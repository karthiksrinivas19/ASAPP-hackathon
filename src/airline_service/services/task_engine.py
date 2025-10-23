"""
Task engine for executing individual tasks within workflows
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Type
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from ..interfaces.workflow_orchestrator import (
    TaskEngineInterface, 
    TaskHandlerInterface,
    AutoDataRetrieverInterface
)
from ..types import (
    TaskDefinition, TaskContext, TaskResult, TaskType, RequestType,
    ExtractedEntity, EntityType, FlightIdentifiers,
    BookingDetails, FlightDetails, CustomerSearchInfo,
    APIResponse, ResponseFormat, PolicyInfo
)
from ..clients.airline_api_client import AirlineAPIClient
from ..services.policy_service import PolicyService
from ..ml.improved_entity_extractor import ImprovedEntityExtractor
from ..services.enhanced_identifier_extractor import EnhancedIdentifierExtractor
from ..services.intelligent_booking_selector import IntelligentBookingSelector, SelectionContext
from .response_formatter import ResponseFormatter


class TaskEngine(TaskEngineInterface):
    """Main task engine that executes individual tasks"""
    
    def __init__(self, airline_client: AirlineAPIClient, policy_service: PolicyService):
        self.airline_client = airline_client
        self.policy_service = policy_service
        self.entity_extractor = ImprovedEntityExtractor()
        self.enhanced_extractor = EnhancedIdentifierExtractor()
        self.task_handlers: Dict[TaskType, TaskHandlerInterface] = {}
        self.auto_retriever = AutoDataRetriever(airline_client, self.enhanced_extractor)
        
        # Register default task handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default task handlers"""
        self.register_task_handler(TaskType.GET_CUSTOMER_INFO, 
                                 GetCustomerInfoHandler(self.enhanced_extractor, self.auto_retriever))
        self.register_task_handler(TaskType.API_CALL, 
                                 APICallHandler(self.airline_client, self.auto_retriever))
        self.register_task_handler(TaskType.POLICY_LOOKUP, 
                                 PolicyLookupHandler(self.policy_service))
        self.register_task_handler(TaskType.INFORM_CUSTOMER, 
                                 InformCustomerHandler())
    
    async def execute_task(self, task: TaskDefinition, context: TaskContext) -> TaskResult:
        """Execute individual task"""
        handler = self.task_handlers.get(task.task_type)
        if not handler:
            return TaskResult(
                success=False,
                error=f"No handler registered for task type: {task.task_type}",
                duration=0.0
            )
        
        start_time = datetime.now()
        
        try:
            result = await handler.execute(task.parameters, context)
            duration = (datetime.now() - start_time).total_seconds()
            
            return TaskResult(
                success=True,
                data=result,
                duration=duration
            )
        
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return TaskResult(
                success=False,
                error=str(e),
                duration=duration
            )
    
    def register_task_handler(self, task_type: TaskType, handler: TaskHandlerInterface) -> None:
        """Register task handler for task type"""
        self.task_handlers[task_type] = handler
    
    async def auto_retrieve_data(self, required_data: List[str], context: TaskContext) -> Dict[str, Any]:
        """Automatically retrieve required data"""
        return await self.auto_retriever.retrieve_data(required_data, context)


class AutoDataRetriever(AutoDataRetrieverInterface):
    """Enhanced automatic data retrieval service with fallback strategies"""
    
    def __init__(self, airline_client: AirlineAPIClient, entity_extractor):
        self.airline_client = airline_client
        self.entity_extractor = entity_extractor
        self.booking_selector = IntelligentBookingSelector()
        self.retrieval_cache = {}  # Simple cache for session-based data
        self.fallback_strategies = [
            self._try_pnr_retrieval,
            self._try_customer_search_retrieval,
            self._try_flight_search_retrieval,
            self._try_partial_info_retrieval,
            self._try_cached_retrieval
        ]
    
    async def retrieve_data(self, required_data: List[str], context: TaskContext) -> Dict[str, Any]:
        """Retrieve required data automatically with enhanced fallback strategies"""
        retrieved_data = {}
        
        for data_field in required_data:
            try:
                if data_field == "booking_details":
                    booking = await self._auto_retrieve_booking_details(context)
                    if booking:
                        retrieved_data["booking_details"] = booking
                
                elif data_field == "flight_info":
                    flight_info = await self._auto_retrieve_flight_info(context)
                    if flight_info:
                        retrieved_data["flight_info"] = flight_info
                
                elif data_field == "customer_info":
                    customer_info = await self._auto_retrieve_customer_info(context)
                    if customer_info:
                        retrieved_data["customer_info"] = customer_info
                
                elif data_field == "customer_bookings":
                    bookings = await self._auto_retrieve_customer_bookings(context)
                    if bookings:
                        retrieved_data["customer_bookings"] = bookings
                
            except Exception as e:
                # Log error but continue with other data
                print(f"Failed to retrieve {data_field}: {e}")
        
        return retrieved_data
    
    async def _auto_retrieve_booking_details(self, context: TaskContext) -> Optional[BookingDetails]:
        """Auto-retrieve booking details using multiple strategies"""
        
        # Try each fallback strategy in order
        for strategy in self.fallback_strategies:
            try:
                booking = await strategy(context, "booking_details")
                if booking:
                    # Cache successful retrieval
                    self._cache_data(context.session_id, "booking_details", booking)
                    return booking
            except Exception as e:
                print(f"Strategy failed: {strategy.__name__}: {e}")
                continue
        
        return None
    
    async def _auto_retrieve_flight_info(self, context: TaskContext) -> Optional[Dict[str, Any]]:
        """Auto-retrieve flight information using multiple strategies"""
        
        # First try to get from booking details
        booking = await self._auto_retrieve_booking_details(context)
        if booking:
            return {
                "flight_id": booking.flight_id,
                "flight_number": f"{booking.source_airport_code}{booking.flight_id}",
                "source_airport": booking.source_airport_code,
                "destination_airport": booking.destination_airport_code,
                "departure_date": booking.scheduled_departure,
                "arrival_date": booking.scheduled_arrival,
                "status": booking.current_status
            }
        
        # Try flight-specific strategies
        for strategy in self.fallback_strategies:
            try:
                flight_info = await strategy(context, "flight_info")
                if flight_info:
                    self._cache_data(context.session_id, "flight_info", flight_info)
                    return flight_info
            except Exception as e:
                print(f"Flight info strategy failed: {strategy.__name__}: {e}")
                continue
        
        return None
    
    async def _auto_retrieve_customer_info(self, context: TaskContext) -> Optional[CustomerSearchInfo]:
        """Auto-retrieve customer information from entities"""
        customer_info = CustomerSearchInfo()
        
        for entity in context.extracted_entities:
            if entity.type == EntityType.PHONE_NUMBER:
                customer_info.phone = entity.value
            elif entity.type == EntityType.EMAIL:
                customer_info.email = entity.value
            elif entity.type == EntityType.PASSENGER_NAME:
                customer_info.name = entity.value
        
        # Return only if we have at least one identifier
        if customer_info.phone or customer_info.email or customer_info.name:
            return customer_info
        
        return None
    
    async def _auto_retrieve_customer_bookings(self, context: TaskContext) -> Optional[List[BookingDetails]]:
        """Auto-retrieve customer bookings using customer search"""
        customer_info = await self._auto_retrieve_customer_info(context)
        
        if customer_info:
            try:
                bookings = await self.airline_client.search_bookings_by_customer(customer_info)
                if bookings:
                    self._cache_data(context.session_id, "customer_bookings", bookings)
                    return bookings
            except Exception as e:
                print(f"Customer bookings retrieval failed: {e}")
        
        return None
    
    # Fallback Strategy Implementations
    
    async def _try_pnr_retrieval(self, context: TaskContext, data_type: str) -> Optional[Any]:
        """Strategy 1: Direct PNR lookup"""
        pnr = self._extract_pnr_from_context(context)
        if not pnr:
            return None
        
        if data_type == "booking_details":
            return await self.retrieve_booking_details(pnr)
        elif data_type == "flight_info":
            booking = await self.retrieve_booking_details(pnr)
            if booking:
                return {
                    "flight_id": booking.flight_id,
                    "pnr": booking.pnr,
                    "source_airport": booking.source_airport_code,
                    "destination_airport": booking.destination_airport_code,
                    "departure_date": booking.scheduled_departure
                }
        
        return None
    
    async def _try_customer_search_retrieval(self, context: TaskContext, data_type: str) -> Optional[Any]:
        """Strategy 2: Customer search using identifiers with intelligent selection"""
        customer_info = await self._auto_retrieve_customer_info(context)
        if not customer_info:
            return None
        
        try:
            bookings = await self.airline_client.search_bookings_by_customer(customer_info)
            if not bookings:
                return None
            
            if data_type == "booking_details":
                # Use intelligent booking selector to choose the best booking
                selection_context = SelectionContext(
                    request_type=context.request_type,
                    prefer_upcoming=True,
                    time_sensitivity=True
                )
                
                # Select based on request type
                if context.request_type == RequestType.CANCEL_TRIP:
                    best_booking = self.booking_selector.select_booking_for_cancellation(bookings)
                elif context.request_type == RequestType.FLIGHT_STATUS:
                    best_booking = self.booking_selector.select_booking_for_status_check(bookings)
                elif context.request_type == RequestType.SEAT_AVAILABILITY:
                    best_booking = self.booking_selector.select_booking_for_seat_availability(bookings)
                else:
                    best_booking = self.booking_selector.select_best_booking(bookings, selection_context)
                
                if best_booking:
                    # Cache the selection reasoning for later use
                    self._cache_data(context.session_id, "selection_reasoning", {
                        "selected_booking": best_booking.booking.pnr,
                        "reasons": best_booking.reasons,
                        "score": best_booking.score
                    })
                    return best_booking.booking
                
                # Fallback to simple selection
                upcoming_bookings = [b for b in bookings if b.scheduled_departure > datetime.now()]
                if upcoming_bookings:
                    return max(upcoming_bookings, key=lambda x: x.scheduled_departure)
                return max(bookings, key=lambda x: x.scheduled_departure)
            
            elif data_type == "customer_bookings":
                return bookings
            
        except Exception as e:
            print(f"Customer search failed: {e}")
        
        return None
    
    async def _try_flight_search_retrieval(self, context: TaskContext, data_type: str) -> Optional[Any]:
        """Strategy 3: Flight-based search"""
        flight_number = self._extract_flight_number_from_context(context)
        date = self._extract_date_from_context(context)
        
        if not flight_number:
            return None
        
        try:
            # Use current date if no date specified
            search_date = date or datetime.now()
            bookings = await self.airline_client.search_bookings_by_flight(flight_number, search_date)
            
            if bookings and data_type == "booking_details":
                # If we have passenger name, try to match
                passenger_name = self._extract_passenger_name_from_context(context)
                if passenger_name:
                    # In a real implementation, we'd match against passenger names in bookings
                    # For now, return first booking
                    return bookings[0]
                return bookings[0]
            
        except Exception as e:
            print(f"Flight search failed: {e}")
        
        return None
    
    async def _try_partial_info_retrieval(self, context: TaskContext, data_type: str) -> Optional[Any]:
        """Strategy 4: Partial information search"""
        # Build search criteria from all available entities
        search_criteria = {}
        
        for entity in context.extracted_entities:
            if entity.type == EntityType.PNR and len(entity.value) >= 3:
                search_criteria["partial_pnr"] = entity.value
            elif entity.type == EntityType.FLIGHT_NUMBER:
                search_criteria["flight_number"] = entity.value
            elif entity.type == EntityType.PASSENGER_NAME:
                search_criteria["passenger_name"] = entity.value
            elif entity.type == EntityType.PHONE_NUMBER:
                search_criteria["phone"] = entity.value
            elif entity.type == EntityType.EMAIL:
                search_criteria["email"] = entity.value
            elif entity.type == EntityType.AIRPORT_CODE:
                if "source_airport" not in search_criteria:
                    search_criteria["source_airport"] = entity.value
                else:
                    search_criteria["destination_airport"] = entity.value
        
        if not search_criteria:
            return None
        
        try:
            bookings = await self.airline_client.search_bookings_by_partial_info(search_criteria)
            if bookings and data_type == "booking_details":
                return bookings[0]  # Return first match
            elif bookings and data_type == "customer_bookings":
                return bookings
        except Exception as e:
            print(f"Partial info search failed: {e}")
        
        return None
    
    async def _try_cached_retrieval(self, context: TaskContext, data_type: str) -> Optional[Any]:
        """Strategy 5: Retrieve from session cache"""
        cached_data = self._get_cached_data(context.session_id, data_type)
        return cached_data
    
    # Enhanced extraction methods
    
    def _extract_flight_number_from_context(self, context: TaskContext) -> Optional[str]:
        """Extract flight number from context entities"""
        for entity in context.extracted_entities:
            if entity.type == EntityType.FLIGHT_NUMBER:
                return entity.value
        return None
    
    def _extract_date_from_context(self, context: TaskContext) -> Optional[datetime]:
        """Extract date from context entities"""
        for entity in context.extracted_entities:
            if entity.type == EntityType.DATE:
                # In a real implementation, we'd parse the date string
                # For now, return current date as fallback
                try:
                    if isinstance(entity.value, str):
                        # Simple date parsing - in production, use proper date parser
                        if entity.value.lower() in ["today", "now"]:
                            return datetime.now()
                        elif entity.value.lower() == "tomorrow":
                            return datetime.now() + timedelta(days=1)
                        # Add more date parsing logic as needed
                    return entity.value if isinstance(entity.value, datetime) else None
                except:
                    return None
        return None
    
    def _extract_passenger_name_from_context(self, context: TaskContext) -> Optional[str]:
        """Extract passenger name from context entities"""
        for entity in context.extracted_entities:
            if entity.type == EntityType.PASSENGER_NAME:
                return entity.value
        return None
    
    # Caching methods
    
    def _cache_data(self, session_id: str, data_type: str, data: Any) -> None:
        """Cache data for session"""
        if session_id not in self.retrieval_cache:
            self.retrieval_cache[session_id] = {}
        
        self.retrieval_cache[session_id][data_type] = {
            "data": data,
            "timestamp": datetime.now()
        }
        
        # Simple cache cleanup - keep only recent sessions
        if len(self.retrieval_cache) > 100:
            oldest_sessions = sorted(
                self.retrieval_cache.keys(),
                key=lambda x: self.retrieval_cache[x].get("timestamp", datetime.min)
            )[:20]
            
            for old_session in oldest_sessions:
                del self.retrieval_cache[old_session]
    
    def _get_cached_data(self, session_id: str, data_type: str) -> Optional[Any]:
        """Get cached data for session"""
        if session_id not in self.retrieval_cache:
            return None
        
        session_cache = self.retrieval_cache[session_id]
        if data_type not in session_cache:
            return None
        
        cached_item = session_cache[data_type]
        
        # Check if cache is still valid (within 10 minutes)
        if datetime.now() - cached_item["timestamp"] > timedelta(minutes=10):
            del session_cache[data_type]
            return None
        
        return cached_item["data"]
    
    async def retrieve_booking_details(self, pnr: str) -> BookingDetails:
        """Retrieve booking details by PNR"""
        return await self.airline_client.get_booking_details(pnr)
    
    def extract_pnr_from_query(self, utterance: str) -> Optional[str]:
        """Extract PNR from customer query"""
        # PNR pattern: 6 alphanumeric characters
        pnr_pattern = r'\b[A-Z0-9]{6}\b'
        matches = re.findall(pnr_pattern, utterance.upper())
        return matches[0] if matches else None
    
    def extract_flight_details_from_query(self, utterance: str) -> Dict[str, Any]:
        """Extract flight details from query"""
        entities = self.entity_extractor.extract_entities(utterance)
        
        flight_details = {}
        for entity in entities:
            if entity.type == EntityType.PNR:
                flight_details["pnr"] = entity.value
            elif entity.type == EntityType.FLIGHT_NUMBER:
                flight_details["flight_number"] = entity.value
            elif entity.type == EntityType.DATE:
                flight_details["date"] = entity.value
            elif entity.type == EntityType.AIRPORT_CODE:
                if "source_airport" not in flight_details:
                    flight_details["source_airport"] = entity.value
                else:
                    flight_details["destination_airport"] = entity.value
        
        return flight_details
    
    def _has_pnr(self, context: TaskContext) -> bool:
        """Check if context has PNR"""
        return any(entity.type == EntityType.PNR for entity in context.extracted_entities)
    
    def _extract_pnr_from_context(self, context: TaskContext) -> Optional[str]:
        """Extract PNR from context entities"""
        for entity in context.extracted_entities:
            if entity.type == EntityType.PNR:
                return entity.value
        return None
    
    def _has_flight_identifiers(self, context: TaskContext) -> bool:
        """Check if context has flight identifiers"""
        return any(entity.type in [EntityType.FLIGHT_NUMBER, EntityType.PNR] 
                  for entity in context.extracted_entities)
    
    async def _retrieve_flight_info(self, context: TaskContext) -> Dict[str, Any]:
        """Retrieve flight information from context"""
        # Try PNR first
        pnr = self._extract_pnr_from_context(context)
        if pnr:
            booking = await self.airline_client.get_booking_details(pnr)
            return {
                "flight_id": booking.flight_id,
                "flight_number": f"{booking.source_airport_code}{booking.flight_id}",
                "source_airport": booking.source_airport_code,
                "destination_airport": booking.destination_airport_code,
                "departure_date": booking.scheduled_departure
            }
        
        # Try flight number
        flight_number = None
        for entity in context.extracted_entities:
            if entity.type == EntityType.FLIGHT_NUMBER:
                flight_number = entity.value
                break
        
        if flight_number:
            # For now, return basic flight info
            # In a real implementation, we'd call a flight info API
            return {
                "flight_number": flight_number,
                "source_airport": "Unknown",
                "destination_airport": "Unknown"
            }
        
        return {}
    
    def _extract_customer_info(self, context: TaskContext) -> CustomerSearchInfo:
        """Extract customer information from context"""
        customer_info = CustomerSearchInfo()
        
        for entity in context.extracted_entities:
            if entity.type == EntityType.PHONE_NUMBER:
                customer_info.phone = entity.value
            elif entity.type == EntityType.EMAIL:
                customer_info.email = entity.value
            elif entity.type == EntityType.PASSENGER_NAME:
                customer_info.name = entity.value
        
        return customer_info


class GetCustomerInfoHandler(TaskHandlerInterface):
    """Handler for GET_CUSTOMER_INFO tasks - extracts identifiers from customer queries"""
    
    def __init__(self, entity_extractor, auto_retriever: AutoDataRetriever):
        self.entity_extractor = entity_extractor
        self.auto_retriever = auto_retriever
    
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Any:
        """Execute customer info extraction task with enhanced extraction capabilities"""
        extract_types = parameters.get("extract_types", [])
        optional = parameters.get("optional", False)
        
        # Extract entities from utterance using enhanced extractor
        utterance = context.metadata.get("utterance", "")
        
        # Use enhanced extractor for comprehensive identifier extraction
        if hasattr(self.entity_extractor, 'extract_all_identifiers'):
            try:
                flight_identifiers = self.entity_extractor.extract_all_identifiers(utterance)
                
                # Also get customer info and contextual clues
                customer_info = self.entity_extractor.extract_customer_info(utterance)
                contextual_clues = self.entity_extractor.extract_contextual_clues(utterance)
                booking_context = self.entity_extractor.extract_booking_context(utterance)
                
                # Convert flight identifiers to entities for compatibility
                entities = []
                if flight_identifiers.pnr and isinstance(flight_identifiers.pnr, str):
                    entities.append(ExtractedEntity(
                        type=EntityType.PNR,
                        value=flight_identifiers.pnr,
                        confidence=0.9,
                        start_index=0,
                        end_index=len(flight_identifiers.pnr)
                    ))
                
                if flight_identifiers.flight_number and isinstance(flight_identifiers.flight_number, str):
                    entities.append(ExtractedEntity(
                        type=EntityType.FLIGHT_NUMBER,
                        value=flight_identifiers.flight_number,
                        confidence=0.9,
                        start_index=0,
                        end_index=len(flight_identifiers.flight_number)
                    ))
                
                if flight_identifiers.passenger_name and isinstance(flight_identifiers.passenger_name, str):
                    entities.append(ExtractedEntity(
                        type=EntityType.PASSENGER_NAME,
                        value=flight_identifiers.passenger_name,
                        confidence=0.8,
                        start_index=0,
                        end_index=len(flight_identifiers.passenger_name)
                    ))
                
                # Add customer info entities
                if customer_info.get("phone") and isinstance(customer_info["phone"], str):
                    entities.append(ExtractedEntity(
                        type=EntityType.PHONE_NUMBER,
                        value=customer_info["phone"],
                        confidence=0.9,
                        start_index=0,
                        end_index=len(customer_info["phone"])
                    ))
                
                if customer_info.get("email") and isinstance(customer_info["email"], str):
                    entities.append(ExtractedEntity(
                        type=EntityType.EMAIL,
                        value=customer_info["email"],
                        confidence=0.9,
                        start_index=0,
                        end_index=len(customer_info["email"])
                    ))
                
            except Exception as e:
                # Fallback to original method if enhanced extraction fails
                entities = self.entity_extractor.extract_entities(utterance)
                flight_identifiers = FlightIdentifiers()
                customer_info = {}
                contextual_clues = {}
                booking_context = None
                
                for entity in entities:
                    if entity.type == EntityType.PNR:
                        flight_identifiers.pnr = entity.value
                    elif entity.type == EntityType.FLIGHT_NUMBER:
                        flight_identifiers.flight_number = entity.value
                    elif entity.type == EntityType.PASSENGER_NAME:
                        flight_identifiers.passenger_name = entity.value
                        customer_info["name"] = entity.value
                    elif entity.type == EntityType.DATE:
                        flight_identifiers.date = entity.value
                    elif entity.type == EntityType.PHONE_NUMBER:
                        customer_info["phone"] = entity.value
                    elif entity.type == EntityType.EMAIL:
                        customer_info["email"] = entity.value
            
        else:
            # Fallback to original entity extractor
            entities = self.entity_extractor.extract_entities(utterance)
            
            # Create flight identifiers from entities
            flight_identifiers = FlightIdentifiers()
            customer_info = {}
            
            for entity in entities:
                if entity.type == EntityType.PNR:
                    flight_identifiers.pnr = entity.value
                elif entity.type == EntityType.FLIGHT_NUMBER:
                    flight_identifiers.flight_number = entity.value
                elif entity.type == EntityType.PASSENGER_NAME:
                    flight_identifiers.passenger_name = entity.value
                    customer_info["name"] = entity.value
                elif entity.type == EntityType.DATE:
                    flight_identifiers.date = entity.value
                elif entity.type == EntityType.PHONE_NUMBER:
                    customer_info["phone"] = entity.value
                elif entity.type == EntityType.EMAIL:
                    customer_info["email"] = entity.value
            
            contextual_clues = {}
            booking_context = None
        
        # Filter entities based on requested types if specified
        filtered_entities = entities
        if extract_types:
            type_mapping = {
                "pnr": EntityType.PNR,
                "flight_number": EntityType.FLIGHT_NUMBER,
                "passenger_name": EntityType.PASSENGER_NAME,
                "date": EntityType.DATE,
                "route": EntityType.AIRPORT_CODE,
                "phone": EntityType.PHONE_NUMBER,
                "email": EntityType.EMAIL,
                "pet_type": EntityType.PET_TYPE,
                "fare_type": "fare_type",
                "destination": EntityType.DESTINATION
            }
            
            requested_types = [type_mapping.get(t, t) for t in extract_types]
            filtered_entities = [e for e in entities if e.type in requested_types]
        
        # Enhanced result with additional context
        result = {
            "extracted_entities": filtered_entities,
            "flight_identifiers": flight_identifiers,
            "customer_info": customer_info,
            "success": True
        }
        
        # Add enhanced context if available
        if contextual_clues:
            result["contextual_clues"] = contextual_clues
        
        if booking_context:
            result["booking_context"] = booking_context
        
        # Check if we found useful information
        if not filtered_entities and not optional:
            result["warning"] = "No identifiers found in customer query"
            
            # Add suggestions if available
            if booking_context and hasattr(booking_context, 'suggested_actions'):
                result["suggestions"] = booking_context.suggested_actions
        
        return result
    
    def get_required_data(self) -> List[str]:
        """Get list of required data fields"""
        return ["utterance"]
    
    def can_auto_retrieve(self, data_field: str) -> bool:
        """Check if data field can be auto-retrieved"""
        return data_field in ["utterance"]


class APICallHandler(TaskHandlerInterface):
    """Handler for API_CALL tasks - makes API calls with automatic parameter resolution"""
    
    def __init__(self, airline_client: AirlineAPIClient, auto_retriever: AutoDataRetriever):
        self.airline_client = airline_client
        self.auto_retriever = auto_retriever
    
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Any:
        """Execute API call task"""
        api_method = parameters.get("api_method")
        required_data = parameters.get("required_data", [])
        
        if not api_method:
            raise ValueError("api_method parameter is required")
        
        # Auto-retrieve required data
        retrieved_data = await self.auto_retriever.retrieve_data(required_data, context)
        
        # Also check workflow data for previously retrieved information
        workflow_data = context.metadata.get("workflow_data", {})
        
        # Execute the appropriate API method
        if api_method == "get_booking_details":
            pnr = self._get_pnr(retrieved_data, workflow_data, context)
            if not pnr:
                raise ValueError("PNR is required for get_booking_details")
            
            booking_details = await self.airline_client.get_booking_details(pnr)
            return {
                "booking_details": booking_details,
                "api_method": api_method,
                "success": True
            }
        
        elif api_method == "cancel_flight":
            booking_details = self._get_booking_details(retrieved_data, workflow_data)
            if not booking_details:
                raise ValueError("Booking details are required for cancel_flight")
            
            # Extract PNR from booking details for cancellation
            pnr = booking_details.pnr if hasattr(booking_details, 'pnr') else str(booking_details)
            cancellation_result = await self.airline_client.cancel_flight(pnr, "Customer request")
            return {
                "cancellation_result": cancellation_result,
                "booking_details": booking_details,  # Include original booking details
                "api_method": api_method,
                "success": True
            }
        
        elif api_method == "get_available_seats":
            # Get flight info from booking details or flight identifiers
            booking_details = self._get_booking_details(retrieved_data, workflow_data)
            if booking_details:
                # Use booking details to get seat availability
                flight_id = str(booking_details.flight_id)
                date = booking_details.scheduled_departure.strftime("%Y-%m-%d")
            else:
                # Fallback to flight info
                flight_info = self._get_flight_info(retrieved_data, workflow_data)
                if not flight_info:
                    raise ValueError("Flight info or booking details are required for get_available_seats")
                
                flight_id = str(flight_info.get("flight_id", flight_info.get("flight_number", "1001")))
                date = flight_info.get("departure_date", "2024-01-15")
                if hasattr(date, 'strftime'):
                    date = date.strftime("%Y-%m-%d")
            
            seat_availability = await self.airline_client.get_available_seats(flight_id, date)
            return {
                "seat_availability": seat_availability,
                "flight_id": flight_id,
                "date": date,
                "api_method": api_method,
                "success": True
            }
        
        else:
            raise ValueError(f"Unknown API method: {api_method}")
    
    def _get_pnr(self, retrieved_data: Dict[str, Any], workflow_data: Dict[str, Any], context: TaskContext) -> Optional[str]:
        """Get PNR from various sources"""
        # Check retrieved data
        if "booking_details" in retrieved_data:
            return retrieved_data["booking_details"].pnr
        
        # Check workflow data
        for task_data in workflow_data.values():
            if isinstance(task_data, dict):
                if "flight_identifiers" in task_data:
                    identifiers = task_data["flight_identifiers"]
                    if hasattr(identifiers, 'pnr') and identifiers.pnr:
                        return identifiers.pnr
                
                if "extracted_entities" in task_data:
                    for entity in task_data["extracted_entities"]:
                        if hasattr(entity, 'type') and entity.type == EntityType.PNR:
                            return entity.value
        
        # Check context entities
        for entity in context.extracted_entities:
            if entity.type == EntityType.PNR:
                return entity.value
        
        return None
    
    def _get_booking_details(self, retrieved_data: Dict[str, Any], workflow_data: Dict[str, Any]) -> Optional[BookingDetails]:
        """Get booking details from various sources"""
        # Check retrieved data
        if "booking_details" in retrieved_data:
            return retrieved_data["booking_details"]
        
        # Check workflow data
        for task_data in workflow_data.values():
            if isinstance(task_data, dict) and "booking_details" in task_data:
                return task_data["booking_details"]
        
        return None
    
    def _get_flight_info(self, retrieved_data: Dict[str, Any], workflow_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get flight info from various sources"""
        # Check retrieved data
        if "flight_info" in retrieved_data:
            return retrieved_data["flight_info"]
        
        # Check for booking details that can provide flight info
        booking_details = self._get_booking_details(retrieved_data, workflow_data)
        if booking_details:
            return {
                "flight_id": booking_details.flight_id,
                "pnr": booking_details.pnr,
                "source_airport": booking_details.source_airport_code,
                "destination_airport": booking_details.destination_airport_code
            }
        
        return None
    
    def get_required_data(self) -> List[str]:
        """Get list of required data fields"""
        return ["api_method"]
    
    def can_auto_retrieve(self, data_field: str) -> bool:
        """Check if data field can be auto-retrieved"""
        return data_field in ["booking_details", "flight_info", "pnr"]


class PolicyLookupHandler(TaskHandlerInterface):
    """Handler for POLICY_LOOKUP tasks - retrieves policy information with caching"""
    
    def __init__(self, policy_service: PolicyService):
        self.policy_service = policy_service
    
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Any:
        """Execute policy lookup task"""
        policy_type = parameters.get("policy_type")
        context_aware = parameters.get("context_aware", False)
        
        if not policy_type:
            raise ValueError("policy_type parameter is required")
        
        try:
            # Add timeout to prevent hanging
            policy_info = await asyncio.wait_for(
                self._get_policy_with_fallback(policy_type),
                timeout=5.0  # 5 second timeout
            )
        except asyncio.TimeoutError:
            # Fallback response if policy lookup times out
            policy_info = self._create_fallback_policy(policy_type)
        except Exception as e:
            # Fallback response for any other errors
            print(f"Policy lookup error: {e}")
            policy_info = self._create_fallback_policy(policy_type)
        
        return {
            "policy_info": policy_info,
            "policy_type": policy_type,
            "context_aware": context_aware,
            "success": True
        }
    
    async def _get_policy_with_fallback(self, policy_type: str) -> PolicyInfo:
        """Get policy with fallback handling"""
        if policy_type == "cancellation":
            return await self.policy_service.get_general_cancellation_policy()
        elif policy_type == "pet_travel":
            return await self.policy_service.get_pet_travel_policy()
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    def _create_fallback_policy(self, policy_type: str) -> PolicyInfo:
        """Create fallback policy response when lookup fails"""
        fallback_content = {
            "cancellation": """
**JetBlue Cancellation Policy Summary**

• **24-Hour Rule**: Cancel within 24 hours of booking for a full refund (if booked 7+ days before departure)
• **Blue Basic**: Non-refundable, but can cancel for JetBlue credit minus fees
• **Blue/Blue Plus/Blue Extra**: Refundable fares with varying fees
• **Same-day changes**: Available for a fee
• **Weather/JetBlue delays**: Full refund or rebooking at no charge

For specific fare rules and current fees, please visit jetblue.com or contact customer service.
            """,
            "pet_travel": """
**JetBlue Pet Travel Policy Summary**

• **In-Cabin Pets**: Small cats and dogs in approved carriers
• **Pet Fee**: $125 each way for in-cabin pets
• **Carrier Requirements**: Must fit under the seat in front of you
• **Health Certificate**: Required for some destinations
• **Service Animals**: Travel free with proper documentation
• **Restrictions**: No pets in cargo, limited to in-cabin only

For complete pet travel requirements, visit jetblue.com/pets or contact customer service.
            """
        }
        
        return PolicyInfo(
            policy_type=policy_type,
            content=fallback_content.get(policy_type, f"{policy_type.title()} policy information is currently unavailable. Please contact customer service for assistance."),
            last_updated=datetime.now(),
            applicable_conditions=[policy_type]
        )
    
    def _extract_flight_details(self, context: TaskContext) -> Optional[FlightDetails]:
        """Extract flight details from context for policy lookup"""
        flight_details = FlightDetails()
        
        # Check context entities
        for entity in context.extracted_entities:
            if entity.type == EntityType.PNR:
                flight_details.pnr = entity.value
            elif entity.type == EntityType.FLIGHT_NUMBER:
                flight_details.flight_number = entity.value
            elif entity.type == EntityType.DATE:
                flight_details.departure_date = entity.value
        
        # Return None if no relevant details found
        if not any([flight_details.pnr, flight_details.flight_number, flight_details.departure_date]):
            return None
        
        return flight_details
    
    def get_required_data(self) -> List[str]:
        """Get list of required data fields"""
        return ["policy_type"]
    
    def can_auto_retrieve(self, data_field: str) -> bool:
        """Check if data field can be auto-retrieved"""
        return False


class InformCustomerHandler(TaskHandlerInterface):
    """Handler for INFORM_CUSTOMER tasks - formats responses for customers"""
    
    def __init__(self):
        self.response_formatter = ResponseFormatter()
        self.response_type_mapping = {
            ResponseFormat.CANCELLATION_RESULT: RequestType.CANCEL_TRIP,
            ResponseFormat.FLIGHT_STATUS: RequestType.FLIGHT_STATUS,
            ResponseFormat.SEAT_AVAILABILITY: RequestType.SEAT_AVAILABILITY,
            ResponseFormat.POLICY_INFO: RequestType.CANCELLATION_POLICY,
            ResponseFormat.BOOKING_CONFIRMATION: RequestType.CANCEL_TRIP,
            ResponseFormat.SIMPLE_MESSAGE: None  # Generic response
        }
    
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Any:
        """Execute customer information task"""
        response_type = parameters.get("response_type")
        
        if not response_type:
            raise ValueError("response_type parameter is required")
        
        # Get data from workflow
        workflow_data = context.metadata.get("workflow_data", {})
        
        # Find the most recent task result that contains relevant data
        relevant_data = self._find_relevant_data(workflow_data, response_type)
        
        # Get the corresponding request type for formatting
        request_type = self.response_type_mapping.get(ResponseFormat(response_type))
        
        # Format response using the new response formatter
        response_format = ResponseFormat(response_type)
        
        # Check if we have a specific format builder
        if hasattr(self.response_formatter, 'format_response_by_format'):
            formatted_response = self.response_formatter.format_response_by_format(
                response_format, relevant_data
            )
        elif request_type:
            formatted_response = self.response_formatter.format_response(
                request_type, relevant_data
            )
        else:
            # Handle simple message or generic responses
            message = parameters.get("message", "Request processed successfully")
            formatted_response = self.response_formatter.create_completed_response(
                message, relevant_data if isinstance(relevant_data, dict) else {"result": relevant_data}
            )
        
        return {
            "response": formatted_response,
            "response_type": response_type,
            "success": True
        }
    
    def _find_relevant_data(self, workflow_data: Dict[str, Any], response_type: str) -> Any:
        """Find relevant data for response formatting"""
        # Look for specific data types based on response type
        if response_type == "cancellation_result":
            for task_data in workflow_data.values():
                if isinstance(task_data, dict) and "cancellation_result" in task_data:
                    return task_data["cancellation_result"]
        
        elif response_type == "flight_status":
            for task_data in workflow_data.values():
                if isinstance(task_data, dict) and "booking_details" in task_data:
                    return task_data["booking_details"]
        
        elif response_type == "seat_availability":
            for task_data in workflow_data.values():
                if isinstance(task_data, dict) and "seat_availability" in task_data:
                    return task_data["seat_availability"]
        
        elif response_type == "policy_info":
            for task_data in workflow_data.values():
                if isinstance(task_data, dict) and "policy_info" in task_data:
                    return task_data["policy_info"]
        
        return None
        """Format cancellation result response"""
    
    def get_required_data(self) -> List[str]:
        """Get list of required data fields"""
        return ["response_type"]
    
    def can_auto_retrieve(self, data_field: str) -> bool:
        """Check if data field can be auto-retrieved"""
        return False