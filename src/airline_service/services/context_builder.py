"""
Context builder for automated data resolution with fallback strategies
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from ..types import (
    RequestType, ExtractedEntity, EntityType, BookingDetails, 
    FlightDetails, CustomerSearchInfo, CustomerProfile, CustomerIdentifier
)
from ..clients.airline_api_client import AirlineAPIClient, AirlineAPIError
from ..services.booking_selector import BookingSelector


class DataSource(str, Enum):
    """Available data sources for context resolution"""
    EXTRACTED_ENTITIES = "extracted_entities"
    API_BOOKING_LOOKUP = "api_booking_lookup"
    API_CUSTOMER_SEARCH = "api_customer_search"
    API_FLIGHT_SEARCH = "api_flight_search"
    CUSTOMER_PROFILE = "customer_profile"
    SESSION_HISTORY = "session_history"
    FALLBACK_INFERENCE = "fallback_inference"


class ResolutionStrategy(str, Enum):
    """Data resolution strategies"""
    DIRECT_LOOKUP = "direct_lookup"
    FUZZY_MATCHING = "fuzzy_matching"
    PARTIAL_SEARCH = "partial_search"
    INFERENCE_BASED = "inference_based"
    PROFILE_BASED = "profile_based"


@dataclass
class ResolvedData:
    """Container for resolved context data"""
    booking_details: Optional[BookingDetails] = None
    flight_details: Optional[FlightDetails] = None
    customer_profile: Optional[CustomerProfile] = None
    extracted_entities: List[ExtractedEntity] = field(default_factory=list)
    confidence_score: float = 0.0
    resolution_path: List[Tuple[DataSource, ResolutionStrategy]] = field(default_factory=list)
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextResolutionRequest:
    """Request for context resolution"""
    utterance: str
    request_type: RequestType
    session_id: str
    customer_id: Optional[str] = None
    extracted_entities: List[ExtractedEntity] = field(default_factory=list)
    session_history: List[Dict[str, Any]] = field(default_factory=list)
    customer_preferences: Optional[Dict[str, Any]] = None


class EntityExtractor:
    """Enhanced entity extraction with pattern matching"""
    
    # Regex patterns for entity extraction
    PATTERNS = {
        EntityType.PNR: [
            r'\b[A-Z0-9]{6}\b',  # Standard 6-character PNR
            r'(?:PNR|confirmation|booking)\s*:?\s*([A-Z0-9]{6})',
            r'(?:reference|ref)\s*:?\s*([A-Z0-9]{6})'
        ],
        EntityType.FLIGHT_NUMBER: [
            r'\b[A-Z]{1,3}\s*\d{1,4}\b',  # AA123, JB1234
            r'(?:flight|flt)\s*:?\s*([A-Z]{1,3}\s*\d{1,4})',
            r'\b\d{4}\b'  # 4-digit flight numbers
        ],
        EntityType.AIRPORT_CODE: [
            r'\b[A-Z]{3}\b',  # 3-letter airport codes
            r'(?:from|to|via)\s+([A-Z]{3})',
            r'(?:airport|apt)\s*:?\s*([A-Z]{3})'
        ],
        EntityType.DATE: [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',  # YYYY-MM-DD
            r'\b(?:today|tomorrow|yesterday)\b',
            r'\b(?:next|this)\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b'
        ],
        EntityType.PHONE_NUMBER: [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US phone numbers
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
            r'\+\d{1,3}\s*\d{3,4}[-.]?\d{3,4}[-.]?\d{4}'
        ],
        EntityType.EMAIL: [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ],
        EntityType.PASSENGER_NAME: [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'(?:name|passenger)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'(?:mr|mrs|ms|miss)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        ]
    }
    
    @classmethod
    def extract_entities(cls, text: str) -> List[ExtractedEntity]:
        """Extract entities from text using pattern matching"""
        entities = []
        text_upper = text.upper()
        
        for entity_type, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)
                    
                    # Clean and validate the extracted value
                    cleaned_value = cls._clean_entity_value(value, entity_type)
                    if cleaned_value and cls._validate_entity(cleaned_value, entity_type):
                        entity = ExtractedEntity(
                            type=entity_type,
                            value=cleaned_value,
                            confidence=cls._calculate_confidence(pattern, match.group(0)),
                            start_index=match.start(),
                            end_index=match.end()
                        )
                        entities.append(entity)
        
        # Remove duplicates and sort by confidence
        entities = cls._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        return entities
    
    @classmethod
    def _clean_entity_value(cls, value: str, entity_type: EntityType) -> str:
        """Clean extracted entity value"""
        value = value.strip()
        
        if entity_type == EntityType.PNR:
            return value.upper().replace(' ', '')
        elif entity_type == EntityType.FLIGHT_NUMBER:
            return re.sub(r'\s+', '', value.upper())
        elif entity_type == EntityType.AIRPORT_CODE:
            return value.upper()
        elif entity_type == EntityType.PHONE_NUMBER:
            return re.sub(r'[^\d]', '', value)
        elif entity_type == EntityType.EMAIL:
            return value.lower()
        elif entity_type == EntityType.PASSENGER_NAME:
            return ' '.join(word.capitalize() for word in value.split())
        
        return value
    
    @classmethod
    def _validate_entity(cls, value: str, entity_type: EntityType) -> bool:
        """Validate extracted entity"""
        if entity_type == EntityType.PNR:
            return len(value) == 6 and value.isalnum()
        elif entity_type == EntityType.AIRPORT_CODE:
            return len(value) == 3 and value.isalpha()
        elif entity_type == EntityType.PHONE_NUMBER:
            return len(value) >= 10
        elif entity_type == EntityType.EMAIL:
            return '@' in value and '.' in value
        elif entity_type == EntityType.FLIGHT_NUMBER:
            return len(value) >= 3 and any(c.isdigit() for c in value)
        
        return True
    
    @classmethod
    def _calculate_confidence(cls, pattern: str, matched_text: str) -> float:
        """Calculate confidence score for entity extraction"""
        base_confidence = 0.7
        
        # Higher confidence for more specific patterns
        if 'PNR' in pattern or 'confirmation' in pattern:
            base_confidence = 0.9
        elif 'flight' in pattern or 'flt' in pattern:
            base_confidence = 0.85
        elif len(matched_text) >= 6:  # Longer matches are more reliable
            base_confidence = 0.8
        
        return min(1.0, base_confidence)
    
    @classmethod
    def _deduplicate_entities(cls, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.type, entity.value.upper())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities


class FallbackStrategy:
    """Implements fallback strategies for data resolution"""
    
    def __init__(self, api_client: AirlineAPIClient, booking_selector: BookingSelector):
        self.api_client = api_client
        self.booking_selector = booking_selector
    
    async def resolve_with_partial_pnr(self, partial_pnr: str) -> Optional[BookingDetails]:
        """Try to resolve booking with partial PNR"""
        if len(partial_pnr) < 3:
            return None
        
        # This would require an enhanced API that supports partial matching
        # For now, we'll simulate the logic
        try:
            # In a real implementation, this would call an API with partial matching
            # For demo purposes, we'll try exact match
            if len(partial_pnr) == 6:
                return await self.api_client.get_booking_details(partial_pnr)
        except AirlineAPIError:
            pass
        
        return None
    
    async def resolve_with_flight_and_name(
        self, 
        flight_number: str, 
        passenger_name: str, 
        date: Optional[datetime] = None
    ) -> Optional[BookingDetails]:
        """Try to resolve booking with flight number and passenger name"""
        try:
            # Search bookings by flight
            search_date = date or datetime.now()
            bookings = await self.api_client.search_bookings_by_flight(flight_number, search_date)
            
            # Filter by passenger name (would need passenger info in booking details)
            # This is a simplified implementation
            if bookings:
                return bookings[0]  # Return first match for demo
        except (AirlineAPIError, NotImplementedError):
            pass
        
        return None
    
    async def resolve_with_route_and_date(
        self, 
        source: str, 
        destination: str, 
        date: datetime,
        passenger_name: Optional[str] = None
    ) -> List[BookingDetails]:
        """Try to resolve bookings with route and date"""
        try:
            # This would use enhanced search APIs
            # For demo, return empty list
            return []
        except (AirlineAPIError, NotImplementedError):
            pass
        
        return []
    
    async def infer_from_context(
        self, 
        context: Dict[str, Any], 
        session_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Infer missing data from context and session history"""
        inferred_data = {}
        
        # Look for patterns in session history
        for session_item in session_history:
            if 'booking_details' in session_item:
                # Use previous booking as context
                prev_booking = session_item['booking_details']
                inferred_data['likely_customer'] = {
                    'previous_pnr': prev_booking.get('pnr'),
                    'preferred_route': f"{prev_booking.get('source_airport_code')}-{prev_booking.get('destination_airport_code')}"
                }
        
        # Infer from request patterns
        if context.get('request_type') == RequestType.CANCEL_TRIP:
            inferred_data['urgency'] = 'high'
            inferred_data['likely_upcoming_flight'] = True
        
        return inferred_data


class ContextBuilder:
    """Main context builder for automated data resolution"""
    
    def __init__(self, api_client: AirlineAPIClient):
        self.api_client = api_client
        self.booking_selector = BookingSelector()
        self.entity_extractor = EntityExtractor()
        self.fallback_strategy = FallbackStrategy(api_client, self.booking_selector)
        self.session_cache: Dict[str, Dict[str, Any]] = {}
    
    async def build_context(self, request: ContextResolutionRequest) -> ResolvedData:
        """Build comprehensive context from request"""
        resolved_data = ResolvedData()
        
        # Step 1: Extract entities from utterance
        if not request.extracted_entities:
            request.extracted_entities = self.entity_extractor.extract_entities(request.utterance)
        
        resolved_data.extracted_entities = request.extracted_entities
        
        # Step 2: Try direct resolution with extracted entities
        success = await self._try_direct_resolution(request, resolved_data)
        
        if not success:
            # Step 3: Try fallback strategies
            await self._try_fallback_resolution(request, resolved_data)
        
        # Step 4: Enhance with customer profile if available
        await self._enhance_with_customer_profile(request, resolved_data)
        
        # Step 5: Calculate final confidence score
        resolved_data.confidence_score = self._calculate_confidence_score(resolved_data)
        
        # Step 6: Cache results for session
        self._cache_session_data(request.session_id, resolved_data)
        
        return resolved_data
    
    async def _try_direct_resolution(
        self, 
        request: ContextResolutionRequest, 
        resolved_data: ResolvedData
    ) -> bool:
        """Try direct resolution using extracted entities"""
        
        # Look for PNR in entities
        pnr_entities = [e for e in request.extracted_entities if e.type == EntityType.PNR]
        
        for pnr_entity in pnr_entities:
            try:
                booking = await self.api_client.get_booking_details(pnr_entity.value)
                resolved_data.booking_details = booking
                resolved_data.resolution_path.append((DataSource.API_BOOKING_LOOKUP, ResolutionStrategy.DIRECT_LOOKUP))
                
                # Extract flight details from booking
                resolved_data.flight_details = FlightDetails(
                    pnr=booking.pnr,
                    flight_number=str(booking.flight_id),  # Simplified
                    departure_date=booking.scheduled_departure,
                    source_airport=booking.source_airport_code,
                    destination_airport=booking.destination_airport_code
                )
                
                return True
                
            except AirlineAPIError:
                continue
        
        # Try customer search if we have customer identifiers
        customer_info = self._build_customer_search_info(request.extracted_entities)
        if customer_info:
            try:
                bookings = await self.api_client.search_bookings_by_customer(customer_info)
                if bookings:
                    # Use booking selector to choose the best one
                    best_booking = self.booking_selector.select_best_booking(
                        bookings, 
                        context={'request_type': request.request_type},
                        customer_preferences=request.customer_preferences
                    )
                    
                    if best_booking:
                        resolved_data.booking_details = best_booking.booking
                        resolved_data.resolution_path.append((DataSource.API_CUSTOMER_SEARCH, ResolutionStrategy.DIRECT_LOOKUP))
                        return True
                        
            except (AirlineAPIError, NotImplementedError):
                pass
        
        return False
    
    async def _try_fallback_resolution(
        self, 
        request: ContextResolutionRequest, 
        resolved_data: ResolvedData
    ) -> bool:
        """Try fallback resolution strategies"""
        
        resolved_data.fallback_used = True
        
        # Strategy 1: Partial PNR matching
        partial_pnrs = [e.value for e in request.extracted_entities if e.type == EntityType.PNR and len(e.value) >= 3]
        for partial_pnr in partial_pnrs:
            booking = await self.fallback_strategy.resolve_with_partial_pnr(partial_pnr)
            if booking:
                resolved_data.booking_details = booking
                resolved_data.resolution_path.append((DataSource.API_BOOKING_LOOKUP, ResolutionStrategy.FUZZY_MATCHING))
                return True
        
        # Strategy 2: Flight number + passenger name
        flight_numbers = [e.value for e in request.extracted_entities if e.type == EntityType.FLIGHT_NUMBER]
        passenger_names = [e.value for e in request.extracted_entities if e.type == EntityType.PASSENGER_NAME]
        
        for flight_number in flight_numbers:
            for passenger_name in passenger_names:
                booking = await self.fallback_strategy.resolve_with_flight_and_name(
                    flight_number, passenger_name
                )
                if booking:
                    resolved_data.booking_details = booking
                    resolved_data.resolution_path.append((DataSource.API_FLIGHT_SEARCH, ResolutionStrategy.PARTIAL_SEARCH))
                    return True
        
        # Strategy 3: Session history lookup
        session_data = self.session_cache.get(request.session_id, {})
        if 'previous_booking' in session_data:
            resolved_data.booking_details = session_data['previous_booking']
            resolved_data.resolution_path.append((DataSource.SESSION_HISTORY, ResolutionStrategy.PROFILE_BASED))
            return True
        
        # Strategy 4: Inference from context
        inferred_data = await self.fallback_strategy.infer_from_context(
            {'request_type': request.request_type}, 
            request.session_history
        )
        
        if inferred_data:
            resolved_data.metadata.update(inferred_data)
            resolved_data.resolution_path.append((DataSource.FALLBACK_INFERENCE, ResolutionStrategy.INFERENCE_BASED))
        
        return False
    
    async def _enhance_with_customer_profile(
        self, 
        request: ContextResolutionRequest, 
        resolved_data: ResolvedData
    ):
        """Enhance context with customer profile data"""
        
        if request.customer_id:
            try:
                # Try to get customer profile
                identifier = CustomerIdentifier(customer_id=request.customer_id)
                profile = await self.api_client.get_customer_profile(identifier)
                resolved_data.customer_profile = profile
                resolved_data.resolution_path.append((DataSource.CUSTOMER_PROFILE, ResolutionStrategy.DIRECT_LOOKUP))
                
                # If we don't have booking details, try recent bookings
                if not resolved_data.booking_details and profile.recent_bookings:
                    best_booking = self.booking_selector.select_best_booking(
                        profile.recent_bookings,
                        context={'request_type': request.request_type}
                    )
                    
                    if best_booking:
                        resolved_data.booking_details = best_booking.booking
                        resolved_data.resolution_path.append((DataSource.CUSTOMER_PROFILE, ResolutionStrategy.PROFILE_BASED))
                
            except (AirlineAPIError, NotImplementedError):
                pass
        
        # Try to extract customer identifiers from entities
        customer_identifiers = self._extract_customer_identifiers(request.extracted_entities)
        for identifier in customer_identifiers:
            try:
                profile = await self.api_client.get_customer_profile(identifier)
                resolved_data.customer_profile = profile
                resolved_data.resolution_path.append((DataSource.CUSTOMER_PROFILE, ResolutionStrategy.DIRECT_LOOKUP))
                break
            except (AirlineAPIError, NotImplementedError):
                continue
    
    def _build_customer_search_info(self, entities: List[ExtractedEntity]) -> Optional[CustomerSearchInfo]:
        """Build customer search info from extracted entities"""
        phone = None
        email = None
        name = None
        
        for entity in entities:
            if entity.type == EntityType.PHONE_NUMBER:
                phone = entity.value
            elif entity.type == EntityType.EMAIL:
                email = entity.value
            elif entity.type == EntityType.PASSENGER_NAME:
                name = entity.value
        
        if phone or email or name:
            return CustomerSearchInfo(phone=phone, email=email, name=name)
        
        return None
    
    def _extract_customer_identifiers(self, entities: List[ExtractedEntity]) -> List[CustomerIdentifier]:
        """Extract customer identifiers from entities"""
        identifiers = []
        
        for entity in entities:
            if entity.type == EntityType.PHONE_NUMBER:
                identifiers.append(CustomerIdentifier(phone=entity.value))
            elif entity.type == EntityType.EMAIL:
                identifiers.append(CustomerIdentifier(email=entity.value))
        
        return identifiers
    
    def _calculate_confidence_score(self, resolved_data: ResolvedData) -> float:
        """Calculate overall confidence score for resolved data"""
        base_score = 0.5
        
        # Boost for successful booking resolution
        if resolved_data.booking_details:
            base_score += 0.3
        
        # Boost for customer profile
        if resolved_data.customer_profile:
            base_score += 0.1
        
        # Boost for high-confidence entities
        high_conf_entities = [e for e in resolved_data.extracted_entities if e.confidence > 0.8]
        base_score += len(high_conf_entities) * 0.05
        
        # Penalty for using fallback
        if resolved_data.fallback_used:
            base_score -= 0.1
        
        # Boost for direct resolution
        direct_resolutions = [path for path in resolved_data.resolution_path 
                            if path[1] == ResolutionStrategy.DIRECT_LOOKUP]
        base_score += len(direct_resolutions) * 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def _cache_session_data(self, session_id: str, resolved_data: ResolvedData):
        """Cache resolved data for session"""
        if session_id not in self.session_cache:
            self.session_cache[session_id] = {}
        
        if resolved_data.booking_details:
            self.session_cache[session_id]['previous_booking'] = resolved_data.booking_details
        
        if resolved_data.customer_profile:
            self.session_cache[session_id]['customer_profile'] = resolved_data.customer_profile
        
        # Keep only recent sessions (simple cleanup)
        if len(self.session_cache) > 100:
            # Remove oldest sessions
            oldest_sessions = list(self.session_cache.keys())[:20]
            for old_session in oldest_sessions:
                del self.session_cache[old_session]
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get cached context for session"""
        return self.session_cache.get(session_id, {})
    
    def clear_session_cache(self, session_id: Optional[str] = None):
        """Clear session cache"""
        if session_id:
            self.session_cache.pop(session_id, None)
        else:
            self.session_cache.clear()


# Global context builder instance
def create_context_builder(api_client: AirlineAPIClient) -> ContextBuilder:
    """Factory function to create context builder"""
    return ContextBuilder(api_client)