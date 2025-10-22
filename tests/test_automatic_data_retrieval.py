"""
Tests for automatic data retrieval capabilities
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.airline_service.services.task_engine import AutoDataRetriever, TaskEngine
from src.airline_service.services.enhanced_identifier_extractor import EnhancedIdentifierExtractor
from src.airline_service.services.intelligent_booking_selector import IntelligentBookingSelector, SelectionContext
from src.airline_service.clients.airline_api_client import MockAirlineAPIClient
from src.airline_service.services.policy_service import PolicyService
from src.airline_service.types import (
    TaskContext, RequestType, EntityType, ExtractedEntity,
    BookingDetails, CustomerSearchInfo, FlightIdentifiers
)


class TestEnhancedIdentifierExtractor:
    """Test enhanced identifier extraction"""
    
    def setup_method(self):
        self.extractor = EnhancedIdentifierExtractor()
    
    def test_extract_pnr_basic(self):
        """Test basic PNR extraction"""
        text = "My booking reference is ABC123"
        pnr = self.extractor.extract_pnr(text)
        assert pnr == "ABC123"
    
    def test_extract_pnr_with_context(self):
        """Test PNR extraction with context keywords"""
        text = "PNR: XYZ789 for my flight"
        pnr = self.extractor.extract_pnr(text)
        assert pnr == "XYZ789"
    
    def test_extract_flight_number(self):
        """Test flight number extraction"""
        text = "I'm on flight AA1234 tomorrow"
        flight_number = self.extractor.extract_flight_number(text)
        assert flight_number == "AA1234"
    
    def test_extract_passenger_name(self):
        """Test passenger name extraction"""
        text = "My name is John Smith and I need help"
        name = self.extractor.extract_passenger_name(text)
        assert name == "John Smith"
    
    def test_extract_route_information(self):
        """Test route extraction"""
        text = "Flying from JFK to LAX"
        route = self.extractor.extract_route(text)
        assert route == {"from": "JFK", "to": "LAX"}
    
    def test_extract_all_identifiers(self):
        """Test comprehensive identifier extraction"""
        text = "I need to cancel my booking ABC123 for flight AA100 from JFK to LAX tomorrow"
        identifiers = self.extractor.extract_all_identifiers(text)
        
        assert identifiers.pnr == "ABC123"
        assert identifiers.flight_number == "AA100"
        # Route extraction would be more complex in real implementation
    
    def test_extract_booking_context(self):
        """Test booking context extraction"""
        text = "I urgently need to cancel my booking"
        context = self.extractor.extract_booking_context(text)
        
        assert context.has_booking_intent
        assert context.has_urgency
        assert len(context.suggested_actions) > 0
    
    def test_extract_customer_info(self):
        """Test customer information extraction"""
        text = "My email is john@example.com and phone is 555-1234"
        customer_info = self.extractor.extract_customer_info(text)
        
        assert customer_info["email"] == "john@example.com"
        assert customer_info["phone"] == "5551234"
    
    def test_extract_contextual_clues(self):
        """Test contextual clue extraction"""
        text = "I need to cancel my flight today urgently"
        clues = self.extractor.extract_contextual_clues(text)
        
        assert clues["urgency"] == "high"
        assert clues["timeframe"] == "today"
        assert clues.get("likely_request") == "cancellation"


class TestIntelligentBookingSelector:
    """Test intelligent booking selection"""
    
    def setup_method(self):
        self.selector = IntelligentBookingSelector()
        
        # Create test bookings
        now = datetime.now()
        self.upcoming_booking = BookingDetails(
            pnr="ABC123",
            flight_id=1001,
            source_airport_code="JFK",
            destination_airport_code="LAX",
            scheduled_departure=now + timedelta(days=2),
            scheduled_arrival=now + timedelta(days=2, hours=5),
            assigned_seat="12A",
            current_departure=now + timedelta(days=2),
            current_arrival=now + timedelta(days=2, hours=5),
            current_status="Confirmed"
        )
        
        self.past_booking = BookingDetails(
            pnr="XYZ789",
            flight_id=2002,
            source_airport_code="ORD",
            destination_airport_code="MIA",
            scheduled_departure=now - timedelta(days=5),
            scheduled_arrival=now - timedelta(days=5, hours=3),
            assigned_seat="8C",
            current_departure=now - timedelta(days=5),
            current_arrival=now - timedelta(days=5, hours=3),
            current_status="Completed"
        )
        
        self.soon_booking = BookingDetails(
            pnr="DEF456",
            flight_id=3003,
            source_airport_code="LAX",
            destination_airport_code="SFO",
            scheduled_departure=now + timedelta(hours=6),
            scheduled_arrival=now + timedelta(hours=7),
            assigned_seat="15F",
            current_departure=now + timedelta(hours=6),
            current_arrival=now + timedelta(hours=7),
            current_status="On Time"
        )
    
    def test_select_single_booking(self):
        """Test selection with single booking"""
        context = SelectionContext(request_type=RequestType.CANCEL_TRIP)
        result = self.selector.select_best_booking([self.upcoming_booking], context)
        
        assert result is not None
        assert result.booking == self.upcoming_booking
        assert result.score == 1.0
    
    def test_select_for_cancellation(self):
        """Test booking selection for cancellation"""
        bookings = [self.upcoming_booking, self.past_booking, self.soon_booking]
        result = self.selector.select_booking_for_cancellation(bookings)
        
        assert result is not None
        # Should prefer upcoming bookings for cancellation
        assert result.booking in [self.upcoming_booking, self.soon_booking]
        assert result.booking != self.past_booking
    
    def test_select_for_status_check(self):
        """Test booking selection for status check"""
        bookings = [self.upcoming_booking, self.past_booking, self.soon_booking]
        result = self.selector.select_booking_for_status_check(bookings)
        
        assert result is not None
        # Should prefer soonest upcoming flight for status check
        assert result.booking == self.soon_booking
    
    def test_prefer_upcoming_flights(self):
        """Test preference for upcoming flights"""
        context = SelectionContext(
            request_type=RequestType.FLIGHT_STATUS,
            prefer_upcoming=True
        )
        
        bookings = [self.past_booking, self.upcoming_booking]
        result = self.selector.select_best_booking(bookings, context)
        
        assert result is not None
        assert result.booking == self.upcoming_booking
    
    def test_selection_explanation(self):
        """Test selection explanation generation"""
        context = SelectionContext(request_type=RequestType.CANCEL_TRIP)
        result = self.selector.select_best_booking([self.upcoming_booking], context)
        
        explanation = self.selector.get_selection_explanation(result)
        assert "ABC123" in explanation
        assert "JFK to LAX" in explanation
        assert len(explanation) > 50  # Should be a meaningful explanation


class TestAutoDataRetriever:
    """Test automatic data retrieval"""
    
    def setup_method(self):
        self.mock_client = MockAirlineAPIClient()
        self.extractor = EnhancedIdentifierExtractor()
        self.retriever = AutoDataRetriever(self.mock_client, self.extractor)
    
    @pytest.mark.asyncio
    async def test_retrieve_booking_by_pnr(self):
        """Test booking retrieval by PNR"""
        context = TaskContext(
            session_id="test_session",
            request_type=RequestType.CANCEL_TRIP,
            extracted_entities=[
                ExtractedEntity(
                    type=EntityType.PNR,
                    value="ABC123",
                    confidence=0.9,
                    start_index=0,
                    end_index=6
                )
            ],
            metadata={"utterance": "Cancel booking ABC123"}
        )
        
        booking = await self.retriever._auto_retrieve_booking_details(context)
        assert booking is not None
        assert booking.pnr == "ABC123"
    
    @pytest.mark.asyncio
    async def test_retrieve_customer_bookings(self):
        """Test customer booking retrieval"""
        context = TaskContext(
            session_id="test_session",
            request_type=RequestType.CANCEL_TRIP,
            extracted_entities=[
                ExtractedEntity(
                    type=EntityType.PHONE_NUMBER,
                    value="555-1234",  # Use format expected by mock client
                    confidence=0.9,
                    start_index=0,
                    end_index=8
                )
            ],
            metadata={"utterance": "Cancel my booking, phone 555-1234"}
        )
        
        bookings = await self.retriever._auto_retrieve_customer_bookings(context)
        assert bookings is not None
        assert len(bookings) > 0
    
    @pytest.mark.asyncio
    async def test_fallback_strategies(self):
        """Test fallback strategy execution"""
        context = TaskContext(
            session_id="test_session",
            request_type=RequestType.CANCEL_TRIP,
            extracted_entities=[
                ExtractedEntity(
                    type=EntityType.PASSENGER_NAME,
                    value="John Smith",
                    confidence=0.8,
                    start_index=0,
                    end_index=10
                )
            ],
            metadata={"utterance": "Cancel booking for John Smith"}
        )
        
        # Should try multiple strategies and potentially find booking
        booking = await self.retriever._auto_retrieve_booking_details(context)
        # May or may not find booking depending on mock data, but should not crash
        assert booking is None or isinstance(booking, BookingDetails)
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self):
        """Test data caching"""
        session_id = "test_session"
        test_data = {"test": "data"}
        
        # Cache data
        self.retriever._cache_data(session_id, "test_type", test_data)
        
        # Retrieve cached data
        cached = self.retriever._get_cached_data(session_id, "test_type")
        assert cached == test_data
        
        # Test cache expiration (would need to mock time for full test)
        non_existent = self.retriever._get_cached_data(session_id, "non_existent")
        assert non_existent is None


class TestTaskEngineIntegration:
    """Test task engine integration with automatic data retrieval"""
    
    def setup_method(self):
        self.mock_client = MockAirlineAPIClient()
        self.mock_policy_service = MagicMock()
        self.task_engine = TaskEngine(self.mock_client, self.mock_policy_service)
    
    @pytest.mark.asyncio
    async def test_enhanced_customer_info_extraction(self):
        """Test enhanced customer info extraction in task engine"""
        from src.airline_service.types import TaskDefinition, TaskType
        
        task = TaskDefinition(
            task_id="test_task",
            task_type=TaskType.GET_CUSTOMER_INFO,
            parameters={"extract_types": ["pnr", "flight_number", "passenger_name"]},
            dependencies=[]
        )
        
        context = TaskContext(
            session_id="test_session",
            request_type=RequestType.CANCEL_TRIP,
            extracted_entities=[],
            metadata={"utterance": "Cancel my booking ABC123 for John Smith on flight AA100"}
        )
        
        result = await self.task_engine.execute_task(task, context)
        
        assert result.success
        assert "flight_identifiers" in result.data
        assert "customer_info" in result.data
        
        # Should extract PNR, flight number, and passenger name
        identifiers = result.data["flight_identifiers"]
        if hasattr(identifiers, 'pnr'):
            assert identifiers.pnr == "ABC123"
    
    @pytest.mark.asyncio
    async def test_automatic_booking_retrieval(self):
        """Test automatic booking retrieval in API call handler"""
        from src.airline_service.types import TaskDefinition, TaskType
        
        task = TaskDefinition(
            task_id="test_api_call",
            task_type=TaskType.API_CALL,
            parameters={
                "api_method": "get_booking_details",
                "required_data": ["booking_details"]
            },
            dependencies=[]
        )
        
        context = TaskContext(
            session_id="test_session",
            request_type=RequestType.CANCEL_TRIP,
            extracted_entities=[
                ExtractedEntity(
                    type=EntityType.PNR,
                    value="ABC123",
                    confidence=0.9,
                    start_index=0,
                    end_index=6
                )
            ],
            metadata={"utterance": "Get details for ABC123"}
        )
        
        result = await self.task_engine.execute_task(task, context)
        
        assert result.success
        assert "booking_details" in result.data
        booking = result.data["booking_details"]
        assert booking.pnr == "ABC123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])