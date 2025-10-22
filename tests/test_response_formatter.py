"""
Tests for response formatting service
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from src.airline_service.services.response_formatter import ResponseFormatter, AutomatedResponseBuilder
from src.airline_service.types import (
    APIResponse, RequestType, ResponseFormat, BookingDetails, FlightInfo,
    CancellationResult, SeatAvailability, SeatInfo, PolicyInfo,
    APIError, MissingDataError, ClassificationError
)


class TestResponseFormatter:
    """Test response formatter functionality"""
    
    @pytest.fixture
    def formatter(self):
        """Create response formatter instance"""
        return ResponseFormatter()
    
    @pytest.fixture
    def sample_booking_details(self):
        """Sample booking details for testing"""
        return BookingDetails(
            pnr="ABC123",
            flight_id=12345,
            source_airport_code="JFK",
            destination_airport_code="LAX",
            scheduled_departure=datetime(2024, 1, 15, 10, 0),
            scheduled_arrival=datetime(2024, 1, 15, 13, 0),
            assigned_seat="12A",
            current_departure=datetime(2024, 1, 15, 10, 15),
            current_arrival=datetime(2024, 1, 15, 13, 15),
            current_status="On Time"
        )
    
    @pytest.fixture
    def sample_flight_info(self):
        """Sample flight info for testing"""
        return FlightInfo(
            flight_id=12345,
            flight_number="AA100",
            source_airport_code="JFK",
            destination_airport_code="LAX",
            scheduled_departure=datetime(2024, 1, 15, 10, 0),
            scheduled_arrival=datetime(2024, 1, 15, 13, 0),
            current_status="Delayed"
        )
    
    @pytest.fixture
    def sample_cancellation_result(self):
        """Sample cancellation result for testing"""
        return CancellationResult(
            message="Flight cancelled successfully",
            cancellation_charges=50.0,
            refund_amount=150.0,
            refund_date=datetime(2024, 1, 20)
        )
    
    @pytest.fixture
    def sample_seat_availability(self):
        """Sample seat availability for testing"""
        seats = [
            SeatInfo(**{"row_number": 12, "column_letter": "A", "price": 25.0, "class": "Economy"}),
            SeatInfo(**{"row_number": 12, "column_letter": "B", "price": 25.0, "class": "Economy"}),
            SeatInfo(**{"row_number": 5, "column_letter": "A", "price": 75.0, "class": "Business"})
        ]
        return SeatAvailability(
            flight_id=12345,
            pnr="ABC123",
            available_seats=seats
        )
    
    @pytest.fixture
    def sample_policy_info(self):
        """Sample policy info for testing"""
        return PolicyInfo(
            policy_type="cancellation_policy",
            content="Cancellation fees apply based on fare type and timing.",
            last_updated=datetime(2024, 1, 1),
            applicable_conditions=["24 hours before departure", "Refundable fare"]
        )
    
    def test_create_completed_response(self, formatter):
        """Test creating completed response"""
        message = "Request processed successfully"
        data = {"result": "success"}
        
        response = formatter.create_completed_response(message, data)
        
        assert isinstance(response, APIResponse)
        assert response.status == "completed"
        assert response.message == message
        assert response.data == data
        assert response.error_code is None
        assert isinstance(response.timestamp, datetime)
    
    def test_create_error_response(self, formatter):
        """Test creating error response"""
        error_message = "Something went wrong"
        error_code = "TEST_ERROR"
        
        response = formatter.create_error_response(error_message, error_code)
        
        assert isinstance(response, APIResponse)
        assert response.status == "error"
        assert response.message == error_message
        assert response.error_code == error_code
        assert response.data is None
        assert isinstance(response.timestamp, datetime)
    
    def test_flight_status_response_with_booking_details(self, formatter, sample_booking_details):
        """Test flight status response formatting with booking details"""
        response = formatter.format_response(
            RequestType.FLIGHT_STATUS, 
            sample_booking_details
        )
        
        assert response.status == "completed"
        assert "ABC123" in response.message
        assert "On Time" in response.message
        assert response.data["pnr"] == "ABC123"
        assert response.data["route"]["from"] == "JFK"
        assert response.data["route"]["to"] == "LAX"
        assert response.data["current_status"] == "On Time"
        assert response.data["assigned_seat"] == "12A"
    
    def test_flight_status_response_with_flight_info(self, formatter, sample_flight_info):
        """Test flight status response formatting with flight info"""
        response = formatter.format_response(
            RequestType.FLIGHT_STATUS, 
            sample_flight_info
        )
        
        assert response.status == "completed"
        assert "AA100" in response.message
        assert "Delayed" in response.message
        assert response.data["flight_number"] == "AA100"
        assert response.data["current_status"] == "Delayed"
        assert "pnr" not in response.data  # Should not have PNR for flight info
    
    def test_cancellation_response(self, formatter, sample_cancellation_result):
        """Test cancellation response formatting"""
        response = formatter.format_response(
            RequestType.CANCEL_TRIP, 
            sample_cancellation_result
        )
        
        assert response.status == "completed"
        assert "cancelled" in response.message.lower()
        assert "$50.00" in response.message
        assert "$150.00" in response.message
        assert response.data["cancellation_charges"] == 50.0
        assert response.data["refund_amount"] == 150.0
        assert "2024-01-20" in response.data["refund_date"]
    
    def test_seat_availability_response(self, formatter, sample_seat_availability):
        """Test seat availability response formatting"""
        response = formatter.format_response(
            RequestType.SEAT_AVAILABILITY, 
            sample_seat_availability
        )
        
        assert response.status == "completed"
        assert "3 available seats" in response.message
        assert "$25.00" in response.message  # Min price
        assert "$75.00" in response.message  # Max price
        assert len(response.data["available_seats"]) == 3
        
        # Check seat formatting
        seat = response.data["available_seats"][0]
        assert seat["seat"] == "12A"
        assert seat["row"] == 12
        assert seat["column"] == "A"
        assert seat["price"] == 25.0
        assert seat["class"] == "Economy"
    
    def test_policy_response(self, formatter, sample_policy_info):
        """Test policy response formatting"""
        response = formatter.format_response(
            RequestType.CANCELLATION_POLICY, 
            sample_policy_info
        )
        
        assert response.status == "completed"
        assert "cancellation_policy information" in response.message
        assert response.data["policy_type"] == "cancellation_policy"
        assert response.data["content"] == sample_policy_info.content
        assert len(response.data["applicable_conditions"]) == 2
    
    def test_policy_response_with_string(self, formatter):
        """Test policy response formatting with string content"""
        policy_content = "Pet travel is allowed with proper documentation."
        
        response = formatter.format_response(
            RequestType.PET_TRAVEL, 
            policy_content
        )
        
        assert response.status == "completed"
        assert "policy information" in response.message
        assert response.data["policy_content"] == policy_content
        assert "retrieved_at" in response.data
    
    def test_generic_response_for_unknown_type(self, formatter):
        """Test generic response for unknown request types"""
        data = {"custom": "data"}
        
        response = formatter.format_response(
            RequestType.UNKNOWN, 
            data
        )
        
        assert response.status == "completed"
        assert "processed successfully" in response.message
        assert response.data == data
    
    def test_format_api_error_404(self, formatter):
        """Test formatting API error with 404 status"""
        error = APIError("Not found", 404)
        
        response = formatter.format_exception_response(error)
        
        assert response.status == "error"
        assert "not found" in response.message.lower()
        assert response.error_code == "BOOKING_NOT_FOUND"
    
    def test_format_api_error_400(self, formatter):
        """Test formatting API error with 400 status"""
        error = APIError("Bad request", 400)
        
        response = formatter.format_exception_response(error)
        
        assert response.status == "error"
        assert "invalid request" in response.message.lower()
        assert response.error_code == "INVALID_REQUEST"
    
    def test_format_api_error_500(self, formatter):
        """Test formatting API error with 500 status"""
        error = APIError("Internal server error", 500)
        
        response = formatter.format_exception_response(error)
        
        assert response.status == "error"
        assert "temporarily unavailable" in response.message.lower()
        assert response.error_code == "SERVICE_UNAVAILABLE"
    
    def test_format_missing_data_error_pnr(self, formatter):
        """Test formatting missing data error for PNR"""
        error = MissingDataError("pnr")
        
        response = formatter.format_exception_response(error)
        
        assert response.status == "error"
        assert "booking reference" in response.message.lower()
        assert "pnr" in response.message.lower()
        assert response.error_code == "MISSING_DATA"
    
    def test_format_missing_data_error_flight_details(self, formatter):
        """Test formatting missing data error for flight details"""
        error = MissingDataError("flight_details")
        
        response = formatter.format_exception_response(error)
        
        assert response.status == "error"
        assert "flight details" in response.message.lower()
        assert response.error_code == "MISSING_DATA"
    
    def test_format_classification_error(self, formatter):
        """Test formatting classification error"""
        error = ClassificationError("Could not classify request")
        
        response = formatter.format_exception_response(error)
        
        assert response.status == "error"
        assert "couldn't understand" in response.message.lower()
        assert response.error_code == "CLASSIFICATION_ERROR"
    
    def test_format_unexpected_error(self, formatter):
        """Test formatting unexpected error"""
        error = ValueError("Unexpected error")
        
        response = formatter.format_exception_response(error)
        
        assert response.status == "error"
        assert "unexpected error" in response.message.lower()
        assert response.error_code == "INTERNAL_ERROR"
    
    def test_empty_seat_availability(self, formatter):
        """Test seat availability response with no seats"""
        empty_availability = SeatAvailability(
            flight_id=12345,
            pnr="ABC123",
            available_seats=[]
        )
        
        response = formatter.format_response(
            RequestType.SEAT_AVAILABILITY, 
            empty_availability
        )
        
        assert response.status == "completed"
        assert "no seats" in response.message.lower()
        assert len(response.data["available_seats"]) == 0


class TestAutomatedResponseBuilder:
    """Test automated response builder functionality"""
    
    @pytest.fixture
    def formatter(self):
        """Create response formatter instance"""
        return ResponseFormatter()
    
    @pytest.fixture
    def builder(self, formatter):
        """Create automated response builder instance"""
        return AutomatedResponseBuilder(formatter)
    
    def test_build_general_info_response(self, builder):
        """Test building general information response"""
        message = "General information provided"
        data = {"info": "test"}
        
        response = builder.build_general_info_response(message, data)
        
        assert response.status == "completed"
        assert response.message == message
        assert response.data == data
    
    def test_build_guidance_response_cancel_trip(self, builder):
        """Test building guidance response for cancel trip"""
        response = builder.build_guidance_response(RequestType.CANCEL_TRIP)
        
        assert response.status == "completed"
        assert "booking reference" in response.message.lower()
        assert "pnr" in response.message.lower()
        assert response.data["request_type"] == "cancel_trip"
        assert response.data["guidance_provided"] is True
        assert len(response.data["alternative_options"]) == 3
    
    def test_build_guidance_response_flight_status(self, builder):
        """Test building guidance response for flight status"""
        response = builder.build_guidance_response(RequestType.FLIGHT_STATUS)
        
        assert response.status == "completed"
        assert "flight status" in response.message.lower()
        assert response.data["request_type"] == "flight_status"
        assert any("website" in option.lower() for option in response.data["alternative_options"])
    
    def test_build_guidance_response_unknown_type(self, builder):
        """Test building guidance response for unknown request type"""
        response = builder.build_guidance_response(RequestType.UNKNOWN)
        
        assert response.status == "completed"
        assert "assistance" in response.message.lower()
        assert response.data["request_type"] == "unknown"