"""
Response formatting service for airline customer service system.

This service provides structured response formatting for different request types
and handles error response formatting with appropriate error codes.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging

from ..types import (
    APIResponse, RequestType, ResponseFormat, BookingDetails, FlightInfo,
    CancellationResult, SeatAvailability, SeatInfo, PolicyInfo,
    AirlineServiceError, APIError, MissingDataError, ClassificationError
)

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Service for formatting responses based on request type and data.
    
    Provides structured JSON responses with consistent format across all request types.
    Handles error response formatting with appropriate error codes.
    """
    
    def __init__(self):
        """Initialize the response formatter."""
        self.response_builders = {
            RequestType.FLIGHT_STATUS: self._build_flight_status_response,
            RequestType.CANCEL_TRIP: self._build_cancellation_response,
            RequestType.SEAT_AVAILABILITY: self._build_seat_availability_response,
            RequestType.CANCELLATION_POLICY: self._build_policy_response,
            RequestType.PET_TRAVEL: self._build_policy_response,
        }
        
        # Special response builders for specific response formats
        self.format_builders = {
            ResponseFormat.BOOKING_CONFIRMATION: self._build_booking_confirmation_response,
        }
    
    def format_response(
        self, 
        request_type: RequestType, 
        data: Any, 
        message: Optional[str] = None
    ) -> APIResponse:
        """
        Format a successful response based on request type and data.
        
        Args:
            request_type: The type of request being responded to
            data: The response data to format
            message: Optional custom message
            
        Returns:
            Formatted APIResponse object
        """
        try:
            builder = self.response_builders.get(request_type)
            if not builder:
                return self._build_generic_response(data, message)
            
            return builder(data, message)
            
        except Exception as e:
            logger.error(f"Error formatting response for {request_type}: {str(e)}")
            return self.create_error_response(
                "Failed to format response",
                "FORMATTING_ERROR"
            )
    
    def create_completed_response(
        self, 
        message: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Create a completed response with message and optional data.
        
        Args:
            message: Human-readable response message
            data: Optional response data
            
        Returns:
            APIResponse with completed status
        """
        return APIResponse(
            status="completed",
            message=message,
            data=data,
            timestamp=datetime.now()
        )
    
    def create_error_response(
        self, 
        error_message: str, 
        error_code: Optional[str] = None
    ) -> APIResponse:
        """
        Create an error response with appropriate error code.
        
        Args:
            error_message: Human-readable error message
            error_code: Optional error code for programmatic handling
            
        Returns:
            APIResponse with error status
        """
        return APIResponse(
            status="error",
            message=error_message,
            error_code=error_code,
            timestamp=datetime.now()
        )
    
    def format_exception_response(self, exception: Exception) -> APIResponse:
        """
        Format an exception into an appropriate error response.
        
        Args:
            exception: The exception to format
            
        Returns:
            APIResponse with appropriate error message and code
        """
        if isinstance(exception, APIError):
            return self._handle_api_error(exception)
        elif isinstance(exception, MissingDataError):
            return self._handle_missing_data_error(exception)
        elif isinstance(exception, ClassificationError):
            return self._handle_classification_error(exception)
        else:
            logger.error(f"Unexpected error: {str(exception)}")
            return self.create_error_response(
                "An unexpected error occurred. Please try again later.",
                "INTERNAL_ERROR"
            )
    
    def _build_flight_status_response(
        self, 
        data: Union[BookingDetails, FlightInfo, Dict[str, Any]], 
        message: Optional[str] = None
    ) -> APIResponse:
        """Build response for flight status requests."""
        if isinstance(data, BookingDetails):
            formatted_data = {
                "pnr": data.pnr,
                "flight_number": getattr(data, 'flight_number', f"Flight {data.flight_id}"),
                "route": {
                    "from": data.source_airport_code,
                    "to": data.destination_airport_code
                },
                "scheduled_departure": data.scheduled_departure.isoformat(),
                "scheduled_arrival": data.scheduled_arrival.isoformat(),
                "current_departure": data.current_departure.isoformat(),
                "current_arrival": data.current_arrival.isoformat(),
                "current_status": data.current_status,
                "assigned_seat": data.assigned_seat
            }
            
            default_message = (
                f"Flight status for {data.pnr}: {data.current_status}. "
                f"Departure: {data.current_departure.strftime('%Y-%m-%d %H:%M')} "
                f"from {data.source_airport_code} to {data.destination_airport_code}."
            )
            
        elif isinstance(data, FlightInfo):
            formatted_data = {
                "flight_number": data.flight_number,
                "route": {
                    "from": data.source_airport_code,
                    "to": data.destination_airport_code
                },
                "scheduled_departure": data.scheduled_departure.isoformat(),
                "scheduled_arrival": data.scheduled_arrival.isoformat(),
                "current_status": data.current_status
            }
            
            default_message = (
                f"Flight {data.flight_number} status: {data.current_status}. "
                f"Departure: {data.scheduled_departure.strftime('%Y-%m-%d %H:%M')} "
                f"from {data.source_airport_code} to {data.destination_airport_code}."
            )
            
        else:
            # Handle dictionary data
            formatted_data = data
            default_message = "Flight status information retrieved successfully."
        
        return self.create_completed_response(
            message or default_message,
            formatted_data
        )
    
    def _build_cancellation_response(
        self, 
        data: Union[CancellationResult, Dict[str, Any]], 
        message: Optional[str] = None
    ) -> APIResponse:
        """Build response for trip cancellation requests."""
        if isinstance(data, CancellationResult):
            formatted_data = {
                "cancellation_charges": data.cancellation_charges,
                "refund_amount": data.refund_amount,
                "refund_date": data.refund_date.isoformat(),
                "message": data.message
            }
            
            default_message = (
                f"Your booking has been cancelled. "
                f"Cancellation charges: ${data.cancellation_charges:.2f}. "
                f"Refund amount: ${data.refund_amount:.2f}. "
                f"Refund date: {data.refund_date.strftime('%Y-%m-%d')}."
            )
            
        else:
            # Handle dictionary data
            formatted_data = data
            charges = data.get('cancellation_charges', 0)
            refund = data.get('refund_amount', 0)
            refund_date = data.get('refund_date', 'TBD')
            
            default_message = (
                f"Your booking has been cancelled. "
                f"Cancellation charges: ${charges:.2f}. "
                f"Refund amount: ${refund:.2f}. "
                f"Refund date: {refund_date}."
            )
        
        return self.create_completed_response(
            message or default_message,
            formatted_data
        )
    
    def _build_seat_availability_response(
        self, 
        data: Union[SeatAvailability, List[SeatInfo], Dict[str, Any]], 
        message: Optional[str] = None
    ) -> APIResponse:
        """Build response for seat availability requests."""
        if isinstance(data, SeatAvailability):
            formatted_data = {
                "flight_id": data.flight_id,
                "pnr": data.pnr,
                "available_seats": [
                    {
                        "seat": f"{seat.row_number}{seat.column_letter}",
                        "row": seat.row_number,
                        "column": seat.column_letter,
                        "price": seat.price,
                        "class": seat.seat_class
                    }
                    for seat in data.available_seats
                ]
            }
            
            seat_count = len(data.available_seats)
            default_message = (
                f"Found {seat_count} available seats on your flight. "
                f"Prices range from ${min(s.price for s in data.available_seats):.2f} "
                f"to ${max(s.price for s in data.available_seats):.2f}."
                if seat_count > 0 else "No seats are currently available on this flight."
            )
            
        elif isinstance(data, list):
            # Handle list of SeatInfo objects
            formatted_data = {
                "available_seats": [
                    {
                        "seat": f"{seat.row_number}{seat.column_letter}",
                        "row": seat.row_number,
                        "column": seat.column_letter,
                        "price": seat.price,
                        "class": seat.seat_class
                    }
                    for seat in data
                ]
            }
            
            seat_count = len(data)
            default_message = (
                f"Found {seat_count} available seats. "
                f"Prices range from ${min(s.price for s in data):.2f} "
                f"to ${max(s.price for s in data):.2f}."
                if seat_count > 0 else "No seats are currently available."
            )
            
        else:
            # Handle dictionary data
            formatted_data = data
            seat_count = len(data.get('available_seats', []))
            default_message = (
                f"Found {seat_count} available seats."
                if seat_count > 0 else "No seats are currently available."
            )
        
        return self.create_completed_response(
            message or default_message,
            formatted_data
        )
    
    def _build_policy_response(
        self, 
        data: Union[PolicyInfo, str, Dict[str, Any]], 
        message: Optional[str] = None
    ) -> APIResponse:
        """Build response for policy information requests."""
        if isinstance(data, PolicyInfo):
            formatted_data = {
                "policy_type": data.policy_type,
                "content": data.content,
                "last_updated": data.last_updated.isoformat(),
                "applicable_conditions": data.applicable_conditions
            }
            
            default_message = f"Here is the {data.policy_type} information you requested."
            
        elif isinstance(data, str):
            # Handle string policy content
            formatted_data = {
                "policy_content": data,
                "retrieved_at": datetime.now().isoformat()
            }
            
            default_message = "Here is the policy information you requested."
            
        else:
            # Handle dictionary data
            formatted_data = data
            policy_type = data.get('policy_type', 'policy')
            default_message = f"Here is the {policy_type} information you requested."
        
        return self.create_completed_response(
            message or default_message,
            formatted_data
        )
    
    def _build_generic_response(
        self, 
        data: Any, 
        message: Optional[str] = None
    ) -> APIResponse:
        """Build a generic response for unknown request types."""
        formatted_data = data if isinstance(data, dict) else {"result": data}
        default_message = "Request processed successfully."
        
        return self.create_completed_response(
            message or default_message,
            formatted_data
        )
    
    def _build_booking_confirmation_response(
        self, 
        data: Union[BookingDetails, Dict[str, Any]], 
        message: Optional[str] = None
    ) -> APIResponse:
        """Build response for booking confirmation before cancellation."""
        if isinstance(data, BookingDetails):
            formatted_data = {
                "pnr": data.pnr,
                "flight_number": getattr(data, 'flight_number', f"Flight {data.flight_id}"),
                "route": {
                    "from": data.source_airport_code,
                    "to": data.destination_airport_code
                },
                "scheduled_departure": data.scheduled_departure.isoformat(),
                "scheduled_arrival": data.scheduled_arrival.isoformat(),
                "assigned_seat": getattr(data, 'assigned_seat', None),
                "fare_type": getattr(data, 'fare_type', 'Standard'),
                "confirmation_required": True
            }
            
            default_message = (
                f"I found your booking {data.pnr} for flight from {data.source_airport_code} "
                f"to {data.destination_airport_code} on {data.scheduled_departure.strftime('%Y-%m-%d')}. "
                f"Would you like me to proceed with the cancellation?"
            )
            
        else:
            # Handle dictionary data
            formatted_data = data
            formatted_data["confirmation_required"] = True
            default_message = "Please confirm the booking details before proceeding with cancellation."
        
        return self.create_completed_response(
            message or default_message,
            formatted_data
        )
    
    def format_response_by_format(
        self, 
        response_format: ResponseFormat, 
        data: Any, 
        message: Optional[str] = None
    ) -> APIResponse:
        """
        Format response by specific response format type.
        
        Args:
            response_format: The specific response format to use
            data: The response data to format
            message: Optional custom message
            
        Returns:
            Formatted APIResponse object
        """
        try:
            builder = self.format_builders.get(response_format)
            if builder:
                return builder(data, message)
            else:
                # Fall back to generic response
                return self._build_generic_response(data, message)
                
        except Exception as e:
            logger.error(f"Error formatting response for format {response_format}: {str(e)}")
            return self.create_error_response(
                "Failed to format response",
                "FORMATTING_ERROR"
            )
    
    def _handle_api_error(self, error: APIError) -> APIResponse:
        """Handle API-specific errors with appropriate messages."""
        if error.status_code == 404:
            return self.create_error_response(
                "Booking or flight not found. Please verify your details and try again.",
                "BOOKING_NOT_FOUND"
            )
        elif error.status_code == 400:
            return self.create_error_response(
                "Invalid request. Please check your flight information and try again.",
                "INVALID_REQUEST"
            )
        elif error.status_code == 500:
            return self.create_error_response(
                "Airline service is temporarily unavailable. Please try again later.",
                "SERVICE_UNAVAILABLE"
            )
        elif error.status_code == 503:
            return self.create_error_response(
                "Service temporarily unavailable due to maintenance. Please try again later.",
                "SERVICE_MAINTENANCE"
            )
        else:
            return self.create_error_response(
                f"Airline API error: {str(error)}",
                error.error_code or "API_ERROR"
            )
    
    def _handle_missing_data_error(self, error: MissingDataError) -> APIResponse:
        """Handle missing data errors with helpful guidance."""
        data_type = error.data_type.lower()
        
        if data_type == "pnr":
            message = (
                "To process your request, please provide your booking reference (PNR) "
                "or flight number and date."
            )
        elif data_type == "flight_number":
            message = (
                "To check flight status, please provide your flight number "
                "or booking reference (PNR)."
            )
        elif data_type == "flight_details":
            message = (
                "To process your request, please provide your flight details such as "
                "booking reference (PNR), flight number, or route information."
            )
        else:
            message = f"Missing required information: {data_type}. Please provide this information to continue."
        
        return self.create_error_response(message, "MISSING_DATA")
    
    def _handle_classification_error(self, error: ClassificationError) -> APIResponse:
        """Handle classification errors when request type cannot be determined."""
        return self.create_error_response(
            "I couldn't understand your request. Please try rephrasing or specify what you need help with "
            "(flight status, cancellation, seat availability, policies, or pet travel).",
            "CLASSIFICATION_ERROR"
        )


class AutomatedResponseBuilder:
    """
    Builder for automated responses that don't require additional customer input.
    
    Creates self-contained responses based on available data and request type.
    """
    
    def __init__(self, formatter: ResponseFormatter):
        """Initialize with a response formatter instance."""
        self.formatter = formatter
    
    def build_flight_status_response(self, flight_data: Any) -> APIResponse:
        """Build automated flight status response."""
        return self.formatter._build_flight_status_response(flight_data)
    
    def build_cancellation_response(self, cancellation_result: Any) -> APIResponse:
        """Build automated cancellation response."""
        return self.formatter._build_cancellation_response(cancellation_result)
    
    def build_seat_availability_response(self, seats: Any) -> APIResponse:
        """Build automated seat availability response."""
        return self.formatter._build_seat_availability_response(seats)
    
    def build_policy_response(self, policy_info: Any) -> APIResponse:
        """Build automated policy information response."""
        return self.formatter._build_policy_response(policy_info)
    
    def build_general_info_response(self, message: str, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Build general information response."""
        return self.formatter.create_completed_response(message, data)
    
    def build_guidance_response(self, request_type: RequestType) -> APIResponse:
        """Build guidance response when specific data is not available."""
        guidance_messages = {
            RequestType.CANCEL_TRIP: (
                "To cancel your booking, please provide your booking reference (PNR) "
                "or flight number and date. You can also visit our website or call customer service."
            ),
            RequestType.FLIGHT_STATUS: (
                "To check flight status, please provide your booking reference (PNR) "
                "or flight number. You can also check our website or mobile app."
            ),
            RequestType.SEAT_AVAILABILITY: (
                "To check seat availability, please provide your booking reference (PNR) "
                "or flight details. You can also manage your booking online."
            ),
            RequestType.CANCELLATION_POLICY: (
                "Here is our general cancellation policy. For specific details about your booking, "
                "please provide your booking reference (PNR) or flight information."
            ),
            RequestType.PET_TRAVEL: (
                "Here is our pet travel policy. For specific assistance with your booking, "
                "please contact customer service or visit our website."
            )
        }
        
        message = guidance_messages.get(
            request_type, 
            "For assistance with your request, please provide more specific information or contact customer service."
        )
        
        return self.formatter.create_completed_response(
            message,
            {
                "request_type": request_type.value,
                "guidance_provided": True,
                "alternative_options": [
                    "Visit our website",
                    "Use our mobile app", 
                    "Call customer service"
                ]
            }
        )
    
    def format_response_by_format(
        self, 
        response_format: ResponseFormat, 
        data: Any, 
        message: Optional[str] = None
    ) -> APIResponse:
        """
        Format response by specific response format type.
        
        Args:
            response_format: The specific response format to use
            data: The response data to format
            message: Optional custom message
            
        Returns:
            Formatted APIResponse object
        """
        try:
            builder = self.format_builders.get(response_format)
            if builder:
                return builder(data, message)
            else:
                # Fall back to generic response
                return self._build_generic_response(data, message)
                
        except Exception as e:
            logger.error(f"Error formatting response for format {response_format}: {str(e)}")
            return self.create_error_response(
                "Failed to format response",
                "FORMATTING_ERROR"
            )