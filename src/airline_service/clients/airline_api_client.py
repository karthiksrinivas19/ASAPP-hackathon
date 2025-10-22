"""
Airline API client implementation with retry logic and error handling
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx
from enum import Enum

from ..types import (
    BookingDetails, CancellationResult, SeatAvailability, SeatInfo,
    FlightInfo, RouteInfo, CustomerSearchInfo, CustomerProfile,
    CustomerIdentifier, APIError
)
from ..interfaces.airline_api import AirlineAPIInterface
from ..config import config


class AirlineAPIError(Exception):
    """Custom exception for airline API errors"""
    
    def __init__(self, message: str, status_code: int = 500, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


class AirlineAPIClient(AirlineAPIInterface):
    """
    HTTP client for airline API with retry logic and error handling
    """
    
    def __init__(self, base_url: str = None, timeout: int = None, api_key: str = None):
        self.base_url = base_url or config.airline_api.base_url
        self.timeout = timeout or config.airline_api.timeout / 1000  # Convert to seconds
        self.api_key = api_key or config.airline_api.api_key
        
        # HTTP client configuration
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_default_headers()
        )
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AirlineCustomerService/1.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    async def _make_request(
        self, 
        method: HTTPMethod, 
        endpoint: str, 
        params: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with error handling
        """
        try:
            response = await self.client.request(
                method=method.value,
                url=endpoint,
                params=params,
                json=json_data
            )
            
            # Handle different status codes
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 400:
                error_data = response.json() if response.content else {}
                raise AirlineAPIError(
                    message=error_data.get("message", "Invalid Request"),
                    status_code=400,
                    error_code="INVALID_REQUEST"
                )
            
            elif response.status_code == 404:
                error_data = response.json() if response.content else {}
                raise AirlineAPIError(
                    message=error_data.get("message", "Not Found"),
                    status_code=404,
                    error_code="NOT_FOUND"
                )
            
            else:
                # Handle other status codes
                raise AirlineAPIError(
                    message=f"API request failed with status {response.status_code}",
                    status_code=response.status_code,
                    error_code="API_ERROR"
                )
        
        except httpx.TimeoutException:
            raise AirlineAPIError(
                message="Request timeout",
                status_code=408,
                error_code="TIMEOUT"
            )
        
        except httpx.ConnectError:
            raise AirlineAPIError(
                message="Connection error - unable to reach airline API",
                status_code=503,
                error_code="CONNECTION_ERROR"
            )
        
        except Exception as e:
            if isinstance(e, AirlineAPIError):
                raise
            
            raise AirlineAPIError(
                message=f"Unexpected error: {str(e)}",
                status_code=500,
                error_code="INTERNAL_ERROR"
            )
    
    async def get_booking_details(self, pnr: str) -> BookingDetails:
        """
        Get booking details by PNR
        
        GET /flight/booking?pnr={pnr}
        """
        if not pnr or len(pnr) != 6:
            raise AirlineAPIError(
                message="Invalid PNR format. PNR must be 6 characters.",
                status_code=400,
                error_code="INVALID_PNR"
            )
        
        params = {"pnr": pnr.upper()}
        
        try:
            response_data = await self._make_request(
                method=HTTPMethod.GET,
                endpoint="/flight/booking",
                params=params
            )
            
            # Parse response into BookingDetails model
            return BookingDetails(
                pnr=response_data["pnr"],
                flight_id=response_data["flight_id"],
                source_airport_code=response_data["source_airport_code"],
                destination_airport_code=response_data["destination_airport_code"],
                scheduled_departure=datetime.fromisoformat(response_data["scheduled_departure"]),
                scheduled_arrival=datetime.fromisoformat(response_data["scheduled_arrival"]),
                assigned_seat=response_data["assigned_seat"],
                current_departure=datetime.fromisoformat(response_data["current_departure"]),
                current_arrival=datetime.fromisoformat(response_data["current_arrival"]),
                current_status=response_data["current_status"]
            )
        
        except AirlineAPIError as e:
            if e.status_code == 404:
                raise AirlineAPIError(
                    message=f"Booking not found for PNR: {pnr}",
                    status_code=404,
                    error_code="BOOKING_NOT_FOUND"
                )
            raise
    
    async def cancel_flight(self, booking_details: BookingDetails) -> CancellationResult:
        """
        Cancel flight booking
        
        POST /flight/cancel
        """
        request_data = {
            "pnr": booking_details.pnr,
            "flight_id": booking_details.flight_id,
            "source_airport_code": booking_details.source_airport_code,
            "destination_airport_code": booking_details.destination_airport_code,
            "scheduled_departure": booking_details.scheduled_departure.isoformat(),
            "scheduled_arrival": booking_details.scheduled_arrival.isoformat()
        }
        
        try:
            response_data = await self._make_request(
                method=HTTPMethod.POST,
                endpoint="/flight/cancel",
                json_data=request_data
            )
            
            return CancellationResult(
                message=response_data["message"],
                cancellation_charges=response_data["cancellation_charges"],
                refund_amount=response_data["refund_amount"],
                refund_date=datetime.fromisoformat(response_data["refund_date"])
            )
        
        except AirlineAPIError as e:
            if e.status_code == 404:
                raise AirlineAPIError(
                    message=f"Booking not found for cancellation: {booking_details.pnr}",
                    status_code=404,
                    error_code="BOOKING_NOT_FOUND"
                )
            elif e.status_code == 400:
                raise AirlineAPIError(
                    message="Invalid cancellation request. Please check booking details.",
                    status_code=400,
                    error_code="INVALID_CANCELLATION"
                )
            raise
    
    async def get_available_seats(self, flight_info: FlightInfo) -> SeatAvailability:
        """
        Get available seats for flight
        
        POST /flight/available_seats
        """
        request_data = {
            "pnr": getattr(flight_info, 'pnr', ''),  # PNR might not be in FlightInfo
            "flight_id": flight_info.flight_id,
            "source_airport_code": flight_info.source_airport_code,
            "destination_airport_code": flight_info.destination_airport_code,
            "scheduled_departure": flight_info.scheduled_departure.isoformat(),
            "scheduled_arrival": flight_info.scheduled_arrival.isoformat()
        }
        
        try:
            response_data = await self._make_request(
                method=HTTPMethod.POST,
                endpoint="/flight/available_seats",
                json_data=request_data
            )
            
            # Parse seat information
            available_seats = []
            for seat_data in response_data["available_seats"]:
                seat_info = SeatInfo(
                    row_number=seat_data["row_number"],
                    column_letter=seat_data["column_letter"],
                    price=seat_data["price"],
                    seat_class=seat_data["class"]
                )
                available_seats.append(seat_info)
            
            return SeatAvailability(
                flight_id=response_data["flight_id"],
                pnr=response_data["pnr"],
                available_seats=available_seats
            )
        
        except AirlineAPIError as e:
            if e.status_code == 404:
                raise AirlineAPIError(
                    message=f"Flight not found: {flight_info.flight_id}",
                    status_code=404,
                    error_code="FLIGHT_NOT_FOUND"
                )
            raise
    
    async def search_bookings_by_flight(self, flight_number: str, date: datetime) -> List[BookingDetails]:
        """
        Search bookings by flight number and date
        
        Note: This is an enhanced endpoint that may not exist in the basic API
        """
        # This would be implemented when the enhanced API is available
        raise NotImplementedError("Enhanced search endpoints not yet implemented")
    
    async def search_bookings_by_route(
        self, 
        route: RouteInfo, 
        date: datetime, 
        passenger_name: Optional[str] = None
    ) -> List[BookingDetails]:
        """
        Search bookings by route and date
        
        Note: This is an enhanced endpoint that may not exist in the basic API
        """
        # This would be implemented when the enhanced API is available
        raise NotImplementedError("Enhanced search endpoints not yet implemented")
    
    async def get_flights_by_route(self, source: str, destination: str, date: datetime) -> List[FlightInfo]:
        """
        Get flights by route and date
        
        Note: This is an enhanced endpoint that may not exist in the basic API
        """
        # This would be implemented when the enhanced API is available
        raise NotImplementedError("Enhanced search endpoints not yet implemented")
    
    async def get_flight_by_number(self, flight_number: str, date: datetime) -> FlightInfo:
        """
        Get flight info by flight number and date
        
        Note: This is an enhanced endpoint that may not exist in the basic API
        """
        # This would be implemented when the enhanced API is available
        raise NotImplementedError("Enhanced search endpoints not yet implemented")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the airline API is healthy
        """
        try:
            # Try a simple request to check API health
            response_data = await self._make_request(
                method=HTTPMethod.GET,
                endpoint="/health"  # Assuming the API has a health endpoint
            )
            return {
                "status": "healthy",
                "api_available": True,
                "response_time_ms": response_data.get("response_time", 0)
            }
        
        except AirlineAPIError:
            return {
                "status": "unhealthy",
                "api_available": False,
                "error": "API not responding"
            }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            asyncio.create_task(self.close())
        except:
            pass  # Ignore errors during cleanup


class MockAirlineAPIClient(AirlineAPIInterface):
    """
    Mock airline API client for testing and development
    """
    
    def __init__(self):
        self.mock_bookings = self._generate_mock_bookings()
    
    def _generate_mock_bookings(self) -> Dict[str, BookingDetails]:
        """Generate mock booking data"""
        return {
            "ABC123": BookingDetails(
                pnr="ABC123",
                flight_id=1001,
                source_airport_code="JFK",
                destination_airport_code="LAX",
                scheduled_departure=datetime(2024, 1, 15, 10, 30),
                scheduled_arrival=datetime(2024, 1, 15, 13, 45),
                assigned_seat="12A",
                current_departure=datetime(2024, 1, 15, 10, 30),
                current_arrival=datetime(2024, 1, 15, 13, 45),
                current_status="On Time"
            ),
            "XYZ789": BookingDetails(
                pnr="XYZ789",
                flight_id=2002,
                source_airport_code="ORD",
                destination_airport_code="MIA",
                scheduled_departure=datetime(2024, 1, 16, 14, 20),
                scheduled_arrival=datetime(2024, 1, 16, 17, 30),
                assigned_seat="8C",
                current_departure=datetime(2024, 1, 16, 14, 35),
                current_arrival=datetime(2024, 1, 16, 17, 45),
                current_status="Delayed"
            )
        }
    
    async def get_booking_details(self, pnr: str) -> BookingDetails:
        """Mock get booking details"""
        pnr = pnr.upper()
        
        if pnr in self.mock_bookings:
            return self.mock_bookings[pnr]
        
        raise AirlineAPIError(
            message=f"PNR Not Found: {pnr}",
            status_code=404,
            error_code="PNR_NOT_FOUND"
        )
    
    async def cancel_flight(self, booking_details: BookingDetails) -> CancellationResult:
        """Mock cancel flight"""
        return CancellationResult(
            message="Flight Cancelled",
            cancellation_charges=50.0,
            refund_amount=150.0,
            refund_date=datetime(2024, 1, 20)
        )
    
    async def get_available_seats(self, flight_info: FlightInfo) -> SeatAvailability:
        """Mock get available seats"""
        mock_seats = [
            SeatInfo(row_number=10, column_letter="A", price=0.0, seat_class="economy"),
            SeatInfo(row_number=10, column_letter="B", price=0.0, seat_class="economy"),
            SeatInfo(row_number=15, column_letter="F", price=25.0, seat_class="economy"),
            SeatInfo(row_number=5, column_letter="A", price=75.0, seat_class="business"),
        ]
        
        return SeatAvailability(
            flight_id=flight_info.flight_id,
            pnr="MOCK01",
            available_seats=mock_seats
        )
    
    async def search_bookings_by_flight(self, flight_number: str, date: datetime) -> List[BookingDetails]:
        """Mock search by flight"""
        return list(self.mock_bookings.values())[:1]  # Return first booking
    
    async def search_bookings_by_route(
        self, 
        route: RouteInfo, 
        date: datetime, 
        passenger_name: Optional[str] = None
    ) -> List[BookingDetails]:
        """Mock search by route"""
        return list(self.mock_bookings.values())
    
    async def get_flights_by_route(self, source: str, destination: str, date: datetime) -> List[FlightInfo]:
        """Mock get flights by route"""
        return [
            FlightInfo(
                flight_id=1001,
                flight_number="AA100",
                source_airport_code=source,
                destination_airport_code=destination,
                scheduled_departure=datetime(2024, 1, 15, 10, 30),
                scheduled_arrival=datetime(2024, 1, 15, 13, 45),
                current_status="On Time"
            )
        ]
    
    async def get_flight_by_number(self, flight_number: str, date: datetime) -> FlightInfo:
        """Mock get flight by number"""
        return FlightInfo(
            flight_id=1001,
            flight_number=flight_number,
            source_airport_code="JFK",
            destination_airport_code="LAX",
            scheduled_departure=datetime(2024, 1, 15, 10, 30),
            scheduled_arrival=datetime(2024, 1, 15, 13, 45),
            current_status="On Time"
        )