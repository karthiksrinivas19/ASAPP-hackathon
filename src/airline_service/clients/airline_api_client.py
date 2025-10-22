"""
Airline API client implementation with retry logic and error handling
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx
from enum import Enum

from ..types import (
    BookingDetails, CancellationResult, SeatAvailability, SeatInfo,
    FlightInfo, RouteInfo, CustomerSearchInfo, CustomerProfile,
    CustomerIdentifier, APIError
)
from ..interfaces.airline_api import AirlineAPIInterface, EnhancedAirlineAPIInterface
from ..config import config
from ..services.connection_pool import pooled_http_client
from ..services.cache_service import api_cache


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


class AirlineAPIClient(AirlineAPIInterface, EnhancedAirlineAPIInterface):
    """
    HTTP client for airline API with retry logic and error handling
    """
    
    def __init__(self, base_url: str = None, timeout: int = None, api_key: str = None):
        self.base_url = base_url or config.airline_api.base_url
        self.timeout = timeout or config.airline_api.timeout / 1000  # Convert to seconds
        self.api_key = api_key or config.airline_api.api_key
        
        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # 1 second base delay
        self.circuit_breaker_threshold = 5  # Number of consecutive failures before circuit opens
        self.circuit_breaker_timeout = 60  # Seconds to wait before trying again
        self.consecutive_failures = 0
        self.circuit_open_time = None
        
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
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_open_time is None:
            return False
        
        # Check if enough time has passed to try again
        if time.time() - self.circuit_open_time > self.circuit_breaker_timeout:
            self.circuit_open_time = None
            self.consecutive_failures = 0
            return False
        
        return True
    
    def _should_retry(self, error: AirlineAPIError) -> bool:
        """Determine if request should be retried based on error type"""
        # Don't retry client errors (400, 404) - these are permanent
        if error.status_code in [400, 404]:
            return False
        
        # Retry server errors, timeouts, and connection errors
        return error.status_code in [500, 502, 503, 504, 408] or error.error_code in ["TIMEOUT", "CONNECTION_ERROR"]
    
    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        return self.base_delay * (2 ** attempt)
    
    async def _make_request_with_retry(
        self, 
        method: HTTPMethod, 
        endpoint: str, 
        params: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and circuit breaker
        """
        # Check circuit breaker
        if self._is_circuit_open():
            raise AirlineAPIError(
                message="Service temporarily unavailable - circuit breaker open",
                status_code=503,
                error_code="CIRCUIT_BREAKER_OPEN"
            )
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                response = await self._make_request_single(method, endpoint, params, json_data)
                
                # Success - reset circuit breaker
                self.consecutive_failures = 0
                self.circuit_open_time = None
                
                return response
                
            except AirlineAPIError as e:
                last_error = e
                
                # Don't retry on final attempt or non-retryable errors
                if attempt == self.max_retries or not self._should_retry(e):
                    # Track failures for circuit breaker
                    if self._should_retry(e):
                        self.consecutive_failures += 1
                        if self.consecutive_failures >= self.circuit_breaker_threshold:
                            self.circuit_open_time = time.time()
                    
                    raise e
                
                # Calculate delay and wait before retry
                delay = self._get_retry_delay(attempt)
                await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_error
    
    async def _make_request_single(
        self, 
        method: HTTPMethod, 
        endpoint: str, 
        params: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make single HTTP request with error handling (no retry logic)
        """
        try:
            # Use pooled HTTP client for better performance
            if method == HTTPMethod.GET:
                response = await pooled_http_client.get(
                    url=endpoint,
                    base_url=self.base_url,
                    params=params
                )
            elif method == HTTPMethod.POST:
                response = await pooled_http_client.post(
                    url=endpoint,
                    base_url=self.base_url,
                    json=json_data
                )
            elif method == HTTPMethod.PUT:
                response = await pooled_http_client.put(
                    url=endpoint,
                    base_url=self.base_url,
                    json=json_data
                )
            elif method == HTTPMethod.DELETE:
                response = await pooled_http_client.delete(
                    url=endpoint,
                    base_url=self.base_url
                )
            else:
                # Fallback to generic request
                response = await pooled_http_client.request(
                    method=method.value,
                    url=endpoint,
                    base_url=self.base_url,
                    params=params,
                    json=json_data
                )
            
            # Handle different status codes
            if response.status == 200:
                return await response.json()
            
            elif response.status == 400:
                try:
                    error_data = await response.json()
                except:
                    error_data = {}
                raise AirlineAPIError(
                    message=error_data.get("message", "Invalid Request"),
                    status_code=400,
                    error_code="INVALID_REQUEST"
                )
            
            elif response.status == 404:
                try:
                    error_data = await response.json()
                except:
                    error_data = {}
                raise AirlineAPIError(
                    message=error_data.get("message", "Not Found"),
                    status_code=404,
                    error_code="NOT_FOUND"
                )
            
            else:
                # Handle other status codes
                raise AirlineAPIError(
                    message=f"API request failed with status {response.status}",
                    status_code=response.status,
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
    
    async def _make_request(
        self, 
        method: HTTPMethod, 
        endpoint: str, 
        params: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic, error handling, and caching
        """
        # Check cache for GET requests
        if use_cache and method == HTTPMethod.GET:
            cached_response = await api_cache.get_api_response(
                endpoint=endpoint,
                method=method.value,
                params=params
            )
            if cached_response:
                return cached_response
        
        # Make request with retry logic
        response_data = await self._make_request_with_retry(method, endpoint, params, json_data)
        
        # Cache successful GET responses
        if use_cache and method == HTTPMethod.GET and response_data:
            await api_cache.set_api_response(
                endpoint=endpoint,
                method=method.value,
                response_data=response_data,
                params=params
            )
        
        return response_data
    
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
            
            # Validate response contains required fields
            required_fields = [
                "pnr", "flight_id", "source_airport_code", "destination_airport_code",
                "scheduled_departure", "scheduled_arrival", "assigned_seat",
                "current_departure", "current_arrival", "current_status"
            ]
            
            missing_fields = [field for field in required_fields if field not in response_data]
            if missing_fields:
                raise AirlineAPIError(
                    message=f"Invalid API response - missing fields: {missing_fields}",
                    status_code=500,
                    error_code="INVALID_RESPONSE"
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
                    **{"class": seat_data["class"]}
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
    
    async def search_bookings_by_customer(self, customer_info: CustomerSearchInfo) -> List[BookingDetails]:
        """
        Search bookings by customer information (phone, email, name)
        
        POST /flight/search/customer
        """
        # Build search parameters
        search_params = {}
        if customer_info.phone:
            search_params["phone"] = customer_info.phone
        if customer_info.email:
            search_params["email"] = customer_info.email
        if customer_info.name:
            search_params["name"] = customer_info.name
        if customer_info.date_range:
            if "from" in customer_info.date_range:
                search_params["date_from"] = customer_info.date_range["from"].isoformat()
            if "to" in customer_info.date_range:
                search_params["date_to"] = customer_info.date_range["to"].isoformat()
        
        if not search_params:
            raise AirlineAPIError(
                message="At least one search parameter (phone, email, or name) is required",
                status_code=400,
                error_code="MISSING_SEARCH_PARAMS"
            )
        
        try:
            response_data = await self._make_request(
                method=HTTPMethod.POST,
                endpoint="/flight/search/customer",
                json_data=search_params
            )
            
            # Validate response contains bookings array
            if "bookings" not in response_data:
                raise AirlineAPIError(
                    message="Invalid API response - missing bookings array",
                    status_code=500,
                    error_code="INVALID_RESPONSE"
                )
            
            # Parse bookings
            bookings = []
            for booking_data in response_data["bookings"]:
                # Validate each booking has required fields
                required_fields = [
                    "pnr", "flight_id", "source_airport_code", "destination_airport_code",
                    "scheduled_departure", "scheduled_arrival", "assigned_seat",
                    "current_departure", "current_arrival", "current_status"
                ]
                
                missing_fields = [field for field in required_fields if field not in booking_data]
                if missing_fields:
                    continue  # Skip invalid bookings
                
                booking = BookingDetails(
                    pnr=booking_data["pnr"],
                    flight_id=booking_data["flight_id"],
                    source_airport_code=booking_data["source_airport_code"],
                    destination_airport_code=booking_data["destination_airport_code"],
                    scheduled_departure=datetime.fromisoformat(booking_data["scheduled_departure"]),
                    scheduled_arrival=datetime.fromisoformat(booking_data["scheduled_arrival"]),
                    assigned_seat=booking_data["assigned_seat"],
                    current_departure=datetime.fromisoformat(booking_data["current_departure"]),
                    current_arrival=datetime.fromisoformat(booking_data["current_arrival"]),
                    current_status=booking_data["current_status"]
                )
                bookings.append(booking)
            
            return bookings
            
        except AirlineAPIError as e:
            if e.status_code == 404:
                # No bookings found - return empty list
                return []
            raise
    
    async def get_recent_bookings(self, customer_id: str, days: int = 30) -> List[BookingDetails]:
        """
        Get recent bookings for a customer
        
        GET /flight/customer/{customer_id}/bookings?days={days}
        """
        if not customer_id:
            raise AirlineAPIError(
                message="Customer ID is required",
                status_code=400,
                error_code="MISSING_CUSTOMER_ID"
            )
        
        params = {"days": days}
        
        try:
            response_data = await self._make_request(
                method=HTTPMethod.GET,
                endpoint=f"/flight/customer/{customer_id}/bookings",
                params=params
            )
            
            # Validate response contains bookings array
            if "bookings" not in response_data:
                raise AirlineAPIError(
                    message="Invalid API response - missing bookings array",
                    status_code=500,
                    error_code="INVALID_RESPONSE"
                )
            
            # Parse bookings (same logic as search_bookings_by_customer)
            bookings = []
            for booking_data in response_data["bookings"]:
                required_fields = [
                    "pnr", "flight_id", "source_airport_code", "destination_airport_code",
                    "scheduled_departure", "scheduled_arrival", "assigned_seat",
                    "current_departure", "current_arrival", "current_status"
                ]
                
                missing_fields = [field for field in required_fields if field not in booking_data]
                if missing_fields:
                    continue  # Skip invalid bookings
                
                booking = BookingDetails(
                    pnr=booking_data["pnr"],
                    flight_id=booking_data["flight_id"],
                    source_airport_code=booking_data["source_airport_code"],
                    destination_airport_code=booking_data["destination_airport_code"],
                    scheduled_departure=datetime.fromisoformat(booking_data["scheduled_departure"]),
                    scheduled_arrival=datetime.fromisoformat(booking_data["scheduled_arrival"]),
                    assigned_seat=booking_data["assigned_seat"],
                    current_departure=datetime.fromisoformat(booking_data["current_departure"]),
                    current_arrival=datetime.fromisoformat(booking_data["current_arrival"]),
                    current_status=booking_data["current_status"]
                )
                bookings.append(booking)
            
            return bookings
            
        except AirlineAPIError as e:
            if e.status_code == 404:
                # Customer not found or no bookings - return empty list
                return []
            raise
    
    async def search_bookings_by_flight(self, flight_number: str, date: datetime) -> List[BookingDetails]:
        """
        Search bookings by flight number and date
        
        GET /flight/search/flight?flight_number={flight_number}&date={date}
        """
        if not flight_number:
            raise AirlineAPIError(
                message="Flight number is required",
                status_code=400,
                error_code="MISSING_FLIGHT_NUMBER"
            )
        
        params = {
            "flight_number": flight_number,
            "date": date.isoformat()
        }
        
        try:
            response_data = await self._make_request(
                method=HTTPMethod.GET,
                endpoint="/flight/search/flight",
                params=params
            )
            
            # Validate response contains bookings array
            if "bookings" not in response_data:
                raise AirlineAPIError(
                    message="Invalid API response - missing bookings array",
                    status_code=500,
                    error_code="INVALID_RESPONSE"
                )
            
            # Parse bookings (same logic as other search methods)
            bookings = []
            for booking_data in response_data["bookings"]:
                required_fields = [
                    "pnr", "flight_id", "source_airport_code", "destination_airport_code",
                    "scheduled_departure", "scheduled_arrival", "assigned_seat",
                    "current_departure", "current_arrival", "current_status"
                ]
                
                missing_fields = [field for field in required_fields if field not in booking_data]
                if missing_fields:
                    continue  # Skip invalid bookings
                
                booking = BookingDetails(
                    pnr=booking_data["pnr"],
                    flight_id=booking_data["flight_id"],
                    source_airport_code=booking_data["source_airport_code"],
                    destination_airport_code=booking_data["destination_airport_code"],
                    scheduled_departure=datetime.fromisoformat(booking_data["scheduled_departure"]),
                    scheduled_arrival=datetime.fromisoformat(booking_data["scheduled_arrival"]),
                    assigned_seat=booking_data["assigned_seat"],
                    current_departure=datetime.fromisoformat(booking_data["current_departure"]),
                    current_arrival=datetime.fromisoformat(booking_data["current_arrival"]),
                    current_status=booking_data["current_status"]
                )
                bookings.append(booking)
            
            return bookings
            
        except AirlineAPIError as e:
            if e.status_code == 404:
                # Flight not found or no bookings - return empty list
                return []
            raise
    
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
    
    async def search_bookings_by_partial_info(self, search_criteria: dict) -> List[BookingDetails]:
        """
        Search bookings by partial information
        
        POST /flight/search/partial
        """
        if not search_criteria:
            raise AirlineAPIError(
                message="Search criteria cannot be empty",
                status_code=400,
                error_code="EMPTY_SEARCH_CRITERIA"
            )
        
        try:
            response_data = await self._make_request(
                method=HTTPMethod.POST,
                endpoint="/flight/search/partial",
                json_data=search_criteria
            )
            
            # Validate response contains bookings array
            if "bookings" not in response_data:
                raise AirlineAPIError(
                    message="Invalid API response - missing bookings array",
                    status_code=500,
                    error_code="INVALID_RESPONSE"
                )
            
            # Parse bookings
            bookings = []
            for booking_data in response_data["bookings"]:
                required_fields = [
                    "pnr", "flight_id", "source_airport_code", "destination_airport_code",
                    "scheduled_departure", "scheduled_arrival", "assigned_seat",
                    "current_departure", "current_arrival", "current_status"
                ]
                
                missing_fields = [field for field in required_fields if field not in booking_data]
                if missing_fields:
                    continue  # Skip invalid bookings
                
                booking = BookingDetails(
                    pnr=booking_data["pnr"],
                    flight_id=booking_data["flight_id"],
                    source_airport_code=booking_data["source_airport_code"],
                    destination_airport_code=booking_data["destination_airport_code"],
                    scheduled_departure=datetime.fromisoformat(booking_data["scheduled_departure"]),
                    scheduled_arrival=datetime.fromisoformat(booking_data["scheduled_arrival"]),
                    assigned_seat=booking_data["assigned_seat"],
                    current_departure=datetime.fromisoformat(booking_data["current_departure"]),
                    current_arrival=datetime.fromisoformat(booking_data["current_arrival"]),
                    current_status=booking_data["current_status"]
                )
                bookings.append(booking)
            
            return bookings
            
        except AirlineAPIError as e:
            if e.status_code == 404:
                return []
            raise
    
    async def get_customer_profile(self, identifier: CustomerIdentifier) -> CustomerProfile:
        """
        Get customer profile and recent activity
        
        POST /customer/profile
        """
        # Build identifier parameters
        identifier_params = {}
        if identifier.phone:
            identifier_params["phone"] = identifier.phone
        if identifier.email:
            identifier_params["email"] = identifier.email
        if identifier.customer_id:
            identifier_params["customer_id"] = identifier.customer_id
        if identifier.loyalty_number:
            identifier_params["loyalty_number"] = identifier.loyalty_number
        
        if not identifier_params:
            raise AirlineAPIError(
                message="At least one customer identifier is required",
                status_code=400,
                error_code="MISSING_IDENTIFIER"
            )
        
        try:
            response_data = await self._make_request(
                method=HTTPMethod.POST,
                endpoint="/customer/profile",
                json_data=identifier_params
            )
            
            # Validate response contains required fields
            required_fields = ["customer_id", "recent_bookings", "upcoming_flights", "preferences"]
            missing_fields = [field for field in required_fields if field not in response_data]
            if missing_fields:
                raise AirlineAPIError(
                    message=f"Invalid API response - missing fields: {missing_fields}",
                    status_code=500,
                    error_code="INVALID_RESPONSE"
                )
            
            # Parse recent bookings
            recent_bookings = []
            for booking_data in response_data["recent_bookings"]:
                booking = BookingDetails(
                    pnr=booking_data["pnr"],
                    flight_id=booking_data["flight_id"],
                    source_airport_code=booking_data["source_airport_code"],
                    destination_airport_code=booking_data["destination_airport_code"],
                    scheduled_departure=datetime.fromisoformat(booking_data["scheduled_departure"]),
                    scheduled_arrival=datetime.fromisoformat(booking_data["scheduled_arrival"]),
                    assigned_seat=booking_data["assigned_seat"],
                    current_departure=datetime.fromisoformat(booking_data["current_departure"]),
                    current_arrival=datetime.fromisoformat(booking_data["current_arrival"]),
                    current_status=booking_data["current_status"]
                )
                recent_bookings.append(booking)
            
            # Parse upcoming flights
            upcoming_flights = []
            for booking_data in response_data["upcoming_flights"]:
                booking = BookingDetails(
                    pnr=booking_data["pnr"],
                    flight_id=booking_data["flight_id"],
                    source_airport_code=booking_data["source_airport_code"],
                    destination_airport_code=booking_data["destination_airport_code"],
                    scheduled_departure=datetime.fromisoformat(booking_data["scheduled_departure"]),
                    scheduled_arrival=datetime.fromisoformat(booking_data["scheduled_arrival"]),
                    assigned_seat=booking_data["assigned_seat"],
                    current_departure=datetime.fromisoformat(booking_data["current_departure"]),
                    current_arrival=datetime.fromisoformat(booking_data["current_arrival"]),
                    current_status=booking_data["current_status"]
                )
                upcoming_flights.append(booking)
            
            return CustomerProfile(
                customer_id=response_data["customer_id"],
                recent_bookings=recent_bookings,
                upcoming_flights=upcoming_flights,
                preferences=response_data["preferences"]
            )
            
        except AirlineAPIError as e:
            if e.status_code == 404:
                raise AirlineAPIError(
                    message="Customer not found",
                    status_code=404,
                    error_code="CUSTOMER_NOT_FOUND"
                )
            raise
    
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
            # Only attempt cleanup if there's an active event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.close())
            else:
                # If no event loop, try to run the cleanup synchronously
                asyncio.run(self.close())
        except:
            pass  # Ignore errors during cleanup


class MockAirlineAPIClient(AirlineAPIInterface, EnhancedAirlineAPIInterface):
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
            SeatInfo(row_number=10, column_letter="A", price=0.0, **{"class": "economy"}),
            SeatInfo(row_number=10, column_letter="B", price=0.0, **{"class": "economy"}),
            SeatInfo(row_number=15, column_letter="F", price=25.0, **{"class": "economy"}),
            SeatInfo(row_number=5, column_letter="A", price=75.0, **{"class": "business"}),
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
    
    async def search_bookings_by_customer(self, customer_info: CustomerSearchInfo) -> List[BookingDetails]:
        """Mock search bookings by customer"""
        # Return mock bookings based on search criteria
        if customer_info.phone == "555-1234" or customer_info.email == "john@example.com" or customer_info.name == "John Smith":
            return [self.mock_bookings["ABC123"]]
        elif customer_info.phone == "555-5678" or customer_info.email == "jane@example.com" or customer_info.name == "Jane Doe":
            return [self.mock_bookings["XYZ789"]]
        else:
            return []  # No bookings found
    
    async def get_recent_bookings(self, customer_id: str, days: int = 30) -> List[BookingDetails]:
        """Mock get recent bookings"""
        # Return mock bookings for known customer IDs
        if customer_id in ["CUST001", "john.smith"]:
            return [self.mock_bookings["ABC123"]]
        elif customer_id in ["CUST002", "jane.doe"]:
            return [self.mock_bookings["XYZ789"]]
        else:
            return []
    
    async def search_bookings_by_partial_info(self, search_criteria: dict) -> List[BookingDetails]:
        """Mock search bookings by partial info"""
        # Simple mock logic - return bookings if any criteria match
        if any(key in search_criteria for key in ["pnr", "flight_number", "phone", "email", "name"]):
            return list(self.mock_bookings.values())
        return []
    
    async def get_customer_profile(self, identifier: CustomerIdentifier) -> CustomerProfile:
        """Mock get customer profile"""
        # Return mock customer profile
        customer_id = identifier.customer_id or "CUST001"
        
        return CustomerProfile(
            customer_id=customer_id,
            recent_bookings=[self.mock_bookings["ABC123"]],
            upcoming_flights=[self.mock_bookings["XYZ789"]],
            preferences={
                "seat_preference": "window",
                "meal_preference": "vegetarian",
                "notification_method": "email"
            }
        )