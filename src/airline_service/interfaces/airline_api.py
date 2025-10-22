"""
Airline API interface definitions
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from ..types import (
    BookingDetails,
    CancellationResult,
    SeatAvailability,
    FlightInfo,
    RouteInfo,
    CustomerSearchInfo,
    CustomerProfile,
    CustomerIdentifier
)


class AirlineAPIInterface(ABC):
    """Interface for basic airline API operations"""
    
    @abstractmethod
    async def get_booking_details(self, pnr: str) -> BookingDetails:
        """Get booking details by PNR"""
        pass
    
    @abstractmethod
    async def cancel_flight(self, booking_details: BookingDetails) -> CancellationResult:
        """Cancel flight booking"""
        pass
    
    @abstractmethod
    async def get_available_seats(self, flight_info: FlightInfo) -> SeatAvailability:
        """Get available seats for flight"""
        pass
    
    @abstractmethod
    async def search_bookings_by_flight(self, flight_number: str, date: datetime) -> List[BookingDetails]:
        """Search bookings by flight number and date"""
        pass
    
    @abstractmethod
    async def search_bookings_by_route(
        self, 
        route: RouteInfo, 
        date: datetime, 
        passenger_name: Optional[str] = None
    ) -> List[BookingDetails]:
        """Search bookings by route and date"""
        pass
    
    @abstractmethod
    async def get_flights_by_route(self, source: str, destination: str, date: datetime) -> List[FlightInfo]:
        """Get flights by route and date"""
        pass
    
    @abstractmethod
    async def get_flight_by_number(self, flight_number: str, date: datetime) -> FlightInfo:
        """Get flight info by flight number and date"""
        pass


class EnhancedAirlineAPIInterface(ABC):
    """Interface for enhanced airline API operations"""
    
    @abstractmethod
    async def search_bookings_by_customer(self, customer_info: CustomerSearchInfo) -> List[BookingDetails]:
        """Search bookings by customer information"""
        pass
    
    @abstractmethod
    async def get_recent_bookings(self, customer_id: str, days: int) -> List[BookingDetails]:
        """Get recent bookings for customer"""
        pass
    
    @abstractmethod
    async def search_bookings_by_partial_info(self, search_criteria: dict) -> List[BookingDetails]:
        """Search bookings by partial information"""
        pass
    
    @abstractmethod
    async def get_customer_profile(self, identifier: CustomerIdentifier) -> CustomerProfile:
        """Get customer profile and recent activity"""
        pass


class APIErrorHandlerInterface(ABC):
    """Interface for API error handling"""
    
    @abstractmethod
    async def handle_error(self, error: Exception) -> dict:
        """Handle API errors"""
        pass
    
    @abstractmethod
    def should_retry(self, error: Exception) -> bool:
        """Determine if error should trigger retry"""
        pass
    
    @abstractmethod
    def get_retry_delay(self, attempt_number: int) -> float:
        """Get delay for retry attempt"""
        pass


class ExponentialBackoffRetryInterface(ABC):
    """Interface for exponential backoff retry logic"""
    
    @abstractmethod
    async def execute_with_retry(self, operation, *args, **kwargs):
        """Execute operation with retry logic"""
        pass