"""
Airline API Client for external airline system integration
"""

import asyncio
import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from ..types import BookingDetails, FlightDetails, SeatInfo
from ..config import config


class AirlineAPIError(Exception):
    """Custom exception for airline API errors"""
    
    def __init__(self, message: str, status_code: int = 500, error_code: str = "API_ERROR"):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


class BookingResponse(BaseModel):
    """Response model for booking details"""
    pnr: str
    flight_id: str
    source_airport_code: str
    destination_airport_code: str
    scheduled_departure: str
    scheduled_arrival: str
    flight_status: str
    seat_number: Optional[str] = None
    fare_type: Optional[str] = None


class CancellationResponse(BaseModel):
    """Response model for flight cancellation"""
    status: str
    message: str
    refund_amount: Optional[float] = None
    cancellation_charges: Optional[float] = None
    refund_date: Optional[str] = None


class SeatAvailabilityResponse(BaseModel):
    """Response model for seat availability"""
    flight_id: str
    available_seats: List[Dict[str, Any]]


class AirlineAPIClient:
    """HTTP client for airline API operations"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url or config.airline_api.base_url
        self.api_key = api_key or config.airline_api.api_key
        self.timeout = 30.0
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "JetBlue-CustomerService/1.0"
            }
        )
    
    async def get_booking_details(self, pnr: str) -> BookingDetails:
        """Get booking details by PNR"""
        if not pnr or len(pnr) != 6:
            raise AirlineAPIError("Invalid PNR format. PNR must be 6 characters.", 400, "INVALID_PNR")
        
        try:
            response = await self.client.get(f"/flight/booking", params={"pnr": pnr})
            
            if response.status_code == 404:
                raise AirlineAPIError(f"PNR Not Found: {pnr}", 404, "PNR_NOT_FOUND")
            elif response.status_code == 400:
                raise AirlineAPIError(f"Invalid PNR: {pnr}", 400, "INVALID_PNR")
            
            response.raise_for_status()
            data = response.json()
            
            # Convert to BookingDetails
            return BookingDetails(
                pnr=data["pnr"],
                flight_id=data["flight_id"],
                source_airport_code=data["source_airport_code"],
                destination_airport_code=data["destination_airport_code"],
                scheduled_departure=data["scheduled_departure"],
                scheduled_arrival=data["scheduled_arrival"],
                assigned_seat=data.get("assigned_seat", data.get("seat_number", "Not assigned")),
                current_departure=data.get("current_departure", data["scheduled_departure"]),
                current_arrival=data.get("current_arrival", data["scheduled_arrival"]),
                current_status=data.get("current_status", data.get("flight_status", "Unknown"))
            )
            
        except httpx.RequestError as e:
            raise AirlineAPIError(f"Connection error: {str(e)}", 503, "CONNECTION_ERROR")
        except httpx.HTTPStatusError as e:
            raise AirlineAPIError(f"HTTP error: {e.response.status_code}", e.response.status_code, "HTTP_ERROR")
        except Exception as e:
            raise AirlineAPIError(f"Unexpected error: {str(e)}", 500, "UNEXPECTED_ERROR")
    
    async def cancel_flight(self, pnr: str, reason: str = "Customer request") -> Dict[str, Any]:
        """Cancel flight booking"""
        try:
            response = await self.client.post(
                "/flight/cancel",
                json={"pnr": pnr, "reason": reason}
            )
            
            if response.status_code == 404:
                raise AirlineAPIError(f"PNR Not Found: {pnr}", 404, "PNR_NOT_FOUND")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            raise AirlineAPIError(f"Connection error: {str(e)}", 503, "CONNECTION_ERROR")
        except Exception as e:
            raise AirlineAPIError(f"Cancellation failed: {str(e)}", 500, "CANCELLATION_ERROR")
    
    async def get_available_seats(self, flight_id: str, date: str) -> Dict[str, Any]:
        """Get available seats for a flight"""
        try:
            response = await self.client.post(
                "/flight/available_seats",
                json={"flight_id": flight_id, "date": date}
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.RequestError as e:
            raise AirlineAPIError(f"Connection error: {str(e)}", 503, "CONNECTION_ERROR")
        except Exception as e:
            raise AirlineAPIError(f"Seat availability check failed: {str(e)}", 500, "SEAT_CHECK_ERROR")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = await self.client.get("/health")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "available": response.status_code == 200
            }
        except:
            return {"status": "unhealthy", "available": False}
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class MockAirlineAPIClient(AirlineAPIClient):
    """Mock airline API client for testing and development"""
    
    def __init__(self):
        # Don't call super().__init__() to avoid creating real HTTP client
        self.mock_bookings = {
            "ABC123": {
                "pnr": "ABC123",
                "flight_id": 1001,
                "source_airport_code": "JFK",
                "destination_airport_code": "LAX",
                "scheduled_departure": "2024-01-15T10:00:00Z",
                "scheduled_arrival": "2024-01-15T13:30:00Z",
                "assigned_seat": "12A",
                "current_departure": "2024-01-15T10:00:00Z",
                "current_arrival": "2024-01-15T13:30:00Z",
                "current_status": "On Time",
                "fare_type": "Blue Plus"
            }
        }
    
    async def get_booking_details(self, pnr: str) -> BookingDetails:
        """Mock get booking details"""
        if not pnr or len(pnr) != 6:
            raise AirlineAPIError("Invalid PNR format. PNR must be 6 characters.", 400, "INVALID_PNR")
        
        if pnr not in self.mock_bookings:
            raise AirlineAPIError(f"PNR Not Found: {pnr}", 404, "PNR_NOT_FOUND")
        
        data = self.mock_bookings[pnr]
        return BookingDetails(**data)
    
    async def cancel_flight(self, pnr: str, reason: str = "Customer request") -> Dict[str, Any]:
        """Mock cancel flight"""
        if pnr not in self.mock_bookings:
            raise AirlineAPIError(f"PNR Not Found: {pnr}", 404, "PNR_NOT_FOUND")
        
        return {
            "status": "cancelled",
            "message": "Flight Cancelled",
            "refund_amount": 150.0,
            "cancellation_charges": 50.0,
            "refund_date": "2024-01-20"
        }
    
    async def get_available_seats(self, flight_id: str, date: str) -> Dict[str, Any]:
        """Mock get available seats"""
        return {
            "flight_id": flight_id,
            "available_seats": [
                {"seat": "10A", "type": "economy", "price": 0.0},
                {"seat": "10B", "type": "economy", "price": 0.0},
                {"seat": "15F", "type": "economy", "price": 25.0},
                {"seat": "2A", "type": "business", "price": 150.0}
            ]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Mock health check"""
        return {"status": "healthy", "available": True}
    
    async def close(self):
        """Mock close - no-op"""
        pass