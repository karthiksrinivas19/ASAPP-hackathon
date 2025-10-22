"""
Customer interaction interface definitions
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from ..types import APIResponse, ResponseFormat


class CustomerInteractionInterface(ABC):
    """Interface for customer interaction service"""
    
    @abstractmethod
    def format_response(self, data: Any, format_type: ResponseFormat) -> APIResponse:
        """Format response data"""
        pass
    
    @abstractmethod
    def create_completed_response(self, message: str, data: Any = None) -> APIResponse:
        """Create completed response"""
        pass
    
    @abstractmethod
    def create_error_response(self, error: str, error_code: str = None) -> APIResponse:
        """Create error response"""
        pass
    
    @abstractmethod
    async def log_interaction(self, request: str, response: APIResponse) -> None:
        """Log customer interaction"""
        pass


class AutomatedResponseBuilderInterface(ABC):
    """Interface for automated response building"""
    
    @abstractmethod
    def build_flight_status_response(self, flight_data: Any) -> APIResponse:
        """Build flight status response"""
        pass
    
    @abstractmethod
    def build_cancellation_response(self, cancellation_result: Any) -> APIResponse:
        """Build cancellation response"""
        pass
    
    @abstractmethod
    def build_seat_availability_response(self, seats: list) -> APIResponse:
        """Build seat availability response"""
        pass
    
    @abstractmethod
    def build_policy_response(self, policy_info: str) -> APIResponse:
        """Build policy information response"""
        pass
    
    @abstractmethod
    def build_general_info_response(self, message: str) -> APIResponse:
        """Build general information response"""
        pass


class ResponseFormatterInterface(ABC):
    """Interface for response formatting"""
    
    @abstractmethod
    def format_flight_status(self, data: Any) -> Dict[str, Any]:
        """Format flight status data"""
        pass
    
    @abstractmethod
    def format_cancellation_result(self, data: Any) -> Dict[str, Any]:
        """Format cancellation result data"""
        pass
    
    @abstractmethod
    def format_seat_availability(self, data: Any) -> Dict[str, Any]:
        """Format seat availability data"""
        pass
    
    @abstractmethod
    def format_policy_info(self, data: Any) -> Dict[str, Any]:
        """Format policy information data"""
        pass
    
    @abstractmethod
    def format_error_message(self, error: str, code: str = None) -> Dict[str, Any]:
        """Format error message"""
        pass