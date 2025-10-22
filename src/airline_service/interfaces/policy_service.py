"""
Policy service interface definitions
"""

from abc import ABC, abstractmethod
from typing import Optional
from ..types import PolicyInfo, FlightDetails


class PolicyServiceInterface(ABC):
    """Interface for policy service operations"""
    
    @abstractmethod
    async def get_cancellation_policy(self, flight_details: Optional[FlightDetails] = None) -> PolicyInfo:
        """Get cancellation policy information"""
        pass
    
    @abstractmethod
    async def get_pet_travel_policy(self) -> PolicyInfo:
        """Get pet travel policy information"""
        pass
    
    @abstractmethod
    async def refresh_policy_cache(self) -> None:
        """Refresh policy cache"""
        pass


class PolicyCacheInterface(ABC):
    """Interface for policy caching"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[PolicyInfo]:
        """Get policy from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, policy: PolicyInfo, ttl: Optional[int] = None) -> None:
        """Set policy in cache"""
        pass
    
    @abstractmethod
    async def invalidate(self, key: str) -> None:
        """Invalidate cache entry"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear entire cache"""
        pass


class PolicyRetrieverInterface(ABC):
    """Interface for policy retrieval"""
    
    @abstractmethod
    async def fetch_policy_from_url(self, url: str) -> str:
        """Fetch policy content from URL"""
        pass
    
    @abstractmethod
    def parse_policy_content(self, content: str, policy_type: str) -> PolicyInfo:
        """Parse policy content into structured format"""
        pass
    
    @abstractmethod
    def extract_relevant_sections(self, content: str, flight_details: Optional[FlightDetails] = None) -> str:
        """Extract relevant policy sections"""
        pass