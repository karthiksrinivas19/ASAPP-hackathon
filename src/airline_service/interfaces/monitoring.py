"""
Monitoring interface definitions
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from ..types import APIResponse


class PerformanceMetricsInterface(ABC):
    """Interface for performance metrics"""
    
    @property
    @abstractmethod
    def request_latency(self) -> float:
        """Request latency in milliseconds"""
        pass
    
    @property
    @abstractmethod
    def api_call_latency(self) -> float:
        """API call latency in milliseconds"""
        pass
    
    @property
    @abstractmethod
    def policy_lookup_latency(self) -> float:
        """Policy lookup latency in milliseconds"""
        pass
    
    @property
    @abstractmethod
    def error_rate(self) -> float:
        """Error rate percentage"""
        pass
    
    @property
    @abstractmethod
    def throughput(self) -> float:
        """Requests per second"""
        pass
    
    @property
    @abstractmethod
    def availability(self) -> float:
        """System availability percentage"""
        pass


class MonitoringInterface(ABC):
    """Interface for monitoring service"""
    
    @abstractmethod
    def record_metric(self, metric: str, value: float) -> None:
        """Record a metric value"""
        pass
    
    @abstractmethod
    def get_metrics(self, time_range: Dict[str, Any]) -> Dict[str, Any]:
        """Get metrics for time range"""
        pass
    
    @abstractmethod
    def create_alert(self, condition: Dict[str, Any]) -> None:
        """Create alert for condition"""
        pass


class AuditLoggerInterface(ABC):
    """Interface for audit logging"""
    
    @abstractmethod
    async def log_request(self, request: Dict[str, Any]) -> None:
        """Log incoming request"""
        pass
    
    @abstractmethod
    async def log_response(self, response: APIResponse) -> None:
        """Log outgoing response"""
        pass
    
    @abstractmethod
    async def log_error(self, error: Exception) -> None:
        """Log error occurrence"""
        pass
    
    @abstractmethod
    async def log_api_call(self, endpoint: str, duration: float, success: bool) -> None:
        """Log API call details"""
        pass


class HealthCheckerInterface(ABC):
    """Interface for health checking"""
    
    @abstractmethod
    async def check_api_health(self) -> bool:
        """Check airline API health"""
        pass
    
    @abstractmethod
    async def check_database_health(self) -> bool:
        """Check database health"""
        pass
    
    @abstractmethod
    async def check_cache_health(self) -> bool:
        """Check cache health"""
        pass
    
    @abstractmethod
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        pass