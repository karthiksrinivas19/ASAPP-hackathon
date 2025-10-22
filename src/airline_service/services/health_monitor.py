"""
Health monitoring service for airline customer service system.

This service provides comprehensive health checks, availability monitoring,
and alerting capabilities for the system.
"""

import structlog
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

from .performance_monitor import performance_monitor, MetricType
from .metrics_collector import metrics_collector
from .audit_logger import audit_logger
from ..config import config


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Performance alert"""
    id: str
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    actual_value: float
    threshold_value: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    availability: float
    response_time_ms: float
    error_rate: float
    active_alerts: List[Alert]
    health_checks: List[HealthCheck]
    timestamp: datetime


class HealthMonitor:
    """
    Comprehensive health monitoring service.
    
    Provides health checks, availability monitoring, alerting,
    and integration with external monitoring systems.
    """
    
    def __init__(self):
        """Initialize the health monitor."""
        self.logger = structlog.get_logger("health_monitor")
        self.enabled = config.logging.enable_metrics
        
        # Alert storage
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_lock = threading.Lock()
        
        # Health check registry
        self._health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Register default health checks
        self._register_default_health_checks()
        
        # Register alert callback with performance monitor
        performance_monitor.add_alert_callback(self._handle_performance_alert)
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheck]) -> None:
        """
        Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns a HealthCheck result
        """
        self._health_checks[name] = check_func
        self.logger.info("Health check registered", name=name)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Add a callback function to be called when alerts are triggered.
        
        Args:
            callback: Function to call with Alert object
        """
        self._alert_callbacks.append(callback)
    
    async def run_health_checks(self) -> List[HealthCheck]:
        """
        Run all registered health checks.
        
        Returns:
            List of health check results
        """
        if not self.enabled:
            return []
        
        results = []
        
        for name, check_func in self._health_checks.items():
            try:
                start_time = time.time()
                result = await self._run_check_safely(check_func)
                result.response_time_ms = (time.time() - start_time) * 1000
                results.append(result)
                
                self.logger.debug(
                    "Health check completed",
                    name=name,
                    status=result.status.value,
                    response_time_ms=result.response_time_ms
                )
                
            except Exception as e:
                error_result = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    response_time_ms=None,
                    details={"error": str(e)}
                )
                results.append(error_result)
                
                self.logger.error(
                    "Health check failed",
                    name=name,
                    error=str(e)
                )
        
        return results
    
    async def get_system_health(self) -> SystemHealth:
        """
        Get comprehensive system health status.
        
        Returns:
            SystemHealth object with overall status and details
        """
        if not self.enabled:
            return self._get_default_system_health()
        
        # Run health checks
        health_checks = await self.run_health_checks()
        
        # Get performance metrics
        system_metrics = metrics_collector.get_system_health()
        
        # Get active alerts
        active_alerts = self.get_active_alerts()
        
        # Determine overall health status
        overall_status = self._calculate_overall_health(
            health_checks, system_metrics, active_alerts
        )
        
        return SystemHealth(
            status=overall_status,
            availability=system_metrics.availability,
            response_time_ms=system_metrics.average_response_time_ms,
            error_rate=system_metrics.error_rate,
            active_alerts=active_alerts,
            health_checks=health_checks,
            timestamp=datetime.now()
        )
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get list of active (unresolved) alerts.
        
        Returns:
            List of active alerts
        """
        with self._alert_lock:
            return [
                alert for alert in self._active_alerts.values()
                if not alert.resolved
            ]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if alert was found and acknowledged
        """
        with self._alert_lock:
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].acknowledged = True
                self.logger.info("Alert acknowledged", alert_id=alert_id)
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if alert was found and resolved
        """
        with self._alert_lock:
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].resolved = True
                self.logger.info("Alert resolved", alert_id=alert_id)
                return True
        
        return False
    
    def get_availability_report(
        self, 
        time_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """
        Get availability report for the specified time window.
        
        Args:
            time_window: Time window for the report
            
        Returns:
            Availability report with SLA metrics
        """
        if not self.enabled:
            return {"availability": 1.0, "sla_met": True}
        
        # Get current availability
        current_availability = performance_monitor.get_current_availability(time_window)
        
        # Calculate SLA compliance
        target_availability = config.performance.target_availability
        sla_met = current_availability >= target_availability
        
        # Get downtime estimate
        total_minutes = time_window.total_seconds() / 60
        downtime_minutes = total_minutes * (1 - current_availability)
        
        return {
            "availability": current_availability,
            "availability_percentage": f"{current_availability * 100:.3f}%",
            "target_availability": target_availability,
            "target_percentage": f"{target_availability * 100:.3f}%",
            "sla_met": sla_met,
            "downtime_minutes": downtime_minutes,
            "uptime_minutes": total_minutes - downtime_minutes,
            "time_window_hours": time_window.total_seconds() / 3600,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive data for performance dashboard.
        
        Returns:
            Dashboard data including metrics, alerts, and health status
        """
        if not self.enabled:
            return self._get_default_dashboard_data()
        
        # Get system health
        health = await self.get_system_health()
        
        # Get detailed metrics
        detailed_metrics = metrics_collector.get_detailed_metrics()
        
        # Get availability report
        availability_report = self.get_availability_report()
        
        return {
            "system_status": {
                "overall_health": health.status.value,
                "availability": f"{health.availability * 100:.2f}%",
                "response_time": f"{health.response_time_ms:.0f}ms",
                "error_rate": f"{health.error_rate * 100:.2f}%",
                "active_alerts_count": len(health.active_alerts)
            },
            "performance_metrics": self._format_metrics_for_dashboard(detailed_metrics),
            "availability_report": availability_report,
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "metric": alert.metric_type.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged
                }
                for alert in health.active_alerts
            ],
            "health_checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "response_time_ms": check.response_time_ms,
                    "timestamp": check.timestamp.isoformat()
                }
                for check in health.health_checks
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks."""
        
        async def check_classifier_health() -> HealthCheck:
            """Check ML classifier health."""
            try:
                from ..services.request_classifier_service import ClassifierFactory
                classifier = ClassifierFactory.create_classifier()
                
                if classifier.is_loaded():
                    # Test classification with a simple query
                    test_result = await classifier.classify_request("test query")
                    
                    return HealthCheck(
                        name="ml_classifier",
                        status=HealthStatus.HEALTHY,
                        message="ML classifier is loaded and responding",
                        details={
                            "model_loaded": True,
                            "test_confidence": test_result.confidence
                        }
                    )
                else:
                    return HealthCheck(
                        name="ml_classifier",
                        status=HealthStatus.DEGRADED,
                        message="Using mock classifier - model not loaded",
                        details={"model_loaded": False}
                    )
                    
            except Exception as e:
                return HealthCheck(
                    name="ml_classifier",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Classifier check failed: {str(e)}",
                    details={"error": str(e)}
                )
        
        async def check_airline_api_health() -> HealthCheck:
            """Check airline API client health."""
            try:
                from ..clients.airline_api_client import MockAirlineAPIClient
                client = MockAirlineAPIClient()
                
                # For mock client, always healthy
                return HealthCheck(
                    name="airline_api",
                    status=HealthStatus.HEALTHY,
                    message="Airline API client is healthy (mock)",
                    details={"client_type": "mock"}
                )
                
            except Exception as e:
                return HealthCheck(
                    name="airline_api",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Airline API check failed: {str(e)}",
                    details={"error": str(e)}
                )
        
        async def check_policy_service_health() -> HealthCheck:
            """Check policy service health."""
            try:
                from ..services.policy_service import policy_service
                
                # Test policy retrieval
                policy_info = await policy_service.get_pet_travel_policy()
                
                return HealthCheck(
                    name="policy_service",
                    status=HealthStatus.HEALTHY,
                    message="Policy service is responding",
                    details={
                        "policy_retrieved": bool(policy_info),
                        "policy_length": len(policy_info.content) if policy_info else 0
                    }
                )
                
            except Exception as e:
                return HealthCheck(
                    name="policy_service",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Policy service check failed: {str(e)}",
                    details={"error": str(e)}
                )
        
        async def check_cache_health() -> HealthCheck:
            """Check cache service health."""
            try:
                from ..services.cache_service import cache_service
                
                # Test cache connectivity
                stats = await cache_service.get_cache_stats()
                
                if stats.get("status") == "connected":
                    return HealthCheck(
                        name="cache_service",
                        status=HealthStatus.HEALTHY,
                        message="Cache service is connected and responding",
                        details={
                            "redis_version": stats.get("redis_version"),
                            "hit_rate": stats.get("hit_rate", 0.0)
                        }
                    )
                else:
                    return HealthCheck(
                        name="cache_service",
                        status=HealthStatus.DEGRADED,
                        message="Cache service is not connected",
                        details=stats
                    )
                    
            except Exception as e:
                return HealthCheck(
                    name="cache_service",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Cache service check failed: {str(e)}",
                    details={"error": str(e)}
                )
        
        # Register all health checks
        self.register_health_check("ml_classifier", check_classifier_health)
        self.register_health_check("airline_api", check_airline_api_health)
        self.register_health_check("policy_service", check_policy_service_health)
        self.register_health_check("cache_service", check_cache_health)
    
    async def _run_check_safely(self, check_func: Callable) -> HealthCheck:
        """Run a health check function safely with timeout."""
        try:
            # Run with timeout
            return await asyncio.wait_for(check_func(), timeout=5.0)
        except asyncio.TimeoutError:
            return HealthCheck(
                name="unknown",
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                details={"timeout": True}
            )
    
    def _handle_performance_alert(
        self, 
        metric_type: MetricType, 
        actual_value: float, 
        threshold_value: float
    ) -> None:
        """Handle performance threshold violations."""
        
        # Determine severity based on how much threshold is exceeded
        ratio = actual_value / threshold_value if threshold_value > 0 else 1.0
        
        if ratio >= 2.0:
            severity = AlertSeverity.CRITICAL
        elif ratio >= 1.5:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        # Create alert
        alert_id = f"{metric_type.value}_{int(datetime.now().timestamp())}"
        alert = Alert(
            id=alert_id,
            severity=severity,
            metric_type=metric_type,
            message=f"{metric_type.value} exceeded threshold: {actual_value:.1f} > {threshold_value:.1f}",
            actual_value=actual_value,
            threshold_value=threshold_value,
            timestamp=datetime.now()
        )
        
        # Store alert
        with self._alert_lock:
            self._active_alerts[alert_id] = alert
        
        # Trigger callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(
                    "Alert callback failed",
                    error=str(e),
                    alert_id=alert_id
                )
        
        # Log alert
        audit_logger.log_performance_alert(
            session_id="system",
            alert_type=metric_type.value,
            severity=severity.value,
            message=alert.message,
            actual_value=actual_value,
            threshold_value=threshold_value
        )
    
    def _calculate_overall_health(
        self, 
        health_checks: List[HealthCheck],
        system_metrics: Any,
        active_alerts: List[Alert]
    ) -> HealthStatus:
        """Calculate overall system health status."""
        
        # Check for critical alerts
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            return HealthStatus.CRITICAL
        
        # Check health check results
        unhealthy_checks = [c for c in health_checks if c.status == HealthStatus.UNHEALTHY]
        if unhealthy_checks:
            return HealthStatus.UNHEALTHY
        
        degraded_checks = [c for c in health_checks if c.status == HealthStatus.DEGRADED]
        warning_alerts = [a for a in active_alerts if a.severity == AlertSeverity.WARNING]
        
        # Check system metrics
        if (system_metrics.availability < 0.95 or 
            system_metrics.error_rate > 0.1 or
            system_metrics.average_response_time_ms > 5000):
            return HealthStatus.UNHEALTHY
        
        if (degraded_checks or warning_alerts or
            system_metrics.availability < 0.99 or
            system_metrics.error_rate > 0.05 or
            system_metrics.average_response_time_ms > 2000):
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def _format_metrics_for_dashboard(self, detailed_metrics: Any) -> Dict[str, Any]:
        """Format detailed metrics for dashboard display."""
        formatted = {}
        
        for metric_name in ["request_latency", "api_call_latency", "classification_latency", "workflow_latency"]:
            metric_summary = getattr(detailed_metrics, metric_name, None)
            if metric_summary:
                formatted[metric_name] = {
                    "avg_ms": f"{metric_summary.avg_value:.0f}",
                    "p95_ms": f"{metric_summary.p95_value:.0f}",
                    "p99_ms": f"{metric_summary.p99_value:.0f}",
                    "count": metric_summary.count,
                    "threshold_violations": metric_summary.threshold_violations,
                    "status": "warning" if metric_summary.threshold_violations > 0 else "ok"
                }
        
        return formatted
    
    def _get_default_system_health(self) -> SystemHealth:
        """Get default system health when monitoring is disabled."""
        return SystemHealth(
            status=HealthStatus.HEALTHY,
            availability=1.0,
            response_time_ms=0.0,
            error_rate=0.0,
            active_alerts=[],
            health_checks=[],
            timestamp=datetime.now()
        )
    
    def _get_default_dashboard_data(self) -> Dict[str, Any]:
        """Get default dashboard data when monitoring is disabled."""
        return {
            "system_status": {
                "overall_health": "healthy",
                "availability": "100.00%",
                "response_time": "0ms",
                "error_rate": "0.00%",
                "active_alerts_count": 0
            },
            "performance_metrics": {},
            "availability_report": {"availability": 1.0, "sla_met": True},
            "active_alerts": [],
            "health_checks": [],
            "timestamp": datetime.now().isoformat()
        }


# Global health monitor instance
health_monitor = HealthMonitor()