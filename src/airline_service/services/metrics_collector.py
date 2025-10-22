"""
Metrics collection service for airline customer service system.

This service provides centralized metrics collection and reporting,
integrating with the performance monitor and audit logger.
"""

import structlog
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .performance_monitor import (
    performance_monitor, MetricType, MetricSummary, PerformanceMonitor
)
from .audit_logger import audit_logger, AuditEventType
from ..config import config


@dataclass
class SystemHealthMetrics:
    """Overall system health metrics"""
    availability: float
    average_response_time_ms: float
    error_rate: float
    requests_per_minute: int
    api_success_rate: float
    classification_accuracy: float
    timestamp: datetime


@dataclass
class DetailedMetrics:
    """Detailed performance metrics"""
    request_latency: Optional[MetricSummary]
    api_call_latency: Optional[MetricSummary]
    policy_lookup_latency: Optional[MetricSummary]
    classification_latency: Optional[MetricSummary]
    workflow_latency: Optional[MetricSummary]
    task_latency: Optional[MetricSummary]
    error_rate: Optional[MetricSummary]
    throughput: Optional[MetricSummary]
    availability: Optional[MetricSummary]
    timestamp: datetime


class MetricsCollector:
    """
    Service for collecting and aggregating system metrics.
    
    Provides unified access to performance metrics, health status,
    and system statistics for monitoring and alerting.
    """
    
    def __init__(self, monitor: PerformanceMonitor = None):
        """Initialize the metrics collector."""
        self.logger = structlog.get_logger("metrics")
        self.monitor = monitor or performance_monitor
        self.enabled = config.logging.enable_metrics
        
        # Metrics cache
        self._cached_health_metrics: Optional[SystemHealthMetrics] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=30)  # Cache for 30 seconds
    
    def get_system_health(self, force_refresh: bool = False) -> SystemHealthMetrics:
        """
        Get overall system health metrics.
        
        Args:
            force_refresh: Force refresh of cached metrics
            
        Returns:
            System health metrics
        """
        if not self.enabled:
            return self._get_default_health_metrics()
        
        # Check cache
        now = datetime.now()
        if (not force_refresh and 
            self._cached_health_metrics and 
            self._cache_timestamp and 
            now - self._cache_timestamp < self._cache_ttl):
            return self._cached_health_metrics
        
        # Calculate fresh metrics
        time_window = timedelta(minutes=5)
        
        # Get availability
        availability = self.monitor.get_current_availability(time_window)
        
        # Get average response time
        request_latency_summary = self.monitor.get_metric_summary(
            MetricType.REQUEST_LATENCY, time_window
        )
        avg_response_time = (
            request_latency_summary.avg_value 
            if request_latency_summary else 0.0
        )
        
        # Get error rate
        error_rate_summary = self.monitor.get_metric_summary(
            MetricType.ERROR_RATE, time_window
        )
        error_rate = (
            error_rate_summary.avg_value 
            if error_rate_summary else 0.0
        )
        
        # Get throughput (requests per minute)
        throughput_summary = self.monitor.get_metric_summary(
            MetricType.THROUGHPUT, time_window
        )
        requests_per_minute = int(
            throughput_summary.avg_value 
            if throughput_summary else 0.0
        )
        
        # Calculate API success rate
        api_latency_summary = self.monitor.get_metric_summary(
            MetricType.API_CALL_LATENCY, time_window
        )
        api_success_rate = self._calculate_api_success_rate(api_latency_summary)
        
        # Estimate classification accuracy (placeholder - would need actual tracking)
        classification_accuracy = 0.95  # Default assumption
        
        health_metrics = SystemHealthMetrics(
            availability=availability,
            average_response_time_ms=avg_response_time,
            error_rate=error_rate,
            requests_per_minute=requests_per_minute,
            api_success_rate=api_success_rate,
            classification_accuracy=classification_accuracy,
            timestamp=now
        )
        
        # Update cache
        self._cached_health_metrics = health_metrics
        self._cache_timestamp = now
        
        return health_metrics
    
    def get_detailed_metrics(
        self, 
        time_window: timedelta = timedelta(minutes=5)
    ) -> DetailedMetrics:
        """
        Get detailed performance metrics for all metric types.
        
        Args:
            time_window: Time window for metric aggregation
            
        Returns:
            Detailed metrics summary
        """
        if not self.enabled:
            return self._get_default_detailed_metrics()
        
        summaries = self.monitor.get_all_metrics_summary(time_window)
        
        return DetailedMetrics(
            request_latency=summaries.get(MetricType.REQUEST_LATENCY),
            api_call_latency=summaries.get(MetricType.API_CALL_LATENCY),
            policy_lookup_latency=summaries.get(MetricType.POLICY_LOOKUP_LATENCY),
            classification_latency=summaries.get(MetricType.CLASSIFICATION_LATENCY),
            workflow_latency=summaries.get(MetricType.WORKFLOW_LATENCY),
            task_latency=summaries.get(MetricType.TASK_LATENCY),
            error_rate=summaries.get(MetricType.ERROR_RATE),
            throughput=summaries.get(MetricType.THROUGHPUT),
            availability=summaries.get(MetricType.AVAILABILITY),
            timestamp=datetime.now()
        )
    
    def get_metrics_for_dashboard(self) -> Dict[str, Any]:
        """
        Get metrics formatted for dashboard display.
        
        Returns:
            Dictionary of metrics suitable for dashboard consumption
        """
        health = self.get_system_health()
        detailed = self.get_detailed_metrics()
        
        dashboard_data = {
            "system_health": {
                "status": self._get_health_status(health),
                "availability": f"{health.availability * 100:.2f}%",
                "avg_response_time": f"{health.average_response_time_ms:.0f}ms",
                "error_rate": f"{health.error_rate * 100:.2f}%",
                "requests_per_minute": health.requests_per_minute,
                "api_success_rate": f"{health.api_success_rate * 100:.1f}%",
                "classification_accuracy": f"{health.classification_accuracy * 100:.1f}%"
            },
            "performance_metrics": {},
            "alerts": self._get_active_alerts(detailed),
            "timestamp": health.timestamp.isoformat()
        }
        
        # Add detailed metrics
        for metric_name, summary in [
            ("request_latency", detailed.request_latency),
            ("api_call_latency", detailed.api_call_latency),
            ("policy_lookup_latency", detailed.policy_lookup_latency),
            ("classification_latency", detailed.classification_latency),
            ("workflow_latency", detailed.workflow_latency)
        ]:
            if summary:
                dashboard_data["performance_metrics"][metric_name] = {
                    "avg": f"{summary.avg_value:.0f}ms",
                    "p95": f"{summary.p95_value:.0f}ms",
                    "p99": f"{summary.p99_value:.0f}ms",
                    "count": summary.count,
                    "threshold_violations": summary.threshold_violations
                }
        
        return dashboard_data
    
    def record_request_metrics(
        self, 
        session_id: str,
        processing_time_ms: int,
        success: bool = True,
        request_type: Optional[str] = None
    ) -> None:
        """
        Record metrics for a completed request.
        
        Args:
            session_id: Session identifier
            processing_time_ms: Total processing time
            success: Whether request was successful
            request_type: Type of request processed
        """
        if not self.enabled:
            return
        
        # Record latency
        self.monitor.record_metric(
            MetricType.REQUEST_LATENCY,
            processing_time_ms,
            session_id=session_id,
            context={"request_type": request_type} if request_type else None
        )
        
        # Record request for availability calculation
        self.monitor.record_request(session_id, success)
        
        # Record throughput (1 request)
        self.monitor.record_metric(
            MetricType.THROUGHPUT,
            1.0,
            session_id=session_id
        )
    
    def record_api_metrics(
        self, 
        session_id: str,
        endpoint: str,
        response_time_ms: int,
        success: bool = True,
        status_code: Optional[int] = None
    ) -> None:
        """
        Record metrics for an API call.
        
        Args:
            session_id: Session identifier
            endpoint: API endpoint called
            response_time_ms: API response time
            success: Whether API call was successful
            status_code: HTTP status code
        """
        if not self.enabled:
            return
        
        self.monitor.record_metric(
            MetricType.API_CALL_LATENCY,
            response_time_ms,
            session_id=session_id,
            context={
                "endpoint": endpoint,
                "success": success,
                "status_code": status_code
            }
        )
    
    def record_classification_metrics(
        self, 
        session_id: str,
        processing_time_ms: int,
        confidence: float,
        request_type: str
    ) -> None:
        """
        Record metrics for request classification.
        
        Args:
            session_id: Session identifier
            processing_time_ms: Classification processing time
            confidence: Classification confidence score
            request_type: Classified request type
        """
        if not self.enabled:
            return
        
        self.monitor.record_metric(
            MetricType.CLASSIFICATION_LATENCY,
            processing_time_ms,
            session_id=session_id,
            context={
                "confidence": confidence,
                "request_type": request_type
            }
        )
    
    def record_workflow_metrics(
        self, 
        session_id: str,
        processing_time_ms: int,
        task_count: int,
        success: bool = True
    ) -> None:
        """
        Record metrics for workflow execution.
        
        Args:
            session_id: Session identifier
            processing_time_ms: Workflow processing time
            task_count: Number of tasks executed
            success: Whether workflow was successful
        """
        if not self.enabled:
            return
        
        self.monitor.record_metric(
            MetricType.WORKFLOW_LATENCY,
            processing_time_ms,
            session_id=session_id,
            context={
                "task_count": task_count,
                "success": success
            }
        )
    
    def record_error_metrics(
        self,
        error_code: str,
        severity: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record metrics for errors.
        
        Args:
            error_code: Error code identifier
            severity: Error severity level
            session_id: Optional session identifier
            context: Optional error context
        """
        if not self.enabled:
            return
        
        # Record error rate metric
        self.monitor.record_metric(
            MetricType.ERROR_RATE,
            1.0,  # 1 error occurred
            session_id=session_id,
            context={
                "error_code": error_code,
                "severity": severity,
                **(context or {})
            }
        )
        
        # Log error event for audit
        if session_id:
            audit_logger.log_error(
                session_id=session_id,
                error_type=error_code,
                error_message=f"Error recorded with severity: {severity}",
                error_code=error_code,
                context=context
            )
    
    def get_health_check_data(self) -> Dict[str, Any]:
        """
        Get health check data for monitoring systems.
        
        Returns:
            Health check data including status and key metrics
        """
        health = self.get_system_health()
        
        # Determine overall health status
        status = "healthy"
        if health.availability < 0.99:
            status = "degraded"
        if health.availability < 0.95 or health.error_rate > 0.1:
            status = "unhealthy"
        
        return {
            "status": status,
            "timestamp": health.timestamp.isoformat(),
            "metrics": {
                "availability": health.availability,
                "avg_response_time_ms": health.average_response_time_ms,
                "error_rate": health.error_rate,
                "requests_per_minute": health.requests_per_minute
            },
            "checks": {
                "availability_ok": health.availability >= 0.99,
                "response_time_ok": health.average_response_time_ms <= 2000,
                "error_rate_ok": health.error_rate <= 0.05
            }
        }
    
    def _get_health_status(self, health: SystemHealthMetrics) -> str:
        """Determine health status from metrics."""
        if health.availability >= 0.999 and health.error_rate <= 0.01:
            return "excellent"
        elif health.availability >= 0.99 and health.error_rate <= 0.05:
            return "good"
        elif health.availability >= 0.95 and health.error_rate <= 0.1:
            return "degraded"
        else:
            return "poor"
    
    def _get_active_alerts(self, detailed: DetailedMetrics) -> List[Dict[str, Any]]:
        """Get list of active alerts based on metrics."""
        alerts = []
        
        # Check each metric for threshold violations
        for metric_name, summary in [
            ("request_latency", detailed.request_latency),
            ("api_call_latency", detailed.api_call_latency),
            ("policy_lookup_latency", detailed.policy_lookup_latency)
        ]:
            if summary and summary.threshold_violations > 0:
                alerts.append({
                    "type": "threshold_violation",
                    "metric": metric_name,
                    "violations": summary.threshold_violations,
                    "max_value": f"{summary.max_value:.0f}ms",
                    "severity": "warning" if summary.threshold_violations < 5 else "critical"
                })
        
        return alerts
    
    def _calculate_api_success_rate(self, api_summary: Optional[MetricSummary]) -> float:
        """Calculate API success rate from latency metrics."""
        if not api_summary:
            return 1.0
        
        # Assume that very high latencies (>10s) indicate failures
        # This is a simplified calculation - in practice, you'd track success/failure explicitly
        failure_threshold = 10000  # 10 seconds
        if api_summary.max_value > failure_threshold:
            # Estimate failure rate based on p99 latency
            if api_summary.p99_value > failure_threshold:
                return 0.95  # Assume 5% failure rate
            else:
                return 0.99  # Assume 1% failure rate
        
        return 1.0
    
    def _get_default_health_metrics(self) -> SystemHealthMetrics:
        """Get default health metrics when monitoring is disabled."""
        return SystemHealthMetrics(
            availability=1.0,
            average_response_time_ms=0.0,
            error_rate=0.0,
            requests_per_minute=0,
            api_success_rate=1.0,
            classification_accuracy=1.0,
            timestamp=datetime.now()
        )
    
    def _get_default_detailed_metrics(self) -> DetailedMetrics:
        """Get default detailed metrics when monitoring is disabled."""
        return DetailedMetrics(
            request_latency=None,
            api_call_latency=None,
            policy_lookup_latency=None,
            classification_latency=None,
            workflow_latency=None,
            task_latency=None,
            error_rate=None,
            throughput=None,
            availability=None,
            timestamp=datetime.now()
        )


# Global metrics collector instance
metrics_collector = MetricsCollector()