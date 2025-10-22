"""
Performance monitoring service for airline customer service system.

This service tracks latency, availability, and other performance metrics,
providing alerts when thresholds are exceeded.
"""

import time
import structlog
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import threading

from ..config import config
from .audit_logger import audit_logger


class MetricType(str, Enum):
    """Types of performance metrics"""
    REQUEST_LATENCY = "request_latency"
    API_CALL_LATENCY = "api_call_latency"
    POLICY_LOOKUP_LATENCY = "policy_lookup_latency"
    CLASSIFICATION_LATENCY = "classification_latency"
    WORKFLOW_LATENCY = "workflow_latency"
    TASK_LATENCY = "task_latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric"""
    metric_type: MetricType
    count: int
    min_value: float
    max_value: float
    avg_value: float
    p95_value: float
    p99_value: float
    threshold_violations: int
    time_window: timedelta


@dataclass
class PerformanceThresholds:
    """Performance thresholds configuration"""
    request_latency_ms: int = 2000  # 2 seconds
    api_call_latency_ms: int = 5000  # 5 seconds
    policy_lookup_latency_ms: int = 3000  # 3 seconds
    classification_latency_ms: int = 1000  # 1 second
    workflow_latency_ms: int = 2000  # 2 seconds
    task_latency_ms: int = 1000  # 1 second
    error_rate_threshold: float = 0.05  # 5%
    availability_threshold: float = 0.999  # 99.9%


class PerformanceMonitor:
    """
    Service for monitoring system performance and tracking metrics.
    
    Provides real-time performance monitoring with threshold alerting
    and metric aggregation for dashboards and reporting.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.logger = structlog.get_logger("performance")
        self.enabled = config.logging.enable_metrics
        
        # Performance thresholds
        self.thresholds = PerformanceThresholds(
            request_latency_ms=config.performance.max_request_latency,
            api_call_latency_ms=config.performance.max_api_latency,
            policy_lookup_latency_ms=config.performance.max_policy_latency
        )
        
        # Metric storage (in-memory for now, could be extended to use Redis/InfluxDB)
        self._metrics: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._metric_lock = threading.Lock()
        
        # Request tracking for availability calculation
        self._request_counts = defaultdict(int)
        self._error_counts = defaultdict(int)
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[MetricType, float, float], None]] = []
    
    def record_metric(
        self, 
        metric_type: MetricType, 
        value: float,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a performance metric.
        
        Args:
            metric_type: Type of metric being recorded
            value: Metric value
            session_id: Optional session identifier
            request_id: Optional request identifier
            context: Additional context information
        """
        if not self.enabled:
            return
        
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            session_id=session_id,
            request_id=request_id,
            context=context or {}
        )
        
        with self._metric_lock:
            self._metrics[metric_type].append(metric)
        
        # Check thresholds and alert if exceeded
        self._check_threshold(metric)
        
        # Log metric
        self.logger.debug(
            "Performance metric recorded",
            metric_type=metric_type.value,
            value=value,
            session_id=session_id,
            request_id=request_id,
            context=context
        )
    
    @contextmanager
    def measure_latency(
        self, 
        metric_type: MetricType,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for measuring operation latency.
        
        Args:
            metric_type: Type of latency metric
            session_id: Optional session identifier
            request_id: Optional request identifier
            context: Additional context information
            
        Usage:
            with monitor.measure_latency(MetricType.API_CALL_LATENCY, session_id):
                # API call code here
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_metric(
                metric_type, 
                duration_ms, 
                session_id, 
                request_id, 
                context
            )
    
    def record_request(self, session_id: str, success: bool = True) -> None:
        """
        Record a request for availability calculation.
        
        Args:
            session_id: Session identifier
            success: Whether the request was successful
        """
        if not self.enabled:
            return
        
        current_minute = datetime.now().replace(second=0, microsecond=0)
        
        with self._metric_lock:
            self._request_counts[current_minute] += 1
            if not success:
                self._error_counts[current_minute] += 1
        
        # Calculate and record current error rate
        self._calculate_error_rate()
    
    def get_metric_summary(
        self, 
        metric_type: MetricType, 
        time_window: timedelta = timedelta(minutes=5)
    ) -> Optional[MetricSummary]:
        """
        Get summary statistics for a metric within a time window.
        
        Args:
            metric_type: Type of metric to summarize
            time_window: Time window for the summary
            
        Returns:
            Metric summary or None if no data available
        """
        if not self.enabled:
            return None
        
        cutoff_time = datetime.now() - time_window
        
        with self._metric_lock:
            metrics = self._metrics[metric_type]
            recent_metrics = [
                m for m in metrics 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return None
        
        values = [m.value for m in recent_metrics]
        values.sort()
        
        count = len(values)
        min_value = min(values)
        max_value = max(values)
        avg_value = sum(values) / count
        
        # Calculate percentiles
        p95_index = int(0.95 * count)
        p99_index = int(0.99 * count)
        p95_value = values[min(p95_index, count - 1)]
        p99_value = values[min(p99_index, count - 1)]
        
        # Count threshold violations
        threshold = self._get_threshold(metric_type)
        threshold_violations = sum(1 for v in values if v > threshold) if threshold else 0
        
        return MetricSummary(
            metric_type=metric_type,
            count=count,
            min_value=min_value,
            max_value=max_value,
            avg_value=avg_value,
            p95_value=p95_value,
            p99_value=p99_value,
            threshold_violations=threshold_violations,
            time_window=time_window
        )
    
    def get_all_metrics_summary(
        self, 
        time_window: timedelta = timedelta(minutes=5)
    ) -> Dict[MetricType, MetricSummary]:
        """
        Get summary statistics for all metrics.
        
        Args:
            time_window: Time window for the summaries
            
        Returns:
            Dictionary of metric summaries
        """
        summaries = {}
        
        for metric_type in MetricType:
            summary = self.get_metric_summary(metric_type, time_window)
            if summary:
                summaries[metric_type] = summary
        
        return summaries
    
    def get_current_availability(self, time_window: timedelta = timedelta(minutes=5)) -> float:
        """
        Calculate current system availability.
        
        Args:
            time_window: Time window for availability calculation
            
        Returns:
            Availability percentage (0.0 to 1.0)
        """
        if not self.enabled:
            return 1.0
        
        cutoff_time = datetime.now() - time_window
        
        with self._metric_lock:
            total_requests = sum(
                count for timestamp, count in self._request_counts.items()
                if timestamp >= cutoff_time
            )
            total_errors = sum(
                count for timestamp, count in self._error_counts.items()
                if timestamp >= cutoff_time
            )
        
        if total_requests == 0:
            return 1.0
        
        availability = 1.0 - (total_errors / total_requests)
        
        # Record availability metric
        self.record_metric(MetricType.AVAILABILITY, availability)
        
        return availability
    
    def add_alert_callback(self, callback: Callable[[MetricType, float, float], None]) -> None:
        """
        Add a callback function to be called when thresholds are exceeded.
        
        Args:
            callback: Function to call with (metric_type, actual_value, threshold_value)
        """
        self._alert_callbacks.append(callback)
    
    def cleanup_old_metrics(self, retention_period: timedelta = timedelta(hours=1)) -> None:
        """
        Clean up old metrics to prevent memory growth.
        
        Args:
            retention_period: How long to retain metrics
        """
        cutoff_time = datetime.now() - retention_period
        
        with self._metric_lock:
            for metric_type in self._metrics:
                metrics = self._metrics[metric_type]
                # Remove old metrics
                while metrics and metrics[0].timestamp < cutoff_time:
                    metrics.popleft()
            
            # Clean up request/error counts
            old_timestamps = [
                timestamp for timestamp in self._request_counts.keys()
                if timestamp < cutoff_time
            ]
            for timestamp in old_timestamps:
                del self._request_counts[timestamp]
                if timestamp in self._error_counts:
                    del self._error_counts[timestamp]
    
    def _check_threshold(self, metric: PerformanceMetric) -> None:
        """Check if metric exceeds threshold and trigger alerts."""
        threshold = self._get_threshold(metric.metric_type)
        
        if threshold and metric.value > threshold:
            # Log threshold violation
            audit_logger.log_performance_threshold_exceeded(
                session_id=metric.session_id or "system",
                metric_name=metric.metric_type.value,
                actual_value=metric.value,
                threshold_value=threshold,
                request_id=metric.request_id
            )
            
            # Trigger alert callbacks
            for callback in self._alert_callbacks:
                try:
                    callback(metric.metric_type, metric.value, threshold)
                except Exception as e:
                    self.logger.error(
                        "Alert callback failed",
                        error=str(e),
                        metric_type=metric.metric_type.value
                    )
    
    def _get_threshold(self, metric_type: MetricType) -> Optional[float]:
        """Get threshold value for a metric type."""
        threshold_map = {
            MetricType.REQUEST_LATENCY: self.thresholds.request_latency_ms,
            MetricType.API_CALL_LATENCY: self.thresholds.api_call_latency_ms,
            MetricType.POLICY_LOOKUP_LATENCY: self.thresholds.policy_lookup_latency_ms,
            MetricType.CLASSIFICATION_LATENCY: self.thresholds.classification_latency_ms,
            MetricType.WORKFLOW_LATENCY: self.thresholds.workflow_latency_ms,
            MetricType.TASK_LATENCY: self.thresholds.task_latency_ms,
            MetricType.ERROR_RATE: self.thresholds.error_rate_threshold,
            MetricType.AVAILABILITY: self.thresholds.availability_threshold,
        }
        
        return threshold_map.get(metric_type)
    
    def _calculate_error_rate(self) -> None:
        """Calculate and record current error rate."""
        current_minute = datetime.now().replace(second=0, microsecond=0)
        
        with self._metric_lock:
            requests = self._request_counts.get(current_minute, 0)
            errors = self._error_counts.get(current_minute, 0)
        
        if requests > 0:
            error_rate = errors / requests
            self.record_metric(MetricType.ERROR_RATE, error_rate)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()