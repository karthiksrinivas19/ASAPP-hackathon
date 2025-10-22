"""
Tests for performance monitoring service
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.airline_service.services.performance_monitor import (
    PerformanceMonitor, MetricType, PerformanceMetric, MetricSummary
)


class TestPerformanceMonitor:
    """Test performance monitor functionality"""
    
    @pytest.fixture
    def monitor(self):
        """Create performance monitor instance"""
        return PerformanceMonitor()
    
    def test_record_metric(self, monitor):
        """Test recording a performance metric"""
        monitor.record_metric(
            MetricType.REQUEST_LATENCY,
            1500.0,
            session_id="test_session",
            request_id="req_123",
            context={"endpoint": "/api/test"}
        )
        
        # Check that metric was stored
        assert len(monitor._metrics[MetricType.REQUEST_LATENCY]) == 1
        
        metric = monitor._metrics[MetricType.REQUEST_LATENCY][0]
        assert metric.metric_type == MetricType.REQUEST_LATENCY
        assert metric.value == 1500.0
        assert metric.session_id == "test_session"
        assert metric.request_id == "req_123"
        assert metric.context["endpoint"] == "/api/test"
    
    def test_measure_latency_context_manager(self, monitor):
        """Test latency measurement context manager"""
        with monitor.measure_latency(
            MetricType.API_CALL_LATENCY,
            session_id="test_session",
            context={"api": "test"}
        ):
            time.sleep(0.01)  # Sleep for 10ms
        
        # Check that latency was recorded
        assert len(monitor._metrics[MetricType.API_CALL_LATENCY]) == 1
        
        metric = monitor._metrics[MetricType.API_CALL_LATENCY][0]
        assert metric.value >= 10.0  # Should be at least 10ms
        assert metric.session_id == "test_session"
        assert metric.context["api"] == "test"
    
    def test_record_request_success(self, monitor):
        """Test recording successful request"""
        current_minute = datetime.now().replace(second=0, microsecond=0)
        
        monitor.record_request("test_session", success=True)
        
        assert monitor._request_counts[current_minute] == 1
        assert monitor._error_counts[current_minute] == 0
    
    def test_record_request_failure(self, monitor):
        """Test recording failed request"""
        current_minute = datetime.now().replace(second=0, microsecond=0)
        
        monitor.record_request("test_session", success=False)
        
        assert monitor._request_counts[current_minute] == 1
        assert monitor._error_counts[current_minute] == 1
    
    def test_get_metric_summary(self, monitor):
        """Test getting metric summary"""
        # Record some test metrics
        values = [100.0, 200.0, 300.0, 400.0, 500.0]
        for value in values:
            monitor.record_metric(MetricType.REQUEST_LATENCY, value)
        
        summary = monitor.get_metric_summary(
            MetricType.REQUEST_LATENCY,
            timedelta(minutes=5)
        )
        
        assert summary is not None
        assert summary.metric_type == MetricType.REQUEST_LATENCY
        assert summary.count == 5
        assert summary.min_value == 100.0
        assert summary.max_value == 500.0
        assert summary.avg_value == 300.0
        assert summary.p95_value == 500.0  # 95th percentile
        assert summary.p99_value == 500.0  # 99th percentile
    
    def test_get_metric_summary_empty(self, monitor):
        """Test getting summary for metric with no data"""
        summary = monitor.get_metric_summary(
            MetricType.REQUEST_LATENCY,
            timedelta(minutes=5)
        )
        
        assert summary is None
    
    def test_get_all_metrics_summary(self, monitor):
        """Test getting summary for all metrics"""
        # Record metrics for different types
        monitor.record_metric(MetricType.REQUEST_LATENCY, 1000.0)
        monitor.record_metric(MetricType.API_CALL_LATENCY, 500.0)
        monitor.record_metric(MetricType.CLASSIFICATION_LATENCY, 200.0)
        
        summaries = monitor.get_all_metrics_summary(timedelta(minutes=5))
        
        assert MetricType.REQUEST_LATENCY in summaries
        assert MetricType.API_CALL_LATENCY in summaries
        assert MetricType.CLASSIFICATION_LATENCY in summaries
        
        assert summaries[MetricType.REQUEST_LATENCY].count == 1
        assert summaries[MetricType.API_CALL_LATENCY].count == 1
        assert summaries[MetricType.CLASSIFICATION_LATENCY].count == 1
    
    def test_get_current_availability(self, monitor):
        """Test availability calculation"""
        current_minute = datetime.now().replace(second=0, microsecond=0)
        
        # Record 10 successful requests and 1 failed request
        monitor._request_counts[current_minute] = 11
        monitor._error_counts[current_minute] = 1
        
        availability = monitor.get_current_availability(timedelta(minutes=5))
        
        expected_availability = 1.0 - (1 / 11)  # 10/11 success rate
        assert abs(availability - expected_availability) < 0.001
    
    def test_get_current_availability_no_requests(self, monitor):
        """Test availability calculation with no requests"""
        availability = monitor.get_current_availability(timedelta(minutes=5))
        
        assert availability == 1.0  # 100% availability when no requests
    
    def test_threshold_violation_alert(self, monitor):
        """Test threshold violation alerting"""
        alert_called = False
        alert_metric_type = None
        alert_value = None
        alert_threshold = None
        
        def alert_callback(metric_type, value, threshold):
            nonlocal alert_called, alert_metric_type, alert_value, alert_threshold
            alert_called = True
            alert_metric_type = metric_type
            alert_value = value
            alert_threshold = threshold
        
        monitor.add_alert_callback(alert_callback)
        
        # Record a metric that exceeds threshold (request latency > 2000ms)
        with patch.object(monitor, '_get_threshold', return_value=2000.0):
            monitor.record_metric(MetricType.REQUEST_LATENCY, 3000.0)
        
        assert alert_called
        assert alert_metric_type == MetricType.REQUEST_LATENCY
        assert alert_value == 3000.0
        assert alert_threshold == 2000.0
    
    def test_threshold_no_violation(self, monitor):
        """Test no alert when threshold not exceeded"""
        alert_called = False
        
        def alert_callback(metric_type, value, threshold):
            nonlocal alert_called
            alert_called = True
        
        monitor.add_alert_callback(alert_callback)
        
        # Record a metric that doesn't exceed threshold
        with patch.object(monitor, '_get_threshold', return_value=2000.0):
            monitor.record_metric(MetricType.REQUEST_LATENCY, 1500.0)
        
        assert not alert_called
    
    def test_cleanup_old_metrics(self, monitor):
        """Test cleanup of old metrics"""
        # Record some metrics
        monitor.record_metric(MetricType.REQUEST_LATENCY, 1000.0)
        monitor.record_metric(MetricType.API_CALL_LATENCY, 500.0)
        
        # Manually set old timestamp for one metric
        old_time = datetime.now() - timedelta(hours=2)
        monitor._metrics[MetricType.REQUEST_LATENCY][0].timestamp = old_time
        
        # Add old request counts
        old_minute = datetime.now() - timedelta(hours=2)
        monitor._request_counts[old_minute] = 5
        monitor._error_counts[old_minute] = 1
        
        # Cleanup with 1 hour retention
        monitor.cleanup_old_metrics(timedelta(hours=1))
        
        # Old REQUEST_LATENCY metric should be removed
        assert len(monitor._metrics[MetricType.REQUEST_LATENCY]) == 0
        # Recent API_CALL_LATENCY metric should remain
        assert len(monitor._metrics[MetricType.API_CALL_LATENCY]) == 1
        
        # Old request counts should be removed
        assert old_minute not in monitor._request_counts
        assert old_minute not in monitor._error_counts
    
    def test_get_threshold_values(self, monitor):
        """Test threshold value retrieval"""
        assert monitor._get_threshold(MetricType.REQUEST_LATENCY) == 2000
        assert monitor._get_threshold(MetricType.API_CALL_LATENCY) == 5000
        assert monitor._get_threshold(MetricType.POLICY_LOOKUP_LATENCY) == 3000
        assert monitor._get_threshold(MetricType.ERROR_RATE) == 0.05
        assert monitor._get_threshold(MetricType.AVAILABILITY) == 0.999
    
    def test_disabled_monitoring(self):
        """Test that monitoring is disabled when configured"""
        with patch('src.airline_service.services.performance_monitor.config') as mock_config:
            mock_config.logging.enable_metrics = False
            
            monitor = PerformanceMonitor()
            
            # Recording metrics should do nothing
            monitor.record_metric(MetricType.REQUEST_LATENCY, 1000.0)
            monitor.record_request("test_session", success=True)
            
            assert len(monitor._metrics[MetricType.REQUEST_LATENCY]) == 0
            assert len(monitor._request_counts) == 0
    
    def test_error_rate_calculation(self, monitor):
        """Test error rate calculation and recording"""
        current_minute = datetime.now().replace(second=0, microsecond=0)
        
        # Set up request and error counts
        monitor._request_counts[current_minute] = 10
        monitor._error_counts[current_minute] = 2
        
        # Trigger error rate calculation
        monitor._calculate_error_rate()
        
        # Check that error rate metric was recorded
        error_rate_metrics = monitor._metrics[MetricType.ERROR_RATE]
        assert len(error_rate_metrics) == 1
        assert error_rate_metrics[0].value == 0.2  # 2/10 = 20% error rate
    
    def test_concurrent_metric_recording(self, monitor):
        """Test thread-safe metric recording"""
        import threading
        
        def record_metrics():
            for i in range(100):
                monitor.record_metric(MetricType.REQUEST_LATENCY, float(i))
        
        # Start multiple threads recording metrics
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=record_metrics)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have 500 metrics total (5 threads * 100 metrics each)
        assert len(monitor._metrics[MetricType.REQUEST_LATENCY]) == 500