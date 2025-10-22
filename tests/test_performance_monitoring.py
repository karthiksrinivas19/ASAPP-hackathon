"""
Tests for performance monitoring functionality.

This module tests the comprehensive performance monitoring system including
health checks, metrics collection, alerting, and dashboard integration.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.airline_service.services.health_monitor import (
    HealthMonitor, HealthStatus, AlertSeverity, HealthCheck, Alert
)
from src.airline_service.services.performance_monitor import (
    PerformanceMonitor, MetricType, PerformanceThresholds
)
from src.airline_service.services.metrics_collector import (
    MetricsCollector, SystemHealthMetrics
)


class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create a health monitor instance for testing."""
        return HealthMonitor()
    
    @pytest.mark.asyncio
    async def test_health_check_registration(self, health_monitor):
        """Test health check registration and execution."""
        
        # Create a mock health check
        async def mock_health_check():
            return HealthCheck(
                name="test_check",
                status=HealthStatus.HEALTHY,
                message="Test check passed"
            )
        
        # Register the health check
        health_monitor.register_health_check("test_check", mock_health_check)
        
        # Run health checks
        results = await health_monitor.run_health_checks()
        
        # Verify results
        assert len(results) >= 1
        test_result = next((r for r in results if r.name == "test_check"), None)
        assert test_result is not None
        assert test_result.status == HealthStatus.HEALTHY
        assert test_result.message == "Test check passed"
        assert test_result.response_time_ms is not None
    
    @pytest.mark.asyncio
    async def test_system_health_calculation(self, health_monitor):
        """Test overall system health calculation."""
        
        # Mock metrics collector
        with patch('src.airline_service.services.health_monitor.metrics_collector') as mock_collector:
            mock_collector.get_system_health.return_value = SystemHealthMetrics(
                availability=0.99,
                average_response_time_ms=150.0,
                error_rate=0.02,
                requests_per_minute=100,
                api_success_rate=0.98,
                classification_accuracy=0.95,
                timestamp=datetime.now()
            )
            
            # Get system health
            system_health = await health_monitor.get_system_health()
            
            # Verify health calculation
            assert system_health.availability == 0.99
            assert system_health.response_time_ms == 150.0
            assert system_health.error_rate == 0.02
            assert system_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    def test_alert_management(self, health_monitor):
        """Test alert acknowledgment and resolution."""
        
        # Create a test alert
        alert = Alert(
            id="test_alert_1",
            severity=AlertSeverity.WARNING,
            metric_type=MetricType.REQUEST_LATENCY,
            message="Test alert",
            actual_value=2500.0,
            threshold_value=2000.0,
            timestamp=datetime.now()
        )
        
        # Add alert to monitor
        health_monitor._active_alerts["test_alert_1"] = alert
        
        # Test acknowledgment
        success = health_monitor.acknowledge_alert("test_alert_1")
        assert success is True
        assert health_monitor._active_alerts["test_alert_1"].acknowledged is True
        
        # Test resolution
        success = health_monitor.resolve_alert("test_alert_1")
        assert success is True
        assert health_monitor._active_alerts["test_alert_1"].resolved is True
        
        # Test non-existent alert
        success = health_monitor.acknowledge_alert("non_existent")
        assert success is False
    
    def test_availability_report(self, health_monitor):
        """Test availability report generation."""
        
        with patch('src.airline_service.services.health_monitor.performance_monitor') as mock_monitor:
            mock_monitor.get_current_availability.return_value = 0.995
            
            # Get availability report
            report = health_monitor.get_availability_report(timedelta(hours=24))
            
            # Verify report structure
            assert "availability" in report
            assert "availability_percentage" in report
            assert "target_availability" in report
            assert "sla_met" in report
            assert "downtime_minutes" in report
            assert "uptime_minutes" in report
            
            # Verify calculations
            assert report["availability"] == 0.995
            assert report["sla_met"] is True  # Assuming target is 99.9%


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor instance for testing."""
        monitor = PerformanceMonitor()
        monitor.enabled = True  # Ensure monitoring is enabled for tests
        return monitor
    
    def test_metric_recording(self, performance_monitor):
        """Test metric recording functionality."""
        
        # Record a test metric
        performance_monitor.record_metric(
            MetricType.REQUEST_LATENCY,
            1500.0,
            session_id="test_session",
            context={"test": "data"}
        )
        
        # Verify metric was recorded
        metrics = performance_monitor._metrics[MetricType.REQUEST_LATENCY]
        assert len(metrics) == 1
        
        recorded_metric = metrics[0]
        assert recorded_metric.metric_type == MetricType.REQUEST_LATENCY
        assert recorded_metric.value == 1500.0
        assert recorded_metric.session_id == "test_session"
        assert recorded_metric.context["test"] == "data"
    
    def test_latency_measurement(self, performance_monitor):
        """Test latency measurement context manager."""
        
        import time
        
        # Measure latency
        with performance_monitor.measure_latency(
            MetricType.API_CALL_LATENCY,
            session_id="test_session"
        ):
            time.sleep(0.1)  # Simulate work
        
        # Verify latency was recorded
        metrics = performance_monitor._metrics[MetricType.API_CALL_LATENCY]
        assert len(metrics) == 1
        
        recorded_metric = metrics[0]
        assert recorded_metric.value >= 100.0  # At least 100ms
        assert recorded_metric.session_id == "test_session"
    
    def test_metric_summary(self, performance_monitor):
        """Test metric summary calculation."""
        
        # Record multiple metrics
        values = [100, 200, 300, 400, 500]
        for value in values:
            performance_monitor.record_metric(
                MetricType.REQUEST_LATENCY,
                float(value),
                session_id="test_session"
            )
        
        # Get summary
        summary = performance_monitor.get_metric_summary(
            MetricType.REQUEST_LATENCY,
            timedelta(minutes=5)
        )
        
        # Verify summary calculations
        assert summary is not None
        assert summary.count == 5
        assert summary.min_value == 100.0
        assert summary.max_value == 500.0
        assert summary.avg_value == 300.0
        assert summary.p95_value >= 400.0  # 95th percentile
    
    def test_threshold_checking(self, performance_monitor):
        """Test threshold violation detection."""
        
        # Set up alert callback
        alerts_received = []
        
        def alert_callback(metric_type, actual_value, threshold_value):
            alerts_received.append((metric_type, actual_value, threshold_value))
        
        performance_monitor.add_alert_callback(alert_callback)
        
        # Record metric that exceeds threshold
        performance_monitor.record_metric(
            MetricType.REQUEST_LATENCY,
            3000.0,  # Exceeds default 2000ms threshold
            session_id="test_session"
        )
        
        # Verify alert was triggered
        assert len(alerts_received) == 1
        assert alerts_received[0][0] == MetricType.REQUEST_LATENCY
        assert alerts_received[0][1] == 3000.0
        assert alerts_received[0][2] == 2000.0


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector instance for testing."""
        collector = MetricsCollector()
        collector.enabled = True  # Ensure collection is enabled for tests
        return collector
    
    def test_system_health_collection(self, metrics_collector):
        """Test system health metrics collection."""
        
        with patch('src.airline_service.services.metrics_collector.performance_monitor') as mock_monitor:
            # Mock performance monitor responses
            mock_monitor.get_current_availability.return_value = 0.995
            mock_monitor.get_metric_summary.return_value = Mock(
                avg_value=150.0,
                count=100,
                threshold_violations=0
            )
            
            # Get system health
            health = metrics_collector.get_system_health()
            
            # Verify health metrics
            assert health.availability == 0.995
            assert health.average_response_time_ms == 150.0
            assert health.requests_per_minute >= 0
            assert health.timestamp is not None
    
    def test_dashboard_data_formatting(self, metrics_collector):
        """Test dashboard data formatting."""
        
        with patch('src.airline_service.services.metrics_collector.performance_monitor') as mock_monitor:
            # Mock performance monitor
            mock_monitor.get_current_availability.return_value = 0.99
            mock_monitor.get_all_metrics_summary.return_value = {
                MetricType.REQUEST_LATENCY: Mock(
                    avg_value=200.0,
                    p95_value=350.0,
                    p99_value=500.0,
                    count=50,
                    threshold_violations=2
                )
            }
            
            # Get dashboard data
            dashboard_data = metrics_collector.get_metrics_for_dashboard()
            
            # Verify dashboard structure
            assert "system_health" in dashboard_data
            assert "performance_metrics" in dashboard_data
            assert "alerts" in dashboard_data
            assert "timestamp" in dashboard_data
            
            # Verify system health data
            system_health = dashboard_data["system_health"]
            assert "availability" in system_health
            assert "avg_response_time" in system_health
            assert "error_rate" in system_health
    
    def test_request_metrics_recording(self, metrics_collector):
        """Test request metrics recording."""
        
        with patch('src.airline_service.services.metrics_collector.performance_monitor') as mock_monitor:
            # Record request metrics
            metrics_collector.record_request_metrics(
                session_id="test_session",
                processing_time_ms=250,
                success=True,
                request_type="cancel_trip"
            )
            
            # Verify metrics were recorded
            assert mock_monitor.record_metric.call_count >= 2  # Latency and throughput
            
            # Verify request was recorded for availability
            mock_monitor.record_request.assert_called_once_with("test_session", True)


@pytest.mark.asyncio
async def test_monitoring_endpoints_integration():
    """Test integration of monitoring endpoints."""
    
    from fastapi.testclient import TestClient
    from src.airline_service.main import app
    
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/api/v1/health")
    assert response.status_code in [200, 503]  # OK or Service Unavailable
    
    health_data = response.json()
    assert "status" in health_data
    assert "timestamp" in health_data
    assert "components" in health_data
    
    # Test metrics endpoint
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    
    metrics_data = response.json()
    assert "system_status" in metrics_data
    assert "performance_metrics" in metrics_data
    assert "timestamp" in metrics_data
    
    # Test metrics summary endpoint
    response = client.get("/api/v1/metrics/summary?window_minutes=5")
    assert response.status_code == 200
    
    summary_data = response.json()
    assert "system_health" in summary_data
    assert "latency_metrics" in summary_data
    assert "time_window_minutes" in summary_data
    
    # Test availability report endpoint
    response = client.get("/api/v1/metrics/availability?hours=24")
    assert response.status_code == 200
    
    availability_data = response.json()
    assert "availability" in availability_data
    assert "sla_met" in availability_data
    assert "downtime_minutes" in availability_data


def test_performance_thresholds_configuration():
    """Test performance thresholds configuration."""
    
    thresholds = PerformanceThresholds()
    
    # Verify default thresholds
    assert thresholds.request_latency_ms == 2000
    assert thresholds.api_call_latency_ms == 5000
    assert thresholds.policy_lookup_latency_ms == 3000
    assert thresholds.error_rate_threshold == 0.05
    assert thresholds.availability_threshold == 0.999
    
    # Test custom thresholds
    custom_thresholds = PerformanceThresholds(
        request_latency_ms=1500,
        api_call_latency_ms=3000,
        error_rate_threshold=0.02
    )
    
    assert custom_thresholds.request_latency_ms == 1500
    assert custom_thresholds.api_call_latency_ms == 3000
    assert custom_thresholds.error_rate_threshold == 0.02


if __name__ == "__main__":
    pytest.main([__file__])