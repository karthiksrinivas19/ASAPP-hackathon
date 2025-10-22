"""
Tests for metrics collection service
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.airline_service.services.metrics_collector import (
    MetricsCollector, SystemHealthMetrics, DetailedMetrics
)
from src.airline_service.services.performance_monitor import (
    PerformanceMonitor, MetricType, MetricSummary
)


class TestMetricsCollector:
    """Test metrics collector functionality"""
    
    @pytest.fixture
    def mock_monitor(self):
        """Create mock performance monitor"""
        monitor = MagicMock(spec=PerformanceMonitor)
        return monitor
    
    @pytest.fixture
    def collector(self, mock_monitor):
        """Create metrics collector with mock monitor"""
        return MetricsCollector(monitor=mock_monitor)
    
    @pytest.fixture
    def sample_metric_summary(self):
        """Sample metric summary for testing"""
        return MetricSummary(
            metric_type=MetricType.REQUEST_LATENCY,
            count=100,
            min_value=50.0,
            max_value=2000.0,
            avg_value=500.0,
            p95_value=1200.0,
            p99_value=1800.0,
            threshold_violations=5,
            time_window=timedelta(minutes=5)
        )
    
    def test_get_system_health(self, collector, mock_monitor, sample_metric_summary):
        """Test getting system health metrics"""
        # Mock monitor responses
        mock_monitor.get_current_availability.return_value = 0.995
        mock_monitor.get_metric_summary.side_effect = lambda metric_type, window: {
            MetricType.REQUEST_LATENCY: sample_metric_summary,
            MetricType.ERROR_RATE: MetricSummary(
                metric_type=MetricType.ERROR_RATE,
                count=50,
                min_value=0.0,
                max_value=0.1,
                avg_value=0.02,
                p95_value=0.05,
                p99_value=0.08,
                threshold_violations=2,
                time_window=timedelta(minutes=5)
            ),
            MetricType.THROUGHPUT: MetricSummary(
                metric_type=MetricType.THROUGHPUT,
                count=300,
                min_value=1.0,
                max_value=1.0,
                avg_value=1.0,
                p95_value=1.0,
                p99_value=1.0,
                threshold_violations=0,
                time_window=timedelta(minutes=5)
            ),
            MetricType.API_CALL_LATENCY: sample_metric_summary
        }.get(metric_type)
        
        health = collector.get_system_health()
        
        assert isinstance(health, SystemHealthMetrics)
        assert health.availability == 0.995
        assert health.average_response_time_ms == 500.0
        assert health.error_rate == 0.02
        assert health.requests_per_minute == 1  # Throughput summary provided with avg_value=1.0
        assert health.api_success_rate > 0.0
        assert health.classification_accuracy == 0.95  # Default value
        assert isinstance(health.timestamp, datetime)
    
    def test_get_system_health_caching(self, collector, mock_monitor):
        """Test system health metrics caching"""
        mock_monitor.get_current_availability.return_value = 0.99
        mock_monitor.get_metric_summary.return_value = None
        
        # First call should query monitor
        health1 = collector.get_system_health()
        assert mock_monitor.get_current_availability.call_count == 1
        
        # Second call within cache TTL should use cache
        health2 = collector.get_system_health()
        assert mock_monitor.get_current_availability.call_count == 1  # No additional call
        assert health1.timestamp == health2.timestamp
        
        # Force refresh should query monitor again
        health3 = collector.get_system_health(force_refresh=True)
        assert mock_monitor.get_current_availability.call_count == 2
    
    def test_get_detailed_metrics(self, collector, mock_monitor, sample_metric_summary):
        """Test getting detailed metrics"""
        # Mock all metrics summary
        mock_monitor.get_all_metrics_summary.return_value = {
            MetricType.REQUEST_LATENCY: sample_metric_summary,
            MetricType.API_CALL_LATENCY: sample_metric_summary,
            MetricType.CLASSIFICATION_LATENCY: sample_metric_summary
        }
        
        detailed = collector.get_detailed_metrics()
        
        assert isinstance(detailed, DetailedMetrics)
        assert detailed.request_latency == sample_metric_summary
        assert detailed.api_call_latency == sample_metric_summary
        assert detailed.classification_latency == sample_metric_summary
        assert detailed.policy_lookup_latency is None  # Not in mock response
        assert isinstance(detailed.timestamp, datetime)
    
    def test_get_metrics_for_dashboard(self, collector, mock_monitor, sample_metric_summary):
        """Test getting dashboard-formatted metrics"""
        # Mock system health
        mock_monitor.get_current_availability.return_value = 0.998
        mock_monitor.get_metric_summary.side_effect = lambda metric_type, window: {
            MetricType.REQUEST_LATENCY: sample_metric_summary,
            MetricType.ERROR_RATE: MetricSummary(
                metric_type=MetricType.ERROR_RATE,
                count=50,
                min_value=0.0,
                max_value=0.05,
                avg_value=0.01,
                p95_value=0.03,
                p99_value=0.04,
                threshold_violations=0,
                time_window=timedelta(minutes=5)
            )
        }.get(metric_type)
        
        # Mock detailed metrics
        mock_monitor.get_all_metrics_summary.return_value = {
            MetricType.REQUEST_LATENCY: sample_metric_summary,
            MetricType.API_CALL_LATENCY: sample_metric_summary
        }
        
        dashboard = collector.get_metrics_for_dashboard()
        
        assert "system_health" in dashboard
        assert "performance_metrics" in dashboard
        assert "alerts" in dashboard
        assert "timestamp" in dashboard
        
        # Check system health formatting
        health = dashboard["system_health"]
        assert health["status"] == "good"  # 99.8% availability, 1% error rate
        assert health["availability"] == "99.80%"
        assert health["avg_response_time"] == "500ms"
        assert health["error_rate"] == "1.00%"
        
        # Check performance metrics formatting
        perf = dashboard["performance_metrics"]
        assert "request_latency" in perf
        assert perf["request_latency"]["avg"] == "500ms"
        assert perf["request_latency"]["p95"] == "1200ms"
        assert perf["request_latency"]["p99"] == "1800ms"
        assert perf["request_latency"]["count"] == 100
        assert perf["request_latency"]["threshold_violations"] == 5
    
    def test_record_request_metrics(self, collector, mock_monitor):
        """Test recording request metrics"""
        collector.record_request_metrics(
            session_id="test_session",
            processing_time_ms=1500,
            success=True,
            request_type="cancel_trip"
        )
        
        # Should record latency, request, and throughput
        assert mock_monitor.record_metric.call_count == 2  # Latency and throughput
        assert mock_monitor.record_request.call_count == 1
        
        # Check latency metric call
        latency_call = mock_monitor.record_metric.call_args_list[0]
        assert latency_call[0][0] == MetricType.REQUEST_LATENCY
        assert latency_call[0][1] == 1500
        assert latency_call[1]["session_id"] == "test_session"
        assert latency_call[1]["context"]["request_type"] == "cancel_trip"
        
        # Check request recording
        request_call = mock_monitor.record_request.call_args
        assert request_call[0][0] == "test_session"
        assert request_call[0][1] is True
    
    def test_record_api_metrics(self, collector, mock_monitor):
        """Test recording API metrics"""
        collector.record_api_metrics(
            session_id="test_session",
            endpoint="/flight/booking",
            response_time_ms=800,
            success=True,
            status_code=200
        )
        
        mock_monitor.record_metric.assert_called_once_with(
            MetricType.API_CALL_LATENCY,
            800,
            session_id="test_session",
            context={
                "endpoint": "/flight/booking",
                "success": True,
                "status_code": 200
            }
        )
    
    def test_record_classification_metrics(self, collector, mock_monitor):
        """Test recording classification metrics"""
        collector.record_classification_metrics(
            session_id="test_session",
            processing_time_ms=300,
            confidence=0.92,
            request_type="cancel_trip"
        )
        
        mock_monitor.record_metric.assert_called_once_with(
            MetricType.CLASSIFICATION_LATENCY,
            300,
            session_id="test_session",
            context={
                "confidence": 0.92,
                "request_type": "cancel_trip"
            }
        )
    
    def test_record_workflow_metrics(self, collector, mock_monitor):
        """Test recording workflow metrics"""
        collector.record_workflow_metrics(
            session_id="test_session",
            processing_time_ms=2000,
            task_count=3,
            success=True
        )
        
        mock_monitor.record_metric.assert_called_once_with(
            MetricType.WORKFLOW_LATENCY,
            2000,
            session_id="test_session",
            context={
                "task_count": 3,
                "success": True
            }
        )
    
    def test_get_health_check_data(self, collector, mock_monitor):
        """Test getting health check data"""
        # Mock system health
        mock_monitor.get_current_availability.return_value = 0.995
        mock_monitor.get_metric_summary.side_effect = lambda metric_type, window: {
            MetricType.REQUEST_LATENCY: MetricSummary(
                metric_type=MetricType.REQUEST_LATENCY,
                count=100,
                min_value=100.0,
                max_value=1800.0,
                avg_value=800.0,
                p95_value=1500.0,
                p99_value=1700.0,
                threshold_violations=2,
                time_window=timedelta(minutes=5)
            ),
            MetricType.ERROR_RATE: MetricSummary(
                metric_type=MetricType.ERROR_RATE,
                count=50,
                min_value=0.0,
                max_value=0.08,
                avg_value=0.03,
                p95_value=0.06,
                p99_value=0.07,
                threshold_violations=1,
                time_window=timedelta(minutes=5)
            )
        }.get(metric_type)
        
        health_check = collector.get_health_check_data()
        
        assert health_check["status"] == "healthy"  # 99.5% availability, 3% error rate
        assert "timestamp" in health_check
        assert "metrics" in health_check
        assert "checks" in health_check
        
        # Check metrics
        metrics = health_check["metrics"]
        assert metrics["availability"] == 0.995
        assert metrics["avg_response_time_ms"] == 800.0
        assert metrics["error_rate"] == 0.03
        
        # Check health checks
        checks = health_check["checks"]
        assert checks["availability_ok"] is True  # >= 0.99
        assert checks["response_time_ok"] is True  # <= 2000ms
        assert checks["error_rate_ok"] is True  # <= 0.05
    
    def test_get_health_status_levels(self, collector, mock_monitor):
        """Test different health status levels"""
        test_cases = [
            # (availability, error_rate, expected_status)
            (0.999, 0.005, "excellent"),
            (0.995, 0.02, "good"),
            (0.98, 0.08, "degraded"),
            (0.90, 0.15, "poor")
        ]
        
        for availability, error_rate, expected_status in test_cases:
            # Mock system health
            mock_monitor.get_current_availability.return_value = availability
            mock_monitor.get_metric_summary.side_effect = lambda metric_type, window: {
                MetricType.ERROR_RATE: MetricSummary(
                    metric_type=MetricType.ERROR_RATE,
                    count=100,
                    min_value=0.0,
                    max_value=error_rate,
                    avg_value=error_rate,
                    p95_value=error_rate,
                    p99_value=error_rate,
                    threshold_violations=0,
                    time_window=timedelta(minutes=5)
                )
            }.get(metric_type)
            
            # Clear cache to get fresh data
            collector._cached_health_metrics = None
            
            # Mock detailed metrics to return None for most metrics
            mock_monitor.get_all_metrics_summary.return_value = {}
            
            dashboard = collector.get_metrics_for_dashboard()
            assert dashboard["system_health"]["status"] == expected_status
    
    def test_get_active_alerts(self, collector, mock_monitor, sample_metric_summary):
        """Test active alerts detection"""
        # Create metric summary with threshold violations
        violation_summary = MetricSummary(
            metric_type=MetricType.REQUEST_LATENCY,
            count=100,
            min_value=100.0,
            max_value=5000.0,  # High max value
            avg_value=1000.0,
            p95_value=3000.0,
            p99_value=4500.0,
            threshold_violations=8,  # High violation count
            time_window=timedelta(minutes=5)
        )
        
        # Mock detailed metrics with violations
        mock_monitor.get_all_metrics_summary.return_value = {
            MetricType.REQUEST_LATENCY: violation_summary,
            MetricType.API_CALL_LATENCY: sample_metric_summary  # Has 5 violations
        }
        
        detailed = collector.get_detailed_metrics()
        alerts = collector._get_active_alerts(detailed)
        
        assert len(alerts) == 2
        
        # Check request latency alert
        request_alert = next(a for a in alerts if a["metric"] == "request_latency")
        assert request_alert["type"] == "threshold_violation"
        assert request_alert["violations"] == 8
        assert request_alert["severity"] == "critical"  # >= 5 violations
        
        # Check API latency alert
        api_alert = next(a for a in alerts if a["metric"] == "api_call_latency")
        assert api_alert["violations"] == 5
        assert api_alert["severity"] == "critical"  # >= 5 violations
    
    def test_disabled_metrics(self):
        """Test behavior when metrics are disabled"""
        with patch('src.airline_service.config.config') as mock_config:
            mock_config.logging.enable_metrics = False
            
            collector = MetricsCollector()
            
            # Should return default values
            health = collector.get_system_health()
            assert health.availability == 1.0
            assert health.average_response_time_ms == 0.0
            assert health.error_rate == 0.0
            
            detailed = collector.get_detailed_metrics()
            assert detailed.request_latency is None
            assert detailed.api_call_latency is None