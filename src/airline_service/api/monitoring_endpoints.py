"""
Monitoring and health check API endpoints.

This module provides comprehensive monitoring endpoints for health checks,
performance metrics, and system status monitoring.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import structlog

from ..services.health_monitor import health_monitor
from ..services.metrics_collector import metrics_collector
from ..services.performance_monitor import performance_monitor, MetricType
from ..services.cache_service import cache_service
from ..services.connection_pool import connection_pool_manager
from ..config import config

logger = structlog.get_logger("monitoring_api")

# Create router for monitoring endpoints
router = APIRouter(prefix="/api/v1", tags=["monitoring"])


@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns detailed health status of all system components including
    ML classifier, API clients, cache, and performance metrics.
    """
    try:
        # Get comprehensive system health
        system_health = await health_monitor.get_system_health()
        
        # Format response
        health_response = {
            "status": system_health.status.value,
            "timestamp": system_health.timestamp.isoformat(),
            "version": "1.0.0",
            "environment": config.server.environment,
            "overall_metrics": {
                "availability": f"{system_health.availability * 100:.3f}%",
                "response_time_ms": f"{system_health.response_time_ms:.0f}",
                "error_rate": f"{system_health.error_rate * 100:.2f}%",
                "active_alerts": len(system_health.active_alerts)
            },
            "components": {
                check.name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time_ms": check.response_time_ms,
                    "details": check.details
                }
                for check in system_health.health_checks
            },
            "alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "metric": alert.metric_type.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged,
                    "resolved": alert.resolved
                }
                for alert in system_health.active_alerts
            ]
        }
        
        # Set appropriate HTTP status code based on health
        if system_health.status.value == "critical":
            status_code = 503  # Service Unavailable
        elif system_health.status.value == "unhealthy":
            status_code = 503  # Service Unavailable
        elif system_health.status.value == "degraded":
            status_code = 200  # OK but degraded
        else:
            status_code = 200  # OK
        
        return JSONResponse(content=health_response, status_code=status_code)
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": "Health check failed",
                "message": str(e)
            },
            status_code=503
        )


@router.get("/health/simple")
async def simple_health_check():
    """
    Simple health check for load balancers and basic monitoring.
    
    Returns minimal health status with fast response time.
    """
    try:
        # Quick availability check
        availability = performance_monitor.get_current_availability(timedelta(minutes=1))
        
        if availability >= 0.95:
            return {"status": "ok", "timestamp": datetime.now().isoformat()}
        else:
            return JSONResponse(
                content={
                    "status": "degraded", 
                    "availability": f"{availability * 100:.1f}%",
                    "timestamp": datetime.now().isoformat()
                },
                status_code=503
            )
            
    except Exception as e:
        logger.error("Simple health check failed", error=str(e))
        return JSONResponse(
            content={
                "status": "error",
                "timestamp": datetime.now().isoformat()
            },
            status_code=503
        )


@router.get("/metrics")
async def get_performance_metrics():
    """
    Get comprehensive performance metrics for monitoring dashboards.
    
    Returns detailed performance metrics, system health, and alerting data
    suitable for integration with monitoring systems like Grafana, DataDog, etc.
    """
    try:
        dashboard_data = await health_monitor.get_performance_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        logger.error("Failed to retrieve performance metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to retrieve performance metrics",
                "error_code": "METRICS_UNAVAILABLE",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/metrics/summary")
async def get_metrics_summary(
    window_minutes: int = Query(default=5, ge=1, le=60, description="Time window in minutes")
):
    """
    Get performance metrics summary for specified time window.
    
    Args:
        window_minutes: Time window for metrics aggregation (1-60 minutes)
    
    Returns:
        Summary of key performance indicators and system health
    """
    try:
        time_window = timedelta(minutes=window_minutes)
        
        # Get system health
        health = metrics_collector.get_system_health()
        detailed = metrics_collector.get_detailed_metrics(time_window)
        
        summary = {
            "time_window_minutes": window_minutes,
            "timestamp": health.timestamp.isoformat(),
            "system_health": {
                "status": "healthy" if health.availability > 0.99 and health.error_rate < 0.05 else "degraded",
                "availability": health.availability,
                "availability_percentage": f"{health.availability * 100:.3f}%",
                "avg_response_time_ms": health.average_response_time_ms,
                "error_rate": health.error_rate,
                "error_rate_percentage": f"{health.error_rate * 100:.2f}%",
                "requests_per_minute": health.requests_per_minute,
                "api_success_rate": f"{health.api_success_rate * 100:.1f}%",
                "classification_accuracy": f"{health.classification_accuracy * 100:.1f}%"
            },
            "latency_metrics": {}
        }
        
        # Add latency summaries
        for metric_name, metric_summary in [
            ("request_latency", detailed.request_latency),
            ("api_call_latency", detailed.api_call_latency),
            ("classification_latency", detailed.classification_latency),
            ("workflow_latency", detailed.workflow_latency),
            ("policy_lookup_latency", detailed.policy_lookup_latency)
        ]:
            if metric_summary:
                summary["latency_metrics"][metric_name] = {
                    "avg_ms": f"{metric_summary.avg_value:.0f}",
                    "min_ms": f"{metric_summary.min_value:.0f}",
                    "max_ms": f"{metric_summary.max_value:.0f}",
                    "p95_ms": f"{metric_summary.p95_value:.0f}",
                    "p99_ms": f"{metric_summary.p99_value:.0f}",
                    "count": metric_summary.count,
                    "threshold_violations": metric_summary.threshold_violations,
                    "status": "warning" if metric_summary.threshold_violations > 0 else "ok"
                }
        
        return summary
        
    except Exception as e:
        logger.error("Failed to retrieve metrics summary", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to retrieve metrics summary",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/metrics/availability")
async def get_availability_report(
    hours: int = Query(default=24, ge=1, le=168, description="Time window in hours")
):
    """
    Get detailed availability report for SLA monitoring.
    
    Args:
        hours: Time window for availability calculation (1-168 hours)
    
    Returns:
        Detailed availability metrics and SLA compliance status
    """
    try:
        time_window = timedelta(hours=hours)
        availability_report = health_monitor.get_availability_report(time_window)
        
        return availability_report
        
    except Exception as e:
        logger.error("Failed to retrieve availability report", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to retrieve availability report",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/alerts")
async def get_active_alerts():
    """
    Get list of active performance alerts.
    
    Returns:
        List of active alerts with severity levels and acknowledgment status
    """
    try:
        active_alerts = health_monitor.get_active_alerts()
        
        return {
            "active_alerts_count": len(active_alerts),
            "alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "metric_type": alert.metric_type.value,
                    "message": alert.message,
                    "actual_value": alert.actual_value,
                    "threshold_value": alert.threshold_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged,
                    "resolved": alert.resolved,
                    "age_minutes": int((datetime.now() - alert.timestamp).total_seconds() / 60)
                }
                for alert in active_alerts
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to retrieve alerts", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to retrieve alerts",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """
    Acknowledge a performance alert.
    
    Args:
        alert_id: ID of the alert to acknowledge
    
    Returns:
        Acknowledgment status
    """
    try:
        success = health_monitor.acknowledge_alert(alert_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Alert {alert_id} acknowledged",
                "alert_id": alert_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "error",
                    "message": f"Alert {alert_id} not found",
                    "alert_id": alert_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to acknowledge alert", alert_id=alert_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to acknowledge alert",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """
    Resolve a performance alert.
    
    Args:
        alert_id: ID of the alert to resolve
    
    Returns:
        Resolution status
    """
    try:
        success = health_monitor.resolve_alert(alert_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Alert {alert_id} resolved",
                "alert_id": alert_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "error",
                    "message": f"Alert {alert_id} not found",
                    "alert_id": alert_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve alert", alert_id=alert_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to resolve alert",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/cache/stats")
async def get_cache_statistics():
    """
    Get cache performance statistics.
    
    Returns:
        Cache hit rates, memory usage, and connection statistics
    """
    try:
        cache_stats = await cache_service.get_cache_stats()
        connection_stats = await connection_pool_manager.get_connection_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cache": cache_stats,
            "connections": connection_stats,
            "summary": {
                "cache_healthy": cache_stats.get("status") == "connected",
                "hit_rate": cache_stats.get("hit_rate", 0.0),
                "total_connections": connection_stats.get("total_sessions", 0),
                "healthy_connections": connection_stats.get("total_sessions", 0) - connection_stats.get("unhealthy_sessions", 0)
            }
        }
        
    except Exception as e:
        logger.error("Failed to retrieve cache statistics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to retrieve cache statistics",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post("/cache/invalidate")
async def invalidate_cache(request: Dict[str, Any]):
    """
    Invalidate cache entries based on pattern or type.
    
    Request body:
    {
        "pattern": "policy:*",  // Redis key pattern
        "cache_type": "policy"  // Specific cache type to invalidate
    }
    
    Returns:
        Cache invalidation results
    """
    try:
        pattern = request.get("pattern")
        cache_type = request.get("cache_type")
        
        if not pattern and not cache_type:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": "Either 'pattern' or 'cache_type' must be provided"
                }
            )
        
        deleted_count = 0
        
        if pattern:
            deleted_count = await cache_service.delete_pattern(pattern)
        elif cache_type == "policy":
            # Invalidate policy cache
            deleted_count = await cache_service.delete_pattern("policy:*")
        elif cache_type == "api":
            deleted_count = await cache_service.delete_pattern("api_response:*")
        elif cache_type == "all":
            deleted_count = await cache_service.flush_all()
        
        return {
            "status": "success",
            "message": f"Cache invalidation completed",
            "deleted_entries": deleted_count,
            "pattern": pattern,
            "cache_type": cache_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Cache invalidation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Cache invalidation failed",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/status")
async def get_service_status():
    """
    Get detailed service status and configuration information.
    
    Returns:
        Comprehensive service status including capabilities and configuration
    """
    try:
        from ..services.request_classifier_service import ClassifierFactory
        from ..services.workflow_orchestrator import workflow_orchestrator
        from ..types import RequestType
        
        # Get classifier info
        classifier = ClassifierFactory.create_classifier()
        classifier_info = classifier.get_model_info() if classifier.is_loaded() else {"status": "not_loaded"}
        
        # Get workflow info
        workflow_definitions = workflow_orchestrator.registry.get_all_workflows()
        
        status_info = {
            "service": {
                "name": "Airline Customer Service API",
                "version": "1.0.0",
                "environment": config.server.environment,
                "debug_mode": config.server.debug,
                "timestamp": datetime.now().isoformat()
            },
            "capabilities": {
                "supported_request_types": [rt.value for rt in RequestType],
                "ml_classification": classifier_info.get("status") == "loaded",
                "workflow_orchestration": True,
                "automatic_data_retrieval": True,
                "policy_lookup": True,
                "performance_monitoring": config.logging.enable_metrics,
                "audit_logging": config.logging.enable_audit
            },
            "classifier": classifier_info,
            "workflows": {
                request_type.value: {
                    "tasks": len(tasks),
                    "task_types": list(set(task.task_type.value for task in tasks))
                }
                for request_type, tasks in workflow_definitions.items()
            },
            "configuration": {
                "max_request_latency_ms": config.performance.max_request_latency,
                "max_api_latency_ms": config.performance.max_api_latency,
                "target_availability": config.performance.target_availability,
                "log_level": config.logging.level,
                "cache_enabled": True  # Assuming cache is always enabled
            }
        }
        
        return status_info
        
    except Exception as e:
        logger.error("Status check failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to retrieve service status",
                "error_code": "STATUS_CHECK_FAILED",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus format for monitoring integration.
    
    Returns:
        Metrics formatted for Prometheus scraping
    """
    try:
        # Get current metrics
        health = metrics_collector.get_system_health()
        detailed = metrics_collector.get_detailed_metrics()
        
        # Format as Prometheus metrics
        metrics_lines = []
        
        # System health metrics
        metrics_lines.append(f"# HELP airline_service_availability System availability percentage")
        metrics_lines.append(f"# TYPE airline_service_availability gauge")
        metrics_lines.append(f"airline_service_availability {health.availability}")
        
        metrics_lines.append(f"# HELP airline_service_response_time_ms Average response time in milliseconds")
        metrics_lines.append(f"# TYPE airline_service_response_time_ms gauge")
        metrics_lines.append(f"airline_service_response_time_ms {health.average_response_time_ms}")
        
        metrics_lines.append(f"# HELP airline_service_error_rate Error rate percentage")
        metrics_lines.append(f"# TYPE airline_service_error_rate gauge")
        metrics_lines.append(f"airline_service_error_rate {health.error_rate}")
        
        metrics_lines.append(f"# HELP airline_service_requests_per_minute Requests per minute")
        metrics_lines.append(f"# TYPE airline_service_requests_per_minute gauge")
        metrics_lines.append(f"airline_service_requests_per_minute {health.requests_per_minute}")
        
        # Latency metrics
        for metric_name, metric_summary in [
            ("request_latency", detailed.request_latency),
            ("api_call_latency", detailed.api_call_latency),
            ("classification_latency", detailed.classification_latency),
            ("workflow_latency", detailed.workflow_latency)
        ]:
            if metric_summary:
                safe_name = metric_name.replace("_", "_")
                metrics_lines.append(f"# HELP airline_service_{safe_name}_avg_ms Average {metric_name} in milliseconds")
                metrics_lines.append(f"# TYPE airline_service_{safe_name}_avg_ms gauge")
                metrics_lines.append(f"airline_service_{safe_name}_avg_ms {metric_summary.avg_value}")
                
                metrics_lines.append(f"# HELP airline_service_{safe_name}_p95_ms 95th percentile {metric_name} in milliseconds")
                metrics_lines.append(f"# TYPE airline_service_{safe_name}_p95_ms gauge")
                metrics_lines.append(f"airline_service_{safe_name}_p95_ms {metric_summary.p95_value}")
        
        # Active alerts count
        active_alerts = health_monitor.get_active_alerts()
        metrics_lines.append(f"# HELP airline_service_active_alerts_total Number of active alerts")
        metrics_lines.append(f"# TYPE airline_service_active_alerts_total gauge")
        metrics_lines.append(f"airline_service_active_alerts_total {len(active_alerts)}")
        
        return "\n".join(metrics_lines)
        
    except Exception as e:
        logger.error("Failed to generate Prometheus metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to generate Prometheus metrics",
                "timestamp": datetime.now().isoformat()
            }
        )