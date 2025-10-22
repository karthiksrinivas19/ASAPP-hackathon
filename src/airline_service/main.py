"""
Main application entry point
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import structlog
from datetime import datetime

from .config import config
from .types import CustomerRequest, APIResponse, RequestType
from .services import (
    audit_logger, performance_monitor, metrics_collector, MetricType,
    cache_service, connection_pool_manager, health_monitor
)
from .api.monitoring_endpoints import router as monitoring_router
from .container import container, configure_environment
from .error_handlers import (
    ErrorHandler, ErrorCode, GracefulDegradationHandler, ExceptionMapper, ErrorLogger,
    global_exception_handler, http_exception_handler, validation_exception_handler
)


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if config.logging.log_format == "json" else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Airline Customer Service API", version="1.0.0")
    
    # Configure environment-specific settings
    configure_environment()
    
    # Initialize dependency injection container
    try:
        await container.initialize()
        logger.info("Service container initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize service container", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Airline Customer Service API")
    
    # Cleanup resources
    try:
        await container.cleanup()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error("Error during cleanup", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="Airline Customer Service API",
    description="Automated airline customer service system with ML-based request classification",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if config.server.debug else None,
    redoc_url="/redoc" if config.server.debug else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.is_development else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if config.is_development else ["yourdomain.com", "*.yourdomain.com"]
)

# Include monitoring endpoints
app.include_router(monitoring_router)

# Add global error handlers
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(422, validation_exception_handler)


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the service and its dependencies
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "environment": config.server.environment,
            "components": {}
        }
        
        # Check services through container
        if container.is_initialized():
            # Check ML classifier
            try:
                classifier = container.get_classifier()
                if classifier.is_loaded():
                    health_status["components"]["classifier"] = {
                        "status": "healthy",
                        "model_info": classifier.get_model_info()
                    }
                else:
                    health_status["components"]["classifier"] = {
                        "status": "degraded",
                        "message": "Using mock classifier"
                    }
            except Exception as e:
                health_status["components"]["classifier"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check airline API client
            try:
                client = container.get_airline_client()
                health_status["components"]["airline_api"] = {
                    "status": "healthy",
                    "type": "mock"
                }
            except Exception as e:
                health_status["components"]["airline_api"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check policy service
            try:
                policy_svc = container.get_policy_service()
                health_status["components"]["policy_service"] = {
                    "status": "healthy"
                }
            except Exception as e:
                health_status["components"]["policy_service"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check workflow orchestrator
            try:
                orchestrator = container.get_workflow_orchestrator()
                health_status["components"]["workflow_orchestrator"] = {
                    "status": "healthy",
                    "registered_workflows": len(orchestrator.registry.get_all_workflows())
                }
            except Exception as e:
                health_status["components"]["workflow_orchestrator"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            health_status["components"]["container"] = {
                "status": "unhealthy",
                "message": "Service container not initialized"
            }
        
        # Add performance metrics
        try:
            system_health = metrics_collector.get_system_health()
            health_status["performance"] = {
                "availability": f"{system_health.availability * 100:.2f}%",
                "avg_response_time_ms": f"{system_health.average_response_time_ms:.0f}",
                "error_rate": f"{system_health.error_rate * 100:.2f}%",
                "requests_per_minute": system_health.requests_per_minute
            }
        except Exception as e:
            health_status["performance"] = {
                "status": "unavailable",
                "error": str(e)
            }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if any(status == "unhealthy" for status in component_statuses):
            health_status["status"] = "unhealthy"
        elif any(status == "degraded" for status in component_statuses):
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        audit_logger.log_error(
            session_id="system",
            error_type="HEALTH_CHECK_FAILED",
            error_message=str(e)
        )
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.get("/api/v1/metrics")
async def get_metrics():
    """
    Get detailed performance metrics
    
    Returns comprehensive performance metrics for monitoring and alerting
    """
    try:
        dashboard_data = metrics_collector.get_metrics_for_dashboard()
        return dashboard_data
        
    except Exception as e:
        logger.error("Failed to retrieve metrics", error=str(e))
        audit_logger.log_error(
            session_id="system",
            error_type="METRICS_RETRIEVAL_FAILED",
            error_message=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to retrieve metrics",
                "error_code": "METRICS_UNAVAILABLE",
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/api/v1/metrics/summary")
async def get_metrics_summary():
    """
    Get metrics summary for the last 5 minutes
    
    Returns key performance indicators and system health summary
    """
    try:
        health = metrics_collector.get_system_health()
        detailed = metrics_collector.get_detailed_metrics()
        
        summary = {
            "timestamp": health.timestamp.isoformat(),
            "system_health": {
                "status": "healthy" if health.availability > 0.99 and health.error_rate < 0.05 else "degraded",
                "availability": health.availability,
                "avg_response_time_ms": health.average_response_time_ms,
                "error_rate": health.error_rate,
                "requests_per_minute": health.requests_per_minute
            },
            "latency_metrics": {}
        }
        
        # Add latency summaries
        for metric_name, metric_summary in [
            ("request_latency", detailed.request_latency),
            ("api_call_latency", detailed.api_call_latency),
            ("classification_latency", detailed.classification_latency),
            ("workflow_latency", detailed.workflow_latency)
        ]:
            if metric_summary:
                summary["latency_metrics"][metric_name] = {
                    "avg_ms": metric_summary.avg_value,
                    "p95_ms": metric_summary.p95_value,
                    "p99_ms": metric_summary.p99_value,
                    "count": metric_summary.count,
                    "threshold_violations": metric_summary.threshold_violations
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


@app.get("/api/v1/status")
async def service_status():
    """
    Detailed service status endpoint
    
    Returns detailed information about service capabilities and configuration
    """
    try:
        if not container.is_initialized():
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error",
                    "message": "Service container not initialized",
                    "error_code": "SERVICE_UNAVAILABLE",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Get services from container
        classifier = container.get_classifier()
        orchestrator = container.get_workflow_orchestrator()
        
        # Get classifier info
        classifier_info = classifier.get_model_info() if classifier.is_loaded() else {"status": "not_loaded"}
        
        # Get workflow info
        workflow_definitions = orchestrator.registry.get_all_workflows()
        
        status_info = {
            "service": {
                "name": "Airline Customer Service API",
                "version": "1.0.0",
                "environment": config.server.environment,
                "uptime": "N/A",  # Would track actual uptime in production
                "timestamp": datetime.now().isoformat(),
                "container_initialized": container.is_initialized()
            },
            "capabilities": {
                "supported_request_types": [rt.value for rt in RequestType],
                "ml_classification": classifier_info.get("status") == "loaded",
                "workflow_orchestration": True,
                "automatic_data_retrieval": True,
                "policy_lookup": True
            },
            "classifier": classifier_info,
            "workflows": {
                request_type.value: {
                    "tasks": len(tasks),
                    "task_types": list(set(task.task_type.value for task in tasks))
                }
                for request_type, tasks in workflow_definitions.items()
            },
            "services": container.list_services(),
            "configuration": {
                "debug_mode": config.server.debug,
                "log_level": config.logging.level,
                "max_request_size": "1000 characters"
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


@app.post("/api/v1/customer-service/query", response_model=APIResponse)
async def process_customer_query(request: CustomerRequest):
    """
    Main customer service query endpoint
    
    This endpoint handles all customer service requests using the complete workflow system:
    - Flight cancellations
    - Flight status inquiries  
    - Seat availability checks
    - Policy information requests
    - Pet travel inquiries
    
    The processing pipeline:
    1. Request validation
    2. ML-based request classification
    3. Workflow orchestration
    4. Task execution
    5. Response formatting
    """
    request_start_time = datetime.now()
    session_id = request.session_id or f"session_{int(datetime.now().timestamp())}"
    
    try:
        # Generate request ID for tracking
        request_id = f"req_{int(datetime.now().timestamp() * 1000)}"
        
        # Log incoming request with audit logging
        audit_logger.log_customer_request(session_id, request, request_id)
        
        logger.info(
            "Processing customer query",
            session_id=session_id,
            request_id=request_id,
            utterance=request.utterance[:100] + "..." if len(request.utterance) > 100 else request.utterance,
            customer_id=request.customer_id
        )
        
        # 1. Request Validation
        if not request.utterance or not request.utterance.strip():
            logger.warning("Empty utterance received", session_id=session_id)
            raise HTTPException(
                status_code=400,
                detail=ErrorHandler.create_error_response(
                    status_code=400,
                    message=ErrorHandler.get_user_friendly_message(ErrorCode.UTTERANCE_EMPTY),
                    error_code=ErrorCode.UTTERANCE_EMPTY,
                    session_id=session_id,
                    request_id=request_id
                )
            )
        
        if len(request.utterance) > 1000:  # Reasonable limit
            logger.warning("Utterance too long", session_id=session_id, length=len(request.utterance))
            raise HTTPException(
                status_code=400,
                detail=ErrorHandler.create_error_response(
                    status_code=400,
                    message=ErrorHandler.get_user_friendly_message(ErrorCode.UTTERANCE_TOO_LONG),
                    error_code=ErrorCode.UTTERANCE_TOO_LONG,
                    session_id=session_id,
                    request_id=request_id
                )
            )
        
        # 2. Get Services from Container with Graceful Degradation
        try:
            if not container.is_initialized():
                logger.error("Service container not initialized", session_id=session_id)
                # Try graceful degradation for classifier
                return APIResponse(**(await GracefulDegradationHandler.handle_classifier_unavailable(request.utterance)))
            
            # Get services from dependency injection container
            classifier = container.get_classifier()
            workflow_orchestrator = container.get_workflow_orchestrator()
            
            if not classifier.is_loaded():
                logger.warning("Classifier not loaded, using graceful degradation", session_id=session_id)
                return APIResponse(**(await GracefulDegradationHandler.handle_classifier_unavailable(request.utterance)))
            
        except Exception as e:
            ErrorLogger.log_error(
                exception=e,
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                session_id=session_id,
                request_id=request_id,
                context={"step": "service_initialization"}
            )
            # Try graceful degradation
            return APIResponse(**(await GracefulDegradationHandler.handle_classifier_unavailable(request.utterance)))
        
        # 3. Classify Request
        try:
            classification_start = datetime.now()
            
            # Classify with performance monitoring
            with performance_monitor.measure_latency(
                MetricType.CLASSIFICATION_LATENCY, 
                session_id=session_id, 
                request_id=request_id
            ):
                classification_result = await classifier.classify_request(request.utterance)
            
            classification_time = (datetime.now() - classification_start).total_seconds()
            
            # Log classification result
            audit_logger.log_classification_result(
                session_id=session_id,
                classification=classification_result,
                processing_time_ms=int(classification_time * 1000),
                request_id=request_id
            )
            
            # Record classification metrics
            metrics_collector.record_classification_metrics(
                session_id=session_id,
                processing_time_ms=int(classification_time * 1000),
                confidence=classification_result.confidence,
                request_type=classification_result.request_type.value
            )
            
            logger.info(
                "Request classified",
                session_id=session_id,
                request_id=request_id,
                request_type=classification_result.request_type.value,
                confidence=classification_result.confidence,
                classification_time_ms=int(classification_time * 1000),
                entities_found=len(classification_result.extracted_entities)
            )
            
            # Check classification confidence
            if classification_result.confidence < 0.3:
                logger.warning(
                    "Low classification confidence",
                    session_id=session_id,
                    confidence=classification_result.confidence
                )
                
                return APIResponse(
                    status="completed",
                    message="I'm not sure I understand your request. Could you please provide more details about what you'd like help with?",
                    data={
                        "request_type": "unknown",
                        "confidence": classification_result.confidence,
                        "suggestions": [
                            "Try mentioning specific details like your booking reference (PNR)",
                            "Specify what you need help with (cancellation, flight status, etc.)",
                            "Include your flight number or travel dates"
                        ]
                    }
                )
            
        except Exception as e:
            ErrorLogger.log_error(
                exception=e,
                error_code=ErrorCode.CLASSIFICATION_FAILED,
                session_id=session_id,
                request_id=request_id,
                context={"step": "classification", "utterance_length": len(request.utterance)}
            )
            # Try graceful degradation
            return APIResponse(**(await GracefulDegradationHandler.handle_classifier_unavailable(request.utterance)))
        
        # 4. Create Request Context
        request_context = RequestContext(
            session_id=session_id,
            customer_id=request.customer_id,
            request_type=classification_result.request_type,
            utterance=request.utterance,
            timestamp=request_start_time,
            metadata={
                "extracted_entities": classification_result.extracted_entities,
                "confidence": classification_result.confidence,
                "alternatives": classification_result.alternative_intents,
                "classification_time_ms": int(classification_time * 1000)
            }
        )
        
        # 5. Execute Workflow
        try:
            workflow_start = datetime.now()
            
            # Execute workflow with performance monitoring
            with performance_monitor.measure_latency(
                MetricType.WORKFLOW_LATENCY, 
                session_id=session_id, 
                request_id=request_id
            ):
                workflow_result = await workflow_orchestrator.execute_workflow(request_context)
            
            workflow_time = (datetime.now() - workflow_start).total_seconds()
            
            # Log workflow execution
            audit_logger.log_workflow_execution(
                session_id=session_id,
                workflow_result=workflow_result,
                request_type=classification_result.request_type,
                processing_time_ms=int(workflow_time * 1000),
                request_id=request_id
            )
            
            # Record workflow metrics
            metrics_collector.record_workflow_metrics(
                session_id=session_id,
                processing_time_ms=int(workflow_time * 1000),
                task_count=len(workflow_result.executed_tasks),
                success=workflow_result.success
            )
            
            logger.info(
                "Workflow completed",
                session_id=session_id,
                request_id=request_id,
                success=workflow_result.success,
                executed_tasks=workflow_result.executed_tasks,
                workflow_time_ms=int(workflow_time * 1000),
                total_time_ms=int((datetime.now() - request_start_time).total_seconds() * 1000)
            )
            
        except Exception as e:
            ErrorLogger.log_error(
                exception=e,
                error_code=ErrorCode.WORKFLOW_FAILED,
                session_id=session_id,
                request_id=request_id,
                context={
                    "step": "workflow_execution",
                    "request_type": classification_result.request_type.value,
                    "confidence": classification_result.confidence
                }
            )
            
            # Try graceful degradation based on request type
            try:
                return APIResponse(**(await GracefulDegradationHandler.handle_api_unavailable(
                    classification_result.request_type.value
                )))
            except Exception:
                # Final fallback
                raise HTTPException(
                    status_code=500,
                    detail=ErrorHandler.create_error_response(
                        status_code=500,
                        message=ErrorHandler.get_user_friendly_message(ErrorCode.WORKFLOW_FAILED),
                        error_code=ErrorCode.WORKFLOW_FAILED,
                        session_id=session_id,
                        request_id=request_id
                    )
                )
        
        # 6. Format and Return Response
        if workflow_result.success:
            # Extract the final response from the workflow result
            if workflow_result.data and isinstance(workflow_result.data, dict):
                if "response" in workflow_result.data:
                    # Return the formatted response from INFORM_CUSTOMER task
                    final_response = workflow_result.data["response"]
                    
                    # Add metadata to response
                    if hasattr(final_response, 'data') and final_response.data:
                        final_response.data.update({
                            "session_id": session_id,
                            "processing_time_ms": int((datetime.now() - request_start_time).total_seconds() * 1000),
                            "executed_tasks": workflow_result.executed_tasks
                        })
                    
                    # Log successful response
                    total_processing_time = int((datetime.now() - request_start_time).total_seconds() * 1000)
                    audit_logger.log_customer_response(
                        session_id=session_id,
                        response=final_response,
                        request_id=request_id,
                        processing_time_ms=total_processing_time
                    )
                    
                    # Record request metrics
                    metrics_collector.record_request_metrics(
                        session_id=session_id,
                        processing_time_ms=total_processing_time,
                        success=True,
                        request_type=classification_result.request_type.value
                    )
                    
                    return final_response
                else:
                    # Fallback response with workflow data
                    total_processing_time = int((datetime.now() - request_start_time).total_seconds() * 1000)
                    fallback_response = APIResponse(
                        status="completed",
                        message=workflow_result.message or "Request processed successfully",
                        data={
                            "session_id": session_id,
                            "request_type": classification_result.request_type.value,
                            "processing_time_ms": total_processing_time,
                            "executed_tasks": workflow_result.executed_tasks,
                            "result": workflow_result.data
                        }
                    )
                    
                    # Log response
                    audit_logger.log_customer_response(
                        session_id=session_id,
                        response=fallback_response,
                        request_id=request_id,
                        processing_time_ms=total_processing_time
                    )
                    
                    # Record metrics
                    metrics_collector.record_request_metrics(
                        session_id=session_id,
                        processing_time_ms=total_processing_time,
                        success=True,
                        request_type=classification_result.request_type.value
                    )
                    
                    return fallback_response
            else:
                # Simple success response
                total_processing_time = int((datetime.now() - request_start_time).total_seconds() * 1000)
                simple_response = APIResponse(
                    status="completed",
                    message=workflow_result.message or "Request processed successfully",
                    data={
                        "session_id": session_id,
                        "request_type": classification_result.request_type.value,
                        "processing_time_ms": total_processing_time,
                        "executed_tasks": workflow_result.executed_tasks
                    }
                )
                
                # Log response
                audit_logger.log_customer_response(
                    session_id=session_id,
                    response=simple_response,
                    request_id=request_id,
                    processing_time_ms=total_processing_time
                )
                
                # Record metrics
                metrics_collector.record_request_metrics(
                    session_id=session_id,
                    processing_time_ms=total_processing_time,
                    success=True,
                    request_type=classification_result.request_type.value
                )
                
                return simple_response
        else:
            # Workflow failed - return appropriate error response
            error_message = workflow_result.message or "Unable to process your request"
            
            # Determine appropriate HTTP status code based on error type
            if "not found" in error_message.lower() or "booking not found" in error_message.lower():
                status_code = 404
                error_code = "RESOURCE_NOT_FOUND"
            elif "invalid" in error_message.lower() or "missing" in error_message.lower():
                status_code = 400
                error_code = "INVALID_REQUEST"
            else:
                status_code = 500
                error_code = "PROCESSING_FAILED"
            
            logger.warning(
                "Workflow failed",
                session_id=session_id,
                request_id=request_id,
                error_message=error_message,
                executed_tasks=workflow_result.executed_tasks
            )
            
            # Log error and record metrics for failed request
            total_processing_time = int((datetime.now() - request_start_time).total_seconds() * 1000)
            audit_logger.log_error(
                session_id=session_id,
                error_type="WORKFLOW_FAILED",
                error_message=error_message,
                error_code=error_code,
                request_id=request_id,
                context={"executed_tasks": workflow_result.executed_tasks}
            )
            
            metrics_collector.record_request_metrics(
                session_id=session_id,
                processing_time_ms=total_processing_time,
                success=False,
                request_type=classification_result.request_type.value
            )
            
            raise HTTPException(
                status_code=status_code,
                detail=ErrorHandler.create_error_response(
                    status_code=status_code,
                    message=ErrorHandler.get_user_friendly_message(error_code, error_message),
                    error_code=error_code,
                    details={"executed_tasks": workflow_result.executed_tasks},
                    session_id=session_id,
                    request_id=request_id
                )
            )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        # Catch-all for unexpected errors
        total_time = (datetime.now() - request_start_time).total_seconds()
        
        # Log error with comprehensive error handling
        ErrorLogger.log_error(
            exception=e,
            error_code=ErrorCode.INTERNAL_ERROR,
            session_id=session_id,
            request_id=request_id if 'request_id' in locals() else None,
            context={
                "utterance_length": len(request.utterance),
                "processing_time_ms": int(total_time * 1000),
                "step": "main_processing"
            }
        )
        
        # Record failed request metrics
        metrics_collector.record_request_metrics(
            session_id=session_id,
            processing_time_ms=int(total_time * 1000),
            success=False
        )
        
        # Try final graceful degradation
        try:
            return APIResponse(**(await GracefulDegradationHandler.handle_classifier_unavailable(request.utterance)))
        except Exception:
            # Absolute final fallback
            raise HTTPException(
                status_code=500,
                detail=ErrorHandler.create_error_response(
                    status_code=500,
                    message=ErrorHandler.get_user_friendly_message(ErrorCode.INTERNAL_ERROR),
                    error_code=ErrorCode.INTERNAL_ERROR,
                    session_id=session_id,
                    request_id=request_id if 'request_id' in locals() else None
                )
            )


@app.post("/api/v1/customer-service/query/simple", response_model=APIResponse)
async def process_customer_query_simple(request: CustomerRequest):
    """
    Simple customer query processing (original implementation for comparison)
    """
    try:
        logger.info("Processing simple customer query", utterance=request.utterance)
        
        # Get classifier from container with graceful degradation
        if not container.is_initialized():
            logger.warning("Service container not initialized, using fallback")
            return APIResponse(**(await GracefulDegradationHandler.handle_classifier_unavailable(request.utterance)))
        
        classifier = container.get_classifier()
        
        if not classifier.is_loaded():
            logger.warning("Classifier not loaded, using graceful degradation")
            return APIResponse(**(await GracefulDegradationHandler.handle_classifier_unavailable(request.utterance)))
        
        # 1. Classify request using ML model
        classification_result = await classifier.classify_request(request.utterance)
        
        # 2. Extract entities (already done in classifier)
        entities = classification_result.extracted_entities
        
        # 3. Create response based on classification
        intent = classification_result.request_type
        confidence = classification_result.confidence
        
        # Generate appropriate response based on intent
        if intent == "cancel_trip":
            message = "I can help you cancel your flight. To proceed with the cancellation, I'll need your booking reference (PNR) or flight details."
            if entities:
                pnr_entities = [e for e in entities if e.type == "pnr"]
                if pnr_entities:
                    pnr = pnr_entities[0].value
                    message = f"I found your booking reference {pnr}. I can help you cancel this flight. Please note that cancellation fees may apply based on your fare type."
        
        elif intent == "flight_status":
            message = "I can check your flight status. Let me look up the current information for your flight."
            if entities:
                flight_entities = [e for e in entities if e.type == "flight_number"]
                pnr_entities = [e for e in entities if e.type == "pnr"]
                if flight_entities:
                    flight = flight_entities[0].value
                    message = f"Checking status for flight {flight}. Your flight is currently on time with no delays reported."
                elif pnr_entities:
                    pnr = pnr_entities[0].value
                    message = f"Checking status for booking {pnr}. Your flight is currently on time with no delays reported."
        
        elif intent == "seat_availability":
            message = "I can show you available seats on your flight. Let me check the current seat map."
            if entities:
                class_entities = [e for e in entities if e.type == "class"]
                seat_type_entities = [e for e in entities if e.type == "seat_type"]
                if class_entities:
                    seat_class = class_entities[0].value
                    message = f"Checking available seats in {seat_class} class. I found several options for you."
                elif seat_type_entities:
                    seat_type = seat_type_entities[0].value
                    message = f"Looking for {seat_type} seats. I found several {seat_type} seats available."
        
        elif intent == "cancellation_policy":
            message = "Our cancellation policy varies by fare type. Generally, you can cancel flights up to 24 hours before departure. Fees may apply depending on your ticket type."
        
        elif intent == "pet_travel":
            message = "I can help you with pet travel information. Small pets can travel in the cabin in approved carriers, while larger pets may need to travel as cargo. Service animals are always welcome."
            if entities:
                pet_entities = [e for e in entities if e.type == "pet_type"]
                if pet_entities:
                    pet_type = pet_entities[0].value
                    message = f"For {pet_type} travel, specific requirements apply. Small {pet_type}s can travel in cabin with proper carriers."
        
        else:
            message = "I understand you need assistance. Could you please provide more details about what you'd like to help with?"
        
        # 4. Return formatted response
        return APIResponse(
            status="completed",
            message=message,
            data={
                "intent": intent.value,
                "confidence": confidence,
                "entities": [
                    {
                        "type": entity.type.value,
                        "value": entity.value,
                        "confidence": entity.confidence
                    } for entity in entities
                ],
                "alternatives": [
                    {
                        "intent": alt["type"].value,
                        "confidence": alt["confidence"]
                    } for alt in classification_result.alternative_intents
                ]
            }
        )
        
    except Exception as e:
        ErrorLogger.log_error(
            exception=e,
            error_code=ErrorCode.INTERNAL_ERROR,
            context={
                "endpoint": "simple",
                "utterance_length": len(request.utterance)
            }
        )
        
        # Try graceful degradation
        try:
            return APIResponse(**(await GracefulDegradationHandler.handle_classifier_unavailable(request.utterance)))
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=ErrorHandler.create_error_response(
                    status_code=500,
                    message=ErrorHandler.get_user_friendly_message(ErrorCode.INTERNAL_ERROR),
                    error_code=ErrorCode.INTERNAL_ERROR
                )
            )


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all HTTP requests"""
    start_time = datetime.now()
    
    # Generate request ID for tracing
    request_id = f"req_{int(start_time.timestamp() * 1000)}"
    
    # Log request
    logger.info(
        "HTTP request received",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(
        "HTTP request completed",
        request_id=request_id,
        status_code=response.status_code,
        duration_ms=int(duration * 1000)
    )
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response


# Custom 404 handler for API endpoints
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors with helpful information"""
    logger.warning("Endpoint not found", url=str(request.url), method=request.method)
    
    return JSONResponse(
        status_code=404,
        content=ErrorHandler.create_error_response(
            status_code=404,
            message=f"Endpoint not found: {request.method} {request.url.path}",
            error_code="ENDPOINT_NOT_FOUND",
            details={
                "available_endpoints": [
                    "GET /health",
                    "GET /api/v1/status", 
                    "POST /api/v1/customer-service/query",
                    "POST /api/v1/customer-service/query/simple",
                    "GET /api/v1/metrics",
                    "GET /api/v1/metrics/summary"
                ]
            }
        )
    )


def main():
    """Main entry point"""
    uvicorn.run(
        "airline_service.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.is_development,
        log_level=config.logging.level.lower(),
        access_log=config.logging.enable_audit,
    )


if __name__ == "__main__":
    main()