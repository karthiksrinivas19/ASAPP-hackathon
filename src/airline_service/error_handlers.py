"""
Comprehensive error handling system for airline service
"""

import traceback
from typing import Dict, Any, Optional, Union
from datetime import datetime
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import structlog

from .types import APIError, MissingDataError, ClassificationError, AirlineServiceError
from .services import audit_logger, metrics_collector

logger = structlog.get_logger()


class ErrorCode:
    """Standard error codes for the airline service"""
    
    # Client errors (4xx)
    INVALID_REQUEST = "INVALID_REQUEST"
    UTTERANCE_TOO_LONG = "UTTERANCE_TOO_LONG"
    UTTERANCE_EMPTY = "UTTERANCE_EMPTY"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    BOOKING_NOT_FOUND = "BOOKING_NOT_FOUND"
    FLIGHT_NOT_FOUND = "FLIGHT_NOT_FOUND"
    INVALID_PNR = "INVALID_PNR"
    INVALID_FLIGHT_NUMBER = "INVALID_FLIGHT_NUMBER"
    MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"
    
    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    API_UNAVAILABLE = "API_UNAVAILABLE"
    CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"
    WORKFLOW_FAILED = "WORKFLOW_FAILED"
    DATABASE_ERROR = "DATABASE_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    
    # Service-specific errors
    ML_MODEL_ERROR = "ML_MODEL_ERROR"
    POLICY_LOOKUP_FAILED = "POLICY_LOOKUP_FAILED"
    AIRLINE_API_ERROR = "AIRLINE_API_ERROR"
    CACHE_ERROR = "CACHE_ERROR"


class ErrorSeverity:
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorHandler:
    """Centralized error handling and response formatting"""
    
    @staticmethod
    def create_error_response(
        status_code: int,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        
        error_response = {
            "status": "error",
            "message": message,
            "error_code": error_code,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            error_response["details"] = details
        
        if session_id:
            error_response["session_id"] = session_id
            
        if request_id:
            error_response["request_id"] = request_id
        
        return error_response
    
    @staticmethod
    def get_user_friendly_message(error_code: str, original_message: str = None) -> str:
        """Get user-friendly error messages"""
        
        user_messages = {
            ErrorCode.BOOKING_NOT_FOUND: "We couldn't find your booking. Please check your booking reference (PNR) and try again.",
            ErrorCode.FLIGHT_NOT_FOUND: "We couldn't find your flight. Please verify your flight number and date.",
            ErrorCode.INVALID_PNR: "The booking reference you provided appears to be invalid. Please check and try again.",
            ErrorCode.INVALID_FLIGHT_NUMBER: "The flight number you provided appears to be invalid. Please check and try again.",
            ErrorCode.MISSING_REQUIRED_DATA: "We need more information to help you. Please provide your booking reference (PNR) or flight details.",
            ErrorCode.API_UNAVAILABLE: "Our booking system is temporarily unavailable. Please try again in a few minutes.",
            ErrorCode.SERVICE_UNAVAILABLE: "Our service is temporarily unavailable. Please try again later.",
            ErrorCode.TIMEOUT_ERROR: "The request took too long to process. Please try again.",
            ErrorCode.CLASSIFICATION_FAILED: "We're having trouble understanding your request. Could you please rephrase it?",
            ErrorCode.WORKFLOW_FAILED: "We encountered an issue processing your request. Please try again or contact support.",
            ErrorCode.VALIDATION_ERROR: "There's an issue with the information provided. Please check and try again.",
            ErrorCode.INTERNAL_ERROR: "We're experiencing technical difficulties. Please try again later."
        }
        
        return user_messages.get(error_code, original_message or "An unexpected error occurred. Please try again.")
    
    @staticmethod
    def determine_error_severity(error_code: str, exception: Exception = None) -> str:
        """Determine error severity for monitoring and alerting"""
        
        critical_errors = {
            ErrorCode.SERVICE_UNAVAILABLE,
            ErrorCode.DATABASE_ERROR,
            ErrorCode.INTERNAL_ERROR
        }
        
        high_errors = {
            ErrorCode.API_UNAVAILABLE,
            ErrorCode.WORKFLOW_FAILED,
            ErrorCode.ML_MODEL_ERROR
        }
        
        medium_errors = {
            ErrorCode.CLASSIFICATION_FAILED,
            ErrorCode.POLICY_LOOKUP_FAILED,
            ErrorCode.TIMEOUT_ERROR
        }
        
        if error_code in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_code in high_errors:
            return ErrorSeverity.HIGH
        elif error_code in medium_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    @staticmethod
    def should_retry(error_code: str) -> bool:
        """Determine if the operation should be retried"""
        
        retryable_errors = {
            ErrorCode.TIMEOUT_ERROR,
            ErrorCode.API_UNAVAILABLE,
            ErrorCode.CACHE_ERROR
        }
        
        return error_code in retryable_errors


class GracefulDegradationHandler:
    """Handle graceful degradation when services are unavailable"""
    
    @staticmethod
    async def handle_classifier_unavailable(utterance: str) -> Dict[str, Any]:
        """Handle when ML classifier is unavailable"""
        logger.warning("Classifier unavailable, using keyword-based fallback")
        
        # Simple keyword-based classification as fallback
        utterance_lower = utterance.lower()
        
        if any(word in utterance_lower for word in ["cancel", "refund", "cancellation"]):
            request_type = "cancel_trip"
            message = "I can help you with cancellation. To proceed, I'll need your booking reference (PNR)."
        elif any(word in utterance_lower for word in ["status", "delayed", "on time", "departure"]):
            request_type = "flight_status"
            message = "I can check your flight status. Please provide your booking reference (PNR) or flight number."
        elif any(word in utterance_lower for word in ["seat", "upgrade", "available"]):
            request_type = "seat_availability"
            message = "I can help you check seat availability. Please provide your booking reference (PNR)."
        elif any(word in utterance_lower for word in ["policy", "rules", "terms"]):
            request_type = "cancellation_policy"
            message = "I can provide information about our policies. What specific policy would you like to know about?"
        elif any(word in utterance_lower for word in ["pet", "animal", "dog", "cat"]):
            request_type = "pet_travel"
            message = "I can help with pet travel information. What would you like to know about traveling with pets?"
        else:
            request_type = "unknown"
            message = "I'd be happy to help! Could you please tell me what you need assistance with? For example: flight status, cancellation, seat selection, or policy information."
        
        return {
            "status": "completed",
            "message": message,
            "data": {
                "request_type": request_type,
                "confidence": 0.5,
                "fallback_mode": True,
                "note": "Using simplified processing due to service limitations"
            }
        }
    
    @staticmethod
    async def handle_api_unavailable(request_type: str) -> Dict[str, Any]:
        """Handle when airline API is unavailable"""
        logger.warning("Airline API unavailable, providing cached/general information")
        
        fallback_messages = {
            "cancel_trip": "Our booking system is temporarily unavailable. For immediate cancellation assistance, please call our customer service at 1-800-JETBLUE or visit jetblue.com.",
            "flight_status": "Flight status information is temporarily unavailable. Please check jetblue.com or the JetBlue app for the latest updates.",
            "seat_availability": "Seat selection is temporarily unavailable. You can select seats during online check-in or at the airport.",
            "cancellation_policy": "Our standard cancellation policy allows changes and cancellations up to 24 hours before departure. Fees may apply based on fare type.",
            "pet_travel": "Small pets can travel in the cabin for $125 each way. Pets must be in an approved carrier that fits under the seat. Service animals fly free."
        }
        
        message = fallback_messages.get(request_type, "Our systems are temporarily unavailable. Please try again later or contact customer service.")
        
        return {
            "status": "completed",
            "message": message,
            "data": {
                "request_type": request_type,
                "fallback_mode": True,
                "note": "Limited functionality due to system unavailability",
                "alternative_channels": {
                    "phone": "1-800-JETBLUE",
                    "website": "jetblue.com",
                    "app": "JetBlue mobile app"
                }
            }
        }
    
    @staticmethod
    async def handle_policy_unavailable() -> Dict[str, Any]:
        """Handle when policy service is unavailable"""
        logger.warning("Policy service unavailable, providing general information")
        
        return {
            "status": "completed",
            "message": "Policy information is temporarily unavailable. For the most current policies, please visit jetblue.com or contact customer service.",
            "data": {
                "fallback_mode": True,
                "general_info": {
                    "cancellation": "Generally, you can cancel flights up to 24 hours before departure",
                    "changes": "Flight changes are typically allowed with applicable fees",
                    "refunds": "Refund eligibility depends on fare type and timing"
                },
                "contact_info": {
                    "website": "jetblue.com/flying-with-us/our-fares",
                    "phone": "1-800-JETBLUE"
                }
            }
        }


class ExceptionMapper:
    """Map exceptions to appropriate HTTP responses"""
    
    @staticmethod
    def map_exception(exception: Exception, session_id: str = None, request_id: str = None) -> HTTPException:
        """Map various exceptions to appropriate HTTP exceptions"""
        
        if isinstance(exception, HTTPException):
            return exception
        
        elif isinstance(exception, MissingDataError):
            return HTTPException(
                status_code=400,
                detail=ErrorHandler.create_error_response(
                    status_code=400,
                    message=ErrorHandler.get_user_friendly_message(ErrorCode.MISSING_REQUIRED_DATA),
                    error_code=ErrorCode.MISSING_REQUIRED_DATA,
                    details={"missing_data_type": exception.data_type},
                    session_id=session_id,
                    request_id=request_id
                )
            )
        
        elif isinstance(exception, ClassificationError):
            return HTTPException(
                status_code=500,
                detail=ErrorHandler.create_error_response(
                    status_code=500,
                    message=ErrorHandler.get_user_friendly_message(ErrorCode.CLASSIFICATION_FAILED),
                    error_code=ErrorCode.CLASSIFICATION_FAILED,
                    session_id=session_id,
                    request_id=request_id
                )
            )
        
        elif isinstance(exception, APIError):
            if exception.status_code == 404:
                error_code = ErrorCode.RESOURCE_NOT_FOUND
                message = ErrorHandler.get_user_friendly_message(ErrorCode.BOOKING_NOT_FOUND)
            elif exception.status_code == 400:
                error_code = ErrorCode.INVALID_REQUEST
                message = ErrorHandler.get_user_friendly_message(ErrorCode.INVALID_REQUEST)
            else:
                error_code = ErrorCode.API_UNAVAILABLE
                message = ErrorHandler.get_user_friendly_message(ErrorCode.API_UNAVAILABLE)
            
            return HTTPException(
                status_code=exception.status_code,
                detail=ErrorHandler.create_error_response(
                    status_code=exception.status_code,
                    message=message,
                    error_code=error_code,
                    session_id=session_id,
                    request_id=request_id
                )
            )
        
        elif isinstance(exception, TimeoutError):
            return HTTPException(
                status_code=504,
                detail=ErrorHandler.create_error_response(
                    status_code=504,
                    message=ErrorHandler.get_user_friendly_message(ErrorCode.TIMEOUT_ERROR),
                    error_code=ErrorCode.TIMEOUT_ERROR,
                    session_id=session_id,
                    request_id=request_id
                )
            )
        
        else:
            # Generic internal server error
            return HTTPException(
                status_code=500,
                detail=ErrorHandler.create_error_response(
                    status_code=500,
                    message=ErrorHandler.get_user_friendly_message(ErrorCode.INTERNAL_ERROR),
                    error_code=ErrorCode.INTERNAL_ERROR,
                    details={"exception_type": type(exception).__name__},
                    session_id=session_id,
                    request_id=request_id
                )
            )


class ErrorLogger:
    """Enhanced error logging with context and metrics"""
    
    @staticmethod
    def log_error(
        exception: Exception,
        error_code: str,
        session_id: str = None,
        request_id: str = None,
        context: Dict[str, Any] = None,
        include_traceback: bool = True
    ):
        """Log error with full context and metrics"""
        
        severity = ErrorHandler.determine_error_severity(error_code, exception)
        
        log_data = {
            "error_code": error_code,
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "severity": severity,
            "session_id": session_id,
            "request_id": request_id
        }
        
        if context:
            log_data["context"] = context
        
        if include_traceback:
            log_data["traceback"] = traceback.format_exc()
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", **log_data)
        elif severity == ErrorSeverity.HIGH:
            logger.error("High severity error occurred", **log_data)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error occurred", **log_data)
        else:
            logger.info("Low severity error occurred", **log_data)
        
        # Log to audit system
        if session_id:
            audit_logger.log_error(
                session_id=session_id,
                error_type=error_code,
                error_message=str(exception),
                error_code=error_code,
                request_id=request_id,
                context=context
            )
        
        # Record error metrics
        metrics_collector.record_error_metrics(
            error_code=error_code,
            severity=severity,
            session_id=session_id
        )


# Global error handlers for FastAPI
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions"""
    
    request_id = getattr(request.state, 'request_id', None)
    session_id = getattr(request.state, 'session_id', None)
    
    # Log the error
    ErrorLogger.log_error(
        exception=exc,
        error_code=ErrorCode.INTERNAL_ERROR,
        session_id=session_id,
        request_id=request_id,
        context={
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers)
        }
    )
    
    # Map to appropriate HTTP exception
    http_exception = ExceptionMapper.map_exception(exc, session_id, request_id)
    
    return JSONResponse(
        status_code=http_exception.status_code,
        content=http_exception.detail
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler for HTTP exceptions"""
    
    request_id = getattr(request.state, 'request_id', None)
    
    # Add request ID to response if available
    if request_id and isinstance(exc.detail, dict):
        exc.detail["request_id"] = request_id
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )


async def validation_exception_handler(request: Request, exc) -> JSONResponse:
    """Handler for request validation exceptions"""
    
    request_id = getattr(request.state, 'request_id', None)
    
    logger.warning(
        "Request validation failed",
        error=str(exc),
        url=str(request.url),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=422,
        content=ErrorHandler.create_error_response(
            status_code=422,
            message="Request validation failed. Please check your input and try again.",
            error_code=ErrorCode.VALIDATION_ERROR,
            details=exc.errors() if hasattr(exc, 'errors') else str(exc),
            request_id=request_id
        )
    )