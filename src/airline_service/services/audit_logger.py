"""
Audit logging service for airline customer service system.

This service provides comprehensive audit logging for all customer interactions,
API calls, and system events while ensuring PII protection.
"""

import json
import structlog
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from ..types import (
    CustomerRequest, APIResponse, RequestType, ClassificationResult,
    WorkflowResult, TaskResult, ExtractedEntity
)
from ..config import config


class AuditEventType(str, Enum):
    """Types of audit events"""
    CUSTOMER_REQUEST = "customer_request"
    CUSTOMER_RESPONSE = "customer_response"
    REQUEST_CLASSIFICATION = "request_classification"
    WORKFLOW_EXECUTION = "workflow_execution"
    TASK_EXECUTION = "task_execution"
    API_CALL = "api_call"
    API_RESPONSE = "api_response"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_THRESHOLD_EXCEEDED = "performance_threshold_exceeded"


class AuditLogger:
    """
    Service for audit logging with PII protection and structured logging.
    
    Logs all customer interactions and system events for compliance and monitoring
    while ensuring sensitive data is properly handled.
    """
    
    def __init__(self):
        """Initialize the audit logger."""
        self.logger = structlog.get_logger("audit")
        self.enabled = config.logging.enable_audit
        
        # PII fields that should be masked or excluded
        self.pii_fields = {
            'phone', 'email', 'passenger_name', 'customer_name', 
            'name', 'phone_number', 'email_address'
        }
    
    def log_customer_request(
        self, 
        session_id: str, 
        request: CustomerRequest,
        request_id: Optional[str] = None
    ) -> None:
        """
        Log incoming customer request with PII protection.
        
        Args:
            session_id: Session identifier
            request: Customer request object
            request_id: Optional request identifier
        """
        if not self.enabled:
            return
        
        # Sanitize utterance to remove potential PII
        sanitized_utterance = self._sanitize_utterance(request.utterance)
        
        self.logger.info(
            "Customer request received",
            event_type=AuditEventType.CUSTOMER_REQUEST,
            session_id=session_id,
            request_id=request_id,
            utterance_length=len(request.utterance),
            utterance_preview=sanitized_utterance[:100] + "..." if len(sanitized_utterance) > 100 else sanitized_utterance,
            customer_id=request.customer_id,
            timestamp=request.timestamp.isoformat() if request.timestamp else datetime.now().isoformat(),
            has_customer_id=bool(request.customer_id)
        )
    
    def log_customer_response(
        self, 
        session_id: str, 
        response: APIResponse,
        request_id: Optional[str] = None,
        processing_time_ms: Optional[int] = None
    ) -> None:
        """
        Log customer response with sanitized data.
        
        Args:
            session_id: Session identifier
            response: API response object
            request_id: Optional request identifier
            processing_time_ms: Processing time in milliseconds
        """
        if not self.enabled:
            return
        
        # Sanitize response data
        sanitized_data = self._sanitize_data(response.data) if response.data else None
        
        self.logger.info(
            "Customer response sent",
            event_type=AuditEventType.CUSTOMER_RESPONSE,
            session_id=session_id,
            request_id=request_id,
            response_status=response.status,
            message_length=len(response.message),
            has_data=bool(response.data),
            error_code=response.error_code,
            processing_time_ms=processing_time_ms,
            timestamp=response.timestamp.isoformat()
        )
        
        # Log sanitized data separately if present
        if sanitized_data:
            self.logger.debug(
                "Response data details",
                event_type=AuditEventType.CUSTOMER_RESPONSE,
                session_id=session_id,
                request_id=request_id,
                data_keys=list(sanitized_data.keys()) if isinstance(sanitized_data, dict) else None,
                data_preview=str(sanitized_data)[:200] + "..." if len(str(sanitized_data)) > 200 else str(sanitized_data)
            )
    
    def log_classification_result(
        self, 
        session_id: str, 
        classification: ClassificationResult,
        processing_time_ms: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> None:
        """
        Log request classification results.
        
        Args:
            session_id: Session identifier
            classification: Classification result
            processing_time_ms: Classification processing time
            request_id: Optional request identifier
        """
        if not self.enabled:
            return
        
        # Sanitize extracted entities
        sanitized_entities = [
            {
                "type": entity.type.value,
                "confidence": entity.confidence,
                "start_index": entity.start_index,
                "end_index": entity.end_index,
                "value_length": len(entity.value),
                "has_pii": entity.type.value.lower() in self.pii_fields
            }
            for entity in classification.extracted_entities
        ]
        
        self.logger.info(
            "Request classified",
            event_type=AuditEventType.REQUEST_CLASSIFICATION,
            session_id=session_id,
            request_id=request_id,
            request_type=classification.request_type.value,
            confidence=classification.confidence,
            alternative_intents_count=len(classification.alternative_intents),
            extracted_entities_count=len(classification.extracted_entities),
            entities=sanitized_entities,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now().isoformat()
        )
    
    def log_workflow_execution(
        self, 
        session_id: str, 
        workflow_result: WorkflowResult,
        request_type: RequestType,
        processing_time_ms: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> None:
        """
        Log workflow execution results.
        
        Args:
            session_id: Session identifier
            workflow_result: Workflow execution result
            request_type: Type of request processed
            processing_time_ms: Workflow processing time
            request_id: Optional request identifier
        """
        if not self.enabled:
            return
        
        self.logger.info(
            "Workflow executed",
            event_type=AuditEventType.WORKFLOW_EXECUTION,
            session_id=session_id,
            request_id=request_id,
            request_type=request_type.value,
            workflow_success=workflow_result.success,
            executed_tasks=workflow_result.executed_tasks,
            task_count=len(workflow_result.executed_tasks),
            processing_time_ms=processing_time_ms,
            workflow_duration_ms=int(workflow_result.duration * 1000),
            has_data=bool(workflow_result.data),
            timestamp=datetime.now().isoformat()
        )
        
        # Log workflow failure details if applicable
        if not workflow_result.success:
            self.logger.warning(
                "Workflow execution failed",
                event_type=AuditEventType.ERROR_OCCURRED,
                session_id=session_id,
                request_id=request_id,
                request_type=request_type.value,
                error_message=workflow_result.message,
                executed_tasks=workflow_result.executed_tasks,
                timestamp=datetime.now().isoformat()
            )
    
    def log_task_execution(
        self, 
        session_id: str, 
        task_id: str,
        task_result: TaskResult,
        request_id: Optional[str] = None
    ) -> None:
        """
        Log individual task execution.
        
        Args:
            session_id: Session identifier
            task_id: Task identifier
            task_result: Task execution result
            request_id: Optional request identifier
        """
        if not self.enabled:
            return
        
        self.logger.debug(
            "Task executed",
            event_type=AuditEventType.TASK_EXECUTION,
            session_id=session_id,
            request_id=request_id,
            task_id=task_id,
            task_success=task_result.success,
            task_duration_ms=int(task_result.duration * 1000),
            has_data=bool(task_result.data),
            error=task_result.error,
            timestamp=datetime.now().isoformat()
        )
    
    def log_api_call(
        self, 
        session_id: str,
        endpoint: str,
        method: str,
        status_code: Optional[int] = None,
        response_time_ms: Optional[int] = None,
        error: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> None:
        """
        Log API call details.
        
        Args:
            session_id: Session identifier
            endpoint: API endpoint called
            method: HTTP method
            status_code: Response status code
            response_time_ms: API response time
            error: Error message if call failed
            request_id: Optional request identifier
        """
        if not self.enabled:
            return
        
        if error:
            self.logger.warning(
                "API call failed",
                event_type=AuditEventType.API_CALL,
                session_id=session_id,
                request_id=request_id,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                error=error,
                timestamp=datetime.now().isoformat()
            )
        else:
            self.logger.info(
                "API call completed",
                event_type=AuditEventType.API_CALL,
                session_id=session_id,
                request_id=request_id,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                timestamp=datetime.now().isoformat()
            )
    
    def log_performance_threshold_exceeded(
        self, 
        session_id: str,
        metric_name: str,
        actual_value: float,
        threshold_value: float,
        request_id: Optional[str] = None
    ) -> None:
        """
        Log when performance thresholds are exceeded.
        
        Args:
            session_id: Session identifier
            metric_name: Name of the performance metric
            actual_value: Actual measured value
            threshold_value: Configured threshold value
            request_id: Optional request identifier
        """
        if not self.enabled:
            return
        
        self.logger.warning(
            "Performance threshold exceeded",
            event_type=AuditEventType.PERFORMANCE_THRESHOLD_EXCEEDED,
            session_id=session_id,
            request_id=request_id,
            metric_name=metric_name,
            actual_value=actual_value,
            threshold_value=threshold_value,
            threshold_exceeded_by=actual_value - threshold_value,
            timestamp=datetime.now().isoformat()
        )
    
    def log_performance_alert(
        self, 
        session_id: str,
        alert_type: str,
        severity: str,
        message: str,
        actual_value: float,
        threshold_value: float,
        request_id: Optional[str] = None
    ) -> None:
        """
        Log performance alerts with severity levels.
        
        Args:
            session_id: Session identifier
            alert_type: Type of performance alert
            severity: Alert severity level
            message: Alert message
            actual_value: Actual measured value
            threshold_value: Configured threshold value
            request_id: Optional request identifier
        """
        if not self.enabled:
            return
        
        log_level = "critical" if severity in ["critical", "emergency"] else "warning"
        
        log_func = self.logger.critical if log_level == "critical" else self.logger.warning
        
        log_func(
            "Performance alert triggered",
            event_type=AuditEventType.PERFORMANCE_THRESHOLD_EXCEEDED,
            session_id=session_id,
            request_id=request_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            actual_value=actual_value,
            threshold_value=threshold_value,
            threshold_ratio=actual_value / threshold_value if threshold_value > 0 else 0,
            timestamp=datetime.now().isoformat()
        )
    
    def log_error(
        self, 
        session_id: str,
        error_type: str,
        error_message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> None:
        """
        Log system errors with context.
        
        Args:
            session_id: Session identifier
            error_type: Type of error
            error_message: Error message
            error_code: Optional error code
            context: Additional context information
            request_id: Optional request identifier
        """
        if not self.enabled:
            return
        
        # Sanitize context data
        sanitized_context = self._sanitize_data(context) if context else None
        
        self.logger.error(
            "System error occurred",
            event_type=AuditEventType.ERROR_OCCURRED,
            session_id=session_id,
            request_id=request_id,
            error_type=error_type,
            error_message=error_message,
            error_code=error_code,
            context=sanitized_context,
            timestamp=datetime.now().isoformat()
        )
    
    def _sanitize_utterance(self, utterance: str) -> str:
        """
        Sanitize customer utterance to remove potential PII.
        
        Args:
            utterance: Original utterance
            
        Returns:
            Sanitized utterance with PII patterns masked
        """
        import re
        
        # Mask potential PNRs (6 alphanumeric characters)
        utterance = re.sub(r'\b[A-Z0-9]{6}\b', '[PNR]', utterance, flags=re.IGNORECASE)
        
        # Mask potential phone numbers
        utterance = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', utterance)
        
        # Mask potential email addresses
        utterance = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', utterance)
        
        # Mask potential flight numbers (2-3 letters followed by 1-4 digits)
        utterance = re.sub(r'\b[A-Z]{2,3}\d{1,4}\b', '[FLIGHT]', utterance, flags=re.IGNORECASE)
        
        return utterance
    
    def _sanitize_data(self, data: Any) -> Any:
        """
        Recursively sanitize data to remove or mask PII.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data with PII removed or masked
        """
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if key.lower() in self.pii_fields:
                    # Mask PII fields
                    if isinstance(value, str) and value:
                        sanitized[key] = f"[MASKED_{key.upper()}]"
                    else:
                        sanitized[key] = "[MASKED]"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        
        elif isinstance(data, str):
            # Apply utterance sanitization to string values
            return self._sanitize_utterance(data)
        
        else:
            return data


# Global audit logger instance
audit_logger = AuditLogger()