"""
Tests for audit logging service
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.airline_service.services.audit_logger import AuditLogger, AuditEventType
from src.airline_service.types import (
    CustomerRequest, APIResponse, RequestType, ClassificationResult,
    WorkflowResult, TaskResult, ExtractedEntity, EntityType
)


class TestAuditLogger:
    """Test audit logger functionality"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger instance"""
        return AuditLogger()
    
    @pytest.fixture
    def sample_customer_request(self):
        """Sample customer request for testing"""
        return CustomerRequest(
            utterance="I want to cancel my flight ABC123",
            session_id="test_session",
            customer_id="customer_123",
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def sample_api_response(self):
        """Sample API response for testing"""
        return APIResponse(
            status="completed",
            message="Flight cancelled successfully",
            data={"pnr": "ABC123", "refund_amount": 150.0},
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def sample_classification_result(self):
        """Sample classification result for testing"""
        entities = [
            ExtractedEntity(
                type=EntityType.PNR,
                value="ABC123",
                confidence=0.95,
                start_index=25,
                end_index=31
            )
        ]
        
        return ClassificationResult(
            request_type=RequestType.CANCEL_TRIP,
            confidence=0.92,
            alternative_intents=[],
            extracted_entities=entities
        )
    
    @pytest.fixture
    def sample_workflow_result(self):
        """Sample workflow result for testing"""
        return WorkflowResult(
            success=True,
            message="Workflow completed successfully",
            data={"result": "success"},
            executed_tasks=["task1", "task2"],
            duration=1.5
        )
    
    def test_log_customer_request(self, audit_logger, sample_customer_request):
        """Test logging customer request"""
        with patch.object(audit_logger.logger, 'info') as mock_info:
            audit_logger.log_customer_request(
                session_id="test_session",
                request=sample_customer_request,
                request_id="req_123"
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[1]
            
            assert call_args["event_type"] == AuditEventType.CUSTOMER_REQUEST
            assert call_args["session_id"] == "test_session"
            assert call_args["request_id"] == "req_123"
            assert call_args["customer_id"] == "customer_123"
            assert call_args["has_customer_id"] is True
            assert "ABC123" not in call_args["utterance_preview"]  # PNR should be masked
    
    def test_log_customer_response(self, audit_logger, sample_api_response):
        """Test logging customer response"""
        with patch.object(audit_logger.logger, 'info') as mock_info:
            audit_logger.log_customer_response(
                session_id="test_session",
                response=sample_api_response,
                request_id="req_123",
                processing_time_ms=1500
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[1]
            
            assert call_args["event_type"] == AuditEventType.CUSTOMER_RESPONSE
            assert call_args["session_id"] == "test_session"
            assert call_args["request_id"] == "req_123"
            assert call_args["response_status"] == "completed"
            assert call_args["processing_time_ms"] == 1500
            assert call_args["has_data"] is True
    
    def test_log_classification_result(self, audit_logger, sample_classification_result):
        """Test logging classification result"""
        with patch.object(audit_logger.logger, 'info') as mock_info:
            audit_logger.log_classification_result(
                session_id="test_session",
                classification=sample_classification_result,
                processing_time_ms=500,
                request_id="req_123"
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[1]
            
            assert call_args["event_type"] == AuditEventType.REQUEST_CLASSIFICATION
            assert call_args["request_type"] == "cancel_trip"
            assert call_args["confidence"] == 0.92
            assert call_args["extracted_entities_count"] == 1
            assert len(call_args["entities"]) == 1
            assert call_args["entities"][0]["type"] == "pnr"
            assert call_args["entities"][0]["has_pii"] is False  # 'pnr' is not in pii_fields by default
    
    def test_log_workflow_execution_success(self, audit_logger, sample_workflow_result):
        """Test logging successful workflow execution"""
        with patch.object(audit_logger.logger, 'info') as mock_info:
            audit_logger.log_workflow_execution(
                session_id="test_session",
                workflow_result=sample_workflow_result,
                request_type=RequestType.CANCEL_TRIP,
                processing_time_ms=2000,
                request_id="req_123"
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[1]
            
            assert call_args["event_type"] == AuditEventType.WORKFLOW_EXECUTION
            assert call_args["workflow_success"] is True
            assert call_args["request_type"] == "cancel_trip"
            assert call_args["task_count"] == 2
            assert call_args["executed_tasks"] == ["task1", "task2"]
    
    def test_log_workflow_execution_failure(self, audit_logger):
        """Test logging failed workflow execution"""
        failed_result = WorkflowResult(
            success=False,
            message="Workflow failed",
            data=None,
            executed_tasks=["task1"],
            duration=0.5
        )
        
        with patch.object(audit_logger.logger, 'info') as mock_info, \
             patch.object(audit_logger.logger, 'warning') as mock_warning:
            
            audit_logger.log_workflow_execution(
                session_id="test_session",
                workflow_result=failed_result,
                request_type=RequestType.CANCEL_TRIP,
                processing_time_ms=1000,
                request_id="req_123"
            )
            
            mock_info.assert_called_once()
            mock_warning.assert_called_once()
            
            warning_args = mock_warning.call_args[1]
            assert warning_args["event_type"] == AuditEventType.ERROR_OCCURRED
            assert warning_args["error_message"] == "Workflow failed"
    
    def test_log_api_call_success(self, audit_logger):
        """Test logging successful API call"""
        with patch.object(audit_logger.logger, 'info') as mock_info:
            audit_logger.log_api_call(
                session_id="test_session",
                endpoint="/flight/booking",
                method="GET",
                status_code=200,
                response_time_ms=300,
                request_id="req_123"
            )
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args[1]
            
            assert call_args["event_type"] == AuditEventType.API_CALL
            # Check that the first positional argument (message) contains "completed"
            assert "completed" in mock_info.call_args[0][0]
            assert call_args["endpoint"] == "/flight/booking"
            assert call_args["method"] == "GET"
            assert call_args["status_code"] == 200
            assert call_args["response_time_ms"] == 300
    
    def test_log_api_call_failure(self, audit_logger):
        """Test logging failed API call"""
        with patch.object(audit_logger.logger, 'warning') as mock_warning:
            audit_logger.log_api_call(
                session_id="test_session",
                endpoint="/flight/booking",
                method="GET",
                status_code=500,
                response_time_ms=5000,
                error="Internal server error",
                request_id="req_123"
            )
            
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[1]
            
            assert call_args["event_type"] == AuditEventType.API_CALL
            # Check that the first positional argument (message) contains "failed"
            assert "failed" in mock_warning.call_args[0][0]
            assert call_args["error"] == "Internal server error"
    
    def test_log_performance_threshold_exceeded(self, audit_logger):
        """Test logging performance threshold violations"""
        with patch.object(audit_logger.logger, 'warning') as mock_warning:
            audit_logger.log_performance_threshold_exceeded(
                session_id="test_session",
                metric_name="request_latency",
                actual_value=3000.0,
                threshold_value=2000.0,
                request_id="req_123"
            )
            
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[1]
            
            assert call_args["event_type"] == AuditEventType.PERFORMANCE_THRESHOLD_EXCEEDED
            assert call_args["metric_name"] == "request_latency"
            assert call_args["actual_value"] == 3000.0
            assert call_args["threshold_value"] == 2000.0
            assert call_args["threshold_exceeded_by"] == 1000.0
    
    def test_log_error(self, audit_logger):
        """Test logging system errors"""
        context = {"user_id": "123", "action": "cancel_flight"}
        
        with patch.object(audit_logger.logger, 'error') as mock_error:
            audit_logger.log_error(
                session_id="test_session",
                error_type="DATABASE_ERROR",
                error_message="Connection timeout",
                error_code="DB_TIMEOUT",
                context=context,
                request_id="req_123"
            )
            
            mock_error.assert_called_once()
            call_args = mock_error.call_args[1]
            
            assert call_args["event_type"] == AuditEventType.ERROR_OCCURRED
            assert call_args["error_type"] == "DATABASE_ERROR"
            assert call_args["error_message"] == "Connection timeout"
            assert call_args["error_code"] == "DB_TIMEOUT"
            assert call_args["context"] == context
    
    def test_sanitize_utterance(self, audit_logger):
        """Test PII sanitization in utterances"""
        utterance = "My PNR is ABC123 and my phone is 555-123-4567, email test@example.com"
        
        sanitized = audit_logger._sanitize_utterance(utterance)
        
        assert "ABC123" not in sanitized
        assert "555-123-4567" not in sanitized
        assert "test@example.com" not in sanitized
        assert "[PNR]" in sanitized
        assert "[PHONE]" in sanitized
        assert "[EMAIL]" in sanitized
    
    def test_sanitize_data_dict(self, audit_logger):
        """Test PII sanitization in data dictionaries"""
        data = {
            "pnr": "ABC123",
            "passenger_name": "John Doe",
            "phone": "555-123-4567",
            "flight_number": "AA100",
            "safe_data": "This is safe"
        }
        
        sanitized = audit_logger._sanitize_data(data)
        
        assert sanitized["pnr"] == "[PNR]"  # PNR gets sanitized by utterance sanitization
        assert sanitized["passenger_name"] == "[MASKED_PASSENGER_NAME]"
        assert sanitized["phone"] == "[MASKED_PHONE]"
        assert sanitized["flight_number"] == "[FLIGHT]"  # Flight numbers get sanitized by utterance sanitization
        assert sanitized["safe_data"] == "This is safe"
    
    def test_sanitize_data_nested(self, audit_logger):
        """Test PII sanitization in nested data structures"""
        data = {
            "booking": {
                "pnr": "ABC123",
                "passenger": {
                    "name": "John Doe",
                    "email": "john@example.com"
                }
            },
            "contacts": [
                {"phone": "555-123-4567"},
                {"email": "contact@example.com"}
            ]
        }
        
        sanitized = audit_logger._sanitize_data(data)
        
        assert sanitized["booking"]["pnr"] == "[PNR]"  # PNR gets sanitized by utterance sanitization
        assert sanitized["booking"]["passenger"]["email"] == "[MASKED_EMAIL]"  # Email is in PII fields
        assert sanitized["contacts"][0]["phone"] == "[MASKED_PHONE]"
        assert "[EMAIL]" in sanitized["contacts"][1]["email"]
    
    def test_disabled_logging(self):
        """Test that logging is disabled when configured"""
        with patch('src.airline_service.services.audit_logger.config') as mock_config:
            mock_config.logging.enable_audit = False
            
            audit_logger = AuditLogger()
            
            with patch.object(audit_logger.logger, 'info') as mock_info:
                audit_logger.log_customer_request(
                    session_id="test_session",
                    request=CustomerRequest(utterance="test"),
                    request_id="req_123"
                )
                
                mock_info.assert_not_called()