"""
Tests for the main customer service query endpoint
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime

from src.airline_service.main import app
from src.airline_service.types import (
    CustomerRequest, APIResponse, RequestType, ClassificationResult, 
    ExtractedEntity, EntityType, WorkflowResult
)


class TestMainEndpoint:
    """Test the main customer service query endpoint"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
    
    def test_status_endpoint(self):
        """Test service status endpoint"""
        response = self.client.get("/api/v1/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "capabilities" in data
        assert "workflows" in data
        assert data["service"]["name"] == "Airline Customer Service API"
    
    def test_query_endpoint_validation_empty_utterance(self):
        """Test query endpoint with empty utterance"""
        response = self.client.post(
            "/api/v1/customer-service/query",
            json={"utterance": ""}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "INVALID_REQUEST"
        assert "empty" in data["detail"]["message"].lower()
    
    def test_query_endpoint_validation_long_utterance(self):
        """Test query endpoint with too long utterance"""
        long_utterance = "x" * 1001  # Over the 1000 character limit
        
        response = self.client.post(
            "/api/v1/customer-service/query",
            json={"utterance": long_utterance}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "UTTERANCE_TOO_LONG"
    
    @patch('src.airline_service.services.request_classifier_service.ClassifierFactory.create_classifier')
    @patch('src.airline_service.main.workflow_orchestrator.execute_workflow')
    def test_query_endpoint_successful_processing(self, mock_workflow, mock_classifier_factory):
        """Test successful query processing"""
        # Mock classifier
        mock_classifier = MagicMock()
        mock_classifier.is_loaded.return_value = True
        mock_classifier.classify_request = AsyncMock(return_value=ClassificationResult(
            request_type=RequestType.CANCEL_TRIP,
            confidence=0.95,
            alternative_intents=[],
            extracted_entities=[
                ExtractedEntity(
                    type=EntityType.PNR,
                    value="ABC123",
                    confidence=0.9,
                    start_index=0,
                    end_index=6
                )
            ]
        ))
        mock_classifier_factory.return_value = mock_classifier
        
        # Mock workflow result
        mock_workflow.return_value = WorkflowResult(
            success=True,
            message="Flight cancelled successfully",
            data={
                "response": APIResponse(
                    status="completed",
                    message="Your flight has been cancelled successfully",
                    data={"pnr": "ABC123", "refund_amount": 150.0}
                )
            },
            executed_tasks=["extract_identifiers", "get_booking_details", "cancel_flight", "inform_cancellation_result"],
            duration=1.5
        )
        
        # Make request
        response = self.client.post(
            "/api/v1/customer-service/query",
            json={
                "utterance": "Cancel my flight ABC123",
                "session_id": "test_session"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "ABC123" in data["message"]
        assert data["data"]["pnr"] == "ABC123"
    
    @patch('src.airline_service.services.request_classifier_service.ClassifierFactory.create_classifier')
    def test_query_endpoint_low_confidence_classification(self, mock_classifier_factory):
        """Test query with low classification confidence"""
        # Mock classifier with low confidence
        mock_classifier = MagicMock()
        mock_classifier.is_loaded.return_value = True
        mock_classifier.classify_request = AsyncMock(return_value=ClassificationResult(
            request_type=RequestType.UNKNOWN,
            confidence=0.2,  # Low confidence
            alternative_intents=[],
            extracted_entities=[]
        ))
        mock_classifier_factory.return_value = mock_classifier
        
        response = self.client.post(
            "/api/v1/customer-service/query",
            json={"utterance": "hello there"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "not sure" in data["message"].lower()
        assert "suggestions" in data["data"]
    
    @patch('src.airline_service.services.request_classifier_service.ClassifierFactory.create_classifier')
    @patch('src.airline_service.main.workflow_orchestrator.execute_workflow')
    def test_query_endpoint_workflow_failure(self, mock_workflow, mock_classifier_factory):
        """Test query processing with workflow failure"""
        # Mock classifier
        mock_classifier = MagicMock()
        mock_classifier.is_loaded.return_value = True
        mock_classifier.classify_request = AsyncMock(return_value=ClassificationResult(
            request_type=RequestType.CANCEL_TRIP,
            confidence=0.95,
            alternative_intents=[],
            extracted_entities=[]
        ))
        mock_classifier_factory.return_value = mock_classifier
        
        # Mock workflow failure
        mock_workflow.return_value = WorkflowResult(
            success=False,
            message="Booking not found for PNR: XYZ999",
            data=None,
            executed_tasks=["extract_identifiers", "get_booking_details"],
            duration=0.5
        )
        
        response = self.client.post(
            "/api/v1/customer-service/query",
            json={"utterance": "Cancel flight XYZ999"}
        )
        
        assert response.status_code == 404  # Should map to 404 for "not found"
        data = response.json()
        assert data["detail"]["error_code"] == "RESOURCE_NOT_FOUND"
        assert "not found" in data["detail"]["message"].lower()
    
    @patch('src.airline_service.services.request_classifier_service.ClassifierFactory.create_classifier')
    def test_query_endpoint_classification_failure(self, mock_classifier_factory):
        """Test query processing with classification failure"""
        # Mock classifier that raises exception
        mock_classifier = MagicMock()
        mock_classifier.is_loaded.return_value = True
        mock_classifier.classify_request = AsyncMock(side_effect=Exception("Classification failed"))
        mock_classifier_factory.return_value = mock_classifier
        
        response = self.client.post(
            "/api/v1/customer-service/query",
            json={"utterance": "Cancel my flight"}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert data["detail"]["error_code"] == "CLASSIFICATION_FAILED"
    
    def test_query_endpoint_invalid_json(self):
        """Test query endpoint with invalid JSON"""
        response = self.client.post(
            "/api/v1/customer-service/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_endpoint_missing_utterance(self):
        """Test query endpoint with missing utterance field"""
        response = self.client.post(
            "/api/v1/customer-service/query",
            json={"session_id": "test"}  # Missing utterance
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_simple_query_endpoint(self):
        """Test the simple query endpoint (legacy)"""
        response = self.client.post(
            "/api/v1/customer-service/query/simple",
            json={"utterance": "Cancel my flight"}
        )
        
        # Should work with mock classifier
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "data" in data
        assert "intent" in data["data"]
    
    def test_404_handler(self):
        """Test 404 error handler"""
        response = self.client.get("/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert data["error_code"] == "ENDPOINT_NOT_FOUND"
        assert "available_endpoints" in data
    
    def test_request_logging_middleware(self):
        """Test that request logging middleware adds request ID"""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"].startswith("req_")


class TestEndpointIntegration:
    """Integration tests for the endpoint with real components"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    def test_full_integration_cancel_request(self):
        """Test full integration with a cancellation request"""
        response = self.client.post(
            "/api/v1/customer-service/query",
            json={
                "utterance": "I want to cancel my booking ABC123",
                "session_id": "integration_test"
            }
        )
        
        # Should work with mock components
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["completed", "error"]
        
        # Should have session_id in response
        if "data" in data and data["data"]:
            assert data["data"].get("session_id") == "integration_test"
    
    def test_full_integration_flight_status_request(self):
        """Test full integration with a flight status request"""
        response = self.client.post(
            "/api/v1/customer-service/query",
            json={
                "utterance": "What's the status of flight AA100?",
                "customer_id": "test_customer"
            }
        )
        
        # Should work with mock components
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["completed", "error"]
    
    def test_full_integration_policy_request(self):
        """Test full integration with a policy request"""
        response = self.client.post(
            "/api/v1/customer-service/query",
            json={
                "utterance": "What is your cancellation policy?"
            }
        )
        
        # Should work with mock components
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["completed", "error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])