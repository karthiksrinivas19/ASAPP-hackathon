"""
Tests for main application
"""

import pytest
from fastapi.testclient import TestClient
from airline_service.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    assert "environment" in data


def test_customer_query_endpoint():
    """Test customer query endpoint"""
    request_data = {
        "utterance": "I want to cancel my flight",
        "customer_id": "test-customer"
    }
    
    response = client.post("/api/v1/customer-service/query", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "completed"
    assert "message" in data
    assert "timestamp" in data


def test_invalid_endpoint():
    """Test invalid endpoint returns 404"""
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404


def test_customer_query_validation():
    """Test customer query validation"""
    # Test missing utterance
    response = client.post("/api/v1/customer-service/query", json={})
    assert response.status_code == 422  # Validation error
    
    # Test invalid data type
    response = client.post("/api/v1/customer-service/query", json={"utterance": 123})
    assert response.status_code == 422