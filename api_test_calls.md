# 🚀 Complete API Test Calls for Airline Customer Service System

This document contains all the API endpoints and test calls you can use to test the airline customer service system.

## 📋 Prerequisites

1. **Start the server first:**
   ```bash
   python run_server.py
   ```

2. **Server will be running on:** `http://localhost:8000`

---

## 🔍 **1. Health & Status Endpoints**

### **1.1 Basic Health Check**
```bash
curl -X GET http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-22T21:46:43.766955",
  "version": "1.0.0",
  "environment": "development",
  "components": {
    "classifier": {"status": "healthy", "model_info": {...}},
    "airline_api": {"status": "healthy", "type": "mock"},
    "policy_service": {"status": "healthy"},
    "workflow_orchestrator": {"status": "healthy", "registered_workflows": 5}
  },
  "performance": {
    "availability": "100.00%",
    "avg_response_time_ms": "0",
    "error_rate": "0.00%",
    "requests_per_minute": 0
  }
}
```

### **1.2 Detailed Service Status**
```bash
curl -X GET http://localhost:8000/api/v1/status
```

**Expected Response:**
```json
{
  "service": {
    "name": "Airline Customer Service API",
    "version": "1.0.0",
    "environment": "development",
    "container_initialized": true
  },
  "capabilities": {
    "supported_request_types": ["cancel_trip", "flight_status", "seat_availability", "cancellation_policy", "pet_travel"],
    "ml_classification": true,
    "workflow_orchestration": true,
    "automatic_data_retrieval": true,
    "policy_lookup": true
  },
  "classifier": {
    "status": "loaded",
    "type": "simple",
    "version": "1.0.0"
  },
  "workflows": {
    "cancel_trip": {"tasks": 4, "task_types": ["api_call", "inform_customer", "get_customer_info"]},
    "flight_status": {"tasks": 3, "task_types": ["api_call", "inform_customer", "get_customer_info"]},
    "seat_availability": {"tasks": 4, "task_types": ["api_call", "inform_customer", "get_customer_info"]},
    "cancellation_policy": {"tasks": 3, "task_types": ["policy_lookup", "inform_customer", "get_customer_info"]},
    "pet_travel": {"tasks": 3, "task_types": ["policy_lookup", "inform_customer", "get_customer_info"]}
  },
  "services": {...},
  "configuration": {
    "debug_mode": false,
    "log_level": "INFO",
    "max_request_size": "1000 characters"
  }
}
```

---

## 📊 **2. Metrics & Monitoring Endpoints**

### **2.1 Detailed Metrics**
```bash
curl -X GET http://localhost:8000/api/v1/metrics
```

### **2.2 Metrics Summary**
```bash
curl -X GET http://localhost:8000/api/v1/metrics/summary
```

**Expected Response:**
```json
{
  "timestamp": "2025-10-22T21:47:54.743903",
  "system_health": {
    "status": "healthy",
    "availability": 1.0,
    "avg_response_time_ms": 0.4,
    "error_rate": 0.0,
    "requests_per_minute": 5
  },
  "latency_metrics": {
    "request_latency": {
      "avg_ms": "150",
      "p95_ms": "300",
      "p99_ms": "500",
      "count": 10,
      "threshold_violations": 0
    },
    "classification_latency": {
      "avg_ms": "50",
      "p95_ms": "100",
      "p99_ms": "200",
      "count": 10,
      "threshold_violations": 0
    }
  }
}
```

---

## 🎯 **3. Main Customer Service Endpoints**

### **3.1 Flight Cancellation Requests**

#### **Basic Cancellation Request**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "I want to cancel my flight",
    "session_id": "test-cancel-1"
  }'
```

#### **Cancellation with PNR**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "Please cancel my booking ABC123",
    "session_id": "test-cancel-2",
    "customer_id": "customer-001"
  }'
```

#### **Urgent Cancellation**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "I need to cancel my flight immediately, booking reference XYZ789",
    "session_id": "test-cancel-3"
  }'
```

### **3.2 Flight Status Inquiries**

#### **Status by Flight Number**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "What is the status of flight JB1234?",
    "session_id": "test-status-1"
  }'
```

#### **Status by PNR**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "Check my flight status for booking ABC123",
    "session_id": "test-status-2"
  }'
```

#### **General Status Check**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "Is my flight on time?",
    "session_id": "test-status-3"
  }'
```

### **3.3 Seat Availability & Selection**

#### **General Seat Availability**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "Can I see available seats on my flight?",
    "session_id": "test-seats-1"
  }'
```

#### **Business Class Seats**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "Show me available business class seats",
    "session_id": "test-seats-2"
  }'
```

#### **Window Seat Request**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "I want a window seat, booking ABC123",
    "session_id": "test-seats-3"
  }'
```

### **3.4 Policy Information**

#### **Cancellation Policy**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "What is your cancellation policy?",
    "session_id": "test-policy-1"
  }'
```

#### **Refund Policy**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "Can I get a refund if I cancel my flight?",
    "session_id": "test-policy-2"
  }'
```

#### **Change Fee Policy**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "How much does it cost to change my flight?",
    "session_id": "test-policy-3"
  }'
```

### **3.5 Pet Travel**

#### **General Pet Travel**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "Can I travel with my dog on the plane?",
    "session_id": "test-pet-1"
  }'
```

#### **Pet Carrier Requirements**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "What are the requirements for pet carriers?",
    "session_id": "test-pet-2"
  }'
```

#### **Service Animal**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "I have a service animal, what do I need to do?",
    "session_id": "test-pet-3"
  }'
```

---

## 🔧 **4. Simple Endpoint (Alternative Processing)**

### **4.1 Simple Query Processing**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query/simple \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "I want to cancel my flight booking ABC123",
    "session_id": "test-simple-1"
  }'
```

---

## ❌ **5. Error Handling Tests**

### **5.1 Empty Request**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "",
    "session_id": "test-error-1"
  }'
```

### **5.2 Too Long Request**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "'$(python -c "print('a' * 1001)")'",
    "session_id": "test-error-2"
  }'
```

### **5.3 Invalid JSON**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{"utterance": "test", "session_id": "test-error-3"'
```

### **5.4 Missing Required Fields**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-error-4"
  }'
```

### **5.5 Invalid Endpoint**
```bash
curl -X GET http://localhost:8000/api/v1/nonexistent
```

---

## 🧪 **6. Advanced Test Scenarios**

### **6.1 Complex Multi-Entity Request**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "I need to cancel flight JB1234 on December 25th, my booking is ABC123 and my name is John Smith",
    "session_id": "test-complex-1",
    "customer_id": "john.smith@email.com"
  }'
```

### **6.2 Ambiguous Request**
```bash
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "I have a problem with my flight",
    "session_id": "test-ambiguous-1"
  }'
```

### **6.3 Multiple Requests in Session**
```bash
# First request
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "What is the status of my flight?",
    "session_id": "test-session-1"
  }'

# Follow-up request (same session)
curl -X POST http://localhost:8000/api/v1/customer-service/query \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "My booking reference is ABC123",
    "session_id": "test-session-1"
  }'
```

---

## 📱 **7. Load Testing**

### **7.1 Concurrent Requests**
```bash
# Run multiple requests simultaneously
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/v1/customer-service/query \
    -H "Content-Type: application/json" \
    -d "{\"utterance\": \"Test request $i\", \"session_id\": \"load-test-$i\"}" &
done
wait
```

### **7.2 Rapid Sequential Requests**
```bash
for i in {1..20}; do
  curl -X POST http://localhost:8000/api/v1/customer-service/query \
    -H "Content-Type: application/json" \
    -d "{\"utterance\": \"Sequential test $i\", \"session_id\": \"seq-test-$i\"}"
  sleep 0.1
done
```

---

## 🎯 **8. Expected Response Formats**

### **Successful Response Format:**
```json
{
  "status": "completed",
  "message": "Human-readable response message",
  "data": {
    "request_type": "cancel_trip|flight_status|seat_availability|cancellation_policy|pet_travel",
    "confidence": 0.95,
    "session_id": "session-id",
    "processing_time_ms": 150,
    "executed_tasks": ["get_customer_info", "api_call", "inform_customer"]
  },
  "error_code": null,
  "timestamp": "2025-10-22T21:47:04.920606"
}
```

### **Error Response Format:**
```json
{
  "status": "error",
  "message": "User-friendly error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-10-22T21:48:04.564355",
  "session_id": "session-id",
  "request_id": "req_1761149884561"
}
```

---

## 🚀 **9. Quick Test Script**

Save this as `quick_test.sh` and run with `bash quick_test.sh`:

```bash
#!/bin/bash

echo "🧪 Testing Airline Customer Service API"
echo "======================================"

BASE_URL="http://localhost:8000"

echo "1. Health Check..."
curl -s "$BASE_URL/health" | jq '.status'

echo -e "\n2. Service Status..."
curl -s "$BASE_URL/api/v1/status" | jq '.service.name'

echo -e "\n3. Flight Cancellation..."
curl -s -X POST "$BASE_URL/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{"utterance": "Cancel my flight ABC123", "session_id": "quick-test-1"}' | jq '.message'

echo -e "\n4. Flight Status..."
curl -s -X POST "$BASE_URL/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{"utterance": "Status of flight JB1234", "session_id": "quick-test-2"}' | jq '.message'

echo -e "\n5. Metrics Summary..."
curl -s "$BASE_URL/api/v1/metrics/summary" | jq '.system_health.status'

echo -e "\n✅ All tests completed!"
```

---

## 📝 **Notes**

1. **Server must be running** on `http://localhost:8000`
2. **All endpoints return JSON** responses
3. **Session IDs** help track conversations
4. **Customer IDs** are optional but helpful for personalization
5. **Error responses** include helpful error codes and messages
6. **Metrics endpoints** provide system performance data
7. **Health endpoints** show system status and component health

This covers all the available API endpoints in the airline customer service system! 🎉