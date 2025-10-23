# 🚀 Airline Customer Service AI System - Complete Runbook

## 📋 **Table of Contents**
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Starting the System](#starting-the-system)
5. [Health Checks](#health-checks)
6. [Feature Testing](#feature-testing)
7. [Troubleshooting](#troubleshooting)
8. [Monitoring](#monitoring)
9. [Maintenance](#maintenance)
10. [Emergency Procedures](#emergency-procedures)

---

## 🎯 **System Overview**

**Purpose:** Automated airline customer service system using AI to handle customer queries
**Technology Stack:** Python, FastAPI, Machine Learning, RAG (Retrieval Augmented Generation)
**Capabilities:** Flight cancellation, policy lookup, status checks, seat availability, pet travel

**Key Components:**
- 🧠 AI Request Classifier
- ⚙️ Workflow Orchestrator  
- 🔗 Airline API Client
- 📚 Policy RAG System
- 💬 Response Formatter

---

## 🔧 **Prerequisites**

### **System Requirements:**
- **OS:** macOS, Linux, or Windows
- **Python:** 3.8 or higher
- **Memory:** 4GB RAM minimum, 8GB recommended
- **Storage:** 2GB free space
- **Network:** Internet connection for policy scraping

### **Dependencies:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Check pip
pip --version
```

---

## 📦 **Installation & Setup**

### **Step 1: Clone/Navigate to Project**
```bash
cd /path/to/ASAPP
```

### **Step 2: Install Dependencies**
```bash
# Install required packages
pip install -r requirements.txt

# Verify key packages
pip list | grep -E "(fastapi|uvicorn|httpx|sentence-transformers|beautifulsoup4)"
```

### **Step 3: Verify Project Structure**
```bash
# Check key files exist
ls -la src/airline_service/main.py
ls -la src/airline_service/services/
ls -la src/airline_service/clients/
```

### **Step 4: Environment Setup (Optional)**
```bash
# Create .env file if needed
echo "DEBUG=true" > .env
echo "LOG_LEVEL=info" >> .env
```

---

## 🚀 **Starting the System**

### **Method 1: Direct Start (Recommended)**
```bash
# Start the server
python run_server.py

# Expected output:
# ✅ Classifier model loaded successfully
# 🚀 Starting Airline Customer Service API
# INFO: Uvicorn running on http://0.0.0.0:8000
```

### **Method 2: Development Mode**
```bash
# Start with auto-reload
uvicorn src.airline_service.main:app --reload --host 0.0.0.0 --port 8000
```

### **Startup Sequence:**
1. **Loading ML Models** (~10-30 seconds first time)
2. **Initializing Services** (~5 seconds)
3. **Policy Service Setup** (~5-10 seconds)
4. **Server Ready** - Look for "Uvicorn running on..."

### **Startup Logs to Expect:**
```
2025-10-23 [info] Health check registered name=ml_classifier
2025-10-23 [info] Health check registered name=airline_api
2025-10-23 [info] Health check registered name=policy_service
INFO: Started server process [12345]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## 🧪 **Feature Testing**

### **Test 1: Flight Cancellation**
```bash
curl -X POST "http://localhost:8000/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "I want to cancel my flight ABC123",
    "session_id": "test-cancel-1"
  }'

# ✅ Success Indicators:
# - status: "completed"
# - executed_tasks: ["extract_identifiers", "get_booking_details", "confirm_booking_details", "cancel_flight", "inform_cancellation_result"]
# - refund_amount: 150.0
# - cancellation_charges: 50.0
```

### **Test 2: Cancellation Policy**
```bash
curl -X POST "http://localhost:8000/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "What is your cancellation policy?",
    "session_id": "test-policy-1"
  }'

# ✅ Success Indicators:
# - status: "completed"
# - content contains "24-Hour Cancellation Rule"
# - content contains "Blue Basic Fares"
# - executed_tasks: ["extract_flight_context", "get_cancellation_policy", "inform_policy_info"]
```

### **Test 3: Flight Status**
```bash
curl -X POST "http://localhost:8000/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "What is the status of flight ABC123?",
    "session_id": "test-status-1"
  }'

# ✅ Success Indicators:
# - status: "completed"
# - pnr: "ABC123"
# - current_status: "On Time"
# - source_airport_code: "JFK"
# - destination_airport_code: "LAX"
```

### **Test 4: Seat Availability**
```bash
curl -X POST "http://localhost:8000/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "Show me available seats for ABC123",
    "session_id": "test-seats-1"
  }'

# ✅ Success Indicators:
# - status: "completed"
# - available_seats array with seat options
# - seats include "10A", "10B", "15F", "2A"
# - pricing information included
```

### **Test 5: Pet Travel Policy**
```bash
curl -X POST "http://localhost:8000/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{
    "utterance": "Can I bring my dog on the plane?",
    "session_id": "test-pet-1"
  }'

# ✅ Success Indicators:
# - status: "completed"
# - content contains "In-Cabin Pet Travel"
# - content contains "$125 each way"
# - content contains "Service Animals"
```
