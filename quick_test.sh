#!/bin/bash

# Quick API test script for airline customer service system

BASE_URL="http://localhost:8000"

echo "🚀 Quick API Test for Airline Customer Service"
echo "=============================================="

# Check if server is running
echo "🔍 Checking if server is running..."
if ! curl -s "$BASE_URL/health" > /dev/null; then
    echo "❌ Server is not running!"
    echo "💡 Start the server first: python run_server.py"
    exit 1
fi

echo "✅ Server is running!"
echo ""

# Test 1: Health Check
echo "1️⃣  Health Check"
echo "----------------"
curl -s "$BASE_URL/health" | jq -r '.status // "No status found"'
echo ""

# Test 2: Service Status
echo "2️⃣  Service Status"
echo "----------------"
curl -s "$BASE_URL/api/v1/status" | jq -r '.service.name // "No service name found"'
echo ""

# Test 3: Flight Cancellation
echo "3️⃣  Flight Cancellation Test"
echo "---------------------------"
curl -s -X POST "$BASE_URL/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{"utterance": "I want to cancel my flight ABC123", "session_id": "quick-test-1"}' | \
  jq -r '.message // "No message found"'
echo ""

# Test 4: Flight Status
echo "4️⃣  Flight Status Test"
echo "--------------------"
curl -s -X POST "$BASE_URL/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{"utterance": "What is the status of flight JB1234?", "session_id": "quick-test-2"}' | \
  jq -r '.message // "No message found"'
echo ""

# Test 5: Seat Availability
echo "5️⃣  Seat Availability Test"
echo "-------------------------"
curl -s -X POST "$BASE_URL/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{"utterance": "Can I see available seats?", "session_id": "quick-test-3"}' | \
  jq -r '.message // "No message found"'
echo ""

# Test 6: Policy Information
echo "6️⃣  Policy Information Test"
echo "--------------------------"
curl -s -X POST "$BASE_URL/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{"utterance": "What is your cancellation policy?", "session_id": "quick-test-4"}' | \
  jq -r '.message // "No message found"'
echo ""

# Test 7: Pet Travel
echo "7️⃣  Pet Travel Test"
echo "------------------"
curl -s -X POST "$BASE_URL/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{"utterance": "Can I travel with my dog?", "session_id": "quick-test-5"}' | \
  jq -r '.message // "No message found"'
echo ""

# Test 8: Error Handling
echo "8️⃣  Error Handling Test"
echo "----------------------"
curl -s -X POST "$BASE_URL/api/v1/customer-service/query" \
  -H "Content-Type: application/json" \
  -d '{"utterance": "", "session_id": "quick-test-6"}' | \
  jq -r '.message // "No message found"'
echo ""

# Test 9: Metrics Summary
echo "9️⃣  Metrics Summary"
echo "------------------"
curl -s "$BASE_URL/api/v1/metrics/summary" | \
  jq -r '.system_health.status // "No status found"'
echo ""

echo "🎉 All quick tests completed!"
echo ""
echo "📋 For more detailed testing:"
echo "   • Run: python test_all_apis.py"
echo "   • Check: api_test_calls.md for all available endpoints"