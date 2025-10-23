#!/usr/bin/env python3
"""
Comprehensive API test script for airline customer service system
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_test_result(test_name, response):
    """Print formatted test result"""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    
    try:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_health_endpoints():
    """Test health and status endpoints"""
    
    # Basic health check
    response = requests.get(f"{BASE_URL}/health")
    print_test_result("Health Check", response)
    
    # Detailed status
    response = requests.get(f"{BASE_URL}/api/v1/status")
    print_test_result("Service Status", response)
    
    # Metrics summary
    response = requests.get(f"{BASE_URL}/api/v1/metrics/summary")
    print_test_result("Metrics Summary", response)

def test_customer_service_endpoints():
    """Test main customer service endpoints"""
    
    test_cases = [
        {
            "name": "Flight Cancellation",
            "data": {
                "utterance": "I want to cancel my flight booking ABC123",
                "session_id": "test-cancel-1"
            }
        },
        {
            "name": "Flight Status",
            "data": {
                "utterance": "What's the status of flight JB1234?",
                "session_id": "test-status-1"
            }
        },
        {
            "name": "Seat Availability",
            "data": {
                "utterance": "Can I see available seats on my flight?",
                "session_id": "test-seats-1"
            }
        },
        {
            "name": "Cancellation Policy",
            "data": {
                "utterance": "What's your cancellation policy?",
                "session_id": "test-policy-1"
            }
        },
        {
            "name": "Pet Travel",
            "data": {
                "utterance": "Can I travel with my dog?",
                "session_id": "test-pet-1"
            }
        },
        {
            "name": "Complex Request",
            "data": {
                "utterance": "I need to cancel flight JB1234, my booking is ABC123",
                "session_id": "test-complex-1",
                "customer_id": "john.doe@email.com"
            }
        }
    ]
    
    for test_case in test_cases:
        response = requests.post(
            f"{BASE_URL}/api/v1/customer-service/query",
            headers={"Content-Type": "application/json"},
            json=test_case["data"]
        )
        print_test_result(test_case["name"], response)
        time.sleep(0.5)  # Small delay between requests

def test_error_handling():
    """Test error handling scenarios"""
    
    error_tests = [
        {
            "name": "Empty Utterance",
            "data": {
                "utterance": "",
                "session_id": "test-error-1"
            }
        },
        {
            "name": "Too Long Utterance",
            "data": {
                "utterance": "a" * 1001,  # Over 1000 character limit
                "session_id": "test-error-2"
            }
        },
        {
            "name": "Missing Utterance",
            "data": {
                "session_id": "test-error-3"
            }
        }
    ]
    
    for test_case in error_tests:
        response = requests.post(
            f"{BASE_URL}/api/v1/customer-service/query",
            headers={"Content-Type": "application/json"},
            json=test_case["data"]
        )
        print_test_result(f"Error Test: {test_case['name']}", response)
        time.sleep(0.5)

def test_simple_endpoint():
    """Test the simple processing endpoint"""
    
    response = requests.post(
        f"{BASE_URL}/api/v1/customer-service/query/simple",
        headers={"Content-Type": "application/json"},
        json={
            "utterance": "I want to cancel my flight",
            "session_id": "test-simple-1"
        }
    )
    print_test_result("Simple Endpoint", response)

def test_invalid_endpoints():
    """Test invalid endpoints"""
    
    # Non-existent endpoint
    response = requests.get(f"{BASE_URL}/api/v1/nonexistent")
    print_test_result("Invalid Endpoint", response)
    
    # Wrong method
    response = requests.get(f"{BASE_URL}/api/v1/customer-service/query")
    print_test_result("Wrong HTTP Method", response)

def main():
    """Run all API tests"""
    
    print("🚀 Airline Customer Service API Test Suite")
    print("=" * 60)
    print(f"Testing server at: {BASE_URL}")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test if server is running
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding properly!")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print(f"Error: {e}")
        print("\n💡 Make sure to start the server first:")
        print("   python run_server.py")
        return
    
    print("✅ Server is running!")
    
    # Run all test suites
    print("\n" + "="*60)
    print("📊 TESTING HEALTH & STATUS ENDPOINTS")
    print("="*60)
    test_health_endpoints()
    
    print("\n" + "="*60)
    print("🎯 TESTING CUSTOMER SERVICE ENDPOINTS")
    print("="*60)
    test_customer_service_endpoints()
    
    print("\n" + "="*60)
    print("🔧 TESTING SIMPLE ENDPOINT")
    print("="*60)
    test_simple_endpoint()
    
    print("\n" + "="*60)
    print("❌ TESTING ERROR HANDLING")
    print("="*60)
    test_error_handling()
    
    print("\n" + "="*60)
    print("🚫 TESTING INVALID ENDPOINTS")
    print("="*60)
    test_invalid_endpoints()
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*60)
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final metrics check
    try:
        response = requests.get(f"{BASE_URL}/api/v1/metrics/summary")
        if response.status_code == 200:
            metrics = response.json()
            print(f"\n📈 Final System Metrics:")
            print(f"   Requests processed: {metrics.get('system_health', {}).get('requests_per_minute', 'N/A')}")
            print(f"   Average response time: {metrics.get('system_health', {}).get('avg_response_time_ms', 'N/A')}ms")
            print(f"   Error rate: {metrics.get('system_health', {}).get('error_rate', 'N/A')}")
    except:
        pass

if __name__ == "__main__":
    main()