#!/usr/bin/env python3
"""
Test client for the airline customer service API
"""

import requests
import json
import time
from datetime import datetime


def test_api():
    """Test the airline customer service API"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Airline Customer Service API")
    print("=" * 40)
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   Status: {health_data['status']}")
            print(f"   Classifier loaded: {health_data['classifier_loaded']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to server. Make sure it's running on localhost:8000")
        print("   Run: python simple_server.py")
        return
    
    # Test customer queries
    test_queries = [
        {
            "utterance": "I want to cancel my flight ABC123",
            "customer_id": "test1"
        },
        {
            "utterance": "What's the status of flight UA100",
            "customer_id": "test2"
        },
        {
            "utterance": "Show me available window seats",
            "customer_id": "test3"
        },
        {
            "utterance": "What's your cancellation policy",
            "customer_id": "test4"
        },
        {
            "utterance": "Can I bring my dog on the flight",
            "customer_id": "test5"
        }
    ]
    
    print(f"\n2. Testing customer queries...")
    
    for i, query_data in enumerate(test_queries, 1):
        print(f"\n   Query {i}: '{query_data['utterance']}'")
        
        try:
            response = requests.post(
                f"{base_url}/api/v1/customer-service/query",
                json=query_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Response received")
                print(f"   Intent: {result['data']['intent']}")
                print(f"   Confidence: {result['data']['confidence']:.3f}")
                print(f"   Message: {result['message'][:100]}...")
                
                if result['data']['entities']:
                    print(f"   Entities: {len(result['data']['entities'])}")
                    for entity in result['data']['entities']:
                        print(f"     - {entity['type']}: {entity['value']}")
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                print(f"   Error: {response.text}")
        
        except Exception as e:
            print(f"   ❌ Request error: {e}")
    
    # Performance test
    print(f"\n3. Performance test...")
    
    test_query = {
        "utterance": "I want to cancel my flight",
        "customer_id": "perf_test"
    }
    
    num_requests = 10
    start_time = time.time()
    
    successful_requests = 0
    
    for i in range(num_requests):
        try:
            response = requests.post(
                f"{base_url}/api/v1/customer-service/query",
                json=test_query,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                successful_requests += 1
        
        except Exception:
            pass
    
    total_time = time.time() - start_time
    avg_latency = (total_time / num_requests) * 1000  # ms
    
    print(f"   Requests: {successful_requests}/{num_requests}")
    print(f"   Average latency: {avg_latency:.2f}ms")
    print(f"   Throughput: {successful_requests/total_time:.1f} req/sec")
    
    print(f"\n✅ API testing completed!")


def interactive_test():
    """Interactive testing mode"""
    
    base_url = "http://localhost:8000"
    
    print("🎯 Interactive API Testing")
    print("=" * 30)
    print("Enter customer queries to test the API (type 'quit' to exit)")
    
    while True:
        try:
            query = input("\nCustomer query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            query_data = {
                "utterance": query,
                "customer_id": "interactive_test"
            }
            
            response = requests.post(
                f"{base_url}/api/v1/customer-service/query",
                json=query_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n📋 Response:")
                print(f"   Intent: {result['data']['intent']}")
                print(f"   Confidence: {result['data']['confidence']:.3f}")
                print(f"   Message: {result['message']}")
                
                if result['data']['entities']:
                    print(f"   Entities:")
                    for entity in result['data']['entities']:
                        print(f"     - {entity['type']}: {entity['value']} (confidence: {entity['confidence']:.2f})")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Interactive testing ended")


def main():
    """Main function"""
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_test()
    else:
        test_api()


if __name__ == "__main__":
    main()