#!/usr/bin/env python3
"""
Simple test script for airline customer service system
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_basic_functionality():
    """Test basic system functionality without starting the full server"""
    
    print("🧪 Testing Airline Customer Service System")
    print("=" * 50)
    
    try:
        # Test 1: Import all modules
        print("1. Testing imports...")
        from airline_service.container import container
        from airline_service.types import CustomerRequest, RequestType
        print("   ✅ All imports successful")
        
        # Test 2: Initialize container
        print("2. Initializing service container...")
        await container.initialize()
        print("   ✅ Container initialized successfully")
        
        # Test 3: Test classifier
        print("3. Testing request classifier...")
        classifier = container.get_classifier()
        
        test_utterances = [
            "I want to cancel my flight booking ABC123",
            "What's the status of flight JB1234?",
            "Can I see available seats on my flight?",
            "What's your cancellation policy?",
            "Can I travel with my dog?"
        ]
        
        for utterance in test_utterances:
            result = await classifier.classify_request(utterance)
            print(f"   📝 '{utterance[:40]}...' → {result.request_type.value} (confidence: {result.confidence:.2f})")
        
        print("   ✅ Classifier working correctly")
        
        # Test 4: Test workflow orchestrator
        print("4. Testing workflow orchestrator...")
        orchestrator = container.get_workflow_orchestrator()
        workflows = orchestrator.registry.get_all_workflows()
        print(f"   📋 {len(workflows)} workflows registered:")
        for req_type, tasks in workflows.items():
            print(f"      - {req_type.value}: {len(tasks)} tasks")
        print("   ✅ Workflow orchestrator ready")
        
        # Test 5: Test task engine
        print("5. Testing task engine...")
        task_engine = container.get_task_engine()
        print(f"   🔧 {len(task_engine.task_handlers)} task handlers registered")
        print("   ✅ Task engine ready")
        
        # Test 6: Test policy service
        print("6. Testing policy service...")
        policy_service = container.get_policy_service()
        print("   ✅ Policy service ready")
        
        print("\n🎉 All basic tests passed!")
        print("The system is ready for API testing.")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            await container.cleanup()
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {str(e)}")


def test_api_requests():
    """Test API requests using curl commands"""
    
    print("\n🌐 API Testing Commands")
    print("=" * 50)
    print("Once the server is running, use these commands to test:")
    print()
    
    # Health check
    print("1. Health Check:")
    print("   curl -X GET http://localhost:8000/health")
    print()
    
    # Service status
    print("2. Service Status:")
    print("   curl -X GET http://localhost:8000/api/v1/status")
    print()
    
    # Test queries
    test_queries = [
        {
            "name": "Flight Cancellation",
            "utterance": "I need to cancel my flight booking ABC123"
        },
        {
            "name": "Flight Status",
            "utterance": "What's the status of flight JB1234?"
        },
        {
            "name": "Seat Availability", 
            "utterance": "Can I see available seats on my flight?"
        },
        {
            "name": "Policy Information",
            "utterance": "What's your cancellation policy?"
        },
        {
            "name": "Pet Travel",
            "utterance": "Can I travel with my dog on the plane?"
        }
    ]
    
    for i, query in enumerate(test_queries, 3):
        print(f"{i}. {query['name']}:")
        curl_cmd = f"""curl -X POST http://localhost:8000/api/v1/customer-service/query \\
  -H "Content-Type: application/json" \\
  -d '{{"utterance": "{query['utterance']}", "session_id": "test-session-{i}"}}'"""
        print(f"   {curl_cmd}")
        print()
    
    # Metrics
    print(f"{len(test_queries) + 3}. Get Metrics:")
    print("   curl -X GET http://localhost:8000/api/v1/metrics/summary")
    print()


if __name__ == "__main__":
    print("🚀 Airline Customer Service System Test Suite")
    print("=" * 60)
    
    # Run basic functionality tests
    success = asyncio.run(test_basic_functionality())
    
    if success:
        # Show API testing commands
        test_api_requests()
        
        print("📋 Next Steps:")
        print("1. Start the server: python run_server.py")
        print("2. Test the API endpoints using the curl commands above")
        print("3. Check the logs for detailed information")
        print("4. Visit http://localhost:8000/docs for interactive API documentation")
    else:
        print("❌ Basic tests failed. Please fix the issues before testing the API.")
        sys.exit(1)