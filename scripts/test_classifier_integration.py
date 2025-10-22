#!/usr/bin/env python3
"""
Test classifier integration with the main application
"""

import sys
import json
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock the logger
class MockLogger:
    def info(self, msg, **kwargs): print(f"INFO: {msg}")
    def debug(self, msg, **kwargs): print(f"DEBUG: {msg}")
    def error(self, msg, **kwargs): print(f"ERROR: {msg}")
    def warning(self, msg, **kwargs): print(f"WARNING: {msg}")

# Mock logger imports
import airline_service.utils.logger
airline_service.utils.logger.get_logger = lambda name: MockLogger()

from airline_service.services.request_classifier_service import ClassifierFactory


async def test_classifier():
    """Test the classifier service"""
    
    print("🧪 Testing Request Classifier Integration")
    print("=" * 50)
    
    # Test with trained model if available
    print("\n📚 Loading classifier...")
    classifier = ClassifierFactory.create_classifier("models/simple-classifier")
    
    if not classifier.is_loaded():
        print("⚠️  Trained model not found, using mock classifier")
        classifier = ClassifierFactory.create_mock_classifier()
    
    # Get model info
    model_info = classifier.get_model_info()
    print(f"Model status: {model_info['status']}")
    print(f"Classes: {model_info.get('num_classes', 'N/A')}")
    
    # Test examples
    test_cases = [
        "I want to cancel my flight ABC123",
        "What's the status of flight UA100",
        "Show me available window seats",
        "What's your cancellation policy",
        "Can I bring my dog on the flight",
        "I need help with my booking",
        "Cancel my reservation PNR XYZ789",
        "Is my flight delayed",
        "Available seats in business class",
        "Pet travel requirements"
    ]
    
    print(f"\n🔍 Testing {len(test_cases)} examples...")
    print("-" * 50)
    
    for i, utterance in enumerate(test_cases, 1):
        try:
            result = await classifier.classify_request(utterance)
            
            print(f"{i:2d}. '{utterance}'")
            print(f"    → Intent: {result.request_type.value}")
            print(f"    → Confidence: {result.confidence:.3f}")
            
            if result.extracted_entities:
                print(f"    → Entities: {len(result.extracted_entities)}")
                for entity in result.extracted_entities[:3]:  # Show first 3
                    print(f"      - {entity.type.value}: {entity.value} ({entity.confidence:.2f})")
            
            if result.alternative_intents:
                alt = result.alternative_intents[0]
                print(f"    → Alternative: {alt['type'].value} ({alt['confidence']:.3f})")
            
            print()
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            print()
    
    # Performance test
    print("⚡ Performance Test...")
    
    import time
    test_text = "I want to cancel my flight"
    iterations = 100
    
    start_time = time.time()
    for _ in range(iterations):
        await classifier.classify_request(test_text)
    
    total_time = time.time() - start_time
    avg_latency = (total_time / iterations) * 1000  # ms
    throughput = iterations / total_time  # requests/sec
    
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Throughput: {throughput:.1f} requests/sec")
    print(f"  Target <100ms: {'✅ PASS' if avg_latency < 100 else '❌ FAIL'}")
    
    # Batch test
    print(f"\n📦 Batch Processing Test...")
    batch_texts = test_cases[:5]
    
    start_time = time.time()
    batch_results = await classifier.predict_batch(batch_texts)
    batch_time = time.time() - start_time
    
    print(f"  Processed {len(batch_texts)} requests in {batch_time*1000:.2f}ms")
    print(f"  Average per request: {(batch_time/len(batch_texts))*1000:.2f}ms")
    
    for text, result in zip(batch_texts, batch_results):
        print(f"  '{text[:30]}...' → {result.request_type.value}")
    
    print(f"\n✅ Classifier integration test completed!")


async def test_api_simulation():
    """Simulate API calls"""
    
    print(f"\n🌐 API Simulation Test")
    print("-" * 30)
    
    # Simulate the main app logic
    from airline_service.types import CustomerRequest, APIResponse
    
    test_requests = [
        {"utterance": "I want to cancel my flight", "customer_id": "test1"},
        {"utterance": "What's my flight status", "customer_id": "test2"},
        {"utterance": "Show available seats", "customer_id": "test3"},
    ]
    
    classifier = ClassifierFactory.create_classifier("models/simple-classifier")
    if not classifier.is_loaded():
        classifier = ClassifierFactory.create_mock_classifier()
    
    for req_data in test_requests:
        print(f"\nRequest: {req_data['utterance']}")
        
        # Classify request
        result = await classifier.classify_request(req_data['utterance'])
        
        # Generate response (simplified version of main app logic)
        intent = result.request_type.value
        confidence = result.confidence
        entities = result.extracted_entities
        
        if intent == "cancel_trip":
            message = "I can help you cancel your flight."
        elif intent == "flight_status":
            message = "I can check your flight status."
        elif intent == "seat_availability":
            message = "I can show you available seats."
        else:
            message = f"I understand you need help with {intent}."
        
        # Create API response
        response = {
            "status": "completed",
            "message": message,
            "data": {
                "intent": intent,
                "confidence": confidence,
                "entities": len(entities)
            }
        }
        
        print(f"Response: {response['message']}")
        print(f"Intent: {intent} (confidence: {confidence:.3f})")


def main():
    """Main test function"""
    
    try:
        # Run async tests
        asyncio.run(test_classifier())
        asyncio.run(test_api_simulation())
        
        print(f"\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()