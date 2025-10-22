#!/usr/bin/env python3
"""
Simple test runner for classifier (no pytest dependency)
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock logger
class MockLogger:
    def info(self, msg, **kwargs): pass
    def debug(self, msg, **kwargs): pass
    def error(self, msg, **kwargs): pass
    def warning(self, msg, **kwargs): pass

import airline_service.utils.logger
airline_service.utils.logger.get_logger = lambda name: MockLogger()

from airline_service.services.request_classifier_service import ClassifierFactory
from airline_service.types import RequestType


async def test_basic_classification():
    """Test basic classification functionality"""
    
    print("🧪 Testing Basic Classification")
    
    classifier = ClassifierFactory.create_mock_classifier()
    
    test_cases = [
        ("I want to cancel my flight", RequestType.CANCEL_TRIP),
        ("What's my flight status", RequestType.FLIGHT_STATUS),
        ("Show available seats", RequestType.SEAT_AVAILABILITY),
        ("What's your cancellation policy", RequestType.CANCELLATION_POLICY),
        ("Can I bring my pet", RequestType.PET_TRAVEL)
    ]
    
    passed = 0
    total = len(test_cases)
    
    for utterance, expected_intent in test_cases:
        result = await classifier.classify_request(utterance)
        
        if result.request_type == expected_intent:
            print(f"  ✅ '{utterance}' → {result.request_type.value}")
            passed += 1
        else:
            print(f"  ❌ '{utterance}' → {result.request_type.value} (expected {expected_intent.value})")
    
    print(f"  Result: {passed}/{total} tests passed")
    return passed == total


async def test_confidence_scores():
    """Test confidence scoring"""
    
    print("\n🎯 Testing Confidence Scores")
    
    classifier = ClassifierFactory.create_mock_classifier()
    
    result = await classifier.classify_request("I want to cancel my flight")
    
    # Check confidence is in valid range
    if 0.0 <= result.confidence <= 1.0:
        print(f"  ✅ Confidence in valid range: {result.confidence:.3f}")
        confidence_ok = True
    else:
        print(f"  ❌ Confidence out of range: {result.confidence:.3f}")
        confidence_ok = False
    
    # Check alternatives exist
    if len(result.alternative_intents) > 0:
        print(f"  ✅ Alternative intents provided: {len(result.alternative_intents)}")
        alternatives_ok = True
    else:
        print(f"  ❌ No alternative intents provided")
        alternatives_ok = False
    
    return confidence_ok and alternatives_ok


async def test_batch_processing():
    """Test batch processing"""
    
    print("\n📦 Testing Batch Processing")
    
    classifier = ClassifierFactory.create_mock_classifier()
    
    utterances = [
        "I want to cancel my flight",
        "What's my flight status",
        "Show available seats"
    ]
    
    results = await classifier.predict_batch(utterances)
    
    if len(results) == len(utterances):
        print(f"  ✅ Batch processing: {len(results)} results for {len(utterances)} inputs")
        batch_ok = True
    else:
        print(f"  ❌ Batch processing: {len(results)} results for {len(utterances)} inputs")
        batch_ok = False
    
    # Check each result
    for i, (utterance, result) in enumerate(zip(utterances, results)):
        print(f"    {i+1}. '{utterance}' → {result.request_type.value}")
    
    return batch_ok


async def test_performance():
    """Test performance requirements"""
    
    print("\n⚡ Testing Performance")
    
    classifier = ClassifierFactory.create_mock_classifier()
    
    import time
    
    # Latency test
    utterance = "I want to cancel my flight"
    iterations = 100
    
    start_time = time.time()
    for _ in range(iterations):
        await classifier.classify_request(utterance)
    
    total_time = time.time() - start_time
    avg_latency = (total_time / iterations) * 1000  # ms
    throughput = iterations / total_time  # requests/sec
    
    latency_ok = avg_latency < 100
    throughput_ok = throughput > 100
    
    print(f"  Average latency: {avg_latency:.2f}ms ({'✅ PASS' if latency_ok else '❌ FAIL'} <100ms)")
    print(f"  Throughput: {throughput:.1f} req/sec ({'✅ PASS' if throughput_ok else '❌ FAIL'} >100 req/sec)")
    
    return latency_ok and throughput_ok


async def test_model_info():
    """Test model information"""
    
    print("\n📋 Testing Model Information")
    
    classifier = ClassifierFactory.create_mock_classifier()
    
    # Test model loading status
    if classifier.is_loaded():
        print("  ✅ Model loaded successfully")
        loaded_ok = True
    else:
        print("  ❌ Model not loaded")
        loaded_ok = False
    
    # Test model info
    model_info = classifier.get_model_info()
    
    required_fields = ["status", "num_classes", "classes"]
    info_ok = all(field in model_info for field in required_fields)
    
    if info_ok:
        print("  ✅ Model info contains required fields")
        print(f"    Status: {model_info['status']}")
        print(f"    Classes: {model_info['num_classes']}")
    else:
        print("  ❌ Model info missing required fields")
    
    # Test supported intents
    intents = classifier.get_supported_intents()
    intents_ok = len(intents) == 5
    
    if intents_ok:
        print(f"  ✅ Supported intents: {len(intents)}")
    else:
        print(f"  ❌ Supported intents: {len(intents)} (expected 5)")
    
    return loaded_ok and info_ok and intents_ok


async def test_edge_cases():
    """Test edge cases"""
    
    print("\n🔍 Testing Edge Cases")
    
    classifier = ClassifierFactory.create_mock_classifier()
    
    edge_cases = [
        ("", "empty string"),
        ("a", "single character"),
        ("I want to cancel my flight " * 20, "very long utterance"),
        ("xyz123", "nonsense input"),
        ("CANCEL MY FLIGHT", "all caps"),
        ("i want to cancel my flight", "all lowercase")
    ]
    
    passed = 0
    total = len(edge_cases)
    
    for utterance, description in edge_cases:
        try:
            result = await classifier.classify_request(utterance)
            
            # Check that we get a valid result
            if (hasattr(result, 'request_type') and 
                hasattr(result, 'confidence') and
                0.0 <= result.confidence <= 1.0):
                print(f"  ✅ {description}: {result.request_type.value} ({result.confidence:.3f})")
                passed += 1
            else:
                print(f"  ❌ {description}: Invalid result")
        
        except Exception as e:
            print(f"  ❌ {description}: Exception - {e}")
    
    print(f"  Result: {passed}/{total} edge cases handled")
    return passed == total


async def main():
    """Run all tests"""
    
    print("🚀 Running Classifier Tests")
    print("=" * 40)
    
    tests = [
        test_basic_classification,
        test_confidence_scores,
        test_batch_processing,
        test_performance,
        test_model_info,
        test_edge_cases
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎉 Test Summary")
    print(f"  Passed: {passed}/{total}")
    print(f"  Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("  🎊 All tests passed!")
    else:
        print("  ⚠️  Some tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)