#!/usr/bin/env python3
"""
Test improved entity extractor
"""

import sys
import json
from pathlib import Path
import time

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

from airline_service.ml.improved_entity_extractor import ImprovedEntityExtractor
from airline_service.types import EntityType


def test_improved_extractor():
    """Test the improved entity extractor"""
    
    print("🔍 Testing Improved Entity Extractor")
    print("=" * 40)
    
    # Initialize extractor
    extractor = ImprovedEntityExtractor()
    
    # Get stats
    stats = extractor.get_extraction_stats()
    print(f"Algorithm: {stats['algorithm']}")
    print(f"Supported entities: {len(stats['supported_entities'])}")
    print(f"Context aware: {stats['context_aware']}")
    
    # Test cases with expected results
    test_cases = [
        {
            'text': "I want to cancel my flight AA100 with PNR ABC123",
            'expected': ['flight_number', 'pnr']
        },
        {
            'text': "My name is John Smith, email john@gmail.com, phone (555) 123-4567",
            'expected': ['passenger_name', 'email', 'phone_number']
        },
        {
            'text': "Check status of booking XYZ789 for tomorrow's flight to JFK",
            'expected': ['pnr', 'date', 'airport_code']
        },
        {
            'text': "Show me window seats in business class",
            'expected': ['seat_type', 'class']
        },
        {
            'text': "Can I bring my dog on the flight",
            'expected': ['pet_type']
        },
        {
            'text': "Flight status for January 15th",
            'expected': ['date']
        },
        {
            'text': "I need help with my booking",
            'expected': []  # Should not extract false positives
        }
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} examples:")
    print("-" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case['text']
        expected_types = set(test_case['expected'])
        
        entities = extractor.extract_entities(text)
        extracted_types = set(entity.type.value for entity in entities)
        
        # Check if we got the expected entity types
        correct = extracted_types == expected_types
        
        print(f"{i}. '{text}'")
        print(f"   Expected: {sorted(expected_types) if expected_types else 'None'}")
        print(f"   Extracted: {sorted(extracted_types) if extracted_types else 'None'}")
        
        if entities:
            for entity in entities:
                print(f"     - {entity.type.value}: '{entity.value}' (confidence: {entity.confidence:.2f})")
        
        print(f"   Result: {'✅ PASS' if correct else '❌ FAIL'}")
        print()
        
        total_tests += 1
        if correct:
            passed_tests += 1
    
    print(f"Test Results: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
    
    # Performance test
    print(f"\n⚡ Performance Test:")
    
    test_texts = [case['text'] for case in test_cases]
    iterations = 100
    
    start_time = time.time()
    total_entities = 0
    
    for _ in range(iterations):
        for text in test_texts:
            entities = extractor.extract_entities(text)
            total_entities += len(entities)
    
    total_time = time.time() - start_time
    total_extractions = iterations * len(test_texts)
    
    avg_latency = (total_time / total_extractions) * 1000
    throughput = total_extractions / total_time
    
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Throughput: {throughput:.1f} extractions/sec")
    print(f"  Total entities: {total_entities}")
    print(f"  Entities per extraction: {total_entities/total_extractions:.2f}")
    
    # Test specific entity types
    print(f"\n🎯 Testing Specific Entity Types:")
    
    specific_tests = {
        EntityType.PNR: [
            ("Cancel booking ABC123", True),
            ("PNR XYZ789", True),
            ("My flight", False)  # Should not extract
        ],
        EntityType.FLIGHT_NUMBER: [
            ("Flight AA100", True),
            ("Status of UA200", True),
            ("My trip", False)  # Should not extract
        ],
        EntityType.EMAIL: [
            ("Email john@gmail.com", True),
            ("Contact test@example.com", True),
            ("No email here", False)
        ],
        EntityType.DATE: [
            ("Flight tomorrow", True),
            ("Travel on January 15", True),
            ("My trip", False)
        ],
        EntityType.AIRPORT_CODE: [
            ("Flying to JFK", True),
            ("From LAX airport", True),
            ("Going home", False)  # Should not extract
        ]
    }
    
    for entity_type, tests in specific_tests.items():
        print(f"\n{entity_type.value}:")
        
        for text, should_extract in tests:
            entities = extractor.extract_specific_entity_type(text, entity_type)
            extracted = len(entities) > 0
            
            result = "✅ PASS" if extracted == should_extract else "❌ FAIL"
            
            if entities:
                entity_values = [f"'{e.value}'" for e in entities]
                print(f"  '{text}' → {', '.join(entity_values)} ({result})")
            else:
                print(f"  '{text}' → No entities ({result})")
    
    # Edge cases
    print(f"\n🔍 Testing Edge Cases:")
    
    edge_cases = [
        "",  # Empty string
        "Hello world",  # No entities
        "ABC123 UA100 john@test.com tomorrow JFK",  # Multiple entities
        "FLIGHT AA100 PNR ABC123",  # All caps
        "flight aa100 pnr abc123",  # All lowercase
    ]
    
    for text in edge_cases:
        entities = extractor.extract_entities(text)
        print(f"  '{text}' → {len(entities)} entities")
        for entity in entities[:3]:  # Show first 3
            print(f"    - {entity.type.value}: '{entity.value}'")
    
    return passed_tests == total_tests


def main():
    """Main test function"""
    
    try:
        success = test_improved_extractor()
        
        print(f"\n🎉 Test Summary:")
        if success:
            print("  ✅ All tests passed!")
            print("  ✅ Improved extractor working correctly")
            print("  ✅ False positive reduction successful")
        else:
            print("  ⚠️  Some tests failed")
            print("  ⚠️  Review extraction logic")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)