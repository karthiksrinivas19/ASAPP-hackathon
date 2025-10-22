#!/usr/bin/env python3
"""
Train and evaluate entity extractor
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock logger to avoid dependencies
class MockLogger:
    def info(self, msg, **kwargs): print(f"INFO: {msg}")
    def debug(self, msg, **kwargs): print(f"DEBUG: {msg}")
    def error(self, msg, **kwargs): print(f"ERROR: {msg}")
    def warning(self, msg, **kwargs): print(f"WARNING: {msg}")

# Mock logger imports
import airline_service.utils.logger
airline_service.utils.logger.get_logger = lambda name: MockLogger()

from airline_service.ml.hybrid_entity_extractor import HybridEntityExtractor, EntityExtractionPipeline


def main():
    """Train and evaluate entity extractor"""
    
    print("🔍 Entity Extraction Training & Evaluation")
    print("=" * 50)
    
    # Paths
    dataset_path = "data/final_training_dataset.json"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run the dataset generation script first.")
        return
    
    try:
        # Initialize extraction pipeline
        print("\n📚 Initializing entity extraction pipeline...")
        pipeline = EntityExtractionPipeline()
        
        # Get extractor stats
        stats = pipeline.extractor.get_extraction_stats()
        print(f"spaCy available: {stats['spacy_available']}")
        print(f"Supported entities: {len(stats['supported_entities'])}")
        print(f"Regex patterns: {stats['regex_patterns']}")
        
        # Test basic extraction
        print(f"\n🧪 Testing basic entity extraction...")
        test_examples = [
            "I want to cancel my flight AA100 with PNR ABC123",
            "My name is John Smith, email john@gmail.com, phone (555) 123-4567",
            "Check status of booking XYZ789 for tomorrow's flight to JFK",
            "Show me window seats in business class",
            "Can I bring my dog on the flight"
        ]
        
        for i, text in enumerate(test_examples, 1):
            print(f"\n{i}. Text: '{text}'")
            entities = pipeline.extractor.extract_entities(text)
            
            if entities:
                print(f"   Entities found: {len(entities)}")
                for entity in entities:
                    print(f"     - {entity.type.value}: '{entity.value}' (confidence: {entity.confidence:.2f})")
            else:
                print("   No entities found")
        
        # Analyze full dataset
        print(f"\n📊 Analyzing full dataset...")
        dataset_results = pipeline.extract_from_dataset(dataset_path)
        
        print(f"Total examples: {dataset_results['total_examples']}")
        print(f"Examples with entities: {dataset_results['examples_with_entities']} ({dataset_results['examples_with_entities']/dataset_results['total_examples']*100:.1f}%)")
        print(f"Extraction errors: {dataset_results['extraction_errors']}")
        
        print(f"\nEntity type distribution:")
        for entity_type, count in sorted(dataset_results['entity_counts'].items()):
            print(f"  {entity_type}: {count}")
        
        # Performance benchmark
        print(f"\n⚡ Performance benchmark...")
        performance = pipeline.benchmark_performance(test_examples, iterations=50)
        
        print(f"Average latency: {performance['average_latency_ms']:.2f}ms")
        print(f"Throughput: {performance['throughput_extractions_per_sec']:.1f} extractions/sec")
        print(f"Entities per extraction: {performance['entities_per_extraction']:.2f}")
        print(f"Total entities extracted: {performance['total_entities_extracted']}")
        
        # Test specific entity types
        print(f"\n🎯 Testing specific entity type extraction...")
        
        from airline_service.types import EntityType
        
        test_cases = {
            EntityType.PNR: "Cancel booking ABC123",
            EntityType.FLIGHT_NUMBER: "Status of flight UA100",
            EntityType.EMAIL: "Contact me at john@example.com",
            EntityType.PHONE_NUMBER: "Call me at (555) 123-4567",
            EntityType.DATE: "Flight on tomorrow",
            EntityType.AIRPORT_CODE: "Flying to JFK airport",
            EntityType.PASSENGER_NAME: "My name is John Smith",
            EntityType.SEAT_TYPE: "I want a window seat",
            EntityType.CLASS: "Upgrade to business class",
            EntityType.PET_TYPE: "Traveling with my dog"
        }
        
        for entity_type, text in test_cases.items():
            entities = pipeline.extractor.extract_specific_entity_type(text, entity_type)
            if entities:
                entity = entities[0]
                print(f"  {entity_type.value}: '{text}' → '{entity.value}' (confidence: {entity.confidence:.2f})")
            else:
                print(f"  {entity_type.value}: '{text}' → No entities found")
        
        # Create and save report
        print(f"\n📋 Creating extraction report...")
        report = pipeline.create_extraction_report(dataset_results, performance)
        
        # Save report
        report_path = Path("models") / "entity_extraction_report.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
        
        # Test edge cases
        print(f"\n🔍 Testing edge cases...")
        edge_cases = [
            "",  # Empty string
            "No entities here",  # No entities
            "PNR ABC123 flight UA100 john@email.com (555) 123-4567",  # Multiple entities
            "CANCEL MY FLIGHT",  # All caps
            "i want to cancel my flight abc123",  # All lowercase
            "Flight AA100 on 12/25/2024 to LAX for John Smith"  # Multiple entity types
        ]
        
        for i, text in enumerate(edge_cases, 1):
            entities = pipeline.extractor.extract_entities(text)
            print(f"  {i}. '{text}' → {len(entities)} entities")
            for entity in entities[:3]:  # Show first 3
                print(f"     - {entity.type.value}: '{entity.value}'")
        
        # Summary
        print(f"\n🎉 Entity Extraction Summary:")
        print(f"  ✅ Extractor initialized and tested")
        print(f"  ✅ Dataset analyzed: {dataset_results['examples_with_entities']}/{dataset_results['total_examples']} examples with entities")
        print(f"  ✅ Performance: {performance['average_latency_ms']:.2f}ms latency")
        print(f"  ✅ Entity types: {len(dataset_results['entity_counts'])} different types extracted")
        print(f"  ✅ Hybrid approach: {'spaCy + Regex' if stats['spacy_available'] else 'Regex only'}")
        
        # Validation
        extraction_rate = dataset_results['examples_with_entities'] / dataset_results['total_examples']
        latency_ok = performance['average_latency_ms'] < 100
        
        print(f"\n📊 Validation Results:")
        print(f"  Entity extraction rate: {extraction_rate*100:.1f}% ({'✅ GOOD' if extraction_rate > 0.5 else '⚠️ LOW'})")
        print(f"  Latency requirement: {'✅ PASS' if latency_ok else '❌ FAIL'} (<100ms)")
        print(f"  Error rate: {dataset_results['extraction_errors']/dataset_results['total_examples']*100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Entity extraction training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()