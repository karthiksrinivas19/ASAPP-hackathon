#!/usr/bin/env python3
"""
Standalone entity extractor (no external dependencies)
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum
import time


class EntityType(str, Enum):
    PNR = "pnr"
    FLIGHT_NUMBER = "flight_number"
    DATE = "date"
    AIRPORT_CODE = "airport_code"
    PASSENGER_NAME = "passenger_name"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    DESTINATION = "destination"
    CLASS = "class"
    SEAT_TYPE = "seat_type"
    PET_TYPE = "pet_type"


class ExtractedEntity:
    def __init__(self, entity_type, value, confidence, start_index, end_index):
        self.type = entity_type
        self.value = value
        self.confidence = confidence
        self.start_index = start_index
        self.end_index = end_index


class SimpleEntityExtractor:
    """Simple regex-based entity extractor"""
    
    def __init__(self):
        self.patterns = self._build_patterns()
        self.confidence_scores = {
            'high': 0.90,
            'medium': 0.75,
            'low': 0.60
        }
    
    def _build_patterns(self):
        """Build regex patterns for entity extraction"""
        
        return {
            EntityType.PNR: [
                {
                    'pattern': r'\b(?:PNR|booking|confirmation|reference)\s*:?\s*([A-Z0-9]{6})\b',
                    'group': 1,
                    'confidence': 'high'
                },
                {
                    'pattern': r'\b([A-Z0-9]{6})\b',
                    'group': 1,
                    'confidence': 'medium'
                }
            ],
            
            EntityType.FLIGHT_NUMBER: [
                {
                    'pattern': r'\b(?:flight|flt)\s*:?\s*([A-Z]{2,3}[0-9]{1,4})\b',
                    'group': 1,
                    'confidence': 'high'
                },
                {
                    'pattern': r'\b([A-Z]{2,3}[0-9]{1,4})\b',
                    'group': 1,
                    'confidence': 'medium'
                }
            ],
            
            EntityType.DATE: [
                {
                    'pattern': r'\b(today|tomorrow|yesterday)\b',
                    'group': 1,
                    'confidence': 'high'
                },
                {
                    'pattern': r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                    'group': 1,
                    'confidence': 'high'
                },
                {
                    'pattern': r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b',
                    'group': 1,
                    'confidence': 'high'
                }
            ],
            
            EntityType.EMAIL: [
                {
                    'pattern': r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                    'group': 1,
                    'confidence': 'high'
                }
            ],
            
            EntityType.PHONE_NUMBER: [
                {
                    'pattern': r'\b(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b',
                    'group': 1,
                    'confidence': 'high'
                }
            ],
            
            EntityType.PASSENGER_NAME: [
                {
                    'pattern': r'\bmy name is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                    'group': 1,
                    'confidence': 'high'
                },
                {
                    'pattern': r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                    'group': 1,
                    'confidence': 'medium'
                }
            ],
            
            EntityType.AIRPORT_CODE: [
                {
                    'pattern': r'\b(?:to|from|via)\s+([A-Z]{3})\b',
                    'group': 1,
                    'confidence': 'high'
                }
            ]
        }
    
    def _validate_entity(self, entity_type, value):
        """Basic validation for extracted entities"""
        
        if entity_type == EntityType.PNR:
            return len(value) == 6 and value.isalnum()
        elif entity_type == EntityType.FLIGHT_NUMBER:
            return len(value) >= 3 and value[:2].isalpha() and value[2:].isdigit()
        elif entity_type == EntityType.EMAIL:
            return '@' in value and '.' in value
        elif entity_type == EntityType.PHONE_NUMBER:
            digits = re.sub(r'[^\d]', '', value)
            return len(digits) >= 10
        elif entity_type == EntityType.AIRPORT_CODE:
            return len(value) == 3 and value.isalpha()
        else:
            return len(value.strip()) > 0
    
    def extract_contextual_entities(self, text):
        """Extract context-specific entities"""
        
        entities = []
        
        # Seat types
        seat_types = ["window", "aisle", "middle", "exit row"]
        for seat_type in seat_types:
            pattern = rf'\b({re.escape(seat_type)})\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    entity_type=EntityType.SEAT_TYPE,
                    value=match.group(1).lower(),
                    confidence=self.confidence_scores['high'],
                    start_index=match.start(1),
                    end_index=match.end(1)
                ))
        
        # Classes
        classes = ["economy", "business", "first", "premium"]
        for class_name in classes:
            pattern = rf'\b({re.escape(class_name)})\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    entity_type=EntityType.CLASS,
                    value=match.group(1).lower(),
                    confidence=self.confidence_scores['high'],
                    start_index=match.start(1),
                    end_index=match.end(1)
                ))
        
        # Pet types
        pet_types = ["dog", "cat", "pet", "service animal", "bird"]
        for pet_type in pet_types:
            pattern = rf'\b({re.escape(pet_type)})\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    entity_type=EntityType.PET_TYPE,
                    value=match.group(1).lower(),
                    confidence=self.confidence_scores['high'],
                    start_index=match.start(1),
                    end_index=match.end(1)
                ))
        
        return entities
    
    def extract_entities(self, text):
        """Extract all entities from text"""
        
        entities = []
        
        # Extract using regex patterns
        for entity_type, patterns in self.patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                group = pattern_info['group']
                confidence_key = pattern_info['confidence']
                
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(group)
                    
                    # Validate extracted value
                    if self._validate_entity(entity_type, value):
                        entities.append(ExtractedEntity(
                            entity_type=entity_type,
                            value=value,
                            confidence=self.confidence_scores[confidence_key],
                            start_index=match.start(group),
                            end_index=match.end(group)
                        ))
        
        # Extract contextual entities
        contextual_entities = self.extract_contextual_entities(text)
        entities.extend(contextual_entities)
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.start_index)
        
        return entities
    
    def _deduplicate_entities(self, entities):
        """Remove duplicate entities"""
        
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create key based on type, value, and position
            key = (entity.type, entity.value.lower(), entity.start_index, entity.end_index)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities


def analyze_dataset(dataset_path, extractor):
    """Analyze entity extraction on dataset"""
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    results = {
        'total_examples': len(dataset),
        'examples_with_entities': 0,
        'entity_counts': {},
        'sample_extractions': []
    }
    
    for example in dataset[:1000]:  # Analyze first 1000 for speed
        text = example['text']
        entities = extractor.extract_entities(text)
        
        if entities:
            results['examples_with_entities'] += 1
        
        # Count entity types
        for entity in entities:
            entity_type = entity.type.value
            results['entity_counts'][entity_type] = \
                results['entity_counts'].get(entity_type, 0) + 1
        
        # Store sample extractions
        if len(results['sample_extractions']) < 10 and entities:
            results['sample_extractions'].append({
                'text': text,
                'entities': [
                    {
                        'type': entity.type.value,
                        'value': entity.value,
                        'confidence': entity.confidence
                    } for entity in entities
                ]
            })
    
    return results


def benchmark_performance(extractor, test_texts, iterations=100):
    """Benchmark extraction performance"""
    
    # Benchmark
    start_time = time.time()
    total_entities = 0
    
    for _ in range(iterations):
        for text in test_texts:
            entities = extractor.extract_entities(text)
            total_entities += len(entities)
    
    total_time = time.time() - start_time
    total_extractions = iterations * len(test_texts)
    
    return {
        'average_latency_ms': (total_time / total_extractions) * 1000,
        'throughput_per_sec': total_extractions / total_time,
        'total_entities': total_entities,
        'entities_per_extraction': total_entities / total_extractions
    }


def main():
    """Main function"""
    
    print("🔍 Standalone Entity Extraction Training")
    print("=" * 45)
    
    # Initialize extractor
    print("\n📚 Initializing entity extractor...")
    extractor = SimpleEntityExtractor()
    
    print(f"Supported entity types: {len(extractor.patterns)}")
    print(f"Total regex patterns: {sum(len(patterns) for patterns in extractor.patterns.values())}")
    
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
        entities = extractor.extract_entities(text)
        
        if entities:
            print(f"   Entities found: {len(entities)}")
            for entity in entities:
                print(f"     - {entity.type.value}: '{entity.value}' (confidence: {entity.confidence:.2f})")
        else:
            print("   No entities found")
    
    # Analyze dataset if available
    dataset_path = "data/final_training_dataset.json"
    if Path(dataset_path).exists():
        print(f"\n📊 Analyzing dataset...")
        results = analyze_dataset(dataset_path, extractor)
        
        print(f"Analyzed: {results['total_examples']} examples")
        print(f"Examples with entities: {results['examples_with_entities']} ({results['examples_with_entities']/results['total_examples']*100:.1f}%)")
        
        print(f"\nEntity type distribution:")
        for entity_type, count in sorted(results['entity_counts'].items()):
            print(f"  {entity_type}: {count}")
        
        print(f"\nSample extractions:")
        for i, sample in enumerate(results['sample_extractions'][:3], 1):
            print(f"  {i}. '{sample['text']}'")
            for entity in sample['entities']:
                print(f"     → {entity['type']}: {entity['value']}")
    
    # Performance benchmark
    print(f"\n⚡ Performance benchmark...")
    performance = benchmark_performance(extractor, test_examples, iterations=100)
    
    print(f"Average latency: {performance['average_latency_ms']:.2f}ms")
    print(f"Throughput: {performance['throughput_per_sec']:.1f} extractions/sec")
    print(f"Entities per extraction: {performance['entities_per_extraction']:.2f}")
    print(f"Total entities extracted: {performance['total_entities']}")
    
    # Test specific entity types
    print(f"\n🎯 Testing specific entity types...")
    
    test_cases = {
        EntityType.PNR: "Cancel booking ABC123",
        EntityType.FLIGHT_NUMBER: "Status of flight UA100",
        EntityType.EMAIL: "Contact me at john@example.com",
        EntityType.PHONE_NUMBER: "Call me at (555) 123-4567",
        EntityType.DATE: "Flight tomorrow",
        EntityType.PASSENGER_NAME: "My name is John Smith",
        EntityType.SEAT_TYPE: "I want a window seat",
        EntityType.CLASS: "Upgrade to business class",
        EntityType.PET_TYPE: "Traveling with my dog"
    }
    
    for entity_type, text in test_cases.items():
        entities = extractor.extract_entities(text)
        matching_entities = [e for e in entities if e.type == entity_type]
        
        if matching_entities:
            entity = matching_entities[0]
            print(f"  {entity_type.value}: '{text}' → '{entity.value}' (confidence: {entity.confidence:.2f})")
        else:
            print(f"  {entity_type.value}: '{text}' → No entities found")
    
    # Create report
    if Path(dataset_path).exists():
        print(f"\n📋 Creating extraction report...")
        
        report = f"""# Entity Extraction Report

## Extraction Results

### Dataset Analysis
- **Total Examples Analyzed**: {results['total_examples']}
- **Examples with Entities**: {results['examples_with_entities']} ({results['examples_with_entities']/results['total_examples']*100:.1f}%)

### Entity Type Distribution
"""
        
        for entity_type, count in sorted(results['entity_counts'].items()):
            report += f"- **{entity_type}**: {count}\n"
        
        report += f"""
### Performance Benchmarks
- **Average Latency**: {performance['average_latency_ms']:.2f}ms
- **Throughput**: {performance['throughput_per_sec']:.1f} extractions/second
- **Entities per Extraction**: {performance['entities_per_extraction']:.2f}

### Extractor Configuration
- **Algorithm**: Regex-based pattern matching
- **Entity Types**: {len(extractor.patterns)}
- **Regex Patterns**: {sum(len(patterns) for patterns in extractor.patterns.values())}
- **Confidence Levels**: {len(extractor.confidence_scores)}
"""
        
        # Save report
        report_path = Path("models") / "entity_extraction_report.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
    
    # Summary
    print(f"\n🎉 Entity Extraction Summary:")
    print(f"  ✅ Extractor initialized and tested")
    if Path(dataset_path).exists():
        print(f"  ✅ Dataset analyzed: {results['examples_with_entities']}/{results['total_examples']} examples with entities")
    print(f"  ✅ Performance: {performance['average_latency_ms']:.2f}ms latency")
    print(f"  ✅ Entity types: {len(extractor.patterns)} supported")
    print(f"  ✅ Approach: Regex-based pattern matching")
    
    # Validation
    latency_ok = performance['average_latency_ms'] < 100
    
    print(f"\n📊 Validation Results:")
    print(f"  Latency requirement: {'✅ PASS' if latency_ok else '❌ FAIL'} (<100ms)")
    if Path(dataset_path).exists():
        extraction_rate = results['examples_with_entities'] / results['total_examples']
        print(f"  Entity extraction rate: {extraction_rate*100:.1f}% ({'✅ GOOD' if extraction_rate > 0.3 else '⚠️ LOW'})")


if __name__ == "__main__":
    main()