"""
Hybrid ML-based entity extraction with regex fallbacks
"""

import re
import spacy
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

from ..types import ExtractedEntity, EntityType
from ..utils.validators import (
    validate_pnr, validate_flight_number, validate_email, 
    validate_phone, validate_airport_code
)


class HybridEntityExtractor:
    """Hybrid entity extractor combining ML (spaCy) with regex fallbacks"""
    
    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self.nlp = None
        
        # Initialize spaCy if available and requested
        if use_spacy:
            try:
                import spacy
                # Try to load English model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("Warning: spaCy English model not found. Using regex-only extraction.")
                    self.use_spacy = False
            except ImportError:
                print("Warning: spaCy not available. Using regex-only extraction.")
                self.use_spacy = False
        
        # Regex patterns for fallback extraction
        self.patterns = self._build_patterns()
        
        # Entity confidence scores
        self.confidence_scores = {
            'spacy_high': 0.95,
            'spacy_medium': 0.85,
            'regex_high': 0.90,
            'regex_medium': 0.75,
            'regex_low': 0.60
        }
    
    def _build_patterns(self) -> Dict[EntityType, List[Dict]]:
        """Build regex patterns for entity extraction"""
        
        return {
            EntityType.PNR: [
                {
                    'pattern': r'\b(?:PNR|booking|confirmation|reference)\s*:?\s*([A-Z0-9]{6})\b',
                    'group': 1,
                    'confidence': 'regex_high'
                },
                {
                    'pattern': r'\b([A-Z0-9]{6})\b',
                    'group': 1,
                    'confidence': 'regex_medium'
                }
            ],
            
            EntityType.FLIGHT_NUMBER: [
                {
                    'pattern': r'\b(?:flight|flt)\s*:?\s*([A-Z]{2,3}[0-9]{1,4})\b',
                    'group': 1,
                    'confidence': 'regex_high'
                },
                {
                    'pattern': r'\b([A-Z]{2,3}[0-9]{1,4})\b',
                    'group': 1,
                    'confidence': 'regex_medium'
                }
            ],
            
            EntityType.DATE: [
                {
                    'pattern': r'\b(today|tomorrow|yesterday)\b',
                    'group': 1,
                    'confidence': 'regex_high'
                },
                {
                    'pattern': r'\b(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b',
                    'group': 1,
                    'confidence': 'regex_high'
                },
                {
                    'pattern': r'\b((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?)\b',
                    'group': 1,
                    'confidence': 'regex_high'
                },
                {
                    'pattern': r'\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?)\b',
                    'group': 1,
                    'confidence': 'regex_medium'
                },
                {
                    'pattern': r'\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b',
                    'group': 1,
                    'confidence': 'regex_medium'
                }
            ],
            
            EntityType.AIRPORT_CODE: [
                {
                    'pattern': r'\b(?:from|to|via|airport)\s+([A-Z]{3})\b',
                    'group': 1,
                    'confidence': 'regex_high'
                },
                {
                    'pattern': r'\b([A-Z]{3})\s+(?:airport|to)\b',
                    'group': 1,
                    'confidence': 'regex_high'
                }
            ],
            
            EntityType.EMAIL: [
                {
                    'pattern': r'\b(?:email|e-mail)\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                    'group': 1,
                    'confidence': 'regex_high'
                },
                {
                    'pattern': r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                    'group': 1,
                    'confidence': 'regex_medium'
                }
            ],
            
            EntityType.PHONE_NUMBER: [
                {
                    'pattern': r'\b(?:phone|call|number)\s*:?\s*(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b',
                    'group': 1,
                    'confidence': 'regex_high'
                },
                {
                    'pattern': r'\b(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b',
                    'group': 1,
                    'confidence': 'regex_medium'
                }
            ],
            
            EntityType.PASSENGER_NAME: [
                {
                    'pattern': r'\b(?:name|passenger)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                    'group': 1,
                    'confidence': 'regex_high'
                },
                {
                    'pattern': r'\bmy name is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                    'group': 1,
                    'confidence': 'regex_high'
                },
                {
                    'pattern': r'\bI am\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                    'group': 1,
                    'confidence': 'regex_medium'
                }
            ]
        }
    
    def extract_entities_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER"""
        
        if not self.use_spacy or not self.nlp:
            return []
        
        entities = []
        doc = self.nlp(text)
        
        # Map spaCy entity types to our entity types
        spacy_mapping = {
            'PERSON': EntityType.PASSENGER_NAME,
            'DATE': EntityType.DATE,
            'TIME': EntityType.DATE,
            'ORG': EntityType.AIRPORT_CODE,  # Sometimes airports are recognized as organizations
            'GPE': EntityType.AIRPORT_CODE,  # Geopolitical entities (cities/airports)
        }
        
        for ent in doc.ents:
            if ent.label_ in spacy_mapping:
                entity_type = spacy_mapping[ent.label_]
                
                # Determine confidence based on spaCy's confidence and entity type
                if ent.label_ in ['PERSON', 'DATE']:
                    confidence_key = 'spacy_high'
                else:
                    confidence_key = 'spacy_medium'
                
                entities.append(ExtractedEntity(
                    type=entity_type,
                    value=ent.text,
                    confidence=self.confidence_scores[confidence_key],
                    start_index=ent.start_char,
                    end_index=ent.end_char
                ))
        
        return entities
    
    def extract_entities_regex(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns"""
        
        entities = []
        
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
                            type=entity_type,
                            value=value,
                            confidence=self.confidence_scores[confidence_key],
                            start_index=match.start(group),
                            end_index=match.end(group)
                        ))
        
        return entities
    
    def _validate_entity(self, entity_type: EntityType, value: str) -> bool:
        """Validate extracted entity value"""
        
        try:
            if entity_type == EntityType.PNR:
                return validate_pnr(value)
            elif entity_type == EntityType.FLIGHT_NUMBER:
                return validate_flight_number(value)
            elif entity_type == EntityType.EMAIL:
                return validate_email(value)
            elif entity_type == EntityType.PHONE_NUMBER:
                return validate_phone(value)
            elif entity_type == EntityType.AIRPORT_CODE:
                return validate_airport_code(value)
            else:
                # For other types, basic validation
                return len(value.strip()) > 0
        except:
            return False
    
    def extract_contextual_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract context-specific entities (seat types, classes, pet types)"""
        
        entities = []
        
        # Seat types
        seat_types = ["window", "aisle", "middle", "exit row", "bulkhead"]
        for seat_type in seat_types:
            pattern = rf'\b({re.escape(seat_type)})\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    type=EntityType.SEAT_TYPE,
                    value=match.group(1).lower(),
                    confidence=self.confidence_scores['regex_high'],
                    start_index=match.start(1),
                    end_index=match.end(1)
                ))
        
        # Classes
        classes = ["economy", "business", "first", "premium economy", "premium"]
        for class_name in classes:
            pattern = rf'\b({re.escape(class_name)})\s*class\b|\b({re.escape(class_name)})\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group(1) or match.group(2)
                if value:
                    entities.append(ExtractedEntity(
                        type=EntityType.CLASS,
                        value=value.lower(),
                        confidence=self.confidence_scores['regex_high'],
                        start_index=match.start(),
                        end_index=match.end()
                    ))
        
        # Pet types
        pet_types = ["dog", "cat", "service animal", "emotional support animal", "bird", "rabbit", "pet"]
        for pet_type in pet_types:
            pattern = rf'\b({re.escape(pet_type)})\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    type=EntityType.PET_TYPE,
                    value=match.group(1).lower(),
                    confidence=self.confidence_scores['regex_high'],
                    start_index=match.start(1),
                    end_index=match.end(1)
                ))
        
        return entities
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract all entities using hybrid approach"""
        
        all_entities = []
        
        # 1. Extract using spaCy (if available)
        if self.use_spacy:
            spacy_entities = self.extract_entities_spacy(text)
            all_entities.extend(spacy_entities)
        
        # 2. Extract using regex patterns
        regex_entities = self.extract_entities_regex(text)
        all_entities.extend(regex_entities)
        
        # 3. Extract contextual entities
        contextual_entities = self.extract_contextual_entities(text)
        all_entities.extend(contextual_entities)
        
        # 4. Deduplicate and merge overlapping entities
        merged_entities = self._merge_entities(all_entities)
        
        # 5. Sort by position
        merged_entities.sort(key=lambda x: x.start_index)
        
        return merged_entities
    
    def _merge_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Merge overlapping entities, keeping the one with higher confidence"""
        
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_index)
        
        merged = []
        
        for entity in entities:
            # Check if this entity overlaps with any existing merged entity
            overlaps = False
            
            for i, existing in enumerate(merged):
                if self._entities_overlap(entity, existing):
                    overlaps = True
                    
                    # Keep the entity with higher confidence
                    if entity.confidence > existing.confidence:
                        merged[i] = entity
                    
                    break
            
            if not overlaps:
                merged.append(entity)
        
        return merged
    
    def _entities_overlap(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> bool:
        """Check if two entities overlap in text position"""
        
        return not (entity1.end_index <= entity2.start_index or 
                   entity2.end_index <= entity1.start_index)
    
    def extract_specific_entity_type(self, text: str, entity_type: EntityType) -> List[ExtractedEntity]:
        """Extract only specific entity type"""
        
        all_entities = self.extract_entities(text)
        return [entity for entity in all_entities if entity.type == entity_type]
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics and capabilities"""
        
        return {
            'spacy_available': self.use_spacy,
            'spacy_model': 'en_core_web_sm' if self.use_spacy else None,
            'supported_entities': [entity_type.value for entity_type in EntityType],
            'regex_patterns': len(sum([patterns for patterns in self.patterns.values()], [])),
            'confidence_levels': list(self.confidence_scores.keys())
        }


class EntityExtractionPipeline:
    """Complete entity extraction pipeline with training and evaluation"""
    
    def __init__(self):
        self.extractor = HybridEntityExtractor()
    
    def extract_from_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Extract entities from entire dataset for analysis"""
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        extraction_results = {
            'total_examples': len(dataset),
            'examples_with_entities': 0,
            'entity_counts': {},
            'extraction_errors': 0,
            'sample_extractions': []
        }
        
        for i, example in enumerate(dataset):
            try:
                text = example['text']
                extracted_entities = self.extractor.extract_entities(text)
                
                if extracted_entities:
                    extraction_results['examples_with_entities'] += 1
                
                # Count entity types
                for entity in extracted_entities:
                    entity_type = entity.type.value
                    extraction_results['entity_counts'][entity_type] = \
                        extraction_results['entity_counts'].get(entity_type, 0) + 1
                
                # Store sample extractions
                if len(extraction_results['sample_extractions']) < 10 and extracted_entities:
                    extraction_results['sample_extractions'].append({
                        'text': text,
                        'entities': [
                            {
                                'type': entity.type.value,
                                'value': entity.value,
                                'confidence': entity.confidence
                            } for entity in extracted_entities
                        ]
                    })
            
            except Exception as e:
                extraction_results['extraction_errors'] += 1
        
        return extraction_results
    
    def benchmark_performance(self, test_texts: List[str], iterations: int = 100) -> Dict[str, float]:
        """Benchmark entity extraction performance"""
        
        import time
        
        # Warm up
        for text in test_texts[:5]:
            self.extractor.extract_entities(text)
        
        # Benchmark
        start_time = time.time()
        total_entities = 0
        
        for _ in range(iterations):
            for text in test_texts:
                entities = self.extractor.extract_entities(text)
                total_entities += len(entities)
        
        total_time = time.time() - start_time
        total_extractions = iterations * len(test_texts)
        
        avg_latency = (total_time / total_extractions) * 1000  # ms
        throughput = total_extractions / total_time  # extractions/sec
        
        return {
            'average_latency_ms': avg_latency,
            'throughput_extractions_per_sec': throughput,
            'total_entities_extracted': total_entities,
            'entities_per_extraction': total_entities / total_extractions,
            'total_time_seconds': total_time
        }
    
    def create_extraction_report(self, dataset_results: Dict, performance: Dict) -> str:
        """Create comprehensive extraction report"""
        
        report = f"""
# Entity Extraction Report

## Extraction Results

### Dataset Analysis
- **Total Examples**: {dataset_results['total_examples']}
- **Examples with Entities**: {dataset_results['examples_with_entities']} ({dataset_results['examples_with_entities']/dataset_results['total_examples']*100:.1f}%)
- **Extraction Errors**: {dataset_results['extraction_errors']}

### Entity Type Distribution
"""
        
        for entity_type, count in sorted(dataset_results['entity_counts'].items()):
            report += f"- **{entity_type}**: {count}\n"
        
        report += f"""
### Performance Benchmarks
- **Average Latency**: {performance['average_latency_ms']:.2f}ms
- **Throughput**: {performance['throughput_extractions_per_sec']:.1f} extractions/second
- **Entities per Extraction**: {performance['entities_per_extraction']:.2f}
- **Total Entities Extracted**: {performance['total_entities_extracted']}

### Sample Extractions
"""
        
        for i, sample in enumerate(dataset_results['sample_extractions'][:5], 1):
            report += f"\n**Example {i}:**\n"
            report += f"Text: \"{sample['text']}\"\n"
            report += f"Entities:\n"
            for entity in sample['entities']:
                report += f"  - {entity['type']}: {entity['value']} (confidence: {entity['confidence']:.2f})\n"
        
        # Add extractor stats
        stats = self.extractor.get_extraction_stats()
        report += f"""
### Extractor Configuration
- **spaCy Available**: {stats['spacy_available']}
- **spaCy Model**: {stats['spacy_model']}
- **Supported Entities**: {len(stats['supported_entities'])}
- **Regex Patterns**: {stats['regex_patterns']}
- **Confidence Levels**: {len(stats['confidence_levels'])}
"""
        
        return report