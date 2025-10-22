"""
Improved entity extractor with better validation and precision
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..types import ExtractedEntity, EntityType
from ..utils.validators import (
    validate_pnr, validate_flight_number, validate_email, 
    validate_phone, validate_airport_code
)


class ImprovedEntityExtractor:
    """Improved entity extractor with better precision and validation"""
    
    def __init__(self):
        self.patterns = self._build_improved_patterns()
        self.confidence_scores = {
            'very_high': 0.95,
            'high': 0.85,
            'medium': 0.70,
            'low': 0.55
        }
        
        # Known airport codes for validation
        self.known_airports = {
            "JFK", "LAX", "ORD", "DFW", "DEN", "ATL", "SFO", "SEA", "LAS", "PHX",
            "IAH", "CLT", "MIA", "BOS", "MSP", "DTW", "PHL", "LGA", "FLL", "BWI",
            "MDW", "TPA", "SAN", "STL", "HNL", "PDX", "SLC", "RDU", "AUS", "BNA"
        }
        
        # Common first and last names for validation
        self.common_names = {
            "first": {"john", "jane", "michael", "sarah", "david", "lisa", "robert", "mary", 
                     "james", "jennifer", "william", "patricia", "richard", "linda"},
            "last": {"smith", "johnson", "williams", "brown", "jones", "garcia", "miller", 
                    "davis", "rodriguez", "martinez", "hernandez", "lopez", "wilson"}
        }
    
    def _build_improved_patterns(self) -> Dict[EntityType, List[Dict]]:
        """Build improved regex patterns with better precision"""
        
        return {
            EntityType.PNR: [
                {
                    'pattern': r'\b(?:PNR|booking|confirmation|reference)\s*:?\s*([A-Z0-9]{6})\b',
                    'group': 1,
                    'confidence': 'very_high',
                    'context_required': True
                },
                {
                    'pattern': r'\bbooking\s+([A-Z0-9]{6})\b',
                    'group': 1,
                    'confidence': 'high',
                    'context_required': True
                }
            ],
            
            EntityType.FLIGHT_NUMBER: [
                {
                    'pattern': r'\b(?:flight|flt)\s*:?\s*([A-Z]{2,3}[0-9]{1,4})\b',
                    'group': 1,
                    'confidence': 'very_high',
                    'context_required': True
                },
                {
                    'pattern': r'\b([A-Z]{2}[0-9]{1,4})\b(?=\s*(?:on|to|from|is|was|will))',
                    'group': 1,
                    'confidence': 'high',
                    'context_required': False
                }
            ],
            
            EntityType.DATE: [
                {
                    'pattern': r'\b(today|tomorrow|yesterday)\b',
                    'group': 1,
                    'confidence': 'very_high',
                    'context_required': False
                },
                {
                    'pattern': r'\b(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b',
                    'group': 1,
                    'confidence': 'very_high',
                    'context_required': False
                },
                {
                    'pattern': r'\b((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?)\b',
                    'group': 1,
                    'confidence': 'high',
                    'context_required': False
                },
                {
                    'pattern': r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
                    'group': 1,
                    'confidence': 'high',
                    'context_required': False
                }
            ],
            
            EntityType.AIRPORT_CODE: [
                {
                    'pattern': r'\b(?:to|from|via|airport)\s+([A-Z]{3})\b',
                    'group': 1,
                    'confidence': 'very_high',
                    'context_required': True
                },
                {
                    'pattern': r'\b([A-Z]{3})\s+airport\b',
                    'group': 1,
                    'confidence': 'very_high',
                    'context_required': True
                }
            ],
            
            EntityType.EMAIL: [
                {
                    'pattern': r'\b(?:email|e-mail)\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                    'group': 1,
                    'confidence': 'very_high',
                    'context_required': True
                },
                {
                    'pattern': r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                    'group': 1,
                    'confidence': 'high',
                    'context_required': False
                }
            ],
            
            EntityType.PHONE_NUMBER: [
                {
                    'pattern': r'\b(?:phone|call|number)\s*:?\s*(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b',
                    'group': 1,
                    'confidence': 'very_high',
                    'context_required': True
                },
                {
                    'pattern': r'\b(\(?[0-9]{3}\)?[-.\s][0-9]{3}[-.\s][0-9]{4})\b',
                    'group': 1,
                    'confidence': 'high',
                    'context_required': False
                }
            ],
            
            EntityType.PASSENGER_NAME: [
                {
                    'pattern': r'\bmy name is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                    'group': 1,
                    'confidence': 'very_high',
                    'context_required': True
                },
                {
                    'pattern': r'\bI am\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                    'group': 1,
                    'confidence': 'high',
                    'context_required': True
                }
            ]
        }
    
    def _validate_entity_strict(self, entity_type: EntityType, value: str, text: str) -> bool:
        """Strict validation for extracted entities"""
        
        try:
            if entity_type == EntityType.PNR:
                return validate_pnr(value) and len(value) == 6
            
            elif entity_type == EntityType.FLIGHT_NUMBER:
                return validate_flight_number(value) and len(value) >= 3 and len(value) <= 7
            
            elif entity_type == EntityType.EMAIL:
                return validate_email(value)
            
            elif entity_type == EntityType.PHONE_NUMBER:
                return validate_phone(value)
            
            elif entity_type == EntityType.AIRPORT_CODE:
                return (validate_airport_code(value) and 
                       value.upper() in self.known_airports)
            
            elif entity_type == EntityType.PASSENGER_NAME:
                # Validate passenger name
                parts = value.split()
                if len(parts) != 2:
                    return False
                
                first_name, last_name = parts
                first_name_lower = first_name.lower()
                last_name_lower = last_name.lower()
                
                # Check if it's a common name combination
                return (first_name_lower in self.common_names["first"] or
                       last_name_lower in self.common_names["last"])
            
            elif entity_type == EntityType.DATE:
                # Basic date validation
                return len(value.strip()) > 2
            
            else:
                return len(value.strip()) > 0
                
        except:
            return False
    
    def extract_contextual_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract context-specific entities with improved precision"""
        
        entities = []
        
        # Seat types - only extract when in seat context
        seat_context_pattern = r'\b(?:seat|seats|seating)\b'
        if re.search(seat_context_pattern, text, re.IGNORECASE):
            seat_types = ["window", "aisle", "middle", "exit row", "bulkhead"]
            for seat_type in seat_types:
                pattern = rf'\b({re.escape(seat_type)})\s*(?:seat|seats)?\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(ExtractedEntity(
                        type=EntityType.SEAT_TYPE,
                        value=match.group(1).lower(),
                        confidence=self.confidence_scores['high'],
                        start_index=match.start(1),
                        end_index=match.end(1)
                    ))
        
        # Classes - only extract when in class context
        class_context_pattern = r'\b(?:class|cabin|upgrade|fare)\b'
        if re.search(class_context_pattern, text, re.IGNORECASE):
            classes = ["economy", "business", "first", "premium"]
            for class_name in classes:
                pattern = rf'\b({re.escape(class_name)})\s*(?:class|cabin)?\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(ExtractedEntity(
                        type=EntityType.CLASS,
                        value=match.group(1).lower(),
                        confidence=self.confidence_scores['high'],
                        start_index=match.start(1),
                        end_index=match.end(1)
                    ))
        
        # Pet types - only extract when in pet context
        pet_context_pattern = r'\b(?:pet|animal|dog|cat|bring|travel)\b'
        if re.search(pet_context_pattern, text, re.IGNORECASE):
            pet_types = ["dog", "cat", "service animal", "emotional support animal", "bird", "rabbit"]
            for pet_type in pet_types:
                pattern = rf'\b({re.escape(pet_type)})\b'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append(ExtractedEntity(
                        type=EntityType.PET_TYPE,
                        value=match.group(1).lower(),
                        confidence=self.confidence_scores['high'],
                        start_index=match.start(1),
                        end_index=match.end(1)
                    ))
        
        return entities
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities with improved precision"""
        
        entities = []
        
        # Extract using improved regex patterns
        for entity_type, patterns in self.patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                group = pattern_info['group']
                confidence_key = pattern_info['confidence']
                context_required = pattern_info.get('context_required', False)
                
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(group)
                    
                    # Strict validation
                    if self._validate_entity_strict(entity_type, value, text):
                        entities.append(ExtractedEntity(
                            type=entity_type,
                            value=value,
                            confidence=self.confidence_scores[confidence_key],
                            start_index=match.start(group),
                            end_index=match.end(group)
                        ))
        
        # Extract contextual entities
        contextual_entities = self.extract_contextual_entities(text)
        entities.extend(contextual_entities)
        
        # Remove duplicates and overlaps
        entities = self._remove_overlaps(entities)
        
        # Sort by position
        entities.sort(key=lambda x: x.start_index)
        
        return entities
    
    def _remove_overlaps(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove overlapping entities, keeping the one with higher confidence"""
        
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_index)
        
        filtered = []
        
        for entity in entities:
            # Check if this entity overlaps with any existing filtered entity
            overlaps = False
            
            for i, existing in enumerate(filtered):
                if self._entities_overlap(entity, existing):
                    overlaps = True
                    
                    # Keep the entity with higher confidence
                    if entity.confidence > existing.confidence:
                        filtered[i] = entity
                    
                    break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered
    
    def _entities_overlap(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> bool:
        """Check if two entities overlap in text position"""
        
        return not (entity1.end_index <= entity2.start_index or 
                   entity2.end_index <= entity1.start_index)
    
    def extract_specific_entity_type(self, text: str, entity_type: EntityType) -> List[ExtractedEntity]:
        """Extract only specific entity type"""
        
        all_entities = self.extract_entities(text)
        return [entity for entity in all_entities if entity.type == entity_type]
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        
        return {
            'algorithm': 'Improved Regex with Validation',
            'supported_entities': [entity_type.value for entity_type in EntityType],
            'regex_patterns': len(sum([patterns for patterns in self.patterns.values()], [])),
            'confidence_levels': list(self.confidence_scores.keys()),
            'validation': 'Strict validation with known values',
            'context_aware': True
        }