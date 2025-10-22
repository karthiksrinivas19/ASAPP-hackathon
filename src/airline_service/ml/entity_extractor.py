"""
Entity extraction for customer queries
"""

import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from ..types import ExtractedEntity, EntityType
from ..utils.validators import (
    validate_pnr, validate_flight_number, validate_email, 
    validate_phone, validate_airport_code
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EntityExtractor:
    """Extract entities from customer queries using regex and NLP"""
    
    def __init__(self):
        self.pnr_patterns = self._build_pnr_patterns()
        self.flight_patterns = self._build_flight_patterns()
        self.date_patterns = self._build_date_patterns()
        self.airport_patterns = self._build_airport_patterns()
        self.phone_patterns = self._build_phone_patterns()
        self.email_patterns = self._build_email_patterns()
        self.name_patterns = self._build_name_patterns()
    
    def _build_pnr_patterns(self) -> List[Dict[str, Any]]:
        """Build PNR extraction patterns"""
        return [
            {
                "pattern": r'\b(?:PNR|booking|confirmation|reference)\s*:?\s*([A-Z0-9]{6})\b',
                "group": 1,
                "confidence": 0.95
            },
            {
                "pattern": r'\b([A-Z0-9]{6})\b',
                "group": 1,
                "confidence": 0.7
            }
        ]
    
    def _build_flight_patterns(self) -> List[Dict[str, Any]]:
        """Build flight number extraction patterns"""
        return [
            {
                "pattern": r'\b(?:flight|flt)\s*:?\s*([A-Z]{2,3}[0-9]{1,4})\b',
                "group": 1,
                "confidence": 0.95
            },
            {
                "pattern": r'\b([A-Z]{2,3}[0-9]{1,4})\b',
                "group": 1,
                "confidence": 0.8
            }
        ]
    
    def _build_date_patterns(self) -> List[Dict[str, Any]]:
        """Build date extraction patterns"""
        return [
            {
                "pattern": r'\b(today|tomorrow|yesterday)\b',
                "group": 1,
                "confidence": 0.9
            },
            {
                "pattern": r'\b(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b',
                "group": 1,
                "confidence": 0.85
            },
            {
                "pattern": r'\b((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?)\b',
                "group": 1,
                "confidence": 0.9
            },
            {
                "pattern": r'\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?)\b',
                "group": 1,
                "confidence": 0.85
            },
            {
                "pattern": r'\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b',
                "group": 1,
                "confidence": 0.8
            },
            {
                "pattern": r'\b(\d{4}-\d{1,2}-\d{1,2})\b',
                "group": 1,
                "confidence": 0.9
            }
        ]
    
    def _build_airport_patterns(self) -> List[Dict[str, Any]]:
        """Build airport code extraction patterns"""
        return [
            {
                "pattern": r'\b(?:from|to|via)\s+([A-Z]{3})\b',
                "group": 1,
                "confidence": 0.9
            },
            {
                "pattern": r'\b([A-Z]{3})\s+(?:to|airport)\b',
                "group": 1,
                "confidence": 0.85
            },
            {
                "pattern": r'\b([A-Z]{3})\b',
                "group": 1,
                "confidence": 0.6
            }
        ]
    
    def _build_phone_patterns(self) -> List[Dict[str, Any]]:
        """Build phone number extraction patterns"""
        return [
            {
                "pattern": r'\b(?:phone|call|number)\s*:?\s*(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b',
                "group": 1,
                "confidence": 0.95
            },
            {
                "pattern": r'\b(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b',
                "group": 1,
                "confidence": 0.8
            }
        ]
    
    def _build_email_patterns(self) -> List[Dict[str, Any]]:
        """Build email extraction patterns"""
        return [
            {
                "pattern": r'\b(?:email|e-mail)\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                "group": 1,
                "confidence": 0.95
            },
            {
                "pattern": r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
                "group": 1,
                "confidence": 0.9
            }
        ]
    
    def _build_name_patterns(self) -> List[Dict[str, Any]]:
        """Build passenger name extraction patterns"""
        return [
            {
                "pattern": r'\b(?:name|passenger)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                "group": 1,
                "confidence": 0.9
            },
            {
                "pattern": r'\bmy name is\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                "group": 1,
                "confidence": 0.95
            },
            {
                "pattern": r'\bI am\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                "group": 1,
                "confidence": 0.9
            }
        ]
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract all entities from text"""
        entities = []
        text_upper = text.upper()
        
        # Extract PNRs
        entities.extend(self._extract_pnrs(text_upper))
        
        # Extract flight numbers
        entities.extend(self._extract_flight_numbers(text_upper))
        
        # Extract dates
        entities.extend(self._extract_dates(text.lower()))
        
        # Extract airport codes
        entities.extend(self._extract_airport_codes(text_upper))
        
        # Extract phone numbers
        entities.extend(self._extract_phone_numbers(text))
        
        # Extract emails
        entities.extend(self._extract_emails(text.lower()))
        
        # Extract passenger names
        entities.extend(self._extract_passenger_names(text))
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.start_index)
        
        return entities
    
    def _extract_pnrs(self, text: str) -> List[ExtractedEntity]:
        """Extract PNR codes from text"""
        entities = []
        
        for pattern_info in self.pnr_patterns:
            pattern = pattern_info["pattern"]
            group = pattern_info["group"]
            confidence = pattern_info["confidence"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                pnr = match.group(group)
                if validate_pnr(pnr):
                    entities.append(ExtractedEntity(
                        type=EntityType.PNR,
                        value=pnr,
                        confidence=confidence,
                        start_index=match.start(group),
                        end_index=match.end(group)
                    ))
        
        return entities
    
    def _extract_flight_numbers(self, text: str) -> List[ExtractedEntity]:
        """Extract flight numbers from text"""
        entities = []
        
        for pattern_info in self.flight_patterns:
            pattern = pattern_info["pattern"]
            group = pattern_info["group"]
            confidence = pattern_info["confidence"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                flight_number = match.group(group)
                if validate_flight_number(flight_number):
                    entities.append(ExtractedEntity(
                        type=EntityType.FLIGHT_NUMBER,
                        value=flight_number,
                        confidence=confidence,
                        start_index=match.start(group),
                        end_index=match.end(group)
                    ))
        
        return entities
    
    def _extract_dates(self, text: str) -> List[ExtractedEntity]:
        """Extract dates from text"""
        entities = []
        
        for pattern_info in self.date_patterns:
            pattern = pattern_info["pattern"]
            group = pattern_info["group"]
            confidence = pattern_info["confidence"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(group)
                entities.append(ExtractedEntity(
                    type=EntityType.DATE,
                    value=date_str,
                    confidence=confidence,
                    start_index=match.start(group),
                    end_index=match.end(group)
                ))
        
        return entities
    
    def _extract_airport_codes(self, text: str) -> List[ExtractedEntity]:
        """Extract airport codes from text"""
        entities = []
        
        for pattern_info in self.airport_patterns:
            pattern = pattern_info["pattern"]
            group = pattern_info["group"]
            confidence = pattern_info["confidence"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                airport_code = match.group(group)
                if validate_airport_code(airport_code):
                    entities.append(ExtractedEntity(
                        type=EntityType.AIRPORT_CODE,
                        value=airport_code,
                        confidence=confidence,
                        start_index=match.start(group),
                        end_index=match.end(group)
                    ))
        
        return entities
    
    def _extract_phone_numbers(self, text: str) -> List[ExtractedEntity]:
        """Extract phone numbers from text"""
        entities = []
        
        for pattern_info in self.phone_patterns:
            pattern = pattern_info["pattern"]
            group = pattern_info["group"]
            confidence = pattern_info["confidence"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phone = match.group(group)
                if validate_phone(phone):
                    entities.append(ExtractedEntity(
                        type=EntityType.PHONE_NUMBER,
                        value=phone,
                        confidence=confidence,
                        start_index=match.start(group),
                        end_index=match.end(group)
                    ))
        
        return entities
    
    def _extract_emails(self, text: str) -> List[ExtractedEntity]:
        """Extract email addresses from text"""
        entities = []
        
        for pattern_info in self.email_patterns:
            pattern = pattern_info["pattern"]
            group = pattern_info["group"]
            confidence = pattern_info["confidence"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                email = match.group(group)
                if validate_email(email):
                    entities.append(ExtractedEntity(
                        type=EntityType.EMAIL,
                        value=email.lower(),
                        confidence=confidence,
                        start_index=match.start(group),
                        end_index=match.end(group)
                    ))
        
        return entities
    
    def _extract_passenger_names(self, text: str) -> List[ExtractedEntity]:
        """Extract passenger names from text"""
        entities = []
        
        for pattern_info in self.name_patterns:
            pattern = pattern_info["pattern"]
            group = pattern_info["group"]
            confidence = pattern_info["confidence"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(group)
                entities.append(ExtractedEntity(
                    type=EntityType.PASSENGER_NAME,
                    value=name.title(),
                    confidence=confidence,
                    start_index=match.start(group),
                    end_index=match.end(group)
                ))
        
        return entities
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create a key based on type, value, and position
            key = (entity.type, entity.value.lower(), entity.start_index, entity.end_index)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_specific_entity(self, text: str, entity_type: EntityType) -> Optional[ExtractedEntity]:
        """Extract specific entity type from text"""
        entities = self.extract_entities(text)
        
        # Find the entity with highest confidence for the requested type
        matching_entities = [e for e in entities if e.type == entity_type]
        if matching_entities:
            return max(matching_entities, key=lambda x: x.confidence)
        
        return None
    
    def has_entity_type(self, text: str, entity_type: EntityType) -> bool:
        """Check if text contains specific entity type"""
        return self.extract_specific_entity(text, entity_type) is not None