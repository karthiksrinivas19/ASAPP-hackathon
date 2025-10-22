"""
Enhanced identifier extraction service for automatic data retrieval
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..types import (
    ExtractedEntity, EntityType, FlightIdentifiers, 
    BookingContext, CustomerSearchInfo, CustomerIdentifier
)
from ..interfaces.workflow_orchestrator import EnhancedIdentifierExtractorInterface


@dataclass
class ExtractionContext:
    """Context for entity extraction"""
    utterance: str
    confidence_threshold: float = 0.7
    enable_fuzzy_matching: bool = True
    enable_context_inference: bool = True


class EnhancedIdentifierExtractor(EnhancedIdentifierExtractorInterface):
    """Enhanced identifier extraction with advanced pattern matching and context inference"""
    
    # Enhanced regex patterns with confidence scoring
    ENHANCED_PATTERNS = {
        EntityType.PNR: [
            (r'(?:PNR|confirmation|booking|reference|ref)\s*:?\s*([A-Z0-9]{6})', 0.95),
            (r'(?:code|number)\s*:?\s*([A-Z0-9]{6})', 0.85),
            (r'\b[A-Z]{2}[0-9]{4}\b', 0.8),  # Airline code + 4 digits
            (r'\b[0-9]{2}[A-Z]{4}\b', 0.8),  # 2 digits + 4 letters
            (r'\b[A-Z]{3}[0-9]{3}\b', 0.85),  # 3 letters + 3 digits
            (r'\b[0-9]{3}[A-Z]{3}\b', 0.85),  # 3 digits + 3 letters
            (r'\b[A-Z0-9]{6}\b', 0.7),  # Standard 6-character PNR (lower priority)
        ],
        EntityType.FLIGHT_NUMBER: [
            (r'\b[A-Z]{1,3}\s*\d{1,4}\b', 0.9),  # AA123, JB1234
            (r'(?:flight|flt)\s*:?\s*([A-Z]{1,3}\s*\d{1,4})', 0.95),
            (r'\b\d{4}\b', 0.7),  # 4-digit flight numbers (lower confidence)
            (r'(?:number|no|#)\s*:?\s*([A-Z]{1,3}\s*\d{1,4})', 0.9),
        ],
        EntityType.AIRPORT_CODE: [
            (r'\b[A-Z]{3}\b', 0.8),  # 3-letter airport codes
            (r'(?:from|to|via|airport|apt)\s+([A-Z]{3})', 0.9),
            (r'(?:depart|departure|departing)\s+(?:from\s+)?([A-Z]{3})', 0.9),
            (r'(?:arrive|arrival|arriving)\s+(?:at\s+|in\s+)?([A-Z]{3})', 0.9),
        ],
        EntityType.DATE: [
            (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 0.9),  # MM/DD/YYYY
            (r'\b\d{4}-\d{1,2}-\d{1,2}\b', 0.95),  # YYYY-MM-DD (ISO format)
            (r'\b(?:today|tomorrow|yesterday)\b', 0.9),
            (r'\b(?:next|this)\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 0.8),
            (r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b', 0.85),
            (r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', 0.85),
        ],
        EntityType.PHONE_NUMBER: [
            (r'(?:phone|tel|mobile|cell)\s*:?\s*is\s*(\d{3}[-.]?\d{4})', 0.9),  # 7-digit with context
            (r'(?:phone|tel|mobile|cell)\s*:?\s*is\s*(\d{3}[-.]?\d{3}[-.]?\d{4})', 0.95),  # 10-digit with context
            (r'(?:phone|tel|mobile|cell)\s*:?\s*(\d{3}[-.]?\d{4})', 0.9),  # 7-digit with context
            (r'(?:phone|tel|mobile|cell)\s*:?\s*(\d{3}[-.]?\d{3}[-.]?\d{4})', 0.95),  # 10-digit with context
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 0.9),  # Full US phone numbers
            (r'\b\d{3}[-.]?\d{4}\b', 0.8),  # 7-digit phone numbers
            (r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', 0.95),
            (r'\+\d{1,3}\s*\d{3,4}[-.]?\d{3,4}[-.]?\d{4}', 0.9),
        ],
        EntityType.EMAIL: [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 0.95),
            (r'(?:email|e-mail)\s*:?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', 0.98),
        ],
        EntityType.PASSENGER_NAME: [
            (r'(?:my name is|i am|this is)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', 0.9),
            (r'(?:name|passenger)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)', 0.9),
            (r'(?:mr|mrs|ms|miss)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 0.85),
            (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', 0.7),  # First Last (lower priority)
        ]
    }
    
    # Context keywords for intent detection
    CONTEXT_KEYWORDS = {
        "urgency": ["urgent", "emergency", "asap", "immediately", "quickly", "rush"],
        "cancellation": ["cancel", "refund", "return", "get money back", "don't want"],
        "booking_reference": ["booking", "reservation", "confirmation", "pnr", "reference"],
        "flight_reference": ["flight", "trip", "journey", "travel"],
        "customer_service": ["help", "assist", "support", "service", "problem", "issue"],
        "time_sensitive": ["today", "tomorrow", "now", "soon", "this week"]
    }
    
    def __init__(self):
        self.extraction_cache = {}
    
    def extract_all_identifiers(self, utterance: str) -> FlightIdentifiers:
        """Extract all possible flight identifiers from utterance"""
        context = ExtractionContext(utterance=utterance)
        entities = self._extract_entities_with_context(context)
        
        identifiers = FlightIdentifiers()
        
        for entity in entities:
            if entity.type == EntityType.PNR:
                identifiers.pnr = entity.value
            elif entity.type == EntityType.FLIGHT_NUMBER:
                identifiers.flight_number = entity.value
            elif entity.type == EntityType.PASSENGER_NAME:
                identifiers.passenger_name = entity.value
            elif entity.type == EntityType.DATE:
                identifiers.date = self._parse_date_entity(entity.value)
            elif entity.type == EntityType.AIRPORT_CODE:
                # Handle route extraction - skip for now as RouteInfo requires both fields
                # In a full implementation, we'd collect all airport codes and determine route
                pass
        
        return identifiers
    
    def extract_pnr(self, text: str) -> Optional[str]:
        """Extract PNR from text with high confidence"""
        patterns = self.ENHANCED_PATTERNS[EntityType.PNR]
        
        for pattern, confidence in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1) if match.groups() else match.group(0)
                cleaned_value = self._clean_pnr(value)
                
                if self._validate_pnr(cleaned_value) and confidence >= 0.8:
                    return cleaned_value
        
        return None
    
    def extract_flight_number(self, text: str) -> Optional[str]:
        """Extract flight number from text"""
        patterns = self.ENHANCED_PATTERNS[EntityType.FLIGHT_NUMBER]
        
        for pattern, confidence in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1) if match.groups() else match.group(0)
                cleaned_value = self._clean_flight_number(value)
                
                if self._validate_flight_number(cleaned_value) and confidence >= 0.7:
                    return cleaned_value
        
        return None
    
    def extract_route(self, text: str) -> Optional[Dict[str, str]]:
        """Extract route information from text"""
        airport_codes = []
        patterns = self.ENHANCED_PATTERNS[EntityType.AIRPORT_CODE]
        
        for pattern, confidence in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1) if match.groups() else match.group(0)
                cleaned_value = value.upper().strip()
                
                if self._validate_airport_code(cleaned_value) and confidence >= 0.8:
                    airport_codes.append(cleaned_value)
        
        # Remove duplicates while preserving order
        unique_codes = []
        for code in airport_codes:
            if code not in unique_codes:
                unique_codes.append(code)
        
        if len(unique_codes) >= 2:
            return {"from": unique_codes[0], "to": unique_codes[1]}
        elif len(unique_codes) == 1:
            # Try to infer direction from context
            if any(word in text.lower() for word in ["from", "depart", "leaving"]):
                return {"from": unique_codes[0], "to": ""}
            elif any(word in text.lower() for word in ["to", "arrive", "going"]):
                return {"from": "", "to": unique_codes[0]}
        
        return None
    
    def extract_date(self, text: str) -> Optional[str]:
        """Extract date from text"""
        patterns = self.ENHANCED_PATTERNS[EntityType.DATE]
        
        for pattern, confidence in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1) if match.groups() else match.group(0)
                
                if confidence >= 0.7:
                    return value.strip()
        
        return None
    
    def extract_passenger_name(self, text: str) -> Optional[str]:
        """Extract passenger name from text"""
        patterns = self.ENHANCED_PATTERNS[EntityType.PASSENGER_NAME]
        
        for pattern, confidence in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1) if match.groups() else match.group(0)
                cleaned_value = self._clean_passenger_name(value)
                
                if self._validate_passenger_name(cleaned_value) and confidence >= 0.7:
                    return cleaned_value
        
        return None
    
    def extract_booking_context(self, text: str) -> BookingContext:
        """Extract booking context and intent from text"""
        text_lower = text.lower()
        
        # Check for booking intent
        has_booking_intent = any(
            keyword in text_lower 
            for keyword in self.CONTEXT_KEYWORDS["booking_reference"] + self.CONTEXT_KEYWORDS["flight_reference"]
        )
        
        # Check for urgency
        has_urgency = any(
            keyword in text_lower 
            for keyword in self.CONTEXT_KEYWORDS["urgency"] + self.CONTEXT_KEYWORDS["time_sensitive"]
        )
        
        # Check for partial information
        entities = self._extract_entities_with_context(ExtractionContext(utterance=text))
        has_partial_info = len(entities) > 0 and len(entities) < 3  # Some info but not complete
        
        # Generate suggested actions based on context
        suggested_actions = []
        
        if not has_booking_intent:
            suggested_actions.append("Include your PNR or booking reference")
        
        if "cancel" in text_lower and not any(e.type == EntityType.PNR for e in entities):
            suggested_actions.append("Provide your PNR for cancellation")
        
        if "flight" in text_lower and not any(e.type == EntityType.FLIGHT_NUMBER for e in entities):
            suggested_actions.append("Include your flight number")
        
        if has_urgency and not has_booking_intent:
            suggested_actions.append("Call customer service for immediate assistance")
        
        # Always provide at least one suggestion if we have booking intent but missing info
        if has_booking_intent and not suggested_actions:
            suggested_actions.append("Provide additional booking details for faster service")
        
        return BookingContext(
            has_booking_intent=has_booking_intent,
            has_urgency=has_urgency,
            has_partial_info=has_partial_info,
            suggested_actions=suggested_actions
        )
    
    def extract_customer_info(self, utterance: str) -> Dict[str, Any]:
        """Extract customer information from utterance"""
        context = ExtractionContext(utterance=utterance)
        entities = self._extract_entities_with_context(context)
        
        customer_info = {}
        
        for entity in entities:
            if entity.type == EntityType.PHONE_NUMBER:
                customer_info["phone"] = entity.value
            elif entity.type == EntityType.EMAIL:
                customer_info["email"] = entity.value
            elif entity.type == EntityType.PASSENGER_NAME:
                customer_info["name"] = entity.value
        
        return customer_info
    
    def extract_contextual_clues(self, utterance: str) -> Dict[str, Any]:
        """Extract contextual clues from utterance"""
        text_lower = utterance.lower()
        clues = {}
        
        # Urgency detection
        urgency_score = sum(1 for keyword in self.CONTEXT_KEYWORDS["urgency"] if keyword in text_lower)
        time_urgency = sum(1 for keyword in self.CONTEXT_KEYWORDS["time_sensitive"] if keyword in text_lower)
        
        total_urgency = urgency_score + time_urgency
        if total_urgency >= 2:
            clues["urgency"] = "high"
        elif total_urgency >= 1:
            clues["urgency"] = "medium"
        else:
            clues["urgency"] = "low"
        
        # Time frame detection
        if any(word in text_lower for word in ["today", "now", "immediately"]):
            clues["timeframe"] = "today"
        elif any(word in text_lower for word in ["tomorrow", "next day"]):
            clues["timeframe"] = "tomorrow"
        elif any(word in text_lower for word in ["this week", "soon"]):
            clues["timeframe"] = "this_week"
        else:
            clues["timeframe"] = "future"
        
        # Travel type inference
        if any(word in text_lower for word in ["business", "work", "meeting", "conference"]):
            clues["travel_type"] = "business"
        elif any(word in text_lower for word in ["vacation", "holiday", "family", "personal"]):
            clues["travel_type"] = "personal"
        else:
            clues["travel_type"] = "unknown"
        
        # Request type hints
        if any(word in text_lower for word in self.CONTEXT_KEYWORDS["cancellation"]):
            clues["likely_request"] = "cancellation"
        elif any(word in text_lower for word in ["status", "delayed", "on time"]):
            clues["likely_request"] = "flight_status"
        elif any(word in text_lower for word in ["seat", "upgrade", "change seat"]):
            clues["likely_request"] = "seat_availability"
        
        return clues
    
    def infer_booking_intent(self, utterance: str) -> Dict[str, Any]:
        """Infer booking intent from utterance"""
        text_lower = utterance.lower()
        intent_info = {}
        
        # Intent confidence scoring
        cancellation_score = sum(1 for word in self.CONTEXT_KEYWORDS["cancellation"] if word in text_lower)
        booking_score = sum(1 for word in self.CONTEXT_KEYWORDS["booking_reference"] if word in text_lower)
        
        intent_info["has_cancellation_intent"] = cancellation_score > 0
        intent_info["has_booking_reference_intent"] = booking_score > 0
        intent_info["confidence"] = min(1.0, (cancellation_score + booking_score) * 0.3)
        
        # Missing information detection
        entities = self._extract_entities_with_context(ExtractionContext(utterance=utterance))
        entity_types = [e.type for e in entities]
        
        missing_info = []
        if EntityType.PNR not in entity_types and EntityType.FLIGHT_NUMBER not in entity_types:
            missing_info.append("booking_identifier")
        
        if cancellation_score > 0 and EntityType.PNR not in entity_types:
            missing_info.append("pnr_for_cancellation")
        
        intent_info["missing_information"] = missing_info
        
        # Suggested next steps
        next_steps = []
        if missing_info:
            if "booking_identifier" in missing_info:
                next_steps.append("Provide your PNR or flight number")
            if "pnr_for_cancellation" in missing_info:
                next_steps.append("Include your PNR to proceed with cancellation")
        
        intent_info["suggested_next_steps"] = next_steps
        
        return intent_info
    
    # Private helper methods
    
    def _extract_entities_with_context(self, context: ExtractionContext) -> List[ExtractedEntity]:
        """Extract entities with enhanced context awareness"""
        entities = []
        text = context.utterance
        
        for entity_type, patterns in self.ENHANCED_PATTERNS.items():
            for pattern, base_confidence in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)
                    cleaned_value = self._clean_entity_value(value, entity_type)
                    
                    if cleaned_value and self._validate_entity(cleaned_value, entity_type):
                        # Adjust confidence based on context
                        adjusted_confidence = self._adjust_confidence_for_context(
                            base_confidence, entity_type, cleaned_value, text
                        )
                        
                        if adjusted_confidence >= context.confidence_threshold:
                            entity = ExtractedEntity(
                                type=entity_type,
                                value=cleaned_value,
                                confidence=adjusted_confidence,
                                start_index=match.start(),
                                end_index=match.end()
                            )
                            entities.append(entity)
        
        # Remove duplicates and sort by confidence
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        return entities
    
    def _adjust_confidence_for_context(
        self, 
        base_confidence: float, 
        entity_type: EntityType, 
        value: str, 
        text: str
    ) -> float:
        """Adjust confidence based on surrounding context"""
        text_lower = text.lower()
        
        # Boost confidence for entities with context keywords
        if entity_type == EntityType.PNR:
            if any(keyword in text_lower for keyword in ["pnr", "confirmation", "booking", "reference"]):
                base_confidence += 0.1
        
        elif entity_type == EntityType.FLIGHT_NUMBER:
            if any(keyword in text_lower for keyword in ["flight", "flt", "number"]):
                base_confidence += 0.1
        
        elif entity_type == EntityType.PASSENGER_NAME:
            if any(keyword in text_lower for keyword in ["name", "passenger", "mr", "mrs", "ms"]):
                base_confidence += 0.15
        
        # Reduce confidence for ambiguous contexts
        if entity_type == EntityType.DATE and len(value) <= 2:  # Single/double digit numbers
            base_confidence -= 0.2
        
        return min(1.0, max(0.0, base_confidence))
    
    def _clean_entity_value(self, value: str, entity_type: EntityType) -> str:
        """Clean extracted entity value"""
        value = value.strip()
        
        if entity_type == EntityType.PNR:
            return self._clean_pnr(value)
        elif entity_type == EntityType.FLIGHT_NUMBER:
            return self._clean_flight_number(value)
        elif entity_type == EntityType.AIRPORT_CODE:
            return value.upper()
        elif entity_type == EntityType.PHONE_NUMBER:
            # Clean but preserve format for readability
            cleaned = re.sub(r'[^\d]', '', value)
            return cleaned
        elif entity_type == EntityType.EMAIL:
            return value.lower()
        elif entity_type == EntityType.PASSENGER_NAME:
            return self._clean_passenger_name(value)
        
        return value
    
    def _clean_pnr(self, value: str) -> str:
        """Clean PNR value"""
        return value.upper().replace(' ', '').replace('-', '')
    
    def _clean_flight_number(self, value: str) -> str:
        """Clean flight number value"""
        return re.sub(r'\s+', '', value.upper())
    
    def _clean_passenger_name(self, value: str) -> str:
        """Clean passenger name value"""
        # Remove titles and clean up spacing
        value = re.sub(r'\b(?:mr|mrs|ms|miss|dr|prof)\b\.?\s*', '', value, flags=re.IGNORECASE)
        return ' '.join(word.capitalize() for word in value.split())
    
    def _validate_entity(self, value: str, entity_type: EntityType) -> bool:
        """Validate extracted entity"""
        if entity_type == EntityType.PNR:
            return self._validate_pnr(value)
        elif entity_type == EntityType.AIRPORT_CODE:
            return self._validate_airport_code(value)
        elif entity_type == EntityType.PHONE_NUMBER:
            return len(value) >= 7  # Accept 7-digit and 10-digit phone numbers
        elif entity_type == EntityType.EMAIL:
            return '@' in value and '.' in value
        elif entity_type == EntityType.FLIGHT_NUMBER:
            return self._validate_flight_number(value)
        elif entity_type == EntityType.PASSENGER_NAME:
            return self._validate_passenger_name(value)
        
        return True
    
    def _validate_pnr(self, value: str) -> bool:
        """Validate PNR format"""
        if len(value) != 6 or not value.isalnum():
            return False
        
        # Exclude common English words that might match PNR pattern
        common_words = {
            'CANCEL', 'FLIGHT', 'BOOKING', 'TRAVEL', 'TICKET', 'REFUND',
            'STATUS', 'CHANGE', 'UPDATE', 'POLICY', 'PLEASE', 'THANKS'
        }
        
        if value.upper() in common_words:
            return False
        
        # PNR should have at least one digit or be mixed case
        has_digit = any(c.isdigit() for c in value)
        has_letter = any(c.isalpha() for c in value)
        
        return has_digit and has_letter
    
    def _validate_airport_code(self, value: str) -> bool:
        """Validate airport code format"""
        return len(value) == 3 and value.isalpha()
    
    def _validate_flight_number(self, value: str) -> bool:
        """Validate flight number format"""
        return len(value) >= 3 and any(c.isdigit() for c in value) and any(c.isalpha() for c in value)
    
    def _validate_passenger_name(self, value: str) -> bool:
        """Validate passenger name format"""
        parts = value.split()
        return len(parts) >= 2 and all(part.isalpha() for part in parts)
    
    def _parse_date_entity(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        date_str_lower = date_str.lower().strip()
        
        # Handle relative dates
        if date_str_lower == "today":
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str_lower == "tomorrow":
            return (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str_lower == "yesterday":
            return (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Handle "next/this" + day of week
        if "next" in date_str_lower or "this" in date_str_lower:
            # Simplified - in production, use proper date parsing library
            return datetime.now() + timedelta(days=7)
        
        # Try to parse standard date formats
        date_formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m-%d-%Y",
            "%d/%m/%Y",
            "%d-%m-%Y"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.type, entity.value.upper())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities