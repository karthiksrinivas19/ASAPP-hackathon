"""
Data augmentation for training dataset enhancement
"""

import random
import re
from typing import List, Dict, Set
from ..types import TrainingExample, RequestType, ExtractedEntity, EntityType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataAugmentation:
    """Data augmentation techniques for training examples"""
    
    def __init__(self):
        self.synonym_dict = self._build_synonym_dict()
        self.typo_patterns = self._build_typo_patterns()
        self.paraphrase_patterns = self._build_paraphrase_patterns()
    
    def _build_synonym_dict(self) -> Dict[str, List[str]]:
        """Build dictionary of synonyms for common words"""
        return {
            # Action words
            "cancel": ["cancel", "terminate", "abort", "stop", "end", "void"],
            "check": ["check", "verify", "look up", "see", "view", "get"],
            "show": ["show", "display", "list", "give me", "provide"],
            "want": ["want", "need", "would like", "wish to", "require"],
            
            # Flight related
            "flight": ["flight", "trip", "journey", "travel"],
            "booking": ["booking", "reservation", "ticket", "confirmation"],
            "seat": ["seat", "chair", "place", "spot"],
            "status": ["status", "info", "information", "details", "update"],
            
            # Time related
            "today": ["today", "now", "currently", "at present"],
            "tomorrow": ["tomorrow", "next day", "the following day"],
            
            # Politeness
            "please": ["please", "kindly", "could you", "would you"],
            "help": ["help", "assist", "support", "aid"],
            
            # Pet related
            "pet": ["pet", "animal", "companion"],
            "dog": ["dog", "canine", "puppy"],
            "cat": ["cat", "feline", "kitten"],
            
            # Policy related
            "policy": ["policy", "rules", "regulations", "terms", "conditions"],
            "refund": ["refund", "money back", "reimbursement", "return"],
            "fee": ["fee", "charge", "cost", "price"]
        }
    
    def _build_typo_patterns(self) -> Dict[str, List[str]]:
        """Build common typo patterns"""
        return {
            "cancel": ["cancle", "cancell", "cansel"],
            "flight": ["flght", "fligt", "flihgt"],
            "booking": ["boking", "bookng", "bookin"],
            "status": ["staus", "stauts", "sttus"],
            "available": ["availabe", "availble", "avaliable"],
            "policy": ["polcy", "policey", "poilcy"],
            "refund": ["refund", "refudn", "refnd"]
        }
    
    def _build_paraphrase_patterns(self) -> Dict[RequestType, List[str]]:
        """Build paraphrase patterns for each intent"""
        return {
            RequestType.CANCEL_TRIP: [
                "I {want} to {cancel} my {flight}",
                "{Cancel} my {booking}",
                "I {need} to {cancel} my {trip}",
                "{Please} {cancel} my reservation",
                "I can't travel anymore",
                "I {want} a {refund}",
                "I {need} to get my money back"
            ],
            RequestType.FLIGHT_STATUS: [
                "What's my {flight} {status}",
                "{Check} my {flight} {info}",
                "Is my {flight} on time",
                "When does my {flight} depart",
                "{Show} me my {flight} details",
                "Has my {flight} been delayed"
            ],
            RequestType.SEAT_AVAILABILITY: [
                "{Show} available {seats}",
                "What {seats} are free",
                "I {want} to change my {seat}",
                "{Check} {seat} availability",
                "Can I upgrade my {seat}",
                "{Show} me the {seat} map"
            ],
            RequestType.CANCELLATION_POLICY: [
                "What's your cancellation {policy}",
                "Can I get a {refund}",
                "What are the cancellation {rules}",
                "How much does it cost to {cancel}",
                "What's the {refund} {policy}",
                "Cancellation {fees} information"
            ],
            RequestType.PET_TRAVEL: [
                "Can I bring my {pet}",
                "{Pet} travel {policy}",
                "Flying with my {pet}",
                "What are the {rules} for {pets}",
                "{Pet} carrier requirements",
                "{Pet} travel {fees}"
            ]
        }
    
    def paraphrase_examples(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Generate paraphrased versions of examples"""
        augmented = []
        
        for example in examples:
            # Original example
            augmented.append(example)
            
            # Generate paraphrases
            paraphrases = self._generate_paraphrases(example.text, example.intent)
            for paraphrase in paraphrases[:2]:  # Limit to 2 paraphrases per example
                augmented.append(TrainingExample(
                    text=paraphrase,
                    intent=example.intent,
                    entities=example.entities  # Keep original entities for simplicity
                ))
        
        return augmented
    
    def _generate_paraphrases(self, text: str, intent: RequestType) -> List[str]:
        """Generate paraphrases for a given text"""
        paraphrases = []
        
        # Synonym replacement
        synonym_version = self._replace_with_synonyms(text)
        if synonym_version != text:
            paraphrases.append(synonym_version)
        
        # Pattern-based paraphrasing
        if intent in self.paraphrase_patterns:
            pattern_version = self._apply_paraphrase_patterns(text, intent)
            if pattern_version and pattern_version != text:
                paraphrases.append(pattern_version)
        
        return paraphrases
    
    def _replace_with_synonyms(self, text: str) -> str:
        """Replace words with synonyms"""
        words = text.lower().split()
        new_words = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word in self.synonym_dict and random.random() < 0.3:
                # 30% chance to replace with synonym
                synonym = random.choice(self.synonym_dict[clean_word])
                # Preserve original punctuation
                new_word = word.replace(clean_word, synonym)
                new_words.append(new_word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def _apply_paraphrase_patterns(self, text: str, intent: RequestType) -> str:
        """Apply paraphrase patterns for specific intent"""
        patterns = self.paraphrase_patterns.get(intent, [])
        if not patterns:
            return text
        
        # Simple pattern matching and replacement
        pattern = random.choice(patterns)
        
        # Replace placeholders with synonyms
        for word, synonyms in self.synonym_dict.items():
            placeholder = f"{{{word}}}"
            if placeholder in pattern:
                synonym = random.choice(synonyms)
                pattern = pattern.replace(placeholder, synonym)
        
        return pattern
    
    def add_typos(self, examples: List[TrainingExample], typo_rate: float = 0.1) -> List[TrainingExample]:
        """Add typos to examples"""
        augmented = []
        
        for example in examples:
            augmented.append(example)
            
            if random.random() < typo_rate:
                typo_text = self._add_typos_to_text(example.text)
                if typo_text != example.text:
                    augmented.append(TrainingExample(
                        text=typo_text,
                        intent=example.intent,
                        entities=example.entities
                    ))
        
        return augmented
    
    def _add_typos_to_text(self, text: str) -> str:
        """Add typos to text"""
        words = text.split()
        new_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word in self.typo_patterns and random.random() < 0.5:
                typo = random.choice(self.typo_patterns[clean_word])
                new_word = word.replace(clean_word, typo)
                new_words.append(new_word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def add_contextual_variations(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Add contextual variations to examples"""
        augmented = []
        
        context_prefixes = [
            "Hi, ", "Hello, ", "Excuse me, ", "I need help. ",
            "Can you help me? ", "I have a question. "
        ]
        
        context_suffixes = [
            " please", " thank you", " thanks", " ASAP", 
            " urgently", " immediately", " if possible"
        ]
        
        for example in examples:
            augmented.append(example)
            
            # Add prefix variation
            if random.random() < 0.2:
                prefix = random.choice(context_prefixes)
                new_text = prefix + example.text
                augmented.append(TrainingExample(
                    text=new_text,
                    intent=example.intent,
                    entities=example.entities
                ))
            
            # Add suffix variation
            if random.random() < 0.2:
                suffix = random.choice(context_suffixes)
                new_text = example.text + suffix
                augmented.append(TrainingExample(
                    text=new_text,
                    intent=example.intent,
                    entities=example.entities
                ))
        
        return augmented
    
    def augment_dataset(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Apply all augmentation techniques"""
        logger.info(f"Starting augmentation with {len(examples)} examples")
        
        # Apply paraphrasing
        augmented = self.paraphrase_examples(examples)
        logger.info(f"After paraphrasing: {len(augmented)} examples")
        
        # Add typos
        augmented = self.add_typos(augmented, typo_rate=0.05)
        logger.info(f"After adding typos: {len(augmented)} examples")
        
        # Add contextual variations
        augmented = self.add_contextual_variations(augmented)
        logger.info(f"After contextual variations: {len(augmented)} examples")
        
        # Shuffle the augmented dataset
        random.shuffle(augmented)
        
        return augmented


class EntityVariation:
    """Generate variations of entities for training"""
    
    def __init__(self):
        self.pnr_formats = self._generate_pnr_formats()
        self.flight_formats = self._generate_flight_formats()
        self.date_formats = self._generate_date_formats()
    
    def _generate_pnr_formats(self) -> List[str]:
        """Generate different PNR format patterns"""
        formats = []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        
        # Different patterns for PNRs
        for _ in range(50):
            # 3 letters + 3 digits
            pnr = ''.join(random.choices(letters, k=3)) + ''.join(random.choices(digits, k=3))
            formats.append(pnr)
            
            # 2 letters + 4 digits
            pnr = ''.join(random.choices(letters, k=2)) + ''.join(random.choices(digits, k=4))
            formats.append(pnr)
            
            # 4 letters + 2 digits
            pnr = ''.join(random.choices(letters, k=4)) + ''.join(random.choices(digits, k=2))
            formats.append(pnr)
        
        return formats
    
    def _generate_flight_formats(self) -> List[str]:
        """Generate different flight number formats"""
        airlines = [
            "AA", "UA", "DL", "JB", "SW", "NK", "F9", "B6", "AS", "WN",
            "AC", "BA", "LH", "AF", "KL", "IB", "AZ", "OS", "LX", "SK"
        ]
        
        formats = []
        for airline in airlines:
            for i in range(1, 100):
                # Standard format: AA123
                formats.append(f"{airline}{i}")
                # With leading zeros: AA0123
                formats.append(f"{airline}{i:04d}")
        
        return formats[:200]
    
    def _generate_date_formats(self) -> List[str]:
        """Generate various date format patterns"""
        return [
            "today", "tomorrow", "yesterday", "next week", "next month",
            "this weekend", "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday", "next Monday", "this Friday",
            "January 15", "Jan 15", "15 Jan", "15th January",
            "01/15", "1/15", "01-15", "1-15-2024", "2024-01-15",
            "15/01/2024", "15.01.2024", "Jan 15th", "January 15th"
        ]
    
    def generate_entity_variations(self, entity_type: EntityType, count: int = 100) -> List[str]:
        """Generate variations for specific entity type"""
        if entity_type == EntityType.PNR:
            return random.sample(self.pnr_formats, min(count, len(self.pnr_formats)))
        elif entity_type == EntityType.FLIGHT_NUMBER:
            return random.sample(self.flight_formats, min(count, len(self.flight_formats)))
        elif entity_type == EntityType.DATE:
            return random.sample(self.date_formats, min(count, len(self.date_formats)))
        else:
            return []