"""
Training dataset generation for request classification
"""

import random
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from ..types import RequestType, TrainingExample, ExtractedEntity, EntityType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatasetGenerator:
    """Generate synthetic training data for request classification"""
    
    def __init__(self):
        self.pnr_patterns = self._generate_pnr_patterns()
        self.flight_numbers = self._generate_flight_numbers()
        self.airport_codes = self._generate_airport_codes()
        self.dates = self._generate_date_variations()
        self.names = self._generate_names()
        self.phone_numbers = self._generate_phone_numbers()
        self.emails = self._generate_emails()
    
    def _generate_pnr_patterns(self) -> List[str]:
        """Generate realistic PNR patterns"""
        pnrs = []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        
        for _ in range(100):
            # 6-character alphanumeric PNR
            pnr = ''.join(random.choices(letters + digits, k=6))
            pnrs.append(pnr)
        
        return pnrs
    
    def _generate_flight_numbers(self) -> List[str]:
        """Generate realistic flight numbers"""
        airlines = ["AA", "UA", "DL", "JB", "SW", "NK", "F9", "B6", "AS", "WN"]
        flights = []
        
        for airline in airlines:
            for num in range(1, 9999, 100):
                flights.append(f"{airline}{num}")
                if len(flights) >= 200:
                    break
        
        return flights[:200]
    
    def _generate_airport_codes(self) -> List[str]:
        """Generate common airport codes"""
        return [
            "JFK", "LAX", "ORD", "DFW", "DEN", "ATL", "SFO", "SEA", "LAS", "PHX",
            "IAH", "CLT", "MIA", "BOS", "MSP", "DTW", "PHL", "LGA", "FLL", "BWI",
            "MDW", "TPA", "SAN", "STL", "HNL", "PDX", "SLC", "RDU", "AUS", "BNA"
        ]
    
    def _generate_date_variations(self) -> List[str]:
        """Generate various date formats"""
        base_date = datetime.now()
        variations = []
        
        # Relative dates
        relative_dates = [
            "today", "tomorrow", "yesterday", "next week", "next month",
            "this weekend", "Monday", "Tuesday", "Wednesday", "Thursday", 
            "Friday", "Saturday", "Sunday"
        ]
        variations.extend(relative_dates)
        
        # Specific dates
        for i in range(30):
            date = base_date + timedelta(days=i)
            variations.extend([
                date.strftime("%B %d"),
                date.strftime("%b %d"),
                date.strftime("%m/%d"),
                date.strftime("%m-%d-%Y"),
                date.strftime("%Y-%m-%d")
            ])
        
        return variations[:50]
    
    def _generate_names(self) -> List[str]:
        """Generate common passenger names"""
        first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Mary",
            "James", "Jennifer", "William", "Patricia", "Richard", "Linda", "Joseph",
            "Elizabeth", "Thomas", "Barbara", "Christopher", "Susan", "Daniel", "Jessica"
        ]
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
        ]
        
        names = []
        for first in first_names[:15]:
            for last in last_names[:15]:
                names.append(f"{first} {last}")
        
        return names[:50]
    
    def _generate_phone_numbers(self) -> List[str]:
        """Generate phone number variations"""
        phones = []
        for _ in range(20):
            area = random.randint(200, 999)
            exchange = random.randint(200, 999)
            number = random.randint(1000, 9999)
            
            phones.extend([
                f"{area}{exchange}{number}",
                f"({area}) {exchange}-{number}",
                f"{area}-{exchange}-{number}",
                f"+1{area}{exchange}{number}",
                f"1-{area}-{exchange}-{number}"
            ])
        
        return phones[:30]
    
    def _generate_emails(self) -> List[str]:
        """Generate email variations"""
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
        names = ["john", "jane", "mike", "sarah", "david", "lisa", "bob", "mary"]
        
        emails = []
        for name in names:
            for domain in domains:
                emails.extend([
                    f"{name}@{domain}",
                    f"{name}.smith@{domain}",
                    f"{name}123@{domain}",
                    f"{name}.doe@{domain}"
                ])
        
        return emails[:40]


class TrainingDatasetBuilder:
    """Build comprehensive training dataset for all request types"""
    
    def __init__(self):
        self.generator = DatasetGenerator()
        self.logger = get_logger(__name__)
    
    def generate_cancel_trip_examples(self, count: int = 2000) -> List[TrainingExample]:
        """Generate cancel trip training examples"""
        examples = []
        
        # Base templates for cancel trip intent
        templates = [
            "I want to cancel my flight",
            "Cancel my booking",
            "I need to cancel my trip",
            "Please cancel my reservation",
            "Cancel flight {flight_number}",
            "I want to cancel booking {pnr}",
            "Cancel my flight {pnr}",
            "I need to get a refund for my booking",
            "Cancel my reservation {pnr}",
            "I can't travel anymore, cancel my flight",
            "Emergency cancellation needed for booking {pnr}",
            "Refund my ticket",
            "I want to cancel and get my money back",
            "Cancel my trip to {destination}",
            "I need to cancel my flight on {date}",
            "Cancel my {date} flight",
            "Please cancel my booking for {flight_number}",
            "I want to cancel my flight from {source} to {destination}",
            "Cancel reservation for {name}",
            "I need to cancel my booking urgently"
        ]
        
        # Variations and modifiers
        urgency_modifiers = ["urgently", "immediately", "as soon as possible", "right now", ""]
        politeness = ["Please", "Could you", "Can you", "I would like to", ""]
        
        for i in range(count):
            template = random.choice(templates)
            
            # Add entities based on template
            entities = []
            text = template
            
            if "{pnr}" in template:
                pnr = random.choice(self.generator.pnr_patterns)
                text = text.replace("{pnr}", pnr)
                start_idx = text.find(pnr)
                entities.append(ExtractedEntity(
                    type=EntityType.PNR,
                    value=pnr,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(pnr)
                ))
            
            if "{flight_number}" in template:
                flight = random.choice(self.generator.flight_numbers)
                text = text.replace("{flight_number}", flight)
                start_idx = text.find(flight)
                entities.append(ExtractedEntity(
                    type=EntityType.FLIGHT_NUMBER,
                    value=flight,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(flight)
                ))
            
            if "{destination}" in template:
                dest = random.choice(self.generator.airport_codes)
                text = text.replace("{destination}", dest)
                start_idx = text.find(dest)
                entities.append(ExtractedEntity(
                    type=EntityType.DESTINATION,
                    value=dest,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(dest)
                ))
            
            if "{source}" in template:
                source = random.choice(self.generator.airport_codes)
                text = text.replace("{source}", source)
            
            if "{date}" in template:
                date = random.choice(self.generator.dates)
                text = text.replace("{date}", date)
                start_idx = text.find(date)
                entities.append(ExtractedEntity(
                    type=EntityType.DATE,
                    value=date,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(date)
                ))
            
            if "{name}" in template:
                name = random.choice(self.generator.names)
                text = text.replace("{name}", name)
                start_idx = text.find(name)
                entities.append(ExtractedEntity(
                    type=EntityType.PASSENGER_NAME,
                    value=name,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(name)
                ))
            
            # Add random modifiers
            if random.random() < 0.3:
                modifier = random.choice(urgency_modifiers)
                if modifier:
                    text = f"{text} {modifier}"
            
            if random.random() < 0.4:
                politeness_word = random.choice(politeness)
                if politeness_word:
                    text = f"{politeness_word} {text.lower()}"
            
            examples.append(TrainingExample(
                text=text,
                intent=RequestType.CANCEL_TRIP,
                entities=entities
            ))
        
        return examples
    
    def generate_flight_status_examples(self, count: int = 2000) -> List[TrainingExample]:
        """Generate flight status training examples"""
        examples = []
        
        templates = [
            "What's my flight status",
            "Is flight {flight_number} on time",
            "Check status of booking {pnr}",
            "When does my flight depart",
            "Flight status for {date} trip to {destination}",
            "Is my flight delayed",
            "Current status of {pnr}",
            "Track my flight",
            "Flight departure time",
            "Has my flight been cancelled",
            "What time does flight {flight_number} arrive",
            "Check my flight {pnr}",
            "Status of my booking",
            "Is {flight_number} delayed",
            "When is my flight",
            "Flight info for {pnr}",
            "Check flight {flight_number} on {date}",
            "My flight status",
            "Update on flight {flight_number}",
            "Is my {date} flight on time"
        ]
        
        for i in range(count):
            template = random.choice(templates)
            entities = []
            text = template
            
            # Replace placeholders with actual values
            if "{pnr}" in template:
                pnr = random.choice(self.generator.pnr_patterns)
                text = text.replace("{pnr}", pnr)
                start_idx = text.find(pnr)
                entities.append(ExtractedEntity(
                    type=EntityType.PNR,
                    value=pnr,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(pnr)
                ))
            
            if "{flight_number}" in template:
                flight = random.choice(self.generator.flight_numbers)
                text = text.replace("{flight_number}", flight)
                start_idx = text.find(flight)
                entities.append(ExtractedEntity(
                    type=EntityType.FLIGHT_NUMBER,
                    value=flight,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(flight)
                ))
            
            if "{destination}" in template:
                dest = random.choice(self.generator.airport_codes)
                text = text.replace("{destination}", dest)
            
            if "{date}" in template:
                date = random.choice(self.generator.dates)
                text = text.replace("{date}", date)
                start_idx = text.find(date)
                entities.append(ExtractedEntity(
                    type=EntityType.DATE,
                    value=date,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(date)
                ))
            
            examples.append(TrainingExample(
                text=text,
                intent=RequestType.FLIGHT_STATUS,
                entities=entities
            ))
        
        return examples
    
    def generate_seat_availability_examples(self, count: int = 2000) -> List[TrainingExample]:
        """Generate seat availability training examples"""
        examples = []
        
        templates = [
            "Show available seats",
            "What seats are free on my flight",
            "Check seat availability for booking {pnr}",
            "I want to change my seat",
            "Available seats in {class} class",
            "Show me {seat_type} seats",
            "Seat map for flight {flight_number}",
            "Can I upgrade my seat",
            "Empty seats on my flight",
            "Seat selection options",
            "Available seats on {flight_number}",
            "What seats can I choose",
            "Seat availability for {pnr}",
            "Show seat map",
            "I need a {seat_type} seat",
            "Upgrade options for my flight",
            "Available {class} class seats",
            "Change my seat assignment",
            "What seats are open",
            "Seat options for flight {flight_number}"
        ]
        
        seat_types = ["window", "aisle", "middle", "exit row", "bulkhead"]
        classes = ["economy", "business", "first", "premium economy"]
        
        for i in range(count):
            template = random.choice(templates)
            entities = []
            text = template
            
            if "{pnr}" in template:
                pnr = random.choice(self.generator.pnr_patterns)
                text = text.replace("{pnr}", pnr)
                start_idx = text.find(pnr)
                entities.append(ExtractedEntity(
                    type=EntityType.PNR,
                    value=pnr,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(pnr)
                ))
            
            if "{flight_number}" in template:
                flight = random.choice(self.generator.flight_numbers)
                text = text.replace("{flight_number}", flight)
                start_idx = text.find(flight)
                entities.append(ExtractedEntity(
                    type=EntityType.FLIGHT_NUMBER,
                    value=flight,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(flight)
                ))
            
            if "{seat_type}" in template:
                seat_type = random.choice(seat_types)
                text = text.replace("{seat_type}", seat_type)
                start_idx = text.find(seat_type)
                entities.append(ExtractedEntity(
                    type=EntityType.SEAT_TYPE,
                    value=seat_type,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(seat_type)
                ))
            
            if "{class}" in template:
                seat_class = random.choice(classes)
                text = text.replace("{class}", seat_class)
                start_idx = text.find(seat_class)
                entities.append(ExtractedEntity(
                    type=EntityType.CLASS,
                    value=seat_class,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(seat_class)
                ))
            
            examples.append(TrainingExample(
                text=text,
                intent=RequestType.SEAT_AVAILABILITY,
                entities=entities
            ))
        
        return examples
    
    def generate_cancellation_policy_examples(self, count: int = 2000) -> List[TrainingExample]:
        """Generate cancellation policy training examples"""
        examples = []
        
        templates = [
            "What's your cancellation policy",
            "Cancellation fees for my booking",
            "Can I get a refund if I cancel",
            "Policy for cancelling flights",
            "How much does it cost to cancel",
            "Refund policy information",
            "What are the cancellation rules",
            "Free cancellation period",
            "Cancellation terms and conditions",
            "When can I cancel without penalty",
            "Refund rules for {class} class",
            "Cancellation policy for {flight_number}",
            "What's the cancellation fee",
            "Can I cancel for free",
            "Refund terms",
            "Cancellation charges",
            "Policy on flight changes",
            "What happens if I cancel",
            "Cancellation deadline",
            "Refund eligibility"
        ]
        
        classes = ["economy", "business", "first", "premium"]
        
        for i in range(count):
            template = random.choice(templates)
            entities = []
            text = template
            
            if "{flight_number}" in template:
                flight = random.choice(self.generator.flight_numbers)
                text = text.replace("{flight_number}", flight)
                start_idx = text.find(flight)
                entities.append(ExtractedEntity(
                    type=EntityType.FLIGHT_NUMBER,
                    value=flight,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(flight)
                ))
            
            if "{class}" in template:
                seat_class = random.choice(classes)
                text = text.replace("{class}", seat_class)
                start_idx = text.find(seat_class)
                entities.append(ExtractedEntity(
                    type=EntityType.CLASS,
                    value=seat_class,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(seat_class)
                ))
            
            examples.append(TrainingExample(
                text=text,
                intent=RequestType.CANCELLATION_POLICY,
                entities=entities
            ))
        
        return examples
    
    def generate_pet_travel_examples(self, count: int = 2000) -> List[TrainingExample]:
        """Generate pet travel training examples"""
        examples = []
        
        templates = [
            "Can I bring my pet on the flight",
            "Pet travel policy",
            "Flying with my {pet_type}",
            "What are the rules for pets",
            "Pet carrier requirements",
            "Can {pet_type}s fly in cabin",
            "Pet documentation needed",
            "{pet_type} policy",
            "Pet fees and charges",
            "{pet_type} rules",
            "Traveling with pets",
            "Pet restrictions",
            "Can I bring my {pet_type}",
            "Pet travel requirements",
            "Flying with animals",
            "Pet cargo policy",
            "In-cabin pet policy",
            "Pet health requirements",
            "Pet travel costs",
            "Animal travel rules"
        ]
        
        pet_types = [
            "dog", "cat", "service animal", "emotional support animal", 
            "bird", "rabbit", "hamster", "guinea pig", "ferret"
        ]
        
        for i in range(count):
            template = random.choice(templates)
            entities = []
            text = template
            
            if "{pet_type}" in template:
                pet_type = random.choice(pet_types)
                text = text.replace("{pet_type}", pet_type)
                start_idx = text.find(pet_type)
                entities.append(ExtractedEntity(
                    type=EntityType.PET_TYPE,
                    value=pet_type,
                    confidence=1.0,
                    start_index=start_idx,
                    end_index=start_idx + len(pet_type)
                ))
            
            examples.append(TrainingExample(
                text=text,
                intent=RequestType.PET_TRAVEL,
                entities=entities
            ))
        
        return examples
    
    def generate_complete_dataset(self) -> List[TrainingExample]:
        """Generate complete training dataset with all intent classes"""
        self.logger.info("Generating complete training dataset...")
        
        all_examples = []
        
        # Generate examples for each intent class
        all_examples.extend(self.generate_cancel_trip_examples(2000))
        all_examples.extend(self.generate_flight_status_examples(2000))
        all_examples.extend(self.generate_seat_availability_examples(2000))
        all_examples.extend(self.generate_cancellation_policy_examples(2000))
        all_examples.extend(self.generate_pet_travel_examples(2000))
        
        # Shuffle the dataset
        random.shuffle(all_examples)
        
        self.logger.info(f"Generated {len(all_examples)} training examples")
        return all_examples
    
    def save_dataset(self, examples: List[TrainingExample], filepath: str) -> None:
        """Save dataset to JSON file"""
        data = []
        for example in examples:
            entities_data = []
            for entity in example.entities:
                entities_data.append({
                    "type": entity.type.value,
                    "value": entity.value,
                    "confidence": entity.confidence,
                    "start_index": entity.start_index,
                    "end_index": entity.end_index
                })
            
            data.append({
                "text": example.text,
                "intent": example.intent.value,
                "entities": entities_data
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(examples)} examples to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[TrainingExample]:
        """Load dataset from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            entities = []
            for entity_data in item.get("entities", []):
                entities.append(ExtractedEntity(
                    type=EntityType(entity_data["type"]),
                    value=entity_data["value"],
                    confidence=entity_data["confidence"],
                    start_index=entity_data["start_index"],
                    end_index=entity_data["end_index"]
                ))
            
            examples.append(TrainingExample(
                text=item["text"],
                intent=RequestType(item["intent"]),
                entities=entities
            ))
        
        self.logger.info(f"Loaded {len(examples)} examples from {filepath}")
        return examples