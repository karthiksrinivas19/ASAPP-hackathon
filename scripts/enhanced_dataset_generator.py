#!/usr/bin/env python3
"""
Enhanced dataset generator with entity extraction
"""

import json
import random
import re
from pathlib import Path
from enum import Enum
from typing import List, Dict, Any, Tuple


class RequestType(str, Enum):
    CANCEL_TRIP = "cancel_trip"
    CANCELLATION_POLICY = "cancellation_policy"
    FLIGHT_STATUS = "flight_status"
    SEAT_AVAILABILITY = "seat_availability"
    PET_TRAVEL = "pet_travel"


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


class EnhancedDatasetGenerator:
    """Enhanced dataset generator with entity extraction"""
    
    def __init__(self):
        self.pnrs = self._generate_pnrs()
        self.flights = self._generate_flights()
        self.airports = self._generate_airports()
        self.dates = self._generate_dates()
        self.names = self._generate_names()
        self.phones = self._generate_phones()
        self.emails = self._generate_emails()
        self.seat_types = ["window", "aisle", "middle", "exit row", "bulkhead"]
        self.classes = ["economy", "business", "first", "premium economy"]
        self.pet_types = ["dog", "cat", "service animal", "emotional support animal", "bird", "rabbit"]
    
    def _generate_pnrs(self) -> List[str]:
        """Generate realistic PNR patterns"""
        pnrs = []
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        
        for _ in range(200):
            pnr = ''.join(random.choices(letters + digits, k=6))
            pnrs.append(pnr)
        
        return pnrs
    
    def _generate_flights(self) -> List[str]:
        """Generate realistic flight numbers"""
        airlines = ["AA", "UA", "DL", "JB", "SW", "NK", "F9", "B6", "AS", "WN"]
        flights = []
        
        for airline in airlines:
            for num in range(1, 1000, 50):
                flights.append(f"{airline}{num}")
        
        return flights
    
    def _generate_airports(self) -> List[str]:
        """Generate airport codes"""
        return [
            "JFK", "LAX", "ORD", "DFW", "DEN", "ATL", "SFO", "SEA", "LAS", "PHX",
            "IAH", "CLT", "MIA", "BOS", "MSP", "DTW", "PHL", "LGA", "FLL", "BWI"
        ]
    
    def _generate_dates(self) -> List[str]:
        """Generate date variations"""
        return [
            "today", "tomorrow", "yesterday", "next week", "Monday", "Tuesday", 
            "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January 15", "Jan 15", "01/15", "12/25", "next month", "this weekend"
        ]
    
    def _generate_names(self) -> List[str]:
        """Generate passenger names"""
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Mary"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        
        names = []
        for first in first_names:
            for last in last_names:
                names.append(f"{first} {last}")
        
        return names[:30]
    
    def _generate_phones(self) -> List[str]:
        """Generate phone numbers"""
        phones = []
        for _ in range(20):
            area = random.randint(200, 999)
            exchange = random.randint(200, 999)
            number = random.randint(1000, 9999)
            phones.append(f"({area}) {exchange}-{number}")
        
        return phones
    
    def _generate_emails(self) -> List[str]:
        """Generate email addresses"""
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
        names = ["john", "jane", "mike", "sarah", "david", "lisa"]
        
        emails = []
        for name in names:
            for domain in domains:
                emails.append(f"{name}@{domain}")
        
        return emails
    
    def _extract_entities(self, text: str, template_entities: List[Dict]) -> List[Dict]:
        """Extract entities from text based on template"""
        entities = []
        
        for entity_info in template_entities:
            entity_type = entity_info["type"]
            value = entity_info["value"]
            
            # Find the position of the entity in the text
            start_idx = text.find(value)
            if start_idx != -1:
                entities.append({
                    "type": entity_type,
                    "value": value,
                    "confidence": 1.0,
                    "start_index": start_idx,
                    "end_index": start_idx + len(value)
                })
        
        return entities
    
    def generate_cancel_trip_examples(self, count: int = 2000) -> List[Dict]:
        """Generate cancel trip examples with entities"""
        examples = []
        
        templates = [
            ("I want to cancel my flight", []),
            ("Cancel my booking", []),
            ("Cancel flight {flight}", [{"type": EntityType.FLIGHT_NUMBER.value, "value": "{flight}"}]),
            ("I want to cancel booking {pnr}", [{"type": EntityType.PNR.value, "value": "{pnr}"}]),
            ("Cancel my flight {pnr}", [{"type": EntityType.PNR.value, "value": "{pnr}"}]),
            ("Cancel my trip to {airport}", [{"type": EntityType.DESTINATION.value, "value": "{airport}"}]),
            ("I need to cancel my flight on {date}", [{"type": EntityType.DATE.value, "value": "{date}"}]),
            ("Cancel reservation for {name}", [{"type": EntityType.PASSENGER_NAME.value, "value": "{name}"}]),
            ("My name is {name}, cancel my booking {pnr}", [
                {"type": EntityType.PASSENGER_NAME.value, "value": "{name}"},
                {"type": EntityType.PNR.value, "value": "{pnr}"}
            ]),
            ("Cancel {flight} on {date}", [
                {"type": EntityType.FLIGHT_NUMBER.value, "value": "{flight}"},
                {"type": EntityType.DATE.value, "value": "{date}"}
            ])
        ]
        
        for i in range(count):
            template, entity_templates = random.choice(templates)
            text = template
            actual_entities = []
            
            # Replace placeholders with actual values
            if "{flight}" in text:
                flight = random.choice(self.flights)
                text = text.replace("{flight}", flight)
                for et in entity_templates:
                    if et["value"] == "{flight}":
                        et["value"] = flight
            
            if "{pnr}" in text:
                pnr = random.choice(self.pnrs)
                text = text.replace("{pnr}", pnr)
                for et in entity_templates:
                    if et["value"] == "{pnr}":
                        et["value"] = pnr
            
            if "{airport}" in text:
                airport = random.choice(self.airports)
                text = text.replace("{airport}", airport)
                for et in entity_templates:
                    if et["value"] == "{airport}":
                        et["value"] = airport
            
            if "{date}" in text:
                date = random.choice(self.dates)
                text = text.replace("{date}", date)
                for et in entity_templates:
                    if et["value"] == "{date}":
                        et["value"] = date
            
            if "{name}" in text:
                name = random.choice(self.names)
                text = text.replace("{name}", name)
                for et in entity_templates:
                    if et["value"] == "{name}":
                        et["value"] = name
            
            # Extract entities from the final text
            entities = self._extract_entities(text, entity_templates)
            
            # Add variations
            if random.random() < 0.3:
                text = f"Please {text.lower()}"
            if random.random() < 0.2:
                text = f"{text} urgently"
            
            examples.append({
                "text": text,
                "intent": RequestType.CANCEL_TRIP.value,
                "entities": entities
            })
        
        return examples
    
    def generate_flight_status_examples(self, count: int = 2000) -> List[Dict]:
        """Generate flight status examples with entities"""
        examples = []
        
        templates = [
            ("What's my flight status", []),
            ("Is flight {flight} on time", [{"type": EntityType.FLIGHT_NUMBER.value, "value": "{flight}"}]),
            ("Check status of booking {pnr}", [{"type": EntityType.PNR.value, "value": "{pnr}"}]),
            ("Flight status for {date} trip", [{"type": EntityType.DATE.value, "value": "{date}"}]),
            ("Current status of {pnr}", [{"type": EntityType.PNR.value, "value": "{pnr}"}]),
            ("Check flight {flight} on {date}", [
                {"type": EntityType.FLIGHT_NUMBER.value, "value": "{flight}"},
                {"type": EntityType.DATE.value, "value": "{date}"}
            ]),
            ("Status of my {date} flight to {airport}", [
                {"type": EntityType.DATE.value, "value": "{date}"},
                {"type": EntityType.DESTINATION.value, "value": "{airport}"}
            ])
        ]
        
        for i in range(count):
            template, entity_templates = random.choice(templates)
            text = template
            
            # Replace placeholders
            if "{flight}" in text:
                flight = random.choice(self.flights)
                text = text.replace("{flight}", flight)
                for et in entity_templates:
                    if et["value"] == "{flight}":
                        et["value"] = flight
            
            if "{pnr}" in text:
                pnr = random.choice(self.pnrs)
                text = text.replace("{pnr}", pnr)
                for et in entity_templates:
                    if et["value"] == "{pnr}":
                        et["value"] = pnr
            
            if "{date}" in text:
                date = random.choice(self.dates)
                text = text.replace("{date}", date)
                for et in entity_templates:
                    if et["value"] == "{date}":
                        et["value"] = date
            
            if "{airport}" in text:
                airport = random.choice(self.airports)
                text = text.replace("{airport}", airport)
                for et in entity_templates:
                    if et["value"] == "{airport}":
                        et["value"] = airport
            
            entities = self._extract_entities(text, entity_templates)
            
            examples.append({
                "text": text,
                "intent": RequestType.FLIGHT_STATUS.value,
                "entities": entities
            })
        
        return examples
    
    def generate_seat_availability_examples(self, count: int = 2000) -> List[Dict]:
        """Generate seat availability examples with entities"""
        examples = []
        
        templates = [
            ("Show available seats", []),
            ("Check seat availability for {pnr}", [{"type": EntityType.PNR.value, "value": "{pnr}"}]),
            ("Available seats in {class} class", [{"type": EntityType.CLASS.value, "value": "{class}"}]),
            ("Show me {seat_type} seats", [{"type": EntityType.SEAT_TYPE.value, "value": "{seat_type}"}]),
            ("Seat map for flight {flight}", [{"type": EntityType.FLIGHT_NUMBER.value, "value": "{flight}"}]),
            ("Available {seat_type} seats in {class}", [
                {"type": EntityType.SEAT_TYPE.value, "value": "{seat_type}"},
                {"type": EntityType.CLASS.value, "value": "{class}"}
            ])
        ]
        
        for i in range(count):
            template, entity_templates = random.choice(templates)
            text = template
            
            if "{pnr}" in text:
                pnr = random.choice(self.pnrs)
                text = text.replace("{pnr}", pnr)
                for et in entity_templates:
                    if et["value"] == "{pnr}":
                        et["value"] = pnr
            
            if "{class}" in text:
                seat_class = random.choice(self.classes)
                text = text.replace("{class}", seat_class)
                for et in entity_templates:
                    if et["value"] == "{class}":
                        et["value"] = seat_class
            
            if "{seat_type}" in text:
                seat_type = random.choice(self.seat_types)
                text = text.replace("{seat_type}", seat_type)
                for et in entity_templates:
                    if et["value"] == "{seat_type}":
                        et["value"] = seat_type
            
            if "{flight}" in text:
                flight = random.choice(self.flights)
                text = text.replace("{flight}", flight)
                for et in entity_templates:
                    if et["value"] == "{flight}":
                        et["value"] = flight
            
            entities = self._extract_entities(text, entity_templates)
            
            examples.append({
                "text": text,
                "intent": RequestType.SEAT_AVAILABILITY.value,
                "entities": entities
            })
        
        return examples
    
    def generate_cancellation_policy_examples(self, count: int = 2000) -> List[Dict]:
        """Generate cancellation policy examples"""
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
            "When can I cancel without penalty"
        ]
        
        for i in range(count):
            text = random.choice(templates)
            
            examples.append({
                "text": text,
                "intent": RequestType.CANCELLATION_POLICY.value,
                "entities": []
            })
        
        return examples
    
    def generate_pet_travel_examples(self, count: int = 2000) -> List[Dict]:
        """Generate pet travel examples with entities"""
        examples = []
        
        templates = [
            ("Can I bring my pet on the flight", []),
            ("Pet travel policy", []),
            ("Flying with my {pet_type}", [{"type": EntityType.PET_TYPE.value, "value": "{pet_type}"}]),
            ("What are the rules for pets", []),
            ("Can {pet_type}s fly in cabin", [{"type": EntityType.PET_TYPE.value, "value": "{pet_type}"}]),
            ("{pet_type} policy", [{"type": EntityType.PET_TYPE.value, "value": "{pet_type}"}]),
            ("Pet fees for {pet_type}", [{"type": EntityType.PET_TYPE.value, "value": "{pet_type}"}])
        ]
        
        for i in range(count):
            template, entity_templates = random.choice(templates)
            text = template
            
            if "{pet_type}" in text:
                pet_type = random.choice(self.pet_types)
                text = text.replace("{pet_type}", pet_type)
                for et in entity_templates:
                    if et["value"] == "{pet_type}":
                        et["value"] = pet_type
            
            entities = self._extract_entities(text, entity_templates)
            
            examples.append({
                "text": text,
                "intent": RequestType.PET_TRAVEL.value,
                "entities": entities
            })
        
        return examples
    
    def generate_complete_dataset(self) -> List[Dict]:
        """Generate complete dataset with all intents"""
        all_examples = []
        
        print("Generating cancel trip examples...")
        all_examples.extend(self.generate_cancel_trip_examples(2000))
        
        print("Generating flight status examples...")
        all_examples.extend(self.generate_flight_status_examples(2000))
        
        print("Generating seat availability examples...")
        all_examples.extend(self.generate_seat_availability_examples(2000))
        
        print("Generating cancellation policy examples...")
        all_examples.extend(self.generate_cancellation_policy_examples(2000))
        
        print("Generating pet travel examples...")
        all_examples.extend(self.generate_pet_travel_examples(2000))
        
        # Shuffle dataset
        random.shuffle(all_examples)
        
        return all_examples


def main():
    """Generate enhanced dataset with entities"""
    print("Starting enhanced dataset generation...")
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Generate dataset
    generator = EnhancedDatasetGenerator()
    dataset = generator.generate_complete_dataset()
    
    # Save dataset
    output_path = output_dir / "enhanced_training_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset generation complete!")
    print(f"Total examples: {len(dataset)}")
    print(f"Saved to: {output_path}")
    
    # Statistics
    intent_counts = {}
    entity_counts = {}
    
    for example in dataset:
        intent = example["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        for entity in example["entities"]:
            entity_type = entity["type"]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    print("\nIntent distribution:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")
    
    print("\nEntity distribution:")
    for entity_type, count in entity_counts.items():
        print(f"  {entity_type}: {count}")


if __name__ == "__main__":
    main()