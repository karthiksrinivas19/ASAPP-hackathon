#!/usr/bin/env python3
"""
Final dataset generator with proper entity extraction
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


def extract_entities_from_text(text: str) -> List[Dict]:
    """Extract entities from text using regex patterns"""
    entities = []
    
    # PNR pattern (6 alphanumeric characters)
    pnr_pattern = r'\b([A-Z0-9]{6})\b'
    for match in re.finditer(pnr_pattern, text.upper()):
        entities.append({
            "type": EntityType.PNR.value,
            "value": match.group(1),
            "confidence": 0.9,
            "start_index": match.start(1),
            "end_index": match.end(1)
        })
    
    # Flight number pattern (2-3 letters + 1-4 digits)
    flight_pattern = r'\b([A-Z]{2,3}[0-9]{1,4})\b'
    for match in re.finditer(flight_pattern, text.upper()):
        entities.append({
            "type": EntityType.FLIGHT_NUMBER.value,
            "value": match.group(1),
            "confidence": 0.9,
            "start_index": match.start(1),
            "end_index": match.end(1)
        })
    
    # Date patterns
    date_patterns = [
        r'\b(today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b',
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b',
        r'\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b'
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text.lower()):
            entities.append({
                "type": EntityType.DATE.value,
                "value": match.group(1),
                "confidence": 0.8,
                "start_index": match.start(1),
                "end_index": match.end(1)
            })
    
    # Airport codes (3 letters)
    airport_pattern = r'\b([A-Z]{3})\b'
    for match in re.finditer(airport_pattern, text.upper()):
        # Common airport codes
        common_airports = ["JFK", "LAX", "ORD", "DFW", "DEN", "ATL", "SFO", "SEA", "LAS", "PHX"]
        if match.group(1) in common_airports:
            entities.append({
                "type": EntityType.AIRPORT_CODE.value,
                "value": match.group(1),
                "confidence": 0.9,
                "start_index": match.start(1),
                "end_index": match.end(1)
            })
    
    # Email pattern
    email_pattern = r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
    for match in re.finditer(email_pattern, text):
        entities.append({
            "type": EntityType.EMAIL.value,
            "value": match.group(1).lower(),
            "confidence": 0.95,
            "start_index": match.start(1),
            "end_index": match.end(1)
        })
    
    # Phone pattern
    phone_pattern = r'\b(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b'
    for match in re.finditer(phone_pattern, text):
        entities.append({
            "type": EntityType.PHONE_NUMBER.value,
            "value": match.group(1),
            "confidence": 0.9,
            "start_index": match.start(1),
            "end_index": match.end(1)
        })
    
    # Seat types
    seat_types = ["window", "aisle", "middle", "exit row", "bulkhead"]
    for seat_type in seat_types:
        pattern = rf'\b({re.escape(seat_type)})\b'
        for match in re.finditer(pattern, text.lower()):
            entities.append({
                "type": EntityType.SEAT_TYPE.value,
                "value": match.group(1),
                "confidence": 0.9,
                "start_index": match.start(1),
                "end_index": match.end(1)
            })
    
    # Classes
    classes = ["economy", "business", "first", "premium economy", "premium"]
    for class_name in classes:
        pattern = rf'\b({re.escape(class_name)})\b'
        for match in re.finditer(pattern, text.lower()):
            entities.append({
                "type": EntityType.CLASS.value,
                "value": match.group(1),
                "confidence": 0.9,
                "start_index": match.start(1),
                "end_index": match.end(1)
            })
    
    # Pet types
    pet_types = ["dog", "cat", "service animal", "emotional support animal", "bird", "rabbit", "pet"]
    for pet_type in pet_types:
        pattern = rf'\b({re.escape(pet_type)})\b'
        for match in re.finditer(pattern, text.lower()):
            entities.append({
                "type": EntityType.PET_TYPE.value,
                "value": match.group(1),
                "confidence": 0.9,
                "start_index": match.start(1),
                "end_index": match.end(1)
            })
    
    # Passenger names (simple pattern)
    name_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
    for match in re.finditer(name_pattern, text):
        entities.append({
            "type": EntityType.PASSENGER_NAME.value,
            "value": match.group(1),
            "confidence": 0.8,
            "start_index": match.start(1),
            "end_index": match.end(1)
        })
    
    # Remove duplicates and sort by position
    unique_entities = []
    seen = set()
    
    for entity in entities:
        key = (entity["type"], entity["value"], entity["start_index"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    
    unique_entities.sort(key=lambda x: x["start_index"])
    return unique_entities


def generate_examples_with_entities():
    """Generate training examples with proper entity extraction"""
    
    # Generate data pools
    pnrs = [''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6)) for _ in range(100)]
    flights = [f"{airline}{num}" for airline in ["AA", "UA", "DL", "JB", "SW"] for num in range(100, 1000, 100)]
    airports = ["JFK", "LAX", "ORD", "DFW", "DEN", "ATL", "SFO", "SEA", "LAS", "PHX"]
    dates = ["today", "tomorrow", "Monday", "Tuesday", "January 15", "Jan 15", "01/15"]
    names = ["John Smith", "Jane Doe", "Michael Johnson", "Sarah Williams", "David Brown"]
    emails = ["john@gmail.com", "jane@yahoo.com", "mike@hotmail.com", "sarah@outlook.com"]
    phones = ["(555) 123-4567", "555-987-6543", "(212) 555-0123"]
    
    all_examples = []
    
    # Cancel Trip Examples (2000)
    cancel_templates = [
        "I want to cancel my flight",
        "Cancel my booking",
        "Please cancel my reservation",
        f"Cancel flight {random.choice(flights)}",
        f"I want to cancel booking {random.choice(pnrs)}",
        f"Cancel my flight {random.choice(pnrs)}",
        f"Cancel my trip to {random.choice(airports)}",
        f"I need to cancel my flight on {random.choice(dates)}",
        f"Cancel reservation for {random.choice(names)}",
        f"My name is {random.choice(names)}, cancel my booking {random.choice(pnrs)}",
        f"Hi, I'm {random.choice(names)} at {random.choice(emails)}, cancel my flight",
        f"Cancel {random.choice(flights)} on {random.choice(dates)}",
        "I can't travel anymore, cancel my flight",
        "Emergency cancellation needed",
        "Refund my ticket",
        "I want my money back"
    ]
    
    for i in range(2000):
        if i < len(cancel_templates):
            text = cancel_templates[i]
        else:
            text = random.choice(cancel_templates)
        
        # Add variations
        if random.random() < 0.3:
            text = f"Please {text.lower()}"
        if random.random() < 0.2:
            text = f"{text} urgently"
        
        entities = extract_entities_from_text(text)
        
        all_examples.append({
            "text": text,
            "intent": RequestType.CANCEL_TRIP.value,
            "entities": entities
        })
    
    # Flight Status Examples (2000)
    status_templates = [
        "What's my flight status",
        "Is my flight on time",
        "When does my flight depart",
        "Has my flight been cancelled",
        f"Is flight {random.choice(flights)} on time",
        f"Check status of booking {random.choice(pnrs)}",
        f"Flight status for {random.choice(dates)} trip",
        f"Current status of {random.choice(pnrs)}",
        f"Check flight {random.choice(flights)} on {random.choice(dates)}",
        f"Status of my {random.choice(dates)} flight to {random.choice(airports)}",
        "Track my flight",
        "Flight departure time",
        "Flight arrival time",
        "Is my flight delayed"
    ]
    
    for i in range(2000):
        if i < len(status_templates):
            text = status_templates[i]
        else:
            text = random.choice(status_templates)
        
        entities = extract_entities_from_text(text)
        
        all_examples.append({
            "text": text,
            "intent": RequestType.FLIGHT_STATUS.value,
            "entities": entities
        })
    
    # Seat Availability Examples (2000)
    seat_templates = [
        "Show available seats",
        "What seats are free on my flight",
        "I want to change my seat",
        "Can I upgrade my seat",
        "Seat selection options",
        f"Check seat availability for booking {random.choice(pnrs)}",
        "Available seats in business class",
        "Available seats in economy class",
        "Available seats in first class",
        "Show me window seats",
        "Show me aisle seats",
        f"Seat map for flight {random.choice(flights)}",
        "Available window seats in business",
        "Available aisle seats in economy",
        "Empty seats on my flight"
    ]
    
    for i in range(2000):
        if i < len(seat_templates):
            text = seat_templates[i]
        else:
            text = random.choice(seat_templates)
        
        entities = extract_entities_from_text(text)
        
        all_examples.append({
            "text": text,
            "intent": RequestType.SEAT_AVAILABILITY.value,
            "entities": entities
        })
    
    # Cancellation Policy Examples (2000)
    policy_templates = [
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
        "What's the cancellation fee",
        "Can I cancel for free",
        "Refund terms",
        "Cancellation charges",
        "What happens if I cancel"
    ]
    
    for i in range(2000):
        if i < len(policy_templates):
            text = policy_templates[i]
        else:
            text = random.choice(policy_templates)
        
        entities = extract_entities_from_text(text)
        
        all_examples.append({
            "text": text,
            "intent": RequestType.CANCELLATION_POLICY.value,
            "entities": entities
        })
    
    # Pet Travel Examples (2000)
    pet_templates = [
        "Can I bring my pet on the flight",
        "Pet travel policy",
        "What are the rules for pets",
        "Pet carrier requirements",
        "Pet documentation needed",
        "Pet fees and charges",
        "Flying with my dog",
        "Flying with my cat",
        "Can dogs fly in cabin",
        "Can cats fly in cabin",
        "Service animal policy",
        "Emotional support animal rules",
        "Pet travel requirements",
        "Flying with animals",
        "Pet restrictions"
    ]
    
    for i in range(2000):
        if i < len(pet_templates):
            text = pet_templates[i]
        else:
            text = random.choice(pet_templates)
        
        entities = extract_entities_from_text(text)
        
        all_examples.append({
            "text": text,
            "intent": RequestType.PET_TRAVEL.value,
            "entities": entities
        })
    
    # Shuffle the dataset
    random.shuffle(all_examples)
    
    return all_examples


def main():
    """Generate final dataset with proper entity extraction"""
    print("Generating final training dataset with entity extraction...")
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Generate dataset
    dataset = generate_examples_with_entities()
    
    # Save dataset
    output_path = output_dir / "final_training_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset generation complete!")
    print(f"Total examples: {len(dataset)}")
    print(f"Saved to: {output_path}")
    
    # Statistics
    intent_counts = {}
    entity_counts = {}
    examples_with_entities = 0
    
    for example in dataset:
        intent = example["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        if example["entities"]:
            examples_with_entities += 1
        
        for entity in example["entities"]:
            entity_type = entity["type"]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    print("\nIntent distribution:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")
    
    print(f"\nExamples with entities: {examples_with_entities} ({examples_with_entities/len(dataset)*100:.1f}%)")
    
    print("\nEntity distribution:")
    for entity_type, count in sorted(entity_counts.items()):
        print(f"  {entity_type}: {count}")


if __name__ == "__main__":
    main()