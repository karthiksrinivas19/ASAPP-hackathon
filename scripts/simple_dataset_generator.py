#!/usr/bin/env python3
"""
Simple script to generate training dataset without external dependencies
"""

import json
import random
import os
from pathlib import Path
from enum import Enum
from typing import List, Dict, Any


class RequestType(str, Enum):
    CANCEL_TRIP = "cancel_trip"
    CANCELLATION_POLICY = "cancellation_policy"
    FLIGHT_STATUS = "flight_status"
    SEAT_AVAILABILITY = "seat_availability"
    PET_TRAVEL = "pet_travel"
    UNKNOWN = "unknown"


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


def generate_pnr_patterns():
    """Generate realistic PNR patterns"""
    pnrs = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    
    for _ in range(100):
        pnr = ''.join(random.choices(letters + digits, k=6))
        pnrs.append(pnr)
    
    return pnrs


def generate_flight_numbers():
    """Generate realistic flight numbers"""
    airlines = ["AA", "UA", "DL", "JB", "SW", "NK", "F9", "B6", "AS", "WN"]
    flights = []
    
    for airline in airlines:
        for num in range(1, 9999, 100):
            flights.append(f"{airline}{num}")
            if len(flights) >= 200:
                break
    
    return flights[:200]


def generate_airport_codes():
    """Generate common airport codes"""
    return [
        "JFK", "LAX", "ORD", "DFW", "DEN", "ATL", "SFO", "SEA", "LAS", "PHX",
        "IAH", "CLT", "MIA", "BOS", "MSP", "DTW", "PHL", "LGA", "FLL", "BWI"
    ]


def generate_dates():
    """Generate various date formats"""
    return [
        "today", "tomorrow", "yesterday", "next week", "Monday", "Tuesday",
        "January 15", "Jan 15", "01/15", "12/25", "next month"
    ]


def generate_cancel_trip_examples(count=2000):
    """Generate cancel trip training examples"""
    examples = []
    pnrs = generate_pnr_patterns()
    flights = generate_flight_numbers()
    airports = generate_airport_codes()
    dates = generate_dates()
    
    templates = [
        "I want to cancel my flight",
        "Cancel my booking",
        "I need to cancel my trip", 
        "Please cancel my reservation",
        f"Cancel flight {random.choice(flights)}",
        f"I want to cancel booking {random.choice(pnrs)}",
        f"Cancel my flight {random.choice(pnrs)}",
        "I need to get a refund for my booking",
        f"Cancel my reservation {random.choice(pnrs)}",
        "I can't travel anymore, cancel my flight",
        f"Emergency cancellation needed for booking {random.choice(pnrs)}",
        "Refund my ticket",
        "I want to cancel and get my money back",
        f"Cancel my trip to {random.choice(airports)}",
        f"I need to cancel my flight on {random.choice(dates)}",
        f"Cancel my {random.choice(dates)} flight"
    ]
    
    for i in range(count):
        if i < len(templates):
            text = templates[i]
        else:
            text = random.choice(templates)
        
        # Add some variations
        if random.random() < 0.3:
            text = f"Please {text.lower()}"
        if random.random() < 0.2:
            text = f"{text} urgently"
        
        examples.append({
            "text": text,
            "intent": RequestType.CANCEL_TRIP.value,
            "entities": []
        })
    
    return examples


def generate_flight_status_examples(count=2000):
    """Generate flight status training examples"""
    examples = []
    pnrs = generate_pnr_patterns()
    flights = generate_flight_numbers()
    airports = generate_airport_codes()
    dates = generate_dates()
    
    templates = [
        "What's my flight status",
        f"Is flight {random.choice(flights)} on time",
        f"Check status of booking {random.choice(pnrs)}",
        "When does my flight depart",
        f"Flight status for {random.choice(dates)} trip to {random.choice(airports)}",
        "Is my flight delayed",
        f"Current status of {random.choice(pnrs)}",
        "Track my flight",
        "Flight departure time",
        "Has my flight been cancelled",
        f"What time does flight {random.choice(flights)} arrive",
        f"Check my flight {random.choice(pnrs)}",
        "Status of my booking",
        f"Is {random.choice(flights)} delayed",
        "When is my flight"
    ]
    
    for i in range(count):
        if i < len(templates):
            text = templates[i]
        else:
            text = random.choice(templates)
        
        examples.append({
            "text": text,
            "intent": RequestType.FLIGHT_STATUS.value,
            "entities": []
        })
    
    return examples


def generate_seat_availability_examples(count=2000):
    """Generate seat availability training examples"""
    examples = []
    pnrs = generate_pnr_patterns()
    flights = generate_flight_numbers()
    
    seat_types = ["window", "aisle", "middle", "exit row"]
    classes = ["economy", "business", "first", "premium"]
    
    templates = [
        "Show available seats",
        "What seats are free on my flight",
        f"Check seat availability for booking {random.choice(pnrs)}",
        "I want to change my seat",
        f"Available seats in {random.choice(classes)} class",
        f"Show me {random.choice(seat_types)} seats",
        f"Seat map for flight {random.choice(flights)}",
        "Can I upgrade my seat",
        "Empty seats on my flight",
        "Seat selection options",
        f"Available seats on {random.choice(flights)}",
        "What seats can I choose"
    ]
    
    for i in range(count):
        if i < len(templates):
            text = templates[i]
        else:
            text = random.choice(templates)
        
        examples.append({
            "text": text,
            "intent": RequestType.SEAT_AVAILABILITY.value,
            "entities": []
        })
    
    return examples


def generate_cancellation_policy_examples(count=2000):
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
        "What's the cancellation fee",
        "Can I cancel for free",
        "Refund terms",
        "Cancellation charges",
        "What happens if I cancel"
    ]
    
    for i in range(count):
        if i < len(templates):
            text = templates[i]
        else:
            text = random.choice(templates)
        
        examples.append({
            "text": text,
            "intent": RequestType.CANCELLATION_POLICY.value,
            "entities": []
        })
    
    return examples


def generate_pet_travel_examples(count=2000):
    """Generate pet travel training examples"""
    examples = []
    
    pet_types = ["dog", "cat", "service animal", "emotional support animal", "bird"]
    
    templates = [
        "Can I bring my pet on the flight",
        "Pet travel policy",
        f"Flying with my {random.choice(pet_types)}",
        "What are the rules for pets",
        "Pet carrier requirements",
        f"Can {random.choice(pet_types)}s fly in cabin",
        "Pet documentation needed",
        f"{random.choice(pet_types)} policy",
        "Pet fees and charges",
        f"{random.choice(pet_types)} rules",
        "Traveling with pets",
        "Pet restrictions",
        f"Can I bring my {random.choice(pet_types)}"
    ]
    
    for i in range(count):
        if i < len(templates):
            text = templates[i]
        else:
            text = random.choice(templates)
        
        examples.append({
            "text": text,
            "intent": RequestType.PET_TRAVEL.value,
            "entities": []
        })
    
    return examples


def main():
    """Generate and save training dataset"""
    print("Starting dataset generation...")
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Generate examples for each intent
    all_examples = []
    
    print("Generating cancel trip examples...")
    all_examples.extend(generate_cancel_trip_examples(2000))
    
    print("Generating flight status examples...")
    all_examples.extend(generate_flight_status_examples(2000))
    
    print("Generating seat availability examples...")
    all_examples.extend(generate_seat_availability_examples(2000))
    
    print("Generating cancellation policy examples...")
    all_examples.extend(generate_cancellation_policy_examples(2000))
    
    print("Generating pet travel examples...")
    all_examples.extend(generate_pet_travel_examples(2000))
    
    # Shuffle the dataset
    random.shuffle(all_examples)
    
    # Save dataset
    output_path = output_dir / "training_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset generation complete!")
    print(f"Total examples: {len(all_examples)}")
    print(f"Saved to: {output_path}")
    
    # Count examples by intent
    intent_counts = {}
    for example in all_examples:
        intent = example["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\nExamples per intent:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    main()