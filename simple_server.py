#!/usr/bin/env python3
"""
Simple airline customer service API server (no external dependencies)
"""

import sys
import json
import pickle
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not available. Install with: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False


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
    SEAT_TYPE = "seat_type"
    CLASS = "class"
    PET_TYPE = "pet_type"


class CustomerRequest(BaseModel):
    utterance: str = Field(..., description="Customer's query text")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    customer_id: Optional[str] = Field(None, description="Optional customer identifier")


class APIResponse(BaseModel):
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.now)


class SimpleClassifier:
    """Simple classifier for demonstration"""
    
    def __init__(self):
        self.model_data = None
        self.loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load model if available"""
        model_path = Path("models/simple-classifier/classifier.pkl")
        
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    self.model_data = pickle.load(f)
                self.loaded = True
                print("✅ Classifier model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                self._create_mock_model()
        else:
            print("⚠️ Model not found, using mock classifier")
            self._create_mock_model()
    
    def _create_mock_model(self):
        """Create mock model for demonstration"""
        self.model_data = {
            'classes': ['cancel_trip', 'flight_status', 'seat_availability', 'cancellation_policy', 'pet_travel'],
            'class_probs': {
                'cancel_trip': 0.2,
                'flight_status': 0.2,
                'seat_availability': 0.2,
                'cancellation_policy': 0.2,
                'pet_travel': 0.2
            },
            'vocabulary': {'cancel': 0, 'flight': 1, 'status': 2, 'seat': 3, 'policy': 4, 'pet': 5},
            'word_probs': {
                'cancel_trip': {'cancel': 0.8, 'flight': 0.6, 'status': 0.1, 'seat': 0.1, 'policy': 0.1, 'pet': 0.1},
                'flight_status': {'cancel': 0.1, 'flight': 0.8, 'status': 0.8, 'seat': 0.1, 'policy': 0.1, 'pet': 0.1},
                'seat_availability': {'cancel': 0.1, 'flight': 0.3, 'status': 0.1, 'seat': 0.8, 'policy': 0.1, 'pet': 0.1},
                'cancellation_policy': {'cancel': 0.6, 'flight': 0.2, 'status': 0.1, 'seat': 0.1, 'policy': 0.8, 'pet': 0.1},
                'pet_travel': {'cancel': 0.1, 'flight': 0.3, 'status': 0.1, 'seat': 0.1, 'policy': 0.2, 'pet': 0.8}
            }
        }
        self.loaded = True
    
    def _tokenize(self, text):
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def classify(self, utterance):
        """Classify utterance"""
        if not self.loaded:
            return RequestType.CANCEL_TRIP, 0.5
        
        words = self._tokenize(utterance)
        
        # Calculate probabilities
        class_scores = {}
        
        for class_name in self.model_data['classes']:
            log_prob = np.log(self.model_data['class_probs'][class_name])
            
            for word in words:
                if word in self.model_data['vocabulary']:
                    word_prob = self.model_data['word_probs'][class_name].get(word, 1e-10)
                    log_prob += np.log(word_prob)
            
            class_scores[class_name] = log_prob
        
        # Find best class
        best_class = max(class_scores, key=class_scores.get)
        
        # Convert to probabilities
        max_score = max(class_scores.values())
        exp_scores = {cls: np.exp(score - max_score) for cls, score in class_scores.items()}
        total_exp = sum(exp_scores.values())
        
        probabilities = {cls: exp_score / total_exp for cls, exp_score in exp_scores.items()}
        
        return RequestType(best_class), probabilities[best_class]


class SimpleEntityExtractor:
    """Simple entity extractor"""
    
    def __init__(self):
        self.patterns = {
            EntityType.PNR: r'\b([A-Z0-9]{6})\b',
            EntityType.FLIGHT_NUMBER: r'\b([A-Z]{2,3}[0-9]{1,4})\b',
            EntityType.EMAIL: r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
            EntityType.PHONE_NUMBER: r'\b(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b',
            EntityType.DATE: r'\b(today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        }
    
    def extract_entities(self, text):
        """Extract entities from text"""
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': entity_type.value,
                    'value': match.group(1),
                    'confidence': 0.8
                })
        
        return entities


# Initialize components
classifier = SimpleClassifier()
entity_extractor = SimpleEntityExtractor()

if FASTAPI_AVAILABLE:
    # Create FastAPI app
    app = FastAPI(
        title="Airline Customer Service API",
        description="Simple airline customer service system",
        version="1.0.0"
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "classifier_loaded": classifier.loaded
        }
    
    @app.post("/api/v1/customer-service/query", response_model=APIResponse)
    async def process_customer_query(request: CustomerRequest):
        """Process customer query"""
        try:
            print(f"Processing query: {request.utterance}")
            
            # Classify request
            intent, confidence = classifier.classify(request.utterance)
            
            # Extract entities
            entities = entity_extractor.extract_entities(request.utterance)
            
            # Generate response based on intent
            if intent == RequestType.CANCEL_TRIP:
                message = "I can help you cancel your flight. To proceed with the cancellation, I'll need your booking reference (PNR) or flight details."
                if any(e['type'] == 'pnr' for e in entities):
                    pnr = next(e['value'] for e in entities if e['type'] == 'pnr')
                    message = f"I found your booking reference {pnr}. I can help you cancel this flight. Please note that cancellation fees may apply."
            
            elif intent == RequestType.FLIGHT_STATUS:
                message = "I can check your flight status. Let me look up the current information for your flight."
                if any(e['type'] == 'flight_number' for e in entities):
                    flight = next(e['value'] for e in entities if e['type'] == 'flight_number')
                    message = f"Checking status for flight {flight}. Your flight is currently on time with no delays reported."
            
            elif intent == RequestType.SEAT_AVAILABILITY:
                message = "I can show you available seats on your flight. Let me check the current seat map for available options."
            
            elif intent == RequestType.CANCELLATION_POLICY:
                message = "Our cancellation policy varies by fare type. Generally, you can cancel flights up to 24 hours before departure. Fees may apply depending on your ticket type."
            
            elif intent == RequestType.PET_TRAVEL:
                message = "I can help you with pet travel information. Small pets can travel in the cabin in approved carriers, while larger pets may need to travel as cargo."
            
            else:
                message = "I understand you need assistance. Could you please provide more details about what you'd like help with?"
            
            return APIResponse(
                status="completed",
                message=message,
                data={
                    "intent": intent.value,
                    "confidence": confidence,
                    "entities": entities
                }
            )
            
        except Exception as e:
            print(f"Error processing query: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": "Internal server error occurred",
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def run_server():
        """Run the FastAPI server"""
        print("🚀 Starting Airline Customer Service API")
        print("=" * 40)
        print(f"Classifier loaded: {classifier.loaded}")
        print(f"Server starting on http://localhost:8000")
        print(f"API docs available at: http://localhost:8000/docs")
        print(f"Health check: http://localhost:8000/health")
        
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

else:
    def run_server():
        """Fallback server without FastAPI"""
        print("🚀 Airline Customer Service API (Demo Mode)")
        print("=" * 40)
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        print(f"Classifier loaded: {classifier.loaded}")
        
        # Simple demo
        test_queries = [
            "I want to cancel my flight",
            "What's my flight status",
            "Show available seats",
            "What's your cancellation policy",
            "Can I bring my pet"
        ]
        
        print("\n🧪 Demo Classification Results:")
        for query in test_queries:
            intent, confidence = classifier.classify(query)
            entities = entity_extractor.extract_entities(query)
            print(f"  '{query}' → {intent.value} (confidence: {confidence:.3f})")
            if entities:
                for entity in entities:
                    print(f"    - {entity['type']}: {entity['value']}")


def main():
    """Main entry point"""
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")


if __name__ == "__main__":
    main()