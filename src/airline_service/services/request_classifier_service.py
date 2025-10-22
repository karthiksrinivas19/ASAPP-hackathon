"""
Request classifier service implementation
"""

import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import time

from ..types import RequestType, ClassificationResult, ExtractedEntity
from ..interfaces.request_classifier import RequestClassifierInterface
from ..ml.entity_extractor import EntityExtractor


class RequestClassifierService(RequestClassifierInterface):
    """Production request classifier service"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model_data = None
        self.entity_extractor = EntityExtractor()
        self.loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for classification"""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    async def classify_request(self, utterance: str) -> ClassificationResult:
        """Classify customer utterance into request type"""
        
        if not self.loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Get text classification
        words = self._tokenize(utterance)
        
        # Calculate log probabilities for each class
        class_scores = {}
        
        for class_name in self.model_data['classes']:
            # Start with log of class probability
            log_prob = np.log(self.model_data['class_probs'][class_name])
            
            # Add log probabilities of words
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
        
        # Get alternatives
        sorted_classes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        alternatives = []
        for class_name, prob in sorted_classes[1:4]:  # Top 3 alternatives
            alternatives.append({
                "type": RequestType(class_name),
                "confidence": float(prob)
            })
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(utterance)
        
        classification_time = time.time() - start_time
        
        return ClassificationResult(
            request_type=RequestType(best_class),
            confidence=float(probabilities[best_class]),
            alternative_intents=alternatives,
            extracted_entities=entities
        )
    
    def get_confidence_score(self, utterance: str, request_type: RequestType) -> float:
        """Get confidence score for specific request type"""
        
        if not self.loaded:
            return 0.0
        
        words = self._tokenize(utterance)
        class_name = request_type.value
        
        if class_name not in self.model_data['classes']:
            return 0.0
        
        # Calculate log probability for this class
        log_prob = np.log(self.model_data['class_probs'][class_name])
        
        for word in words:
            if word in self.model_data['vocabulary']:
                word_prob = self.model_data['word_probs'][class_name].get(word, 1e-10)
                log_prob += np.log(word_prob)
        
        # Convert to approximate probability (simplified)
        return min(1.0, max(0.0, np.exp(log_prob)))
    
    async def train_model(self, training_data: List[Any]) -> None:
        """Train model (not implemented for production service)"""
        raise NotImplementedError("Training should be done offline")
    
    async def load_model(self, model_path: str) -> None:
        """Load pre-trained model"""
        
        model_file = Path(model_path) / 'classifier.pkl'
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.model_path = model_path
        self.loaded = True
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        
        if not self.loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "vocabulary_size": len(self.model_data['vocabulary']),
            "num_classes": len(self.model_data['classes']),
            "classes": self.model_data['classes'],
            "algorithm": "Naive Bayes"
        }
    
    async def predict_batch(self, utterances: List[str]) -> List[ClassificationResult]:
        """Classify multiple utterances"""
        
        results = []
        for utterance in utterances:
            result = await self.classify_request(utterance)
            results.append(result)
        
        return results
    
    def get_supported_intents(self) -> List[RequestType]:
        """Get list of supported request types"""
        
        if not self.loaded:
            return []
        
        return [RequestType(class_name) for class_name in self.model_data['classes']]


class ClassifierFactory:
    """Factory for creating classifier instances"""
    
    @staticmethod
    def create_classifier(model_path: str = "models/simple-classifier") -> RequestClassifierService:
        """Create and load classifier instance"""
        
        classifier = RequestClassifierService()
        
        try:
            # Try to load the model synchronously for factory method
            model_file = Path(model_path) / 'classifier.pkl'
            
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    classifier.model_data = pickle.load(f)
                classifier.model_path = model_path
                classifier.loaded = True
            else:
                print(f"Warning: Model not found at {model_path}")
        
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
        
        return classifier
    
    @staticmethod
    def create_mock_classifier() -> RequestClassifierService:
        """Create mock classifier for testing"""
        
        # Create minimal mock model data
        mock_data = {
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
        
        classifier = RequestClassifierService()
        classifier.model_data = mock_data
        classifier.loaded = True
        
        return classifier


class MockClassifier(RequestClassifierInterface):
    """Mock classifier for testing and fallback scenarios"""
    
    def __init__(self):
        self.loaded = True
        self.model_info = {
            "status": "loaded",
            "type": "mock",
            "version": "1.0.0"
        }
    
    def is_loaded(self) -> bool:
        return self.loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        return self.model_info
    
    async def load_model(self, model_path: str = None) -> bool:
        """Mock load model - always returns True"""
        self.loaded = True
        return True
    
    def get_confidence_score(self, utterance: str, predicted_class: str) -> float:
        """Mock confidence score calculation"""
        return 0.8
    
    def train_model(self, dataset_path: str, model_save_path: str = None):
        """Mock training - returns dummy metrics"""
        from ..types import ModelMetrics, RequestType
        return ModelMetrics(
            accuracy=0.95,
            precision={RequestType.CANCEL_TRIP: 0.95, RequestType.FLIGHT_STATUS: 0.95},
            recall={RequestType.CANCEL_TRIP: 0.95, RequestType.FLIGHT_STATUS: 0.95},
            f1_score={RequestType.CANCEL_TRIP: 0.95, RequestType.FLIGHT_STATUS: 0.95},
            confusion_matrix=[[10, 0], [0, 10]]
        )
    
    async def classify_request(self, utterance: str) -> ClassificationResult:
        """Simple keyword-based classification for mock"""
        from ..types import RequestType, ClassificationResult, ExtractedEntity, EntityType
        
        utterance_lower = utterance.lower()
        
        # Simple keyword matching
        if any(word in utterance_lower for word in ["cancel", "refund", "cancellation"]):
            request_type = RequestType.CANCEL_TRIP
            confidence = 0.8
        elif any(word in utterance_lower for word in ["status", "delayed", "on time", "departure"]):
            request_type = RequestType.FLIGHT_STATUS
            confidence = 0.8
        elif any(word in utterance_lower for word in ["seat", "upgrade", "available"]):
            request_type = RequestType.SEAT_AVAILABILITY
            confidence = 0.8
        elif any(word in utterance_lower for word in ["policy", "rules", "terms"]):
            request_type = RequestType.CANCELLATION_POLICY
            confidence = 0.8
        elif any(word in utterance_lower for word in ["pet", "animal", "dog", "cat"]):
            request_type = RequestType.PET_TRAVEL
            confidence = 0.8
        else:
            request_type = RequestType.UNKNOWN
            confidence = 0.3
        
        # Simple entity extraction
        entities = []
        
        # Look for PNR patterns (6 characters alphanumeric)
        import re
        pnr_matches = re.findall(r'\b[A-Z0-9]{6}\b', utterance.upper())
        for match in pnr_matches:
            entities.append(ExtractedEntity(
                type=EntityType.PNR,
                value=match,
                confidence=0.9,
                start_index=utterance.upper().find(match),
                end_index=utterance.upper().find(match) + len(match)
            ))
        
        # Look for flight numbers (2-3 letters + 3-4 digits)
        flight_matches = re.findall(r'\b[A-Z]{2,3}\s*\d{3,4}\b', utterance.upper())
        for match in flight_matches:
            entities.append(ExtractedEntity(
                type=EntityType.FLIGHT_NUMBER,
                value=match.replace(' ', ''),
                confidence=0.9,
                start_index=utterance.upper().find(match),
                end_index=utterance.upper().find(match) + len(match)
            ))
        
        return ClassificationResult(
            request_type=request_type,
            confidence=confidence,
            alternative_intents=[],
            extracted_entities=entities
        )