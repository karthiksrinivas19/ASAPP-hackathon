#!/usr/bin/env python3
"""
Simple test for classifier without dependencies
"""

import pickle
import numpy as np
from pathlib import Path
import re
import time
import asyncio


class SimpleClassifierTest:
    """Simple classifier for testing"""
    
    def __init__(self, model_path):
        self.model_data = None
        self.loaded = False
        
        # Try to load model
        model_file = Path(model_path) / 'classifier.pkl'
        if model_file.exists():
            with open(model_file, 'rb') as f:
                self.model_data = pickle.load(f)
            self.loaded = True
    
    def _tokenize(self, text):
        """Tokenize text"""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    async def classify_request(self, utterance):
        """Classify request"""
        
        if not self.loaded:
            # Return mock result
            return {
                'request_type': 'unknown',
                'confidence': 0.5,
                'entities': []
            }
        
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
        
        return {
            'request_type': best_class,
            'confidence': probabilities[best_class],
            'entities': []  # Simplified for test
        }


async def main():
    """Test the classifier"""
    
    print("🧪 Simple Classifier Test")
    print("=" * 30)
    
    # Initialize classifier
    classifier = SimpleClassifierTest("models/simple-classifier")
    
    if classifier.loaded:
        print("✅ Model loaded successfully")
        print(f"Classes: {classifier.model_data['classes']}")
        print(f"Vocabulary size: {len(classifier.model_data['vocabulary'])}")
    else:
        print("⚠️  Model not found, using mock responses")
    
    # Test cases
    test_cases = [
        "I want to cancel my flight",
        "What's my flight status",
        "Show available seats",
        "What's your cancellation policy",
        "Can I bring my pet",
        "I need help with my booking"
    ]
    
    print(f"\n🔍 Testing {len(test_cases)} examples:")
    print("-" * 40)
    
    for i, utterance in enumerate(test_cases, 1):
        result = await classifier.classify_request(utterance)
        
        print(f"{i}. '{utterance}'")
        print(f"   → {result['request_type']} (confidence: {result['confidence']:.3f})")
        print()
    
    # Performance test
    if classifier.loaded:
        print("⚡ Performance Test:")
        
        test_text = "I want to cancel my flight"
        iterations = 100
        
        start_time = time.time()
        for _ in range(iterations):
            await classifier.classify_request(test_text)
        
        total_time = time.time() - start_time
        avg_latency = (total_time / iterations) * 1000
        throughput = iterations / total_time
        
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Throughput: {throughput:.1f} requests/sec")
        print(f"  Target <100ms: {'✅ PASS' if avg_latency < 100 else '❌ FAIL'}")
    
    print(f"\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())