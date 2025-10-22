#!/usr/bin/env python3
"""
Standalone classifier training (no external dependencies)
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import Counter
import re
import time
from enum import Enum


class RequestType(str, Enum):
    CANCEL_TRIP = "cancel_trip"
    CANCELLATION_POLICY = "cancellation_policy"
    FLIGHT_STATUS = "flight_status"
    SEAT_AVAILABILITY = "seat_availability"
    PET_TRAVEL = "pet_travel"


class ClassificationResult:
    def __init__(self, request_type, confidence, alternatives):
        self.request_type = request_type
        self.confidence = confidence
        self.alternative_intents = alternatives


class SimpleClassifier:
    """Simple Naive Bayes classifier"""
    
    def __init__(self):
        self.vocabulary = {}
        self.class_probs = {}
        self.word_probs = {}
        self.classes = []
        self.trained = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from texts"""
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        
        # Keep words that appear at least 2 times
        self.vocabulary = {word: idx for idx, (word, count) in enumerate(word_counts.items()) if count >= 2}
    
    def train(self, dataset: List[Dict]) -> Dict[str, float]:
        """Train the classifier"""
        
        # Prepare data
        texts = [example['text'] for example in dataset]
        labels = [example['intent'] for example in dataset]
        
        self.classes = sorted(list(set(labels)))
        
        # Build vocabulary
        self._build_vocabulary(texts)
        
        # Calculate class probabilities
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        for class_name in self.classes:
            self.class_probs[class_name] = label_counts[class_name] / total_samples
        
        # Calculate word probabilities for each class
        self.word_probs = {}
        
        for class_name in self.classes:
            class_texts = [texts[i] for i, label in enumerate(labels) if label == class_name]
            
            # Count words in this class
            word_counts = Counter()
            for text in class_texts:
                words = self._tokenize(text)
                word_counts.update(words)
            
            # Calculate probabilities with Laplace smoothing
            total_words = sum(word_counts.values())
            vocab_size = len(self.vocabulary)
            
            self.word_probs[class_name] = {}
            for word in self.vocabulary:
                count = word_counts.get(word, 0)
                # Laplace smoothing
                prob = (count + 1) / (total_words + vocab_size)
                self.word_probs[class_name][word] = prob
        
        self.trained = True
        
        # Evaluate on training data
        correct = 0
        predictions = []
        
        for i, text in enumerate(texts):
            result = self.predict(text)
            predicted_class = result.request_type.value
            true_class = labels[i]
            
            predictions.append(predicted_class)
            if predicted_class == true_class:
                correct += 1
        
        accuracy = correct / len(texts)
        
        # Calculate per-class metrics
        metrics = {}
        for class_name in self.classes:
            tp = sum(1 for i, pred in enumerate(predictions) if pred == class_name and labels[i] == class_name)
            fp = sum(1 for i, pred in enumerate(predictions) if pred == class_name and labels[i] != class_name)
            fn = sum(1 for i, pred in enumerate(predictions) if pred != class_name and labels[i] == class_name)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        metrics['overall_accuracy'] = accuracy
        return metrics
    
    def predict(self, text: str) -> ClassificationResult:
        """Predict class for text"""
        
        if not self.trained:
            raise ValueError("Model not trained")
        
        words = self._tokenize(text)
        
        # Calculate log probabilities for each class
        class_scores = {}
        
        for class_name in self.classes:
            # Start with log of class probability
            log_prob = np.log(self.class_probs[class_name])
            
            # Add log probabilities of words
            for word in words:
                if word in self.vocabulary:
                    word_prob = self.word_probs[class_name].get(word, 1e-10)
                    log_prob += np.log(word_prob)
            
            class_scores[class_name] = log_prob
        
        # Find best class
        best_class = max(class_scores, key=class_scores.get)
        
        # Convert to probabilities (approximate)
        max_score = max(class_scores.values())
        exp_scores = {cls: np.exp(score - max_score) for cls, score in class_scores.items()}
        total_exp = sum(exp_scores.values())
        
        probabilities = {cls: exp_score / total_exp for cls, exp_score in exp_scores.items()}
        
        # Get alternatives
        sorted_classes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        alternatives = [{"type": cls, "confidence": prob} for cls, prob in sorted_classes[1:4]]
        
        return ClassificationResult(
            request_type=RequestType(best_class),
            confidence=float(probabilities[best_class]),
            alternatives=alternatives
        )
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'vocabulary': self.vocabulary,
            'class_probs': self.class_probs,
            'word_probs': self.word_probs,
            'classes': self.classes,
            'trained': self.trained
        }
        
        with open(save_dir / 'classifier.pkl', 'wb') as f:
            pickle.dump(model_data, f)


def benchmark_performance(classifier, test_texts: List[str], iterations: int = 100) -> Dict[str, float]:
    """Benchmark model performance"""
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        for text in test_texts:
            classifier.predict(text)
    
    total_time = time.time() - start_time
    avg_latency = (total_time / (iterations * len(test_texts))) * 1000  # ms
    throughput = (iterations * len(test_texts)) / total_time  # requests/sec
    
    return {
        "average_latency_ms": avg_latency,
        "throughput_rps": throughput,
        "total_predictions": iterations * len(test_texts),
        "total_time_seconds": total_time
    }


def main():
    """Train classifier"""
    
    print("🚀 Training Simple Classifier for Airline Customer Service")
    print("=" * 60)
    
    # Paths
    dataset_path = "data/final_training_dataset.json"
    model_save_path = "models/simple-classifier"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run the dataset generation script first.")
        return
    
    try:
        # Load dataset
        print(f"\n📊 Loading dataset from {dataset_path}...")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        print(f"Dataset size: {len(dataset)} examples")
        
        # Count examples per intent
        intent_counts = {}
        for example in dataset:
            intent = example['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print("\nIntent distribution:")
        for intent, count in intent_counts.items():
            print(f"  {intent}: {count}")
        
        # Initialize and train classifier
        print(f"\n🏋️ Training Simple Naive Bayes classifier...")
        classifier = SimpleClassifier()
        
        # Train model
        metrics = classifier.train(dataset)
        
        print(f"\n✅ Training completed!")
        print(f"Overall accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
        
        # Save model
        print(f"\n💾 Saving model to {model_save_path}...")
        classifier.save_model(model_save_path)
        
        # Test predictions
        print(f"\n🧪 Testing predictions...")
        test_examples = [
            "I want to cancel my flight",
            "What's my flight status", 
            "Show available seats",
            "What's your cancellation policy",
            "Can I bring my pet on the flight"
        ]
        
        for text in test_examples:
            result = classifier.predict(text)
            print(f"  '{text}' → {result.request_type.value} (confidence: {result.confidence:.3f})")
        
        # Performance benchmark
        print(f"\n⚡ Performance benchmark...")
        performance = benchmark_performance(classifier, test_examples, iterations=100)
        print(f"  Average latency: {performance['average_latency_ms']:.2f}ms")
        print(f"  Throughput: {performance['throughput_rps']:.1f} requests/sec")
        
        # Detailed per-class results
        print(f"\n📊 Detailed Results:")
        for intent in classifier.classes:
            if intent in metrics:
                m = metrics[intent]
                print(f"  {intent}:")
                print(f"    Precision: {m['precision']:.3f}")
                print(f"    Recall: {m['recall']:.3f}")
                print(f"    F1-Score: {m['f1']:.3f}")
        
        # Create and save report
        report = f"""# Simple Classifier Training Report

## Model Performance

### Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)

### Per-Class Performance:
"""
        
        for intent in classifier.classes:
            if intent in metrics:
                m = metrics[intent]
                report += f"- **{intent}**: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['f1']:.3f}\n"
        
        report += f"""
### Performance Benchmarks:
- **Average Latency**: {performance['average_latency_ms']:.2f}ms
- **Throughput**: {performance['throughput_rps']:.1f} requests/second
- **Target <100ms latency**: {'✅ PASS' if performance['average_latency_ms'] < 100 else '❌ FAIL'}
- **Target >95% accuracy**: {'✅ PASS' if metrics['overall_accuracy'] > 0.95 else '❌ FAIL'}

### Model Information:
- **Algorithm**: Naive Bayes with Laplace smoothing
- **Vocabulary Size**: {len(classifier.vocabulary)}
- **Number of Classes**: {len(classifier.classes)}
- **Classes**: {', '.join(classifier.classes)}
"""
        
        # Save report
        report_path = Path(model_save_path) / "training_report.md"
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📋 Training report saved to {report_path}")
        
        # Summary
        print(f"\n🎉 Training Summary:")
        print(f"  ✅ Model trained and saved")
        print(f"  ✅ Accuracy: {metrics['overall_accuracy']*100:.2f}%")
        print(f"  ✅ Latency: {performance['average_latency_ms']:.2f}ms")
        print(f"  ✅ Vocabulary size: {len(classifier.vocabulary)}")
        print(f"  ✅ Target >95% accuracy: {'PASS' if metrics['overall_accuracy'] > 0.95 else 'FAIL'}")
        print(f"  ✅ Target <100ms latency: {'PASS' if performance['average_latency_ms'] < 100 else 'FAIL'}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()