"""
Simple classifier implementation for demonstration (without heavy ML dependencies)
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import Counter
import re
import time

from ..types import RequestType, ClassificationResult, ModelMetrics


class SimpleNBClassifier:
    """Simple Naive Bayes classifier for demonstration"""
    
    def __init__(self):
        self.vocabulary = {}
        self.class_probs = {}
        self.word_probs = {}
        self.classes = []
        self.trained = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and extract words
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
    
    def train(self, dataset: List[Dict]) -> ModelMetrics:
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
        
        # Evaluate on training data (simple evaluation)
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
        precision_dict = {}
        recall_dict = {}
        f1_dict = {}
        
        for class_name in self.classes:
            # True positives, false positives, false negatives
            tp = sum(1 for i, pred in enumerate(predictions) if pred == class_name and labels[i] == class_name)
            fp = sum(1 for i, pred in enumerate(predictions) if pred == class_name and labels[i] != class_name)
            fn = sum(1 for i, pred in enumerate(predictions) if pred != class_name and labels[i] == class_name)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_dict[RequestType(class_name)] = precision
            recall_dict[RequestType(class_name)] = recall
            f1_dict[RequestType(class_name)] = f1
        
        # Create confusion matrix
        confusion_matrix = [[0 for _ in self.classes] for _ in self.classes]
        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        
        for i, pred in enumerate(predictions):
            true_idx = class_to_idx[labels[i]]
            pred_idx = class_to_idx[pred]
            confusion_matrix[true_idx][pred_idx] += 1
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            confusion_matrix=confusion_matrix
        )
        
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
        
        # Get top alternatives
        sorted_classes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        alternatives = []
        for class_name, prob in sorted_classes[1:4]:  # Top 3 alternatives
            alternatives.append({
                "type": RequestType(class_name),
                "confidence": float(prob)
            })
        
        return ClassificationResult(
            request_type=RequestType(best_class),
            confidence=float(probabilities[best_class]),
            alternative_intents=alternatives,
            extracted_entities=[]
        )
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model"""
        
        if not self.trained:
            raise ValueError("No trained model to save")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'vocabulary': self.vocabulary,
            'class_probs': self.class_probs,
            'word_probs': self.word_probs,
            'classes': self.classes,
            'trained': self.trained
        }
        
        with open(save_dir / 'simple_classifier.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model"""
        
        model_file = Path(model_path) / 'simple_classifier.pkl'
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocabulary = model_data['vocabulary']
        self.class_probs = model_data['class_probs']
        self.word_probs = model_data['word_probs']
        self.classes = model_data['classes']
        self.trained = model_data['trained']
        
        print(f"Model loaded from {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        
        return {
            "status": "trained" if self.trained else "not_trained",
            "model_type": "Naive Bayes",
            "vocabulary_size": len(self.vocabulary),
            "num_classes": len(self.classes),
            "classes": self.classes
        }


class SimpleModelPipeline:
    """Simple training pipeline"""
    
    def __init__(self):
        self.classifier = SimpleNBClassifier()
    
    def train_and_evaluate(self, dataset_path: str, model_save_path: str = None) -> ModelMetrics:
        """Train and evaluate the simple classifier"""
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"Loaded {len(dataset)} examples")
        
        # Train model
        metrics = self.classifier.train(dataset)
        
        # Save model if path provided
        if model_save_path:
            self.classifier.save_model(model_save_path)
        
        return metrics
    
    def benchmark_performance(self, test_texts: List[str], iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        
        if not self.classifier.trained:
            raise ValueError("Model not trained")
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            for text in test_texts:
                self.classifier.predict(text)
        
        total_time = time.time() - start_time
        avg_latency = (total_time / (iterations * len(test_texts))) * 1000  # ms
        throughput = (iterations * len(test_texts)) / total_time  # requests/sec
        
        return {
            "average_latency_ms": avg_latency,
            "throughput_rps": throughput,
            "total_predictions": iterations * len(test_texts),
            "total_time_seconds": total_time
        }
    
    def create_model_report(self, metrics: ModelMetrics, performance: Dict[str, float]) -> str:
        """Create model report"""
        
        report = f"""
# Simple Classifier Report

## Model Performance

### Accuracy Metrics
- **Overall Accuracy**: {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)

### Per-Class Performance
"""
        
        for intent, precision in metrics.precision.items():
            recall = metrics.recall.get(intent, 0.0)
            f1 = metrics.f1_score.get(intent, 0.0)
            report += f"- **{intent.value}**: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n"
        
        report += f"""
### Performance Benchmarks
- **Average Latency**: {performance['average_latency_ms']:.2f}ms
- **Throughput**: {performance['throughput_rps']:.1f} requests/second
- **Target Latency**: <100ms ({'✅ PASS' if performance['average_latency_ms'] < 100 else '❌ FAIL'})
- **Target Accuracy**: >95% ({'✅ PASS' if metrics.accuracy > 0.95 else '❌ FAIL'})

### Model Information
{json.dumps(self.classifier.get_model_info(), indent=2)}
"""
        
        return report