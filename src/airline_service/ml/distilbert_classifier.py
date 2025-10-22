"""
DistilBERT-based request classifier for airline customer service
"""

import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import time

from ..types import RequestType, TrainingExample, ClassificationResult, ExtractedEntity, ModelMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AirlineDataset(Dataset):
    """PyTorch Dataset for airline service requests"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DistilBERTClassifier:
    """DistilBERT-based request classifier"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized DistilBERT classifier with device: {self.device}")
    
    def _setup_labels(self, labels: List[str]) -> None:
        """Setup label mappings"""
        unique_labels = sorted(list(set(labels)))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        logger.info(f"Label mappings: {self.label_to_id}")
    
    def _prepare_data(self, dataset: List[Dict]) -> Tuple[List[str], List[int]]:
        """Prepare data for training"""
        texts = []
        labels = []
        
        for example in dataset:
            texts.append(example['text'])
            labels.append(example['intent'])
        
        # Setup label mappings
        self._setup_labels(labels)
        
        # Convert labels to integers
        label_ids = [self.label_to_id[label] for label in labels]
        
        return texts, label_ids
    
    def train(self, dataset: List[Dict], validation_split: float = 0.2, 
              batch_size: int = 16, epochs: int = 3, learning_rate: float = 2e-5) -> ModelMetrics:
        """Train the DistilBERT classifier"""
        
        logger.info("Starting DistilBERT training...")
        
        # Prepare data
        texts, labels = self._prepare_data(dataset)
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        logger.info(f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_to_id)
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Create datasets
        train_dataset = AirlineDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = AirlineDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./models/distilbert-airline-classifier',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            report_to=None  # Disable wandb/tensorboard
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        metrics = self.evaluate(val_texts, val_labels)
        
        return metrics
    
    def evaluate(self, texts: List[str], true_labels: List[int]) -> ModelMetrics:
        """Evaluate the model performance"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained or loaded")
        
        logger.info("Evaluating model...")
        
        # Get predictions
        predictions = []
        confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                result = self._predict_single(text)
                predictions.append(self.label_to_id[result.request_type.value])
                confidences.append(result.confidence)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # Create per-class metrics
        precision_dict = {}
        recall_dict = {}
        f1_dict = {}
        
        for label_id, label_name in self.id_to_label.items():
            if label_id < len(precision):
                precision_dict[RequestType(label_name)] = float(precision[label_id])
                recall_dict[RequestType(label_name)] = float(recall[label_id])
                f1_dict[RequestType(label_name)] = float(f1[label_id])
        
        metrics = ModelMetrics(
            accuracy=float(accuracy),
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            confusion_matrix=conf_matrix.tolist()
        )
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Average confidence: {np.mean(confidences):.4f}")
        
        return metrics
    
    def _predict_single(self, text: str) -> ClassificationResult:
        """Predict single text input"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained or loaded")
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(3, len(self.id_to_label)))
            
            # Primary prediction
            predicted_id = top_indices[0][0].item()
            confidence = top_probs[0][0].item()
            predicted_label = self.id_to_label[predicted_id]
            
            # Alternative predictions
            alternatives = []
            for i in range(1, len(top_indices[0])):
                alt_id = top_indices[0][i].item()
                alt_confidence = top_probs[0][i].item()
                alt_label = self.id_to_label[alt_id]
                alternatives.append({
                    "type": RequestType(alt_label),
                    "confidence": float(alt_confidence)
                })
        
        return ClassificationResult(
            request_type=RequestType(predicted_label),
            confidence=float(confidence),
            alternative_intents=alternatives,
            extracted_entities=[]  # Will be filled by entity extractor
        )
    
    def predict(self, text: str) -> ClassificationResult:
        """Public prediction method with timing"""
        start_time = time.time()
        result = self._predict_single(text)
        prediction_time = time.time() - start_time
        
        logger.debug(f"Prediction completed in {prediction_time*1000:.2f}ms")
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Predict batch of texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def save_model(self, save_path: str) -> None:
        """Save the trained model"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model to save")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save label mappings
        label_config = {
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label,
            'max_length': self.max_length
        }
        
        with open(save_dir / 'label_config.json', 'w') as f:
            json.dump(label_config, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model"""
        
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Load label mappings
        with open(model_dir / 'label_config.json', 'r') as f:
            label_config = json.load(f)
        
        self.label_to_id = label_config['label_to_id']
        self.id_to_label = {int(k): v for k, v in label_config['id_to_label'].items()}
        self.max_length = label_config['max_length']
        
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        
        # Move to device
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.model_name,
            "num_labels": len(self.label_to_id),
            "labels": list(self.label_to_id.keys()),
            "max_length": self.max_length,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters())
        }


class ModelTrainingPipeline:
    """Complete training pipeline for DistilBERT classifier"""
    
    def __init__(self):
        self.classifier = DistilBERTClassifier()
        self.logger = get_logger(__name__)
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load training dataset"""
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        self.logger.info(f"Loaded {len(dataset)} examples from {dataset_path}")
        return dataset
    
    def train_and_evaluate(self, dataset_path: str, model_save_path: str = None) -> ModelMetrics:
        """Complete training and evaluation pipeline"""
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Train model
        metrics = self.classifier.train(dataset)
        
        # Save model if path provided
        if model_save_path:
            self.classifier.save_model(model_save_path)
        
        return metrics
    
    def benchmark_performance(self, test_texts: List[str], iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        
        if self.classifier.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Warm up
        for _ in range(10):
            self.classifier.predict(test_texts[0])
        
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
        """Create comprehensive model report"""
        
        report = f"""
# DistilBERT Airline Classifier Report

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