#!/usr/bin/env python3
"""
Train DistilBERT classifier for airline customer service
"""

import sys
import json
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Simple logger for training script
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

# Mock the logger import
import airline_service.utils.logger
airline_service.utils.logger.get_logger = lambda name: SimpleLogger()

from airline_service.ml.distilbert_classifier import DistilBERTClassifier, ModelTrainingPipeline


def main():
    """Train DistilBERT classifier"""
    
    print("🚀 Starting DistilBERT Training for Airline Customer Service")
    print("=" * 60)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    dataset_path = "data/final_training_dataset.json"
    model_save_path = "models/distilbert-airline-classifier"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run the dataset generation script first.")
        return
    
    # Create model directory
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize training pipeline
        print("\n📚 Initializing training pipeline...")
        pipeline = ModelTrainingPipeline()
        
        # Load and inspect dataset
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
        
        # Train model with smaller parameters for faster training
        print(f"\n🏋️ Training DistilBERT classifier...")
        print("Training parameters:")
        print("  - Model: distilbert-base-uncased")
        print("  - Max length: 128 tokens")
        print("  - Batch size: 8 (reduced for memory)")
        print("  - Epochs: 2 (reduced for speed)")
        print("  - Learning rate: 2e-5")
        
        # Initialize classifier with smaller batch size
        classifier = DistilBERTClassifier(max_length=128)
        
        # Prepare data
        texts, labels = classifier._prepare_data(dataset)
        
        # Use smaller subset for demo (first 1000 examples)
        if len(texts) > 1000:
            print(f"\n⚡ Using subset of {1000} examples for faster training...")
            texts = texts[:1000]
            labels = labels[:1000]
        
        # Train with reduced parameters
        metrics = classifier.train(
            dataset[:len(texts)],
            validation_split=0.2,
            batch_size=8,  # Smaller batch size
            epochs=2,      # Fewer epochs
            learning_rate=2e-5
        )
        
        print(f"\n✅ Training completed!")
        print(f"Final accuracy: {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
        
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
            "Can I bring my pet"
        ]
        
        for text in test_examples:
            result = classifier.predict(text)
            print(f"  '{text}' → {result.request_type.value} (confidence: {result.confidence:.3f})")
        
        # Performance benchmark
        print(f"\n⚡ Performance benchmark...")
        performance = pipeline.benchmark_performance(test_examples[:2], iterations=10)
        print(f"  Average latency: {performance['average_latency_ms']:.2f}ms")
        print(f"  Throughput: {performance['throughput_rps']:.1f} requests/sec")
        
        # Create report
        report = pipeline.create_model_report(metrics, performance)
        
        # Save report
        report_path = Path(model_save_path) / "training_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📋 Training report saved to {report_path}")
        
        # Summary
        print(f"\n🎉 Training Summary:")
        print(f"  ✅ Model trained and saved")
        print(f"  ✅ Accuracy: {metrics.accuracy*100:.2f}%")
        print(f"  ✅ Latency: {performance['average_latency_ms']:.2f}ms")
        print(f"  ✅ Target >95% accuracy: {'PASS' if metrics.accuracy > 0.95 else 'FAIL'}")
        print(f"  ✅ Target <100ms latency: {'PASS' if performance['average_latency_ms'] < 100 else 'FAIL'}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()