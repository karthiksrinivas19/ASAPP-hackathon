#!/usr/bin/env python3
"""
Train simple classifier for airline customer service (no heavy dependencies)
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from airline_service.ml.simple_classifier import SimpleModelPipeline


def main():
    """Train simple classifier"""
    
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
    
    # Create model directory
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize training pipeline
        print("\n📚 Initializing training pipeline...")
        pipeline = SimpleModelPipeline()
        
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
        
        # Train model
        print(f"\n🏋️ Training Simple Naive Bayes classifier...")
        print("Training parameters:")
        print("  - Algorithm: Naive Bayes with Laplace smoothing")
        print("  - Features: Bag of words")
        print("  - Vocabulary: Words appearing ≥2 times")
        
        # Train and evaluate
        metrics = pipeline.train_and_evaluate(dataset_path, model_save_path)
        
        print(f"\n✅ Training completed!")
        print(f"Final accuracy: {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
        
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
            result = pipeline.classifier.predict(text)
            print(f"  '{text}' → {result.request_type.value} (confidence: {result.confidence:.3f})")
        
        # Performance benchmark
        print(f"\n⚡ Performance benchmark...")
        performance = pipeline.benchmark_performance(test_examples, iterations=100)
        print(f"  Average latency: {performance['average_latency_ms']:.2f}ms")
        print(f"  Throughput: {performance['throughput_rps']:.1f} requests/sec")
        
        # Create report
        report = pipeline.create_model_report(metrics, performance)
        
        # Save report
        report_path = Path(model_save_path) / "training_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📋 Training report saved to {report_path}")
        
        # Detailed per-class results
        print(f"\n📊 Detailed Results:")
        for intent, precision in metrics.precision.items():
            recall = metrics.recall.get(intent, 0.0)
            f1 = metrics.f1_score.get(intent, 0.0)
            print(f"  {intent.value}:")
            print(f"    Precision: {precision:.3f}")
            print(f"    Recall: {recall:.3f}")
            print(f"    F1-Score: {f1:.3f}")
        
        # Summary
        print(f"\n🎉 Training Summary:")
        print(f"  ✅ Model trained and saved")
        print(f"  ✅ Accuracy: {metrics.accuracy*100:.2f}%")
        print(f"  ✅ Latency: {performance['average_latency_ms']:.2f}ms")
        print(f"  ✅ Vocabulary size: {len(pipeline.classifier.vocabulary)}")
        print(f"  ✅ Target >95% accuracy: {'PASS' if metrics.accuracy > 0.95 else 'FAIL'}")
        print(f"  ✅ Target <100ms latency: {'PASS' if performance['average_latency_ms'] < 100 else 'FAIL'}")
        
        # Model info
        model_info = pipeline.classifier.get_model_info()
        print(f"\n📋 Model Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()