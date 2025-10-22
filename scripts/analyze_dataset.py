#!/usr/bin/env python3
"""
Analyze the generated training dataset
"""

import json
from pathlib import Path
from collections import Counter
import re


def analyze_dataset(dataset_path):
    """Analyze the training dataset"""
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Dataset Analysis: {dataset_path}")
    print("=" * 50)
    
    # Basic statistics
    print(f"Total examples: {len(data)}")
    
    # Intent distribution
    intents = [example['intent'] for example in data]
    intent_counts = Counter(intents)
    
    print("\nIntent Distribution:")
    for intent, count in intent_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {intent}: {count} ({percentage:.1f}%)")
    
    # Text length statistics
    text_lengths = [len(example['text']) for example in data]
    
    print(f"\nText Length Statistics:")
    print(f"  Average length: {sum(text_lengths) / len(text_lengths):.1f} characters")
    print(f"  Min length: {min(text_lengths)} characters")
    print(f"  Max length: {max(text_lengths)} characters")
    
    # Sample examples for each intent
    print(f"\nSample Examples by Intent:")
    for intent in intent_counts.keys():
        examples = [ex['text'] for ex in data if ex['intent'] == intent]
        print(f"\n{intent.upper()}:")
        for i, example in enumerate(examples[:3]):
            print(f"  {i+1}. {example}")
    
    # Word frequency analysis
    all_text = ' '.join([example['text'].lower() for example in data])
    words = re.findall(r'\b\w+\b', all_text)
    word_counts = Counter(words)
    
    print(f"\nMost Common Words:")
    for word, count in word_counts.most_common(20):
        print(f"  {word}: {count}")
    
    # Entity analysis (if entities exist)
    entity_counts = Counter()
    for example in data:
        for entity in example.get('entities', []):
            entity_counts[entity.get('type', 'unknown')] += 1
    
    if entity_counts:
        print(f"\nEntity Distribution:")
        for entity_type, count in entity_counts.items():
            print(f"  {entity_type}: {count}")
    else:
        print(f"\nNo entities found in dataset")


def main():
    """Main function"""
    dataset_path = Path("data/training_dataset.json")
    
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please run the dataset generation script first.")
        return
    
    analyze_dataset(dataset_path)


if __name__ == "__main__":
    main()