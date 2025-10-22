#!/usr/bin/env python3
"""
Script to generate training dataset for airline customer service classification
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from airline_service.ml.dataset_generator import TrainingDatasetBuilder
from airline_service.ml.data_augmentation import DataAugmentation
from airline_service.utils.logger import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Generate and save training dataset"""
    logger.info("Starting dataset generation...")
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize dataset builder
    builder = TrainingDatasetBuilder()
    augmenter = DataAugmentation()
    
    # Generate base dataset
    logger.info("Generating base dataset...")
    base_examples = builder.generate_complete_dataset()
    
    # Save base dataset
    base_path = output_dir / "base_dataset.json"
    builder.save_dataset(base_examples, str(base_path))
    
    # Apply data augmentation
    logger.info("Applying data augmentation...")
    augmented_examples = augmenter.augment_dataset(base_examples)
    
    # Save augmented dataset
    augmented_path = output_dir / "augmented_dataset.json"
    builder.save_dataset(augmented_examples, str(augmented_path))
    
    # Generate statistics
    logger.info("Dataset generation complete!")
    logger.info(f"Base dataset: {len(base_examples)} examples")
    logger.info(f"Augmented dataset: {len(augmented_examples)} examples")
    
    # Count examples by intent
    intent_counts = {}
    for example in augmented_examples:
        intent = example.intent.value
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    logger.info("Examples per intent:")
    for intent, count in intent_counts.items():
        logger.info(f"  {intent}: {count}")
    
    # Count entities
    entity_counts = {}
    for example in augmented_examples:
        for entity in example.entities:
            entity_type = entity.type.value
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    logger.info("Entities extracted:")
    for entity_type, count in entity_counts.items():
        logger.info(f"  {entity_type}: {count}")


if __name__ == "__main__":
    main()