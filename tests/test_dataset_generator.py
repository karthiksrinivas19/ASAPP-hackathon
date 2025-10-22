"""
Tests for dataset generation
"""

import pytest
from airline_service.ml.dataset_generator import TrainingDatasetBuilder, DatasetGenerator
from airline_service.ml.data_augmentation import DataAugmentation, EntityVariation
from airline_service.ml.entity_extractor import EntityExtractor
from airline_service.types import RequestType, EntityType


class TestDatasetGenerator:
    """Test dataset generation functionality"""
    
    def test_generator_initialization(self):
        """Test generator initializes correctly"""
        generator = DatasetGenerator()
        
        assert len(generator.pnr_patterns) > 0
        assert len(generator.flight_numbers) > 0
        assert len(generator.airport_codes) > 0
        assert len(generator.dates) > 0
    
    def test_pnr_generation(self):
        """Test PNR pattern generation"""
        generator = DatasetGenerator()
        pnrs = generator._generate_pnr_patterns()
        
        assert len(pnrs) == 100
        for pnr in pnrs[:10]:  # Test first 10
            assert len(pnr) == 6
            assert pnr.isalnum()
    
    def test_flight_number_generation(self):
        """Test flight number generation"""
        generator = DatasetGenerator()
        flights = generator._generate_flight_numbers()
        
        assert len(flights) > 0
        for flight in flights[:10]:  # Test first 10
            assert len(flight) >= 3
            assert flight[:2].isalpha()
            assert flight[2:].isdigit()


class TestTrainingDatasetBuilder:
    """Test training dataset builder"""
    
    def test_builder_initialization(self):
        """Test builder initializes correctly"""
        builder = TrainingDatasetBuilder()
        assert builder.generator is not None
    
    def test_cancel_trip_examples(self):
        """Test cancel trip example generation"""
        builder = TrainingDatasetBuilder()
        examples = builder.generate_cancel_trip_examples(10)
        
        assert len(examples) == 10
        for example in examples:
            assert example.intent == RequestType.CANCEL_TRIP
            assert len(example.text) > 0
    
    def test_flight_status_examples(self):
        """Test flight status example generation"""
        builder = TrainingDatasetBuilder()
        examples = builder.generate_flight_status_examples(10)
        
        assert len(examples) == 10
        for example in examples:
            assert example.intent == RequestType.FLIGHT_STATUS
            assert len(example.text) > 0
    
    def test_seat_availability_examples(self):
        """Test seat availability example generation"""
        builder = TrainingDatasetBuilder()
        examples = builder.generate_seat_availability_examples(10)
        
        assert len(examples) == 10
        for example in examples:
            assert example.intent == RequestType.SEAT_AVAILABILITY
            assert len(example.text) > 0
    
    def test_cancellation_policy_examples(self):
        """Test cancellation policy example generation"""
        builder = TrainingDatasetBuilder()
        examples = builder.generate_cancellation_policy_examples(10)
        
        assert len(examples) == 10
        for example in examples:
            assert example.intent == RequestType.CANCELLATION_POLICY
            assert len(example.text) > 0
    
    def test_pet_travel_examples(self):
        """Test pet travel example generation"""
        builder = TrainingDatasetBuilder()
        examples = builder.generate_pet_travel_examples(10)
        
        assert len(examples) == 10
        for example in examples:
            assert example.intent == RequestType.PET_TRAVEL
            assert len(example.text) > 0
    
    def test_complete_dataset_generation(self):
        """Test complete dataset generation"""
        builder = TrainingDatasetBuilder()
        examples = builder.generate_complete_dataset()
        
        # Should have 10,000 examples (2,000 per intent)
        assert len(examples) == 10000
        
        # Check distribution
        intent_counts = {}
        for example in examples:
            intent = example.intent
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Each intent should have 2,000 examples
        for intent in RequestType:
            if intent != RequestType.UNKNOWN:
                assert intent_counts.get(intent, 0) == 2000


class TestDataAugmentation:
    """Test data augmentation functionality"""
    
    def test_augmentation_initialization(self):
        """Test augmentation initializes correctly"""
        augmenter = DataAugmentation()
        
        assert len(augmenter.synonym_dict) > 0
        assert len(augmenter.typo_patterns) > 0
        assert len(augmenter.paraphrase_patterns) > 0
    
    def test_synonym_replacement(self):
        """Test synonym replacement"""
        augmenter = DataAugmentation()
        
        text = "I want to cancel my flight"
        result = augmenter._replace_with_synonyms(text)
        
        # Result should be different (with some probability)
        # or same if no synonyms were applied
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_typo_addition(self):
        """Test typo addition"""
        augmenter = DataAugmentation()
        
        text = "cancel my flight booking"
        result = augmenter._add_typos_to_text(text)
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestEntityVariation:
    """Test entity variation generation"""
    
    def test_entity_variation_initialization(self):
        """Test entity variation initializes correctly"""
        variation = EntityVariation()
        
        assert len(variation.pnr_formats) > 0
        assert len(variation.flight_formats) > 0
        assert len(variation.date_formats) > 0
    
    def test_pnr_variations(self):
        """Test PNR variation generation"""
        variation = EntityVariation()
        pnrs = variation.generate_entity_variations(EntityType.PNR, 10)
        
        assert len(pnrs) == 10
        for pnr in pnrs:
            assert len(pnr) == 6
            assert pnr.isalnum()
    
    def test_flight_variations(self):
        """Test flight number variation generation"""
        variation = EntityVariation()
        flights = variation.generate_entity_variations(EntityType.FLIGHT_NUMBER, 10)
        
        assert len(flights) == 10
        for flight in flights:
            assert len(flight) >= 3


class TestEntityExtractor:
    """Test entity extraction functionality"""
    
    def test_extractor_initialization(self):
        """Test extractor initializes correctly"""
        extractor = EntityExtractor()
        
        assert len(extractor.pnr_patterns) > 0
        assert len(extractor.flight_patterns) > 0
        assert len(extractor.date_patterns) > 0
    
    def test_pnr_extraction(self):
        """Test PNR extraction"""
        extractor = EntityExtractor()
        
        text = "My PNR is ABC123"
        entities = extractor.extract_entities(text)
        
        pnr_entities = [e for e in entities if e.type == EntityType.PNR]
        assert len(pnr_entities) > 0
        assert pnr_entities[0].value == "ABC123"
    
    def test_flight_number_extraction(self):
        """Test flight number extraction"""
        extractor = EntityExtractor()
        
        text = "Flight AA100 is delayed"
        entities = extractor.extract_entities(text)
        
        flight_entities = [e for e in entities if e.type == EntityType.FLIGHT_NUMBER]
        assert len(flight_entities) > 0
        assert flight_entities[0].value == "AA100"
    
    def test_email_extraction(self):
        """Test email extraction"""
        extractor = EntityExtractor()
        
        text = "My email is test@example.com"
        entities = extractor.extract_entities(text)
        
        email_entities = [e for e in entities if e.type == EntityType.EMAIL]
        assert len(email_entities) > 0
        assert email_entities[0].value == "test@example.com"
    
    def test_multiple_entity_extraction(self):
        """Test extraction of multiple entities"""
        extractor = EntityExtractor()
        
        text = "Cancel flight AA100 for PNR ABC123 on tomorrow"
        entities = extractor.extract_entities(text)
        
        # Should extract flight number, PNR, and date
        entity_types = {e.type for e in entities}
        assert EntityType.FLIGHT_NUMBER in entity_types
        assert EntityType.PNR in entity_types
        assert EntityType.DATE in entity_types