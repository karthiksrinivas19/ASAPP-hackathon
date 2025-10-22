"""
Tests for request classifier
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock logger to avoid dependency issues
class MockLogger:
    def info(self, msg, **kwargs): pass
    def debug(self, msg, **kwargs): pass
    def error(self, msg, **kwargs): pass
    def warning(self, msg, **kwargs): pass

# Mock the logger import
import airline_service.utils.logger
airline_service.utils.logger.get_logger = lambda name: MockLogger()

from airline_service.services.request_classifier_service import ClassifierFactory, RequestClassifierService
from airline_service.types import RequestType


class TestRequestClassifier:
    """Test request classifier functionality"""
    
    def setup_method(self):
        """Setup test method"""
        self.classifier = ClassifierFactory.create_mock_classifier()
    
    @pytest.mark.asyncio
    async def test_classifier_initialization(self):
        """Test classifier initialization"""
        assert self.classifier.is_loaded()
        
        model_info = self.classifier.get_model_info()
        assert model_info["status"] == "loaded"
        assert model_info["num_classes"] == 5
    
    @pytest.mark.asyncio
    async def test_cancel_trip_classification(self):
        """Test cancel trip intent classification"""
        test_cases = [
            "I want to cancel my flight",
            "Cancel my booking",
            "Please cancel my reservation",
            "I need to cancel my trip"
        ]
        
        for utterance in test_cases:
            result = await self.classifier.classify_request(utterance)
            assert result.request_type == RequestType.CANCEL_TRIP
            assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_flight_status_classification(self):
        """Test flight status intent classification"""
        test_cases = [
            "What's my flight status",
            "Is my flight on time",
            "Check flight status",
            "When does my flight depart"
        ]
        
        for utterance in test_cases:
            result = await self.classifier.classify_request(utterance)
            assert result.request_type == RequestType.FLIGHT_STATUS
            assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_seat_availability_classification(self):
        """Test seat availability intent classification"""
        test_cases = [
            "Show available seats",
            "What seats are free",
            "I want to change my seat",
            "Seat availability"
        ]
        
        for utterance in test_cases:
            result = await self.classifier.classify_request(utterance)
            assert result.request_type == RequestType.SEAT_AVAILABILITY
            assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_cancellation_policy_classification(self):
        """Test cancellation policy intent classification"""
        test_cases = [
            "What's your cancellation policy",
            "Cancellation fees",
            "Can I get a refund",
            "Refund policy"
        ]
        
        for utterance in test_cases:
            result = await self.classifier.classify_request(utterance)
            assert result.request_type == RequestType.CANCELLATION_POLICY
            assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_pet_travel_classification(self):
        """Test pet travel intent classification"""
        test_cases = [
            "Can I bring my pet",
            "Pet travel policy",
            "Flying with my dog",
            "Pet requirements"
        ]
        
        for utterance in test_cases:
            result = await self.classifier.classify_request(utterance)
            assert result.request_type == RequestType.PET_TRAVEL
            assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_confidence_scores(self):
        """Test confidence scoring"""
        utterance = "I want to cancel my flight"
        
        # Test specific confidence score
        confidence = self.classifier.get_confidence_score(utterance, RequestType.CANCEL_TRIP)
        assert 0.0 <= confidence <= 1.0
        
        # Cancel trip should have higher confidence than other intents
        cancel_confidence = self.classifier.get_confidence_score(utterance, RequestType.CANCEL_TRIP)
        status_confidence = self.classifier.get_confidence_score(utterance, RequestType.FLIGHT_STATUS)
        
        assert cancel_confidence > status_confidence
    
    @pytest.mark.asyncio
    async def test_batch_prediction(self):
        """Test batch prediction"""
        utterances = [
            "I want to cancel my flight",
            "What's my flight status",
            "Show available seats"
        ]
        
        results = await self.classifier.predict_batch(utterances)
        
        assert len(results) == len(utterances)
        assert results[0].request_type == RequestType.CANCEL_TRIP
        assert results[1].request_type == RequestType.FLIGHT_STATUS
        assert results[2].request_type == RequestType.SEAT_AVAILABILITY
    
    @pytest.mark.asyncio
    async def test_alternative_intents(self):
        """Test alternative intent suggestions"""
        result = await self.classifier.classify_request("I want to cancel my flight")
        
        assert len(result.alternative_intents) > 0
        
        # Check that alternatives have lower confidence than primary
        primary_confidence = result.confidence
        for alt in result.alternative_intents:
            assert alt["confidence"] < primary_confidence
    
    def test_supported_intents(self):
        """Test getting supported intents"""
        intents = self.classifier.get_supported_intents()
        
        assert len(intents) == 5
        assert RequestType.CANCEL_TRIP in intents
        assert RequestType.FLIGHT_STATUS in intents
        assert RequestType.SEAT_AVAILABILITY in intents
        assert RequestType.CANCELLATION_POLICY in intents
        assert RequestType.PET_TRAVEL in intents
    
    @pytest.mark.asyncio
    async def test_empty_utterance(self):
        """Test handling of empty utterance"""
        result = await self.classifier.classify_request("")
        
        # Should still return a result
        assert result.request_type in [e.value for e in RequestType]
        assert 0.0 <= result.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_long_utterance(self):
        """Test handling of long utterance"""
        long_utterance = "I want to cancel my flight " * 20
        
        result = await self.classifier.classify_request(long_utterance)
        
        # Should still classify correctly
        assert result.request_type == RequestType.CANCEL_TRIP
        assert result.confidence > 0.5


class TestClassifierFactory:
    """Test classifier factory"""
    
    def test_create_mock_classifier(self):
        """Test creating mock classifier"""
        classifier = ClassifierFactory.create_mock_classifier()
        
        assert classifier.is_loaded()
        assert len(classifier.get_supported_intents()) == 5
    
    def test_create_classifier_with_invalid_path(self):
        """Test creating classifier with invalid model path"""
        classifier = ClassifierFactory.create_classifier("invalid/path")
        
        # Should create classifier but not load model
        assert not classifier.is_loaded()


class TestPerformance:
    """Test classifier performance"""
    
    def setup_method(self):
        """Setup test method"""
        self.classifier = ClassifierFactory.create_mock_classifier()
    
    @pytest.mark.asyncio
    async def test_latency_requirement(self):
        """Test that latency meets requirement (<100ms)"""
        import time
        
        utterance = "I want to cancel my flight"
        iterations = 10
        
        start_time = time.time()
        for _ in range(iterations):
            await self.classifier.classify_request(utterance)
        
        total_time = time.time() - start_time
        avg_latency = (total_time / iterations) * 1000  # ms
        
        # Should be well under 100ms
        assert avg_latency < 100
    
    @pytest.mark.asyncio
    async def test_throughput(self):
        """Test classifier throughput"""
        import time
        
        utterances = ["I want to cancel my flight"] * 50
        
        start_time = time.time()
        results = await self.classifier.predict_batch(utterances)
        total_time = time.time() - start_time
        
        throughput = len(utterances) / total_time
        
        # Should handle reasonable throughput
        assert throughput > 100  # requests per second
        assert len(results) == len(utterances)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])