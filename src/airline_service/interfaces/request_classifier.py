"""
Request classifier interface definitions
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..types import (
    RequestType, 
    ClassificationResult, 
    TrainingExample, 
    MLClassifierConfig, 
    ModelMetrics
)


class RequestClassifierInterface(ABC):
    """Interface for request classification"""
    
    @abstractmethod
    async def classify_request(self, utterance: str) -> ClassificationResult:
        """Classify customer utterance into request type"""
        pass
    
    @abstractmethod
    def get_confidence_score(self, utterance: str, request_type: RequestType) -> float:
        """Get confidence score for specific request type"""
        pass
    
    @abstractmethod
    async def train_model(self, training_data: List[TrainingExample]) -> None:
        """Train the classification model"""
        pass
    
    @abstractmethod
    async def load_model(self, model_path: str) -> None:
        """Load pre-trained model"""
        pass


class ModelTrainingPipelineInterface(ABC):
    """Interface for model training pipeline"""
    
    @abstractmethod
    def preprocess_text(self, utterance: str) -> str:
        """Preprocess text for training"""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text for model input"""
        pass
    
    @abstractmethod
    def augment_data(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Augment training data"""
        pass
    
    @abstractmethod
    async def train_model(self, config: MLClassifierConfig, data: List[TrainingExample]) -> Any:
        """Train the model with given configuration"""
        pass
    
    @abstractmethod
    async def evaluate_model(self, model: Any, test_data: List[TrainingExample]) -> ModelMetrics:
        """Evaluate model performance"""
        pass


class DataPreprocessorInterface(ABC):
    """Interface for data preprocessing"""
    
    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        pass
    
    @abstractmethod
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        pass
    
    @abstractmethod
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove stop words"""
        pass
    
    @abstractmethod
    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """Add special tokens for model"""
        pass


class DataAugmentationInterface(ABC):
    """Interface for data augmentation"""
    
    @abstractmethod
    def paraphrase_examples(self, examples: List[str]) -> List[str]:
        """Generate paraphrases of examples"""
        pass
    
    @abstractmethod
    def add_synonyms(self, text: str, synonym_dict: Dict[str, List[str]]) -> List[str]:
        """Add synonym variations"""
        pass
    
    @abstractmethod
    def add_typos(self, text: str, typo_rate: float) -> List[str]:
        """Add typos to text"""
        pass
    
    @abstractmethod
    def add_noise(self, text: str, noise_level: float) -> List[str]:
        """Add noise to text"""
        pass


class EntityVariationInterface(ABC):
    """Interface for entity variation generation"""
    
    @abstractmethod
    def generate_pnr_variations(self) -> List[str]:
        """Generate PNR variations"""
        pass
    
    @abstractmethod
    def generate_flight_numbers(self) -> List[str]:
        """Generate flight number variations"""
        pass
    
    @abstractmethod
    def generate_dates(self) -> List[str]:
        """Generate date variations"""
        pass
    
    @abstractmethod
    def generate_airport_codes(self) -> List[str]:
        """Generate airport code variations"""
        pass