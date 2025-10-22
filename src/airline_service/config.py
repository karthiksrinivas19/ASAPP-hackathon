"""
Configuration management for airline service
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field


class ServerConfig(BaseSettings):
    """Server configuration"""
    port: int = Field(default=8000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    class Config:
        env_prefix = "SERVER_"


class AirlineAPIConfig(BaseSettings):
    """Airline API configuration"""
    base_url: str = Field(default="https://api.airline.com", env="AIRLINE_API_BASE_URL")
    timeout: int = Field(default=5000, env="AIRLINE_API_TIMEOUT")
    retry_attempts: int = Field(default=3, env="AIRLINE_API_RETRY_ATTEMPTS")
    retry_delay: int = Field(default=1000, env="AIRLINE_API_RETRY_DELAY")
    api_key: Optional[str] = Field(default=None, env="AIRLINE_API_KEY")
    
    class Config:
        env_prefix = "AIRLINE_API_"


class PolicyConfig(BaseSettings):
    """Policy configuration"""
    cancellation_policy_url: str = Field(
        default="https://www.jetblue.com/flying-with-us/our-fares",
        env="CANCELLATION_POLICY_URL"
    )
    pet_travel_policy_url: str = Field(
        default="https://www.jetblue.com/traveling-together/traveling-with-pets",
        env="PET_TRAVEL_POLICY_URL"
    )
    cache_ttl: int = Field(default=3600, env="POLICY_CACHE_TTL")  # 1 hour
    
    class Config:
        env_prefix = "POLICY_"


class RedisConfig(BaseSettings):
    """Redis configuration"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    
    class Config:
        env_prefix = "REDIS_"


class MLConfig(BaseSettings):
    """ML model configuration"""
    model_path: str = Field(default="./models/distilbert-classifier", env="ML_MODEL_PATH")
    max_sequence_length: int = Field(default=128, env="ML_MAX_SEQUENCE_LENGTH")
    confidence_threshold: float = Field(default=0.8, env="ML_CONFIDENCE_THRESHOLD")
    batch_size: int = Field(default=16, env="ML_BATCH_SIZE")
    device: str = Field(default="cpu", env="ML_DEVICE")  # cpu or cuda
    
    class Config:
        env_prefix = "ML_"


class PerformanceConfig(BaseSettings):
    """Performance thresholds configuration"""
    max_request_latency: int = Field(default=2000, env="MAX_REQUEST_LATENCY")  # 2 seconds
    max_api_latency: int = Field(default=5000, env="MAX_API_LATENCY")  # 5 seconds
    max_policy_latency: int = Field(default=3000, env="MAX_POLICY_LATENCY")  # 3 seconds
    target_availability: float = Field(default=0.999, env="TARGET_AVAILABILITY")  # 99.9%
    
    class Config:
        env_prefix = "PERFORMANCE_"


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_audit: bool = Field(default=True, env="ENABLE_AUDIT_LOGGING")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    class Config:
        env_prefix = "LOGGING_"


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.server = ServerConfig()
        self.airline_api = AirlineAPIConfig()
        self.policies = PolicyConfig()
        self.redis = RedisConfig()
        self.ml = MLConfig()
        self.performance = PerformanceConfig()
        self.logging = LoggingConfig()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.server.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.server.environment.lower() == "production"


# Global configuration instance
config = Config()