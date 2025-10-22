"""
Dependency injection container for airline service components
"""

import asyncio
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from contextlib import asynccontextmanager
import structlog

from .config import config
from .types import RequestType, TaskType

# Import all services and clients
from .services import (
    TaskEngine, WorkflowOrchestrator, workflow_orchestrator,
    PolicyService, policy_service, RequestClassifierService,
    ContextBuilder, BookingSelector, ResponseFormatter,
    audit_logger, performance_monitor, metrics_collector,
    health_monitor, cache_service, connection_pool_manager
)
from .clients.airline_api_client import MockAirlineAPIClient
# ML classifiers will be imported as needed
from .ml.distilbert_classifier import DistilBERTClassifier

logger = structlog.get_logger()

T = TypeVar('T')


class ServiceContainer:
    """
    Dependency injection container for managing service instances and their dependencies
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize all services and their dependencies"""
        if self._initialized:
            return
            
        logger.info("Initializing service container")
        
        try:
            # 1. Initialize core infrastructure services first
            await self._initialize_infrastructure()
            
            # 2. Initialize ML services
            await self._initialize_ml_services()
            
            # 3. Initialize API clients
            await self._initialize_api_clients()
            
            # 4. Initialize business services
            await self._initialize_business_services()
            
            # 5. Initialize workflow orchestrator with all dependencies
            await self._initialize_workflow_orchestrator()
            
            # 6. Register all services in container
            self._register_services()
            
            self._initialized = True
            logger.info("Service container initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize service container", error=str(e))
            raise
    
    async def _initialize_infrastructure(self):
        """Initialize infrastructure services (cache, monitoring, etc.)"""
        logger.info("Initializing infrastructure services")
        
        # Initialize cache service
        try:
            await cache_service.initialize()
            logger.info("Cache service initialized")
        except Exception as e:
            logger.warning("Cache service initialization failed", error=str(e))
        
        # Connection pool manager is ready (no initialization needed)
        logger.info("Connection pool manager ready")
        
        # Initialize monitoring services (these are already singletons)
        logger.info("Monitoring services ready")
    
    async def _initialize_ml_services(self):
        """Initialize ML services (classifier, entity extractor)"""
        logger.info("Initializing ML services")
        
        # Initialize request classifier
        try:
            # First try to load the trained simple classifier
            from .services.request_classifier_service import ClassifierFactory
            
            # Check if simple classifier model exists
            simple_model_path = "./models/simple-classifier/classifier.pkl"
            import os
            
            if os.path.exists(simple_model_path):
                logger.info("Found trained simple classifier, loading...")
                classifier = ClassifierFactory.create_classifier()
                if classifier.is_loaded():
                    self._singletons['classifier'] = classifier
                    logger.info("Simple classifier loaded successfully")
                    return
            
            # Try to load DistilBERT classifier if simple classifier not available
            logger.info("Trying DistilBERT classifier...")
            classifier = DistilBERTClassifier(
                model_name=config.ml.model_path,
                max_length=config.ml.max_sequence_length
            )
            
            if await classifier.load_model(config.ml.model_path):
                self._singletons['classifier'] = classifier
                logger.info("DistilBERT classifier loaded successfully")
            else:
                # Fallback to mock classifier
                from .services.request_classifier_service import MockClassifier
                mock_classifier = MockClassifier()
                self._singletons['classifier'] = mock_classifier
                logger.info("Using mock classifier as fallback")
                
        except Exception as e:
            logger.error("Failed to initialize classifier", error=str(e))
            # Use mock classifier as last resort
            from .services.request_classifier_service import MockClassifier
            self._singletons['classifier'] = MockClassifier()
            logger.warning("Using mock classifier")
    
    async def _initialize_api_clients(self):
        """Initialize API clients"""
        logger.info("Initializing API clients")
        
        # Initialize airline API client (using mock for now)
        airline_client = MockAirlineAPIClient()
        self._singletons['airline_client'] = airline_client
        logger.info("Airline API client initialized (mock)")
    
    async def _initialize_business_services(self):
        """Initialize business logic services"""
        logger.info("Initializing business services")
        
        # Get dependencies
        airline_client = self._singletons['airline_client']
        
        # Initialize context builder
        context_builder = ContextBuilder(airline_client)
        self._singletons['context_builder'] = context_builder
        
        # Initialize booking selector
        booking_selector = BookingSelector()
        self._singletons['booking_selector'] = booking_selector
        
        # Initialize response formatter
        response_formatter = ResponseFormatter()
        self._singletons['response_formatter'] = response_formatter
        
        # Initialize task engine with all dependencies
        task_engine = TaskEngine(airline_client, policy_service)
        self._singletons['task_engine'] = task_engine
        
        logger.info("Business services initialized")
    
    async def _initialize_workflow_orchestrator(self):
        """Initialize workflow orchestrator with all task handlers"""
        logger.info("Initializing workflow orchestrator")
        
        # Get task engine
        task_engine = self._singletons['task_engine']
        
        # Register task handlers with the global workflow orchestrator
        workflow_orchestrator.task_handlers = {
            TaskType.GET_CUSTOMER_INFO: task_engine.task_handlers[TaskType.GET_CUSTOMER_INFO],
            TaskType.API_CALL: task_engine.task_handlers[TaskType.API_CALL],
            TaskType.POLICY_LOOKUP: task_engine.task_handlers[TaskType.POLICY_LOOKUP],
            TaskType.INFORM_CUSTOMER: task_engine.task_handlers[TaskType.INFORM_CUSTOMER]
        }
        
        self._singletons['workflow_orchestrator'] = workflow_orchestrator
        logger.info("Workflow orchestrator initialized with task handlers")
    
    def _register_services(self):
        """Register all services in the container"""
        # Register singletons
        for name, service in self._singletons.items():
            self._services[name] = service
        
        # Register global services
        self._services['audit_logger'] = audit_logger
        self._services['performance_monitor'] = performance_monitor
        self._services['metrics_collector'] = metrics_collector
        self._services['health_monitor'] = health_monitor
        self._services['cache_service'] = cache_service
        self._services['connection_pool_manager'] = connection_pool_manager
        self._services['policy_service'] = policy_service
        
        logger.info("All services registered in container", service_count=len(self._services))
    
    def get_service(self, service_name: str) -> Any:
        """Get a service by name"""
        if not self._initialized:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        
        if service_name not in self._services:
            raise KeyError(f"Service '{service_name}' not found in container")
        
        return self._services[service_name]
    
    def get_classifier(self) -> RequestClassifierService:
        """Get the request classifier service"""
        return self.get_service('classifier')
    
    def get_workflow_orchestrator(self) -> WorkflowOrchestrator:
        """Get the workflow orchestrator"""
        return self.get_service('workflow_orchestrator')
    
    def get_task_engine(self) -> TaskEngine:
        """Get the task engine"""
        return self.get_service('task_engine')
    
    def get_airline_client(self) -> MockAirlineAPIClient:
        """Get the airline API client"""
        return self.get_service('airline_client')
    
    def get_policy_service(self) -> PolicyService:
        """Get the policy service"""
        return self.get_service('policy_service')
    
    def get_context_builder(self) -> ContextBuilder:
        """Get the context builder"""
        return self.get_service('context_builder')
    
    def get_booking_selector(self) -> BookingSelector:
        """Get the booking selector"""
        return self.get_service('booking_selector')
    
    def get_response_formatter(self) -> ResponseFormatter:
        """Get the response formatter"""
        return self.get_service('response_formatter')
    
    async def cleanup(self):
        """Cleanup all services"""
        if not self._initialized:
            return
            
        logger.info("Cleaning up service container")
        
        try:
            # Cleanup services in reverse order
            await cache_service.close()
            await connection_pool_manager.close_all_sessions()
            
            # Clear all services
            self._services.clear()
            self._singletons.clear()
            self._initialized = False
            
            logger.info("Service container cleanup completed")
            
        except Exception as e:
            logger.error("Error during service container cleanup", error=str(e))
    
    def is_initialized(self) -> bool:
        """Check if container is initialized"""
        return self._initialized
    
    def list_services(self) -> Dict[str, str]:
        """List all registered services"""
        return {name: type(service).__name__ for name, service in self._services.items()}


# Global container instance
container = ServiceContainer()


@asynccontextmanager
async def get_container():
    """Context manager for getting initialized container"""
    if not container.is_initialized():
        await container.initialize()
    
    try:
        yield container
    finally:
        # Don't cleanup here - let the application lifespan handle it
        pass


class ServiceFactory:
    """Factory for creating service instances with proper dependencies"""
    
    @staticmethod
    async def create_classifier() -> RequestClassifierService:
        """Create and initialize a classifier instance"""
        async with get_container() as cont:
            return cont.get_classifier()
    
    @staticmethod
    async def create_workflow_orchestrator() -> WorkflowOrchestrator:
        """Create and initialize a workflow orchestrator instance"""
        async with get_container() as cont:
            return cont.get_workflow_orchestrator()
    
    @staticmethod
    async def create_task_engine() -> TaskEngine:
        """Create and initialize a task engine instance"""
        async with get_container() as cont:
            return cont.get_task_engine()


# Environment-specific configuration
class EnvironmentConfig:
    """Environment-specific service configuration"""
    
    @staticmethod
    def configure_for_development():
        """Configure services for development environment"""
        logger.info("Configuring services for development environment")
        # Development-specific configurations
        
    @staticmethod
    def configure_for_production():
        """Configure services for production environment"""
        logger.info("Configuring services for production environment")
        # Production-specific configurations
        
    @staticmethod
    def configure_for_testing():
        """Configure services for testing environment"""
        logger.info("Configuring services for testing environment")
        # Testing-specific configurations


def configure_environment():
    """Configure services based on current environment"""
    env = config.server.environment.lower()
    
    if env == "development":
        EnvironmentConfig.configure_for_development()
    elif env == "production":
        EnvironmentConfig.configure_for_production()
    elif env == "testing":
        EnvironmentConfig.configure_for_testing()
    else:
        logger.warning("Unknown environment, using development configuration", environment=env)
        EnvironmentConfig.configure_for_development()