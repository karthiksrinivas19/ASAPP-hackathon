"""
Workflow orchestrator interface definitions
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..types import (
    RequestType,
    RequestContext,
    WorkflowResult,
    TaskDefinition,
    TaskContext,
    TaskResult,
    TaskType,
    FlightIdentifiers,
    BookingContext,
    CancellationResult
)


class WorkflowOrchestratorInterface(ABC):
    """Interface for workflow orchestration"""
    
    @abstractmethod
    async def execute_workflow(self, request_type: RequestType, context: RequestContext) -> WorkflowResult:
        """Execute workflow for given request type"""
        pass
    
    @abstractmethod
    def get_workflow_definition(self, request_type: RequestType) -> List[TaskDefinition]:
        """Get workflow definition for request type"""
        pass


class TaskEngineInterface(ABC):
    """Interface for task execution"""
    
    @abstractmethod
    async def execute_task(self, task: TaskDefinition, context: TaskContext) -> TaskResult:
        """Execute individual task"""
        pass
    
    @abstractmethod
    def register_task_handler(self, task_type: TaskType, handler: 'TaskHandlerInterface') -> None:
        """Register task handler for task type"""
        pass
    
    @abstractmethod
    async def auto_retrieve_data(self, required_data: List[str], context: TaskContext) -> Dict[str, Any]:
        """Automatically retrieve required data"""
        pass


class TaskHandlerInterface(ABC):
    """Interface for task handlers"""
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: TaskContext) -> Any:
        """Execute task with parameters"""
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Get list of required data fields"""
        pass
    
    @abstractmethod
    def can_auto_retrieve(self, data_field: str) -> bool:
        """Check if data field can be auto-retrieved"""
        pass


class AutoDataRetrieverInterface(ABC):
    """Interface for automatic data retrieval"""
    
    @abstractmethod
    async def retrieve_booking_details(self, pnr: str) -> Any:
        """Retrieve booking details by PNR"""
        pass
    
    @abstractmethod
    def extract_pnr_from_query(self, utterance: str) -> Optional[str]:
        """Extract PNR from customer query"""
        pass
    
    @abstractmethod
    def extract_flight_details_from_query(self, utterance: str) -> Dict[str, Any]:
        """Extract flight details from query"""
        pass


class FullyAutomatedEngineInterface(ABC):
    """Interface for fully automated processing"""
    
    @abstractmethod
    async def process_request(self, utterance: str, request_type: RequestType) -> Any:
        """Process request with full automation"""
        pass
    
    @abstractmethod
    def extract_all_possible_identifiers(self, utterance: str) -> FlightIdentifiers:
        """Extract all possible identifiers from utterance"""
        pass
    
    @abstractmethod
    async def resolve_flight_data(self, identifiers: FlightIdentifiers) -> Any:
        """Resolve flight data from identifiers"""
        pass


class ContextBuilderInterface(ABC):
    """Interface for context building"""
    
    @abstractmethod
    async def build_context(self, utterance: str, request_type: RequestType) -> TaskContext:
        """Build task context from utterance"""
        pass
    
    @abstractmethod
    async def resolve_data_dependencies(self, required_data: List[str], available_data: Any) -> Any:
        """Resolve data dependencies"""
        pass


class AutomatedBookingSelectorInterface(ABC):
    """Interface for automated booking selection"""
    
    @abstractmethod
    def select_booking_to_cancel(self, bookings: List[Any], context: Any) -> Optional[Any]:
        """Select booking to cancel from multiple options"""
        pass


class EnhancedIdentifierExtractorInterface(ABC):
    """Interface for enhanced identifier extraction"""
    
    @abstractmethod
    def extract_pnr(self, text: str) -> Optional[str]:
        """Extract PNR from text"""
        pass
    
    @abstractmethod
    def extract_flight_number(self, text: str) -> Optional[str]:
        """Extract flight number from text"""
        pass
    
    @abstractmethod
    def extract_route(self, text: str) -> Optional[Dict[str, str]]:
        """Extract route information from text"""
        pass
    
    @abstractmethod
    def extract_date(self, text: str) -> Optional[str]:
        """Extract date from text"""
        pass
    
    @abstractmethod
    def extract_passenger_name(self, text: str) -> Optional[str]:
        """Extract passenger name from text"""
        pass
    
    @abstractmethod
    def extract_booking_context(self, text: str) -> BookingContext:
        """Extract booking context from text"""
        pass
    
    @abstractmethod
    def extract_customer_info(self, utterance: str) -> Dict[str, Any]:
        """Extract customer information from utterance"""
        pass
    
    @abstractmethod
    def extract_contextual_clues(self, utterance: str) -> Dict[str, Any]:
        """Extract contextual clues from utterance"""
        pass
    
    @abstractmethod
    def infer_booking_intent(self, utterance: str) -> Dict[str, Any]:
        """Infer booking intent from utterance"""
        pass