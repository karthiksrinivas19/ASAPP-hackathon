"""
Core data types for the airline customer service system
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field


class RequestType(str, Enum):
    """Types of customer requests that can be processed"""
    CANCEL_TRIP = "cancel_trip"
    CANCELLATION_POLICY = "cancellation_policy"
    FLIGHT_STATUS = "flight_status"
    SEAT_AVAILABILITY = "seat_availability"
    PET_TRAVEL = "pet_travel"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """Types of entities that can be extracted from customer queries"""
    PNR = "pnr"
    FLIGHT_NUMBER = "flight_number"
    DATE = "date"
    AIRPORT_CODE = "airport_code"
    PASSENGER_NAME = "passenger_name"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    DESTINATION = "destination"
    CLASS = "class"
    SEAT_TYPE = "seat_type"
    PET_TYPE = "pet_type"


class TaskType(str, Enum):
    """Types of tasks in workflow execution"""
    GET_CUSTOMER_INFO = "get_customer_info"
    API_CALL = "api_call"
    POLICY_LOOKUP = "policy_lookup"
    INFORM_CUSTOMER = "inform_customer"


class ResponseFormat(str, Enum):
    """Response formatting types"""
    SIMPLE_MESSAGE = "simple_message"
    STRUCTURED_DATA = "structured_data"
    FLIGHT_STATUS = "flight_status"
    CANCELLATION_RESULT = "cancellation_result"
    SEAT_AVAILABILITY = "seat_availability"
    POLICY_INFO = "policy_info"


# Request and Response Models
class CustomerRequest(BaseModel):
    """Customer request model"""
    utterance: str = Field(..., description="Customer's query text")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    customer_id: Optional[str] = Field(None, description="Optional customer identifier")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class APIResponse(BaseModel):
    """Standard API response model"""
    status: str = Field(..., description="Response status: completed or error")
    message: str = Field(..., description="Human-readable response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error_code: Optional[str] = Field(None, description="Error code if applicable")
    timestamp: datetime = Field(default_factory=datetime.now)


# Classification Models
class ExtractedEntity(BaseModel):
    """Extracted entity from customer query"""
    type: EntityType = Field(..., description="Type of extracted entity")
    value: str = Field(..., description="Extracted value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    start_index: int = Field(..., description="Start position in text")
    end_index: int = Field(..., description="End position in text")


class ClassificationResult(BaseModel):
    """Result of request classification"""
    request_type: RequestType = Field(..., description="Classified request type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    alternative_intents: List[Dict[str, Union[RequestType, float]]] = Field(
        default_factory=list, description="Alternative intent suggestions"
    )
    extracted_entities: List[ExtractedEntity] = Field(
        default_factory=list, description="Entities extracted from query"
    )


# Flight and Booking Models
class BookingDetails(BaseModel):
    """Flight booking details from airline API"""
    pnr: str = Field(..., description="Passenger Name Record")
    flight_id: int = Field(..., description="Flight identifier")
    source_airport_code: str = Field(..., description="Departure airport code")
    destination_airport_code: str = Field(..., description="Arrival airport code")
    scheduled_departure: datetime = Field(..., description="Scheduled departure time")
    scheduled_arrival: datetime = Field(..., description="Scheduled arrival time")
    assigned_seat: str = Field(..., description="Assigned seat number")
    current_departure: datetime = Field(..., description="Current departure time")
    current_arrival: datetime = Field(..., description="Current arrival time")
    current_status: str = Field(..., description="Current flight status")


class FlightDetails(BaseModel):
    """Flight details for queries"""
    pnr: Optional[str] = None
    flight_number: Optional[str] = None
    departure_date: Optional[datetime] = None
    source_airport: Optional[str] = None
    destination_airport: Optional[str] = None


class FlightInfo(BaseModel):
    """Flight information"""
    flight_id: int
    flight_number: str
    source_airport_code: str
    destination_airport_code: str
    scheduled_departure: datetime
    scheduled_arrival: datetime
    current_status: str


class CancellationResult(BaseModel):
    """Flight cancellation result"""
    message: str
    cancellation_charges: float
    refund_amount: float
    refund_date: datetime


class SeatInfo(BaseModel):
    """Individual seat information"""
    row_number: int
    column_letter: str
    price: float
    seat_class: str = Field(..., alias="class")


class SeatAvailability(BaseModel):
    """Seat availability response"""
    flight_id: int
    pnr: str
    available_seats: List[SeatInfo]


# Customer and Search Models
class CustomerSearchInfo(BaseModel):
    """Customer search criteria"""
    phone: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    date_range: Optional[Dict[str, datetime]] = None


class CustomerIdentifier(BaseModel):
    """Customer identification options"""
    phone: Optional[str] = None
    email: Optional[str] = None
    customer_id: Optional[str] = None
    loyalty_number: Optional[str] = None


class CustomerProfile(BaseModel):
    """Customer profile information"""
    customer_id: str
    recent_bookings: List[BookingDetails]
    upcoming_flights: List[BookingDetails]
    preferences: Dict[str, Any]


class RouteInfo(BaseModel):
    """Route information"""
    source: str = Field(..., alias="from")
    destination: str = Field(..., alias="to")


# Workflow and Task Models
class RequestContext(BaseModel):
    """Context for request processing"""
    session_id: str
    customer_id: Optional[str] = None
    request_type: RequestType
    utterance: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskContext(BaseModel):
    """Context for task execution"""
    session_id: str
    request_type: RequestType
    extracted_entities: List[ExtractedEntity]
    flight_details: Optional[FlightDetails] = None
    customer_info: Optional[CustomerSearchInfo] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskDefinition(BaseModel):
    """Task definition for workflow"""
    task_id: str
    task_type: TaskType
    parameters: Dict[str, Any]
    dependencies: List[str]


class TaskResult(BaseModel):
    """Result of task execution"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    duration: float


class WorkflowResult(BaseModel):
    """Result of workflow execution"""
    success: bool
    message: str
    data: Optional[Any] = None
    executed_tasks: List[str]
    duration: float


# Policy Models
class PolicyInfo(BaseModel):
    """Policy information"""
    policy_type: str
    content: str
    last_updated: datetime
    applicable_conditions: List[str]


# ML Model Models
class TrainingExample(BaseModel):
    """Training example for ML model"""
    text: str
    intent: RequestType
    entities: List[ExtractedEntity]


class MLClassifierConfig(BaseModel):
    """ML classifier configuration"""
    model_type: str = Field(default="distilbert", description="Model type to use")
    max_sequence_length: int = Field(default=128, description="Maximum sequence length")
    num_labels: int = Field(default=6, description="Number of classification labels")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    batch_size: int = Field(default=16, description="Training batch size")
    epochs: int = Field(default=3, description="Number of training epochs")


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: float
    precision: Dict[RequestType, float]
    recall: Dict[RequestType, float]
    f1_score: Dict[RequestType, float]
    confusion_matrix: List[List[int]]


# Flight Identifiers
class FlightIdentifiers(BaseModel):
    """Identifiers extracted from customer query"""
    pnr: Optional[str] = None
    flight_number: Optional[str] = None
    route: Optional[RouteInfo] = None
    date: Optional[datetime] = None
    passenger_name: Optional[str] = None


class BookingContext(BaseModel):
    """Context about booking intent"""
    has_booking_intent: bool
    has_urgency: bool
    has_partial_info: bool
    suggested_actions: List[str]


# Custom Exceptions
class AirlineServiceError(Exception):
    """Base exception for airline service"""
    pass


class MissingDataError(AirlineServiceError):
    """Exception for missing required data"""
    def __init__(self, data_type: str):
        self.data_type = data_type
        super().__init__(f"Missing required data: {data_type}")


class APIError(AirlineServiceError):
    """Exception for API errors"""
    def __init__(self, message: str, status_code: int, error_code: Optional[str] = None):
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)


class ClassificationError(AirlineServiceError):
    """Exception for classification errors"""
    pass