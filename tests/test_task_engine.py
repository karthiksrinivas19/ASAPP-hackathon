"""
Tests for task engine and task handlers
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from src.airline_service.services.task_engine import (
    TaskEngine, GetCustomerInfoHandler, APICallHandler, 
    PolicyLookupHandler, InformCustomerHandler, AutoDataRetriever
)
from src.airline_service.types import (
    TaskDefinition, TaskContext, TaskType, RequestType,
    ExtractedEntity, EntityType, BookingDetails, PolicyInfo,
    CancellationResult, APIResponse, ResponseFormat
)
from src.airline_service.clients.airline_api_client import MockAirlineAPIClient
from src.airline_service.services.policy_service import PolicyService


class TestGetCustomerInfoHandler:
    """Test GET_CUSTOMER_INFO task handler"""
    
    @pytest.fixture
    def handler(self):
        mock_extractor = Mock()
        mock_retriever = Mock()
        return GetCustomerInfoHandler(mock_extractor, mock_retriever)
    
    @pytest.fixture
    def task_context(self):
        return TaskContext(
            session_id="test_session",
            request_type=RequestType.CANCEL_TRIP,
            extracted_entities=[],
            metadata={"utterance": "I want to cancel my flight ABC123"}
        )
    
    @pytest.mark.asyncio
    async def test_extract_pnr_from_utterance(self, handler, task_context):
        """Test extracting PNR from customer utterance"""
        # Mock entity extractor
        mock_entities = [
            ExtractedEntity(
                type=EntityType.PNR,
                value="ABC123",
                confidence=0.95,
                start_index=30,
                end_index=36
            )
        ]
        handler.entity_extractor.extract_entities.return_value = mock_entities
        
        # Execute task
        parameters = {"extract_types": ["pnr"]}
        result = await handler.execute(parameters, task_context)
        
        # Verify results
        assert result["success"] is True
        assert len(result["extracted_entities"]) == 1
        assert result["extracted_entities"][0].type == EntityType.PNR
        assert result["extracted_entities"][0].value == "ABC123"
        assert result["flight_identifiers"].pnr == "ABC123"
    
    @pytest.mark.asyncio
    async def test_extract_multiple_entities(self, handler, task_context):
        """Test extracting multiple entity types"""
        # Mock entity extractor
        mock_entities = [
            ExtractedEntity(
                type=EntityType.PNR,
                value="ABC123",
                confidence=0.95,
                start_index=30,
                end_index=36
            ),
            ExtractedEntity(
                type=EntityType.PASSENGER_NAME,
                value="John Smith",
                confidence=0.90,
                start_index=10,
                end_index=20
            )
        ]
        handler.entity_extractor.extract_entities.return_value = mock_entities
        
        # Execute task
        parameters = {"extract_types": ["pnr", "passenger_name"]}
        result = await handler.execute(parameters, task_context)
        
        # Verify results
        assert result["success"] is True
        assert len(result["extracted_entities"]) == 2
        assert result["flight_identifiers"].pnr == "ABC123"
        assert result["flight_identifiers"].passenger_name == "John Smith"
        assert result["customer_info"]["name"] == "John Smith"


class TestAPICallHandler:
    """Test API_CALL task handler"""
    
    @pytest.fixture
    def handler(self):
        mock_client = AsyncMock()
        mock_retriever = AsyncMock()
        return APICallHandler(mock_client, mock_retriever)
    
    @pytest.fixture
    def task_context(self):
        return TaskContext(
            session_id="test_session",
            request_type=RequestType.CANCEL_TRIP,
            extracted_entities=[
                ExtractedEntity(
                    type=EntityType.PNR,
                    value="ABC123",
                    confidence=0.95,
                    start_index=0,
                    end_index=6
                )
            ],
            metadata={"workflow_data": {}}
        )
    
    @pytest.mark.asyncio
    async def test_get_booking_details(self, handler, task_context):
        """Test getting booking details via API"""
        # Mock booking details
        mock_booking = BookingDetails(
            pnr="ABC123",
            flight_id=1001,
            source_airport_code="JFK",
            destination_airport_code="LAX",
            scheduled_departure=datetime(2024, 1, 15, 10, 30),
            scheduled_arrival=datetime(2024, 1, 15, 13, 45),
            assigned_seat="12A",
            current_departure=datetime(2024, 1, 15, 10, 30),
            current_arrival=datetime(2024, 1, 15, 13, 45),
            current_status="On Time"
        )
        
        # Mock auto retriever
        handler.auto_retriever.retrieve_data.return_value = {"booking_details": mock_booking}
        
        # Mock airline client
        handler.airline_client.get_booking_details.return_value = mock_booking
        
        # Execute task
        parameters = {"api_method": "get_booking_details", "required_data": ["pnr"]}
        result = await handler.execute(parameters, task_context)
        
        # Verify results
        assert result["success"] is True
        assert result["api_method"] == "get_booking_details"
        assert result["booking_details"].pnr == "ABC123"
    
    @pytest.mark.asyncio
    async def test_cancel_flight(self, handler, task_context):
        """Test cancelling flight via API"""
        # Mock booking details
        mock_booking = BookingDetails(
            pnr="ABC123",
            flight_id=1001,
            source_airport_code="JFK",
            destination_airport_code="LAX",
            scheduled_departure=datetime(2024, 1, 15, 10, 30),
            scheduled_arrival=datetime(2024, 1, 15, 13, 45),
            assigned_seat="12A",
            current_departure=datetime(2024, 1, 15, 10, 30),
            current_arrival=datetime(2024, 1, 15, 13, 45),
            current_status="On Time"
        )
        
        # Mock cancellation result
        mock_cancellation = CancellationResult(
            message="Flight cancelled successfully",
            cancellation_charges=50.0,
            refund_amount=150.0,
            refund_date=datetime(2024, 1, 20)
        )
        
        # Mock auto retriever
        handler.auto_retriever.retrieve_data.return_value = {"booking_details": mock_booking}
        
        # Mock airline client
        handler.airline_client.cancel_flight.return_value = mock_cancellation
        
        # Execute task
        parameters = {"api_method": "cancel_flight", "required_data": ["booking_details"]}
        result = await handler.execute(parameters, task_context)
        
        # Verify results
        assert result["success"] is True
        assert result["api_method"] == "cancel_flight"
        assert result["cancellation_result"].refund_amount == 150.0


class TestPolicyLookupHandler:
    """Test POLICY_LOOKUP task handler"""
    
    @pytest.fixture
    def handler(self):
        mock_policy_service = AsyncMock()
        return PolicyLookupHandler(mock_policy_service)
    
    @pytest.fixture
    def task_context(self):
        return TaskContext(
            session_id="test_session",
            request_type=RequestType.CANCELLATION_POLICY,
            extracted_entities=[],
            metadata={}
        )
    
    @pytest.mark.asyncio
    async def test_get_cancellation_policy(self, handler, task_context):
        """Test getting cancellation policy"""
        # Mock policy info
        mock_policy = PolicyInfo(
            policy_type="cancellation",
            content="Cancellation policy content here...",
            last_updated=datetime.now(),
            applicable_conditions=["fare_type", "booking_class"]
        )
        
        # Mock policy service
        handler.policy_service.get_general_cancellation_policy.return_value = mock_policy
        
        # Execute task
        parameters = {"policy_type": "cancellation", "context_aware": False}
        result = await handler.execute(parameters, task_context)
        
        # Verify results
        assert result["success"] is True
        assert result["policy_type"] == "cancellation"
        assert result["policy_info"].policy_type == "cancellation"
    
    @pytest.mark.asyncio
    async def test_get_pet_travel_policy(self, handler, task_context):
        """Test getting pet travel policy"""
        # Mock policy info
        mock_policy = PolicyInfo(
            policy_type="pet_travel",
            content="Pet travel policy content here...",
            last_updated=datetime.now(),
            applicable_conditions=["pet_type", "destination"]
        )
        
        # Mock policy service
        handler.policy_service.get_pet_travel_policy.return_value = mock_policy
        
        # Execute task
        parameters = {"policy_type": "pet_travel"}
        result = await handler.execute(parameters, task_context)
        
        # Verify results
        assert result["success"] is True
        assert result["policy_type"] == "pet_travel"
        assert result["policy_info"].policy_type == "pet_travel"


class TestInformCustomerHandler:
    """Test INFORM_CUSTOMER task handler"""
    
    @pytest.fixture
    def handler(self):
        return InformCustomerHandler()
    
    @pytest.fixture
    def task_context(self):
        return TaskContext(
            session_id="test_session",
            request_type=RequestType.CANCEL_TRIP,
            extracted_entities=[],
            metadata={
                "workflow_data": {
                    "cancel_flight": {
                        "cancellation_result": CancellationResult(
                            message="Flight cancelled successfully",
                            cancellation_charges=50.0,
                            refund_amount=150.0,
                            refund_date=datetime(2024, 1, 20)
                        )
                    }
                }
            }
        )
    
    @pytest.mark.asyncio
    async def test_format_cancellation_result(self, handler, task_context):
        """Test formatting cancellation result response"""
        # Execute task
        parameters = {"response_type": "cancellation_result"}
        result = await handler.execute(parameters, task_context)
        
        # Verify results
        assert result["success"] is True
        assert result["response_type"] == "cancellation_result"
        
        response = result["response"]
        assert response.status == "completed"
        assert "cancelled" in response.message
        assert response.data["refund_amount"] == 150.0
        assert response.data["cancellation_charges"] == 50.0


class TestTaskEngine:
    """Test complete task engine"""
    
    @pytest.fixture
    def task_engine(self):
        mock_airline_client = MockAirlineAPIClient()
        mock_policy_service = Mock()
        return TaskEngine(mock_airline_client, mock_policy_service)
    
    @pytest.mark.asyncio
    async def test_execute_get_customer_info_task(self, task_engine):
        """Test executing GET_CUSTOMER_INFO task"""
        # Create task definition
        task_def = TaskDefinition(
            task_id="test_extract",
            task_type=TaskType.GET_CUSTOMER_INFO,
            parameters={"extract_types": ["pnr"]},
            dependencies=[]
        )
        
        # Create task context
        context = TaskContext(
            session_id="test_session",
            request_type=RequestType.CANCEL_TRIP,
            extracted_entities=[],
            metadata={"utterance": "Cancel my flight ABC123"}
        )
        
        # Mock entity extractor
        mock_entities = [
            ExtractedEntity(
                type=EntityType.PNR,
                value="ABC123",
                confidence=0.95,
                start_index=18,
                end_index=24
            )
        ]
        task_engine.entity_extractor.extract_entities = Mock(return_value=mock_entities)
        
        # Execute task
        result = await task_engine.execute_task(task_def, context)
        
        # Verify results
        assert result.success is True
        assert result.data["success"] is True
        assert len(result.data["extracted_entities"]) == 1
        assert result.data["flight_identifiers"].pnr == "ABC123"


if __name__ == "__main__":
    pytest.main([__file__])