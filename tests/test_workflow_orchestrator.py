"""
Tests for workflow orchestrator and workflow definitions
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from src.airline_service.services.workflow_orchestrator import (
    WorkflowOrchestrator, WorkflowDefinitionRegistry, DependencyResolver,
    WorkflowInstance, TaskInstance, WorkflowStatus, TaskStatus
)
from src.airline_service.types import (
    RequestType, TaskType, TaskDefinition, RequestContext,
    TaskContext, TaskResult, WorkflowResult, ExtractedEntity, EntityType
)


class TestWorkflowDefinitionRegistry:
    """Test workflow definition registry"""
    
    @pytest.fixture
    def registry(self):
        return WorkflowDefinitionRegistry()
    
    def test_get_cancel_trip_workflow(self, registry):
        """Test getting cancel trip workflow definition"""
        workflow = registry.get_workflow_definition(RequestType.CANCEL_TRIP)
        
        assert len(workflow) == 4
        assert workflow[0].task_id == "extract_identifiers"
        assert workflow[0].task_type == TaskType.GET_CUSTOMER_INFO
        assert workflow[1].task_id == "get_booking_details"
        assert workflow[1].dependencies == ["extract_identifiers"]
        assert workflow[2].task_id == "cancel_flight"
        assert workflow[3].task_id == "inform_cancellation_result"
    
    def test_get_flight_status_workflow(self, registry):
        """Test getting flight status workflow definition"""
        workflow = registry.get_workflow_definition(RequestType.FLIGHT_STATUS)
        
        assert len(workflow) == 3
        assert workflow[0].task_id == "extract_flight_identifiers"
        assert workflow[1].task_id == "get_flight_status"
        assert workflow[2].task_id == "inform_flight_status" 
   
    def test_get_seat_availability_workflow(self, registry):
        """Test getting seat availability workflow definition"""
        workflow = registry.get_workflow_definition(RequestType.SEAT_AVAILABILITY)
        
        assert len(workflow) == 4
        assert workflow[0].task_id == "extract_flight_info"
        assert workflow[1].task_id == "get_flight_details"
        assert workflow[2].task_id == "get_seat_availability"
        assert workflow[3].task_id == "inform_seat_availability"
    
    def test_get_cancellation_policy_workflow(self, registry):
        """Test getting cancellation policy workflow definition"""
        workflow = registry.get_workflow_definition(RequestType.CANCELLATION_POLICY)
        
        assert len(workflow) == 3
        assert workflow[0].task_id == "extract_flight_context"
        assert workflow[1].task_id == "get_cancellation_policy"
        assert workflow[2].task_id == "inform_policy_info"
    
    def test_get_pet_travel_workflow(self, registry):
        """Test getting pet travel workflow definition"""
        workflow = registry.get_workflow_definition(RequestType.PET_TRAVEL)
        
        assert len(workflow) == 3
        assert workflow[0].task_id == "extract_pet_context"
        assert workflow[1].task_id == "get_pet_travel_policy"
        assert workflow[2].task_id == "inform_pet_policy"
    
    def test_register_custom_workflow(self, registry):
        """Test registering custom workflow definition"""
        custom_tasks = [
            TaskDefinition(
                task_id="custom_task",
                task_type=TaskType.GET_CUSTOMER_INFO,
                parameters={},
                dependencies=[]
            )
        ]
        
        registry.register_workflow(RequestType.UNKNOWN, custom_tasks)
        workflow = registry.get_workflow_definition(RequestType.UNKNOWN)
        
        assert len(workflow) == 1
        assert workflow[0].task_id == "custom_task"


class TestDependencyResolver:
    """Test dependency resolution"""
    
    @pytest.fixture
    def resolver(self):
        return DependencyResolver()
    
    def test_resolve_linear_dependencies(self, resolver):
        """Test resolving linear task dependencies"""
        tasks = [
            TaskDefinition(task_id="task_c", task_type=TaskType.API_CALL, parameters={}, dependencies=["task_b"]),
            TaskDefinition(task_id="task_a", task_type=TaskType.GET_CUSTOMER_INFO, parameters={}, dependencies=[]),
            TaskDefinition(task_id="task_b", task_type=TaskType.API_CALL, parameters={}, dependencies=["task_a"])
        ]
        
        execution_order = resolver.resolve_execution_order(tasks)
        
        assert execution_order == ["task_a", "task_b", "task_c"]
    
    def test_resolve_parallel_dependencies(self, resolver):
        """Test resolving parallel task dependencies"""
        tasks = [
            TaskDefinition(task_id="task_d", task_type=TaskType.INFORM_CUSTOMER, parameters={}, dependencies=["task_b", "task_c"]),
            TaskDefinition(task_id="task_a", task_type=TaskType.GET_CUSTOMER_INFO, parameters={}, dependencies=[]),
            TaskDefinition(task_id="task_b", task_type=TaskType.API_CALL, parameters={}, dependencies=["task_a"]),
            TaskDefinition(task_id="task_c", task_type=TaskType.POLICY_LOOKUP, parameters={}, dependencies=["task_a"])
        ]
        
        execution_order = resolver.resolve_execution_order(tasks)
        
        assert execution_order[0] == "task_a"
        assert execution_order[-1] == "task_d"
        assert "task_b" in execution_order[1:3]
        assert "task_c" in execution_order[1:3]
    
    def test_circular_dependency_detection(self, resolver):
        """Test detection of circular dependencies"""
        tasks = [
            TaskDefinition(task_id="task_a", task_type=TaskType.GET_CUSTOMER_INFO, parameters={}, dependencies=["task_b"]),
            TaskDefinition(task_id="task_b", task_type=TaskType.API_CALL, parameters={}, dependencies=["task_a"])
        ]
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            resolver.resolve_execution_order(tasks)   
 
    def test_get_ready_tasks(self, resolver):
        """Test getting ready tasks based on dependencies"""
        # Create task instances
        tasks = {
            "task_a": TaskInstance(
                task_id="task_a",
                task_definition=TaskDefinition(task_id="task_a", task_type=TaskType.GET_CUSTOMER_INFO, parameters={}, dependencies=[]),
                status=TaskStatus.COMPLETED
            ),
            "task_b": TaskInstance(
                task_id="task_b", 
                task_definition=TaskDefinition(task_id="task_b", task_type=TaskType.API_CALL, parameters={}, dependencies=["task_a"]),
                status=TaskStatus.PENDING
            ),
            "task_c": TaskInstance(
                task_id="task_c",
                task_definition=TaskDefinition(task_id="task_c", task_type=TaskType.POLICY_LOOKUP, parameters={}, dependencies=["task_a"]),
                status=TaskStatus.PENDING
            ),
            "task_d": TaskInstance(
                task_id="task_d",
                task_definition=TaskDefinition(task_id="task_d", task_type=TaskType.INFORM_CUSTOMER, parameters={}, dependencies=["task_b", "task_c"]),
                status=TaskStatus.PENDING
            )
        }
        
        execution_order = ["task_a", "task_b", "task_c", "task_d"]
        ready_tasks = resolver.get_ready_tasks(tasks, execution_order)
        
        # task_b and task_c should be ready since task_a is completed
        assert "task_b" in ready_tasks
        assert "task_c" in ready_tasks
        assert "task_d" not in ready_tasks  # Still waiting for task_b and task_c


class TestWorkflowOrchestrator:
    """Test complete workflow orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        return WorkflowOrchestrator()
    
    @pytest.fixture
    def mock_task_handler(self):
        handler = AsyncMock()
        handler.execute.return_value = {"success": True, "data": "test_result"}
        return handler
    
    @pytest.fixture
    def request_context(self):
        return RequestContext(
            session_id="test_session",
            request_type=RequestType.CANCEL_TRIP,
            utterance="I want to cancel my flight ABC123",
            timestamp=datetime.now(),
            metadata={
                "extracted_entities": [
                    ExtractedEntity(
                        type=EntityType.PNR,
                        value="ABC123",
                        confidence=0.95,
                        start_index=30,
                        end_index=36
                    )
                ]
            }
        )
    
    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self, orchestrator, mock_task_handler, request_context):
        """Test executing a simple workflow"""
        # Register mock handler for all task types
        for task_type in TaskType:
            orchestrator.register_task_handler(task_type, mock_task_handler)
        
        # Execute workflow
        result = await orchestrator.execute_workflow(request_context)
        
        # Verify workflow completed successfully
        assert result.success is True
        assert len(result.executed_tasks) == 4  # Cancel trip workflow has 4 tasks
        assert "extract_identifiers" in result.executed_tasks
        assert "get_booking_details" in result.executed_tasks
        assert "cancel_flight" in result.executed_tasks
        assert "inform_cancellation_result" in result.executed_tasks
    
    @pytest.mark.asyncio
    async def test_workflow_with_task_failure(self, orchestrator, request_context):
        """Test workflow behavior when a task fails"""
        # Create a handler that fails on the second task
        failing_handler = AsyncMock()
        failing_handler.execute.side_effect = [
            {"success": True, "data": "first_task_result"},
            Exception("Task failed"),
            {"success": True, "data": "third_task_result"}
        ]
        
        # Register failing handler
        for task_type in TaskType:
            orchestrator.register_task_handler(task_type, failing_handler)
        
        # Execute workflow
        result = await orchestrator.execute_workflow(request_context)
        
        # Verify workflow failed
        assert result.success is False
        assert "failed" in result.message.lower()
        assert len(result.executed_tasks) >= 1  # At least first task should have executed
    
    def test_register_task_handler(self, orchestrator):
        """Test registering task handlers"""
        mock_handler = Mock()
        
        orchestrator.register_task_handler(TaskType.GET_CUSTOMER_INFO, mock_handler)
        
        assert TaskType.GET_CUSTOMER_INFO in orchestrator.task_handlers
        assert orchestrator.task_handlers[TaskType.GET_CUSTOMER_INFO] == mock_handler   
 
    @pytest.mark.asyncio
    async def test_workflow_with_unknown_request_type(self, orchestrator, request_context):
        """Test workflow execution with unknown request type"""
        # Modify request context to have unknown type
        request_context.request_type = RequestType.UNKNOWN
        
        # Execute workflow
        result = await orchestrator.execute_workflow(request_context)
        
        # Verify workflow failed due to no workflow definition
        assert result.success is False
        assert "No workflow defined" in result.message
    
    def test_get_workflow_status(self, orchestrator):
        """Test getting workflow status"""
        # Initially no active workflows
        status = orchestrator.get_workflow_status("nonexistent_id")
        assert status is None
        
        active_workflows = orchestrator.get_active_workflows()
        assert len(active_workflows) == 0
    
    def test_cancel_workflow(self, orchestrator):
        """Test cancelling a workflow"""
        # Test cancelling non-existent workflow
        result = orchestrator.cancel_workflow("nonexistent_id")
        assert result is False


class TestWorkflowInstance:
    """Test workflow instance data structures"""
    
    def test_workflow_instance_creation(self):
        """Test creating workflow instance"""
        context = RequestContext(
            session_id="test_session",
            request_type=RequestType.FLIGHT_STATUS,
            utterance="What's my flight status",
            timestamp=datetime.now(),
            metadata={}
        )
        
        workflow = WorkflowInstance(
            workflow_id="test_workflow",
            request_type=RequestType.FLIGHT_STATUS,
            context=context
        )
        
        assert workflow.workflow_id == "test_workflow"
        assert workflow.request_type == RequestType.FLIGHT_STATUS
        assert workflow.status == WorkflowStatus.CREATED
        assert len(workflow.tasks) == 0
        assert len(workflow.execution_order) == 0
    
    def test_task_instance_creation(self):
        """Test creating task instance"""
        task_def = TaskDefinition(
            task_id="test_task",
            task_type=TaskType.GET_CUSTOMER_INFO,
            parameters={"extract_types": ["pnr"]},
            dependencies=[]
        )
        
        task_instance = TaskInstance(
            task_id="test_task",
            task_definition=task_def
        )
        
        assert task_instance.task_id == "test_task"
        assert task_instance.status == TaskStatus.PENDING
        assert task_instance.result is None
        assert task_instance.retry_count == 0
        assert task_instance.max_retries == 3


if __name__ == "__main__":
    pytest.main([__file__])