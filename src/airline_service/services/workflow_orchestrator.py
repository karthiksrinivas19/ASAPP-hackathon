"""
Workflow orchestrator for managing task execution sequences
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from ..types import (
    RequestType, TaskType, TaskDefinition, TaskResult, WorkflowResult,
    RequestContext, TaskContext, ExtractedEntity
)


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInstance:
    """Instance of a task in workflow execution"""
    task_id: str
    task_definition: TaskDefinition
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkflowInstance:
    """Instance of a workflow execution"""
    workflow_id: str
    request_type: RequestType
    context: RequestContext
    status: WorkflowStatus = WorkflowStatus.CREATED
    tasks: Dict[str, TaskInstance] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    result: Optional[WorkflowResult] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowDefinitionRegistry:
    """Registry for workflow definitions by request type"""
    
    def __init__(self):
        self._workflows: Dict[RequestType, List[TaskDefinition]] = {}
        self._initialize_default_workflows()
    
    def _initialize_default_workflows(self):
        """Initialize default workflow definitions for each request type"""
        
        # Cancel Trip Workflow
        self._workflows[RequestType.CANCEL_TRIP] = [
            TaskDefinition(
                task_id="extract_identifiers",
                task_type=TaskType.GET_CUSTOMER_INFO,
                parameters={"extract_types": ["pnr", "flight_number", "passenger_name", "date"]},
                dependencies=[]
            ),
            TaskDefinition(
                task_id="get_booking_details",
                task_type=TaskType.API_CALL,
                parameters={"api_method": "get_booking_details", "required_data": ["pnr"]},
                dependencies=["extract_identifiers"]
            ),
            TaskDefinition(
                task_id="confirm_booking_details",
                task_type=TaskType.INFORM_CUSTOMER,
                parameters={"response_type": "booking_confirmation", "require_confirmation": True},
                dependencies=["get_booking_details"]
            ),
            TaskDefinition(
                task_id="cancel_flight",
                task_type=TaskType.API_CALL,
                parameters={"api_method": "cancel_flight", "required_data": ["booking_details"]},
                dependencies=["confirm_booking_details"]
            ),
            TaskDefinition(
                task_id="inform_cancellation_result",
                task_type=TaskType.INFORM_CUSTOMER,
                parameters={"response_type": "cancellation_result"},
                dependencies=["cancel_flight"]
            )
        ]
        
        # Flight Status Workflow
        self._workflows[RequestType.FLIGHT_STATUS] = [
            TaskDefinition(
                task_id="extract_flight_identifiers",
                task_type=TaskType.GET_CUSTOMER_INFO,
                parameters={"extract_types": ["pnr", "flight_number", "date", "route"]},
                dependencies=[]
            ),
            TaskDefinition(
                task_id="get_flight_status",
                task_type=TaskType.API_CALL,
                parameters={"api_method": "get_booking_details", "required_data": ["pnr_or_flight"]},
                dependencies=["extract_flight_identifiers"]
            ),
            TaskDefinition(
                task_id="inform_flight_status",
                task_type=TaskType.INFORM_CUSTOMER,
                parameters={"response_type": "flight_status"},
                dependencies=["get_flight_status"]
            )
        ]
        
        # Seat Availability Workflow
        self._workflows[RequestType.SEAT_AVAILABILITY] = [
            TaskDefinition(
                task_id="extract_flight_info",
                task_type=TaskType.GET_CUSTOMER_INFO,
                parameters={"extract_types": ["pnr", "flight_number", "date"]},
                dependencies=[]
            ),
            TaskDefinition(
                task_id="get_flight_details",
                task_type=TaskType.API_CALL,
                parameters={"api_method": "get_booking_details", "required_data": ["pnr"]},
                dependencies=["extract_flight_info"]
            ),
            TaskDefinition(
                task_id="get_seat_availability",
                task_type=TaskType.API_CALL,
                parameters={"api_method": "get_available_seats", "required_data": ["flight_info"]},
                dependencies=["get_flight_details"]
            ),
            TaskDefinition(
                task_id="inform_seat_availability",
                task_type=TaskType.INFORM_CUSTOMER,
                parameters={"response_type": "seat_availability"},
                dependencies=["get_seat_availability"]
            )
        ]
        
        # Cancellation Policy Workflow
        self._workflows[RequestType.CANCELLATION_POLICY] = [
            TaskDefinition(
                task_id="extract_flight_context",
                task_type=TaskType.GET_CUSTOMER_INFO,
                parameters={"extract_types": ["fare_type", "flight_number", "date"], "optional": True},
                dependencies=[]
            ),
            TaskDefinition(
                task_id="get_cancellation_policy",
                task_type=TaskType.POLICY_LOOKUP,
                parameters={"policy_type": "cancellation", "context_aware": True},
                dependencies=["extract_flight_context"]
            ),
            TaskDefinition(
                task_id="inform_policy_info",
                task_type=TaskType.INFORM_CUSTOMER,
                parameters={"response_type": "policy_info"},
                dependencies=["get_cancellation_policy"]
            )
        ]
        
        # Pet Travel Workflow
        self._workflows[RequestType.PET_TRAVEL] = [
            TaskDefinition(
                task_id="extract_pet_context",
                task_type=TaskType.GET_CUSTOMER_INFO,
                parameters={"extract_types": ["pet_type", "destination", "pet_size"], "optional": True},
                dependencies=[]
            ),
            TaskDefinition(
                task_id="get_pet_travel_policy",
                task_type=TaskType.POLICY_LOOKUP,
                parameters={"policy_type": "pet_travel", "context_aware": True},
                dependencies=["extract_pet_context"]
            ),
            TaskDefinition(
                task_id="inform_pet_policy",
                task_type=TaskType.INFORM_CUSTOMER,
                parameters={"response_type": "policy_info"},
                dependencies=["get_pet_travel_policy"]
            )
        ]
    
    def get_workflow_definition(self, request_type: RequestType) -> List[TaskDefinition]:
        """Get workflow definition for request type"""
        return self._workflows.get(request_type, [])
    
    def register_workflow(self, request_type: RequestType, tasks: List[TaskDefinition]):
        """Register a custom workflow definition"""
        self._workflows[request_type] = tasks
    
    def get_all_workflows(self) -> Dict[RequestType, List[TaskDefinition]]:
        """Get all registered workflows"""
        return self._workflows.copy()


class DependencyResolver:
    """Resolves task dependencies and determines execution order"""
    
    def resolve_execution_order(self, tasks: List[TaskDefinition]) -> List[str]:
        """Resolve task execution order based on dependencies"""
        task_map = {task.task_id: task for task in tasks}
        resolved = []
        visited = set()
        temp_visited = set()
        
        def visit(task_id: str):
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving task: {task_id}")
            
            if task_id in visited:
                return
            
            temp_visited.add(task_id)
            
            task = task_map.get(task_id)
            if not task:
                raise ValueError(f"Task not found: {task_id}")
            
            # Visit all dependencies first
            for dep_id in task.dependencies:
                visit(dep_id)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            resolved.append(task_id)
        
        # Visit all tasks
        for task in tasks:
            if task.task_id not in visited:
                visit(task.task_id)
        
        return resolved
    
    def get_ready_tasks(self, tasks: Dict[str, TaskInstance], execution_order: List[str]) -> List[str]:
        """Get tasks that are ready to execute (all dependencies completed)"""
        ready = []
        
        for task_id in execution_order:
            task = tasks[task_id]
            
            # Skip if already completed or running
            if task.status in [TaskStatus.COMPLETED, TaskStatus.RUNNING, TaskStatus.SKIPPED]:
                continue
            
            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in task.task_definition.dependencies:
                dep_task = tasks.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                ready.append(task_id)
        
        return ready


class WorkflowOrchestrator:
    """Main workflow orchestrator that manages task execution sequences"""
    
    def __init__(self):
        self.registry = WorkflowDefinitionRegistry()
        self.dependency_resolver = DependencyResolver()
        self.active_workflows: Dict[str, WorkflowInstance] = {}
        self.task_handlers: Dict[TaskType, Any] = {}
    
    def register_task_handler(self, task_type: TaskType, handler):
        """Register a task handler for specific task type"""
        self.task_handlers[task_type] = handler
    
    async def execute_workflow(self, request_context: RequestContext) -> WorkflowResult:
        """Execute workflow for a request"""
        workflow_id = str(uuid.uuid4())
        
        try:
            # Get workflow definition
            task_definitions = self.registry.get_workflow_definition(request_context.request_type)
            if not task_definitions:
                raise ValueError(f"No workflow defined for request type: {request_context.request_type}")
            
            # Create workflow instance
            workflow = WorkflowInstance(
                workflow_id=workflow_id,
                request_type=request_context.request_type,
                context=request_context,
                started_at=datetime.now()
            )
            
            # Create task instances
            for task_def in task_definitions:
                task_instance = TaskInstance(
                    task_id=task_def.task_id,
                    task_definition=task_def
                )
                workflow.tasks[task_def.task_id] = task_instance
            
            # Resolve execution order
            workflow.execution_order = self.dependency_resolver.resolve_execution_order(task_definitions)
            workflow.status = WorkflowStatus.RUNNING
            
            # Store active workflow
            self.active_workflows[workflow_id] = workflow
            
            # Execute workflow
            result = await self._execute_workflow_tasks(workflow)
            
            # Update workflow status
            workflow.status = WorkflowStatus.COMPLETED if result.success else WorkflowStatus.FAILED
            workflow.completed_at = datetime.now()
            workflow.result = result
            
            return result
        
        except Exception as e:
            # Mark workflow as failed
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].status = WorkflowStatus.FAILED
                self.active_workflows[workflow_id].completed_at = datetime.now()
            
            return WorkflowResult(
                success=False,
                message=f"Workflow execution failed: {str(e)}",
                executed_tasks=[],
                duration=0.0
            )
        
        finally:
            # Clean up completed workflow
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
    
    async def _execute_workflow_tasks(self, workflow: WorkflowInstance) -> WorkflowResult:
        """Execute all tasks in the workflow"""
        start_time = datetime.now()
        executed_tasks = []
        workflow_data = {}
        
        while True:
            # Get ready tasks
            ready_tasks = self.dependency_resolver.get_ready_tasks(
                workflow.tasks, 
                workflow.execution_order
            )
            
            if not ready_tasks:
                # Check if all tasks are completed
                all_completed = all(
                    task.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED]
                    for task in workflow.tasks.values()
                )
                
                if all_completed:
                    break
                else:
                    # No ready tasks but not all completed - likely a dependency issue
                    failed_tasks = [
                        task.task_id for task in workflow.tasks.values()
                        if task.status == TaskStatus.FAILED
                    ]
                    
                    if failed_tasks:
                        return WorkflowResult(
                            success=False,
                            message=f"Workflow failed due to failed tasks: {failed_tasks}",
                            executed_tasks=executed_tasks,
                            duration=(datetime.now() - start_time).total_seconds()
                        )
                    
                    # Wait a bit and try again
                    await asyncio.sleep(0.1)
                    continue
            
            # Execute ready tasks (can be parallel in the future)
            for task_id in ready_tasks:
                task_instance = workflow.tasks[task_id]
                workflow.current_task = task_id
                
                try:
                    # Execute task
                    task_result = await self._execute_task(task_instance, workflow, workflow_data)
                    
                    # Update task status
                    task_instance.result = task_result
                    task_instance.status = TaskStatus.COMPLETED if task_result.success else TaskStatus.FAILED
                    task_instance.completed_at = datetime.now()
                    
                    # Store task data for subsequent tasks
                    if task_result.success and task_result.data:
                        workflow_data[task_id] = task_result.data
                    
                    executed_tasks.append(task_id)
                    
                    # If task failed and is critical, fail the workflow
                    if not task_result.success and not task_instance.task_definition.parameters.get("optional", False):
                        return WorkflowResult(
                            success=False,
                            message=f"Critical task failed: {task_id} - {task_result.error}",
                            executed_tasks=executed_tasks,
                            duration=(datetime.now() - start_time).total_seconds()
                        )
                
                except Exception as e:
                    task_instance.status = TaskStatus.FAILED
                    task_instance.error = str(e)
                    task_instance.completed_at = datetime.now()
                    
                    return WorkflowResult(
                        success=False,
                        message=f"Task execution error: {task_id} - {str(e)}",
                        executed_tasks=executed_tasks,
                        duration=(datetime.now() - start_time).total_seconds()
                    )
        
        # Workflow completed successfully
        final_result = workflow_data.get(executed_tasks[-1]) if executed_tasks else None
        
        return WorkflowResult(
            success=True,
            message="Workflow completed successfully",
            data=final_result,
            executed_tasks=executed_tasks,
            duration=(datetime.now() - start_time).total_seconds()
        )
    
    async def _execute_task(self, task_instance: TaskInstance, workflow: WorkflowInstance, workflow_data: Dict[str, Any]) -> TaskResult:
        """Execute a single task"""
        task_instance.status = TaskStatus.RUNNING
        task_instance.started_at = datetime.now()
        
        # Get task handler
        handler = self.task_handlers.get(task_instance.task_definition.task_type)
        if not handler:
            raise ValueError(f"No handler registered for task type: {task_instance.task_definition.task_type}")
        
        # Create task context
        task_context = TaskContext(
            session_id=workflow.context.session_id,
            request_type=workflow.context.request_type,
            extracted_entities=workflow.context.metadata.get("extracted_entities", []),
            metadata={
                "workflow_data": workflow_data,
                "task_parameters": task_instance.task_definition.parameters,
                "utterance": workflow.context.utterance
            }
        )
        
        # Execute task
        try:
            result = await handler.execute(task_instance.task_definition.parameters, task_context)
            
            return TaskResult(
                success=True,
                data=result,
                duration=(datetime.now() - task_instance.started_at).total_seconds()
            )
        
        except Exception as e:
            return TaskResult(
                success=False,
                error=str(e),
                duration=(datetime.now() - task_instance.started_at).total_seconds()
            )
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """Get status of an active workflow"""
        return self.active_workflows.get(workflow_id)
    
    def get_active_workflows(self) -> Dict[str, WorkflowInstance]:
        """Get all active workflows"""
        return self.active_workflows.copy()
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id].status = WorkflowStatus.CANCELLED
            return True
        return False


# Global workflow orchestrator instance
workflow_orchestrator = WorkflowOrchestrator()