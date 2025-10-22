#!/usr/bin/env python3
"""
Demonstration script showing task handlers in action
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from src.airline_service.services.task_engine import (
    GetCustomerInfoHandler, APICallHandler, PolicyLookupHandler, InformCustomerHandler
)
from src.airline_service.clients.airline_api_client import MockAirlineAPIClient
from src.airline_service.services.policy_service import PolicyService
from src.airline_service.ml.improved_entity_extractor import ImprovedEntityExtractor
from src.airline_service.types import (
    TaskContext, RequestType, ExtractedEntity, EntityType
)


async def demo_get_customer_info_handler():
    """Demonstrate GET_CUSTOMER_INFO handler"""
    print("🔍 GET_CUSTOMER_INFO Handler Demo")
    print("=" * 50)
    
    # Initialize handler
    entity_extractor = ImprovedEntityExtractor()
    from src.airline_service.services.task_engine import AutoDataRetriever
    airline_client = MockAirlineAPIClient()
    auto_retriever = AutoDataRetriever(airline_client, entity_extractor)
    handler = GetCustomerInfoHandler(entity_extractor, auto_retriever)
    
    # Create test context
    context = TaskContext(
        session_id="demo_session",
        request_type=RequestType.CANCEL_TRIP,
        extracted_entities=[],
        metadata={"utterance": "I want to cancel my flight ABC123 for John Smith"}
    )
    
    # Test parameters
    parameters = {"extract_types": ["pnr", "passenger_name", "phone", "email"]}
    
    # Execute handler
    result = await handler.execute(parameters, context)
    
    print(f"📝 Input: {context.metadata['utterance']}")
    print(f"🎯 Extracted entities: {len(result['extracted_entities'])}")
    for entity in result['extracted_entities']:
        print(f"   - {entity.type.value}: {entity.value} (confidence: {entity.confidence:.2f})")
    
    print(f"✈️  Flight identifiers:")
    flight_ids = result['flight_identifiers']
    if flight_ids.pnr:
        print(f"   - PNR: {flight_ids.pnr}")
    if flight_ids.passenger_name:
        print(f"   - Passenger: {flight_ids.passenger_name}")
    
    print(f"👤 Customer info: {result['customer_info']}")
    print()


async def demo_api_call_handler():
    """Demonstrate API_CALL handler"""
    print("🌐 API_CALL Handler Demo")
    print("=" * 50)
    
    # Initialize handler
    airline_client = MockAirlineAPIClient()
    entity_extractor = ImprovedEntityExtractor()
    from src.airline_service.services.task_engine import AutoDataRetriever
    auto_retriever = AutoDataRetriever(airline_client, entity_extractor)
    handler = APICallHandler(airline_client, auto_retriever)
    
    # Create test context with PNR entity
    context = TaskContext(
        session_id="demo_session",
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
    
    # Test get_booking_details
    print("📋 Testing get_booking_details...")
    parameters = {"api_method": "get_booking_details", "required_data": ["pnr"]}
    result = await handler.execute(parameters, context)
    
    if result['success']:
        booking = result['booking_details']
        print(f"✅ Retrieved booking: {booking.pnr}")
        print(f"   - Flight: {booking.source_airport_code} → {booking.destination_airport_code}")
        print(f"   - Departure: {booking.scheduled_departure}")
        print(f"   - Status: {booking.current_status}")
        
        # Test cancel_flight using the retrieved booking
        print("\n🚫 Testing cancel_flight...")
        context.metadata["workflow_data"]["get_booking_details"] = result
        cancel_params = {"api_method": "cancel_flight", "required_data": ["booking_details"]}
        cancel_result = await handler.execute(cancel_params, context)
        
        if cancel_result['success']:
            cancellation = cancel_result['cancellation_result']
            print(f"✅ Cancellation successful!")
            print(f"   - Charges: ${cancellation.cancellation_charges}")
            print(f"   - Refund: ${cancellation.refund_amount}")
            print(f"   - Refund date: {cancellation.refund_date.strftime('%Y-%m-%d')}")
    
    print()


async def demo_policy_lookup_handler():
    """Demonstrate POLICY_LOOKUP handler"""
    print("📋 POLICY_LOOKUP Handler Demo")
    print("=" * 50)
    
    # Initialize handler
    policy_service = PolicyService()
    handler = PolicyLookupHandler(policy_service)
    
    # Create test context
    context = TaskContext(
        session_id="demo_session",
        request_type=RequestType.CANCELLATION_POLICY,
        extracted_entities=[],
        metadata={}
    )
    
    # Test cancellation policy lookup
    print("📜 Testing cancellation policy lookup...")
    parameters = {"policy_type": "cancellation", "context_aware": False}
    
    try:
        result = await handler.execute(parameters, context)
        
        if result['success']:
            policy = result['policy_info']
            print(f"✅ Policy retrieved: {policy.policy_type}")
            print(f"   - Last updated: {policy.last_updated.strftime('%Y-%m-%d %H:%M')}")
            print(f"   - Conditions: {policy.applicable_conditions}")
            print(f"   - Content preview: {policy.content[:100]}...")
    
    except Exception as e:
        print(f"⚠️  Policy lookup failed (expected in demo): {e}")
    
    print()


async def demo_inform_customer_handler():
    """Demonstrate INFORM_CUSTOMER handler"""
    print("💬 INFORM_CUSTOMER Handler Demo")
    print("=" * 50)
    
    # Initialize handler
    handler = InformCustomerHandler()
    
    # Create mock cancellation result
    from src.airline_service.types import CancellationResult
    mock_cancellation = CancellationResult(
        message="Flight cancelled successfully",
        cancellation_charges=50.0,
        refund_amount=150.0,
        refund_date=datetime(2024, 1, 20)
    )
    
    # Create test context with workflow data
    context = TaskContext(
        session_id="demo_session",
        request_type=RequestType.CANCEL_TRIP,
        extracted_entities=[],
        metadata={
            "workflow_data": {
                "cancel_flight": {
                    "cancellation_result": mock_cancellation
                }
            }
        }
    )
    
    # Test cancellation result formatting
    print("📝 Testing cancellation result formatting...")
    parameters = {"response_type": "cancellation_result"}
    result = await handler.execute(parameters, context)
    
    if result['success']:
        response = result['response']
        print(f"✅ Response formatted:")
        print(f"   - Status: {response.status}")
        print(f"   - Message: {response.message}")
        print(f"   - Data keys: {list(response.data.keys())}")
        print(f"   - Refund amount: ${response.data['refund_amount']}")
    
    print()


async def main():
    """Run all handler demonstrations"""
    print("🎭 Task Handler Demonstration")
    print("=" * 60)
    print("This script demonstrates each task handler type in action.\n")
    
    try:
        await demo_get_customer_info_handler()
        await demo_api_call_handler()
        await demo_policy_lookup_handler()
        await demo_inform_customer_handler()
        
        print("🎉 All handler demonstrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())