#!/usr/bin/env python3
"""
Interactive test script for airline customer service
"""

import requests
import json
from datetime import datetime

def test_customer_service():
    """Interactive customer service tester"""
    
    base_url = "http://localhost:8000"
    
    print("🛫 Airline Customer Service - Interactive Tester")
    print("=" * 60)
    print("Server URL:", base_url)
    print("Type 'quit' to exit, 'help' for examples")
    print()
    
    # Test server connection
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and healthy!")
        else:
            print("⚠️  Server responded but may have issues")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Make sure it's running on port 8000")
        return
    
    session_id = f"interactive-{int(datetime.now().timestamp())}"
    
    while True:
        print("\n" + "-" * 60)
        user_input = input("👤 Customer: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'help':
            show_examples()
            continue
        
        if not user_input:
            print("Please enter a message or 'help' for examples")
            continue
        
        # Send request to API
        try:
            payload = {
                "utterance": user_input,
                "session_id": session_id
            }
            
            print("🤖 Processing your request...")
            response = requests.post(
                f"{base_url}/api/v1/customer-service/query",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"🤖 Assistant: {result['message']}")
                
                # Show additional details if available
                if result.get('data'):
                    data = result['data']
                    print(f"   📊 Request Type: {data.get('request_type', 'unknown')}")
                    print(f"   🎯 Confidence: {data.get('confidence', 0):.2f}")
                    
                    if data.get('fallback_mode'):
                        print("   ⚠️  Note: Using simplified processing")
            else:
                error_data = response.json()
                print(f"❌ Error: {error_data.get('message', 'Unknown error')}")
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            print(f"🔌 Connection error: {str(e)}")
        except json.JSONDecodeError:
            print("📄 Invalid response format from server")
        except Exception as e:
            print(f"💥 Unexpected error: {str(e)}")

def show_examples():
    """Show example queries"""
    print("\n💡 Example Customer Queries:")
    print("=" * 40)
    examples = [
        "I want to cancel my flight booking ABC123",
        "What's the status of flight JB1234?",
        "Can I see available seats on my flight?",
        "What's your cancellation policy?",
        "Can I travel with my dog?",
        "I need to change my seat to window",
        "My flight is delayed, what are my options?",
        "How much does it cost to cancel?",
        "I want to upgrade to business class"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i:2d}. {example}")
    print()

if __name__ == "__main__":
    test_customer_service()