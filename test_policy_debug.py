#!/usr/bin/env python3
"""
Debug script to test policy service RAG functionality
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_policy():
    from airline_service.services.policy_service import PolicyService
    
    policy_service = PolicyService()
    await policy_service.initialize()
    
    # Test the RAG search directly
    print('Testing RAG search...')
    results = policy_service.rag.search('cancellation policy', top_k=3)
    print(f'Found {len(results)} results:')
    for i, result in enumerate(results):
        print(f'{i+1}. Score: {result["score"]:.3f}')
        print(f'   Section: {result["section"]}')
        print(f'   Content: {result["content"][:200]}...')
        print()
    
    # Test the policy service method
    print('Testing policy service method...')
    policy_info = await policy_service.get_general_cancellation_policy()
    print(f'Policy content length: {len(policy_info.content)}')
    print(f'Policy content preview: {policy_info.content[:300]}...')
    
    await policy_service.close()

if __name__ == "__main__":
    asyncio.run(test_policy())