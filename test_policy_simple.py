#!/usr/bin/env python3
"""
Simple test to check policy scraping and RAG
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_policy_scraping():
    # Import only what we need
    from airline_service.services.policy_service import PolicyScraper, PolicyRAG
    from airline_service.config import config
    
    print("🧪 Testing Policy Scraping and RAG")
    print("=" * 50)
    
    scraper = PolicyScraper()
    rag = PolicyRAG()
    
    try:
        # Test scraping
        print("1. Testing cancellation policy scraping...")
        print(f"   URL: {config.policies.cancellation_policy_url}")
        
        policy_data = await scraper.scrape_policy_page(config.policies.cancellation_policy_url)
        print(f"   ✅ Successfully scraped!")
        print(f"   Title: {policy_data['title']}")
        print(f"   Sections found: {len(policy_data['sections'])}")
        
        # Show first few sections
        for i, section in enumerate(policy_data['sections'][:3]):
            print(f"   Section {i+1}: {section['title'][:50]}...")
            print(f"   Content preview: {section['content'][:100]}...")
            print()
        
        # Test RAG
        print("2. Testing RAG system...")
        rag.add_policy_data(policy_data)
        
        # Test search
        print("3. Testing search...")
        results = rag.search("cancellation policy", top_k=3)
        print(f"   Found {len(results)} results:")
        
        for i, result in enumerate(results):
            print(f"   Result {i+1}:")
            print(f"     Score: {result['score']:.3f}")
            print(f"     Section: {result['section']}")
            print(f"     Content: {result['content'][:150]}...")
            print()
        
        # Check if we have any content
        if not results:
            print("   ❌ No search results found!")
            print("   Checking RAG chunks...")
            print(f"   Total chunks: {len(rag.chunks)}")
            if rag.chunks:
                print("   First chunk content:")
                print(f"     {rag.chunks[0].content[:200]}...")
        else:
            print("   ✅ RAG search working!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await scraper.close()

if __name__ == "__main__":
    asyncio.run(test_policy_scraping())