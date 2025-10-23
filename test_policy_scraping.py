#!/usr/bin/env python3
"""
Test script to verify policy scraping is working
"""

import sys
import os
import asyncio

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_policy_scraping():
    """Test if policy scraping is working"""
    
    print("🧪 Testing Policy Scraping")
    print("=" * 50)
    
    try:
        from airline_service.services.policy_service import PolicyScraper
        from airline_service.config import config
        
        scraper = PolicyScraper()
        
        # Test scraping the cancellation policy URL
        print("1. Testing cancellation policy scraping...")
        print(f"   URL: {config.policies.cancellation_policy_url}")
        
        try:
            policy_data = await scraper.scrape_policy_page(config.policies.cancellation_policy_url)
            print(f"   ✅ Successfully scraped!")
            print(f"   Title: {policy_data['title']}")
            print(f"   Sections found: {len(policy_data['sections'])}")
            
            # Show first few sections
            for i, section in enumerate(policy_data['sections'][:3]):
                print(f"   Section {i+1}: {section['title'][:50]}...")
                print(f"   Content preview: {section['content'][:100]}...")
                print()
                
        except Exception as e:
            print(f"   ❌ Failed to scrape cancellation policy: {e}")
        
        # Test scraping the pet travel policy URL
        print("2. Testing pet travel policy scraping...")
        print(f"   URL: {config.policies.pet_travel_policy_url}")
        
        try:
            policy_data = await scraper.scrape_policy_page(config.policies.pet_travel_policy_url)
            print(f"   ✅ Successfully scraped!")
            print(f"   Title: {policy_data['title']}")
            print(f"   Sections found: {len(policy_data['sections'])}")
            
            # Show first few sections
            for i, section in enumerate(policy_data['sections'][:3]):
                print(f"   Section {i+1}: {section['title'][:50]}...")
                print(f"   Content preview: {section['content'][:100]}...")
                print()
                
        except Exception as e:
            print(f"   ❌ Failed to scrape pet travel policy: {e}")
        
        await scraper.close()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_policy_scraping())