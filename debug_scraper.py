#!/usr/bin/env python3
"""
Debug script to test policy scraping and see what's actually being extracted
"""

import asyncio
import sys
import os
from bs4 import BeautifulSoup
import httpx

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def debug_scraping():
    print("🔍 Debugging Policy Scraping")
    print("=" * 50)
    
    urls = [
        "https://www.jetblue.com/flying-with-us/our-fares",
        "https://www.jetblue.com/traveling-together/traveling-with-pets"
    ]
    
    client = httpx.AsyncClient(
        timeout=30.0,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    )
    
    for url in urls:
        print(f"\n📄 Testing: {url}")
        print("-" * 40)
        
        try:
            response = await client.get(url)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Get title
                title = soup.find('title')
                print(f"Title: {title.get_text().strip() if title else 'No title'}")
                
                # Look for headings
                headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                print(f"Found {len(headings)} headings:")
                for i, h in enumerate(headings[:5]):  # Show first 5
                    print(f"  {i+1}. {h.name}: {h.get_text().strip()[:80]}...")
                
                # Look for main content areas
                content_selectors = [
                    'main', '.main-content', '.content', '#content', 
                    '.policy-content', 'article', '.article-content',
                    '[role="main"]', '.page-content'
                ]
                
                main_content = None
                for selector in content_selectors:
                    main_content = soup.select_one(selector)
                    if main_content:
                        print(f"Found main content with selector: {selector}")
                        break
                
                if not main_content:
                    main_content = soup.find('body')
                    print("Using body as main content")
                
                # Get all text content
                if main_content:
                    all_text = main_content.get_text(separator=' ', strip=True)
                    print(f"Total text length: {len(all_text)} characters")
                    print(f"Preview: {all_text[:300]}...")
                    
                    # Look for policy-related keywords
                    keywords = ['cancel', 'refund', 'policy', 'fee', 'pet', 'animal', 'travel']
                    found_keywords = [kw for kw in keywords if kw.lower() in all_text.lower()]
                    print(f"Found keywords: {found_keywords}")
                
            else:
                print(f"Failed to fetch: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    await client.aclose()

if __name__ == "__main__":
    asyncio.run(debug_scraping())