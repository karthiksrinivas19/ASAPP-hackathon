"""
Policy service with web scraping and RAG capabilities
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from ..types import PolicyInfo
from ..config import config
from .cache_service import cache_service, policy_cache
from .connection_pool import pooled_http_client


class PolicyChunk:
    """Represents a chunk of policy content with metadata"""
    
    def __init__(self, content: str, url: str, section: str, chunk_id: str):
        self.content = content
        self.url = url
        self.section = section
        self.chunk_id = chunk_id
        self.embedding: Optional[np.ndarray] = None
        self.timestamp = datetime.now()


class PolicyCache:
    """Simple file-based cache for policy content"""
    
    def __init__(self, cache_dir: str = "./cache/policies"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = config.policies.cache_ttl  # seconds
    
    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.json"
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached content if not expired"""
        cache_path = self._get_cache_path(url)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > timedelta(seconds=self.ttl):
                return None
            
            return cached_data
        
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def set(self, url: str, content: str, metadata: Dict[str, Any]) -> None:
        """Cache content with metadata"""
        cache_path = self._get_cache_path(url)
        
        cache_data = {
            'url': url,
            'content': content,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)


class PolicyScraper:
    """Web scraper for JetBlue policy pages"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
    
    async def scrape_policy_page(self, url: str) -> Dict[str, Any]:
        """Scrape a policy page and extract structured content"""
        try:
            # Try multiple requests with different headers to handle JS sites
            headers_list = [
                {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            ]
            
            response = None
            for headers in headers_list:
                try:
                    temp_client = httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True)
                    response = await temp_client.get(url)
                    await temp_client.aclose()
                    
                    if response.status_code == 200 and len(response.content) > 1000:
                        break
                except:
                    continue
            
            if not response or response.status_code != 200:
                raise Exception(f"Failed to fetch content from {url}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements but keep more content
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Policy Page"
            
            # Try multiple content extraction strategies
            sections = []
            
            # Strategy 1: Look for specific JetBlue content patterns
            jetblue_selectors = [
                '[data-testid*="content"]',
                '.rich-text',
                '.policy-section',
                '.fare-details',
                '.content-block',
                '[class*="content"]',
                '[class*="policy"]',
                '[class*="fare"]'
            ]
            
            for selector in jetblue_selectors:
                elements = soup.select(selector)
                if elements:
                    for elem in elements:
                        text = elem.get_text(strip=True)
                        if len(text) > 50:  # Substantial content
                            sections.append({
                                'title': f'Policy Section {len(sections) + 1}',
                                'content': text
                            })
            
            # Strategy 2: Extract from common content areas
            if not sections:
                content_selectors = [
                    'main', '[role="main"]', '.main-content', '.content', 
                    '#content', 'article', '.article-content', '.page-content'
                ]
                
                main_content = None
                for selector in content_selectors:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                
                if main_content:
                    sections = self._extract_sections(main_content)
            
            # Strategy 3: Fallback to all paragraphs and divs
            if not sections:
                all_elements = soup.find_all(['p', 'div', 'section', 'li'])
                content_parts = []
                
                for elem in all_elements:
                    text = elem.get_text(strip=True)
                    if len(text) > 30 and text not in content_parts:  # Avoid duplicates
                        content_parts.append(text)
                
                if content_parts:
                    # Group content into sections
                    chunk_size = 5
                    for i in range(0, len(content_parts), chunk_size):
                        chunk = content_parts[i:i+chunk_size]
                        sections.append({
                            'title': f'Policy Information {i//chunk_size + 1}',
                            'content': ' '.join(chunk)
                        })
            
            # If still no content, this might be a JS-heavy site
            if not sections:
                print(f"Warning: No content extracted from {url} - might be JavaScript-rendered")
                # Return empty sections to trigger fallback
                sections = []
            
            return {
                'url': url,
                'title': title_text,
                'sections': sections,
                'scraped_at': datetime.now().isoformat()
            }
        
        except Exception as e:
            raise Exception(f"Failed to scrape {url}: {str(e)}")
    
    def _extract_sections(self, content_element) -> List[Dict[str, str]]:
        """Extract sections from content element"""
        sections = []
        
        # First try to extract all text content as a single section if no headings found
        all_text = content_element.get_text(separator='\n', strip=True)
        
        # Find all headings and their content
        headings = content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        if not headings:
            # If no headings found, create sections from paragraphs and divs
            paragraphs = content_element.find_all(['p', 'div', 'section', 'article'])
            
            current_section = {
                'title': 'Policy Information',
                'content': '',
                'level': 1
            }
            
            for para in paragraphs:
                text = para.get_text(strip=True)
                if text and len(text) > 20:  # Only include substantial content
                    current_section['content'] += text + '\n\n'
            
            if current_section['content'].strip():
                sections.append(current_section)
        
        else:
            # Process headings normally
            for i, heading in enumerate(headings):
                section_title = heading.get_text().strip()
                
                # Get content until next heading
                content_parts = []
                current = heading.next_sibling
                
                while current:
                    if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        break
                    
                    if hasattr(current, 'get_text'):
                        text = current.get_text().strip()
                        if text:
                            content_parts.append(text)
                    elif isinstance(current, str):
                        text = current.strip()
                        if text:
                            content_parts.append(text)
                    
                    current = current.next_sibling
                
                section_content = ' '.join(content_parts)
                
                if section_content:
                    sections.append({
                        'title': section_title,
                        'content': section_content
                    })
        
        # If no sections found, try alternative extraction methods
        if not sections:
            # Try to find content in common containers
            content_containers = content_element.find_all(['div', 'section', 'article', 'p'])
            
            for container in content_containers:
                text = container.get_text().strip()
                if len(text) > 100:  # Only substantial content
                    # Try to find a title from nearby elements
                    title = "Policy Information"
                    prev_heading = container.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if prev_heading:
                        title = prev_heading.get_text().strip()
                    
                    sections.append({
                        'title': title,
                        'content': text
                    })
                    
                    # Limit to avoid too many sections
                    if len(sections) >= 10:
                        break
        
        # Final fallback: extract all text and split into chunks
        if not sections:
            all_text = content_element.get_text()
            # Split into paragraphs
            paragraphs = [p.strip() for p in all_text.split('\n') if p.strip()]
            
            current_section = []
            current_length = 0
            
            for paragraph in paragraphs:
                if len(paragraph) > 20:  # Skip very short lines
                    if current_length + len(paragraph) > 1000 and current_section:
                        # Create a section
                        sections.append({
                            'title': f'Policy Section {len(sections) + 1}',
                            'content': ' '.join(current_section)
                        })
                        current_section = [paragraph]
                        current_length = len(paragraph)
                    else:
                        current_section.append(paragraph)
                        current_length += len(paragraph)
            
            # Add remaining content
            if current_section:
                sections.append({
                    'title': f'Policy Section {len(sections) + 1}',
                    'content': ' '.join(current_section)
                })
        
        return sections
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class PolicyRAG:
    """RAG system for policy content using sentence transformers and FAISS"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks: List[PolicyChunk] = []
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
    def _load_model(self):
        """Lazy load the sentence transformer model"""
        if self.model is None:
            print(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def _create_chunks(self, policy_data: Dict[str, Any]) -> List[PolicyChunk]:
        """Create chunks from policy data"""
        chunks = []
        url = policy_data['url']
        
        # Debug logging
        print(f"Creating chunks for {url}")
        print(f"Policy data keys: {list(policy_data.keys())}")
        sections = policy_data.get('sections', [])
        print(f"Found {len(sections)} sections")
        
        for section in sections:
            section_title = section['title']
            section_content = section['content']
            
            # Split long sections into smaller chunks
            chunk_size = 500  # characters
            if len(section_content) <= chunk_size:
                chunk_id = f"{url}#{len(chunks)}"
                chunk = PolicyChunk(
                    content=f"{section_title}: {section_content}",
                    url=url,
                    section=section_title,
                    chunk_id=chunk_id
                )
                chunks.append(chunk)
            else:
                # Split into smaller chunks
                words = section_content.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) > chunk_size and current_chunk:
                        chunk_content = f"{section_title}: {' '.join(current_chunk)}"
                        chunk_id = f"{url}#{len(chunks)}"
                        chunk = PolicyChunk(
                            content=chunk_content,
                            url=url,
                            section=section_title,
                            chunk_id=chunk_id
                        )
                        chunks.append(chunk)
                        current_chunk = [word]
                        current_length = len(word)
                    else:
                        current_chunk.append(word)
                        current_length += len(word) + 1
                
                # Add remaining chunk
                if current_chunk:
                    chunk_content = f"{section_title}: {' '.join(current_chunk)}"
                    chunk_id = f"{url}#{len(chunks)}"
                    chunk = PolicyChunk(
                        content=chunk_content,
                        url=url,
                        section=section_title,
                        chunk_id=chunk_id
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def add_policy_data(self, policy_data: Dict[str, Any]):
        """Add policy data to the RAG system"""
        self._load_model()
        
        # Create chunks
        new_chunks = self._create_chunks(policy_data)
        
        # Generate embeddings
        contents = [chunk.content for chunk in new_chunks]
        embeddings = self.model.encode(contents)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(new_chunks, embeddings):
            chunk.embedding = embedding
        
        # Add to chunks list
        self.chunks.extend(new_chunks)
        
        # Rebuild FAISS index
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the FAISS index with all chunks"""
        if not self.chunks:
            return
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        embeddings = np.array([chunk.embedding for chunk in self.chunks])
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant policy content"""
        if not self.chunks or self.index is None:
            return []
        
        self._load_model()
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'content': chunk.content,
                    'url': chunk.url,
                    'section': chunk.section,
                    'score': float(score),
                    'chunk_id': chunk.chunk_id
                })
        
        return results


class PolicyService:
    """Main policy service with caching and RAG capabilities"""
    
    def __init__(self):
        self.cache = PolicyCache()  # Keep file cache as fallback
        self.redis_cache = policy_cache  # Use Redis cache as primary
        self.scraper = PolicyScraper()
        self.rag = PolicyRAG()
        self.policy_urls = {
            'cancellation': config.policies.cancellation_policy_url,
            'pet_travel': config.policies.pet_travel_policy_url
        }
        self._initialized = False
    
    async def initialize(self):
        """Initialize the policy service by loading all policies"""
        if self._initialized:
            return
        
        print("Initializing policy service...")
        
        for policy_type, url in self.policy_urls.items():
            try:
                print(f"Loading {policy_type} policy from {url}")
                policy_data = await self._fetch_policy_data(url)
                self.rag.add_policy_data(policy_data)
                print(f"✓ Loaded {policy_type} policy")
            except Exception as e:
                print(f"⚠️  Failed to load {policy_type} policy: {e}")
        
        self._initialized = True
        print("Policy service initialized")
    
    async def _fetch_policy_data(self, url: str) -> Dict[str, Any]:
        """Fetch policy data with Redis caching and file cache fallback"""
        # Check Redis cache first
        policy_type = self._get_policy_type_from_url(url)
        cached_policy = await self.redis_cache.get_policy(policy_type, url=url)
        
        if cached_policy:
            return {
                'content': cached_policy.content,
                'url': url,
                'last_updated': cached_policy.last_updated,
                'policy_type': cached_policy.policy_type,
                'applicable_conditions': cached_policy.applicable_conditions
            }
        
        # Check file cache as fallback
        cached_data = self.cache.get(url)
        if cached_data:
            return cached_data
        
        # Scrape fresh data
        try:
            policy_data = await self.scraper.scrape_policy_page(url)
        except Exception as e:
            print(f"Scraping failed for {url}: {e}")
            # Use fallback policy data
            policy_data = self._get_fallback_policy_data(url)
        
        # Convert sections to content
        content_parts = []
        sections = policy_data.get('sections', [])
        
        if sections:
            for section in sections:
                if section.get('title'):
                    content_parts.append(f"**{section['title']}**")
                if section.get('content'):
                    content_parts.append(section['content'])
                content_parts.append("")  # Add spacing
        
        combined_content = '\n'.join(content_parts).strip()
        
        # If no sections or content, use fallback
        if not combined_content:
            if 'title' in policy_data:
                combined_content = f"Policy information from {policy_data['title']}"
            else:
                # Use fallback content based on URL
                combined_content = self._get_fallback_content_by_url(url)
        
        # Cache in both Redis and file cache
        policy_info = PolicyInfo(
            policy_type=self._get_policy_type_from_url(url),
            content=combined_content or "Policy content not available",
            last_updated=datetime.now(),
            applicable_conditions=policy_data.get('applicable_conditions', [])
        )
        
        # Update policy_data with processed content
        policy_data['content'] = combined_content
        
        # Cache in Redis
        await self.redis_cache.set_policy(policy_info.policy_type, policy_info, url=url)
        
        # Cache in file system as fallback
        self.cache.set(url, policy_data['title'], policy_data)
        
        return policy_data
    
    def _get_policy_type_from_url(self, url: str) -> str:
        """Extract policy type from URL"""
        if 'pet' in url.lower():
            return 'pet_travel'
        elif 'cancel' in url.lower() or 'fare' in url.lower():
            return 'cancellation'
        else:
            return 'general'
    
    async def get_general_cancellation_policy(self) -> PolicyInfo:
        """Get general cancellation policy without specific flight context"""
        return await self.get_cancellation_policy(flight_details=None)
    
    async def get_cancellation_policy(self, flight_details: Optional[Dict] = None) -> PolicyInfo:
        """Get cancellation policy information based on flight details"""
        await self.initialize()
        
        # Build context-aware search query
        query_parts = ["cancellation policy"]
        
        if flight_details:
            # Add fare type specific terms
            fare_type = flight_details.get('fare_type', '').lower()
            if fare_type:
                query_parts.append(fare_type)
            
            # Add booking class
            booking_class = flight_details.get('booking_class', '').lower()
            if booking_class:
                query_parts.append(booking_class)
            
            # Add time-based context
            departure_time = flight_details.get('departure_time')
            if departure_time:
                # Check if departure is within 24 hours
                if isinstance(departure_time, str):
                    try:
                        departure_time = datetime.fromisoformat(departure_time)
                    except:
                        departure_time = None
                
                if departure_time and (departure_time - datetime.now()).total_seconds() < 86400:
                    query_parts.append("24 hour")
            
            # Add route information for international flights
            source = flight_details.get('source_airport', '')
            destination = flight_details.get('destination_airport', '')
            if source and destination:
                # Simple international flight detection
                if len(source) == 3 and len(destination) == 3:
                    query_parts.append("international" if source[:1] != destination[:1] else "domestic")
        
        query = " ".join(query_parts)
        search_results = self.rag.search(query, top_k=5)
        
        # Debug: Check if we have any results
        print(f"RAG search for '{query}' returned {len(search_results)} results")
        if not search_results:
            print(f"No RAG results found. Total chunks in RAG: {len(self.rag.chunks)}")
            # Return fallback policy content
            return self._get_fallback_cancellation_policy()
        
        # Filter and rank results based on flight details
        filtered_results = self._filter_cancellation_results(search_results, flight_details)
        
        # Combine relevant content
        policy_content = []
        applicable_conditions = []
        
        for result in filtered_results:
            policy_content.append(f"**{result['section']}**\n{result['content']}")
            
            # Extract applicable conditions from content
            content_lower = result['content'].lower()
            if 'blue basic' in content_lower:
                applicable_conditions.append('Blue Basic fare restrictions')
            if 'blue plus' in content_lower:
                applicable_conditions.append('Blue Plus fare benefits')
            if 'blue extra' in content_lower:
                applicable_conditions.append('Blue Extra fare benefits')
            if '24 hour' in content_lower:
                applicable_conditions.append('24-hour cancellation window')
            if 'refund' in content_lower:
                applicable_conditions.append('Refund processing')
        
        combined_content = "\n\n".join(policy_content)
        
        # If still no content, use fallback
        if not combined_content.strip():
            return self._get_fallback_cancellation_policy()
        
        return PolicyInfo(
            policy_type="cancellation",
            content=combined_content,
            last_updated=datetime.now(),
            applicable_conditions=list(set(applicable_conditions)) or ["fare_type", "booking_class", "departure_time"]
        )
    
    def _filter_cancellation_results(self, results: List[Dict], flight_details: Optional[Dict]) -> List[Dict]:
        """Filter cancellation policy results based on flight details"""
        if not flight_details:
            return results
        
        filtered = []
        fare_type = flight_details.get('fare_type', '').lower()
        
        # Prioritize results based on fare type
        for result in results:
            content_lower = result['content'].lower()
            score_boost = 0
            
            # Boost score for matching fare type
            if fare_type:
                if fare_type in content_lower:
                    score_boost += 0.3
                elif 'basic' in fare_type and 'basic' in content_lower:
                    score_boost += 0.2
                elif 'plus' in fare_type and 'plus' in content_lower:
                    score_boost += 0.2
                elif 'extra' in fare_type and 'extra' in content_lower:
                    score_boost += 0.2
            
            # Boost for relevant keywords
            if 'cancellation' in content_lower:
                score_boost += 0.1
            if 'refund' in content_lower:
                score_boost += 0.1
            if 'fee' in content_lower:
                score_boost += 0.1
            
            result['adjusted_score'] = result['score'] + score_boost
            filtered.append(result)
        
        # Sort by adjusted score
        filtered.sort(key=lambda x: x['adjusted_score'], reverse=True)
        return filtered[:3]  # Return top 3 most relevant
    
    async def get_pet_travel_policy(self, pet_details: Optional[Dict] = None) -> PolicyInfo:
        """Get pet travel policy information based on pet details"""
        await self.initialize()
        
        # Build context-aware search query for pets
        query_parts = ["pet travel policy"]
        
        if pet_details:
            # Add pet type
            pet_type = pet_details.get('pet_type', '').lower()
            if pet_type:
                query_parts.append(pet_type)
            
            # Add pet size
            pet_size = pet_details.get('pet_size', '').lower()
            if pet_size:
                query_parts.append(pet_size)
            
            # Add destination context
            destination = pet_details.get('destination', '')
            if destination:
                # Check for international travel
                if len(destination) == 3:  # Airport code
                    query_parts.append("international" if destination[0] not in ['A', 'K', 'N', 'P'] else "domestic")
            
            # Add service animal context
            is_service_animal = pet_details.get('is_service_animal', False)
            if is_service_animal:
                query_parts.append("service animal")
        
        query = " ".join(query_parts)
        search_results = self.rag.search(query, top_k=5)
        
        # Debug: Check if we have any results
        print(f"RAG search for pet travel '{query}' returned {len(search_results)} results")
        if not search_results:
            print(f"No pet travel RAG results found. Total chunks in RAG: {len(self.rag.chunks)}")
            # Return fallback policy content
            return self._get_fallback_pet_policy()
        
        # Filter results based on pet details
        filtered_results = self._filter_pet_travel_results(search_results, pet_details)
        
        # Combine relevant content
        policy_content = []
        applicable_conditions = []
        
        for result in filtered_results:
            policy_content.append(f"**{result['section']}**\n{result['content']}")
            
            # Extract applicable conditions
            content_lower = result['content'].lower()
            if 'cabin' in content_lower:
                applicable_conditions.append('In-cabin travel requirements')
            if 'carrier' in content_lower:
                applicable_conditions.append('Pet carrier specifications')
            if 'health' in content_lower or 'certificate' in content_lower:
                applicable_conditions.append('Health documentation required')
            if 'fee' in content_lower or 'cost' in content_lower:
                applicable_conditions.append('Pet travel fees')
            if 'service animal' in content_lower:
                applicable_conditions.append('Service animal policies')
            if 'international' in content_lower:
                applicable_conditions.append('International travel requirements')
        
        combined_content = "\n\n".join(policy_content)
        
        # If still no content, use fallback
        if not combined_content.strip():
            return self._get_fallback_pet_policy()
        
        return PolicyInfo(
            policy_type="pet_travel",
            content=combined_content,
            last_updated=datetime.now(),
            applicable_conditions=list(set(applicable_conditions)) or ["pet_type", "pet_size", "destination"]
        )
    
    def _filter_pet_travel_results(self, results: List[Dict], pet_details: Optional[Dict]) -> List[Dict]:
        """Filter pet travel policy results based on pet details"""
        if not pet_details:
            return results
        
        filtered = []
        pet_type = pet_details.get('pet_type', '').lower()
        is_service_animal = pet_details.get('is_service_animal', False)
        
        for result in results:
            content_lower = result['content'].lower()
            score_boost = 0
            
            # Boost for matching pet type
            if pet_type:
                if pet_type in content_lower:
                    score_boost += 0.3
                elif 'dog' in pet_type and 'dog' in content_lower:
                    score_boost += 0.2
                elif 'cat' in pet_type and 'cat' in content_lower:
                    score_boost += 0.2
            
            # Boost for service animals
            if is_service_animal and 'service' in content_lower:
                score_boost += 0.4
            
            # Boost for relevant keywords
            if 'pet' in content_lower:
                score_boost += 0.1
            if 'travel' in content_lower:
                score_boost += 0.1
            if 'cabin' in content_lower:
                score_boost += 0.1
            
            result['adjusted_score'] = result['score'] + score_boost
            filtered.append(result)
        
        # Sort by adjusted score
        filtered.sort(key=lambda x: x['adjusted_score'], reverse=True)
        return filtered[:3]
    
    async def search_policy_content(self, query: str, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search across all policy content with optional filters"""
        await self.initialize()
        
        # Get initial search results
        results = self.rag.search(query, top_k=20)
        
        # Apply filters if provided
        if filters:
            results = self._apply_search_filters(results, filters)
        
        return results[:10]  # Return top 10 after filtering
    
    def _apply_search_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to search results"""
        filtered = []
        
        for result in results:
            content_lower = result['content'].lower()
            section_lower = result['section'].lower()
            
            # Apply policy type filter
            policy_type = filters.get('policy_type', '').lower()
            if policy_type:
                if policy_type == 'cancellation' and 'cancel' not in content_lower and 'refund' not in content_lower:
                    continue
                elif policy_type == 'pet' and 'pet' not in content_lower and 'animal' not in content_lower:
                    continue
            
            # Apply fare type filter
            fare_type = filters.get('fare_type', '').lower()
            if fare_type and fare_type not in content_lower:
                continue
            
            # Apply minimum score filter
            min_score = filters.get('min_score', 0.0)
            if result['score'] < min_score:
                continue
            
            # Apply keyword filters
            required_keywords = filters.get('required_keywords', [])
            if required_keywords:
                if not all(keyword.lower() in content_lower for keyword in required_keywords):
                    continue
            
            # Apply exclusion filters
            excluded_keywords = filters.get('excluded_keywords', [])
            if excluded_keywords:
                if any(keyword.lower() in content_lower for keyword in excluded_keywords):
                    continue
            
            filtered.append(result)
        
        return filtered
    
    async def get_policy_by_condition(self, condition: str, context: Optional[Dict] = None) -> PolicyInfo:
        """Get policy information for specific conditions"""
        await self.initialize()
        
        # Map common conditions to search queries
        condition_queries = {
            'cancellation_fees': 'cancellation fees charges cost',
            'refund_processing': 'refund processing time credit card',
            '24_hour_rule': '24 hour cancellation free no penalty',
            'pet_cabin_travel': 'pet cabin travel in-cabin requirements',
            'pet_documentation': 'pet health certificate documentation required',
            'service_animal': 'service animal emotional support documentation',
            'international_pet': 'international pet travel requirements quarantine',
            'fare_differences': 'fare difference change fee upgrade downgrade',
            'weather_cancellation': 'weather delay cancellation compensation',
            'medical_emergency': 'medical emergency cancellation documentation'
        }
        
        query = condition_queries.get(condition, condition)
        
        # Add context to query if provided
        if context:
            context_terms = []
            for key, value in context.items():
                if isinstance(value, str) and value:
                    context_terms.append(value)
            if context_terms:
                query += " " + " ".join(context_terms)
        
        # Search for relevant content
        search_results = self.rag.search(query, top_k=5)
        
        # Combine results
        policy_content = []
        for result in search_results:
            policy_content.append(f"**{result['section']}**\n{result['content']}")
        
        combined_content = "\n\n".join(policy_content)
        
        return PolicyInfo(
            policy_type=condition,
            content=combined_content or f"Policy information for {condition} not available",
            last_updated=datetime.now(),
            applicable_conditions=[condition]
        )
    
    async def get_fare_specific_policy(self, fare_type: str, policy_category: str = "cancellation") -> PolicyInfo:
        """Get policy information specific to fare type"""
        await self.initialize()
        
        # Build fare-specific query
        query = f"{fare_type} {policy_category} policy"
        
        # Add fare-specific terms
        fare_terms = {
            'blue_basic': 'non-refundable basic economy',
            'blue': 'standard economy refundable',
            'blue_plus': 'premium economy extra legroom',
            'blue_extra': 'premium flexible changeable',
            'mint': 'business class premium'
        }
        
        fare_key = fare_type.lower().replace(' ', '_')
        if fare_key in fare_terms:
            query += " " + fare_terms[fare_key]
        
        search_results = self.rag.search(query, top_k=3)
        
        # Filter for fare-specific content
        filtered_results = []
        for result in search_results:
            content_lower = result['content'].lower()
            if fare_type.lower() in content_lower or any(term in content_lower for term in fare_terms.get(fare_key, '').split()):
                filtered_results.append(result)
        
        # Combine content
        policy_content = []
        for result in filtered_results:
            policy_content.append(f"**{result['section']}**\n{result['content']}")
        
        combined_content = "\n\n".join(policy_content)
        
        return PolicyInfo(
            policy_type=f"{fare_type}_{policy_category}",
            content=combined_content or f"{fare_type} {policy_category} policy information not available",
            last_updated=datetime.now(),
            applicable_conditions=[fare_type, policy_category]
        )
    
    async def refresh_policy_cache(self):
        """Refresh all cached policy data"""
        print("Refreshing policy cache...")
        
        for policy_type, url in self.policy_urls.items():
            try:
                print(f"Refreshing {policy_type} policy...")
                policy_data = await self.scraper.scrape_policy_page(url)
                self.cache.set(url, policy_data['title'], policy_data)
                print(f"✓ Refreshed {policy_type} policy")
            except Exception as e:
                print(f"⚠️  Failed to refresh {policy_type} policy: {e}")
        
        # Rebuild RAG system
        self.rag = PolicyRAG()
        self._initialized = False
        await self.initialize()
        
        print("Policy cache refreshed")
    
    def _get_fallback_cancellation_policy(self) -> PolicyInfo:
        """Get fallback cancellation policy when RAG fails"""
        fallback_content = """
**JetBlue Cancellation Policy**

**24-Hour Cancellation Rule**
You can cancel your booking within 24 hours of purchase for a full refund, provided the booking was made at least 7 days before departure.

**Fare-Specific Cancellation Rules**

**Blue Basic Fares**
• Non-refundable
• Can cancel for JetBlue credit minus cancellation fees
• Changes not permitted

**Blue Fares**
• Refundable with cancellation fees
• Changes permitted with fare difference and fees

**Blue Plus & Blue Extra Fares**
• More flexible cancellation and change policies
• Reduced or waived fees depending on fare type

**Same-Day Changes**
Available for a fee, subject to availability.

**Weather and Operational Delays**
If JetBlue cancels or significantly delays your flight, you're entitled to a full refund or rebooking at no additional charge.

**How to Cancel**
• Online at jetblue.com
• Through the JetBlue mobile app
• By calling customer service

For the most current cancellation fees and specific fare rules, please visit jetblue.com or contact customer service at 1-800-JETBLUE.
        """.strip()
        
        return PolicyInfo(
            policy_type="cancellation",
            content=fallback_content,
            last_updated=datetime.now(),
            applicable_conditions=["24_hour_rule", "fare_type", "weather_delays"]
        )
    
    def _get_fallback_pet_policy(self) -> PolicyInfo:
        """Get fallback pet travel policy when RAG fails"""
        fallback_content = """
**JetBlue Pet Travel Policy**

**In-Cabin Pet Travel**
• Small cats and dogs only
• Must be in an approved soft-sided carrier
• Carrier must fit completely under the seat in front of you
• Pet fee: $125 each way

**Pet Carrier Requirements**
• Maximum dimensions: 17" L x 12.5" W x 8" H
• Soft-sided carriers only
• Must be leak-proof and well-ventilated
• Pet must be able to stand and turn around comfortably

**Health Requirements**
• Health certificate may be required for some destinations
• Pets must be at least 8 weeks old
• Current vaccinations required

**Service Animals**
• Travel free of charge
• Proper documentation required
• Must be trained to perform specific tasks

**Restrictions**
• No pets in cargo hold
• Limited to in-cabin travel only
• One pet per carrier, one carrier per customer
• Advance reservations required

For complete pet travel requirements and to make reservations, visit jetblue.com/pets or call 1-800-JETBLUE.
        """.strip()
        
        return PolicyInfo(
            policy_type="pet_travel",
            content=fallback_content,
            last_updated=datetime.now(),
            applicable_conditions=["in_cabin_only", "carrier_requirements", "health_certificate"]
        )

    def _get_fallback_policy_data(self, url: str) -> Dict[str, Any]:
        """Get comprehensive static policy data when scraping fails"""
        policy_type = self._get_policy_type_from_url(url)
        
        if policy_type == 'cancellation':
            return {
                'url': url,
                'title': 'JetBlue Cancellation and Change Policy',
                'sections': [
                    {
                        'title': '24-Hour Risk-Free Cancellation',
                        'content': 'Cancel within 24 hours of booking for a full refund to your original form of payment, as long as your departure date is at least 7 days away. This applies to all fare types including Blue Basic.'
                    },
                    {
                        'title': 'Blue Basic Fare Cancellation',
                        'content': 'Blue Basic fares are non-refundable after the 24-hour window. You can cancel for a JetBlue Travel Credit minus a $100 cancellation fee per person. Changes are not permitted on Blue Basic fares.'
                    },
                    {
                        'title': 'Blue Fare Cancellation',
                        'content': 'Blue fares can be cancelled for a full refund minus a $100 cancellation fee per person when cancelled more than 60 days before departure. Within 60 days, you receive a JetBlue Travel Credit minus the fee.'
                    },
                    {
                        'title': 'Blue Plus Fare Cancellation',
                        'content': 'Blue Plus fares can be cancelled for a full refund minus a $75 cancellation fee per person when cancelled more than 60 days before departure. Within 60 days, you receive a JetBlue Travel Credit minus the fee.'
                    },
                    {
                        'title': 'Blue Extra and Blue Flex Cancellation',
                        'content': 'Blue Extra fares have no cancellation fees when cancelled more than 60 days before departure. Blue Flex fares have no cancellation or change fees at any time.'
                    },
                    {
                        'title': 'Same-Day Flight Changes',
                        'content': 'Same-day flight changes are available for $75 per person, subject to availability. Must be on the same route and date.'
                    },
                    {
                        'title': 'Weather and Operational Cancellations',
                        'content': 'If JetBlue cancels your flight due to weather, mechanical issues, or operational reasons, you are entitled to a full refund or rebooking at no additional charge.'
                    },
                    {
                        'title': 'How to Cancel or Change',
                        'content': 'Cancel or change your booking online at jetblue.com, through the JetBlue mobile app, or by calling 1-800-JETBLUE. Online changes may have lower fees than phone bookings.'
                    }
                ]
            }
        elif policy_type == 'pet_travel':
            return {
                'url': url,
                'title': 'JetBlue Pet Travel Policy',
                'sections': [
                    {
                        'title': 'In-Cabin Pet Travel Overview',
                        'content': 'JetBlue welcomes small cats and dogs in the cabin on most flights. Pets must remain in an approved carrier that fits under the seat in front of you for the entire flight.'
                    },
                    {
                        'title': 'Pet Travel Fees',
                        'content': 'The pet travel fee is $125 each way for in-cabin pets. This fee is non-refundable and must be paid at the time of booking or check-in.'
                    },
                    {
                        'title': 'Pet Carrier Requirements',
                        'content': 'Soft-sided carriers only, maximum dimensions 17" L x 12.5" W x 8" H. Hard-sided carriers are not permitted. The carrier must be leak-proof, well-ventilated, and allow your pet to stand and turn around comfortably.'
                    },
                    {
                        'title': 'Pet Age and Health Requirements',
                        'content': 'Pets must be at least 8 weeks old and weaned. A health certificate from a veterinarian may be required for some destinations. Current vaccinations are required.'
                    },
                    {
                        'title': 'Booking Pet Travel',
                        'content': 'Pet reservations are required and must be made in advance. Limited space is available on each flight. Call 1-800-JETBLUE to add a pet to your reservation.'
                    },
                    {
                        'title': 'Service Animals',
                        'content': 'Trained service animals travel free of charge and are not considered pets. Proper documentation is required including DOT Service Animal Air Transportation Form. Emotional support animals are no longer accepted.'
                    },
                    {
                        'title': 'Pet Travel Restrictions',
                        'content': 'Pets are not allowed in cargo. Only in-cabin travel is permitted. Some aircraft types and routes may have restrictions. Pets cannot travel on flights longer than 6 hours.'
                    },
                    {
                        'title': 'International Pet Travel',
                        'content': 'Additional documentation and health certificates are required for international destinations. Contact your destination country\'s embassy or consulate for specific requirements.'
                    }
                ]
            }
        else:
            return {
                'url': url,
                'title': 'JetBlue Policy Information',
                'sections': [
                    {
                        'title': 'General Policies',
                        'content': 'For the most current and detailed policy information, please visit jetblue.com, use the JetBlue mobile app, or contact customer service at 1-800-JETBLUE.'
                    }
                ]
            }
    
    def _get_fallback_content_by_url(self, url: str) -> str:
        """Get fallback content based on URL when all else fails"""
        if 'cancel' in url.lower() or 'fare' in url.lower():
            return self._get_fallback_cancellation_policy().content
        elif 'pet' in url.lower():
            return self._get_fallback_pet_policy().content
        else:
            return "Policy information is currently unavailable. Please visit jetblue.com or contact customer service for assistance."

    async def close(self):
        """Close the policy service"""
        await self.scraper.close()


# Global policy service instance
policy_service = PolicyService()