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
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Policy Page"
            
            # Extract main content
            content_selectors = [
                'main',
                '.main-content',
                '.content',
                '#content',
                '.policy-content',
                'article',
                '.article-content'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body')
            
            # Extract sections
            sections = self._extract_sections(main_content)
            
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
        
        # Find all headings and their content
        headings = content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
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
        
        for section in policy_data['sections']:
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
        policy_data = await self.scraper.scrape_policy_page(url)
        
        # Cache in both Redis and file cache
        policy_info = PolicyInfo(
            policy_type=self._get_policy_type_from_url(url),
            content=policy_data.get('content', ''),
            last_updated=datetime.now(),
            applicable_conditions=policy_data.get('applicable_conditions', [])
        )
        
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
        
        return PolicyInfo(
            policy_type="cancellation",
            content=combined_content or "Cancellation policy information not available",
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
        
        return PolicyInfo(
            policy_type="pet_travel",
            content=combined_content or "Pet travel policy information not available",
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
    
    async def close(self):
        """Close the policy service"""
        await self.scraper.close()


# Global policy service instance
policy_service = PolicyService()