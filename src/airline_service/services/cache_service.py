"""
Caching service for airline customer service system.

This service provides Redis-based caching for policy information, API responses,
and other frequently accessed data with configurable TTL and invalidation strategies.
"""

import json
import asyncio
import structlog
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib

import aioredis
from aioredis import Redis

from ..config import config
from ..types import PolicyInfo, BookingDetails, FlightInfo, SeatAvailability


logger = structlog.get_logger(__name__)


class CacheKeyType(str, Enum):
    """Types of cache keys"""
    POLICY = "policy"
    API_RESPONSE = "api_response"
    BOOKING_DETAILS = "booking"
    FLIGHT_INFO = "flight"
    SEAT_AVAILABILITY = "seats"
    CUSTOMER_PROFILE = "customer"
    CLASSIFICATION_RESULT = "classification"


@dataclass
class CacheConfig:
    """Cache configuration for different data types"""
    ttl_seconds: int
    invalidation_strategy: str = "ttl"  # ttl, manual, event-based
    compress: bool = False
    serialize_json: bool = True


class CacheService:
    """
    Redis-based caching service with support for different data types,
    TTL management, and cache invalidation strategies.
    """
    
    def __init__(self):
        """Initialize the cache service."""
        self.redis: Optional[Redis] = None
        self.connection_pool: Optional[aioredis.ConnectionPool] = None
        self.logger = structlog.get_logger("cache")
        
        # Cache configurations for different data types
        self.cache_configs = {
            CacheKeyType.POLICY: CacheConfig(
                ttl_seconds=config.policies.cache_ttl,  # 1 hour default
                invalidation_strategy="manual"
            ),
            CacheKeyType.API_RESPONSE: CacheConfig(
                ttl_seconds=300,  # 5 minutes
                invalidation_strategy="ttl"
            ),
            CacheKeyType.BOOKING_DETAILS: CacheConfig(
                ttl_seconds=600,  # 10 minutes
                invalidation_strategy="event-based"
            ),
            CacheKeyType.FLIGHT_INFO: CacheConfig(
                ttl_seconds=180,  # 3 minutes
                invalidation_strategy="ttl"
            ),
            CacheKeyType.SEAT_AVAILABILITY: CacheConfig(
                ttl_seconds=60,  # 1 minute
                invalidation_strategy="ttl"
            ),
            CacheKeyType.CUSTOMER_PROFILE: CacheConfig(
                ttl_seconds=1800,  # 30 minutes
                invalidation_strategy="event-based"
            ),
            CacheKeyType.CLASSIFICATION_RESULT: CacheConfig(
                ttl_seconds=3600,  # 1 hour
                invalidation_strategy="ttl"
            )
        }
        
        # Event listeners for cache invalidation
        self.invalidation_listeners: Dict[str, List[Callable]] = {}
    
    async def initialize(self) -> None:
        """Initialize Redis connection and connection pool."""
        try:
            # Create connection pool
            self.connection_pool = aioredis.ConnectionPool.from_url(
                f"redis://{config.redis.host}:{config.redis.port}/{config.redis.db}",
                password=config.redis.password,
                max_connections=config.redis.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Create Redis client
            self.redis = aioredis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis.ping()
            
            self.logger.info(
                "Cache service initialized",
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
                max_connections=config.redis.max_connections
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize cache service", error=str(e))
            # Fall back to no caching
            self.redis = None
            self.connection_pool = None
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self.redis:
            await self.redis.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()
        
        self.logger.info("Cache service closed")
    
    def _generate_cache_key(
        self, 
        key_type: CacheKeyType, 
        identifier: str, 
        **kwargs
    ) -> str:
        """
        Generate a cache key with consistent format.
        
        Args:
            key_type: Type of cache key
            identifier: Primary identifier
            **kwargs: Additional parameters for key generation
            
        Returns:
            Generated cache key
        """
        # Create base key
        base_key = f"{key_type.value}:{identifier}"
        
        # Add additional parameters if provided
        if kwargs:
            # Sort kwargs for consistent key generation
            sorted_params = sorted(kwargs.items())
            param_str = ":".join(f"{k}={v}" for k, v in sorted_params)
            base_key = f"{base_key}:{param_str}"
        
        # Hash long keys to avoid Redis key length limits
        if len(base_key) > 250:
            key_hash = hashlib.md5(base_key.encode()).hexdigest()
            base_key = f"{key_type.value}:hash:{key_hash}"
        
        return base_key
    
    async def get(
        self, 
        key_type: CacheKeyType, 
        identifier: str, 
        **kwargs
    ) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key_type: Type of cache key
            identifier: Primary identifier
            **kwargs: Additional parameters for key generation
            
        Returns:
            Cached value or None if not found
        """
        if not self.redis:
            return None
        
        try:
            cache_key = self._generate_cache_key(key_type, identifier, **kwargs)
            cached_data = await self.redis.get(cache_key)
            
            if cached_data is None:
                self.logger.debug("Cache miss", key=cache_key, key_type=key_type.value)
                return None
            
            # Deserialize data
            config_obj = self.cache_configs.get(key_type)
            if config_obj and config_obj.serialize_json:
                try:
                    data = json.loads(cached_data)
                    self.logger.debug("Cache hit", key=cache_key, key_type=key_type.value)
                    return data
                except json.JSONDecodeError:
                    self.logger.warning("Failed to deserialize cached data", key=cache_key)
                    return None
            else:
                return cached_data.decode('utf-8')
                
        except Exception as e:
            self.logger.error("Cache get error", error=str(e), key_type=key_type.value)
            return None
    
    async def set(
        self, 
        key_type: CacheKeyType, 
        identifier: str, 
        value: Any, 
        ttl_override: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key_type: Type of cache key
            identifier: Primary identifier
            value: Value to cache
            ttl_override: Override default TTL
            **kwargs: Additional parameters for key generation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            cache_key = self._generate_cache_key(key_type, identifier, **kwargs)
            config_obj = self.cache_configs.get(key_type)
            
            # Serialize data
            if config_obj and config_obj.serialize_json:
                cached_data = json.dumps(value, default=str)
            else:
                cached_data = str(value)
            
            # Determine TTL
            ttl = ttl_override or (config_obj.ttl_seconds if config_obj else 3600)
            
            # Set in Redis
            await self.redis.setex(cache_key, ttl, cached_data)
            
            self.logger.debug(
                "Cache set", 
                key=cache_key, 
                key_type=key_type.value, 
                ttl=ttl
            )
            return True
            
        except Exception as e:
            self.logger.error("Cache set error", error=str(e), key_type=key_type.value)
            return False
    
    async def delete(
        self, 
        key_type: CacheKeyType, 
        identifier: str, 
        **kwargs
    ) -> bool:
        """
        Delete value from cache.
        
        Args:
            key_type: Type of cache key
            identifier: Primary identifier
            **kwargs: Additional parameters for key generation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            cache_key = self._generate_cache_key(key_type, identifier, **kwargs)
            result = await self.redis.delete(cache_key)
            
            self.logger.debug(
                "Cache delete", 
                key=cache_key, 
                key_type=key_type.value, 
                deleted=bool(result)
            )
            return bool(result)
            
        except Exception as e:
            self.logger.error("Cache delete error", error=str(e), key_type=key_type.value)
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "policy:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.redis:
            return 0
        
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                deleted = await self.redis.delete(*keys)
                self.logger.info("Cache pattern delete", pattern=pattern, deleted=deleted)
                return deleted
            return 0
            
        except Exception as e:
            self.logger.error("Cache pattern delete error", error=str(e), pattern=pattern)
            return 0
    
    async def exists(
        self, 
        key_type: CacheKeyType, 
        identifier: str, 
        **kwargs
    ) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key_type: Type of cache key
            identifier: Primary identifier
            **kwargs: Additional parameters for key generation
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            cache_key = self._generate_cache_key(key_type, identifier, **kwargs)
            result = await self.redis.exists(cache_key)
            return bool(result)
            
        except Exception as e:
            self.logger.error("Cache exists error", error=str(e), key_type=key_type.value)
            return False
    
    async def get_ttl(
        self, 
        key_type: CacheKeyType, 
        identifier: str, 
        **kwargs
    ) -> Optional[int]:
        """
        Get TTL for a cached key.
        
        Args:
            key_type: Type of cache key
            identifier: Primary identifier
            **kwargs: Additional parameters for key generation
            
        Returns:
            TTL in seconds or None if key doesn't exist
        """
        if not self.redis:
            return None
        
        try:
            cache_key = self._generate_cache_key(key_type, identifier, **kwargs)
            ttl = await self.redis.ttl(cache_key)
            return ttl if ttl > 0 else None
            
        except Exception as e:
            self.logger.error("Cache TTL error", error=str(e), key_type=key_type.value)
            return None
    
    async def refresh_ttl(
        self, 
        key_type: CacheKeyType, 
        identifier: str, 
        ttl_override: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Refresh TTL for an existing key.
        
        Args:
            key_type: Type of cache key
            identifier: Primary identifier
            ttl_override: Override default TTL
            **kwargs: Additional parameters for key generation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            cache_key = self._generate_cache_key(key_type, identifier, **kwargs)
            config_obj = self.cache_configs.get(key_type)
            ttl = ttl_override or (config_obj.ttl_seconds if config_obj else 3600)
            
            result = await self.redis.expire(cache_key, ttl)
            return bool(result)
            
        except Exception as e:
            self.logger.error("Cache refresh TTL error", error=str(e), key_type=key_type.value)
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.redis:
            return {"status": "disabled"}
        
        try:
            info = await self.redis.info()
            
            stats = {
                "status": "connected",
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "expired_keys": info.get("expired_keys", 0)
            }
            
            # Calculate hit rate
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            if hits + misses > 0:
                stats["hit_rate"] = hits / (hits + misses)
            else:
                stats["hit_rate"] = 0.0
            
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get cache stats", error=str(e))
            return {"status": "error", "error": str(e)}
    
    def add_invalidation_listener(
        self, 
        event_type: str, 
        callback: Callable[[str, Any], None]
    ) -> None:
        """
        Add a listener for cache invalidation events.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        if event_type not in self.invalidation_listeners:
            self.invalidation_listeners[event_type] = []
        
        self.invalidation_listeners[event_type].append(callback)
        self.logger.debug("Added invalidation listener", event_type=event_type)
    
    async def trigger_invalidation(self, event_type: str, data: Any) -> None:
        """
        Trigger cache invalidation based on an event.
        
        Args:
            event_type: Type of event that occurred
            data: Event data
        """
        listeners = self.invalidation_listeners.get(event_type, [])
        
        for callback in listeners:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                self.logger.error(
                    "Invalidation listener error", 
                    error=str(e), 
                    event_type=event_type
                )
        
        self.logger.debug(
            "Triggered invalidation", 
            event_type=event_type, 
            listeners_count=len(listeners)
        )


class PolicyCacheManager:
    """
    Specialized cache manager for policy information with smart invalidation.
    """
    
    def __init__(self, cache_service: CacheService):
        """Initialize policy cache manager."""
        self.cache = cache_service
        self.logger = structlog.get_logger("policy_cache")
    
    async def get_policy(self, policy_type: str, **filters) -> Optional[PolicyInfo]:
        """
        Get policy from cache.
        
        Args:
            policy_type: Type of policy (cancellation, pet_travel, etc.)
            **filters: Additional filters for policy lookup
            
        Returns:
            Cached policy info or None
        """
        cached_data = await self.cache.get(
            CacheKeyType.POLICY, 
            policy_type, 
            **filters
        )
        
        if cached_data:
            try:
                return PolicyInfo(**cached_data)
            except Exception as e:
                self.logger.warning("Failed to deserialize policy", error=str(e))
                return None
        
        return None
    
    async def set_policy(
        self, 
        policy_type: str, 
        policy_info: PolicyInfo, 
        **filters
    ) -> bool:
        """
        Cache policy information.
        
        Args:
            policy_type: Type of policy
            policy_info: Policy information to cache
            **filters: Additional filters for policy lookup
            
        Returns:
            True if successful
        """
        # Convert PolicyInfo to dict for JSON serialization
        policy_data = {
            "policy_type": policy_info.policy_type,
            "content": policy_info.content,
            "last_updated": policy_info.last_updated.isoformat(),
            "applicable_conditions": policy_info.applicable_conditions
        }
        
        return await self.cache.set(
            CacheKeyType.POLICY, 
            policy_type, 
            policy_data, 
            **filters
        )
    
    async def invalidate_policy(self, policy_type: str) -> bool:
        """
        Invalidate all cached policies of a specific type.
        
        Args:
            policy_type: Type of policy to invalidate
            
        Returns:
            True if successful
        """
        pattern = f"policy:{policy_type}*"
        deleted = await self.cache.delete_pattern(pattern)
        
        self.logger.info(
            "Policy cache invalidated", 
            policy_type=policy_type, 
            deleted=deleted
        )
        
        return deleted > 0
    
    async def refresh_all_policies(self) -> int:
        """
        Clear all cached policies to force refresh.
        
        Returns:
            Number of policies cleared
        """
        deleted = await self.cache.delete_pattern("policy:*")
        
        self.logger.info("All policies cache cleared", deleted=deleted)
        
        return deleted


class APIResponseCacheManager:
    """
    Specialized cache manager for API responses with intelligent caching strategies.
    """
    
    def __init__(self, cache_service: CacheService):
        """Initialize API response cache manager."""
        self.cache = cache_service
        self.logger = structlog.get_logger("api_cache")
    
    async def get_api_response(
        self, 
        endpoint: str, 
        method: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached API response.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            params: Request parameters
            
        Returns:
            Cached response or None
        """
        # Create cache identifier from endpoint, method, and params
        identifier = f"{method}:{endpoint}"
        kwargs = {"params": json.dumps(params, sort_keys=True)} if params else {}
        
        return await self.cache.get(CacheKeyType.API_RESPONSE, identifier, **kwargs)
    
    async def set_api_response(
        self, 
        endpoint: str, 
        method: str, 
        response_data: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None,
        ttl_override: Optional[int] = None
    ) -> bool:
        """
        Cache API response.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            response_data: Response data to cache
            params: Request parameters
            ttl_override: Override default TTL
            
        Returns:
            True if successful
        """
        identifier = f"{method}:{endpoint}"
        kwargs = {"params": json.dumps(params, sort_keys=True)} if params else {}
        
        return await self.cache.set(
            CacheKeyType.API_RESPONSE, 
            identifier, 
            response_data,
            ttl_override=ttl_override,
            **kwargs
        )
    
    async def invalidate_endpoint(self, endpoint: str) -> int:
        """
        Invalidate all cached responses for an endpoint.
        
        Args:
            endpoint: API endpoint to invalidate
            
        Returns:
            Number of responses invalidated
        """
        # Clean endpoint for pattern matching
        clean_endpoint = endpoint.replace("/", "_").replace(":", "_")
        pattern = f"api_response:*:{clean_endpoint}*"
        
        deleted = await self.cache.delete_pattern(pattern)
        
        self.logger.info(
            "API endpoint cache invalidated", 
            endpoint=endpoint, 
            deleted=deleted
        )
        
        return deleted


# Global cache service instance
cache_service = CacheService()
policy_cache = PolicyCacheManager(cache_service)
api_cache = APIResponseCacheManager(cache_service)