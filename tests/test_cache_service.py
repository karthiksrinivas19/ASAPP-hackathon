"""
Tests for caching service
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.airline_service.services.cache_service import (
    CacheService, CacheKeyType, PolicyCacheManager, APIResponseCacheManager
)
from src.airline_service.types import PolicyInfo


class TestCacheService:
    """Test cache service functionality"""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client"""
        redis_mock = AsyncMock()
        redis_mock.ping.return_value = True
        redis_mock.get.return_value = None
        redis_mock.setex.return_value = True
        redis_mock.delete.return_value = 1
        redis_mock.exists.return_value = False
        redis_mock.ttl.return_value = 3600
        redis_mock.expire.return_value = True
        redis_mock.keys.return_value = []
        redis_mock.info.return_value = {
            "redis_version": "6.0.0",
            "used_memory_human": "1M",
            "connected_clients": 1,
            "total_commands_processed": 100,
            "keyspace_hits": 80,
            "keyspace_misses": 20,
            "expired_keys": 5
        }
        return redis_mock
    
    @pytest.fixture
    def cache_service(self, mock_redis):
        """Create cache service with mock Redis"""
        service = CacheService()
        service.redis = mock_redis
        return service
    
    def test_generate_cache_key(self, cache_service):
        """Test cache key generation"""
        # Basic key
        key = cache_service._generate_cache_key(CacheKeyType.POLICY, "cancellation")
        assert key == "policy:cancellation"
        
        # Key with parameters
        key = cache_service._generate_cache_key(
            CacheKeyType.API_RESPONSE, 
            "booking", 
            pnr="ABC123",
            date="2024-01-01"
        )
        assert "api_response:booking" in key
        assert "date=2024-01-01" in key
        assert "pnr=ABC123" in key
    
    def test_generate_cache_key_long(self, cache_service):
        """Test cache key generation for long keys"""
        long_identifier = "x" * 300
        key = cache_service._generate_cache_key(CacheKeyType.POLICY, long_identifier)
        
        # Should be hashed due to length
        assert key.startswith("policy:hash:")
        assert len(key) < 250
    
    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_service, mock_redis):
        """Test cache miss scenario"""
        mock_redis.get.return_value = None
        
        result = await cache_service.get(CacheKeyType.POLICY, "cancellation")
        
        assert result is None
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache_service, mock_redis):
        """Test cache hit scenario"""
        test_data = {"policy": "test content"}
        mock_redis.get.return_value = '{"policy": "test content"}'
        
        result = await cache_service.get(CacheKeyType.POLICY, "cancellation")
        
        assert result == test_data
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_cache(self, cache_service, mock_redis):
        """Test setting cache value"""
        test_data = {"policy": "test content"}
        
        result = await cache_service.set(CacheKeyType.POLICY, "cancellation", test_data)
        
        assert result is True
        mock_redis.setex.assert_called_once()
        
        # Check that data was JSON serialized
        call_args = mock_redis.setex.call_args
        assert '"policy": "test content"' in call_args[0][2]
    
    @pytest.mark.asyncio
    async def test_set_cache_with_ttl_override(self, cache_service, mock_redis):
        """Test setting cache value with TTL override"""
        test_data = {"policy": "test content"}
        
        result = await cache_service.set(
            CacheKeyType.POLICY, 
            "cancellation", 
            test_data, 
            ttl_override=7200
        )
        
        assert result is True
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 7200  # TTL should be overridden
    
    @pytest.mark.asyncio
    async def test_delete_cache(self, cache_service, mock_redis):
        """Test deleting cache value"""
        mock_redis.delete.return_value = 1
        
        result = await cache_service.delete(CacheKeyType.POLICY, "cancellation")
        
        assert result is True
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_pattern(self, cache_service, mock_redis):
        """Test deleting cache values by pattern"""
        mock_redis.keys.return_value = ["policy:cancellation", "policy:pet_travel"]
        mock_redis.delete.return_value = 2
        
        result = await cache_service.delete_pattern("policy:*")
        
        assert result == 2
        mock_redis.keys.assert_called_once_with("policy:*")
        mock_redis.delete.assert_called_once_with("policy:cancellation", "policy:pet_travel")
    
    @pytest.mark.asyncio
    async def test_exists(self, cache_service, mock_redis):
        """Test checking if key exists"""
        mock_redis.exists.return_value = 1
        
        result = await cache_service.exists(CacheKeyType.POLICY, "cancellation")
        
        assert result is True
        mock_redis.exists.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_ttl(self, cache_service, mock_redis):
        """Test getting TTL for key"""
        mock_redis.ttl.return_value = 1800
        
        result = await cache_service.get_ttl(CacheKeyType.POLICY, "cancellation")
        
        assert result == 1800
        mock_redis.ttl.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_refresh_ttl(self, cache_service, mock_redis):
        """Test refreshing TTL for key"""
        mock_redis.expire.return_value = True
        
        result = await cache_service.refresh_ttl(CacheKeyType.POLICY, "cancellation")
        
        assert result is True
        mock_redis.expire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, cache_service, mock_redis):
        """Test getting cache statistics"""
        stats = await cache_service.get_cache_stats()
        
        assert stats["status"] == "connected"
        assert stats["redis_version"] == "6.0.0"
        assert stats["hit_rate"] == 0.8  # 80 hits / (80 hits + 20 misses)
        assert stats["keyspace_hits"] == 80
        assert stats["keyspace_misses"] == 20
    
    @pytest.mark.asyncio
    async def test_cache_service_no_redis(self):
        """Test cache service behavior when Redis is not available"""
        service = CacheService()
        service.redis = None
        
        # All operations should return None/False gracefully
        assert await service.get(CacheKeyType.POLICY, "test") is None
        assert await service.set(CacheKeyType.POLICY, "test", {"data": "test"}) is False
        assert await service.delete(CacheKeyType.POLICY, "test") is False
        assert await service.exists(CacheKeyType.POLICY, "test") is False
        
        stats = await service.get_cache_stats()
        assert stats["status"] == "disabled"
    
    def test_add_invalidation_listener(self, cache_service):
        """Test adding invalidation listener"""
        callback = MagicMock()
        
        cache_service.add_invalidation_listener("booking_updated", callback)
        
        assert "booking_updated" in cache_service.invalidation_listeners
        assert callback in cache_service.invalidation_listeners["booking_updated"]
    
    @pytest.mark.asyncio
    async def test_trigger_invalidation(self, cache_service):
        """Test triggering cache invalidation"""
        callback = AsyncMock()
        cache_service.add_invalidation_listener("booking_updated", callback)
        
        await cache_service.trigger_invalidation("booking_updated", {"pnr": "ABC123"})
        
        callback.assert_called_once_with("booking_updated", {"pnr": "ABC123"})


class TestPolicyCacheManager:
    """Test policy cache manager functionality"""
    
    @pytest.fixture
    def mock_cache_service(self):
        """Create mock cache service"""
        cache_mock = AsyncMock()
        return cache_mock
    
    @pytest.fixture
    def policy_cache_manager(self, mock_cache_service):
        """Create policy cache manager with mock cache service"""
        return PolicyCacheManager(mock_cache_service)
    
    @pytest.fixture
    def sample_policy_info(self):
        """Sample policy info for testing"""
        return PolicyInfo(
            policy_type="cancellation",
            content="Cancellation policy content",
            last_updated=datetime(2024, 1, 1),
            applicable_conditions=["24 hours notice", "refundable fare"]
        )
    
    @pytest.mark.asyncio
    async def test_get_policy_hit(self, policy_cache_manager, mock_cache_service, sample_policy_info):
        """Test getting policy from cache (hit)"""
        mock_cache_service.get.return_value = {
            "policy_type": "cancellation",
            "content": "Cancellation policy content",
            "last_updated": "2024-01-01T00:00:00",
            "applicable_conditions": ["24 hours notice", "refundable fare"]
        }
        
        result = await policy_cache_manager.get_policy("cancellation")
        
        assert result is not None
        assert result.policy_type == "cancellation"
        assert result.content == "Cancellation policy content"
        mock_cache_service.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_policy_miss(self, policy_cache_manager, mock_cache_service):
        """Test getting policy from cache (miss)"""
        mock_cache_service.get.return_value = None
        
        result = await policy_cache_manager.get_policy("cancellation")
        
        assert result is None
        mock_cache_service.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_policy(self, policy_cache_manager, mock_cache_service, sample_policy_info):
        """Test setting policy in cache"""
        mock_cache_service.set.return_value = True
        
        result = await policy_cache_manager.set_policy("cancellation", sample_policy_info)
        
        assert result is True
        mock_cache_service.set.assert_called_once()
        
        # Check that PolicyInfo was converted to dict
        call_args = mock_cache_service.set.call_args
        policy_data = call_args[0][2]  # Third argument is the data
        assert policy_data["policy_type"] == "cancellation"
        assert policy_data["content"] == "Cancellation policy content"
    
    @pytest.mark.asyncio
    async def test_invalidate_policy(self, policy_cache_manager, mock_cache_service):
        """Test invalidating policy cache"""
        mock_cache_service.delete_pattern.return_value = 3
        
        result = await policy_cache_manager.invalidate_policy("cancellation")
        
        assert result is True
        mock_cache_service.delete_pattern.assert_called_once_with("policy:cancellation*")
    
    @pytest.mark.asyncio
    async def test_refresh_all_policies(self, policy_cache_manager, mock_cache_service):
        """Test refreshing all policies"""
        mock_cache_service.delete_pattern.return_value = 5
        
        result = await policy_cache_manager.refresh_all_policies()
        
        assert result == 5
        mock_cache_service.delete_pattern.assert_called_once_with("policy:*")


class TestAPIResponseCacheManager:
    """Test API response cache manager functionality"""
    
    @pytest.fixture
    def mock_cache_service(self):
        """Create mock cache service"""
        cache_mock = AsyncMock()
        return cache_mock
    
    @pytest.fixture
    def api_cache_manager(self, mock_cache_service):
        """Create API response cache manager with mock cache service"""
        return APIResponseCacheManager(mock_cache_service)
    
    @pytest.mark.asyncio
    async def test_get_api_response_hit(self, api_cache_manager, mock_cache_service):
        """Test getting API response from cache (hit)"""
        test_response = {"pnr": "ABC123", "status": "confirmed"}
        mock_cache_service.get.return_value = test_response
        
        result = await api_cache_manager.get_api_response(
            endpoint="/flight/booking",
            method="GET",
            params={"pnr": "ABC123"}
        )
        
        assert result == test_response
        mock_cache_service.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_api_response_miss(self, api_cache_manager, mock_cache_service):
        """Test getting API response from cache (miss)"""
        mock_cache_service.get.return_value = None
        
        result = await api_cache_manager.get_api_response(
            endpoint="/flight/booking",
            method="GET",
            params={"pnr": "ABC123"}
        )
        
        assert result is None
        mock_cache_service.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_api_response(self, api_cache_manager, mock_cache_service):
        """Test setting API response in cache"""
        test_response = {"pnr": "ABC123", "status": "confirmed"}
        mock_cache_service.set.return_value = True
        
        result = await api_cache_manager.set_api_response(
            endpoint="/flight/booking",
            method="GET",
            response_data=test_response,
            params={"pnr": "ABC123"}
        )
        
        assert result is True
        mock_cache_service.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalidate_endpoint(self, api_cache_manager, mock_cache_service):
        """Test invalidating API endpoint cache"""
        mock_cache_service.delete_pattern.return_value = 2
        
        result = await api_cache_manager.invalidate_endpoint("/flight/booking")
        
        assert result == 2
        mock_cache_service.delete_pattern.assert_called_once()
        
        # Check that pattern was properly formatted
        call_args = mock_cache_service.delete_pattern.call_args
        pattern = call_args[0][0]
        assert "flight_booking" in pattern  # Slashes should be replaced