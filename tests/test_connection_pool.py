"""
Tests for connection pool manager
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import ClientSession, TCPConnector, ClientTimeout

from src.airline_service.services.connection_pool import (
    ConnectionPoolManager, PooledHTTPClient
)


class TestConnectionPoolManager:
    """Test connection pool manager functionality"""
    
    @pytest.fixture
    def pool_manager(self):
        """Create connection pool manager"""
        return ConnectionPoolManager()
    
    @pytest.fixture
    def mock_session(self):
        """Create mock ClientSession"""
        session_mock = AsyncMock(spec=ClientSession)
        session_mock.closed = False
        session_mock.connector = MagicMock(spec=TCPConnector)
        session_mock.connector.limit = 100
        session_mock.connector.limit_per_host = 30
        session_mock.connector.keepalive_timeout = 30.0
        session_mock.connector._acquired = []
        session_mock.connector._available_connections = []
        return session_mock
    
    @pytest.mark.asyncio
    async def test_create_session(self, pool_manager):
        """Test creating a new session"""
        base_url = "https://api.example.com"
        
        with patch('aiohttp.ClientSession') as mock_client_session:
            mock_session = AsyncMock()
            mock_client_session.return_value = mock_session
            
            await pool_manager._create_session(base_url)
            
            assert base_url in pool_manager.sessions
            assert pool_manager.sessions[base_url] == mock_session
            mock_client_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session_new(self, pool_manager):
        """Test getting session when it doesn't exist"""
        base_url = "https://api.example.com"
        
        with patch.object(pool_manager, '_create_session') as mock_create:
            mock_session = AsyncMock()
            pool_manager.sessions[base_url] = mock_session
            
            session = await pool_manager.get_session(base_url)
            
            assert session == mock_session
            mock_create.assert_called_once_with(base_url)
    
    @pytest.mark.asyncio
    async def test_get_session_existing(self, pool_manager, mock_session):
        """Test getting existing session"""
        base_url = "https://api.example.com"
        pool_manager.sessions[base_url] = mock_session
        
        session = await pool_manager.get_session(base_url)
        
        assert session == mock_session
    
    @pytest.mark.asyncio
    async def test_get_session_closed(self, pool_manager, mock_session):
        """Test getting session when existing session is closed"""
        base_url = "https://api.example.com"
        mock_session.closed = True
        pool_manager.sessions[base_url] = mock_session
        
        with patch.object(pool_manager, '_create_session') as mock_create:
            new_mock_session = AsyncMock()
            new_mock_session.closed = False
            pool_manager.sessions[base_url] = new_mock_session
            
            session = await pool_manager.get_session(base_url)
            
            assert session == new_mock_session
            mock_create.assert_called_once_with(base_url)
    
    @pytest.mark.asyncio
    async def test_close_session(self, pool_manager, mock_session):
        """Test closing a specific session"""
        base_url = "https://api.example.com"
        pool_manager.sessions[base_url] = mock_session
        
        await pool_manager.close_session(base_url)
        
        mock_session.close.assert_called_once()
        assert base_url not in pool_manager.sessions
    
    @pytest.mark.asyncio
    async def test_close_all_sessions(self, pool_manager):
        """Test closing all sessions"""
        # Create multiple mock sessions
        sessions = {}
        for i in range(3):
            base_url = f"https://api{i}.example.com"
            mock_session = AsyncMock()
            mock_session.closed = False
            sessions[base_url] = mock_session
            pool_manager.sessions[base_url] = mock_session
        
        await pool_manager.close_all_sessions()
        
        # Check all sessions were closed
        for session in sessions.values():
            session.close.assert_called_once()
        
        assert len(pool_manager.sessions) == 0
    
    @pytest.mark.asyncio
    async def test_get_connection_stats(self, pool_manager):
        """Test getting connection statistics"""
        # Add mock sessions
        for i in range(2):
            base_url = f"https://api{i}.example.com"
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_connector = MagicMock(spec=TCPConnector)
            mock_connector.limit = 100
            mock_connector.limit_per_host = 30
            mock_connector.keepalive_timeout = 30.0
            mock_connector._acquired = []
            mock_connector._available_connections = []
            mock_session.connector = mock_connector
            pool_manager.sessions[base_url] = mock_session
        
        stats = await pool_manager.get_connection_stats()
        
        assert stats["total_sessions"] == 2
        assert len(stats["sessions"]) == 2
        
        for session_stats in stats["sessions"].values():
            assert session_stats["closed"] is False
            assert session_stats["connector_limit"] == 100
            assert session_stats["limit_per_host"] == 30
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, pool_manager):
        """Test health check with healthy sessions"""
        # Add healthy mock session
        base_url = "https://api.example.com"
        mock_session = AsyncMock()
        mock_session.closed = False
        pool_manager.sessions[base_url] = mock_session
        
        health = await pool_manager.health_check()
        
        assert health["status"] == "healthy"
        assert health["total_sessions"] == 1
        assert health["unhealthy_sessions"] == 0
        assert health["details"][base_url]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_degraded(self, pool_manager):
        """Test health check with some unhealthy sessions"""
        # Add healthy session
        base_url1 = "https://api1.example.com"
        mock_session1 = AsyncMock()
        mock_session1.closed = False
        pool_manager.sessions[base_url1] = mock_session1
        
        # Add unhealthy session
        base_url2 = "https://api2.example.com"
        mock_session2 = AsyncMock()
        mock_session2.closed = True
        pool_manager.sessions[base_url2] = mock_session2
        
        health = await pool_manager.health_check()
        
        assert health["status"] == "degraded"
        assert health["total_sessions"] == 2
        assert health["unhealthy_sessions"] == 1
        assert health["details"][base_url1]["status"] == "healthy"
        assert health["details"][base_url2]["status"] == "closed"
    
    @pytest.mark.asyncio
    async def test_cleanup_idle_connections(self, pool_manager):
        """Test cleaning up idle connections"""
        # Add mock session with TCP connector
        base_url = "https://api.example.com"
        mock_session = AsyncMock()
        mock_connector = AsyncMock(spec=TCPConnector)
        mock_connector._available_connections = [1, 2, 3]  # Mock connections
        mock_session.connector = mock_connector
        pool_manager.sessions[base_url] = mock_session
        
        # Mock cleanup method
        mock_connector._cleanup_closed = AsyncMock()
        
        result = await pool_manager.cleanup_idle_connections()
        
        mock_connector._cleanup_closed.assert_called_once()
        # Result depends on implementation details, just check it's a number
        assert isinstance(result, int)


class TestPooledHTTPClient:
    """Test pooled HTTP client functionality"""
    
    @pytest.fixture
    def mock_pool_manager(self):
        """Create mock connection pool manager"""
        return AsyncMock(spec=ConnectionPoolManager)
    
    @pytest.fixture
    def pooled_client(self, mock_pool_manager):
        """Create pooled HTTP client with mock pool manager"""
        return PooledHTTPClient(mock_pool_manager)
    
    @pytest.fixture
    def mock_session(self):
        """Create mock session"""
        session_mock = AsyncMock()
        return session_mock
    
    @pytest.mark.asyncio
    async def test_request_with_base_url(self, pooled_client, mock_pool_manager, mock_session):
        """Test making request with explicit base URL"""
        mock_pool_manager.get_session.return_value = mock_session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session.request.return_value = mock_response
        
        response = await pooled_client.request(
            method="GET",
            url="/test",
            base_url="https://api.example.com"
        )
        
        assert response == mock_response
        mock_pool_manager.get_session.assert_called_once_with("https://api.example.com")
        mock_session.request.assert_called_once_with("GET", "/test")
    
    @pytest.mark.asyncio
    async def test_request_with_full_url(self, pooled_client, mock_pool_manager, mock_session):
        """Test making request with full URL"""
        mock_pool_manager.get_session.return_value = mock_session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session.request.return_value = mock_response
        
        response = await pooled_client.request(
            method="GET",
            url="https://api.example.com/test"
        )
        
        assert response == mock_response
        mock_pool_manager.get_session.assert_called_once_with("https://api.example.com")
        mock_session.request.assert_called_once_with("GET", "/test")
    
    @pytest.mark.asyncio
    async def test_get_request(self, pooled_client, mock_pool_manager, mock_session):
        """Test GET request convenience method"""
        mock_pool_manager.get_session.return_value = mock_session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session.request.return_value = mock_response
        
        response = await pooled_client.get("/test", base_url="https://api.example.com")
        
        assert response == mock_response
        mock_session.request.assert_called_once_with("GET", "/test")
    
    @pytest.mark.asyncio
    async def test_post_request(self, pooled_client, mock_pool_manager, mock_session):
        """Test POST request convenience method"""
        mock_pool_manager.get_session.return_value = mock_session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session.request.return_value = mock_response
        
        response = await pooled_client.post(
            "/test", 
            base_url="https://api.example.com",
            json={"data": "test"}
        )
        
        assert response == mock_response
        mock_session.request.assert_called_once_with("POST", "/test", json={"data": "test"})
    
    @pytest.mark.asyncio
    async def test_request_error_handling(self, pooled_client, mock_pool_manager, mock_session):
        """Test request error handling"""
        mock_pool_manager.get_session.return_value = mock_session
        mock_session.request.side_effect = Exception("Connection error")
        
        with pytest.raises(Exception, match="Connection error"):
            await pooled_client.request("GET", "/test", base_url="https://api.example.com")
    
    @pytest.mark.asyncio
    async def test_close(self, pooled_client, mock_pool_manager):
        """Test closing all connections"""
        await pooled_client.close()
        
        mock_pool_manager.close_all_sessions.assert_called_once()