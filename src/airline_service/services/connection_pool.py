"""
Connection pool manager for airline API client.

This service provides HTTP connection pooling for efficient API communication
with configurable pool sizes, timeouts, and connection reuse.
"""

import asyncio
import structlog
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from ..config import config


logger = structlog.get_logger(__name__)


class ConnectionPoolManager:
    """
    HTTP connection pool manager for airline API client.
    
    Provides persistent connections, connection reuse, and automatic
    connection management for improved performance.
    """
    
    def __init__(self):
        """Initialize the connection pool manager."""
        self.sessions: Dict[str, ClientSession] = {}
        self.logger = structlog.get_logger("connection_pool")
        
        # Connection pool configuration
        self.pool_config = {
            "connector_limit": 100,  # Total connection pool size
            "connector_limit_per_host": 30,  # Connections per host
            "timeout_total": config.airline_api.timeout / 1000,  # Convert ms to seconds
            "timeout_connect": 10.0,  # Connection timeout
            "timeout_read": 30.0,  # Read timeout
            "keepalive_timeout": 30.0,  # Keep-alive timeout
            "enable_cleanup_closed": True
        }
    
    async def get_session(self, base_url: str) -> ClientSession:
        """
        Get or create a session for the given base URL.
        
        Args:
            base_url: Base URL for the API
            
        Returns:
            Configured ClientSession
        """
        if base_url not in self.sessions:
            await self._create_session(base_url)
        
        session = self.sessions[base_url]
        
        # Check if session is still valid
        if session.closed:
            self.logger.warning("Session was closed, creating new one", base_url=base_url)
            await self._create_session(base_url)
            session = self.sessions[base_url]
        
        return session
    
    async def _create_session(self, base_url: str) -> None:
        """
        Create a new session with connection pooling configuration.
        
        Args:
            base_url: Base URL for the API
        """
        # Create TCP connector with connection pooling
        connector = TCPConnector(
            limit=self.pool_config["connector_limit"],
            limit_per_host=self.pool_config["connector_limit_per_host"],
            keepalive_timeout=self.pool_config["keepalive_timeout"],
            enable_cleanup_closed=self.pool_config["enable_cleanup_closed"],
            use_dns_cache=True,
            ttl_dns_cache=300,  # DNS cache TTL: 5 minutes
            family=0,  # Allow both IPv4 and IPv6
            ssl=False  # Disable SSL verification for development
        )
        
        # Create timeout configuration
        timeout = ClientTimeout(
            total=self.pool_config["timeout_total"],
            connect=self.pool_config["timeout_connect"],
            sock_read=self.pool_config["timeout_read"]
        )
        
        # Create session
        session = ClientSession(
            base_url=base_url,
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "AirlineCustomerService/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            raise_for_status=False  # Handle status codes manually
        )
        
        self.sessions[base_url] = session
        
        self.logger.info(
            "Created new HTTP session",
            base_url=base_url,
            connector_limit=self.pool_config["connector_limit"],
            limit_per_host=self.pool_config["connector_limit_per_host"]
        )
    
    async def close_session(self, base_url: str) -> None:
        """
        Close a specific session.
        
        Args:
            base_url: Base URL of the session to close
        """
        if base_url in self.sessions:
            session = self.sessions[base_url]
            if not session.closed:
                await session.close()
            del self.sessions[base_url]
            
            self.logger.info("Closed HTTP session", base_url=base_url)
    
    async def close_all_sessions(self) -> None:
        """Close all sessions and clean up connections."""
        for base_url, session in self.sessions.items():
            if not session.closed:
                await session.close()
                self.logger.debug("Closed session", base_url=base_url)
        
        self.sessions.clear()
        self.logger.info("All HTTP sessions closed")
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Dictionary with connection statistics
        """
        stats = {
            "total_sessions": len(self.sessions),
            "sessions": {}
        }
        
        for base_url, session in self.sessions.items():
            connector = session.connector
            if isinstance(connector, TCPConnector):
                stats["sessions"][base_url] = {
                    "closed": session.closed,
                    "connector_limit": connector.limit,
                    "limit_per_host": connector.limit_per_host,
                    "acquired_connections": len(connector._acquired),
                    "available_connections": len(connector._available_connections),
                    "keepalive_timeout": connector.keepalive_timeout
                }
            else:
                stats["sessions"][base_url] = {
                    "closed": session.closed,
                    "connector_type": type(connector).__name__
                }
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on connection pools.
        
        Returns:
            Health check results
        """
        health_status = {
            "status": "healthy",
            "total_sessions": len(self.sessions),
            "unhealthy_sessions": 0,
            "details": {}
        }
        
        for base_url, session in self.sessions.items():
            session_health = {
                "status": "healthy" if not session.closed else "closed",
                "base_url": base_url
            }
            
            if session.closed:
                health_status["unhealthy_sessions"] += 1
                session_health["status"] = "closed"
            
            health_status["details"][base_url] = session_health
        
        # Overall health status
        if health_status["unhealthy_sessions"] > 0:
            health_status["status"] = "degraded"
        
        return health_status
    
    async def cleanup_idle_connections(self) -> int:
        """
        Clean up idle connections across all sessions.
        
        Returns:
            Number of connections cleaned up
        """
        cleaned_up = 0
        
        for base_url, session in self.sessions.items():
            connector = session.connector
            if isinstance(connector, TCPConnector):
                # Force cleanup of closed connections
                await connector._cleanup_closed()
                
                # Count available connections before cleanup
                before_count = len(connector._available_connections)
                
                # Cleanup will happen automatically, but we can trigger it
                await asyncio.sleep(0)  # Allow cleanup to run
                
                after_count = len(connector._available_connections)
                session_cleaned = before_count - after_count
                cleaned_up += session_cleaned
                
                if session_cleaned > 0:
                    self.logger.debug(
                        "Cleaned up idle connections",
                        base_url=base_url,
                        cleaned=session_cleaned
                    )
        
        if cleaned_up > 0:
            self.logger.info("Idle connection cleanup completed", total_cleaned=cleaned_up)
        
        return cleaned_up


class PooledHTTPClient:
    """
    HTTP client that uses connection pooling for improved performance.
    """
    
    def __init__(self, pool_manager: ConnectionPoolManager):
        """Initialize pooled HTTP client."""
        self.pool_manager = pool_manager
        self.logger = structlog.get_logger("pooled_http_client")
    
    async def request(
        self,
        method: str,
        url: str,
        base_url: Optional[str] = None,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Make HTTP request using connection pool.
        
        Args:
            method: HTTP method
            url: Request URL (can be relative if base_url provided)
            base_url: Base URL for the request
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response
        """
        # Determine base URL
        if base_url is None:
            # Extract base URL from full URL
            if url.startswith(('http://', 'https://')):
                parts = url.split('/', 3)
                base_url = '/'.join(parts[:3])
                url = '/' + parts[3] if len(parts) > 3 else '/'
            else:
                base_url = config.airline_api.base_url
        
        # Get session from pool
        session = await self.pool_manager.get_session(base_url)
        
        # Make request
        try:
            response = await session.request(method, url, **kwargs)
            
            self.logger.debug(
                "HTTP request completed",
                method=method,
                url=url,
                status=response.status,
                base_url=base_url
            )
            
            return response
            
        except Exception as e:
            self.logger.error(
                "HTTP request failed",
                method=method,
                url=url,
                base_url=base_url,
                error=str(e)
            )
            raise
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make GET request."""
        return await self.request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make POST request."""
        return await self.request("POST", url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make PUT request."""
        return await self.request("PUT", url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make DELETE request."""
        return await self.request("DELETE", url, **kwargs)
    
    async def close(self) -> None:
        """Close all connections."""
        await self.pool_manager.close_all_sessions()


# Global connection pool manager and client
connection_pool_manager = ConnectionPoolManager()
pooled_http_client = PooledHTTPClient(connection_pool_manager)