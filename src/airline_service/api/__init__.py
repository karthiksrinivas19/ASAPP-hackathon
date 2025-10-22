"""
API module initialization
"""

from .monitoring_endpoints import router as monitoring_router

__all__ = [
    'monitoring_router'
]