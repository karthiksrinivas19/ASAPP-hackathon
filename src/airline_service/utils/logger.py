"""
Logging utilities
"""

import structlog
import logging
from typing import Any, Dict
from ..config import config


def setup_logging() -> None:
    """Setup structured logging configuration"""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=getattr(logging, config.logging.level.upper()),
    )
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if config.logging.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


class AuditLogger:
    """Audit logger for customer interactions"""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    async def log_request(self, request_data: Dict[str, Any]) -> None:
        """Log customer request"""
        if config.logging.enable_audit:
            self.logger.info(
                "customer_request",
                utterance=request_data.get("utterance", ""),
                session_id=request_data.get("session_id"),
                customer_id=request_data.get("customer_id"),
                timestamp=request_data.get("timestamp")
            )
    
    async def log_response(self, response_data: Dict[str, Any]) -> None:
        """Log system response"""
        if config.logging.enable_audit:
            self.logger.info(
                "system_response",
                status=response_data.get("status"),
                message=response_data.get("message"),
                timestamp=response_data.get("timestamp")
            )
    
    async def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log error occurrence"""
        self.logger.error(
            "system_error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )
    
    async def log_api_call(self, endpoint: str, duration: float, success: bool) -> None:
        """Log API call details"""
        if config.logging.enable_metrics:
            self.logger.info(
                "api_call",
                endpoint=endpoint,
                duration_ms=duration,
                success=success
            )