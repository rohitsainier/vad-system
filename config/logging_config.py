# config/logging_config.py
"""
Logging configuration for production VAD system
"""
import logging
import sys
from typing import Optional
from pathlib import Path
import structlog
from pythonjsonlogger import jsonlogger
from datetime import datetime


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['service'] = 'vad-system'
        
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None
) -> structlog.BoundLogger:
    """
    Configure structured logging for production
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Whether to use JSON format
        log_file: Optional log file name
        log_dir: Optional log directory
        
    Returns:
        Configured structlog logger
    """
    
    # Determine log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure structlog processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_format:
        # JSON format for production
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Console format for development
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer(colors=True)
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # Configure standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if json_format:
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file or log_dir:
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file_path = log_path / (log_file or f"vad_{datetime.now().strftime('%Y%m%d')}.log")
        else:
            log_file_path = Path(log_file)
        
        file_handler = logging.FileHandler(str(log_file_path))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(CustomJsonFormatter() if json_format else formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    return structlog.get_logger()


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance with the given name"""
    return structlog.get_logger(name)


# Convenience function for quick setup
def configure_for_production():
    """Configure logging for production environment"""
    return setup_logging(
        level="INFO",
        json_format=True,
        log_dir="/var/log/vad"
    )


def configure_for_development():
    """Configure logging for development environment"""
    return setup_logging(
        level="DEBUG",
        json_format=False
    )