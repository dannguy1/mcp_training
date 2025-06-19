"""
Logging utilities for MCP Training Service.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "structured",
    max_size: str = "100MB",
    backup_count: int = 10,
    console_output: bool = True
) -> None:
    """Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Log format ("structured" or "standard")
        max_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
    """
    # Convert string log level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    if log_format == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max_size (e.g., "100MB" -> 100*1024*1024)
        size_bytes = _parse_size(max_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=size_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, level: str, message: str, **context):
    """Log a message with additional context fields.
    
    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **context: Additional context fields
    """
    log_func = getattr(logger, level.lower())
    
    # Create a log record with extra fields
    record = logger.makeRecord(
        logger.name, getattr(logging, level.upper()), 
        "", 0, message, (), None
    )
    record.extra_fields = context
    
    log_func(message, extra={'extra_fields': context})


def _parse_size(size_str: str) -> int:
    """Parse size string to bytes.
    
    Args:
        size_str: Size string (e.g., "100MB", "1GB")
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


class TrainingLogger:
    """Specialized logger for training operations."""
    
    def __init__(self, training_id: str):
        """Initialize training logger.
        
        Args:
            training_id: Unique training job ID
        """
        self.training_id = training_id
        self.logger = get_logger(f"training.{training_id}")
    
    def log_progress(self, step: str, progress: float, message: str = ""):
        """Log training progress.
        
        Args:
            step: Current training step
            progress: Progress percentage (0-100)
            message: Additional message
        """
        log_with_context(
            self.logger, "info", f"Training progress: {step}",
            training_id=self.training_id,
            step=step,
            progress=progress,
            message=message
        )
    
    def log_error(self, error: str, step: str = ""):
        """Log training error.
        
        Args:
            error: Error message
            step: Step where error occurred
        """
        log_with_context(
            self.logger, "error", f"Training error: {error}",
            training_id=self.training_id,
            step=step,
            error=error
        )
    
    def log_completion(self, result: Dict[str, Any]):
        """Log training completion.
        
        Args:
            result: Training results
        """
        log_with_context(
            self.logger, "info", "Training completed successfully",
            training_id=self.training_id,
            result=result
        ) 