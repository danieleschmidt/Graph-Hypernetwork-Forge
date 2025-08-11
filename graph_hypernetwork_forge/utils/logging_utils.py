"""Centralized logging utilities with structured logging support."""

import logging
import logging.handlers
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
from datetime import datetime

# Define log levels
TRACE = 5  # Custom trace level below DEBUG

class StructuredFormatter(logging.Formatter):
    """Structured formatter for JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process,
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)


class GraphHypernetworkLogger:
    """Centralized logger for the Graph Hypernetwork Forge project."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.loggers = {}
            self.default_level = logging.INFO
            self.log_dir = Path("logs")
            self.setup_logging()
            GraphHypernetworkLogger._initialized = True
    
    def setup_logging(self, 
                     log_dir: Optional[Union[str, Path]] = None,
                     level: int = logging.INFO,
                     console_output: bool = True,
                     file_output: bool = True,
                     structured_format: bool = False) -> None:
        """Setup centralized logging configuration.
        
        Args:
            log_dir: Directory for log files
            level: Default logging level
            console_output: Whether to output to console
            file_output: Whether to output to files
            structured_format: Whether to use JSON structured format
        """
        if log_dir:
            self.log_dir = Path(log_dir)
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_level = level
        self.structured_format = structured_format
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all levels
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Setup formatters
        if structured_format:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handlers
        if file_output:
            # Main log file
            main_log_file = self.log_dir / "graph_hypernetwork.log"
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file, maxBytes=50*1024*1024, backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            # Error log file
            error_log_file = self.log_dir / "errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file, maxBytes=50*1024*1024, backupCount=5
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)
            
            # Performance log file (for profiling and timing)
            perf_log_file = self.log_dir / "performance.log"
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_file, maxBytes=50*1024*1024, backupCount=5
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.addFilter(PerformanceFilter())
            perf_handler.setFormatter(formatter)
            root_logger.addHandler(perf_handler)
    
    def get_logger(self, name: str, level: Optional[int] = None) -> logging.Logger:
        """Get or create a logger with the given name.
        
        Args:
            name: Logger name (usually module name)
            level: Optional override for logging level
            
        Returns:
            Configured logger instance
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level or self.default_level)
        
        # Add trace method
        logger.trace = lambda msg, *args, **kwargs: logger.log(TRACE, msg, *args, **kwargs)
        
        self.loggers[name] = logger
        return logger
    
    def log_with_context(self, 
                        logger_name: str, 
                        level: int, 
                        message: str, 
                        **context) -> None:
        """Log with additional context fields.
        
        Args:
            logger_name: Name of the logger
            level: Log level
            message: Log message
            **context: Additional context fields
        """
        logger = self.get_logger(logger_name)
        
        # Create log record with extra fields
        record = logger.makeRecord(
            logger.name, level, "", 0, message, (), None
        )
        record.extra_fields = context
        
        for handler in logger.handlers:
            handler.handle(record)


class PerformanceFilter(logging.Filter):
    """Filter to only allow performance-related log messages."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter performance-related messages."""
        perf_keywords = ['performance', 'timing', 'memory', 'gpu', 'cuda', 'profile']
        message = record.getMessage().lower()
        return any(keyword in message for keyword in perf_keywords)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if self._logger is None:
            logger_manager = GraphHypernetworkLogger()
            self._logger = logger_manager.get_logger(self.__class__.__module__)
        return self._logger
    
    def log_with_context(self, level: int, message: str, **context):
        """Log with additional context."""
        logger_manager = GraphHypernetworkLogger()
        logger_manager.log_with_context(
            self.__class__.__module__, level, message, 
            class_name=self.__class__.__name__, **context
        )


def get_logger(name: str = None) -> logging.Logger:
    """Convenient function to get a logger.
    
    Args:
        name: Logger name (defaults to caller's module)
        
    Returns:
        Configured logger instance
    """
    if name is None:
        # Get caller's module name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'unknown')
    
    logger_manager = GraphHypernetworkLogger()
    return logger_manager.get_logger(name)


def setup_logging(**kwargs):
    """Setup logging with given configuration."""
    logger_manager = GraphHypernetworkLogger()
    logger_manager.setup_logging(**kwargs)


def log_function_call(logger: Optional[logging.Logger] = None):
    """Decorator to log function calls with arguments and return values.
    
    Args:
        logger: Optional logger instance
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            # Log function entry
            logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful completion
                logger.debug(f"Completed {func.__name__} in {execution_time:.4f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log exception
                logger.error(
                    f"Exception in {func.__name__} after {execution_time:.4f}s: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_performance_metrics(logger: logging.Logger, 
                          metrics: Dict[str, Any], 
                          prefix: str = "performance") -> None:
    """Log performance metrics in a structured way.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics to log
        prefix: Prefix for log message
    """
    logger.info(f"{prefix}: {json.dumps(metrics)}")


# Initialize the global logger manager
_logger_manager = GraphHypernetworkLogger()

# Add trace level to logging module
logging.addLevelName(TRACE, "TRACE")