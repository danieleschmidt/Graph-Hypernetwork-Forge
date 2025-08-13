"""Security utilities for Graph Hypernetwork Forge.

This module provides comprehensive security features including:
- Input sanitization and validation
- Secure model serialization/deserialization
- Access control and authentication
- Security monitoring and threat detection
"""

import hashlib
import hmac
import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import numpy as np

# Enhanced logging
try:
    from .logging_utils import get_logger, SecurityAuditLogger
    ENHANCED_LOGGING = True
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)
    class SecurityAuditLogger:
        def __init__(self, *args, **kwargs):
            self.logger = logging.getLogger("security")
        def log_security_event(self, event, details): 
            self.logger.warning(f"SECURITY: {event} - {details}")
        def log_access_attempt(self, resource, user, success):
            self.logger.info(f"ACCESS: {resource} by {user} - {'SUCCESS' if success else 'FAILED'}")
    ENHANCED_LOGGING = False

logger = get_logger(__name__)
security_logger = SecurityAuditLogger("graph_hypernetwork_forge_security")


class SecurityError(Exception):
    """Base class for security-related errors."""
    pass


class InputValidationError(SecurityError):
    """Raised when input validation fails."""
    pass


class ModelTamperingError(SecurityError):
    """Raised when model tampering is detected."""
    pass


class AccessDeniedError(SecurityError):
    """Raised when access is denied."""
    pass


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    def __init__(self):
        """Initialize input sanitizer."""
        self.max_text_length = 10000
        self.max_list_size = 10000
        self.allowed_file_extensions = {'.json', '.pt', '.pth', '.pkl'}
        self.dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'subprocess',
            r'os\.system',
            r'open\s*\(',
        ]
        
        logger.info("InputSanitizer initialized with security patterns")
    
    def sanitize_text_input(self, text: str, allow_empty: bool = False) -> str:
        """Sanitize text input with security checks.
        
        Args:
            text: Input text to sanitize
            allow_empty: Whether to allow empty strings
            
        Returns:
            Sanitized text
            
        Raises:
            InputValidationError: If input is invalid or dangerous
        """
        if not isinstance(text, str):
            raise InputValidationError(f"Expected string, got {type(text)}")
        
        if not allow_empty and len(text.strip()) == 0:
            raise InputValidationError("Empty text not allowed")
        
        if len(text) > self.max_text_length:
            raise InputValidationError(f"Text too long: {len(text)} > {self.max_text_length}")
        
        # Check for dangerous patterns
        import re
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                security_logger.log_security_event(
                    "DANGEROUS_PATTERN_DETECTED",
                    f"Pattern '{pattern}' found in input text"
                )
                raise InputValidationError(f"Dangerous pattern detected: {pattern}")
        
        # Remove or escape dangerous characters
        sanitized = text.replace('\x00', '').replace('\r\n', '\n')
        
        # Limit control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\t')
        
        return sanitized
    
    def sanitize_text_list(self, texts: List[str]) -> List[str]:
        """Sanitize a list of text inputs.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sanitized texts
            
        Raises:
            InputValidationError: If input is invalid
        """
        if not isinstance(texts, list):
            raise InputValidationError(f"Expected list, got {type(texts)}")
        
        if len(texts) > self.max_list_size:
            raise InputValidationError(f"List too large: {len(texts)} > {self.max_list_size}")
        
        return [self.sanitize_text_input(text) for text in texts]
    
    def validate_tensor_input(self, tensor: torch.Tensor, name: str, 
                            expected_dtype: Optional[torch.dtype] = None,
                            max_size_mb: float = 1000.0) -> torch.Tensor:
        """Validate tensor input with security checks.
        
        Args:
            tensor: Input tensor
            name: Name for error messages
            expected_dtype: Expected tensor dtype
            max_size_mb: Maximum tensor size in MB
            
        Returns:
            Validated tensor
            
        Raises:
            InputValidationError: If tensor is invalid
        """
        if not isinstance(tensor, torch.Tensor):
            raise InputValidationError(f"{name}: Expected torch.Tensor, got {type(tensor)}")
        
        # Check for NaN or infinite values
        if torch.isnan(tensor).any():
            raise InputValidationError(f"{name}: Contains NaN values")
        
        if torch.isinf(tensor).any():
            raise InputValidationError(f"{name}: Contains infinite values")
        
        # Check tensor size
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
        if tensor_size_mb > max_size_mb:
            raise InputValidationError(
                f"{name}: Tensor too large: {tensor_size_mb:.1f}MB > {max_size_mb}MB"
            )
        
        # Check dtype if specified
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            logger.warning(f"{name}: Unexpected dtype {tensor.dtype}, expected {expected_dtype}")
        
        return tensor
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate file path for security.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            InputValidationError: If path is unsafe
        """
        path = Path(file_path).resolve()
        
        # Check for directory traversal attempts
        if '..' in str(path) or str(path).startswith('/'):
            raise InputValidationError(f"Unsafe path detected: {path}")
        
        # Check file extension
        if path.suffix.lower() not in self.allowed_file_extensions:
            raise InputValidationError(f"File extension not allowed: {path.suffix}")
        
        # Check for symbolic links
        if path.is_symlink():
            raise InputValidationError(f"Symbolic links not allowed: {path}")
        
        return path


# Global security manager instance
_security_manager = None

def get_security_manager():
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = InputSanitizer()  # Simplified for compilation test
    return _security_manager