"""Custom exception classes for Graph Hypernetwork Forge."""

from typing import Any, Dict, List, Optional, Union


class GraphHypernetworkError(Exception):
    """Base exception class for all Graph Hypernetwork Forge errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        """Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'cause': str(self.cause) if self.cause else None
        }


class ValidationError(GraphHypernetworkError):
    """Exception raised when input validation fails."""
    
    def __init__(self, field: str, value: Any, expected: str, message: Optional[str] = None):
        """Initialize validation error.
        
        Args:
            field: Name of the field that failed validation
            value: Invalid value
            expected: Description of expected value
            message: Optional custom message
        """
        if message is None:
            message = f"Validation failed for field '{field}': got {type(value).__name__} '{value}', expected {expected}"
        
        details = {
            'field': field,
            'value': str(value),
            'value_type': type(value).__name__,
            'expected': expected
        }
        
        super().__init__(message, details)
        self.field = field
        self.value = value
        self.expected = expected


class ConfigurationError(GraphHypernetworkError):
    """Exception raised when there are configuration issues."""
    
    def __init__(self, config_key: str, message: str, config_value: Any = None):
        """Initialize configuration error.
        
        Args:
            config_key: Configuration key that caused the error
            message: Error message
            config_value: Optional configuration value
        """
        details = {
            'config_key': config_key,
            'config_value': str(config_value) if config_value is not None else None
        }
        
        full_message = f"Configuration error for '{config_key}': {message}"
        super().__init__(full_message, details)
        self.config_key = config_key
        self.config_value = config_value


class ModelError(GraphHypernetworkError):
    """Exception raised when there are model-related issues."""
    
    def __init__(self, model_name: str, operation: str, message: str, model_state: Optional[Dict] = None):
        """Initialize model error.
        
        Args:
            model_name: Name of the model
            operation: Operation that failed
            message: Error message
            model_state: Optional model state information
        """
        details = {
            'model_name': model_name,
            'operation': operation,
            'model_state': model_state or {}
        }
        
        full_message = f"Model error in {model_name} during {operation}: {message}"
        super().__init__(full_message, details)
        self.model_name = model_name
        self.operation = operation


class DataError(GraphHypernetworkError):
    """Exception raised when there are data-related issues."""
    
    def __init__(self, data_type: str, issue: str, data_shape: Optional[tuple] = None, expected_shape: Optional[tuple] = None):
        """Initialize data error.
        
        Args:
            data_type: Type of data (e.g., 'node_features', 'edge_index')
            issue: Description of the issue
            data_shape: Actual data shape
            expected_shape: Expected data shape
        """
        details = {
            'data_type': data_type,
            'data_shape': data_shape,
            'expected_shape': expected_shape
        }
        
        message = f"Data error with {data_type}: {issue}"
        if data_shape and expected_shape:
            message += f" (got shape {data_shape}, expected {expected_shape})"
        
        super().__init__(message, details)
        self.data_type = data_type
        self.issue = issue


class GPUError(GraphHypernetworkError):
    """Exception raised when there are GPU/CUDA-related issues."""
    
    def __init__(self, operation: str, cuda_error: str, memory_info: Optional[Dict] = None):
        """Initialize GPU error.
        
        Args:
            operation: Operation that failed
            cuda_error: CUDA error message
            memory_info: Optional GPU memory information
        """
        details = {
            'operation': operation,
            'cuda_error': cuda_error,
            'memory_info': memory_info or {}
        }
        
        message = f"GPU error during {operation}: {cuda_error}"
        super().__init__(message, details)
        self.operation = operation
        self.cuda_error = cuda_error


class MemoryError(GraphHypernetworkError):
    """Exception raised when there are memory-related issues."""
    
    def __init__(self, operation: str, memory_required: Optional[int] = None, memory_available: Optional[int] = None):
        """Initialize memory error.
        
        Args:
            operation: Operation that failed
            memory_required: Required memory in bytes
            memory_available: Available memory in bytes
        """
        details = {
            'operation': operation,
            'memory_required_mb': memory_required / (1024**2) if memory_required else None,
            'memory_available_mb': memory_available / (1024**2) if memory_available else None
        }
        
        message = f"Memory error during {operation}"
        if memory_required and memory_available:
            message += f": required {memory_required / (1024**2):.1f} MB, available {memory_available / (1024**2):.1f} MB"
        
        super().__init__(message, details)
        self.operation = operation
        self.memory_required = memory_required
        self.memory_available = memory_available


class FileIOError(GraphHypernetworkError):
    """Exception raised when there are file I/O issues."""
    
    def __init__(self, file_path: str, operation: str, original_error: Exception):
        """Initialize file I/O error.
        
        Args:
            file_path: Path to the file
            operation: File operation that failed
            original_error: Original exception
        """
        details = {
            'file_path': file_path,
            'operation': operation,
            'original_error_type': type(original_error).__name__
        }
        
        message = f"File I/O error during {operation} of '{file_path}': {original_error}"
        super().__init__(message, details, cause=original_error)
        self.file_path = file_path
        self.operation = operation


class NetworkError(GraphHypernetworkError):
    """Exception raised when there are network/remote model loading issues."""
    
    def __init__(self, model_name: str, url: Optional[str] = None, timeout: Optional[float] = None):
        """Initialize network error.
        
        Args:
            model_name: Name of the model being loaded
            url: Optional URL being accessed
            timeout: Optional timeout value
        """
        details = {
            'model_name': model_name,
            'url': url,
            'timeout': timeout
        }
        
        message = f"Network error loading model '{model_name}'"
        if url:
            message += f" from {url}"
        if timeout:
            message += f" (timeout: {timeout}s)"
        
        super().__init__(message, details)
        self.model_name = model_name
        self.url = url


class TrainingError(GraphHypernetworkError):
    """Exception raised during training process."""
    
    def __init__(self, epoch: int, batch: Optional[int] = None, loss: Optional[float] = None, message: str = "Training failed"):
        """Initialize training error.
        
        Args:
            epoch: Epoch when error occurred
            batch: Optional batch number
            loss: Optional loss value
            message: Error message
        """
        details = {
            'epoch': epoch,
            'batch': batch,
            'loss': loss
        }
        
        full_message = f"Training error at epoch {epoch}"
        if batch is not None:
            full_message += f", batch {batch}"
        if loss is not None:
            full_message += f", loss {loss:.4f}"
        full_message += f": {message}"
        
        super().__init__(full_message, details)
        self.epoch = epoch
        self.batch = batch
        self.loss = loss


class InferenceError(GraphHypernetworkError):
    """Exception raised during model inference."""
    
    def __init__(self, input_shape: tuple, model_name: str, message: str):
        """Initialize inference error.
        
        Args:
            input_shape: Shape of input that caused the error
            model_name: Name of the model
            message: Error message
        """
        details = {
            'input_shape': input_shape,
            'model_name': model_name
        }
        
        full_message = f"Inference error in {model_name} with input shape {input_shape}: {message}"
        super().__init__(full_message, details)
        self.input_shape = input_shape
        self.model_name = model_name


class GraphStructureError(GraphHypernetworkError):
    """Exception raised when there are graph structure issues."""
    
    def __init__(self, graph_id: str, issue: str, num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        """Initialize graph structure error.
        
        Args:
            graph_id: Identifier for the graph
            issue: Description of the structural issue
            num_nodes: Optional number of nodes
            num_edges: Optional number of edges
        """
        details = {
            'graph_id': graph_id,
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }
        
        message = f"Graph structure error in {graph_id}: {issue}"
        if num_nodes is not None:
            message += f" (nodes: {num_nodes}"
            if num_edges is not None:
                message += f", edges: {num_edges}"
            message += ")"
        
        super().__init__(message, details)
        self.graph_id = graph_id
        self.issue = issue


# Error handling utilities
def handle_cuda_out_of_memory(operation: str, required_memory: Optional[int] = None) -> GPUError:
    """Create a standardized CUDA out of memory error.
    
    Args:
        operation: Operation that failed
        required_memory: Optional required memory in bytes
        
    Returns:
        GPUError instance
    """
    import torch
    
    memory_info = {}
    if torch.cuda.is_available():
        memory_info = {
            'allocated_mb': torch.cuda.memory_allocated() / (1024**2),
            'reserved_mb': torch.cuda.memory_reserved() / (1024**2),
            'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024**2),
            'max_reserved_mb': torch.cuda.max_memory_reserved() / (1024**2),
        }
    
    return GPUError(
        operation=operation,
        cuda_error="CUDA out of memory",
        memory_info=memory_info
    )


def create_validation_error(field: str, value: Any, constraints: List[str]) -> ValidationError:
    """Create a validation error with multiple constraints.
    
    Args:
        field: Field name
        value: Invalid value
        constraints: List of constraint descriptions
        
    Returns:
        ValidationError instance
    """
    expected = " and ".join(constraints)
    return ValidationError(field, value, expected)


def log_and_raise_error(logger, error: GraphHypernetworkError) -> None:
    """Log an error and then raise it.
    
    Args:
        logger: Logger instance
        error: Error to log and raise
    """
    logger.error(f"{error.__class__.__name__}: {error.message}", extra=error.details)
    if error.cause:
        logger.error(f"Caused by: {error.cause}", exc_info=error.cause)
    raise error