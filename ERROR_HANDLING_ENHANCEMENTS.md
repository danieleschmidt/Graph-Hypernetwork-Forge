# Error Handling and Validation Enhancements

This document describes the comprehensive error handling and validation enhancements implemented throughout the Graph Hypernetwork Forge codebase.

## Overview

The error handling system has been completely redesigned to provide:

- **Comprehensive input validation** with clear error messages
- **Graceful error recovery** mechanisms
- **Structured logging** throughout the codebase
- **Memory monitoring** and automatic cleanup
- **GPU memory management** with CUDA error handling
- **Runtime type checking** where critical
- **Custom exception hierarchy** for better error categorization

## Core Components

### 1. Centralized Logging System (`utils/logging_utils.py`)

**Features:**
- Structured JSON logging support
- Multiple output handlers (console, file, performance logs)
- Automatic log rotation
- Context-aware logging with metadata
- Performance metrics logging
- Trace-level logging for detailed debugging

**Key Classes:**
- `GraphHypernetworkLogger`: Singleton logger manager
- `StructuredFormatter`: JSON log formatting
- `LoggerMixin`: Easy logging integration for any class
- `PerformanceFilter`: Performance-specific log filtering

**Usage:**
```python
from graph_hypernetwork_forge.utils import get_logger, setup_logging

# Setup logging
setup_logging(level="INFO", structured_format=True)

# Get logger
logger = get_logger(__name__)
logger.info("Processing started", extra={'batch_size': 32})
```

### 2. Custom Exception Hierarchy (`utils/exceptions.py`)

**Base Exception:**
- `GraphHypernetworkError`: Base for all custom exceptions

**Specialized Exceptions:**
- `ValidationError`: Input validation failures
- `ModelError`: Model-related issues
- `DataError`: Data consistency problems
- `GPUError`: CUDA/GPU-related errors
- `MemoryError`: Memory management issues
- `FileIOError`: File operation failures
- `NetworkError`: Network/model loading errors
- `TrainingError`: Training process errors
- `InferenceError`: Inference failures
- `GraphStructureError`: Graph structure violations

**Features:**
- Rich error context with structured details
- Error chaining to preserve original exceptions
- Standardized error handling utilities
- Automatic CUDA OOM error detection

**Usage:**
```python
from graph_hypernetwork_forge.utils import ValidationError

if not isinstance(value, int) or value <= 0:
    raise ValidationError("hidden_dim", value, "positive integer")
```

### 3. Memory Management (`utils/memory_utils.py`)

**Features:**
- Real-time memory monitoring
- Automatic cleanup when thresholds exceeded
- GPU memory estimation and validation
- Safe CUDA operation wrappers
- Memory-managed context managers
- Background monitoring threads

**Key Classes:**
- `MemoryMonitor`: System resource monitoring
- `MemoryInfo`: Memory statistics container

**Key Functions:**
- `memory_management()`: Context manager for automatic cleanup
- `check_gpu_memory_available()`: GPU memory validation
- `safe_cuda_operation()`: Retry logic for CUDA operations
- `estimate_tensor_memory()`: Memory usage estimation

**Usage:**
```python
from graph_hypernetwork_forge.utils import memory_management

with memory_management(cleanup_on_exit=True):
    # Memory-intensive operations
    result = model(large_input)
```

## Enhanced Modules

### 1. HyperGNN Model (`models/hypergnn.py`)

**Enhancements:**
- Comprehensive input validation for all parameters
- GPU memory checks before operations
- CUDA error handling with automatic recovery
- Text input validation and sanitization
- Model state validation
- Automatic memory cleanup
- Structured logging throughout

**Key Improvements:**
- All tensor operations wrapped with error handling
- Input dimension validation
- GPU memory estimation before allocation
- Graceful handling of model loading failures
- Comprehensive error messages with context

### 2. Training Utilities (`utils/training.py`)

**Enhancements:**
- Memory monitoring during training
- Automatic error recovery and retry logic
- Comprehensive validation of training data
- GPU memory management
- Structured logging with metrics
- Automatic cleanup of training resources
- Graceful degradation on errors

**Key Improvements:**
- Training graphs validated before processing
- Memory cleanup between batches
- Error tolerance with configurable thresholds
- Detailed training metrics logging
- Automatic model checkpointing with error handling

### 3. Knowledge Graph (`data/knowledge_graph.py`)

**Enhancements:**
- Comprehensive data validation
- File I/O error handling
- Graph structure validation
- Tensor consistency checks
- Safe file operations with rollback
- Detailed error reporting

**Key Improvements:**
- JSON parsing with detailed error reporting
- Graph structure validation (node indices, dimensions)
- Safe node/edge addition with rollback on failure
- File format validation
- Memory-efficient loading for large graphs

## Usage Guidelines

### 1. Error Handling Best Practices

```python
from graph_hypernetwork_forge.utils import (
    ValidationError, ModelError, log_function_call, get_logger
)

logger = get_logger(__name__)

@log_function_call()
def process_data(data):
    """Process data with comprehensive error handling."""
    try:
        # Validate inputs
        if not isinstance(data, list):
            raise ValidationError("data", type(data).__name__, "list")
        
        # Process data
        result = expensive_operation(data)
        
        logger.info("Data processing completed", extra={
            'input_size': len(data),
            'output_size': len(result)
        })
        
        return result
        
    except ValidationError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Data processing failed: {e}", exc_info=True)
        raise ModelError("data_processor", "process_data", str(e))
```

### 2. Memory Management Integration

```python
from graph_hypernetwork_forge.utils import (
    memory_management, MemoryMonitor, check_gpu_memory_available
)

def train_model(model, data):
    """Train model with memory management."""
    # Check GPU memory before starting
    estimated_memory = estimate_model_memory(model, data)
    check_gpu_memory_available(estimated_memory, "training")
    
    # Setup memory monitoring
    monitor = MemoryMonitor(cleanup_callbacks=[cleanup_training_cache])
    monitor.start_monitoring()
    
    try:
        with memory_management(cleanup_on_exit=True):
            for epoch in range(num_epochs):
                # Training logic
                train_epoch(model, data)
                
    finally:
        monitor.stop_monitoring()
```

### 3. Structured Logging

```python
from graph_hypernetwork_forge.utils import get_logger, setup_logging

# Setup logging once in your application
setup_logging(
    log_dir="logs",
    level="INFO", 
    structured_format=True,
    console_output=True
)

logger = get_logger(__name__)

# Log with context
logger.info("Training started", extra={
    'model_type': 'HyperGNN',
    'dataset_size': 1000,
    'batch_size': 32
})

# Performance logging
import time
start_time = time.time()
# ... operation ...
logger.info("Operation completed", extra={
    'duration_ms': (time.time() - start_time) * 1000,
    'operation': 'model_inference'
})
```

## Configuration

### Logging Configuration

```python
from graph_hypernetwork_forge.utils import setup_logging

setup_logging(
    log_dir="logs",           # Log directory
    level="INFO",             # Log level
    console_output=True,      # Console logging
    file_output=True,         # File logging
    structured_format=False   # JSON vs. text format
)
```

### Memory Monitoring Configuration

```python
from graph_hypernetwork_forge.utils import start_global_memory_monitoring

start_global_memory_monitoring(
    interval=30.0  # Monitoring interval in seconds
)
```

## Testing the Enhancements

Run the demonstration script to see all features in action:

```bash
python examples/enhanced_error_handling_demo.py
```

This will demonstrate:
- Structured logging with different levels
- Memory management and monitoring
- Input validation with clear error messages
- Graceful error recovery
- GPU memory handling
- Custom exception handling

## Benefits

1. **Improved Reliability**: Comprehensive validation prevents invalid states
2. **Better Debugging**: Structured logging provides detailed context
3. **Resource Management**: Automatic memory monitoring prevents OOM errors
4. **User Experience**: Clear error messages help identify issues quickly
5. **Maintainability**: Consistent error handling patterns across codebase
6. **Production Readiness**: Robust error recovery and logging for deployment

## Migration Guide

For existing code using the library:

1. **Import Updates**: New utilities are available in `graph_hypernetwork_forge.utils`
2. **Error Handling**: Catch specific exception types instead of generic `Exception`
3. **Logging**: Replace print statements with structured logging
4. **Memory**: Use memory management contexts for memory-intensive operations

The enhancements are backward compatible, but adopting the new patterns will provide better error handling and debugging capabilities.