"""Memory management and monitoring utilities."""

import gc
import os
import psutil
import torch
import threading
import time
from typing import Dict, Optional, Any, Callable, List
from contextlib import contextmanager
from dataclasses import dataclass
from .logging_utils import get_logger
from .exceptions import MemoryError, GPUError


logger = get_logger(__name__)


@dataclass
class MemoryInfo:
    """Memory information container."""
    system_total_gb: float
    system_available_gb: float
    system_used_gb: float
    system_percent: float
    process_memory_gb: float
    gpu_allocated_gb: Optional[float] = None
    gpu_reserved_gb: Optional[float] = None
    gpu_total_gb: Optional[float] = None


class MemoryMonitor:
    """Monitor and manage memory usage."""
    
    def __init__(self, 
                 warning_threshold: float = 0.8,
                 critical_threshold: float = 0.95,
                 gpu_warning_threshold: float = 0.8,
                 cleanup_callbacks: Optional[List[Callable]] = None):
        """Initialize memory monitor.
        
        Args:
            warning_threshold: System memory threshold for warnings (0.0-1.0)
            critical_threshold: System memory threshold for critical alerts (0.0-1.0)
            gpu_warning_threshold: GPU memory threshold for warnings (0.0-1.0)
            cleanup_callbacks: List of functions to call when memory is low
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.gpu_warning_threshold = gpu_warning_threshold
        self.cleanup_callbacks = cleanup_callbacks or []
        
        self.monitoring = False
        self.monitor_thread = None
        self.last_warning_time = 0
        self.warning_cooldown = 60  # seconds
        
        # Process reference
        self.process = psutil.Process(os.getpid())
    
    def get_memory_info(self) -> MemoryInfo:
        """Get current memory information.
        
        Returns:
            MemoryInfo with current memory statistics
        """
        # System memory
        sys_memory = psutil.virtual_memory()
        
        # Process memory
        process_memory = self.process.memory_info()
        
        memory_info = MemoryInfo(
            system_total_gb=sys_memory.total / (1024**3),
            system_available_gb=sys_memory.available / (1024**3),
            system_used_gb=sys_memory.used / (1024**3),
            system_percent=sys_memory.percent,
            process_memory_gb=process_memory.rss / (1024**3)
        )
        
        # GPU memory if available
        if torch.cuda.is_available():
            try:
                memory_info.gpu_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                memory_info.gpu_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                
                # Get total GPU memory
                if hasattr(torch.cuda, 'get_device_properties'):
                    props = torch.cuda.get_device_properties(torch.cuda.current_device())
                    memory_info.gpu_total_gb = props.total_memory / (1024**3)
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
        
        return memory_info
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and return status.
        
        Returns:
            Dictionary with memory status and alerts
        """
        memory_info = self.get_memory_info()
        status = {
            'memory_info': memory_info,
            'warnings': [],
            'critical': False,
            'should_cleanup': False
        }
        
        # Check system memory
        if memory_info.system_percent / 100.0 >= self.critical_threshold:
            status['critical'] = True
            status['should_cleanup'] = True
            status['warnings'].append(f"Critical system memory usage: {memory_info.system_percent:.1f}%")
        elif memory_info.system_percent / 100.0 >= self.warning_threshold:
            status['warnings'].append(f"High system memory usage: {memory_info.system_percent:.1f}%")
        
        # Check GPU memory
        if memory_info.gpu_total_gb and memory_info.gpu_allocated_gb:
            gpu_usage_ratio = memory_info.gpu_allocated_gb / memory_info.gpu_total_gb
            if gpu_usage_ratio >= self.gpu_warning_threshold:
                status['warnings'].append(
                    f"High GPU memory usage: {gpu_usage_ratio*100:.1f}% "
                    f"({memory_info.gpu_allocated_gb:.2f}/{memory_info.gpu_total_gb:.2f} GB)"
                )
                if gpu_usage_ratio >= 0.95:
                    status['critical'] = True
                    status['should_cleanup'] = True
        
        return status
    
    def cleanup_memory(self) -> Dict[str, Any]:
        """Perform memory cleanup operations.
        
        Returns:
            Dictionary with cleanup results
        """
        logger.info("Starting memory cleanup...")
        
        memory_before = self.get_memory_info()
        
        cleanup_results = {
            'memory_before': memory_before,
            'actions_taken': [],
            'errors': []
        }
        
        try:
            # Python garbage collection
            collected = gc.collect()
            cleanup_results['actions_taken'].append(f"Python GC collected {collected} objects")
            
            # Clear PyTorch cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cleanup_results['actions_taken'].append("Cleared PyTorch CUDA cache")
            
            # Run custom cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    result = callback()
                    if result:
                        cleanup_results['actions_taken'].append(f"Custom cleanup: {result}")
                except Exception as e:
                    cleanup_results['errors'].append(f"Cleanup callback error: {e}")
                    logger.warning(f"Cleanup callback failed: {e}")
            
        except Exception as e:
            cleanup_results['errors'].append(f"Cleanup error: {e}")
            logger.error(f"Memory cleanup error: {e}")
        
        memory_after = self.get_memory_info()
        cleanup_results['memory_after'] = memory_after
        
        # Calculate savings
        system_saved = memory_before.system_used_gb - memory_after.system_used_gb
        process_saved = memory_before.process_memory_gb - memory_after.process_memory_gb
        
        cleanup_results['system_memory_saved_gb'] = system_saved
        cleanup_results['process_memory_saved_gb'] = process_saved
        
        if memory_before.gpu_allocated_gb and memory_after.gpu_allocated_gb:
            gpu_saved = memory_before.gpu_allocated_gb - memory_after.gpu_allocated_gb
            cleanup_results['gpu_memory_saved_gb'] = gpu_saved
        
        logger.info(f"Memory cleanup completed. Saved {system_saved:.2f} GB system, {process_saved:.2f} GB process memory")
        
        return cleanup_results
    
    def start_monitoring(self, interval: float = 30.0):
        """Start continuous memory monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            logger.warning("Memory monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started memory monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
        
        logger.info("Stopped memory monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                status = self.check_memory_usage()
                
                # Log warnings (with cooldown)
                current_time = time.time()
                if status['warnings'] and (current_time - self.last_warning_time) > self.warning_cooldown:
                    for warning in status['warnings']:
                        logger.warning(warning)
                    self.last_warning_time = current_time
                
                # Automatic cleanup if critical
                if status['critical'] and status['should_cleanup']:
                    logger.error("Critical memory usage detected, performing automatic cleanup")
                    cleanup_results = self.cleanup_memory()
                    
                    # Log cleanup results
                    for action in cleanup_results['actions_taken']:
                        logger.info(f"Cleanup action: {action}")
                    
                    for error in cleanup_results['errors']:
                        logger.error(f"Cleanup error: {error}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(interval)
    
    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add a cleanup callback function.
        
        Args:
            callback: Function to call during cleanup
        """
        self.cleanup_callbacks.append(callback)


@contextmanager
def memory_management(threshold_gb: Optional[float] = None,
                     cleanup_on_exit: bool = True):
    """Context manager for automatic memory management.
    
    Args:
        threshold_gb: Optional memory threshold in GB
        cleanup_on_exit: Whether to cleanup memory on exit
    """
    monitor = MemoryMonitor()
    
    memory_before = monitor.get_memory_info()
    logger.debug(f"Entering memory managed context. Memory: {memory_before.process_memory_gb:.2f} GB")
    
    try:
        yield monitor
        
    finally:
        memory_after = monitor.get_memory_info()
        memory_increase = memory_after.process_memory_gb - memory_before.process_memory_gb
        
        logger.debug(f"Exiting memory managed context. Memory change: {memory_increase:+.2f} GB")
        
        # Check if cleanup is needed
        if threshold_gb and memory_after.process_memory_gb > threshold_gb:
            logger.info(f"Memory usage {memory_after.process_memory_gb:.2f} GB exceeds threshold {threshold_gb:.2f} GB")
            cleanup_on_exit = True
        
        if cleanup_on_exit:
            monitor.cleanup_memory()


def check_gpu_memory_available(required_gb: float, operation: str = "operation") -> bool:
    """Check if enough GPU memory is available.
    
    Args:
        required_gb: Required memory in GB
        operation: Description of the operation
        
    Returns:
        True if enough memory is available
        
    Raises:
        GPUError: If not enough GPU memory is available
    """
    if not torch.cuda.is_available():
        raise GPUError(operation, "CUDA not available")
    
    try:
        available_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
        
        if available_gb < required_gb:
            memory_info = {
                'required_gb': required_gb,
                'available_gb': available_gb,
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
            raise GPUError(
                operation=operation,
                cuda_error=f"Insufficient GPU memory: need {required_gb:.2f}GB, have {available_gb:.2f}GB",
                memory_info=memory_info
            )
        
        return True
        
    except Exception as e:
        if isinstance(e, GPUError):
            raise
        raise GPUError(operation, f"Failed to check GPU memory: {e}")


def estimate_tensor_memory(shape: tuple, dtype: torch.dtype = torch.float32) -> float:
    """Estimate memory usage of a tensor in GB.
    
    Args:
        shape: Tensor shape
        dtype: Tensor data type
        
    Returns:
        Estimated memory usage in GB
    """
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    # Bytes per element based on dtype
    bytes_per_element = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.int16: 2,
        torch.int8: 1,
        torch.bool: 1,
    }.get(dtype, 4)  # Default to 4 bytes (float32)
    
    total_bytes = num_elements * bytes_per_element
    return total_bytes / (1024**3)  # Convert to GB


def safe_cuda_operation(operation: Callable, operation_name: str = "CUDA operation", max_retries: int = 3):
    """Safely execute a CUDA operation with error handling and retries.
    
    Args:
        operation: Function to execute
        operation_name: Name of the operation for logging
        max_retries: Maximum number of retries
        
    Returns:
        Result of the operation
        
    Raises:
        GPUError: If operation fails after all retries
    """
    for attempt in range(max_retries):
        try:
            return operation()
        
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"{operation_name} failed with CUDA OOM (attempt {attempt + 1}/{max_retries})")
            
            if attempt < max_retries - 1:
                # Try to free memory and retry
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Cleared CUDA cache and ran garbage collection")
            else:
                # Final attempt failed
                raise GPUError(
                    operation=operation_name,
                    cuda_error=f"CUDA out of memory after {max_retries} attempts: {e}"
                )
        
        except Exception as e:
            logger.error(f"{operation_name} failed with error: {e}")
            if attempt >= max_retries - 1:
                raise GPUError(
                    operation=operation_name,
                    cuda_error=f"Operation failed after {max_retries} attempts: {e}"
                )


# Global memory monitor instance
_global_memory_monitor = None


def get_global_memory_monitor() -> MemoryMonitor:
    """Get the global memory monitor instance."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor()
    return _global_memory_monitor


def start_global_memory_monitoring(interval: float = 30.0):
    """Start global memory monitoring."""
    monitor = get_global_memory_monitor()
    monitor.start_monitoring(interval)


def stop_global_memory_monitoring():
    """Stop global memory monitoring."""
    monitor = get_global_memory_monitor()
    monitor.stop_monitoring()