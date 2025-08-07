"""Production profiling and monitoring utilities."""

import time
import psutil
import torch
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import functools


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_time: float = 0.0
    cpu_time: float = 0.0
    gpu_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    gpu_memory_current_mb: float = 0.0
    call_count: int = 0
    error_count: int = 0
    throughput: float = 0.0  # items/second
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Production-ready performance profiler."""
    
    def __init__(
        self, 
        enabled: bool = True,
        sample_rate: float = 1.0,
        history_size: int = 1000
    ):
        """Initialize profiler.
        
        Args:
            enabled: Whether profiling is enabled
            sample_rate: Sampling rate (0.0 to 1.0)
            history_size: Maximum number of measurements to keep
        """
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.history_size = history_size
        
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self._lock = threading.RLock()
        
        # System monitoring
        self._process = psutil.Process()
        self._start_time = time.time()
        
    def should_profile(self) -> bool:
        """Determine if current operation should be profiled."""
        if not self.enabled:
            return False
        return np.random.random() < self.sample_rate
    
    @contextmanager
    def profile(self, operation_name: str, **context):
        """Profile a code block or function.
        
        Args:
            operation_name: Name of the operation being profiled
            **context: Additional context information
            
        Example:
            with profiler.profile("text_encoding"):
                embeddings = model.encode(texts)
        """
        if not self.should_profile():
            yield
            return
        
        start_time = time.time()
        start_cpu_time = self._process.cpu_times().user
        start_memory = self._get_memory_usage()
        
        # GPU metrics if available
        start_gpu_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0
        
        error_occurred = False
        try:
            yield
        except Exception as e:
            error_occurred = True
            raise
        finally:
            end_time = time.time()
            end_cpu_time = self._process.cpu_times().user
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0
            
            # Calculate metrics
            total_time = end_time - start_time
            cpu_time = end_cpu_time - start_cpu_time
            memory_delta = end_memory - start_memory
            gpu_memory_delta = end_gpu_memory - start_gpu_memory
            
            # Update metrics
            self._update_metrics(
                operation_name,
                total_time=total_time,
                cpu_time=cpu_time,
                memory_delta=memory_delta,
                gpu_memory_delta=gpu_memory_delta,
                error_occurred=error_occurred,
                context=context
            )
    
    def profile_function(self, operation_name: str = None):
        """Decorator to profile a function.
        
        Args:
            operation_name: Name for the operation (uses function name if None)
        """
        def decorator(func):
            name = operation_name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile(name, function=func.__name__):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _update_metrics(
        self,
        operation_name: str,
        total_time: float,
        cpu_time: float,
        memory_delta: float,
        gpu_memory_delta: float,
        error_occurred: bool,
        context: Dict[str, Any]
    ):
        """Update performance metrics for an operation."""
        with self._lock:
            if operation_name not in self._metrics:
                self._metrics[operation_name] = PerformanceMetrics()
            
            metrics = self._metrics[operation_name]
            
            # Update counters
            metrics.call_count += 1
            if error_occurred:
                metrics.error_count += 1
            
            # Update timing
            metrics.total_time += total_time
            metrics.cpu_time += cpu_time
            
            # Update memory (track peaks and current)
            current_memory = self._get_memory_usage()
            current_gpu_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0
            
            metrics.memory_current_mb = current_memory
            metrics.memory_peak_mb = max(metrics.memory_peak_mb, current_memory)
            
            if torch.cuda.is_available():
                metrics.gpu_memory_current_mb = current_gpu_memory
                metrics.gpu_memory_peak_mb = max(metrics.gpu_memory_peak_mb, current_gpu_memory)
            
            # Calculate throughput (if applicable)
            if 'item_count' in context:
                items_processed = context['item_count']
                metrics.throughput = items_processed / total_time
            
            # Store additional context
            for key, value in context.items():
                if key not in metrics.additional_metrics:
                    metrics.additional_metrics[key] = []
                metrics.additional_metrics[key].append(value)
            
            # Add to history
            measurement = {
                'timestamp': time.time(),
                'total_time': total_time,
                'cpu_time': cpu_time,
                'memory_delta': memory_delta,
                'gpu_memory_delta': gpu_memory_delta,
                'error': error_occurred,
                **context
            }
            self._history[operation_name].append(measurement)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
            return 0.0
        except:
            return 0.0
    
    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics.
        
        Args:
            operation_name: Specific operation to get metrics for (all if None)
            
        Returns:
            Dictionary of performance metrics
        """
        with self._lock:
            if operation_name:
                if operation_name in self._metrics:
                    return self._format_metrics(operation_name, self._metrics[operation_name])
                else:
                    return {}
            else:
                return {
                    name: self._format_metrics(name, metrics)
                    for name, metrics in self._metrics.items()
                }
    
    def _format_metrics(self, name: str, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Format metrics for output."""
        avg_time = metrics.total_time / max(metrics.call_count, 1)
        avg_cpu_time = metrics.cpu_time / max(metrics.call_count, 1)
        error_rate = metrics.error_count / max(metrics.call_count, 1)
        
        return {
            'operation': name,
            'call_count': metrics.call_count,
            'error_count': metrics.error_count,
            'error_rate': error_rate,
            'total_time': metrics.total_time,
            'average_time': avg_time,
            'average_cpu_time': avg_cpu_time,
            'memory_peak_mb': metrics.memory_peak_mb,
            'memory_current_mb': metrics.memory_current_mb,
            'gpu_memory_peak_mb': metrics.gpu_memory_peak_mb,
            'gpu_memory_current_mb': metrics.gpu_memory_current_mb,
            'throughput': metrics.throughput,
            'additional_metrics': metrics.additional_metrics
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            system_info = {
                'timestamp': time.time(),
                'uptime_seconds': time.time() - self._start_time,
                'cpu_percent': cpu_percent,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'process_memory_mb': self._get_memory_usage(),
            }
            
            if torch.cuda.is_available():
                system_info.update({
                    'gpu_available': True,
                    'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                    'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                    'gpu_device_count': torch.cuda.device_count(),
                })
            else:
                system_info['gpu_available'] = False
            
            return system_info
        except Exception as e:
            logger.warning(f"Error getting system info: {e}")
            return {'error': str(e)}
    
    def reset_metrics(self, operation_name: Optional[str] = None):
        """Reset performance metrics.
        
        Args:
            operation_name: Specific operation to reset (all if None)
        """
        with self._lock:
            if operation_name:
                if operation_name in self._metrics:
                    del self._metrics[operation_name]
                if operation_name in self._history:
                    self._history[operation_name].clear()
            else:
                self._metrics.clear()
                self._history.clear()
    
    def export_metrics(self, filepath: Path) -> None:
        """Export metrics to JSON file.
        
        Args:
            filepath: Path to export file
        """
        data = {
            'system_info': self.get_system_info(),
            'metrics': self.get_metrics(),
            'history': {
                name: list(history) for name, history in self._history.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def get_summary_report(self) -> str:
        """Generate a summary report of performance metrics."""
        metrics = self.get_metrics()
        system_info = self.get_system_info()
        
        report = []
        report.append("Performance Summary Report")
        report.append("=" * 50)
        
        # System information
        report.append(f"\nSystem Information:")
        report.append(f"  CPU Usage: {system_info.get('cpu_percent', 0):.1f}%")
        report.append(f"  Memory Usage: {system_info.get('memory_percent', 0):.1f}%")
        report.append(f"  Process Memory: {system_info.get('process_memory_mb', 0):.1f} MB")
        
        if system_info.get('gpu_available'):
            report.append(f"  GPU Memory: {system_info.get('gpu_memory_allocated_mb', 0):.1f} MB")
        
        # Operation metrics
        report.append(f"\nOperation Metrics:")
        for operation, data in metrics.items():
            report.append(f"  {operation}:")
            report.append(f"    Calls: {data['call_count']}")
            report.append(f"    Avg Time: {data['average_time']:.4f}s")
            report.append(f"    Error Rate: {data['error_rate']:.2%}")
            report.append(f"    Peak Memory: {data['memory_peak_mb']:.1f} MB")
            if data['throughput'] > 0:
                report.append(f"    Throughput: {data['throughput']:.2f} items/s")
        
        return "\n".join(report)


class ModelProfiler:
    """Specialized profiler for ML models."""
    
    def __init__(self, profiler: PerformanceProfiler = None):
        """Initialize model profiler.
        
        Args:
            profiler: Underlying performance profiler
        """
        self.profiler = profiler or PerformanceProfiler()
    
    def profile_inference(self, model_name: str = "model"):
        """Profile model inference."""
        return self.profiler.profile(f"{model_name}_inference")
    
    def profile_training_step(self, model_name: str = "model"):
        """Profile model training step."""
        return self.profiler.profile(f"{model_name}_training_step")
    
    def profile_data_loading(self, loader_name: str = "data_loader"):
        """Profile data loading."""
        return self.profiler.profile(f"{loader_name}_loading")
    
    def get_model_summary(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Get detailed model summary.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model summary information
        """
        try:
            from .model_utils import count_parameters, calculate_model_size
            
            total_params = count_parameters(model)
            trainable_params = count_parameters(model, trainable_only=True)
            model_size_mb = calculate_model_size(model)
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb,
                'device': str(next(model.parameters()).device),
                'dtype': str(next(model.parameters()).dtype),
            }
        except Exception as e:
            return {'error': str(e)}


# Global profiler instance
_global_profiler = None


def get_profiler(
    enabled: bool = True,
    sample_rate: float = 1.0
) -> PerformanceProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(enabled, sample_rate)
    return _global_profiler


def profile(operation_name: str):
    """Convenient decorator for profiling functions."""
    return get_profiler().profile_function(operation_name)


@contextmanager
def profile_context(operation_name: str, **context):
    """Convenient context manager for profiling code blocks."""
    with get_profiler().profile(operation_name, **context):
        yield


def benchmark_function(
    func: Callable,
    *args,
    num_runs: int = 10,
    warmup_runs: int = 3,
    **kwargs
) -> Dict[str, float]:
    """Benchmark a function with multiple runs.
    
    Args:
        func: Function to benchmark
        *args: Arguments for function
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        **kwargs: Keyword arguments for function
        
    Returns:
        Benchmark statistics
    """
    # Warmup runs
    for _ in range(warmup_runs):
        try:
            func(*args, **kwargs)
        except:
            pass
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        try:
            func(*args, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
        except Exception as e:
            times.append(float('inf'))  # Mark failed runs
    
    # Calculate statistics
    valid_times = [t for t in times if t != float('inf')]
    
    if not valid_times:
        return {
            'mean_time': float('inf'),
            'std_time': 0.0,
            'min_time': float('inf'),
            'max_time': float('inf'),
            'success_rate': 0.0
        }
    
    return {
        'mean_time': np.mean(valid_times),
        'std_time': np.std(valid_times),
        'min_time': np.min(valid_times),
        'max_time': np.max(valid_times),
        'success_rate': len(valid_times) / num_runs
    }