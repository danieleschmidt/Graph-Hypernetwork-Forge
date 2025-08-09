"""Performance optimization utilities."""

import time
import functools
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ModelProfiler:
    """Profiler for tracking model performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        if name not in self.timers:
            raise ValueError(f"Timer {name} not started")
        
        duration = time.time() - self.timers[name]
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        del self.timers[name]
        return duration
    
    def get_average(self, name: str) -> float:
        """Get average time for an operation."""
        if name not in self.metrics:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.timers.clear()
    
    def report(self) -> Dict[str, float]:
        """Generate performance report."""
        return {
            name: self.get_average(name) 
            for name in self.metrics
        }


def profile_function(name: str, profiler: Optional[ModelProfiler] = None):
    """Decorator to profile function execution time."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal profiler
            if profiler is None:
                # Use global profiler if none provided
                profiler = getattr(wrapper, '_profiler', ModelProfiler())
                wrapper._profiler = profiler
            
            profiler.start_timer(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_timer(name)
        
        return wrapper
    return decorator


class BatchProcessor:
    """Efficient batch processing for large-scale inference."""
    
    def __init__(self, batch_size: int = 32, device: str = "cpu"):
        self.batch_size = batch_size
        self.device = device
    
    def process_in_batches(
        self, 
        model: nn.Module, 
        data: torch.Tensor,
        process_fn: Callable = None
    ):
        """Process data in batches to manage memory."""
        model.eval()
        results = []
        
        with torch.no_grad():
            for i in range(0, data.size(0), self.batch_size):
                batch = data[i:i+self.batch_size].to(self.device)
                
                if process_fn:
                    batch_result = process_fn(model, batch)
                else:
                    batch_result = model(batch)
                
                results.append(batch_result.cpu())
        
        return torch.cat(results, dim=0)


class WeightCache:
    """Cache for generated weights to avoid recomputation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
    
    def _generate_key(self, texts: list, input_dim: int, output_dim: int) -> str:
        """Generate cache key from inputs."""
        text_hash = str(hash(tuple(texts)))
        return f"{text_hash}_{input_dim}_{output_dim}"
    
    def get(self, texts: list, input_dim: int, output_dim: int):
        """Get cached weights if available."""
        key = self._generate_key(texts, input_dim, output_dim)
        
        if key in self.cache:
            # Move to end (most recent access)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        return None
    
    def put(self, texts: list, input_dim: int, output_dim: int, weights):
        """Cache generated weights."""
        key = self._generate_key(texts, input_dim, output_dim)
        
        # Remove oldest if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = weights
        if key not in self.access_order:
            self.access_order.append(key)
    
    def clear(self):
        """Clear all cached weights."""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class AdaptiveDropout(nn.Module):
    """Adaptive dropout that adjusts based on training phase."""
    
    def __init__(self, p: float = 0.1, min_p: float = 0.01, max_p: float = 0.5):
        super().__init__()
        self.base_p = p
        self.min_p = min_p
        self.max_p = max_p
        self.current_p = p
    
    def set_dropout_rate(self, epoch: int, total_epochs: int):
        """Adjust dropout rate based on training progress."""
        # Higher dropout early in training, lower later
        progress = epoch / total_epochs
        adaptive_factor = 1.0 - progress * 0.5  # Reduce by up to 50%
        self.current_p = max(self.min_p, 
                           min(self.max_p, self.base_p * adaptive_factor))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.dropout(x, self.current_p, self.training)


class GradientClipping:
    """Advanced gradient clipping utilities."""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.grad_norms = []
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return the norm."""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, self.norm_type
        )
        
        self.grad_norms.append(total_norm.item())
        return total_norm.item()
    
    def adaptive_clip(self, model: nn.Module, percentile: float = 90) -> float:
        """Adaptive clipping based on gradient history."""
        if len(self.grad_norms) > 10:
            # Use percentile of recent gradient norms as clipping threshold
            recent_norms = self.grad_norms[-100:]  # Last 100 updates
            adaptive_max = torch.quantile(torch.tensor(recent_norms), 
                                        percentile / 100.0)
            adaptive_max = float(adaptive_max)
        else:
            adaptive_max = self.max_norm
        
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), adaptive_max, self.norm_type
        )
        
        self.grad_norms.append(total_norm.item())
        return total_norm.item()


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def enable_memory_efficient_attention():
        """Enable memory-efficient attention if available."""
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
    
    @staticmethod
    def optimize_model_memory(model: nn.Module):
        """Apply memory optimizations to model."""
        # Enable gradient checkpointing for large models
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Set models to use memory format
        if torch.cuda.is_available():
            model = model.to(memory_format=torch.channels_last)
        
        return model
    
    @staticmethod
    def clear_cache():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global profiler instance
global_profiler = ModelProfiler()