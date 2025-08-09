"""Advanced monitoring and metrics collection."""

import time
import psutil
import torch
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import json


class SystemMonitor:
    """Monitor system resources during training/inference."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.monitoring = False
        self.monitor_thread = None
        self.interval = 1.0  # seconds
    
    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring system resources."""
        self.interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            timestamp = time.time()
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            self.metrics_history['timestamp'].append(timestamp)
            self.metrics_history['cpu_percent'].append(cpu_percent)
            self.metrics_history['memory_percent'].append(memory.percent)
            self.metrics_history['memory_available_gb'].append(
                memory.available / (1024**3)
            )
            
            # GPU metrics if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                
                self.metrics_history['gpu_memory_gb'].append(gpu_memory)
                self.metrics_history['gpu_memory_reserved_gb'].append(gpu_memory_reserved)
                self.metrics_history['gpu_utilization'].append(gpu_util)
            
            time.sleep(self.interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        metrics = {}
        
        # CPU and Memory
        metrics['cpu_percent'] = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_available_gb'] = memory.available / (1024**3)
        
        # GPU metrics
        if torch.cuda.is_available():
            metrics['gpu_memory_gb'] = torch.cuda.memory_allocated() / (1024**3)
            metrics['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            metrics['gpu_count'] = torch.cuda.device_count()
        
        return metrics
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of monitored metrics."""
        summary = {}
        
        for metric_name, values in self.metrics_history.items():
            if metric_name == 'timestamp' or len(values) == 0:
                continue
            
            values_list = list(values)
            summary[metric_name] = {
                'mean': sum(values_list) / len(values_list),
                'min': min(values_list),
                'max': max(values_list),
                'current': values_list[-1] if values_list else 0.0,
                'count': len(values_list)
            }
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export metrics history to file."""
        export_data = {
            metric_name: list(values)
            for metric_name, values in self.metrics_history.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


class TrainingMonitor:
    """Monitor training progress and performance."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_times = []
        self.batch_times = []
        self.current_epoch = 0
        self.epoch_start_time = None
    
    def start_epoch(self, epoch: int):
        """Mark the start of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
    
    def end_epoch(self):
        """Mark the end of an epoch."""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            self.epoch_start_time = None
    
    def log_batch_time(self, batch_time: float):
        """Log batch processing time."""
        self.batch_times.append(batch_time)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a training metric."""
        if step is None:
            step = len(self.metrics[name])
        
        self.metrics[name].append({'step': step, 'value': value})
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the latest value for each metric."""
        return {
            name: values[-1]['value'] if values else 0.0
            for name, values in self.metrics.items()
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        stats = {
            'total_epochs': self.current_epoch,
            'total_batches': len(self.batch_times),
        }
        
        if self.epoch_times:
            stats['avg_epoch_time'] = sum(self.epoch_times) / len(self.epoch_times)
            stats['total_training_time'] = sum(self.epoch_times)
        
        if self.batch_times:
            recent_batches = self.batch_times[-100:]  # Last 100 batches
            stats['avg_batch_time'] = sum(recent_batches) / len(recent_batches)
            stats['batches_per_second'] = 1.0 / stats['avg_batch_time']
        
        # Metric trends
        for name, values in self.metrics.items():
            if len(values) >= 2:
                recent_values = [v['value'] for v in values[-10:]]  # Last 10 values
                stats[f'{name}_trend'] = recent_values[-1] - recent_values[0]
        
        return stats


class ModelAnalyzer:
    """Analyze model architecture and parameters."""
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    
    @staticmethod
    def analyze_gradients(model: torch.nn.Module) -> Dict[str, Any]:
        """Analyze gradient statistics."""
        grad_stats = {}
        total_grad_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                param_count += 1
                
                grad_stats[name] = {
                    'norm': grad_norm,
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'max': param.grad.max().item(),
                    'min': param.grad.min().item()
                }
        
        grad_stats['total_norm'] = (total_grad_norm ** 0.5) if param_count > 0 else 0.0
        grad_stats['param_count'] = param_count
        
        return grad_stats
    
    @staticmethod
    def get_model_size(model: torch.nn.Module) -> Dict[str, float]:
        """Calculate model size in memory."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'parameters_mb': param_size / (1024 ** 2),
            'buffers_mb': buffer_size / (1024 ** 2),
            'total_mb': total_size / (1024 ** 2)
        }
    
    @staticmethod
    def profile_inference_speed(
        model: torch.nn.Module, 
        sample_input: torch.Tensor, 
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Profile model inference speed."""
        model.eval()
        device = next(model.parameters()).device
        sample_input = sample_input.to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(sample_input)
        
        # Timing runs
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(sample_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end = time.time()
                times.append(end - start)
        
        return {
            'mean_time_ms': (sum(times) / len(times)) * 1000,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
            'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
            'throughput_hz': 1.0 / (sum(times) / len(times))
        }


# Global monitoring instances
system_monitor = SystemMonitor()
training_monitor = TrainingMonitor()