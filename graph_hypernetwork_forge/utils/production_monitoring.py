"""Production-grade monitoring and observability for Graph Hypernetwork Forge."""

import asyncio
import json
import os
import psutil
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock Prometheus classes
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return contextlib.nullcontext()
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
    CollectorRegistry = None
    def generate_latest(*args, **kwargs): return b""

try:
    from .logging_utils import get_logger
    from .exceptions import GraphHypernetworkError
    ENHANCED_FEATURES = True
except ImportError:
    def get_logger(name): return logging.getLogger(name)
    class GraphHypernetworkError(Exception): pass
    ENHANCED_FEATURES = False

logger = get_logger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for model operations."""
    operation: str
    duration_ms: float
    memory_used_mb: float
    gpu_memory_used_mb: float = 0.0
    input_size: int = 0
    output_size: int = 0
    batch_size: int = 1
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class SystemResourceMetrics:
    """System resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_memory_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_temperature_c: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Comprehensive metrics collection and aggregation."""
    
    def __init__(
        self,
        collection_interval: float = 10.0,
        max_metrics_history: int = 10000,
        export_prometheus: bool = True,
        metrics_file: Optional[str] = None,
    ):
        """Initialize metrics collector.
        
        Args:
            collection_interval: Interval for automatic metrics collection
            max_metrics_history: Maximum metrics to keep in memory
            export_prometheus: Whether to export Prometheus metrics
            metrics_file: Optional file to write metrics to
        """
        self.collection_interval = collection_interval
        self.max_metrics_history = max_metrics_history
        self.export_prometheus = export_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_file = Path(metrics_file) if metrics_file else None
        
        # Metrics storage
        self.performance_metrics = deque(maxlen=max_metrics_history)
        self.resource_metrics = deque(maxlen=max_metrics_history)
        self.error_metrics = defaultdict(int)
        
        # Prometheus metrics
        if self.export_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        
        # Background collection
        self._running = False
        self._collection_thread = None
        
        # Locks for thread safety
        self._metrics_lock = threading.Lock()
        
        logger.info(f"Metrics collector initialized (Prometheus: {self.export_prometheus})")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        if not self.export_prometheus:
            return
        
        # Model performance metrics
        self.model_duration_histogram = Histogram(
            'hypergnn_model_operation_duration_seconds',
            'Time spent on model operations',
            ['operation', 'success'],
            registry=self.registry
        )
        
        self.model_memory_gauge = Gauge(
            'hypergnn_model_memory_usage_bytes',
            'Memory usage during model operations',
            ['operation'],
            registry=self.registry
        )
        
        self.model_operations_counter = Counter(
            'hypergnn_model_operations_total',
            'Total model operations',
            ['operation', 'success'],
            registry=self.registry
        )
        
        # System resource metrics
        self.cpu_usage_gauge = Gauge(
            'hypergnn_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage_gauge = Gauge(
            'hypergnn_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.gpu_memory_gauge = Gauge(
            'hypergnn_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            registry=self.registry
        )
        
        self.gpu_utilization_gauge = Gauge(
            'hypergnn_gpu_utilization_percent',
            'GPU utilization percentage',
            registry=self.registry
        )
        
        # Error tracking
        self.error_counter = Counter(
            'hypergnn_errors_total',
            'Total errors by type',
            ['error_type', 'component'],
            registry=self.registry
        )
    
    def record_model_performance(self, metrics: ModelPerformanceMetrics):
        """Record model performance metrics."""
        with self._metrics_lock:
            self.performance_metrics.append(metrics)
            
            if self.export_prometheus:
                # Update Prometheus metrics
                self.model_duration_histogram.labels(
                    operation=metrics.operation,
                    success=str(metrics.success).lower()
                ).observe(metrics.duration_ms / 1000.0)
                
                self.model_memory_gauge.labels(
                    operation=metrics.operation
                ).set(metrics.memory_used_mb * 1024 * 1024)
                
                self.model_operations_counter.labels(
                    operation=metrics.operation,
                    success=str(metrics.success).lower()
                ).inc()
            
            # Write to file if configured
            if self.metrics_file:
                self._write_metrics_to_file(metrics)
        
        logger.debug(f"Recorded model performance: {metrics.operation} "
                    f"({metrics.duration_ms:.1f}ms, {metrics.memory_used_mb:.1f}MB)")
    
    def record_system_resources(self, metrics: SystemResourceMetrics):
        """Record system resource metrics."""
        with self._metrics_lock:
            self.resource_metrics.append(metrics)
            
            if self.export_prometheus:
                # Update Prometheus metrics
                self.cpu_usage_gauge.set(metrics.cpu_percent)
                self.memory_usage_gauge.set(metrics.memory_used_mb * 1024 * 1024)
                
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    self.gpu_memory_gauge.set(metrics.gpu_memory_percent)
                    self.gpu_utilization_gauge.set(metrics.gpu_utilization_percent)
        
        logger.debug(f"Recorded system resources: CPU {metrics.cpu_percent:.1f}%, "
                    f"Memory {metrics.memory_percent:.1f}%")
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        with self._metrics_lock:
            self.error_metrics[f"{component}:{error_type}"] += 1
            
            if self.export_prometheus:
                self.error_counter.labels(
                    error_type=error_type,
                    component=component
                ).inc()
        
        logger.warning(f"Recorded error: {component}:{error_type}")
    
    def start_collection(self):
        """Start automatic metrics collection."""
        if self._running:
            return
        
        self._running = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
        logger.info("Started automatic metrics collection")
    
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join()
        logger.info("Stopped automatic metrics collection")
    
    def _collection_loop(self):
        """Background metrics collection loop."""
        while self._running:
            try:
                # Collect system resource metrics
                resource_metrics = self._collect_system_resources()
                self.record_system_resources(resource_metrics)
                
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_resources(self) -> SystemResourceMetrics:
        """Collect current system resource metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network (delta from last measurement)
        net_io = psutil.net_io_counters()
        
        metrics = SystemResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024 * 1024 * 1024),
            network_bytes_sent=net_io.bytes_sent,
            network_bytes_recv=net_io.bytes_recv,
        )
        
        # GPU metrics if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = gpu_memory.get('allocated_bytes.all.current', 0)
                
                metrics.gpu_memory_percent = (allocated_memory / total_memory) * 100
                metrics.gpu_utilization_percent = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                
                # Try to get temperature (nvidia-ml-py would be needed for this)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics.gpu_temperature_c = temp
                except (ImportError, Exception):
                    metrics.gpu_temperature_c = 0.0
                    
            except Exception as e:
                logger.debug(f"Could not collect GPU metrics: {e}")
        
        return metrics
    
    def _write_metrics_to_file(self, metrics):
        """Write metrics to file."""
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.metrics_file, 'a') as f:
                json.dump(asdict(metrics), f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Failed to write metrics to file: {e}")
    
    def get_performance_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics summary."""
        with self._metrics_lock:
            metrics = list(self.performance_metrics)
        
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        
        if not metrics:
            return {}
        
        durations = [m.duration_ms for m in metrics if m.success]
        memory_usage = [m.memory_used_mb for m in metrics if m.success]
        
        if not durations:
            return {'error': 'No successful operations found'}
        
        return {
            'operation': operation or 'all',
            'total_operations': len(metrics),
            'successful_operations': len(durations),
            'success_rate': len(durations) / len(metrics) * 100,
            'duration_ms': {
                'mean': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'p95': sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0],
                'p99': sorted(durations)[int(len(durations) * 0.99)] if len(durations) > 1 else durations[0],
            },
            'memory_mb': {
                'mean': sum(memory_usage) / len(memory_usage),
                'min': min(memory_usage),
                'max': max(memory_usage),
            },
            'time_window': {
                'start': min(m.timestamp for m in metrics),
                'end': max(m.timestamp for m in metrics),
            }
        }
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get system resource metrics summary."""
        with self._metrics_lock:
            metrics = list(self.resource_metrics)
        
        if not metrics:
            return {}
        
        latest = metrics[-1]
        
        return {
            'current': {
                'cpu_percent': latest.cpu_percent,
                'memory_percent': latest.memory_percent,
                'memory_used_mb': latest.memory_used_mb,
                'disk_usage_percent': latest.disk_usage_percent,
                'gpu_memory_percent': latest.gpu_memory_percent,
                'gpu_utilization_percent': latest.gpu_utilization_percent,
            },
            'averages_last_hour': self._calculate_resource_averages(metrics, 3600),
            'peak_usage': {
                'cpu_percent': max(m.cpu_percent for m in metrics),
                'memory_percent': max(m.memory_percent for m in metrics),
                'gpu_memory_percent': max(m.gpu_memory_percent for m in metrics),
            }
        }
    
    def _calculate_resource_averages(self, metrics: List[SystemResourceMetrics], window_seconds: int) -> Dict[str, float]:
        """Calculate resource averages within a time window."""
        cutoff_time = time.time() - window_seconds
        recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        return {
            'cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'gpu_memory_percent': sum(m.gpu_memory_percent for m in recent_metrics) / len(recent_metrics),
            'gpu_utilization_percent': sum(m.gpu_utilization_percent for m in recent_metrics) / len(recent_metrics),
        }
    
    def get_prometheus_metrics(self) -> bytes:
        """Get Prometheus-formatted metrics."""
        if not self.export_prometheus:
            return b"Prometheus not available"
        
        return generate_latest(self.registry)
    
    def export_metrics(self, filename: str):
        """Export all metrics to JSON file."""
        with self._metrics_lock:
            data = {
                'performance_metrics': [asdict(m) for m in self.performance_metrics],
                'resource_metrics': [asdict(m) for m in self.resource_metrics],
                'error_metrics': dict(self.error_metrics),
                'export_timestamp': time.time(),
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to {filename}")


class PerformanceProfiler:
    """Advanced performance profiling for model operations."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize performance profiler.
        
        Args:
            metrics_collector: Optional metrics collector to use
        """
        self.metrics_collector = metrics_collector
        self.active_profiles = {}
        self._lock = threading.Lock()
        
        logger.info("Performance profiler initialized")
    
    @contextmanager
    def profile_operation(
        self,
        operation: str,
        batch_size: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for profiling operations.
        
        Args:
            operation: Name of the operation
            batch_size: Batch size for the operation
            metadata: Additional metadata
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory_usage()
        
        profile_id = f"{operation}_{threading.current_thread().ident}_{start_time}"
        
        with self._lock:
            self.active_profiles[profile_id] = {
                'operation': operation,
                'start_time': start_time,
                'start_memory': start_memory,
                'start_gpu_memory': start_gpu_memory,
                'batch_size': batch_size,
                'metadata': metadata or {},
            }
        
        try:
            yield
            success = True
            error_message = None
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory_usage()
            
            with self._lock:
                profile_data = self.active_profiles.pop(profile_id, {})
            
            # Calculate metrics
            duration_ms = (end_time - start_time) * 1000
            memory_used_mb = max(0, end_memory - start_memory)
            gpu_memory_used_mb = max(0, end_gpu_memory - start_gpu_memory)
            
            # Create performance metrics
            metrics = ModelPerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                memory_used_mb=memory_used_mb,
                gpu_memory_used_mb=gpu_memory_used_mb,
                batch_size=batch_size,
                success=success,
                error_message=error_message,
                metadata=profile_data.get('metadata', {}),
            )
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_model_performance(metrics)
                if not success:
                    self.metrics_collector.record_error(
                        error_type=type(Exception).__name__,
                        component=operation
                    )
            
            logger.debug(f"Profiled {operation}: {duration_ms:.1f}ms, "
                        f"{memory_used_mb:.1f}MB RAM, {gpu_memory_used_mb:.1f}MB GPU")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            return 0.0


class AlertManager:
    """Alert management for critical system conditions."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector to monitor
        """
        self.metrics_collector = metrics_collector
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Default alert rules
        self._setup_default_rules()
        
        logger.info("Alert manager initialized with default rules")
    
    def _setup_default_rules(self):
        """Setup default alerting rules."""
        
        # High CPU usage
        self.add_alert_rule(
            name="high_cpu_usage",
            condition=lambda metrics: metrics.cpu_percent > 90,
            severity="warning",
            message="High CPU usage detected: {cpu_percent:.1f}%"
        )
        
        # High memory usage
        self.add_alert_rule(
            name="high_memory_usage", 
            condition=lambda metrics: metrics.memory_percent > 85,
            severity="warning",
            message="High memory usage detected: {memory_percent:.1f}%"
        )
        
        # Low disk space
        self.add_alert_rule(
            name="low_disk_space",
            condition=lambda metrics: metrics.disk_usage_percent > 90,
            severity="critical",
            message="Low disk space: {disk_usage_percent:.1f}% used"
        )
        
        # GPU memory exhaustion
        self.add_alert_rule(
            name="gpu_memory_exhaustion",
            condition=lambda metrics: metrics.gpu_memory_percent > 95,
            severity="critical",
            message="GPU memory nearly exhausted: {gpu_memory_percent:.1f}%"
        )
    
    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[SystemResourceMetrics], bool],
        severity: str,
        message: str,
        cooldown_seconds: int = 300  # 5 minutes
    ):
        """Add custom alert rule.
        
        Args:
            name: Unique name for the alert
            condition: Function that takes SystemResourceMetrics and returns bool
            severity: Alert severity (info, warning, critical)
            message: Alert message template
            cooldown_seconds: Cooldown period before re-alerting
        """
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'message': message,
            'cooldown_seconds': cooldown_seconds,
        }
        
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name} ({severity})")
    
    def check_alerts(self):
        """Check all alert conditions against current metrics."""
        if not self.metrics_collector.resource_metrics:
            return
        
        latest_metrics = self.metrics_collector.resource_metrics[-1]
        current_time = time.time()
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](latest_metrics):
                    self._trigger_alert(rule, latest_metrics, current_time)
                else:
                    self._resolve_alert(rule['name'], current_time)
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    def _trigger_alert(self, rule: Dict, metrics: SystemResourceMetrics, current_time: float):
        """Trigger an alert."""
        alert_name = rule['name']
        
        # Check cooldown
        if alert_name in self.active_alerts:
            last_alert_time = self.active_alerts[alert_name]['last_triggered']
            if current_time - last_alert_time < rule['cooldown_seconds']:
                return  # Still in cooldown
        
        # Format message
        message = rule['message'].format(**asdict(metrics))
        
        alert = {
            'name': alert_name,
            'severity': rule['severity'],
            'message': message,
            'triggered_at': current_time,
            'last_triggered': current_time,
            'trigger_count': self.active_alerts.get(alert_name, {}).get('trigger_count', 0) + 1,
            'metrics_snapshot': asdict(metrics),
        }
        
        self.active_alerts[alert_name] = alert
        self.alert_history.append(alert.copy())
        
        # Log alert
        log_level = {
            'info': logger.info,
            'warning': logger.warning,
            'critical': logger.critical,
        }.get(rule['severity'], logger.warning)
        
        log_level(f"ALERT [{rule['severity'].upper()}] {alert_name}: {message}")
    
    def _resolve_alert(self, alert_name: str, current_time: float):
        """Resolve an active alert."""
        if alert_name in self.active_alerts:
            alert = self.active_alerts.pop(alert_name)
            
            resolution = {
                'name': alert_name,
                'severity': 'info',
                'message': f"Alert resolved: {alert_name}",
                'resolved_at': current_time,
                'duration_seconds': current_time - alert['triggered_at'],
            }
            
            self.alert_history.append(resolution)
            logger.info(f"RESOLVED: {alert_name} after {resolution['duration_seconds']:.1f}s")
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all currently active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history."""
        return list(self.alert_history)[-limit:]


# Global instances
_global_metrics_collector = None
_global_profiler = None
_global_alert_manager = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector

def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    global _global_profiler
    if _global_profiler is None:
        collector = get_metrics_collector()
        _global_profiler = PerformanceProfiler(collector)
    return _global_profiler

def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    global _global_alert_manager
    if _global_alert_manager is None:
        collector = get_metrics_collector()
        _global_alert_manager = AlertManager(collector)
    return _global_alert_manager

def monitor_performance(operation: str, **kwargs):
    """Decorator for automatic performance monitoring."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs):
            profiler = get_performance_profiler()
            
            with profiler.profile_operation(operation, **kwargs):
                return func(*args, **func_kwargs)
        
        return wrapper
    return decorator