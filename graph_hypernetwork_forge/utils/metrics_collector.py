"""Advanced metrics collection system for comprehensive monitoring."""

import time
import threading
import queue
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import numpy as np
import torch
import psutil
from pathlib import Path
import sqlite3

from .logging_utils import get_logger
from .exceptions import GraphHypernetworkError

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: Union[float, int]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    mean: float
    std: float
    min_value: float
    max_value: float
    p50: float
    p95: float
    p99: float
    latest_value: float
    latest_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['latest_timestamp'] = self.latest_timestamp.isoformat()
        return result


class MetricsBuffer:
    """Thread-safe buffer for metrics collection."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add(self, metric: MetricPoint):
        """Add a metric to the buffer."""
        with self.lock:
            self.buffer.append(metric)
            
    def get_all(self, clear: bool = False) -> List[MetricPoint]:
        """Get all metrics from buffer."""
        with self.lock:
            metrics = list(self.buffer)
            if clear:
                self.buffer.clear()
            return metrics
            
    def get_recent(self, seconds: int = 60) -> List[MetricPoint]:
        """Get metrics from the last N seconds."""
        cutoff = datetime.now() - timedelta(seconds=seconds)
        with self.lock:
            return [m for m in self.buffer if m.timestamp >= cutoff]
            
    def size(self) -> int:
        """Get buffer size."""
        with self.lock:
            return len(self.buffer)


class PerformanceMetricsCollector:
    """Collect performance-related metrics."""
    
    def __init__(self):
        self.latency_buffer = deque(maxlen=1000)
        self.throughput_buffer = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        
    def record_latency(self, operation: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record operation latency."""
        metric = MetricPoint(
            name=f"{operation}_latency_ms",
            value=duration_ms,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata={"operation": operation}
        )
        self.latency_buffer.append(metric)
        return metric
        
    def record_throughput(self, operation: str, count: int, tags: Optional[Dict[str, str]] = None):
        """Record throughput metric."""
        metric = MetricPoint(
            name=f"{operation}_throughput",
            value=count,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata={"operation": operation}
        )
        self.throughput_buffer.append(metric)
        return metric
        
    def record_error(self, operation: str, error_type: str, tags: Optional[Dict[str, str]] = None):
        """Record error occurrence."""
        self.error_counts[f"{operation}_{error_type}"] += 1
        return MetricPoint(
            name=f"{operation}_errors_total",
            value=self.error_counts[f"{operation}_{error_type}"],
            timestamp=datetime.now(),
            tags={**(tags or {}), "error_type": error_type},
            metadata={"operation": operation, "error_type": error_type}
        )
        
    def record_request(self, operation: str, tags: Optional[Dict[str, str]] = None):
        """Record request count."""
        self.request_counts[operation] += 1
        return MetricPoint(
            name=f"{operation}_requests_total",
            value=self.request_counts[operation],
            timestamp=datetime.now(),
            tags=tags or {},
            metadata={"operation": operation}
        )
        
    def get_latency_stats(self, operation: str, minutes: int = 5) -> Optional[Dict[str, float]]:
        """Get latency statistics for an operation."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        recent_metrics = [
            m for m in self.latency_buffer 
            if m.metadata.get("operation") == operation and m.timestamp >= cutoff
        ]
        
        if not recent_metrics:
            return None
            
        values = [m.value for m in recent_metrics]
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
        
    def get_error_rate(self, operation: str, minutes: int = 5) -> float:
        """Get error rate for an operation."""
        total_requests = self.request_counts.get(operation, 0)
        if total_requests == 0:
            return 0.0
            
        total_errors = sum(
            count for op_error, count in self.error_counts.items()
            if op_error.startswith(f"{operation}_")
        )
        
        return total_errors / total_requests


class ResourceMetricsCollector:
    """Collect system resource metrics."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.collecting = False
        self.collection_thread = None
        self.metrics_buffer = MetricsBuffer()
        
    def start_collection(self):
        """Start collecting resource metrics."""
        if self.collecting:
            return
            
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started resource metrics collection")
        
    def stop_collection(self):
        """Stop collecting resource metrics."""
        if not self.collecting:
            return
            
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("Stopped resource metrics collection")
        
    def _collection_loop(self):
        """Main collection loop."""
        while self.collecting:
            timestamp = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics_buffer.add(MetricPoint(
                name="cpu_usage_percent",
                value=cpu_percent,
                timestamp=timestamp
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_buffer.add(MetricPoint(
                name="memory_usage_percent",
                value=memory.percent,
                timestamp=timestamp
            ))
            self.metrics_buffer.add(MetricPoint(
                name="memory_available_gb",
                value=memory.available / (1024**3),
                timestamp=timestamp
            ))
            self.metrics_buffer.add(MetricPoint(
                name="memory_used_gb",
                value=memory.used / (1024**3),
                timestamp=timestamp
            ))
            
            # Disk metrics
            try:
                disk_usage = psutil.disk_usage('/')
                self.metrics_buffer.add(MetricPoint(
                    name="disk_usage_percent",
                    value=(disk_usage.used / disk_usage.total) * 100,
                    timestamp=timestamp
                ))
                self.metrics_buffer.add(MetricPoint(
                    name="disk_free_gb",
                    value=disk_usage.free / (1024**3),
                    timestamp=timestamp
                ))
            except Exception as e:
                logger.warning(f"Failed to collect disk metrics: {e}")
            
            # GPU metrics
            if torch.cuda.is_available():
                try:
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.set_device(i)
                        
                        # Memory metrics
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        allocated_memory = torch.cuda.memory_allocated(i)
                        reserved_memory = torch.cuda.memory_reserved(i)
                        
                        self.metrics_buffer.add(MetricPoint(
                            name="gpu_memory_allocated_gb",
                            value=allocated_memory / (1024**3),
                            timestamp=timestamp,
                            tags={"gpu": str(i)}
                        ))
                        self.metrics_buffer.add(MetricPoint(
                            name="gpu_memory_reserved_gb",
                            value=reserved_memory / (1024**3),
                            timestamp=timestamp,
                            tags={"gpu": str(i)}
                        ))
                        self.metrics_buffer.add(MetricPoint(
                            name="gpu_memory_usage_percent",
                            value=(allocated_memory / total_memory) * 100,
                            timestamp=timestamp,
                            tags={"gpu": str(i)}
                        ))
                        
                        # Utilization (if available)
                        if hasattr(torch.cuda, 'utilization'):
                            gpu_util = torch.cuda.utilization(i)
                            self.metrics_buffer.add(MetricPoint(
                                name="gpu_utilization_percent",
                                value=gpu_util,
                                timestamp=timestamp,
                                tags={"gpu": str(i)}
                            ))
                            
                except Exception as e:
                    logger.warning(f"Failed to collect GPU metrics: {e}")
                    
            time.sleep(self.collection_interval)
            
    def get_current_resource_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        metrics = {}
        
        # CPU and Memory
        metrics['cpu_percent'] = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_available_gb'] = memory.available / (1024**3)
        
        # GPU
        if torch.cuda.is_available():
            gpu_metrics = {}
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                
                gpu_metrics[f'gpu_{i}_memory_percent'] = (allocated_memory / total_memory) * 100
                gpu_metrics[f'gpu_{i}_memory_allocated_gb'] = allocated_memory / (1024**3)
                
            metrics['gpu'] = gpu_metrics
            
        return metrics


class ModelMetricsCollector:
    """Collect model-specific metrics."""
    
    def __init__(self):
        self.training_metrics = defaultdict(list)
        self.inference_metrics = defaultdict(list)
        self.model_stats = {}
        
    def record_training_metric(self, metric_name: str, value: float, epoch: int, step: int, 
                             tags: Optional[Dict[str, str]] = None):
        """Record training metric."""
        metric = MetricPoint(
            name=f"training_{metric_name}",
            value=value,
            timestamp=datetime.now(),
            tags={**(tags or {}), "epoch": str(epoch), "step": str(step)},
            metadata={"epoch": epoch, "step": step, "metric_type": "training"}
        )
        self.training_metrics[metric_name].append(metric)
        return metric
        
    def record_validation_metric(self, metric_name: str, value: float, epoch: int,
                                tags: Optional[Dict[str, str]] = None):
        """Record validation metric."""
        metric = MetricPoint(
            name=f"validation_{metric_name}",
            value=value,
            timestamp=datetime.now(),
            tags={**(tags or {}), "epoch": str(epoch)},
            metadata={"epoch": epoch, "metric_type": "validation"}
        )
        self.training_metrics[f"val_{metric_name}"].append(metric)
        return metric
        
    def record_inference_metric(self, model_name: str, latency_ms: float, 
                               batch_size: int, tags: Optional[Dict[str, str]] = None):
        """Record inference metrics."""
        metric = MetricPoint(
            name="inference_latency_ms",
            value=latency_ms,
            timestamp=datetime.now(),
            tags={**(tags or {}), "model": model_name, "batch_size": str(batch_size)},
            metadata={"model": model_name, "batch_size": batch_size}
        )
        self.inference_metrics["latency"].append(metric)
        
        # Calculate throughput
        throughput = batch_size / (latency_ms / 1000.0)  # samples per second
        throughput_metric = MetricPoint(
            name="inference_throughput",
            value=throughput,
            timestamp=datetime.now(),
            tags={**(tags or {}), "model": model_name},
            metadata={"model": model_name, "batch_size": batch_size}
        )
        self.inference_metrics["throughput"].append(throughput_metric)
        
        return metric, throughput_metric
        
    def record_model_accuracy(self, model_name: str, accuracy: float, dataset: str,
                            tags: Optional[Dict[str, str]] = None):
        """Record model accuracy metrics."""
        metric = MetricPoint(
            name="model_accuracy",
            value=accuracy,
            timestamp=datetime.now(),
            tags={**(tags or {}), "model": model_name, "dataset": dataset},
            metadata={"model": model_name, "dataset": dataset}
        )
        self.inference_metrics["accuracy"].append(metric)
        return metric
        
    def analyze_model(self, model: torch.nn.Module, model_name: str) -> Dict[str, Any]:
        """Analyze model architecture and record stats."""
        stats = {}
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        stats['total_parameters'] = total_params
        stats['trainable_parameters'] = trainable_params
        stats['frozen_parameters'] = total_params - trainable_params
        
        # Model size
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        stats['model_size_mb'] = (param_size + buffer_size) / (1024**2)
        stats['parameters_size_mb'] = param_size / (1024**2)
        stats['buffers_size_mb'] = buffer_size / (1024**2)
        
        # Store stats
        self.model_stats[model_name] = {
            **stats,
            'timestamp': datetime.now().isoformat(),
            'architecture': str(model.__class__.__name__)
        }
        
        return stats
        
    def get_training_progress(self, metric_name: str, recent_epochs: int = 10) -> Dict[str, Any]:
        """Get training progress for a metric."""
        if metric_name not in self.training_metrics:
            return {}
            
        metrics = self.training_metrics[metric_name]
        if not metrics:
            return {}
            
        # Get recent metrics
        recent_metrics = sorted(metrics, key=lambda x: x.timestamp)[-recent_epochs:]
        values = [m.value for m in recent_metrics]
        
        progress = {
            'current_value': values[-1] if values else 0,
            'trend': 'improving' if len(values) >= 2 and values[-1] > values[0] else 'stable',
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
        
        # Calculate improvement rate
        if len(values) >= 2:
            progress['improvement_rate'] = (values[-1] - values[0]) / len(values)
            
        return progress


class MetricsAggregator:
    """Aggregate and analyze metrics from multiple collectors."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.performance_collector = PerformanceMetricsCollector()
        self.resource_collector = ResourceMetricsCollector()
        self.model_collector = ModelMetricsCollector()
        
        self.storage_path = storage_path
        if storage_path:
            self.db_connection = self._init_database()
        else:
            self.db_connection = None
            
    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for metrics storage."""
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.storage_path, check_same_thread=False)
        
        # Create metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                tags TEXT,
                metadata TEXT
            )
        """)
        
        # Create index for efficient querying
        conn.execute("CREATE INDEX IF NOT EXISTS idx_name_timestamp ON metrics(name, timestamp)")
        conn.commit()
        
        return conn
        
    def store_metric(self, metric: MetricPoint):
        """Store metric in database."""
        if not self.db_connection:
            return
            
        try:
            self.db_connection.execute(
                "INSERT INTO metrics (name, value, timestamp, tags, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    metric.name,
                    metric.value,
                    metric.timestamp.isoformat(),
                    json.dumps(metric.tags),
                    json.dumps(metric.metadata)
                )
            )
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
            
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        if not self.db_connection:
            return None
            
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor = self.db_connection.execute(
            "SELECT value, timestamp FROM metrics WHERE name = ? AND timestamp >= ? ORDER BY timestamp",
            (metric_name, cutoff)
        )
        
        rows = cursor.fetchall()
        if not rows:
            return None
            
        values = [row[0] for row in rows]
        timestamps = [datetime.fromisoformat(row[1]) for row in rows]
        
        return MetricSummary(
            name=metric_name,
            count=len(values),
            mean=np.mean(values),
            std=np.std(values),
            min_value=np.min(values),
            max_value=np.max(values),
            p50=np.percentile(values, 50),
            p95=np.percentile(values, 95),
            p99=np.percentile(values, 99),
            latest_value=values[-1],
            latest_timestamp=timestamps[-1]
        )
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'resource_metrics': self.resource_collector.get_current_resource_metrics(),
            'model_stats': self.model_collector.model_stats,
            'system_health': self._get_system_health_score()
        }
        
        # Get recent performance metrics
        recent_metrics = {}
        if self.db_connection:
            # Get recent latency metrics
            latency_summary = self.get_metric_summary("inference_latency_ms", hours=1)
            if latency_summary:
                recent_metrics['inference_latency'] = latency_summary.to_dict()
                
            # Get recent throughput metrics
            throughput_summary = self.get_metric_summary("inference_throughput", hours=1)
            if throughput_summary:
                recent_metrics['inference_throughput'] = throughput_summary.to_dict()
                
        dashboard_data['performance_metrics'] = recent_metrics
        
        return dashboard_data
        
    def _get_system_health_score(self) -> float:
        """Calculate overall system health score (0-1)."""
        score = 1.0
        
        try:
            # Resource health factors
            current_resources = self.resource_collector.get_current_resource_metrics()
            
            # CPU penalty
            cpu_usage = current_resources.get('cpu_percent', 0)
            if cpu_usage > 80:
                score -= 0.2
            elif cpu_usage > 90:
                score -= 0.4
                
            # Memory penalty
            memory_usage = current_resources.get('memory_percent', 0)
            if memory_usage > 85:
                score -= 0.2
            elif memory_usage > 95:
                score -= 0.4
                
            # GPU memory penalty
            gpu_metrics = current_resources.get('gpu', {})
            for gpu_id, memory_percent in gpu_metrics.items():
                if 'memory_percent' in gpu_id and memory_percent > 90:
                    score -= 0.1
                    
        except Exception as e:
            logger.warning(f"Failed to calculate health score: {e}")
            score = 0.5  # Unknown state
            
        return max(0.0, min(1.0, score))
        
    def start_collection(self):
        """Start all metric collection."""
        self.resource_collector.start_collection()
        logger.info("Started metrics collection")
        
    def stop_collection(self):
        """Stop all metric collection."""
        self.resource_collector.stop_collection()
        if self.db_connection:
            self.db_connection.close()
        logger.info("Stopped metrics collection")
        
    def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics to JSON file."""
        if not self.db_connection:
            logger.warning("No database connection for metrics export")
            return
            
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor = self.db_connection.execute(
            "SELECT * FROM metrics WHERE timestamp >= ? ORDER BY timestamp",
            (cutoff,)
        )
        
        metrics = []
        for row in cursor.fetchall():
            metrics.append({
                'name': row[1],
                'value': row[2],
                'timestamp': row[3],
                'tags': json.loads(row[4]) if row[4] else {},
                'metadata': json.loads(row[5]) if row[5] else {}
            })
            
        with open(filepath, 'w') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'metrics_count': len(metrics),
                'time_range_hours': hours,
                'metrics': metrics
            }, f, indent=2)
            
        logger.info(f"Exported {len(metrics)} metrics to {filepath}")


# Global metrics aggregator
metrics_aggregator: Optional[MetricsAggregator] = None


def get_metrics_aggregator(storage_path: Optional[str] = None) -> MetricsAggregator:
    """Get or create global metrics aggregator."""
    global metrics_aggregator
    
    if metrics_aggregator is None:
        metrics_aggregator = MetricsAggregator(storage_path)
        
    return metrics_aggregator


def setup_metrics_collection(storage_path: Optional[str] = "/tmp/ghf_metrics.db"):
    """Setup and start metrics collection."""
    aggregator = get_metrics_aggregator(storage_path)
    aggregator.start_collection()
    return aggregator