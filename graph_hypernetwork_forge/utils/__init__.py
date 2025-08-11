"""Utility functions and classes with comprehensive error handling and monitoring."""

from .training import HyperGNNTrainer, ZeroShotEvaluator
from .datasets import SyntheticDataGenerator, DatasetSplitter, create_sample_datasets
from . import graph_utils, text_utils, model_utils, evaluation_utils
from . import caching, batch_processing, profiling

# Import key optimization classes
from .caching import EmbeddingCache, WeightCache, get_embedding_cache, get_weight_cache
from .batch_processing import BatchProcessor, GraphBatcher, TextBatcher, auto_batch_size
from .profiling import PerformanceProfiler, ModelProfiler, get_profiler, profile

# Import enhanced error handling and logging utilities
from .logging_utils import (
    get_logger, setup_logging, GraphHypernetworkLogger, 
    LoggerMixin, log_function_call, log_performance_metrics
)
from .exceptions import (
    GraphHypernetworkError, ValidationError, ConfigurationError, ModelError,
    DataError, GPUError, MemoryError, FileIOError, NetworkError, TrainingError,
    InferenceError, GraphStructureError, handle_cuda_out_of_memory,
    create_validation_error, log_and_raise_error
)
from .memory_utils import (
    MemoryMonitor, MemoryInfo, memory_management, check_gpu_memory_available,
    estimate_tensor_memory, safe_cuda_operation, get_global_memory_monitor,
    start_global_memory_monitoring, stop_global_memory_monitoring
)

# Import comprehensive monitoring system
from .monitoring import SystemMonitor, TrainingMonitor, ModelAnalyzer, system_monitor, training_monitor
from .health_checks import (
    HealthStatus, HealthCheckResult, BaseHealthCheck, ModelHealthCheck, 
    MemoryHealthCheck, GPUHealthCheck, DataPipelineHealthCheck, 
    DependenciesHealthCheck, ExternalServiceHealthCheck, HealthCheckRegistry,
    health_registry, setup_default_health_checks, get_health_registry
)
from .metrics_collector import (
    MetricPoint, MetricSummary, MetricsBuffer, PerformanceMetricsCollector,
    ResourceMetricsCollector, ModelMetricsCollector, MetricsAggregator,
    get_metrics_aggregator, setup_metrics_collection
)
from .alerting import (
    AlertSeverity, AlertStatus, Alert, AlertRule, NotificationChannel, 
    AlertEvaluator, EmailNotifier, WebhookNotifier, SlackNotifier, AlertManager,
    get_alert_manager, setup_alerting
)
from .dashboard import (
    DashboardData, WebDashboard, ConsoleDashboard, create_dashboard, 
    setup_monitoring_dashboard
)
from .monitoring_server import (
    MonitoringServer, create_monitoring_server, run_monitoring_server
)

__all__ = [
    # Core training and evaluation
    "HyperGNNTrainer",
    "ZeroShotEvaluator", 
    "SyntheticDataGenerator",
    "DatasetSplitter",
    "create_sample_datasets",
    
    # Utility modules
    "graph_utils",
    "text_utils", 
    "model_utils",
    "evaluation_utils",
    "caching",
    "batch_processing", 
    "profiling",
    
    # Optimization classes
    "EmbeddingCache",
    "WeightCache",
    "get_embedding_cache",
    "get_weight_cache",
    "BatchProcessor",
    "GraphBatcher", 
    "TextBatcher",
    "auto_batch_size",
    "PerformanceProfiler",
    "ModelProfiler",
    "get_profiler",
    "profile",
    
    # Enhanced error handling and logging
    "get_logger",
    "setup_logging",
    "GraphHypernetworkLogger",
    "LoggerMixin",
    "log_function_call",
    "log_performance_metrics",
    
    # Exception classes
    "GraphHypernetworkError",
    "ValidationError",
    "ConfigurationError",
    "ModelError",
    "DataError",
    "GPUError",
    "MemoryError",
    "FileIOError",
    "NetworkError",
    "TrainingError",
    "InferenceError",
    "GraphStructureError",
    "handle_cuda_out_of_memory",
    "create_validation_error",
    "log_and_raise_error",
    
    # Memory management
    "MemoryMonitor",
    "MemoryInfo",
    "memory_management",
    "check_gpu_memory_available",
    "estimate_tensor_memory",
    "safe_cuda_operation",
    "get_global_memory_monitor",
    "start_global_memory_monitoring",
    "stop_global_memory_monitoring",
    
    # Comprehensive monitoring system
    "SystemMonitor",
    "TrainingMonitor", 
    "ModelAnalyzer",
    "system_monitor",
    "training_monitor",
    
    # Health checks
    "HealthStatus",
    "HealthCheckResult",
    "BaseHealthCheck",
    "ModelHealthCheck",
    "MemoryHealthCheck", 
    "GPUHealthCheck",
    "DataPipelineHealthCheck",
    "DependenciesHealthCheck",
    "ExternalServiceHealthCheck",
    "HealthCheckRegistry",
    "health_registry",
    "setup_default_health_checks",
    "get_health_registry",
    
    # Metrics collection
    "MetricPoint",
    "MetricSummary",
    "MetricsBuffer",
    "PerformanceMetricsCollector",
    "ResourceMetricsCollector", 
    "ModelMetricsCollector",
    "MetricsAggregator",
    "get_metrics_aggregator",
    "setup_metrics_collection",
    
    # Alerting system
    "AlertSeverity",
    "AlertStatus",
    "Alert",
    "AlertRule",
    "NotificationChannel",
    "AlertEvaluator",
    "EmailNotifier",
    "WebhookNotifier", 
    "SlackNotifier",
    "AlertManager",
    "get_alert_manager",
    "setup_alerting",
    
    # Dashboard and visualization
    "DashboardData",
    "WebDashboard",
    "ConsoleDashboard",
    "create_dashboard",
    "setup_monitoring_dashboard",
    
    # Monitoring server
    "MonitoringServer",
    "create_monitoring_server",
    "run_monitoring_server",
]