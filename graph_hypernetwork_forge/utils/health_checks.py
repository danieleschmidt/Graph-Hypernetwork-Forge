"""Comprehensive health check system for Graph Hypernetwork Forge."""

import time
import psutil
import torch
import threading
import subprocess
import requests
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

from .logging_utils import get_logger
from .exceptions import GraphHypernetworkError, GPUError, MemoryError
from .memory_utils import get_global_memory_monitor

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"  
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


class BaseHealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout: float = 10.0, critical: bool = False):
        self.name = name
        self.timeout = timeout
        self.critical = critical
        self.last_result: Optional[HealthCheckResult] = None
        
    def check(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = time.time()
        try:
            status, message, details = self._execute()
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details=details
            )
            
            self.last_result = result
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details={"error": str(e), "error_type": type(e).__name__}
            )
            
            self.last_result = result
            logger.error(f"Health check {self.name} failed: {e}")
            return result
    
    def _execute(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Execute the actual health check logic. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _execute method")


class ModelHealthCheck(BaseHealthCheck):
    """Health check for model availability and functionality."""
    
    def __init__(self, model_loader: Optional[Callable] = None, 
                 sample_input_generator: Optional[Callable] = None):
        super().__init__("model_health", timeout=30.0, critical=True)
        self.model_loader = model_loader
        self.sample_input_generator = sample_input_generator
        self._cached_model = None
        
    def _execute(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check model health."""
        details = {}
        
        try:
            # Check if model can be loaded/accessed
            if self.model_loader:
                if self._cached_model is None:
                    self._cached_model = self.model_loader()
                model = self._cached_model
            else:
                # Try to import and create a basic model
                from ..models import HyperGNN
                if self._cached_model is None:
                    self._cached_model = HyperGNN()
                model = self._cached_model
                
            details['model_type'] = type(model).__name__
            
            # Check if model is on expected device
            device = next(model.parameters()).device
            details['device'] = str(device)
            
            # Check model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            details['total_parameters'] = total_params
            details['trainable_parameters'] = trainable_params
            
            # Test inference if sample input generator provided
            if self.sample_input_generator:
                sample_input = self.sample_input_generator()
                model.eval()
                with torch.no_grad():
                    start_inference = time.time()
                    output = model(sample_input)
                    inference_time = (time.time() - start_inference) * 1000
                    
                details['inference_test'] = True
                details['inference_time_ms'] = inference_time
                details['output_shape'] = list(output.shape) if hasattr(output, 'shape') else str(type(output))
                
                if inference_time > 5000:  # > 5 seconds
                    return HealthStatus.DEGRADED, f"Model inference slow: {inference_time:.1f}ms", details
                    
            return HealthStatus.HEALTHY, "Model is healthy and functional", details
            
        except Exception as e:
            details['error'] = str(e)
            return HealthStatus.UNHEALTHY, f"Model health check failed: {str(e)}", details


class MemoryHealthCheck(BaseHealthCheck):
    """Health check for system and GPU memory."""
    
    def __init__(self, memory_threshold_percent: float = 90.0, 
                 gpu_memory_threshold_percent: float = 95.0):
        super().__init__("memory_health", timeout=5.0, critical=True)
        self.memory_threshold = memory_threshold_percent
        self.gpu_memory_threshold = gpu_memory_threshold_percent
        
    def _execute(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check memory health."""
        details = {}
        issues = []
        
        # System memory check
        memory = psutil.virtual_memory()
        details['system_memory_total_gb'] = memory.total / (1024**3)
        details['system_memory_used_gb'] = memory.used / (1024**3)
        details['system_memory_percent'] = memory.percent
        details['system_memory_available_gb'] = memory.available / (1024**3)
        
        if memory.percent > self.memory_threshold:
            issues.append(f"System memory usage high: {memory.percent:.1f}%")
            
        # GPU memory check
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    allocated_memory = torch.cuda.memory_allocated(i)
                    reserved_memory = torch.cuda.memory_reserved(i)
                    
                    allocated_percent = (allocated_memory / total_memory) * 100
                    reserved_percent = (reserved_memory / total_memory) * 100
                    
                    details[f'gpu_{i}_total_memory_gb'] = total_memory / (1024**3)
                    details[f'gpu_{i}_allocated_memory_gb'] = allocated_memory / (1024**3)
                    details[f'gpu_{i}_reserved_memory_gb'] = reserved_memory / (1024**3)
                    details[f'gpu_{i}_allocated_percent'] = allocated_percent
                    details[f'gpu_{i}_reserved_percent'] = reserved_percent
                    
                    if allocated_percent > self.gpu_memory_threshold:
                        issues.append(f"GPU {i} memory usage critical: {allocated_percent:.1f}%")
                    elif allocated_percent > self.gpu_memory_threshold - 10:
                        issues.append(f"GPU {i} memory usage high: {allocated_percent:.1f}%")
                        
                except Exception as e:
                    details[f'gpu_{i}_error'] = str(e)
                    issues.append(f"GPU {i} memory check failed: {str(e)}")
        
        # Memory leak detection (if memory monitor is available)
        try:
            memory_monitor = get_global_memory_monitor()
            if memory_monitor and hasattr(memory_monitor, 'get_memory_trend'):
                trend = memory_monitor.get_memory_trend()
                details['memory_trend'] = trend
                if trend > 100:  # Memory growing by more than 100MB/min
                    issues.append(f"Potential memory leak detected: {trend:.1f}MB/min growth")
        except Exception:
            pass  # Memory monitor not available
            
        if not issues:
            return HealthStatus.HEALTHY, "Memory usage within normal limits", details
        elif len(issues) == 1 and "high" in issues[0]:
            return HealthStatus.DEGRADED, "; ".join(issues), details
        else:
            return HealthStatus.UNHEALTHY, "; ".join(issues), details


class GPUHealthCheck(BaseHealthCheck):
    """Health check for GPU availability and status."""
    
    def __init__(self):
        super().__init__("gpu_health", timeout=10.0, critical=False)
        
    def _execute(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check GPU health."""
        details = {}
        
        # Basic CUDA availability
        details['cuda_available'] = torch.cuda.is_available()
        
        if not torch.cuda.is_available():
            return HealthStatus.DEGRADED, "CUDA not available", details
            
        try:
            device_count = torch.cuda.device_count()
            details['device_count'] = device_count
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                details[f'gpu_{i}_name'] = props.name
                details[f'gpu_{i}_compute_capability'] = f"{props.major}.{props.minor}"
                details[f'gpu_{i}_total_memory_gb'] = props.total_memory / (1024**3)
                
                # Test GPU functionality
                try:
                    torch.cuda.set_device(i)
                    # Create a small tensor and perform operation
                    test_tensor = torch.randn(10, 10, device=f'cuda:{i}')
                    result = torch.matmul(test_tensor, test_tensor.T)
                    torch.cuda.synchronize()
                    
                    details[f'gpu_{i}_functional'] = True
                    
                except Exception as e:
                    details[f'gpu_{i}_functional'] = False
                    details[f'gpu_{i}_error'] = str(e)
                    return HealthStatus.UNHEALTHY, f"GPU {i} functionality test failed: {str(e)}", details
                    
            return HealthStatus.HEALTHY, f"All {device_count} GPU(s) are healthy and functional", details
            
        except Exception as e:
            details['error'] = str(e)
            return HealthStatus.UNHEALTHY, f"GPU health check failed: {str(e)}", details


class DataPipelineHealthCheck(BaseHealthCheck):
    """Health check for data pipeline components."""
    
    def __init__(self, data_loader_factory: Optional[Callable] = None):
        super().__init__("data_pipeline_health", timeout=15.0, critical=True)
        self.data_loader_factory = data_loader_factory
        
    def _execute(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check data pipeline health."""
        details = {}
        issues = []
        
        try:
            # Test data loading
            if self.data_loader_factory:
                data_loader = self.data_loader_factory()
                
                # Test first batch
                start_time = time.time()
                first_batch = next(iter(data_loader))
                load_time = (time.time() - start_time) * 1000
                
                details['first_batch_load_time_ms'] = load_time
                details['batch_size'] = len(first_batch) if hasattr(first_batch, '__len__') else 'unknown'
                
                if load_time > 10000:  # > 10 seconds
                    issues.append(f"Data loading slow: {load_time:.1f}ms for first batch")
                    
            # Check data directory access (if applicable)
            import os
            data_dirs = ['data', 'datasets', '/tmp/data']
            accessible_dirs = []
            
            for data_dir in data_dirs:
                if os.path.exists(data_dir) and os.access(data_dir, os.R_OK):
                    accessible_dirs.append(data_dir)
                    dir_size = sum(os.path.getsize(os.path.join(data_dir, f)) 
                                 for f in os.listdir(data_dir) 
                                 if os.path.isfile(os.path.join(data_dir, f)))
                    details[f'{data_dir}_size_mb'] = dir_size / (1024**2)
                    
            details['accessible_data_directories'] = accessible_dirs
            
            if not issues:
                return HealthStatus.HEALTHY, "Data pipeline is healthy", details
            else:
                return HealthStatus.DEGRADED, "; ".join(issues), details
                
        except Exception as e:
            details['error'] = str(e)
            return HealthStatus.UNHEALTHY, f"Data pipeline check failed: {str(e)}", details


class DependenciesHealthCheck(BaseHealthCheck):
    """Health check for critical dependencies."""
    
    def __init__(self, required_packages: Optional[List[str]] = None):
        super().__init__("dependencies_health", timeout=10.0, critical=True)
        self.required_packages = required_packages or [
            'torch', 'numpy', 'psutil', 'requests', 'transformers'
        ]
        
    def _execute(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check dependencies health."""
        details = {}
        missing_packages = []
        version_info = {}
        
        for package in self.required_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                version_info[package] = version
                details[f'{package}_version'] = version
                
            except ImportError:
                missing_packages.append(package)
                details[f'{package}_available'] = False
                
        details['python_version'] = f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
        
        if missing_packages:
            return HealthStatus.UNHEALTHY, f"Missing required packages: {', '.join(missing_packages)}", details
        else:
            return HealthStatus.HEALTHY, "All dependencies are available", details


class ExternalServiceHealthCheck(BaseHealthCheck):
    """Health check for external services."""
    
    def __init__(self, service_urls: Dict[str, str], timeout: float = 5.0):
        super().__init__("external_services_health", timeout=timeout, critical=False)
        self.service_urls = service_urls
        
    def _execute(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check external services health."""
        details = {}
        issues = []
        
        for service_name, url in self.service_urls.items():
            try:
                start_time = time.time()
                response = requests.get(url, timeout=self.timeout)
                response_time = (time.time() - start_time) * 1000
                
                details[f'{service_name}_status_code'] = response.status_code
                details[f'{service_name}_response_time_ms'] = response_time
                details[f'{service_name}_available'] = response.status_code == 200
                
                if response.status_code != 200:
                    issues.append(f"{service_name} returned {response.status_code}")
                elif response_time > 5000:
                    issues.append(f"{service_name} response slow: {response_time:.1f}ms")
                    
            except requests.exceptions.Timeout:
                details[f'{service_name}_available'] = False
                details[f'{service_name}_error'] = "timeout"
                issues.append(f"{service_name} timeout")
                
            except requests.exceptions.RequestException as e:
                details[f'{service_name}_available'] = False
                details[f'{service_name}_error'] = str(e)
                issues.append(f"{service_name} connection failed")
                
        if not issues:
            return HealthStatus.HEALTHY, "All external services are healthy", details
        elif all("slow" in issue for issue in issues):
            return HealthStatus.DEGRADED, "; ".join(issues), details
        else:
            return HealthStatus.UNHEALTHY, "; ".join(issues), details


class HealthCheckRegistry:
    """Registry and orchestrator for health checks."""
    
    def __init__(self):
        self.health_checks: Dict[str, BaseHealthCheck] = {}
        self.check_history: Dict[str, List[HealthCheckResult]] = {}
        self.max_history = 100
        
    def register(self, health_check: BaseHealthCheck):
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        if health_check.name not in self.check_history:
            self.check_history[health_check.name] = []
            
    def unregister(self, name: str):
        """Unregister a health check."""
        self.health_checks.pop(name, None)
        self.check_history.pop(name, None)
        
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.health_checks:
            raise ValueError(f"Health check '{name}' not registered")
            
        health_check = self.health_checks[name]
        result = health_check.check()
        
        # Store in history
        history = self.check_history[name]
        history.append(result)
        if len(history) > self.max_history:
            history.pop(0)
            
        return result
        
    def run_all_checks(self, include_non_critical: bool = True) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name, health_check in self.health_checks.items():
            if not include_non_critical and not health_check.critical:
                continue
                
            try:
                results[name] = self.run_check(name)
            except Exception as e:
                logger.error(f"Failed to run health check {name}: {e}")
                # Create a failure result
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check execution failed: {str(e)}",
                    timestamp=datetime.now(),
                    duration_ms=0.0,
                    details={"error": str(e)}
                )
                
        return results
        
    def get_overall_health(self, results: Optional[Dict[str, HealthCheckResult]] = None) -> HealthStatus:
        """Get overall system health status."""
        if results is None:
            results = self.run_all_checks()
            
        if not results:
            return HealthStatus.UNKNOWN
            
        critical_results = [
            result for name, result in results.items() 
            if self.health_checks.get(name, BaseHealthCheck("", critical=False)).critical
        ]
        
        # Check critical health checks first
        for result in critical_results:
            if result.status == HealthStatus.UNHEALTHY:
                return HealthStatus.UNHEALTHY
                
        # Check if any critical are degraded
        if any(result.status == HealthStatus.DEGRADED for result in critical_results):
            return HealthStatus.DEGRADED
            
        # Check all results
        if any(result.status == HealthStatus.UNHEALTHY for result in results.values()):
            return HealthStatus.DEGRADED  # Non-critical unhealthy -> overall degraded
            
        if any(result.status == HealthStatus.DEGRADED for result in results.values()):
            return HealthStatus.DEGRADED
            
        if all(result.status == HealthStatus.HEALTHY for result in results.values()):
            return HealthStatus.HEALTHY
            
        return HealthStatus.UNKNOWN
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a comprehensive health summary."""
        results = self.run_all_checks()
        overall_status = self.get_overall_health(results)
        
        summary = {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'checks': {name: result.to_dict() for name, result in results.items()},
            'summary': {
                'total_checks': len(results),
                'healthy': sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                'degraded': sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                'unhealthy': sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
                'unknown': sum(1 for r in results.values() if r.status == HealthStatus.UNKNOWN),
            }
        }
        
        return summary
        
    def get_check_history(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get history for a specific health check."""
        if name not in self.check_history:
            return []
            
        history = self.check_history[name]
        recent_history = history[-limit:] if limit else history
        return [result.to_dict() for result in recent_history]


# Global health check registry
health_registry = HealthCheckRegistry()


def setup_default_health_checks(
    model_loader: Optional[Callable] = None,
    sample_input_generator: Optional[Callable] = None,
    data_loader_factory: Optional[Callable] = None,
    external_services: Optional[Dict[str, str]] = None
) -> HealthCheckRegistry:
    """Setup default health checks for the system."""
    
    # Register default health checks
    health_registry.register(ModelHealthCheck(model_loader, sample_input_generator))
    health_registry.register(MemoryHealthCheck())
    health_registry.register(GPUHealthCheck())
    health_registry.register(DataPipelineHealthCheck(data_loader_factory))
    health_registry.register(DependenciesHealthCheck())
    
    if external_services:
        health_registry.register(ExternalServiceHealthCheck(external_services))
        
    logger.info("Default health checks registered")
    return health_registry


def get_health_registry() -> HealthCheckRegistry:
    """Get the global health check registry."""
    return health_registry