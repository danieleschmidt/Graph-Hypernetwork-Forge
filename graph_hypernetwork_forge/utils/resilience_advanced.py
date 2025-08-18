"""Advanced resilience and fault tolerance utilities for production deployment."""

import asyncio
import contextlib
import functools
import time
import threading
import random
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

try:
    from .logging_utils import get_logger, LoggerMixin
    from .exceptions import GraphHypernetworkError, ModelError, GPUError
    ENHANCED_FEATURES = True
except ImportError:
    def get_logger(name): return logging.getLogger(name)
    class LoggerMixin: pass
    class GraphHypernetworkError(Exception): pass
    class ModelError(Exception): pass  
    class GPUError(Exception): pass
    ENHANCED_FEATURES = False

logger = get_logger(__name__)


class FailureType(Enum):
    """Types of failures that can occur."""
    TRANSIENT = "transient"          # Temporary failures that may resolve
    PERSISTENT = "persistent"        # Failures that require intervention
    CATASTROPHIC = "catastrophic"    # Critical failures requiring system restart
    RESOURCE_EXHAUSTION = "resource" # Memory/GPU/disk exhaustion
    NETWORK = "network"              # Network connectivity issues
    MODEL = "model"                  # Model inference/training failures


@dataclass
class FailureRecord:
    """Record of a failure occurrence."""
    timestamp: float
    failure_type: FailureType
    component: str
    error_message: str
    severity: int  # 1-5, 5 being most severe
    recovery_attempted: bool = False
    recovery_successful: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, blocking requests
    HALF_OPEN = "half_open" # Testing if service has recovered


class CircuitBreaker:
    """Advanced circuit breaker with intelligent failure detection."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        recovery_timeout: float = 300.0,
        success_threshold: int = 3,
        failure_rate_threshold: float = 0.5,
        window_size: int = 100,
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before trying half-open state
            recovery_timeout: Time to wait for full recovery
            success_threshold: Successes needed in half-open to close circuit
            failure_rate_threshold: Failure rate to trigger opening (0.0-1.0)
            window_size: Rolling window size for failure rate calculation
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_rate_threshold = failure_rate_threshold
        self.window_size = window_size
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
        # Rolling window for failure rate calculation
        self.request_results = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
        logger.info(f"Circuit breaker initialized with failure_threshold={failure_threshold}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == CircuitBreakerState.OPEN:
                    if time.time() - self.last_failure_time < self.timeout:
                        raise ModelError(
                            func.__name__, 
                            "circuit_breaker_open",
                            "Circuit breaker is open - service unavailable"
                        )
                    else:
                        self.state = CircuitBreakerState.HALF_OPEN
                        logger.info("Circuit breaker moving to HALF_OPEN state")
            
            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except Exception as e:
                self._record_failure()
                raise e
        
        return wrapper
    
    def _record_success(self):
        """Record a successful operation."""
        with self._lock:
            self.request_results.append(True)
            self.last_success_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker CLOSED - service recovered")
    
    def _record_failure(self):
        """Record a failed operation."""
        with self._lock:
            self.request_results.append(False)
            self.last_failure_time = time.time()
            self.failure_count += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning("Circuit breaker OPEN - service still failing")
            elif self.state == CircuitBreakerState.CLOSED:
                # Check if we should open the circuit
                if (self.failure_count >= self.failure_threshold or 
                    self._get_failure_rate() >= self.failure_rate_threshold):
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker OPEN - failure threshold reached "
                                 f"(failures: {self.failure_count}, rate: {self._get_failure_rate():.2f})")
    
    def _get_failure_rate(self) -> float:
        """Calculate current failure rate."""
        if len(self.request_results) < 10:  # Need minimum samples
            return 0.0
        
        failures = sum(1 for result in self.request_results if not result)
        return failures / len(self.request_results)
    
    @property
    def is_available(self) -> bool:
        """Check if service is available."""
        return self.state != CircuitBreakerState.OPEN


class RetryStrategy:
    """Advanced retry strategy with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Tuple[Exception, ...]] = None,
    ):
        """Initialize retry strategy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types that should trigger retries
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        if retryable_exceptions is None:
            # Default retryable exceptions (transient failures)
            self.retryable_exceptions = (
                ConnectionError, TimeoutError, FutureTimeoutError,
                GPUError,  # CUDA OOM might be recoverable
            )
        else:
            self.retryable_exceptions = retryable_exceptions
        
        logger.info(f"Retry strategy initialized with max_attempts={max_attempts}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not isinstance(e, self.retryable_exceptions):
                        logger.debug(f"Exception {type(e).__name__} not retryable, failing immediately")
                        raise e
                    
                    if attempt == self.max_attempts - 1:
                        logger.error(f"All {self.max_attempts} retry attempts failed")
                        raise e
                    
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed with {type(e).__name__}, "
                                 f"retrying in {delay:.2f}s: {e}")
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add up to 20% jitter
            jitter_amount = delay * 0.2 * random.random()
            delay += jitter_amount
        
        return delay


class HealthCheck:
    """Comprehensive system health monitoring."""
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize health checker.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.health_status = {}
        self.health_history = defaultdict(list)
        self._running = False
        self._thread = None
        
        logger.info(f"Health checker initialized with interval={check_interval}s")
    
    def register_check(self, name: str, check_func: Callable[[], bool], 
                      critical: bool = False) -> None:
        """Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
            critical: Whether this check is critical for system operation
        """
        self.health_status[name] = {
            'func': check_func,
            'critical': critical,
            'status': None,
            'last_check': 0,
            'consecutive_failures': 0
        }
        logger.info(f"Registered health check: {name} (critical: {critical})")
    
    def start_monitoring(self) -> None:
        """Start health monitoring in background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join()
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            self.perform_checks()
            time.sleep(self.check_interval)
    
    def perform_checks(self) -> Dict[str, bool]:
        """Perform all registered health checks."""
        results = {}
        
        for name, check_info in self.health_status.items():
            try:
                start_time = time.time()
                is_healthy = check_info['func']()
                check_time = time.time() - start_time
                
                check_info['status'] = is_healthy
                check_info['last_check'] = time.time()
                
                if is_healthy:
                    check_info['consecutive_failures'] = 0
                else:
                    check_info['consecutive_failures'] += 1
                    logger.warning(f"Health check '{name}' failed "
                                 f"({check_info['consecutive_failures']} consecutive failures)")
                
                # Store in history
                self.health_history[name].append({
                    'timestamp': time.time(),
                    'healthy': is_healthy,
                    'check_time': check_time
                })
                
                # Keep only last 100 checks
                if len(self.health_history[name]) > 100:
                    self.health_history[name].pop(0)
                
                results[name] = is_healthy
                
            except Exception as e:
                logger.error(f"Health check '{name}' threw exception: {e}")
                check_info['status'] = False
                check_info['consecutive_failures'] += 1
                results[name] = False
        
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        overall_healthy = True
        critical_failures = []
        
        for name, check_info in self.health_status.items():
            if check_info['status'] is False:
                if check_info['critical']:
                    overall_healthy = False
                    critical_failures.append(name)
        
        return {
            'healthy': overall_healthy,
            'critical_failures': critical_failures,
            'checks': {name: info['status'] for name, info in self.health_status.items()},
            'timestamp': time.time()
        }


class BulkheadIsolation:
    """Resource isolation using bulkhead pattern."""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        """Initialize bulkhead isolation.
        
        Args:
            max_workers: Maximum number of worker threads
            queue_size: Maximum queue size for pending work
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="bulkhead"
        )
        self.queue_size = queue_size
        self._pending_count = 0
        self._lock = threading.Lock()
        
        logger.info(f"Bulkhead isolation initialized with {max_workers} workers")
    
    def isolate(self, timeout: float = 30.0):
        """Decorator to isolate function execution."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self._lock:
                    if self._pending_count >= self.queue_size:
                        raise ModelError(
                            func.__name__,
                            "bulkhead_overflow",
                            f"Bulkhead queue full ({self.queue_size} pending)"
                        )
                    self._pending_count += 1
                
                try:
                    future = self.executor.submit(func, *args, **kwargs)
                    result = future.result(timeout=timeout)
                    return result
                except FutureTimeoutError:
                    raise ModelError(
                        func.__name__,
                        "bulkhead_timeout",
                        f"Operation timed out after {timeout}s"
                    )
                finally:
                    with self._lock:
                        self._pending_count -= 1
            
            return wrapper
        return decorator
    
    def shutdown(self):
        """Shutdown the bulkhead executor."""
        self.executor.shutdown(wait=True)
        logger.info("Bulkhead isolation shut down")


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system performance."""
    
    def __init__(
        self,
        initial_rate: float = 10.0,  # requests per second
        min_rate: float = 1.0,
        max_rate: float = 100.0,
        adaptation_window: float = 60.0,  # seconds
    ):
        """Initialize adaptive rate limiter.
        
        Args:
            initial_rate: Initial requests per second limit
            min_rate: Minimum rate limit
            max_rate: Maximum rate limit
            adaptation_window: Window for rate adaptation
        """
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adaptation_window = adaptation_window
        
        self.request_times = deque()
        self.success_count = 0
        self.failure_count = 0
        self.last_adaptation = time.time()
        
        self._lock = threading.Lock()
        
        logger.info(f"Adaptive rate limiter initialized at {initial_rate} req/s")
    
    def allow_request(self) -> bool:
        """Check if request should be allowed based on current rate."""
        current_time = time.time()
        
        with self._lock:
            # Remove old requests outside window
            while (self.request_times and 
                   current_time - self.request_times[0] > 1.0):
                self.request_times.popleft()
            
            # Check if we're under the rate limit
            if len(self.request_times) >= self.current_rate:
                return False
            
            self.request_times.append(current_time)
            self._maybe_adapt_rate()
            return True
    
    def record_success(self):
        """Record a successful operation."""
        with self._lock:
            self.success_count += 1
    
    def record_failure(self):
        """Record a failed operation."""
        with self._lock:
            self.failure_count += 1
    
    def _maybe_adapt_rate(self):
        """Adapt rate limit based on recent performance."""
        current_time = time.time()
        
        if current_time - self.last_adaptation < self.adaptation_window:
            return
        
        total_requests = self.success_count + self.failure_count
        if total_requests == 0:
            return
        
        success_rate = self.success_count / total_requests
        
        # Adapt rate based on success rate
        if success_rate > 0.95:
            # High success rate, can increase rate
            new_rate = min(self.current_rate * 1.1, self.max_rate)
        elif success_rate < 0.8:
            # Low success rate, decrease rate
            new_rate = max(self.current_rate * 0.9, self.min_rate)
        else:
            # Acceptable success rate, maintain current rate
            new_rate = self.current_rate
        
        if abs(new_rate - self.current_rate) > 0.1:
            logger.info(f"Rate limit adapted: {self.current_rate:.1f} -> {new_rate:.1f} "
                       f"(success rate: {success_rate:.2%})")
            self.current_rate = new_rate
        
        # Reset counters
        self.success_count = 0
        self.failure_count = 0
        self.last_adaptation = current_time


class ResilienceOrchestrator:
    """Orchestrates all resilience patterns for comprehensive fault tolerance."""
    
    def __init__(self):
        """Initialize resilience orchestrator."""
        self.circuit_breakers = {}
        self.retry_strategies = {}
        self.health_checker = HealthCheck()
        self.bulkhead = BulkheadIsolation()
        self.rate_limiter = AdaptiveRateLimiter()
        self.failure_records = deque(maxlen=1000)
        
        logger.info("Resilience orchestrator initialized")
    
    def register_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Register a circuit breaker for a component."""
        cb = CircuitBreaker(**kwargs)
        self.circuit_breakers[name] = cb
        return cb
    
    def register_retry_strategy(self, name: str, **kwargs) -> RetryStrategy:
        """Register a retry strategy for a component."""
        rs = RetryStrategy(**kwargs)
        self.retry_strategies[name] = rs
        return rs
    
    def resilient_operation(
        self,
        operation_name: str,
        func: Callable,
        use_circuit_breaker: bool = True,
        use_retry: bool = True,
        use_bulkhead: bool = True,
        use_rate_limit: bool = True,
        **kwargs
    ):
        """Execute operation with full resilience patterns."""
        
        # Rate limiting
        if use_rate_limit and not self.rate_limiter.allow_request():
            raise ModelError(
                operation_name,
                "rate_limited",
                "Request rate limit exceeded"
            )
        
        # Get or create circuit breaker
        if use_circuit_breaker:
            if operation_name not in self.circuit_breakers:
                self.circuit_breakers[operation_name] = CircuitBreaker()
            cb = self.circuit_breakers[operation_name]
            
            if not cb.is_available:
                raise ModelError(
                    operation_name,
                    "circuit_breaker_open",
                    "Circuit breaker is open"
                )
        
        # Get or create retry strategy
        if use_retry:
            if operation_name not in self.retry_strategies:
                self.retry_strategies[operation_name] = RetryStrategy()
            rs = self.retry_strategies[operation_name]
            func = rs(func)
        
        # Apply circuit breaker
        if use_circuit_breaker:
            func = cb(func)
        
        # Apply bulkhead isolation
        if use_bulkhead:
            func = self.bulkhead.isolate()(func)
        
        try:
            result = func(**kwargs)
            self.rate_limiter.record_success()
            return result
        except Exception as e:
            self.rate_limiter.record_failure()
            self._record_failure(operation_name, e)
            raise
    
    def _record_failure(self, component: str, error: Exception):
        """Record failure for analysis."""
        failure_record = FailureRecord(
            timestamp=time.time(),
            failure_type=self._classify_failure(error),
            component=component,
            error_message=str(error),
            severity=self._assess_severity(error)
        )
        
        self.failure_records.append(failure_record)
        logger.error(f"Failure recorded for {component}: {failure_record.failure_type} - {error}")
    
    def _classify_failure(self, error: Exception) -> FailureType:
        """Classify the type of failure."""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return FailureType.TRANSIENT
        elif isinstance(error, MemoryError):
            return FailureType.RESOURCE_EXHAUSTION
        elif isinstance(error, GPUError):
            return FailureType.RESOURCE_EXHAUSTION
        elif isinstance(error, ModelError):
            return FailureType.MODEL
        else:
            return FailureType.PERSISTENT
    
    def _assess_severity(self, error: Exception) -> int:
        """Assess severity of error (1-5 scale)."""
        if isinstance(error, (MemoryError, GPUError)):
            return 4  # High severity
        elif isinstance(error, ModelError):
            return 3  # Medium severity
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return 2  # Low severity
        else:
            return 3  # Default medium severity
    
    def get_resilience_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics."""
        return {
            'circuit_breakers': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb.failure_count,
                    'failure_rate': cb._get_failure_rate()
                }
                for name, cb in self.circuit_breakers.items()
            },
            'rate_limiter': {
                'current_rate': self.rate_limiter.current_rate,
                'requests_in_window': len(self.rate_limiter.request_times)
            },
            'system_health': self.health_checker.get_system_health(),
            'recent_failures': [
                {
                    'timestamp': fr.timestamp,
                    'type': fr.failure_type.value,
                    'component': fr.component,
                    'severity': fr.severity
                }
                for fr in list(self.failure_records)[-10:]  # Last 10 failures
            ]
        }
    
    def start(self):
        """Start resilience monitoring."""
        self.health_checker.start_monitoring()
        logger.info("Resilience orchestrator started")
    
    def shutdown(self):
        """Shutdown resilience components."""
        self.health_checker.stop_monitoring()
        self.bulkhead.shutdown()
        logger.info("Resilience orchestrator shut down")


# Global resilience orchestrator
_global_resilience = None

def get_resilience_orchestrator() -> ResilienceOrchestrator:
    """Get global resilience orchestrator."""
    global _global_resilience
    if _global_resilience is None:
        _global_resilience = ResilienceOrchestrator()
    return _global_resilience


def resilient(operation_name: str, **resilience_kwargs):
    """Decorator to make any function resilient with all patterns."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            orchestrator = get_resilience_orchestrator()
            return orchestrator.resilient_operation(
                operation_name=operation_name,
                func=lambda: func(*args, **kwargs),
                **resilience_kwargs
            )
        return wrapper
    return decorator