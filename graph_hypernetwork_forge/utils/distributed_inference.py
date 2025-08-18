"""Distributed inference and high-performance computing for Graph Hypernetwork Forge."""

import asyncio
import concurrent.futures
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as torch_mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock torch classes
    class DDP:
        def __init__(self, *args, **kwargs): pass

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    # Mock ray
    class ray:
        @staticmethod
        def init(*args, **kwargs): pass
        @staticmethod
        def remote(*args, **kwargs): 
            def decorator(func): return func
            return decorator
        @staticmethod
        def get(*args, **kwargs): return args[0] if args else None
        @staticmethod
        def put(obj): return obj

try:
    from .logging_utils import get_logger
    from .exceptions import GraphHypernetworkError, ModelError, GPUError
    from .memory_utils import memory_management, estimate_tensor_memory
    from .resilience_advanced import get_resilience_orchestrator
    ENHANCED_FEATURES = True
except ImportError:
    def get_logger(name): return logging.getLogger(name)
    class GraphHypernetworkError(Exception): pass
    class ModelError(Exception): pass
    class GPUError(Exception): pass
    def memory_management(*args, **kwargs):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContext()
    def estimate_tensor_memory(*args): return 0
    def get_resilience_orchestrator(): 
        class MockOrchestrator:
            def resilient_operation(self, *args, **kwargs): 
                return kwargs.get('func', lambda: None)()
        return MockOrchestrator()
    ENHANCED_FEATURES = False

logger = get_logger(__name__)


class InferenceBackend(Enum):
    """Available inference backends."""
    LOCAL = "local"               # Single process local inference
    THREAD_POOL = "thread_pool"   # Multi-threaded local inference
    PROCESS_POOL = "process_pool" # Multi-process local inference
    DISTRIBUTED = "distributed"   # PyTorch distributed inference
    RAY = "ray"                   # Ray distributed computing
    ASYNC = "async"               # Asynchronous inference


@dataclass
class InferenceRequest:
    """Individual inference request."""
    request_id: str
    edge_index: Any  # torch.Tensor when available
    node_features: Any
    node_texts: List[str]
    priority: int = 1  # 1-10, higher is more priority
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    submitted_at: float = field(default_factory=time.time)


@dataclass
class InferenceResult:
    """Inference result."""
    request_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    memory_used: float = 0.0
    backend_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """Intelligent batch processing for efficient inference."""
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_time: float = 1.0,
        adaptive_batching: bool = True,
        memory_limit_mb: float = 4000.0,
    ):
        """Initialize batch processor.
        
        Args:
            max_batch_size: Maximum batch size
            max_wait_time: Maximum time to wait for batch completion
            adaptive_batching: Whether to adapt batch sizes based on performance
            memory_limit_mb: Memory limit for batches
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.adaptive_batching = adaptive_batching
        self.memory_limit_mb = memory_limit_mb
        
        # Batch tracking
        self.pending_requests = []
        self.batch_stats = {
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0,
            'optimal_batch_size': max_batch_size
        }
        
        self._lock = threading.Lock()
        
        logger.info(f"Batch processor initialized with max_batch_size={max_batch_size}")
    
    def add_request(self, request: InferenceRequest) -> bool:
        """Add request to batch queue.
        
        Args:
            request: Inference request to add
            
        Returns:
            True if added successfully
        """
        with self._lock:
            # Check memory constraints
            estimated_memory = self._estimate_request_memory(request)
            current_batch_memory = sum(
                self._estimate_request_memory(req) for req in self.pending_requests
            )
            
            if current_batch_memory + estimated_memory > self.memory_limit_mb:
                return False  # Would exceed memory limit
            
            # Add to batch with priority sorting
            self.pending_requests.append(request)
            self.pending_requests.sort(key=lambda x: (-x.priority, x.submitted_at))
            
            return True
    
    def get_next_batch(self) -> List[InferenceRequest]:
        """Get next batch for processing.
        
        Returns:
            List of inference requests forming a batch
        """
        with self._lock:
            if not self.pending_requests:
                return []
            
            # Determine batch size
            current_optimal = self.batch_stats['optimal_batch_size']
            batch_size = min(
                len(self.pending_requests),
                int(current_optimal),
                self.max_batch_size
            )
            
            # Check if we should wait for more requests
            if (len(self.pending_requests) < batch_size and 
                time.time() - self.pending_requests[0].submitted_at < self.max_wait_time):
                return []  # Wait for more requests
            
            # Create batch
            batch = self.pending_requests[:batch_size]
            self.pending_requests = self.pending_requests[batch_size:]
            
            return batch
    
    def update_batch_stats(self, batch_size: int, processing_time: float):
        """Update batch processing statistics.
        
        Args:
            batch_size: Size of the processed batch
            processing_time: Time taken to process the batch
        """
        with self._lock:
            self.batch_stats['total_batches'] += 1
            
            # Update running averages
            alpha = 0.1  # Smoothing factor
            self.batch_stats['avg_batch_size'] = (
                alpha * batch_size + 
                (1 - alpha) * self.batch_stats['avg_batch_size']
            )
            self.batch_stats['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.batch_stats['avg_processing_time']
            )
            
            # Adapt optimal batch size based on performance
            if self.adaptive_batching:
                self._adapt_batch_size(batch_size, processing_time)
    
    def _adapt_batch_size(self, batch_size: int, processing_time: float):
        """Adapt optimal batch size based on performance metrics."""
        # Simple adaptation: increase if processing time per item decreases
        if self.batch_stats['total_batches'] > 5:
            time_per_item = processing_time / batch_size
            avg_time_per_item = (
                self.batch_stats['avg_processing_time'] / 
                max(1, self.batch_stats['avg_batch_size'])
            )
            
            if time_per_item < avg_time_per_item * 0.95:  # 5% improvement
                # Increase batch size slightly
                self.batch_stats['optimal_batch_size'] = min(
                    self.max_batch_size,
                    self.batch_stats['optimal_batch_size'] * 1.1
                )
            elif time_per_item > avg_time_per_item * 1.1:  # 10% degradation
                # Decrease batch size
                self.batch_stats['optimal_batch_size'] = max(
                    1,
                    self.batch_stats['optimal_batch_size'] * 0.9
                )
    
    def _estimate_request_memory(self, request: InferenceRequest) -> float:
        """Estimate memory usage for a request in MB."""
        try:
            if TORCH_AVAILABLE and hasattr(request.node_features, 'shape'):
                return estimate_tensor_memory(request.node_features.shape) / (1024 * 1024)
            else:
                # Rough estimate based on number of nodes and texts
                num_nodes = len(request.node_texts)
                return num_nodes * 0.1  # 0.1 MB per node estimate
        except Exception:
            return 10.0  # Default conservative estimate


class DistributedInferenceEngine:
    """High-performance distributed inference engine."""
    
    def __init__(
        self,
        model_factory: Callable,
        backend: InferenceBackend = InferenceBackend.THREAD_POOL,
        max_workers: int = None,
        enable_batching: bool = True,
        batch_config: Optional[Dict] = None,
    ):
        """Initialize distributed inference engine.
        
        Args:
            model_factory: Function that creates model instances
            backend: Inference backend to use
            max_workers: Maximum number of worker processes/threads
            enable_batching: Whether to enable intelligent batching
            batch_config: Batch processor configuration
        """
        self.model_factory = model_factory
        self.backend = backend
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.enable_batching = enable_batching
        
        # Initialize batch processor
        if enable_batching:
            batch_config = batch_config or {}
            self.batch_processor = BatchProcessor(**batch_config)
        else:
            self.batch_processor = None
        
        # Request tracking
        self.active_requests = {}
        self.request_results = {}
        self.request_lock = threading.Lock()
        
        # Backend-specific initialization
        self.executor = None
        self.workers = []
        self._initialize_backend()
        
        logger.info(f"Distributed inference engine initialized with {backend.value} backend")
    
    def _initialize_backend(self):
        """Initialize the specific backend."""
        if self.backend == InferenceBackend.THREAD_POOL:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            
        elif self.backend == InferenceBackend.PROCESS_POOL:
            # Use spawn method for better compatibility
            ctx = mp.get_context('spawn')
            self.executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=ctx
            )
            
        elif self.backend == InferenceBackend.DISTRIBUTED and TORCH_AVAILABLE:
            self._initialize_torch_distributed()
            
        elif self.backend == InferenceBackend.RAY and RAY_AVAILABLE:
            self._initialize_ray_backend()
            
        elif self.backend == InferenceBackend.ASYNC:
            # Async backend uses asyncio internally
            pass
            
        else:
            # Fall back to local backend
            self.backend = InferenceBackend.LOCAL
            logger.warning(f"Backend not available, falling back to {self.backend.value}")
    
    def _initialize_torch_distributed(self):
        """Initialize PyTorch distributed backend."""
        try:
            if not dist.is_initialized():
                # Initialize distributed training
                dist.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    init_method='tcp://localhost:23456',
                    rank=0,
                    world_size=1
                )
            logger.info("PyTorch distributed backend initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize distributed backend: {e}")
            self.backend = InferenceBackend.THREAD_POOL
            self._initialize_backend()
    
    def _initialize_ray_backend(self):
        """Initialize Ray backend."""
        try:
            if not ray.is_initialized():
                ray.init(num_cpus=self.max_workers)
            
            # Create Ray workers
            @ray.remote
            class InferenceWorker:
                def __init__(self, model_factory):
                    self.model = model_factory()
                
                def infer_batch(self, requests):
                    results = []
                    for request in requests:
                        try:
                            start_time = time.time()
                            result = self.model.predict(
                                request.edge_index,
                                request.node_features,
                                request.node_texts
                            )
                            processing_time = time.time() - start_time
                            
                            results.append(InferenceResult(
                                request_id=request.request_id,
                                success=True,
                                result=result,
                                processing_time=processing_time,
                                backend_used="ray"
                            ))
                        except Exception as e:
                            results.append(InferenceResult(
                                request_id=request.request_id,
                                success=False,
                                error=str(e),
                                backend_used="ray"
                            ))
                    return results
            
            # Create worker pool
            self.workers = [
                InferenceWorker.remote(self.model_factory)
                for _ in range(self.max_workers)
            ]
            
            logger.info(f"Ray backend initialized with {len(self.workers)} workers")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Ray backend: {e}")
            self.backend = InferenceBackend.THREAD_POOL
            self._initialize_backend()
    
    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        """Asynchronous inference for single request.
        
        Args:
            request: Inference request
            
        Returns:
            Inference result
        """
        if self.enable_batching:
            return await self._infer_with_batching_async(request)
        else:
            return await self._infer_single_async(request)
    
    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Synchronous inference for single request.
        
        Args:
            request: Inference request
            
        Returns:
            Inference result
        """
        if self.backend == InferenceBackend.ASYNC:
            # Run async inference in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.infer_async(request))
            finally:
                loop.close()
        
        if self.enable_batching:
            return self._infer_with_batching(request)
        else:
            return self._infer_single(request)
    
    def infer_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Process batch of inference requests.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference results
        """
        if not requests:
            return []
        
        start_time = time.time()
        
        if self.backend == InferenceBackend.RAY and self.workers:
            return self._infer_batch_ray(requests)
        elif self.backend == InferenceBackend.DISTRIBUTED:
            return self._infer_batch_distributed(requests)
        elif self.executor:
            return self._infer_batch_executor(requests)
        else:
            return self._infer_batch_local(requests)
    
    def _infer_batch_ray(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Process batch using Ray backend."""
        # Distribute requests across workers
        worker_batches = self._distribute_requests(requests, len(self.workers))
        
        # Submit work to Ray workers
        futures = []
        for i, batch in enumerate(worker_batches):
            if batch:  # Only submit non-empty batches
                future = self.workers[i].infer_batch.remote(batch)
                futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            batch_results = ray.get(future)
            results.extend(batch_results)
        
        return results
    
    def _infer_batch_distributed(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Process batch using PyTorch distributed backend."""
        # For single-node distributed, process sequentially with DDP model
        model = self.model_factory()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            model = DDP(model.cuda())
        
        results = []
        for request in requests:
            try:
                start_time = time.time()
                result = model.module.predict(
                    request.edge_index,
                    request.node_features,
                    request.node_texts
                ) if hasattr(model, 'module') else model.predict(
                    request.edge_index,
                    request.node_features,
                    request.node_texts
                )
                processing_time = time.time() - start_time
                
                results.append(InferenceResult(
                    request_id=request.request_id,
                    success=True,
                    result=result,
                    processing_time=processing_time,
                    backend_used="distributed"
                ))
            except Exception as e:
                results.append(InferenceResult(
                    request_id=request.request_id,
                    success=False,
                    error=str(e),
                    backend_used="distributed"
                ))
        
        return results
    
    def _infer_batch_executor(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Process batch using ThreadPoolExecutor or ProcessPoolExecutor."""
        # Submit individual requests to executor
        futures = {}
        for request in requests:
            future = self.executor.submit(self._infer_single_worker, request)
            futures[future] = request.request_id
        
        # Collect results as they complete
        results = []
        for future in as_completed(futures, timeout=max(req.timeout for req in requests)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                request_id = futures[future]
                results.append(InferenceResult(
                    request_id=request_id,
                    success=False,
                    error=str(e),
                    backend_used=self.backend.value
                ))
        
        return results
    
    def _infer_batch_local(self, requests: List[InferenceRequest]) -> List[InferenceResult:
        """Process batch locally in single thread."""
        model = self.model_factory()
        results = []
        
        for request in requests:
            try:
                start_time = time.time()
                
                with memory_management(cleanup_on_exit=True):
                    result = model.predict(
                        request.edge_index,
                        request.node_features,
                        request.node_texts
                    )
                
                processing_time = time.time() - start_time
                
                results.append(InferenceResult(
                    request_id=request.request_id,
                    success=True,
                    result=result,
                    processing_time=processing_time,
                    backend_used="local"
                ))
                
            except Exception as e:
                results.append(InferenceResult(
                    request_id=request.request_id,
                    success=False,
                    error=str(e),
                    backend_used="local"
                ))
        
        return results
    
    def _infer_single_worker(self, request: InferenceRequest) -> InferenceResult:
        """Worker function for single request inference."""
        try:
            # Use resilience patterns
            orchestrator = get_resilience_orchestrator()
            
            def inference_operation():
                model = self.model_factory()
                return model.predict(
                    request.edge_index,
                    request.node_features,
                    request.node_texts
                )
            
            start_time = time.time()
            
            result = orchestrator.resilient_operation(
                operation_name=f"inference_{request.request_id}",
                func=inference_operation,
                use_circuit_breaker=True,
                use_retry=True,
                use_bulkhead=True
            )
            
            processing_time = time.time() - start_time
            
            return InferenceResult(
                request_id=request.request_id,
                success=True,
                result=result,
                processing_time=processing_time,
                backend_used=self.backend.value
            )
            
        except Exception as e:
            return InferenceResult(
                request_id=request.request_id,
                success=False,
                error=str(e),
                backend_used=self.backend.value
            )
    
    def _infer_single(self, request: InferenceRequest) -> InferenceResult:
        """Synchronous single request inference."""
        if self.executor:
            future = self.executor.submit(self._infer_single_worker, request)
            return future.result(timeout=request.timeout)
        else:
            return self._infer_single_worker(request)
    
    async def _infer_single_async(self, request: InferenceRequest) -> InferenceResult:
        """Asynchronous single request inference."""
        loop = asyncio.get_event_loop()
        
        if self.executor:
            result = await loop.run_in_executor(
                self.executor,
                self._infer_single_worker,
                request
            )
        else:
            result = await loop.run_in_executor(
                None,  # Use default executor
                self._infer_single_worker,
                request
            )
        
        return result
    
    def _infer_with_batching(self, request: InferenceRequest) -> InferenceResult:
        """Inference with intelligent batching."""
        # Add request to batch queue
        if not self.batch_processor.add_request(request):
            # Batch is full, process immediately
            return self._infer_single(request)
        
        # Wait for batch to be ready or timeout
        start_wait = time.time()
        while time.time() - start_wait < request.timeout:
            batch = self.batch_processor.get_next_batch()
            if batch and any(req.request_id == request.request_id for req in batch):
                # Our request is in this batch
                results = self.infer_batch(batch)
                
                # Update batch statistics
                batch_time = sum(r.processing_time for r in results) / len(results)
                self.batch_processor.update_batch_stats(len(batch), batch_time)
                
                # Find our result
                for result in results:
                    if result.request_id == request.request_id:
                        return result
            
            time.sleep(0.01)  # Small sleep to prevent busy waiting
        
        # Timeout waiting for batch, process individually
        return self._infer_single(request)
    
    async def _infer_with_batching_async(self, request: InferenceRequest) -> InferenceResult:
        """Asynchronous inference with batching."""
        # Add request to batch queue
        if not self.batch_processor.add_request(request):
            return await self._infer_single_async(request)
        
        # Wait for batch processing
        start_wait = time.time()
        while time.time() - start_wait < request.timeout:
            batch = self.batch_processor.get_next_batch()
            if batch and any(req.request_id == request.request_id for req in batch):
                # Process batch asynchronously
                results = await self._infer_batch_async(batch)
                
                # Update batch statistics
                batch_time = sum(r.processing_time for r in results) / len(results)
                self.batch_processor.update_batch_stats(len(batch), batch_time)
                
                # Find our result
                for result in results:
                    if result.request_id == request.request_id:
                        return result
            
            await asyncio.sleep(0.01)
        
        # Timeout, process individually
        return await self._infer_single_async(request)
    
    async def _infer_batch_async(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Asynchronous batch processing."""
        loop = asyncio.get_event_loop()
        
        if self.backend == InferenceBackend.RAY and self.workers:
            # Ray is inherently async-friendly
            return self._infer_batch_ray(requests)
        
        # Use thread executor for async batch processing
        executor = self.executor or ThreadPoolExecutor(max_workers=self.max_workers)
        
        tasks = []
        for request in requests:
            task = loop.run_in_executor(executor, self._infer_single_worker, request)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(InferenceResult(
                    request_id=requests[i].request_id,
                    success=False,
                    error=str(result),
                    backend_used=self.backend.value
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _distribute_requests(self, requests: List[InferenceRequest], num_workers: int) -> List[List[InferenceRequest]]:
        """Distribute requests across workers."""
        worker_batches = [[] for _ in range(num_workers)]
        
        for i, request in enumerate(requests):
            worker_idx = i % num_workers
            worker_batches[worker_idx].append(request)
        
        return worker_batches
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'backend': self.backend.value,
            'max_workers': self.max_workers,
            'active_requests': len(self.active_requests),
        }
        
        if self.batch_processor:
            stats['batch_stats'] = self.batch_processor.batch_stats.copy()
            stats['pending_requests'] = len(self.batch_processor.pending_requests)
        
        if self.backend == InferenceBackend.RAY:
            stats['ray_workers'] = len(self.workers)
        
        return stats
    
    def shutdown(self):
        """Shutdown the inference engine."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.backend == InferenceBackend.RAY and RAY_AVAILABLE:
            ray.shutdown()
        
        logger.info("Distributed inference engine shut down")


class LoadBalancer:
    """Intelligent load balancer for multiple inference engines."""
    
    def __init__(self, engines: List[DistributedInferenceEngine]):
        """Initialize load balancer.
        
        Args:
            engines: List of inference engines to balance across
        """
        self.engines = engines
        self.engine_stats = {i: {'requests': 0, 'avg_time': 0.0} for i in range(len(engines))}
        self.round_robin_index = 0
        self._lock = threading.Lock()
        
        logger.info(f"Load balancer initialized with {len(engines)} engines")
    
    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Route request to best available engine.
        
        Args:
            request: Inference request
            
        Returns:
            Inference result
        """
        engine_idx = self._select_engine(request)
        engine = self.engines[engine_idx]
        
        start_time = time.time()
        result = engine.infer(request)
        processing_time = time.time() - start_time
        
        # Update engine statistics
        with self._lock:
            stats = self.engine_stats[engine_idx]
            stats['requests'] += 1
            stats['avg_time'] = (
                0.9 * stats['avg_time'] + 0.1 * processing_time
            )
        
        return result
    
    def _select_engine(self, request: InferenceRequest) -> int:
        """Select best engine for request."""
        with self._lock:
            # For now, use round-robin with basic load awareness
            # Could be enhanced with more sophisticated algorithms
            
            # Find engine with lowest average processing time
            best_engine = min(
                self.engine_stats.items(),
                key=lambda x: x[1]['avg_time'] if x[1]['requests'] > 0 else 0
            )[0]
            
            # Occasionally use round-robin to explore other engines
            if self.round_robin_index % 10 == 0:
                best_engine = self.round_robin_index % len(self.engines)
            
            self.round_robin_index += 1
            return best_engine
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            return {
                'total_engines': len(self.engines),
                'engine_stats': self.engine_stats.copy(),
                'total_requests': sum(stats['requests'] for stats in self.engine_stats.values())
            }


# Convenience functions and factories
def create_inference_engine(
    model_factory: Callable,
    backend: str = "thread_pool",
    **kwargs
) -> DistributedInferenceEngine:
    """Create inference engine with specified backend.
    
    Args:
        model_factory: Function that creates model instances
        backend: Backend type (thread_pool, process_pool, ray, distributed, async)
        **kwargs: Additional configuration options
        
    Returns:
        Configured inference engine
    """
    backend_enum = InferenceBackend(backend.lower())
    
    return DistributedInferenceEngine(
        model_factory=model_factory,
        backend=backend_enum,
        **kwargs
    )


def create_multi_engine_setup(
    model_factory: Callable,
    num_engines: int = 2,
    backend_configs: Optional[List[Dict]] = None
) -> LoadBalancer:
    """Create multiple inference engines with load balancer.
    
    Args:
        model_factory: Function that creates model instances
        num_engines: Number of engines to create
        backend_configs: Configuration for each engine
        
    Returns:
        Load balancer managing multiple engines
    """
    if backend_configs is None:
        backend_configs = [
            {'backend': InferenceBackend.THREAD_POOL, 'max_workers': 4}
            for _ in range(num_engines)
        ]
    
    engines = []
    for config in backend_configs[:num_engines]:
        engine = DistributedInferenceEngine(
            model_factory=model_factory,
            **config
        )
        engines.append(engine)
    
    return LoadBalancer(engines)


# Global distributed engine instance
_global_engine = None

def get_distributed_engine(model_factory: Callable = None) -> Optional[DistributedInferenceEngine]:
    """Get global distributed inference engine."""
    global _global_engine
    if _global_engine is None and model_factory is not None:
        _global_engine = create_inference_engine(model_factory)
    return _global_engine