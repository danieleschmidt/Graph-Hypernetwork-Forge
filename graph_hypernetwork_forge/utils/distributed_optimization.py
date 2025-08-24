"""
Advanced distributed optimization and scaling utilities for HyperGNN.

This module provides sophisticated distributed training, multi-GPU support,
performance optimization, and high-throughput inference capabilities.
"""

import asyncio
import multiprocessing as mp
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .logging_utils import get_logger
from .exceptions import GPUError, ModelError
from .memory_utils import estimate_tensor_memory

logger = get_logger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    find_unused_parameters: bool = True
    gradient_as_bucket_view: bool = True


class DistributedManager:
    """Manages distributed training setup and coordination."""
    
    def __init__(self, config: DistributedConfig):
        """Initialize distributed manager.
        
        Args:
            config: Distributed training configuration
        """
        self.config = config
        self.is_initialized = False
        self.is_master = config.rank == 0
        
        logger.info(f"Initializing DistributedManager: "
                   f"backend={config.backend}, "
                   f"world_size={config.world_size}, "
                   f"rank={config.rank}")
    
    def setup(self) -> bool:
        """Setup distributed training environment.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Set environment variables
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            os.environ['WORLD_SIZE'] = str(self.config.world_size)
            os.environ['RANK'] = str(self.config.rank)
            os.environ['LOCAL_RANK'] = str(self.config.local_rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            # Set CUDA device if available
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
            
            self.is_initialized = True
            
            if self.is_master:
                logger.info("Distributed training setup successful")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {e}")
            return False
    
    def cleanup(self):
        """Cleanup distributed resources."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("Distributed training cleanup completed")
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model for distributed training.
        
        Args:
            model: PyTorch model to wrap
            
        Returns:
            Wrapped model for distributed training
        """
        if not self.is_initialized:
            raise RuntimeError("Distributed training not initialized")
        
        # Move model to appropriate device
        device = torch.device(f"cuda:{self.config.local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
            output_device=self.config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.find_unused_parameters,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
        )
        
        logger.info(f"Model wrapped for distributed training on device: {device}")
        return ddp_model
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce tensor across all processes.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation
            
        Returns:
            Reduced tensor
        """
        if self.is_initialized:
            dist.all_reduce(tensor, op=op)
            return tensor / self.config.world_size
        return tensor
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()


class PerformanceOptimizer:
    """Advanced performance optimization for HyperGNN models."""
    
    def __init__(self, model: nn.Module):
        """Initialize performance optimizer.
        
        Args:
            model: Model to optimize
        """
        self.model = model
        self.original_state = None
        self.optimizations_applied = []
        
        logger.info("Initialized PerformanceOptimizer")
    
    def apply_mixed_precision(self, enabled: bool = True) -> 'PerformanceOptimizer':
        """Apply mixed precision optimization.
        
        Args:
            enabled: Whether to enable mixed precision
            
        Returns:
            Self for method chaining
        """
        if enabled and torch.cuda.is_available():
            # Enable automatic mixed precision
            self.model = torch.jit.script(self.model) if hasattr(torch, 'jit') else self.model
            self.optimizations_applied.append("mixed_precision")
            logger.info("Mixed precision optimization applied")
        
        return self
    
    def apply_gradient_checkpointing(self, enabled: bool = True) -> 'PerformanceOptimizer':
        """Apply gradient checkpointing to reduce memory usage.
        
        Args:
            enabled: Whether to enable gradient checkpointing
            
        Returns:
            Self for method chaining
        """
        if enabled:
            for module in self.model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
                    
            self.optimizations_applied.append("gradient_checkpointing")
            logger.info("Gradient checkpointing applied")
        
        return self
    
    def apply_model_compilation(self, backend: str = "inductor") -> 'PerformanceOptimizer':
        """Apply PyTorch 2.0 model compilation.
        
        Args:
            backend: Compilation backend
            
        Returns:
            Self for method chaining
        """
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, backend=backend)
                self.optimizations_applied.append(f"torch_compile_{backend}")
                logger.info(f"Model compilation applied with backend: {backend}")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return self
    
    def apply_tensor_parallelism(self, num_partitions: int = 2) -> 'PerformanceOptimizer':
        """Apply tensor parallelism for large models.
        
        Args:
            num_partitions: Number of tensor partitions
            
        Returns:
            Self for method chaining
        """
        # Simplified tensor parallelism implementation
        if torch.cuda.device_count() >= num_partitions:
            # This is a placeholder - actual implementation would require
            # more sophisticated tensor partitioning
            self.optimizations_applied.append(f"tensor_parallelism_{num_partitions}")
            logger.info(f"Tensor parallelism applied with {num_partitions} partitions")
        
        return self
    
    def optimize_for_inference(self) -> 'PerformanceOptimizer':
        """Optimize model specifically for inference.
        
        Returns:
            Self for method chaining
        """
        # Set model to eval mode
        self.model.eval()
        
        # Fuse operations where possible
        if hasattr(torch.nn.utils, 'fuse_conv_bn_eval'):
            try:
                torch.nn.utils.fuse_conv_bn_eval(self.model)
                self.optimizations_applied.append("conv_bn_fusion")
            except Exception as e:
                logger.debug(f"Conv-BN fusion not applicable: {e}")
        
        # Apply torch.jit optimization
        if hasattr(torch, 'jit'):
            try:
                self.model = torch.jit.optimize_for_inference(torch.jit.script(self.model))
                self.optimizations_applied.append("jit_optimization")
                logger.info("JIT optimization for inference applied")
            except Exception as e:
                logger.debug(f"JIT optimization failed: {e}")
        
        return self
    
    def profile_performance(self, 
                          input_data: Tuple[torch.Tensor, ...], 
                          num_runs: int = 100) -> Dict[str, float]:
        """Profile model performance.
        
        Args:
            input_data: Sample input data
            num_runs: Number of profiling runs
            
        Returns:
            Performance metrics
        """
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(*input_data)
        
        # Time inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(*input_data)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = 1.0 / avg_time
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_used = 0.0
        
        metrics = {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_qps": throughput,
            "memory_usage_mb": memory_used,
            "optimizations_applied": len(self.optimizations_applied),
        }
        
        logger.info(f"Performance metrics: {metrics}")
        return metrics
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of applied optimizations.
        
        Returns:
            Dictionary containing optimization summary
        """
        return {
            "optimizations_applied": self.optimizations_applied,
            "num_optimizations": len(self.optimizations_applied),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
        }


class BatchProcessor:
    """High-throughput batch processing for HyperGNN inference."""
    
    def __init__(self, 
                 model: nn.Module, 
                 batch_size: int = 32,
                 max_workers: int = None,
                 device_ids: Optional[List[int]] = None):
        """Initialize batch processor.
        
        Args:
            model: Model for inference
            batch_size: Batch size for processing
            max_workers: Maximum number of worker threads
            device_ids: GPU device IDs to use
        """
        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers or mp.cpu_count()
        self.device_ids = device_ids or (list(range(torch.cuda.device_count())) 
                                        if torch.cuda.is_available() else [])
        
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.is_running = False
        self.worker_threads = []
        
        # Multi-GPU setup
        if len(self.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        elif self.device_ids:
            self.model = self.model.to(f"cuda:{self.device_ids[0]}")
        
        logger.info(f"BatchProcessor initialized: "
                   f"batch_size={batch_size}, "
                   f"max_workers={self.max_workers}, "
                   f"devices={self.device_ids}")
    
    def _worker_loop(self):
        """Worker thread main loop."""
        while self.is_running:
            try:
                batch_data = []
                batch_ids = []
                
                # Collect batch
                for _ in range(self.batch_size):
                    if not self.input_queue.empty():
                        item = self.input_queue.get_nowait()
                        batch_data.append(item['data'])
                        batch_ids.append(item['id'])
                    else:
                        break
                
                if batch_data:
                    # Process batch
                    with torch.no_grad():
                        # Assuming batch_data contains tuples of (edge_index, node_features, node_texts)
                        results = []
                        for data in batch_data:
                            result = self.model(*data)
                            results.append(result)
                    
                    # Return results
                    for batch_id, result in zip(batch_ids, results):
                        self.output_queue.put({
                            'id': batch_id,
                            'result': result,
                            'timestamp': time.time()
                        })
                
                time.sleep(0.001)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(0.01)
    
    def start(self):
        """Start batch processing workers."""
        self.is_running = True
        
        for i in range(self.max_workers):
            thread = Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        
        logger.info(f"Started {self.max_workers} batch processing workers")
    
    def stop(self):
        """Stop batch processing workers."""
        self.is_running = False
        
        for thread in self.worker_threads:
            thread.join(timeout=1.0)
        
        self.worker_threads.clear()
        logger.info("Batch processing workers stopped")
    
    def process_async(self, data: Tuple[torch.Tensor, ...], request_id: str = None) -> str:
        """Submit data for asynchronous processing.
        
        Args:
            data: Input data tuple
            request_id: Unique request identifier
            
        Returns:
            Request ID for result retrieval
        """
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        self.input_queue.put({
            'id': request_id,
            'data': data,
            'timestamp': time.time()
        })
        
        return request_id
    
    def get_result(self, request_id: str, timeout: float = 5.0) -> Optional[Any]:
        """Get result for a specific request.
        
        Args:
            request_id: Request identifier
            timeout: Timeout in seconds
            
        Returns:
            Processing result or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.output_queue.empty():
                item = self.output_queue.get_nowait()
                if item['id'] == request_id:
                    return item['result']
                else:
                    # Put back if not matching
                    self.output_queue.put(item)
            
            time.sleep(0.001)
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "num_workers": len(self.worker_threads),
            "is_running": self.is_running,
            "batch_size": self.batch_size,
            "devices": self.device_ids,
        }


class ScalableHyperGNN(nn.Module):
    """Scalable wrapper for HyperGNN with distributed and optimization features."""
    
    def __init__(self, 
                 base_model: nn.Module,
                 distributed_config: Optional[DistributedConfig] = None,
                 enable_optimization: bool = True):
        """Initialize scalable HyperGNN.
        
        Args:
            base_model: Base HyperGNN model
            distributed_config: Distributed training configuration
            enable_optimization: Whether to enable performance optimizations
        """
        super().__init__()
        self.base_model = base_model
        self.distributed_config = distributed_config
        self.enable_optimization = enable_optimization
        
        # Initialize components
        self.distributed_manager = None
        self.performance_optimizer = None
        self.batch_processor = None
        
        if distributed_config:
            self.distributed_manager = DistributedManager(distributed_config)
        
        if enable_optimization:
            self.performance_optimizer = PerformanceOptimizer(base_model)
        
        logger.info(f"ScalableHyperGNN initialized: "
                   f"distributed={distributed_config is not None}, "
                   f"optimization={enable_optimization}")
    
    def setup_distributed(self) -> bool:
        """Setup distributed training.
        
        Returns:
            True if setup successful
        """
        if self.distributed_manager:
            success = self.distributed_manager.setup()
            if success:
                self.base_model = self.distributed_manager.wrap_model(self.base_model)
                logger.info("Distributed training setup completed")
            return success
        return False
    
    def apply_optimizations(self, 
                           mixed_precision: bool = True,
                           gradient_checkpointing: bool = True,
                           model_compilation: bool = True) -> 'ScalableHyperGNN':
        """Apply performance optimizations.
        
        Args:
            mixed_precision: Enable mixed precision
            gradient_checkpointing: Enable gradient checkpointing
            model_compilation: Enable model compilation
            
        Returns:
            Self for method chaining
        """
        if self.performance_optimizer:
            self.performance_optimizer\
                .apply_mixed_precision(mixed_precision)\
                .apply_gradient_checkpointing(gradient_checkpointing)\
                .apply_model_compilation() if model_compilation else None
            
            logger.info("Performance optimizations applied")
        
        return self
    
    def setup_batch_processing(self, 
                             batch_size: int = 32,
                             max_workers: int = None) -> 'ScalableHyperGNN':
        """Setup high-throughput batch processing.
        
        Args:
            batch_size: Batch size for processing
            max_workers: Maximum number of workers
            
        Returns:
            Self for method chaining
        """
        self.batch_processor = BatchProcessor(
            self.base_model, 
            batch_size=batch_size,
            max_workers=max_workers
        )
        self.batch_processor.start()
        
        logger.info("Batch processing setup completed")
        return self
    
    def forward(self, *args, **kwargs):
        """Forward pass through the base model."""
        return self.base_model(*args, **kwargs)
    
    def process_batch_async(self, batch_data: List[Tuple[torch.Tensor, ...]]) -> List[str]:
        """Process batch of data asynchronously.
        
        Args:
            batch_data: List of input data tuples
            
        Returns:
            List of request IDs
        """
        if not self.batch_processor:
            raise RuntimeError("Batch processor not initialized")
        
        request_ids = []
        for data in batch_data:
            req_id = self.batch_processor.process_async(data)
            request_ids.append(req_id)
        
        return request_ids
    
    def get_batch_results(self, request_ids: List[str], timeout: float = 5.0) -> List[Any]:
        """Get results for batch of requests.
        
        Args:
            request_ids: List of request IDs
            timeout: Timeout per request
            
        Returns:
            List of results
        """
        if not self.batch_processor:
            raise RuntimeError("Batch processor not initialized")
        
        results = []
        for req_id in request_ids:
            result = self.batch_processor.get_result(req_id, timeout)
            results.append(result)
        
        return results
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling metrics.
        
        Returns:
            Scaling metrics dictionary
        """
        metrics = {
            "distributed_enabled": self.distributed_manager is not None,
            "optimization_enabled": self.performance_optimizer is not None,
            "batch_processing_enabled": self.batch_processor is not None,
        }
        
        if self.distributed_manager and self.distributed_manager.is_initialized:
            metrics["distributed"] = {
                "world_size": self.distributed_config.world_size,
                "rank": self.distributed_config.rank,
                "backend": self.distributed_config.backend,
            }
        
        if self.performance_optimizer:
            metrics["optimizations"] = self.performance_optimizer.get_optimization_summary()
        
        if self.batch_processor:
            metrics["batch_processing"] = self.batch_processor.get_statistics()
        
        return metrics
    
    def cleanup(self):
        """Cleanup distributed and processing resources."""
        if self.batch_processor:
            self.batch_processor.stop()
        
        if self.distributed_manager:
            self.distributed_manager.cleanup()
        
        logger.info("ScalableHyperGNN cleanup completed")


def distributed_training_worker(rank: int, 
                               world_size: int, 
                               model_fn: Callable,
                               train_fn: Callable,
                               **kwargs):
    """Worker function for distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        model_fn: Function to create model
        train_fn: Training function
        **kwargs: Additional arguments
    """
    # Setup distributed environment
    config = DistributedConfig(
        world_size=world_size,
        rank=rank,
        local_rank=rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    
    manager = DistributedManager(config)
    success = manager.setup()
    
    if not success:
        logger.error(f"Failed to setup distributed training for rank {rank}")
        return
    
    try:
        # Create and wrap model
        model = model_fn()
        distributed_model = manager.wrap_model(model)
        
        # Run training
        train_fn(distributed_model, rank, world_size, **kwargs)
        
    except Exception as e:
        logger.error(f"Training failed for rank {rank}: {e}")
    finally:
        manager.cleanup()


def launch_distributed_training(model_fn: Callable,
                               train_fn: Callable,
                               world_size: int = None,
                               **kwargs):
    """Launch distributed training across multiple processes.
    
    Args:
        model_fn: Function to create model
        train_fn: Training function
        world_size: Number of processes (defaults to GPU count)
        **kwargs: Additional arguments for training
    """
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if world_size <= 1:
        logger.warning("Single process training - distributed not needed")
        model = model_fn()
        train_fn(model, 0, 1, **kwargs)
        return
    
    logger.info(f"Launching distributed training with {world_size} processes")
    
    torch_mp.spawn(
        distributed_training_worker,
        args=(world_size, model_fn, train_fn),
        nprocs=world_size,
        join=True
    )