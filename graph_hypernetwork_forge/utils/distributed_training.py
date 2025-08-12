"""Simplified Distributed Training Framework for Graph Hypernetwork Forge."""

import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import logging


def get_logger(name):
    return logging.getLogger(name)


logger = get_logger(__name__)


class DistributedTrainer:
    """Basic distributed training framework."""
    
    def __init__(self, world_size: int = 1, rank: int = 0):
        """Initialize distributed trainer."""
        self.world_size = world_size
        self.rank = rank
        self.is_main_process = rank == 0
        logger.info(f"DistributedTrainer initialized for rank {rank}/{world_size}")
    
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for distributed training."""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.rank}")
            model = model.to(device)
        
        if self.world_size > 1 and dist.is_initialized():
            model = DDP(model, device_ids=[self.rank] if torch.cuda.is_available() else None)
        
        return model
    
    def train_step(self, model: nn.Module, batch, loss_fn) -> float:
        """Execute single training step."""
        model.train()
        
        # Move batch to device
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.rank}")
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch) if isinstance(batch, dict) else model(batch)
        loss = loss_fn(outputs, batch)
        
        return loss.item()


def get_world_size() -> int:
    """Get world size."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if current process is main process."""
    return get_rank() == 0