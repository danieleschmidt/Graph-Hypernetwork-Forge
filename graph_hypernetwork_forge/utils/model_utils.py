"""Model utility functions for Graph Hypernetwork Forge."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import numpy as np
from pathlib import Path
import pickle
import json
import gc


def count_parameters(
    model: nn.Module, 
    trainable_only: bool = False,
    exclude_embeddings: bool = False
) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
        exclude_embeddings: Exclude embedding layer parameters
        
    Returns:
        Number of parameters
    """
    total_params = 0
    
    for name, param in model.named_parameters():
        # Skip non-trainable parameters if requested
        if trainable_only and not param.requires_grad:
            continue
        
        # Skip embedding layers if requested
        if exclude_embeddings and 'embedding' in name.lower():
            continue
        
        total_params += param.numel()
    
    return total_params


def calculate_model_size(model: nn.Module, unit: str = 'MB') -> float:
    """
    Calculate model memory size.
    
    Args:
        model: PyTorch model
        unit: Size unit ('MB', 'KB', 'GB')
        
    Returns:
        Model size in specified unit
    """
    total_params = 0
    buffer_size = 0
    
    # Count parameters
    for param in model.parameters():
        total_params += param.numel() * param.element_size()
    
    # Count buffers
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size_bytes = total_params + buffer_size
    
    if unit == 'KB':
        return total_size_bytes / 1024
    elif unit == 'MB':
        return total_size_bytes / (1024 ** 2)
    elif unit == 'GB':
        return total_size_bytes / (1024 ** 3)
    else:
        return total_size_bytes


def clip_gradients(
    model: nn.Module, 
    max_norm: float = 1.0,
    norm_type: int = 2
) -> float:
    """
    Clip model gradients and return the gradient norm.
    
    Args:
        model: PyTorch model
        max_norm: Maximum norm for gradients
        norm_type: Type of norm (1, 2, or inf)
        
    Returns:
        Gradient norm after clipping
    """
    if norm_type == float('inf'):
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.abs().max()
                total_norm = max(total_norm, param_norm.item())
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm, 
            norm_type=norm_type
        )
        if isinstance(total_norm, torch.Tensor):
            total_norm = total_norm.item()
    
    return float(total_norm)


def get_model_device(model: nn.Module) -> torch.device:
    """
    Get the device of a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Device of the model
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        # Model has no parameters, return CPU as default
        return torch.device('cpu')


def move_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    """
    Move model to specified device.
    
    Args:
        model: PyTorch model
        device: Target device
        
    Returns:
        Model moved to device
    """
    model = model.to(device)
    return model


def save_model_state(
    model: nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Save model state dictionary with optional training state.
    
    Args:
        model: PyTorch model
        optimizer: Optional optimizer
        scheduler: Optional learning rate scheduler
        epoch: Optional current epoch
        metadata: Optional additional metadata
        
    Returns:
        State dictionary
    """
    state_dict = {
        'model_state_dict': model.state_dict()
    }
    
    if optimizer is not None:
        state_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        state_dict['scheduler_state_dict'] = scheduler.state_dict()
    
    if epoch is not None:
        state_dict['epoch'] = epoch
    
    if metadata is not None:
        state_dict['metadata'] = metadata
    
    return state_dict


def load_model_state(
    model: nn.Module,
    state_dict: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model state from state dictionary.
    
    Args:
        model: PyTorch model
        state_dict: State dictionary to load
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        strict: Whether to enforce strict loading
        
    Returns:
        Dictionary with loading information
    """
    result = {}
    
    # Load model state
    if 'model_state_dict' in state_dict:
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict['model_state_dict'], 
            strict=strict
        )
        result['missing_keys'] = missing_keys
        result['unexpected_keys'] = unexpected_keys
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        result['optimizer_loaded'] = True
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in state_dict:
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        result['scheduler_loaded'] = True
    
    # Extract metadata
    if 'epoch' in state_dict:
        result['epoch'] = state_dict['epoch']
    
    if 'metadata' in state_dict:
        result['metadata'] = state_dict['metadata']
    
    return result


def freeze_parameters(model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
    """
    Freeze parameters in model layers.
    
    Args:
        model: PyTorch model
        layer_names: Optional list of layer names to freeze (if None, freeze all)
    """
    if layer_names is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Freeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False


def unfreeze_parameters(model: nn.Module, layer_names: Optional[List[str]] = None) -> None:
    """
    Unfreeze parameters in model layers.
    
    Args:
        model: PyTorch model
        layer_names: Optional list of layer names to unfreeze (if None, unfreeze all)
    """
    if layer_names is None:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Unfreeze specific layers
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True


def get_layer_names(model: nn.Module, trainable_only: bool = False) -> List[str]:
    """
    Get names of all layers in the model.
    
    Args:
        model: PyTorch model
        trainable_only: Only return names of trainable layers
        
    Returns:
        List of layer names
    """
    layer_names = []
    
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
        layer_names.append(name)
    
    return layer_names


def initialize_weights(model: nn.Module, method: str = 'xavier_uniform') -> None:
    """
    Initialize model weights using specified method.
    
    Args:
        model: PyTorch model
        method: Initialization method ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            else:
                raise ValueError(f"Unknown initialization method: {method}")
            
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)


def model_summary(model: nn.Module, input_size: Optional[tuple] = None) -> Dict[str, Any]:
    """
    Generate a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Optional input size for parameter calculation
        
    Returns:
        Dictionary containing model summary information
    """
    summary = {}
    
    # Basic information
    summary['total_parameters'] = count_parameters(model)
    summary['trainable_parameters'] = count_parameters(model, trainable_only=True)
    summary['model_size_mb'] = calculate_model_size(model)
    summary['device'] = str(get_model_device(model))
    
    # Layer information
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            layer_info = {
                'name': name,
                'type': type(module).__name__,
                'parameters': sum(p.numel() for p in module.parameters())
            }
            layers.append(layer_info)
    
    summary['layers'] = layers
    summary['num_layers'] = len(layers)
    
    return summary


def check_gradients(model: nn.Module) -> Dict[str, Any]:
    """
    Check gradient statistics for debugging.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with gradient statistics
    """
    grad_stats = {
        'has_gradients': False,
        'gradient_norm': 0.0,
        'max_gradient': 0.0,
        'min_gradient': 0.0,
        'num_zero_gradients': 0,
        'num_nan_gradients': 0,
        'layer_gradients': {}
    }
    
    gradients = []
    zero_count = 0
    nan_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats['has_gradients'] = True
            grad_data = param.grad.data.flatten()
            
            # Collect all gradients
            gradients.extend(grad_data.tolist())
            
            # Count zeros and NaNs
            zero_count += (grad_data == 0).sum().item()
            nan_count += torch.isnan(grad_data).sum().item()
            
            # Per-layer statistics
            grad_stats['layer_gradients'][name] = {
                'norm': param.grad.data.norm().item(),
                'max': param.grad.data.max().item(),
                'min': param.grad.data.min().item(),
                'mean': param.grad.data.mean().item(),
                'std': param.grad.data.std().item()
            }
    
    if gradients:
        gradients = torch.tensor(gradients)
        grad_stats['gradient_norm'] = gradients.norm().item()
        grad_stats['max_gradient'] = gradients.max().item()
        grad_stats['min_gradient'] = gradients.min().item()
        grad_stats['num_zero_gradients'] = zero_count
        grad_stats['num_nan_gradients'] = nan_count
    
    return grad_stats


def save_model_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint to file.
    
    Args:
        model: PyTorch model
        filepath: Path to save checkpoint
        optimizer: Optional optimizer
        scheduler: Optional scheduler
        epoch: Optional current epoch
        loss: Optional current loss
        metadata: Optional additional metadata
    """
    checkpoint = save_model_state(model, optimizer, scheduler, epoch, metadata)
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, filepath)


def load_model_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint from file.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optional optimizer
        scheduler: Optional scheduler
        device: Optional device to load to
        strict: Whether to enforce strict loading
        
    Returns:
        Dictionary with loading information
    """
    if device is None:
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=device)
    
    result = load_model_state(model, checkpoint, optimizer, scheduler, strict)
    
    if 'loss' in checkpoint:
        result['loss'] = checkpoint['loss']
    
    return result


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()