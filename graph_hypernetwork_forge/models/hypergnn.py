"""Core HyperGNN model implementation with comprehensive error handling."""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, MessagePassing
from transformers import AutoModel, AutoTokenizer

# Import enhanced error handling and logging (with fallbacks)
try:
    from ..utils.logging_utils import get_logger, LoggerMixin, log_function_call
    from ..utils.exceptions import (
        ValidationError, ModelError, GPUError, MemoryError, NetworkError, 
        InferenceError, handle_cuda_out_of_memory, log_and_raise_error
    )
    from ..utils.memory_utils import (
        check_gpu_memory_available, estimate_tensor_memory, safe_cuda_operation,
        memory_management
    )
    ENHANCED_FEATURES = True
except ImportError:
    # Fallback implementations
    def log_function_call(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)
    
    class LoggerMixin:
        def __init__(self, *args, **kwargs):
            pass
    
    class ValidationError(Exception):
        pass
    
    class ModelError(Exception):
        pass
    
    class GPUError(Exception):
        pass
        
    class MemoryError(Exception):
        pass
    
    class NetworkError(Exception):
        pass
    
    class InferenceError(Exception):
        pass
    
    def handle_cuda_out_of_memory(func):
        return func
    
    def log_and_raise_error(msg, exception_type=Exception):
        raise exception_type(msg)
    
    def check_gpu_memory_available(min_gb=1):
        return True
    
    def estimate_tensor_memory(shape, dtype=torch.float32):
        return 0
    
    def safe_cuda_operation(func, *args, **kwargs):
        # Check if func is a no-argument lambda (from existing code)
        import inspect
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            return func()
        else:
            return func(*args, **kwargs)
    
    def memory_management(*args, **kwargs):
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()
    
    ENHANCED_FEATURES = False

# Import optimization utilities
try:
    from ..utils.optimization import WeightCache, profile_function, AdaptiveDropout
    from ..utils.advanced_resilience import (
        ResilientModelWrapper, AdaptiveCircuitBreaker, ExponentialBackoffRetry,
        ResourceGuard, AutoHealingManager, resilient_model_call
    )
    from ..utils.distributed_optimization import (
        ScalableHyperGNN, PerformanceOptimizer, BatchProcessor,
        DistributedConfig, DistributedManager
    )
    RESILIENCE_FEATURES = True
    SCALING_FEATURES = True
except ImportError:
    # Fallback for when utils are not available
    WeightCache = None
    def profile_function(name, profiler=None): 
        def decorator(func): return func
        return decorator
    AdaptiveDropout = nn.Dropout
    ResilientModelWrapper = None
    def resilient_model_call(*args, **kwargs):
        def decorator(func): return func
        return decorator
    ScalableHyperGNN = None
    PerformanceOptimizer = None
    BatchProcessor = None
    DistributedConfig = None
    DistributedManager = None
    RESILIENCE_FEATURES = False
    SCALING_FEATURES = False

# Initialize logger
logger = get_logger(__name__)


def _validate_tensor_input(tensor: torch.Tensor, name: str, expected_dims: int, min_size: Optional[int] = None) -> None:
    """Validate tensor input parameters.
    
    Args:
        tensor: Input tensor to validate
        name: Name of the tensor for error messages
        expected_dims: Expected number of dimensions
        min_size: Minimum size for the last dimension
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(name, type(tensor).__name__, "torch.Tensor")
    
    if tensor.dim() != expected_dims:
        raise ValidationError(
            f"{name}.dims", tensor.dim(), f"{expected_dims} dimensions"
        )
    
    if min_size is not None and tensor.size(-1) < min_size:
        raise ValidationError(
            f"{name}.size(-1)", tensor.size(-1), f"at least {min_size}"
        )
    
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValidationError(name, "contains NaN/Inf values", "finite values only")


def _validate_text_input(texts: List[str], name: str = "texts") -> None:
    """Validate text input parameters.
    
    Args:
        texts: List of text strings to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(texts, list):
        raise ValidationError(name, type(texts).__name__, "list of strings")
    
    if len(texts) == 0:
        raise ValidationError(name, "empty list", "non-empty list")
    
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValidationError(f"{name}[{i}]", type(text).__name__, "string")
        
        if len(text.strip()) == 0:
            raise ValidationError(f"{name}[{i}]", "empty string", "non-empty string")


class TextEncoder(nn.Module, LoggerMixin):
    """Encodes text descriptions into embeddings with comprehensive error handling."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        freeze_encoder: bool = False,
    ):
        """Initialize text encoder with validation and error handling.
        
        Args:
            model_name: Pre-trained model name or path
            embedding_dim: Output embedding dimension
            freeze_encoder: Whether to freeze encoder weights
            
        Raises:
            ValidationError: If parameters are invalid
            NetworkError: If model cannot be loaded
            ModelError: If encoder initialization fails
        """
        super().__init__()
        
        # Validate parameters
        if not isinstance(model_name, str) or len(model_name.strip()) == 0:
            raise ValidationError("model_name", model_name, "non-empty string")
        
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValidationError("embedding_dim", embedding_dim, "positive integer")
        
        if not isinstance(freeze_encoder, bool):
            raise ValidationError("freeze_encoder", freeze_encoder, "boolean")
        
        # Initialize logger
        if ENHANCED_FEATURES:
            self.logger = get_logger(self.__class__.__name__)
        else:
            self.logger = get_logger(__name__)
            
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.freeze_encoder = freeze_encoder
        
        self.logger.info(f"Initializing TextEncoder with model: {model_name}, dim: {embedding_dim}")
        
        # Initialize encoder based on model type with error handling
        try:
            if "sentence-transformers" in model_name:
                self.logger.info(f"Loading SentenceTransformer model: {model_name}")
                self.encoder = SentenceTransformer(model_name)
                self.is_sentence_transformer = True
                # Get actual embedding dimension
                self.input_dim = self.encoder.get_sentence_embedding_dimension()
                self.logger.info(f"SentenceTransformer loaded, input_dim: {self.input_dim}")
            else:
                # Use Hugging Face transformers
                self.logger.info(f"Loading HuggingFace model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.encoder = AutoModel.from_pretrained(model_name)
                self.is_sentence_transformer = False
                self.input_dim = self.encoder.config.hidden_size
                self.logger.info(f"HuggingFace model loaded, input_dim: {self.input_dim}")
                
        except Exception as e:
            error_msg = f"Failed to load text encoder model '{model_name}': {e}"
            self.logger.error(error_msg)
            if "network" in str(e).lower() or "connection" in str(e).lower() or "timeout" in str(e).lower():
                raise NetworkError(model_name)
            else:
                raise ModelError("TextEncoder", "initialization", error_msg)
        
        # Projection layer if dimensions don't match
        try:
            if self.input_dim != embedding_dim:
                self.projection = nn.Linear(self.input_dim, embedding_dim)
                self.logger.info(f"Created projection layer: {self.input_dim} -> {embedding_dim}")
            else:
                self.projection = nn.Identity()
                self.logger.info("No projection layer needed (dimensions match)")
            
            # Freeze encoder if requested
            if freeze_encoder:
                self._freeze_encoder()
                self.logger.info("Encoder weights frozen")
                
        except Exception as e:
            raise ModelError("TextEncoder", "layer_creation", f"Failed to create projection layer: {e}")
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        if self.is_sentence_transformer:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    @log_function_call()
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into embeddings with comprehensive error handling.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            Text embeddings [batch_size, embedding_dim]
            
        Raises:
            ValidationError: If input texts are invalid
            GPUError: If CUDA operation fails
            ModelError: If encoding fails
        """
        # Validate input
        _validate_text_input(texts, "texts")
        
        self.logger.debug(f"Encoding {len(texts)} texts")
        
        # Check GPU memory if using CUDA
        if next(self.parameters()).is_cuda:
            estimated_memory = estimate_tensor_memory((len(texts), self.embedding_dim))
            check_gpu_memory_available(estimated_memory * 2, "text encoding")  # 2x for safety
        
        try:
            if self.is_sentence_transformer:
                # Use sentence-transformers
                if self.freeze_encoder:
                    with torch.no_grad():
                        embeddings = self.encoder.encode(
                            texts, convert_to_tensor=True, show_progress_bar=False
                        )
                        # Clone to make it compatible with autograd
                        embeddings = embeddings.clone().detach().requires_grad_(False)
                else:
                    embeddings = self.encoder.encode(
                        texts, convert_to_tensor=True, show_progress_bar=False
                    )
                    # Ensure autograd compatibility
                    if not embeddings.requires_grad:
                        embeddings = embeddings.clone().requires_grad_(True)
            else:
                # Use transformers
                inputs = self.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                inputs = {k: v.to(next(self.encoder.parameters()).device) for k, v in inputs.items()}
                
                with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                    outputs = self.encoder(**inputs)
                    # Use [CLS] token embedding
                    embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Apply projection with error handling
            embeddings = safe_cuda_operation(
                lambda: self.projection(embeddings),
                "projection layer forward pass"
            )
            
            # Validate output
            _validate_tensor_input(embeddings, "output_embeddings", 2)
            
            if embeddings.size(1) != self.embedding_dim:
                raise ModelError(
                    "TextEncoder", "forward",
                    f"Output dimension mismatch: got {embeddings.size(1)}, expected {self.embedding_dim}"
                )
            
            self.logger.debug(f"Successfully encoded texts to shape {embeddings.shape}")
            return embeddings
            
        except torch.cuda.OutOfMemoryError as e:
            raise handle_cuda_out_of_memory("text encoding")
        except Exception as e:
            error_msg = f"Text encoding failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise ModelError("TextEncoder", "forward", error_msg)


class HyperNetwork(nn.Module):
    """Generates GNN weights from text embeddings."""
    
    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        num_layers: int,
        gnn_type: str = "GAT",
        dropout: float = 0.1,
    ):
        """Initialize hypernetwork.
        
        Args:
            text_dim: Text embedding dimension
            hidden_dim: GNN hidden dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN backbone (GCN, GAT, SAGE)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Initialize logger
        if ENHANCED_FEATURES:
            self.logger = get_logger(self.__class__.__name__)
        else:
            self.logger = get_logger(__name__)
            
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.upper()
        self.dropout = dropout
        
        # Calculate weight dimensions based on GNN type
        self.weight_dims = self._calculate_weight_dimensions()
        
        # Hypernetwork layers
        self.hypernetwork = nn.ModuleDict()
        
        # Generate weights for each GNN layer
        for layer_idx in range(num_layers):
            layer_generators = nn.ModuleDict()
            
            for weight_name, weight_shape in self.weight_dims[layer_idx].items():
                # Create generator for this weight
                weight_size = torch.prod(torch.tensor(weight_shape)).item()
                
                generator = nn.Sequential(
                    nn.Linear(self.text_dim, self.text_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.text_dim * 2, self.text_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.text_dim, weight_size),
                )
                
                layer_generators[weight_name] = generator
            
            self.hypernetwork[f"layer_{layer_idx}"] = layer_generators
        
        # Weight normalization factors
        self.weight_scales = nn.ParameterDict()
        for layer_idx in range(num_layers):
            layer_scales = nn.ParameterDict()
            for weight_name in self.weight_dims[layer_idx].keys():
                layer_scales[weight_name] = nn.Parameter(torch.ones(1) * 0.1)
            self.weight_scales[f"layer_{layer_idx}"] = layer_scales
    
    def _calculate_weight_dimensions(self) -> List[Dict[str, Tuple[int, ...]]]:
        """Calculate weight dimensions for each layer."""
        weight_dims = []
        
        for layer_idx in range(self.num_layers):
            layer_dims = {}
            
            if layer_idx == 0:
                # First layer: input features to hidden
                in_dim = None  # Will be set dynamically
                out_dim = self.hidden_dim
            elif layer_idx == self.num_layers - 1:
                # Last layer: hidden to output
                in_dim = self.hidden_dim
                out_dim = None  # Will be set dynamically
            else:
                # Middle layers: hidden to hidden
                in_dim = self.hidden_dim
                out_dim = self.hidden_dim
            
            # Weight dimensions depend on GNN type
            if self.gnn_type == "GCN":
                layer_dims["weight"] = (in_dim or self.hidden_dim, out_dim or self.hidden_dim)
                layer_dims["bias"] = (out_dim or self.hidden_dim,)
            elif self.gnn_type == "GAT":
                # For simplicity, assume single head GAT
                layer_dims["weight"] = (in_dim or self.hidden_dim, out_dim or self.hidden_dim)
                layer_dims["att_weight"] = (2 * (out_dim or self.hidden_dim), 1)
                layer_dims["bias"] = (out_dim or self.hidden_dim,)
            elif self.gnn_type == "SAGE":
                layer_dims["lin_l"] = (in_dim or self.hidden_dim, out_dim or self.hidden_dim)
                layer_dims["lin_r"] = (in_dim or self.hidden_dim, out_dim or self.hidden_dim)
                layer_dims["bias"] = (out_dim or self.hidden_dim,)
            else:
                raise ValidationError("gnn_type", self.gnn_type, "GCN, GAT, or SAGE")
            
            weight_dims.append(layer_dims)
        
        return weight_dims
    
    def forward(
        self, 
        text_embeddings: torch.Tensor,
        input_dim: int,
        output_dim: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate GNN weights from text embeddings.
        
        Args:
            text_embeddings: Text embeddings [num_nodes, text_dim]
            input_dim: Input feature dimension
            output_dim: Output dimension
            
        Returns:
            List of weight dictionaries for each layer
        """
        batch_size = text_embeddings.size(0)
        generated_weights = []
        
        for layer_idx in range(self.num_layers):
            layer_weights = {}
            layer_generators = self.hypernetwork[f"layer_{layer_idx}"]
            layer_scales = self.weight_scales[f"layer_{layer_idx}"]
            
            # Update dimensions for first and last layers
            updated_dims = self.weight_dims[layer_idx].copy()
            
            # Determine actual output dimension for this layer
            if layer_idx == self.num_layers - 1:
                actual_output_dim = output_dim
            else:
                actual_output_dim = self.hidden_dim
            
            # Update first layer input dimensions
            if layer_idx == 0:
                for key in updated_dims:
                    if key == "weight" or "lin_l" in key or "lin_r" in key:
                        shape = list(updated_dims[key])
                        shape[0] = input_dim
                        updated_dims[key] = tuple(shape)
            
            # Update output dimensions for all layers
            for key in updated_dims:
                if key == "weight" or "lin_l" in key or "lin_r" in key:
                    shape = list(updated_dims[key])
                    shape[1] = actual_output_dim
                    updated_dims[key] = tuple(shape)
                elif "bias" in key:
                    updated_dims[key] = (actual_output_dim,)
                elif key == "att_weight":
                    shape = list(updated_dims[key])
                    shape[0] = 2 * actual_output_dim  # 2 * output_dim for concatenated attention
                    updated_dims[key] = tuple(shape)
            
            for weight_name, weight_shape in updated_dims.items():
                # Generate weights for each node
                flat_weights = layer_generators[weight_name](text_embeddings)
                
                # Calculate expected output size
                target_elements = torch.prod(torch.tensor(weight_shape)).item()
                
                # Ensure correct output size
                if flat_weights.size(-1) != target_elements:
                    raise RuntimeError(f"Weight generator for {weight_name} in layer {layer_idx} "
                                     f"produces {flat_weights.size(-1)} elements, "
                                     f"but expected {target_elements} for shape {weight_shape}")
                
                # Reshape to proper weight shape
                weight_tensor = flat_weights.view(batch_size, *weight_shape)
                
                # Apply scaling
                scale = layer_scales[weight_name]
                weight_tensor = weight_tensor * scale
                
                layer_weights[weight_name] = weight_tensor
            
            generated_weights.append(layer_weights)
        
        return generated_weights


class DynamicGNN(nn.Module):
    """Dynamic GNN that uses generated weights."""
    
    def __init__(self, gnn_type: str = "GAT", dropout: float = 0.1):
        """Initialize dynamic GNN.
        
        Args:
            gnn_type: Type of GNN (GCN, GAT, SAGE)
            dropout: Dropout probability
        """
        super().__init__()
        
        # Initialize logger
        if ENHANCED_FEATURES:
            self.logger = get_logger(self.__class__.__name__)
        else:
            self.logger = get_logger(__name__)
            
        self.gnn_type = gnn_type.upper()
        self.dropout = dropout
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        generated_weights: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass with generated weights.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            generated_weights: Generated weights for each layer
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        current_x = x
        
        for layer_idx, layer_weights in enumerate(generated_weights):
            # Apply GNN layer with generated weights
            if self.gnn_type == "GCN":
                current_x = self._gcn_layer(current_x, edge_index, layer_weights)
            elif self.gnn_type == "GAT":
                current_x = self._gat_layer(current_x, edge_index, layer_weights)
            elif self.gnn_type == "SAGE":
                current_x = self._sage_layer(current_x, edge_index, layer_weights)
            else:
                raise ValidationError("gnn_type", self.gnn_type, "GCN, GAT, or SAGE")
            
            # Apply activation and dropout (except for last layer)
            if layer_idx < len(generated_weights) - 1:
                current_x = F.relu(current_x)
                current_x = F.dropout(current_x, p=self.dropout, training=self.training)
        
        return current_x
    
    def _gcn_layer(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """GCN layer with dynamic weights."""
        # Simple GCN implementation
        # x: [num_nodes, in_dim], weight: [num_nodes, in_dim, out_dim]
        weight = weights["weight"]  # [num_nodes, in_dim, out_dim]
        bias = weights["bias"]      # [num_nodes, out_dim]
        
        # Apply linear transformation per node
        out = torch.bmm(x.unsqueeze(1), weight).squeeze(1)  # [num_nodes, out_dim]
        out = out + bias
        
        # Apply message passing (simplified aggregation)
        row, col = edge_index
        out_messages = out[row]  # Messages from source nodes
        
        # Aggregate messages at target nodes
        out_agg = torch.zeros_like(out)
        out_agg.scatter_add_(0, col.unsqueeze(1).expand(-1, out.size(1)), out_messages)
        
        # Add self-loops effect
        out_agg = out_agg + out
        
        return out_agg
    
    def _gat_layer(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """GAT layer with dynamic weights."""
        weight = weights["weight"]      # [num_nodes, in_dim, out_dim]
        att_weight = weights["att_weight"]  # [num_nodes, 2*out_dim, 1]
        bias = weights["bias"]          # [num_nodes, out_dim]
        
        # Linear transformation
        h = torch.bmm(x.unsqueeze(1), weight).squeeze(1)  # [num_nodes, out_dim]
        
        # Attention mechanism (simplified)
        row, col = edge_index
        h_i = h[row]  # Source nodes
        h_j = h[col]  # Target nodes
        
        # Attention scores
        att_input = torch.cat([h_i, h_j], dim=1)  # [num_edges, 2*out_dim]
        # Use source node attention weights for each edge
        source_att_weights = att_weight[row]  # [num_edges, 2*out_dim, 1]
        # Batch matrix multiplication for attention scores
        att_scores = torch.bmm(att_input.unsqueeze(1), source_att_weights).squeeze()  # [num_edges]
        att_scores = F.leaky_relu(att_scores, 0.2)
        
        # Apply softmax per target node
        att_weights = torch.zeros_like(att_scores)
        for node in torch.unique(col):
            mask = col == node
            att_weights[mask] = F.softmax(att_scores[mask], dim=0)
        
        # Aggregate with attention
        weighted_messages = h_i * att_weights.unsqueeze(1)
        out = torch.zeros_like(h)
        out.scatter_add_(0, col.unsqueeze(1).expand(-1, h.size(1)), weighted_messages)
        
        # Add bias
        out = out + bias
        
        return out
    
    def _sage_layer(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """SAGE layer with dynamic weights."""
        lin_l = weights["lin_l"]  # [num_nodes, in_dim, out_dim]
        lin_r = weights["lin_r"]  # [num_nodes, in_dim, out_dim] 
        bias = weights["bias"]    # [num_nodes, out_dim]
        
        # Self transformation
        out_self = torch.bmm(x.unsqueeze(1), lin_l).squeeze(1)
        
        # Neighbor aggregation
        row, col = edge_index
        neighbor_messages = x[row]  # Messages from neighbors
        
        # Mean aggregation
        out_neigh = torch.zeros_like(x)
        ones = torch.ones(neighbor_messages.size(0), device=x.device)
        neighbor_count = torch.zeros(x.size(0), device=x.device)
        
        out_neigh.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), neighbor_messages)
        neighbor_count.scatter_add_(0, col, ones)
        neighbor_count = neighbor_count.clamp(min=1).unsqueeze(1)
        out_neigh = out_neigh / neighbor_count
        
        # Transform neighbor features
        out_neigh = torch.bmm(out_neigh.unsqueeze(1), lin_r).squeeze(1)
        
        # Concatenate and add bias
        out = out_self + out_neigh + bias
        
        return out


class HyperGNN(nn.Module, LoggerMixin):
    """Main HyperGNN model with comprehensive error handling and validation."""
    
    def __init__(
        self,
        text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone: str = "GAT",
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        freeze_text_encoder: bool = False,
        enable_caching: bool = True,
        cache_size: int = 1000,
    ):
        """Initialize HyperGNN model with comprehensive validation.
        
        Args:
            text_encoder: Text encoder model name
            gnn_backbone: GNN backbone type (GCN, GAT, SAGE)
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
            freeze_text_encoder: Whether to freeze text encoder
            enable_caching: Whether to enable weight caching
            cache_size: Maximum number of cached weight sets
            
        Raises:
            ValidationError: If parameters are invalid
            ModelError: If model initialization fails
            NetworkError: If text encoder cannot be loaded
        """
        super().__init__()
        
        # Initialize logger from LoggerMixin
        if ENHANCED_FEATURES:
            self.logger = get_logger(self.__class__.__name__)
        else:
            self.logger = get_logger(__name__)
        
        # Validate all parameters
        if not isinstance(text_encoder, str) or len(text_encoder.strip()) == 0:
            raise ValidationError("text_encoder", text_encoder, "non-empty string")
        
        if not isinstance(gnn_backbone, str):
            raise ValidationError("gnn_backbone", gnn_backbone, "string")
        
        if gnn_backbone.upper() not in ["GCN", "GAT", "SAGE"]:
            raise ValidationError("gnn_backbone", gnn_backbone, "one of: GCN, GAT, SAGE")
        
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValidationError("hidden_dim", hidden_dim, "positive integer")
        
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValidationError("num_layers", num_layers, "positive integer")
        
        if not isinstance(dropout, (int, float)) or not (0.0 <= dropout <= 1.0):
            raise ValidationError("dropout", dropout, "float between 0.0 and 1.0")
        
        if not isinstance(freeze_text_encoder, bool):
            raise ValidationError("freeze_text_encoder", freeze_text_encoder, "boolean")
        
        if not isinstance(enable_caching, bool):
            raise ValidationError("enable_caching", enable_caching, "boolean")
        
        if not isinstance(cache_size, int) or cache_size <= 0:
            raise ValidationError("cache_size", cache_size, "positive integer")
        
        self.text_encoder_name = text_encoder
        self.gnn_backbone = gnn_backbone
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.logger.info(f"Initializing HyperGNN: encoder={text_encoder}, backbone={gnn_backbone}, "
                        f"hidden_dim={hidden_dim}, num_layers={num_layers}")
        
        # Optimization features
        self.enable_caching = enable_caching
        if enable_caching and WeightCache is not None:
            self.weight_cache = WeightCache(max_size=cache_size)
        else:
            self.weight_cache = None
        
        # Initialize resilience components if available
        self.resilience_enabled = RESILIENCE_FEATURES
        if RESILIENCE_FEATURES:
            self.circuit_breaker = AdaptiveCircuitBreaker(failure_threshold=3)
            self.retry_strategy = ExponentialBackoffRetry(max_retries=2)
            self.resource_guard = ResourceGuard()
            self.auto_healer = None  # Will be initialized after model creation
            self.logger.info("Resilience features enabled: circuit breaker, retry, resource guard")
        
        # Initialize scaling components if available
        self.scaling_enabled = SCALING_FEATURES
        self.performance_optimizer = None
        self.batch_processor = None
        if SCALING_FEATURES:
            self.logger.info("Scaling features available: performance optimization, batch processing, distributed training")
        
        # Initialize components with error handling
        try:
            self.logger.info("Initializing TextEncoder...")
            self.text_encoder = TextEncoder(
                model_name=text_encoder,
                embedding_dim=hidden_dim,
                freeze_encoder=freeze_text_encoder,
            )
            
            self.logger.info("Initializing HyperNetwork...")
            self.hypernetwork = HyperNetwork(
                text_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                gnn_type=gnn_backbone,
                dropout=dropout,
            )
            
            self.logger.info("Initializing DynamicGNN...")
            self.dynamic_gnn = DynamicGNN(
                gnn_type=gnn_backbone,
                dropout=dropout,
            )
            
            self.logger.info("HyperGNN initialization completed successfully")
            
            # Initialize auto-healer after all components are ready
            if self.resilience_enabled:
                self.auto_healer = AutoHealingManager(self)
                self.logger.info("Auto-healing manager initialized")
            
        except Exception as e:
            error_msg = f"Failed to initialize HyperGNN components: {e}"
            self.logger.error(error_msg, exc_info=True)
            if isinstance(e, (ValidationError, NetworkError, ModelError)):
                raise
            else:
                raise ModelError("HyperGNN", "initialization", error_msg)
    
    @profile_function("generate_weights")
    def generate_weights(self, node_texts: List[str], return_flat: bool = True) -> Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Generate GNN weights from node texts with caching optimization.
        
        Args:
            node_texts: List of node text descriptions
            
        Returns:
            Generated weights for each GNN layer
        """
        # Use model's hidden dimensions
        input_dim = self.hidden_dim
        output_dim = self.hidden_dim
        
        # Check cache first  
        if self.weight_cache is not None:
            cached_weights = self.weight_cache.get(node_texts, input_dim, output_dim)
            if cached_weights is not None:
                return cached_weights
        
        # Generate weights (most expensive operations)
        text_embeddings = self.text_encoder(node_texts)
        weights = self.hypernetwork(text_embeddings, input_dim, output_dim)
        
        # Cache for future use (deep copy to avoid reference issues)
        if self.weight_cache is not None:
            cached_weights = [{k: v.clone().detach() for k, v in layer.items()} for layer in weights]
            self.weight_cache.put(node_texts, input_dim, output_dim, cached_weights)
        
        # Return format based on preference
        if return_flat and len(weights) == 1:
            # For single layer, return the layer dict directly
            return weights[0]
        elif return_flat:
            # For multiple layers, create a flattened structure
            flat_weights = {}
            for i, layer in enumerate(weights):
                for key, value in layer.items():
                    flat_weights[f"layer_{i}_{key}"] = value
            return flat_weights
        else:
            return weights
    
    @profile_function("hypergnn_forward")
    @log_function_call()
    def forward(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        node_texts: List[str],
    ) -> torch.Tensor:
        """Forward pass of HyperGNN with comprehensive validation and error handling.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            node_features: Node features [num_nodes, feature_dim]
            node_texts: List of node text descriptions
            
        Returns:
            Node embeddings [num_nodes, output_dim]
            
        Raises:
            ValidationError: If inputs are invalid
            InferenceError: If inference fails
            GPUError: If CUDA operation fails
        """
        # Comprehensive input validation
        _validate_tensor_input(edge_index, "edge_index", 2)
        _validate_tensor_input(node_features, "node_features", 2)
        _validate_text_input(node_texts, "node_texts")
        
        if edge_index.size(0) != 2:
            raise ValidationError(
                "edge_index.size(0)", edge_index.size(0), "2 (source and target nodes)"
            )
        
        if len(node_texts) != node_features.size(0):
            raise ValidationError(
                "node_texts length", len(node_texts), 
                f"match node_features.size(0)={node_features.size(0)}"
            )
        
        # Check edge indices are valid
        if edge_index.numel() > 0:
            max_edge_idx = edge_index.max().item()
            if max_edge_idx >= node_features.size(0):
                raise ValidationError(
                    "edge_index", f"max index {max_edge_idx}", 
                    f"less than num_nodes ({node_features.size(0)})"
                )
        
        # Check GPU memory requirements
        if node_features.is_cuda:
            estimated_memory = (
                estimate_tensor_memory(node_features.shape) * 3 +  # features, embeddings, weights
                estimate_tensor_memory((len(node_texts), self.hidden_dim)) * 2  # text embeddings
            )
            check_gpu_memory_available(estimated_memory, "HyperGNN forward pass")
        
        self.logger.debug(f"HyperGNN forward: {node_features.size(0)} nodes, {edge_index.size(1)} edges")
        
        try:
            # Encode texts to get embeddings
            text_embeddings = self.text_encoder(node_texts)
            
            # Generate GNN weights
            input_dim = node_features.size(1)
            output_dim = self.hidden_dim  # Can be made configurable
            
            generated_weights = self.hypernetwork(
                text_embeddings, input_dim, output_dim
            )
            
            # Apply dynamic GNN
            node_embeddings = self.dynamic_gnn(
                node_features, edge_index, generated_weights
            )
            
            # Final validation
            _validate_tensor_input(node_embeddings, "output_embeddings", 2)
            
            if node_embeddings.size(0) != node_features.size(0):
                raise InferenceError(
                    node_features.shape, "HyperGNN",
                    f"Output node count mismatch: got {node_embeddings.size(0)}, expected {node_features.size(0)}"
                )
            
            self.logger.debug(f"HyperGNN forward completed successfully, output shape: {node_embeddings.shape}")
            return node_embeddings
            
        except torch.cuda.OutOfMemoryError:
            raise handle_cuda_out_of_memory("HyperGNN forward pass")
        except Exception as e:
            if isinstance(e, (ValidationError, InferenceError, ModelError, GPUError)):
                raise
            error_msg = f"HyperGNN forward pass failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise InferenceError(node_features.shape, "HyperGNN", error_msg)
    
    def forward_with_weights(
        self,
        edge_index: torch.Tensor, 
        node_features: torch.Tensor,
        generated_weights: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        """Forward pass using precomputed weights.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]  
            node_features: Node features [num_nodes, feature_dim]
            generated_weights: Precomputed GNN weights
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Convert flat weights back to list format if needed
        if isinstance(generated_weights, dict):
            if any(key.startswith('layer_') for key in generated_weights.keys()):
                # Convert flat format back to list format
                layer_weights = []
                layer_indices = set()
                for key in generated_weights.keys():
                    if key.startswith('layer_'):
                        layer_idx = int(key.split('_')[1])
                        layer_indices.add(layer_idx)
                
                for i in sorted(layer_indices):
                    layer_dict = {}
                    prefix = f"layer_{i}_"
                    for key, value in generated_weights.items():
                        if key.startswith(prefix):
                            weight_name = key[len(prefix):]
                            layer_dict[weight_name] = value
                    if layer_dict:
                        layer_weights.append(layer_dict)
                
                generated_weights = layer_weights
            else:
                # Single layer dict
                generated_weights = [generated_weights]
        
        # Use DynamicGNN with precomputed weights
        return self.dynamic_gnn(node_features, edge_index, generated_weights)
    
    @log_function_call()
    def predict(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        node_texts: List[str],
    ) -> torch.Tensor:
        """Prediction interface for inference with memory management.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            node_features: Node features [num_nodes, feature_dim]
            node_texts: List of node text descriptions
            
        Returns:
            Predictions [num_nodes, output_dim]
            
        Raises:
            ValidationError: If inputs are invalid
            InferenceError: If prediction fails
        """
        with memory_management(cleanup_on_exit=True):
            self.eval()
            self.logger.debug("Starting prediction in eval mode")
            
            try:
                with torch.no_grad():
                    result = self.forward(edge_index, node_features, node_texts)
                    self.logger.debug("Prediction completed successfully")
                    return result
            except Exception as e:
                error_msg = f"Prediction failed: {e}"
                self.logger.error(error_msg, exc_info=True)
                if isinstance(e, (ValidationError, InferenceError)):
                    raise
                raise InferenceError(node_features.shape, "HyperGNN.predict", error_msg)
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "text_encoder": self.text_encoder_name,
            "gnn_backbone": self.gnn_backbone,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
    
    def forward_resilient(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        node_texts: List[str],
    ) -> torch.Tensor:
        """Resilient forward pass with comprehensive error recovery.
        
        This method provides advanced error handling, automatic recovery,
        and resource management for production deployments.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            node_features: Node features [num_nodes, feature_dim]
            node_texts: List of node text descriptions
            
        Returns:
            Node embeddings [num_nodes, output_dim]
            
        Raises:
            ValidationError: If inputs are invalid
            InferenceError: If inference fails after all recovery attempts
        """
        if not self.resilience_enabled:
            return self.forward(edge_index, node_features, node_texts)
        
        def protected_forward():
            return self.forward(edge_index, node_features, node_texts)
        
        try:
            # Execute with resource protection
            with self.resource_guard.resource_protection():
                # Execute with circuit breaker and retry protection
                return self.circuit_breaker.call(
                    self.retry_strategy.retry, 
                    protected_forward
                )
        except Exception as e:
            # Attempt auto-healing if available
            if self.auto_healer:
                self.logger.info("Attempting auto-healing due to forward pass failure")
                healing_result = self.auto_healer.diagnose_and_heal()
                
                if healing_result["healing_successful"]:
                    self.logger.info("Auto-healing successful, retrying forward pass")
                    return self.circuit_breaker.call(protected_forward)
            
            raise InferenceError(
                node_features.shape, 
                "HyperGNN.forward_resilient", 
                f"Resilient forward pass failed after all recovery attempts: {e}"
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the model.
        
        Returns:
            Dictionary containing health metrics and status information
        """
        status = {
            "timestamp": time.time(),
            "model_info": {
                "text_encoder": self.text_encoder_name,
                "gnn_backbone": self.gnn_backbone,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
            },
            "resilience_enabled": self.resilience_enabled,
            "overall_health": "healthy",
        }
        
        if not self.resilience_enabled:
            return status
        
        # Circuit breaker health
        if self.circuit_breaker:
            circuit_health = self.circuit_breaker.get_health_metrics()
            status["circuit_breaker"] = circuit_health
            if circuit_health["state"] != "healthy":
                status["overall_health"] = "degraded"
        
        # Resource health
        if self.resource_guard:
            healthy, warnings = self.resource_guard.is_resources_healthy()
            status["resources"] = {
                "healthy": healthy,
                "warnings": warnings,
                "current": self.resource_guard.check_resources()
            }
            if not healthy:
                status["overall_health"] = "degraded"
        
        # Auto-healing status
        if self.auto_healer:
            recent_healings = self.auto_healer.healing_history[-5:]  # Last 5 events
            status["auto_healing"] = {
                "recent_healings": recent_healings,
                "total_healings": len(self.auto_healer.healing_history),
                "last_healing": recent_healings[-1] if recent_healings else None,
            }
            
            # Mark as degraded if recent healings occurred
            if recent_healings:
                recent_time = time.time() - 300  # Last 5 minutes
                recent_critical_healings = [
                    h for h in recent_healings 
                    if h["timestamp"] > recent_time and h["healing_successful"]
                ]
                if recent_critical_healings:
                    status["overall_health"] = "degraded"
        
        # Model parameter health check
        param_health = self._check_parameter_health()
        status["parameters"] = param_health
        if param_health["issues_found"]:
            status["overall_health"] = "critical" if param_health["critical_issues"] else "degraded"
        
        return status
    
    def _check_parameter_health(self) -> Dict[str, Any]:
        """Check health of model parameters."""
        health = {
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "nan_params": 0,
            "inf_params": 0,
            "zero_grad_params": 0,
            "issues_found": [],
            "critical_issues": [],
        }
        
        for name, param in self.named_parameters():
            # Check for NaN values
            if torch.isnan(param).any():
                health["nan_params"] += torch.isnan(param).sum().item()
                health["issues_found"].append(f"NaN values in {name}")
                health["critical_issues"].append(name)
            
            # Check for infinite values
            if torch.isinf(param).any():
                health["inf_params"] += torch.isinf(param).sum().item()
                health["issues_found"].append(f"Infinite values in {name}")
                health["critical_issues"].append(name)
            
            # Check gradients if available
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    health["issues_found"].append(f"Invalid gradients in {name}")
                elif param.grad.abs().max() == 0:
                    health["zero_grad_params"] += 1
        
        return health
    
    def create_resilient_wrapper(self) -> 'ResilientModelWrapper':
        """Create a resilient wrapper around this model.
        
        Returns:
            ResilientModelWrapper instance providing additional protection
        """
        if not self.resilience_enabled or ResilientModelWrapper is None:
            raise RuntimeError(
                "Resilience features not available. Install required dependencies."
            )
        
        return ResilientModelWrapper(
            model=self,
            enable_circuit_breaker=True,
            enable_auto_healing=True,
            enable_resource_guard=True,
        )
    
    def optimize_for_performance(self, 
                                mixed_precision: bool = True,
                                gradient_checkpointing: bool = True,
                                model_compilation: bool = True) -> 'HyperGNN':
        """Optimize model for high performance inference and training.
        
        Args:
            mixed_precision: Enable mixed precision training/inference
            gradient_checkpointing: Enable gradient checkpointing for memory efficiency
            model_compilation: Enable PyTorch 2.0 compilation
            
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If scaling features are not available
        """
        if not self.scaling_enabled or PerformanceOptimizer is None:
            raise RuntimeError(
                "Scaling features not available. Install required dependencies."
            )
        
        if self.performance_optimizer is None:
            self.performance_optimizer = PerformanceOptimizer(self)
        
        self.performance_optimizer\
            .apply_mixed_precision(mixed_precision)\
            .apply_gradient_checkpointing(gradient_checkpointing)
        
        if model_compilation:
            self.performance_optimizer.apply_model_compilation()
        
        self.logger.info("Performance optimizations applied")
        return self
    
    def setup_batch_processing(self, 
                             batch_size: int = 32,
                             max_workers: int = None,
                             device_ids: Optional[List[int]] = None) -> 'HyperGNN':
        """Setup high-throughput batch processing for inference.
        
        Args:
            batch_size: Batch size for processing
            max_workers: Maximum number of worker threads
            device_ids: GPU device IDs to use
            
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If scaling features are not available
        """
        if not self.scaling_enabled or BatchProcessor is None:
            raise RuntimeError(
                "Scaling features not available. Install required dependencies."
            )
        
        self.batch_processor = BatchProcessor(
            model=self,
            batch_size=batch_size,
            max_workers=max_workers,
            device_ids=device_ids
        )
        self.batch_processor.start()
        
        self.logger.info(f"Batch processing setup: batch_size={batch_size}, "
                        f"max_workers={self.batch_processor.max_workers}")
        return self
    
    def create_scalable_wrapper(self, 
                               distributed_config: Optional[DistributedConfig] = None,
                               enable_optimization: bool = True) -> 'ScalableHyperGNN':
        """Create a scalable wrapper with distributed and optimization features.
        
        Args:
            distributed_config: Configuration for distributed training
            enable_optimization: Whether to enable performance optimizations
            
        Returns:
            ScalableHyperGNN wrapper instance
            
        Raises:
            RuntimeError: If scaling features are not available
        """
        if not self.scaling_enabled or ScalableHyperGNN is None:
            raise RuntimeError(
                "Scaling features not available. Install required dependencies."
            )
        
        return ScalableHyperGNN(
            base_model=self,
            distributed_config=distributed_config,
            enable_optimization=enable_optimization
        )
    
    def profile_performance(self, 
                          sample_input: Tuple[torch.Tensor, torch.Tensor, List[str]],
                          num_runs: int = 100) -> Dict[str, float]:
        """Profile model performance with sample input.
        
        Args:
            sample_input: Sample input tuple (edge_index, node_features, node_texts)
            num_runs: Number of profiling runs
            
        Returns:
            Performance metrics dictionary
        """
        if self.performance_optimizer is None and self.scaling_enabled:
            self.performance_optimizer = PerformanceOptimizer(self)
        
        if self.performance_optimizer:
            return self.performance_optimizer.profile_performance(sample_input, num_runs)
        else:
            # Fallback basic profiling
            self.eval()
            edge_index, node_features, node_texts = sample_input
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    self.forward(edge_index, node_features, node_texts)
            
            # Time measurement
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs):
                    self.forward(edge_index, node_features, node_texts)
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / num_runs
            return {
                "avg_inference_time_ms": avg_time * 1000,
                "throughput_qps": 1.0 / avg_time,
                "memory_usage_mb": 0.0,  # Not available in fallback
                "optimizations_applied": 0,
            }
    
    def process_batch_async(self, 
                           batch_data: List[Tuple[torch.Tensor, torch.Tensor, List[str]]],
                           timeout: float = 10.0) -> List[torch.Tensor]:
        """Process batch of inputs asynchronously using batch processor.
        
        Args:
            batch_data: List of input tuples
            timeout: Timeout for batch processing
            
        Returns:
            List of output tensors
            
        Raises:
            RuntimeError: If batch processor is not initialized
        """
        if self.batch_processor is None:
            raise RuntimeError(
                "Batch processor not initialized. Call setup_batch_processing() first."
            )
        
        # Submit all requests
        request_ids = []
        for data in batch_data:
            req_id = self.batch_processor.process_async(data)
            request_ids.append(req_id)
        
        # Collect results
        results = []
        for req_id in request_ids:
            result = self.batch_processor.get_result(req_id, timeout)
            results.append(result)
        
        return results
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status and metrics.
        
        Returns:
            Dictionary containing scaling status and metrics
        """
        status = {
            "scaling_enabled": self.scaling_enabled,
            "optimization_applied": self.performance_optimizer is not None,
            "batch_processing_active": self.batch_processor is not None,
        }
        
        if self.performance_optimizer:
            status["optimizations"] = self.performance_optimizer.get_optimization_summary()
        
        if self.batch_processor:
            status["batch_processing"] = self.batch_processor.get_statistics()
        
        return status
    
    def cleanup_scaling_resources(self):
        """Cleanup scaling-related resources."""
        if self.batch_processor:
            self.batch_processor.stop()
            self.batch_processor = None
            self.logger.info("Batch processor stopped and cleaned up")
    
    @classmethod
    def from_config(cls, config: Dict) -> "HyperGNN":
        """Create model from configuration with validation.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            HyperGNN instance
            
        Raises:
            ValidationError: If configuration is invalid
            ModelError: If model creation fails
        """
        if not isinstance(config, dict):
            raise ValidationError("config", type(config).__name__, "dictionary")
        
        logger.info(f"Creating HyperGNN from config: {config}")
        
        try:
            return cls(**config)
        except Exception as e:
            error_msg = f"Failed to create HyperGNN from config: {e}"
            logger.error(error_msg, exc_info=True)
            if isinstance(e, (ValidationError, ModelError, NetworkError)):
                raise
            raise ModelError("HyperGNN", "from_config", error_msg)