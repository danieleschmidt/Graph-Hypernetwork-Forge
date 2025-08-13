"""Research Baseline Models for Comparative Studies.

This module implements state-of-the-art baseline models for comprehensive
benchmarking against the HyperGNN approach. Includes traditional GNNs,
meta-learning methods, and transfer learning approaches.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool

# Enhanced error handling
try:
    from ..utils.logging_utils import get_logger, log_function_call
    from ..utils.exceptions import ValidationError, ModelError
    ENHANCED_FEATURES = True
except ImportError:
    def log_function_call(*args, **kwargs):
        def decorator(func): return func
        return decorator
    def get_logger(name): 
        import logging
        return logging.getLogger(name)
    class ValidationError(Exception): pass
    class ModelError(Exception): pass
    ENHANCED_FEATURES = False

logger = get_logger(__name__)


class TraditionalGNN(nn.Module):
    """Traditional GNN baseline with fixed architecture.
    
    This serves as a strong baseline for comparison with hypernetwork approaches.
    Supports GCN, GAT, and GraphSAGE architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        gnn_type: str = "GAT",
        dropout: float = 0.1,
        heads: int = 8,
    ):
        """Initialize traditional GNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN (GCN, GAT, SAGE)
            dropout: Dropout probability
            heads: Number of attention heads (for GAT)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.upper()
        self.dropout = dropout
        self.heads = heads
        
        logger.info(f"Initializing TraditionalGNN: {gnn_type}, {num_layers} layers")
        
        # Build GNN layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
            
            if i == num_layers - 1:
                out_dim = output_dim
                heads_out = 1  # Single head for output layer
            else:
                out_dim = hidden_dim
                heads_out = heads
            
            if self.gnn_type == "GCN":
                self.layers.append(GCNConv(in_dim, out_dim))
            elif self.gnn_type == "GAT":
                self.layers.append(
                    GATConv(
                        in_dim, 
                        out_dim // heads_out,
                        heads=heads_out,
                        dropout=dropout,
                        concat=(i != num_layers - 1)
                    )
                )
            elif self.gnn_type == "SAGE":
                self.layers.append(SAGEConv(in_dim, out_dim))
            else:
                raise ValidationError("gnn_type", gnn_type, "GCN, GAT, or SAGE")
        
        self.dropout_layer = nn.Dropout(dropout)
    
    @log_function_call()
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through traditional GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector for graph-level tasks
            
        Returns:
            Node or graph embeddings
        """
        # Apply GNN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            
            # Apply activation and dropout (except for last layer)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        # Graph-level pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class MAMLBaseline(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) baseline for few-shot graph learning.
    
    This implements MAML applied to graph neural networks for comparison
    with hypernetwork-based approaches.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        meta_lr: float = 0.001,
        adaptation_steps: int = 5,
    ):
        """Initialize MAML baseline.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of layers
            meta_lr: Meta-learning rate
            adaptation_steps: Number of gradient steps for adaptation
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        
        # Base network (will be adapted)
        self.base_network = TraditionalGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
        
        logger.info(f"Initialized MAML baseline with {adaptation_steps} adaptation steps")
    
    def adapt(
        self,
        support_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        task_type: str = "node_classification",
    ) -> nn.Module:
        """Adapt model to new task using support examples.
        
        Args:
            support_data: List of (features, edge_index, labels) tuples
            task_type: Type of task (node_classification, graph_classification)
            
        Returns:
            Adapted model
        """
        # Create adapted copy
        adapted_model = TraditionalGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
        )
        
        # Copy parameters from base model
        adapted_model.load_state_dict(self.base_network.state_dict())
        
        # Adaptation optimizer
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.meta_lr)
        
        # Adaptation loop
        for step in range(self.adaptation_steps):
            total_loss = 0
            
            for features, edge_index, labels in support_data:
                # Forward pass
                predictions = adapted_model(features, edge_index)
                
                # Compute loss based on task type
                if task_type == "node_classification":
                    loss = F.cross_entropy(predictions, labels)
                elif task_type == "graph_classification":
                    # Assume features is batched and batch vector is provided
                    batch = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
                    predictions = adapted_model(features, edge_index, batch)
                    loss = F.cross_entropy(predictions, labels)
                else:
                    loss = F.mse_loss(predictions, labels)
                
                total_loss += loss
            
            # Gradient step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            logger.debug(f"MAML adaptation step {step}: loss = {total_loss.item():.4f}")
        
        return adapted_model
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass using base network."""
        return self.base_network(x, edge_index)


class PrototypicalNetworkBaseline(nn.Module):
    """Prototypical Networks adapted for graph learning.
    
    This implements prototypical networks for few-shot graph classification,
    serving as a strong meta-learning baseline.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int = 128,
        num_layers: int = 3,
        distance_metric: str = "euclidean",
    ):
        """Initialize Prototypical Network baseline.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            embedding_dim: Embedding dimension for prototypes
            num_layers: Number of GNN layers
            distance_metric: Distance metric (euclidean, cosine)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        
        # Embedding network
        self.encoder = TraditionalGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
        )
        
        logger.info(f"Initialized PrototypicalNetwork with {distance_metric} distance")
    
    def compute_prototypes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Compute class prototypes from embeddings.
        
        Args:
            embeddings: Node/graph embeddings [batch_size, embedding_dim]
            labels: Class labels [batch_size]
            num_classes: Number of classes
            
        Returns:
            Prototypes [num_classes, embedding_dim]
        """
        prototypes = torch.zeros(
            num_classes, self.embedding_dim, 
            device=embeddings.device, dtype=embeddings.dtype
        )
        
        for class_idx in range(num_classes):
            class_mask = labels == class_idx
            if class_mask.any():
                prototypes[class_idx] = embeddings[class_mask].mean(dim=0)
        
        return prototypes
    
    def compute_distances(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distances between embeddings and prototypes.
        
        Args:
            embeddings: Query embeddings [batch_size, embedding_dim]
            prototypes: Class prototypes [num_classes, embedding_dim]
            
        Returns:
            Distances [batch_size, num_classes]
        """
        if self.distance_metric == "euclidean":
            # Euclidean distance
            distances = torch.cdist(embeddings, prototypes, p=2)
        elif self.distance_metric == "cosine":
            # Cosine similarity (converted to distance)
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1)
            similarities = torch.mm(embeddings_norm, prototypes_norm.t())
            distances = 1 - similarities
        else:
            raise ValidationError("distance_metric", self.distance_metric, "euclidean or cosine")
        
        return distances
    
    def forward(
        self,
        query_x: torch.Tensor,
        query_edge_index: torch.Tensor,
        support_x: torch.Tensor,
        support_edge_index: torch.Tensor,
        support_labels: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Forward pass for few-shot classification.
        
        Args:
            query_x: Query node features
            query_edge_index: Query edge connectivity
            support_x: Support node features
            support_edge_index: Support edge connectivity
            support_labels: Support labels
            num_classes: Number of classes
            
        Returns:
            Class predictions for query examples
        """
        # Encode support and query examples
        support_embeddings = self.encoder(support_x, support_edge_index)
        query_embeddings = self.encoder(query_x, query_edge_index)
        
        # Compute prototypes from support set
        prototypes = self.compute_prototypes(support_embeddings, support_labels, num_classes)
        
        # Compute distances and convert to probabilities
        distances = self.compute_distances(query_embeddings, prototypes)
        logits = -distances  # Negative distance as logits
        
        return F.softmax(logits, dim=1)


class FineTuningBaseline(nn.Module):
    """Traditional fine-tuning baseline for transfer learning.
    
    This implements standard pre-training + fine-tuning for graph neural networks,
    serving as a strong transfer learning baseline.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        pretraining_tasks: List[str] = None,
    ):
        """Initialize fine-tuning baseline.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of layers
            pretraining_tasks: List of pretraining task types
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Shared encoder
        self.encoder = TraditionalGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output hidden representation
            num_layers=num_layers - 1,  # Reserve last layer for task-specific head
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        # Default classification head
        self.task_heads["classification"] = nn.Linear(hidden_dim, output_dim)
        
        # Pretraining heads
        if pretraining_tasks:
            for task in pretraining_tasks:
                if task == "node_reconstruction":
                    self.task_heads[task] = nn.Linear(hidden_dim, input_dim)
                elif task == "edge_prediction":
                    self.task_heads[task] = nn.Linear(hidden_dim * 2, 1)
                elif task == "graph_classification":
                    self.task_heads[task] = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"Initialized FineTuning baseline with {len(self.task_heads)} task heads")
    
    def pretrain(
        self,
        data_loader,
        task_type: str = "node_reconstruction",
        num_epochs: int = 100,
        lr: float = 0.001,
    ):
        """Pretrain the encoder on auxiliary tasks.
        
        Args:
            data_loader: DataLoader for pretraining data
            task_type: Type of pretraining task
            num_epochs: Number of pretraining epochs
            lr: Learning rate
        """
        if task_type not in self.task_heads:
            raise ValidationError("task_type", task_type, f"one of {list(self.task_heads.keys())}")
        
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.task_heads[task_type].parameters()),
            lr=lr
        )
        
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in data_loader:
                optimizer.zero_grad()
                
                # Extract batch data
                x, edge_index = batch.x, batch.edge_index
                
                # Forward pass through encoder
                embeddings = self.encoder(x, edge_index)
                
                # Task-specific forward pass
                if task_type == "node_reconstruction":
                    reconstructed = self.task_heads[task_type](embeddings)
                    loss = F.mse_loss(reconstructed, x)
                
                elif task_type == "edge_prediction":
                    # Sample positive and negative edges
                    num_edges = edge_index.size(1)
                    perm = torch.randperm(num_edges)[:num_edges // 2]
                    pos_edges = edge_index[:, perm]
                    
                    # Generate negative edges
                    neg_edges = self._generate_negative_edges(edge_index, x.size(0), pos_edges.size(1))
                    
                    # Edge embeddings
                    pos_embeddings = torch.cat([
                        embeddings[pos_edges[0]], embeddings[pos_edges[1]]
                    ], dim=1)
                    neg_embeddings = torch.cat([
                        embeddings[neg_edges[0]], embeddings[neg_edges[1]]
                    ], dim=1)
                    
                    # Predictions
                    pos_pred = self.task_heads[task_type](pos_embeddings)
                    neg_pred = self.task_heads[task_type](neg_embeddings)
                    
                    # Binary classification loss
                    pos_loss = F.binary_cross_entropy_with_logits(
                        pos_pred.squeeze(), torch.ones(pos_pred.size(0), device=pos_pred.device)
                    )
                    neg_loss = F.binary_cross_entropy_with_logits(
                        neg_pred.squeeze(), torch.zeros(neg_pred.size(0), device=neg_pred.device)
                    )
                    loss = (pos_loss + neg_loss) / 2
                
                elif task_type == "graph_classification":
                    # Graph-level pooling
                    if hasattr(batch, 'batch'):
                        graph_embeddings = global_mean_pool(embeddings, batch.batch)
                        predictions = self.task_heads[task_type](graph_embeddings)
                        loss = F.cross_entropy(predictions, batch.y)
                    else:
                        # Single graph case
                        graph_embedding = embeddings.mean(dim=0, keepdim=True)
                        prediction = self.task_heads[task_type](graph_embedding)
                        loss = F.cross_entropy(prediction, batch.y.unsqueeze(0))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            if epoch % 10 == 0:
                logger.info(f"Pretraining epoch {epoch}: loss = {avg_loss:.4f}")
    
    def _generate_negative_edges(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int, 
        num_neg_edges: int
    ) -> torch.Tensor:
        """Generate negative edges for edge prediction task."""
        device = edge_index.device
        
        # Convert edge_index to set for fast lookup
        edge_set = set(zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()))
        
        neg_edges = []
        while len(neg_edges) < num_neg_edges:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            
            if src != dst and (src, dst) not in edge_set and (dst, src) not in edge_set:
                neg_edges.append([src, dst])
        
        return torch.tensor(neg_edges, device=device).t()
    
    def fine_tune(
        self,
        data_loader,
        task_type: str = "classification",
        num_epochs: int = 50,
        lr: float = 0.0001,
        freeze_encoder: bool = False,
    ):
        """Fine-tune on target task.
        
        Args:
            data_loader: DataLoader for target task data
            task_type: Target task type
            num_epochs: Number of fine-tuning epochs
            lr: Learning rate
            freeze_encoder: Whether to freeze encoder weights
        """
        if task_type not in self.task_heads:
            raise ValidationError("task_type", task_type, f"one of {list(self.task_heads.keys())}")
        
        # Setup optimizer
        if freeze_encoder:
            params = self.task_heads[task_type].parameters()
        else:
            params = list(self.encoder.parameters()) + list(self.task_heads[task_type].parameters())
        
        optimizer = torch.optim.Adam(params, lr=lr)
        
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in data_loader:
                optimizer.zero_grad()
                
                # Forward pass
                x, edge_index = batch.x, batch.edge_index
                embeddings = self.encoder(x, edge_index)
                predictions = self.task_heads[task_type](embeddings)
                
                # Compute loss
                if task_type == "classification":
                    loss = F.cross_entropy(predictions, batch.y)
                else:
                    loss = F.mse_loss(predictions, batch.y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            if epoch % 10 == 0:
                logger.info(f"Fine-tuning epoch {epoch}: loss = {avg_loss:.4f}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, task_type: str = "classification") -> torch.Tensor:
        """Forward pass for inference.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            task_type: Task type for appropriate head
            
        Returns:
            Task predictions
        """
        embeddings = self.encoder(x, edge_index)
        return self.task_heads[task_type](embeddings)


class ZeroShotTextGNN(nn.Module):
    """Zero-shot baseline using text similarity without hypernetworks.
    
    This serves as a simpler baseline that uses text similarity for
    zero-shot transfer without generating dynamic weights.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
    ):
        """Initialize zero-shot text GNN baseline.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            text_encoder: Text encoder model
            similarity_threshold: Similarity threshold for adaptation
        """
        super().__init__()
        
        from sentence_transformers import SentenceTransformer
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.similarity_threshold = similarity_threshold
        
        # Text encoder
        self.text_encoder = SentenceTransformer(text_encoder)
        
        # Base GNN
        self.base_gnn = TraditionalGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        
        # Memory bank for storing source domain examples
        self.memory_bank = {
            "embeddings": [],
            "labels": [],
            "texts": [],
        }
        
        logger.info(f"Initialized ZeroShotTextGNN with similarity threshold {similarity_threshold}")
    
    def add_to_memory(
        self,
        node_texts: List[str],
        node_features: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Add examples to memory bank.
        
        Args:
            node_texts: Node text descriptions
            node_features: Node features
            labels: Node labels
        """
        # Encode texts
        text_embeddings = self.text_encoder.encode(node_texts, convert_to_tensor=True)
        
        # Store in memory bank
        self.memory_bank["embeddings"].append(text_embeddings)
        self.memory_bank["labels"].append(labels)
        self.memory_bank["texts"].extend(node_texts)
    
    def predict_zero_shot(
        self,
        query_texts: List[str],
        query_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict on new examples using text similarity.
        
        Args:
            query_texts: Query text descriptions
            query_features: Query node features
            edge_index: Edge connectivity
            
        Returns:
            Predictions based on similar examples
        """
        if not self.memory_bank["embeddings"]:
            raise RuntimeError("Memory bank is empty. Add source examples first.")
        
        # Encode query texts
        query_embeddings = self.text_encoder.encode(query_texts, convert_to_tensor=True)
        
        # Concatenate all memory embeddings
        memory_embeddings = torch.cat(self.memory_bank["embeddings"], dim=0)
        memory_labels = torch.cat(self.memory_bank["labels"], dim=0)
        
        # Compute similarities
        similarities = F.cosine_similarity(
            query_embeddings.unsqueeze(1), 
            memory_embeddings.unsqueeze(0), 
            dim=2
        )
        
        # Find most similar examples above threshold
        max_similarities, max_indices = torch.max(similarities, dim=1)
        
        # Create predictions based on similar examples
        predictions = torch.zeros(len(query_texts), self.output_dim, device=query_features.device)
        
        for i, (sim, idx) in enumerate(zip(max_similarities, max_indices)):
            if sim > self.similarity_threshold:
                # Use label from most similar example
                similar_label = memory_labels[idx]
                if similar_label.dim() == 0:  # Scalar label
                    predictions[i, similar_label] = 1.0
                else:  # One-hot label
                    predictions[i] = similar_label
            else:
                # Fall back to uniform distribution
                predictions[i] = torch.ones(self.output_dim) / self.output_dim
        
        return predictions
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            node_texts: Node texts for zero-shot prediction
            
        Returns:
            Predictions
        """
        if node_texts is not None:
            # Zero-shot prediction using text similarity
            return self.predict_zero_shot(node_texts, x, edge_index)
        else:
            # Regular GNN forward pass
            return self.base_gnn(x, edge_index)