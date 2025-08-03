"""Training and evaluation utilities."""

import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from ..data.knowledge_graph import TextualKnowledgeGraph
from ..models.hypergnn import HyperGNN


class HyperGNNTrainer:
    """Trainer for HyperGNN models."""
    
    def __init__(
        self,
        model: HyperGNN,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "auto",
        wandb_project: Optional[str] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: HyperGNN model to train
            optimizer: Optimizer (defaults to Adam)
            scheduler: Learning rate scheduler
            device: Device to use ("auto", "cuda", "cpu")
            wandb_project: Wandb project name for logging
        """
        self.model = model
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        
        # Setup wandb
        if wandb_project:
            wandb.init(project=wandb_project, config=self.model.get_config())
        self.use_wandb = wandb_project is not None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def train_epoch(
        self,
        train_graphs: List[TextualKnowledgeGraph],
        loss_fn: nn.Module,
        task_type: str = "node_classification",
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_graphs: List of training knowledge graphs
            loss_fn: Loss function
            task_type: Type of task ("node_classification", "link_prediction", "graph_classification")
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        pbar = tqdm(train_graphs, desc=f"Training Epoch {self.epoch}")
        
        for graph in pbar:
            self.optimizer.zero_grad()
            
            # Move data to device
            edge_index = graph.edge_index.to(self.device)
            node_features = graph.node_features
            if node_features is not None:
                node_features = node_features.to(self.device)
            else:
                # Create default features
                node_features = torch.randn(graph.num_nodes, 128, device=self.device)
            
            # Forward pass
            try:
                predictions = self.model(edge_index, node_features, graph.node_texts)
                
                # Calculate loss based on task type
                if task_type == "node_classification":
                    if graph.node_labels is not None:
                        labels = graph.node_labels.to(self.device)
                        loss = loss_fn(predictions, labels)
                    else:
                        # Skip graphs without labels
                        continue
                elif task_type == "link_prediction":
                    # For link prediction, create positive and negative samples
                    loss = self._link_prediction_loss(predictions, edge_index, loss_fn)
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_samples += 1
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Log to wandb
                if self.use_wandb and self.global_step % 10 == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/epoch": self.epoch,
                        "train/step": self.global_step,
                    })
                
            except Exception as e:
                print(f"Error processing graph: {e}")
                continue
        
        # Calculate average loss
        avg_loss = total_loss / max(num_samples, 1)
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        return {
            "loss": avg_loss,
            "num_samples": num_samples,
        }
    
    def validate(
        self,
        val_graphs: List[TextualKnowledgeGraph],
        loss_fn: nn.Module,
        task_type: str = "node_classification",
    ) -> Dict[str, float]:
        """Validate model.
        
        Args:
            val_graphs: List of validation knowledge graphs
            loss_fn: Loss function
            task_type: Type of task
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for graph in tqdm(val_graphs, desc="Validation"):
                # Move data to device
                edge_index = graph.edge_index.to(self.device)
                node_features = graph.node_features
                if node_features is not None:
                    node_features = node_features.to(self.device)
                else:
                    node_features = torch.randn(graph.num_nodes, 128, device=self.device)
                
                try:
                    # Forward pass
                    predictions = self.model(edge_index, node_features, graph.node_texts)
                    
                    # Calculate loss and accuracy
                    if task_type == "node_classification":
                        if graph.node_labels is not None:
                            labels = graph.node_labels.to(self.device)
                            loss = loss_fn(predictions, labels)
                            
                            # Calculate accuracy
                            pred_classes = torch.argmax(predictions, dim=1)
                            correct = (pred_classes == labels).sum().item()
                            
                            total_loss += loss.item()
                            total_correct += correct
                            total_samples += labels.size(0)
                        else:
                            continue
                    elif task_type == "link_prediction":
                        loss = self._link_prediction_loss(predictions, edge_index, loss_fn)
                        total_loss += loss.item()
                        total_samples += 1
                
                except Exception as e:
                    print(f"Error validating graph: {e}")
                    continue
        
        # Calculate metrics
        avg_loss = total_loss / max(len(val_graphs), 1)
        accuracy = total_correct / max(total_samples, 1) if task_type == "node_classification" else None
        
        metrics = {"val_loss": avg_loss}
        if accuracy is not None:
            metrics["val_accuracy"] = accuracy
        
        return metrics
    
    def train(
        self,
        train_graphs: List[TextualKnowledgeGraph],
        val_graphs: Optional[List[TextualKnowledgeGraph]] = None,
        num_epochs: int = 100,
        task_type: str = "node_classification",
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_graphs: Training knowledge graphs
            val_graphs: Validation knowledge graphs
            num_epochs: Number of training epochs
            task_type: Type of task
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        # Setup loss function
        if task_type == "node_classification":
            loss_fn = nn.CrossEntropyLoss()
        elif task_type == "link_prediction":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_graphs, loss_fn, task_type)
            history["train_loss"].append(train_metrics["loss"])
            
            print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}")
            
            # Validation
            if val_graphs:
                val_metrics = self.validate(val_graphs, loss_fn, task_type)
                history["val_loss"].append(val_metrics["val_loss"])
                
                if "val_accuracy" in val_metrics:
                    history["val_accuracy"].append(val_metrics["val_accuracy"])
                    print(f"Epoch {epoch}: Val Loss = {val_metrics['val_loss']:.4f}, "
                          f"Val Acc = {val_metrics['val_accuracy']:.4f}")
                else:
                    print(f"Epoch {epoch}: Val Loss = {val_metrics['val_loss']:.4f}")
                
                # Early stopping
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    patience_counter = 0
                    
                    # Save best model
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train/epoch_loss": train_metrics["loss"],
                        "val/loss": val_metrics["val_loss"],
                        **({f"val/accuracy": val_metrics["val_accuracy"]} 
                           if "val_accuracy" in val_metrics else {}),
                    })
        
        self.training_history = history
        return history
    
    def _link_prediction_loss(
        self, 
        node_embeddings: torch.Tensor, 
        edge_index: torch.Tensor, 
        loss_fn: nn.Module
    ) -> torch.Tensor:
        """Calculate link prediction loss."""
        # Positive edges
        pos_edge_scores = self._edge_scores(node_embeddings, edge_index)
        pos_labels = torch.ones(pos_edge_scores.size(0), device=self.device)
        
        # Negative edges (random sampling)
        num_nodes = node_embeddings.size(0)
        num_neg_edges = edge_index.size(1)
        
        neg_edges = torch.randint(0, num_nodes, (2, num_neg_edges), device=self.device)
        neg_edge_scores = self._edge_scores(node_embeddings, neg_edges)
        neg_labels = torch.zeros(neg_edge_scores.size(0), device=self.device)
        
        # Combine positive and negative
        all_scores = torch.cat([pos_edge_scores, neg_edge_scores])
        all_labels = torch.cat([pos_labels, neg_labels])
        
        return loss_fn(all_scores, all_labels)
    
    def _edge_scores(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Calculate edge scores for link prediction."""
        row, col = edge_index
        edge_embeddings = node_embeddings[row] * node_embeddings[col]
        return edge_embeddings.sum(dim=1)
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.model.get_config(),
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Model loaded from {path}")


class ZeroShotEvaluator:
    """Evaluator for zero-shot transfer capabilities."""
    
    def __init__(self, model: HyperGNN, device: str = "auto"):
        """Initialize evaluator.
        
        Args:
            model: Trained HyperGNN model
            device: Device to use
        """
        self.model = model
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_transfer(
        self,
        source_graphs: List[TextualKnowledgeGraph],
        target_graphs: List[TextualKnowledgeGraph],
        task_type: str = "node_classification",
    ) -> Dict[str, float]:
        """Evaluate zero-shot transfer performance.
        
        Args:
            source_graphs: Source domain graphs (for reference)
            target_graphs: Target domain graphs (for evaluation)
            task_type: Type of task
            
        Returns:
            Transfer performance metrics
        """
        print(f"Evaluating zero-shot transfer on {len(target_graphs)} target graphs...")
        
        total_correct = 0
        total_samples = 0
        graph_accuracies = []
        
        with torch.no_grad():
            for graph in tqdm(target_graphs, desc="Zero-shot evaluation"):
                # Move data to device
                edge_index = graph.edge_index.to(self.device)
                node_features = graph.node_features
                if node_features is not None:
                    node_features = node_features.to(self.device)
                else:
                    node_features = torch.randn(graph.num_nodes, 128, device=self.device)
                
                try:
                    # Generate predictions
                    predictions = self.model(edge_index, node_features, graph.node_texts)
                    
                    if task_type == "node_classification" and graph.node_labels is not None:
                        labels = graph.node_labels.to(self.device)
                        pred_classes = torch.argmax(predictions, dim=1)
                        correct = (pred_classes == labels).sum().item()
                        
                        graph_accuracy = correct / labels.size(0)
                        graph_accuracies.append(graph_accuracy)
                        
                        total_correct += correct
                        total_samples += labels.size(0)
                
                except Exception as e:
                    print(f"Error evaluating graph: {e}")
                    continue
        
        # Calculate metrics
        overall_accuracy = total_correct / max(total_samples, 1)
        avg_graph_accuracy = sum(graph_accuracies) / max(len(graph_accuracies), 1)
        
        metrics = {
            "zero_shot_accuracy": overall_accuracy,
            "avg_graph_accuracy": avg_graph_accuracy,
            "num_target_graphs": len(target_graphs),
            "total_target_nodes": total_samples,
        }
        
        print(f"Zero-shot transfer results:")
        print(f"  Overall accuracy: {overall_accuracy:.4f}")
        print(f"  Average graph accuracy: {avg_graph_accuracy:.4f}")
        print(f"  Evaluated on {len(target_graphs)} graphs ({total_samples} nodes)")
        
        return metrics
    
    def analyze_text_similarity(
        self,
        source_texts: List[str],
        target_texts: List[str],
    ) -> Dict[str, float]:
        """Analyze text similarity between source and target domains.
        
        Args:
            source_texts: Source domain node texts
            target_texts: Target domain node texts
            
        Returns:
            Similarity metrics
        """
        # Encode texts
        source_embeddings = self.model.text_encoder(source_texts)
        target_embeddings = self.model.text_encoder(target_texts)
        
        # Calculate similarities
        similarities = torch.cosine_similarity(
            source_embeddings.unsqueeze(1), 
            target_embeddings.unsqueeze(0), 
            dim=2
        )
        
        # Calculate metrics
        max_similarities = similarities.max(dim=1)[0]
        avg_max_similarity = max_similarities.mean().item()
        min_max_similarity = max_similarities.min().item()
        
        return {
            "avg_max_similarity": avg_max_similarity,
            "min_max_similarity": min_max_similarity,
            "similarity_std": max_similarities.std().item(),
        }