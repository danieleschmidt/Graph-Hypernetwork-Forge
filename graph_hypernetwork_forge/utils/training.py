"""Training and evaluation utilities with comprehensive error handling."""

import time
import gc
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from ..data.knowledge_graph import TextualKnowledgeGraph
from ..models.hypergnn import HyperGNN
from .logging_utils import get_logger, LoggerMixin, log_function_call
from .exceptions import (
    ValidationError, TrainingError, GPUError, MemoryError, 
    handle_cuda_out_of_memory, log_and_raise_error
)
from .memory_utils import (
    MemoryMonitor, memory_management, check_gpu_memory_available,
    estimate_tensor_memory, safe_cuda_operation
)


# Initialize logger
logger = get_logger(__name__)


class HyperGNNTrainer(LoggerMixin):
    """Trainer for HyperGNN models with comprehensive error handling and monitoring."""
    
    def __init__(
        self,
        model: HyperGNN,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "auto",
        wandb_project: Optional[str] = None,
        memory_monitoring: bool = True,
        memory_cleanup_threshold: float = 0.85,
    ):
        """Initialize trainer with comprehensive validation and monitoring.
        
        Args:
            model: HyperGNN model to train
            optimizer: Optimizer (defaults to Adam)
            scheduler: Learning rate scheduler
            device: Device to use ("auto", "cuda", "cpu")
            wandb_project: Wandb project name for logging
            memory_monitoring: Whether to enable memory monitoring
            memory_cleanup_threshold: Memory threshold for automatic cleanup
            
        Raises:
            ValidationError: If parameters are invalid
        """
        super().__init__()
        
        # Validate inputs
        if not isinstance(model, HyperGNN):
            raise ValidationError("model", type(model).__name__, "HyperGNN instance")
        
        if device not in ["auto", "cuda", "cpu"]:
            raise ValidationError("device", device, "one of: auto, cuda, cpu")
        
        if not isinstance(memory_monitoring, bool):
            raise ValidationError("memory_monitoring", memory_monitoring, "boolean")
        
        if not isinstance(memory_cleanup_threshold, (int, float)) or not (0.0 <= memory_cleanup_threshold <= 1.0):
            raise ValidationError("memory_cleanup_threshold", memory_cleanup_threshold, "float between 0.0 and 1.0")
        
        self.model = model
        self.memory_monitoring = memory_monitoring
        self.memory_cleanup_threshold = memory_cleanup_threshold
        
        # Setup device with error handling
        try:
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            
            self.logger.info(f"Using device: {self.device}")
            
            # Move model to device with memory check
            if self.device.type == "cuda":
                model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
                check_gpu_memory_available(model_size * 2, "model loading")  # 2x for safety
            
            self.model.to(self.device)
            self.logger.info(f"Model moved to {self.device}")
            
        except Exception as e:
            if isinstance(e, GPUError):
                raise
            raise GPUError("model_loading", f"Failed to setup device or move model: {e}")
        
        # Setup optimizer with validation
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
            self.logger.info("Created default Adam optimizer")
        else:
            if not isinstance(optimizer, optim.Optimizer):
                raise ValidationError("optimizer", type(optimizer).__name__, "torch.optim.Optimizer")
            self.optimizer = optimizer
            self.logger.info(f"Using provided optimizer: {type(optimizer).__name__}")
        
        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler._LRScheduler):
            raise ValidationError("scheduler", type(scheduler).__name__, "torch.optim.lr_scheduler._LRScheduler")
        self.scheduler = scheduler
        
        # Setup wandb with error handling
        self.use_wandb = False
        if wandb_project:
            try:
                wandb.init(project=wandb_project, config=self.model.get_config())
                self.use_wandb = True
                self.logger.info(f"Initialized wandb logging for project: {wandb_project}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb logging.")
        
        # Setup memory monitoring
        if self.memory_monitoring:
            self.memory_monitor = MemoryMonitor(
                warning_threshold=self.memory_cleanup_threshold,
                cleanup_callbacks=[self._cleanup_training_memory]
            )
            self.memory_monitor.start_monitoring(interval=30.0)
            self.logger.info("Started memory monitoring")
        else:
            self.memory_monitor = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        self.last_cleanup_step = 0
        
        self.logger.info("HyperGNNTrainer initialized successfully")
    
    def _cleanup_training_memory(self) -> str:
        """Cleanup memory during training.
        
        Returns:
            Description of cleanup actions taken
        """
        actions = []
        
        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)
        actions.append("Cleared gradients")
        
        # Python garbage collection
        collected = gc.collect()
        if collected > 0:
            actions.append(f"Collected {collected} objects")
        
        # CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            actions.append("Cleared CUDA cache")
        
        return "; ".join(actions)
    
    @log_function_call()
    def train_epoch(
        self,
        train_graphs: List[TextualKnowledgeGraph],
        loss_fn: nn.Module,
        task_type: str = "node_classification",
    ) -> Dict[str, float]:
        """Train for one epoch with comprehensive error handling and monitoring.
        
        Args:
            train_graphs: List of training knowledge graphs
            loss_fn: Loss function
            task_type: Type of task ("node_classification", "link_prediction", "graph_classification")
            
        Returns:
            Dictionary with training metrics
            
        Raises:
            ValidationError: If inputs are invalid
            TrainingError: If training fails
        """
        # Validate inputs
        if not isinstance(train_graphs, list) or len(train_graphs) == 0:
            raise ValidationError("train_graphs", "empty or non-list", "non-empty list of TextualKnowledgeGraph")
        
        for i, graph in enumerate(train_graphs):
            if not isinstance(graph, TextualKnowledgeGraph):
                raise ValidationError(f"train_graphs[{i}]", type(graph).__name__, "TextualKnowledgeGraph")
        
        if not isinstance(loss_fn, nn.Module):
            raise ValidationError("loss_fn", type(loss_fn).__name__, "nn.Module")
        
        if task_type not in ["node_classification", "link_prediction", "graph_classification"]:
            raise ValidationError("task_type", task_type, "node_classification, link_prediction, or graph_classification")
        
        self.logger.info(f"Starting training epoch {self.epoch} with {len(train_graphs)} graphs")
        with memory_management(cleanup_on_exit=True):
            self.model.train()
            total_loss = 0.0
            num_samples = 0
            num_errors = 0
            max_errors = len(train_graphs) // 10  # Allow 10% errors
            
            pbar = tqdm(train_graphs, desc=f"Training Epoch {self.epoch}")
        
            for graph_idx, graph in enumerate(pbar):
                try:
                    # Validate graph
                    if graph.num_nodes == 0:
                        self.logger.warning(f"Skipping empty graph at index {graph_idx}")
                        continue
                    
                    if len(graph.node_texts) != graph.num_nodes:
                        self.logger.warning(f"Skipping graph {graph_idx}: text count mismatch")
                        continue
                    
                    # Memory cleanup check
                    if self.global_step - self.last_cleanup_step > 100:
                        if self.memory_monitor:
                            status = self.memory_monitor.check_memory_usage()
                            if status['should_cleanup']:
                                self.logger.info("Performing automatic memory cleanup during training")
                                cleanup_result = self._cleanup_training_memory()
                                self.logger.debug(f"Cleanup actions: {cleanup_result}")
                                self.last_cleanup_step = self.global_step
                    
                    self.optimizer.zero_grad()
                    
                    # Move data to device with error handling
                    try:
                        edge_index = safe_cuda_operation(
                            lambda: graph.edge_index.to(self.device),
                            "edge_index transfer"
                        )
                        
                        node_features = graph.node_features
                        if node_features is not None:
                            node_features = safe_cuda_operation(
                                lambda: node_features.to(self.device),
                                "node_features transfer"
                            )
                        else:
                            # Create default features
                            node_features = torch.randn(graph.num_nodes, 128, device=self.device)
                    
                    except GPUError as e:
                        self.logger.error(f"GPU error moving data for graph {graph_idx}: {e}")
                        num_errors += 1
                        if num_errors > max_errors:
                            raise TrainingError(self.epoch, graph_idx, message="Too many GPU errors")
                        continue
                    
                    # Forward pass with error handling
                    try:
                        predictions = self.model(edge_index, node_features, graph.node_texts)
                        
                        # Calculate loss based on task type
                        if task_type == "node_classification":
                            if graph.node_labels is not None:
                                labels = safe_cuda_operation(
                                    lambda: graph.node_labels.to(self.device),
                                    "labels transfer"
                                )
                                loss = loss_fn(predictions, labels)
                            else:
                                # Skip graphs without labels
                                self.logger.debug(f"Skipping graph {graph_idx}: no labels")
                                continue
                        elif task_type == "link_prediction":
                            # For link prediction, create positive and negative samples
                            loss = self._link_prediction_loss(predictions, edge_index, loss_fn)
                        else:
                            raise ValidationError("task_type", task_type, "supported task type")
                        
                        # Validate loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            self.logger.warning(f"Invalid loss for graph {graph_idx}: {loss.item()}")
                            continue
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # Check for gradient issues
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            self.logger.warning(f"Invalid gradients for graph {graph_idx}, skipping step")
                            self.optimizer.zero_grad()
                            continue
                        
                        # Optimizer step
                        self.optimizer.step()
                        
                        # Update metrics
                        total_loss += loss.item()
                        num_samples += 1
                        self.global_step += 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "errors": num_errors,
                            "grad_norm": f"{grad_norm:.3f}"
                        })
                        
                        # Log to wandb
                        if self.use_wandb and self.global_step % 10 == 0:
                            log_dict = {
                                "train/loss": loss.item(),
                                "train/epoch": self.epoch,
                                "train/step": self.global_step,
                                "train/grad_norm": grad_norm,
                            }
                            
                            # Add memory metrics if monitoring
                            if self.memory_monitor:
                                memory_info = self.memory_monitor.get_memory_info()
                                log_dict.update({
                                    "train/memory_gb": memory_info.process_memory_gb,
                                    "train/gpu_memory_gb": memory_info.gpu_allocated_gb or 0,
                                })
                            
                            wandb.log(log_dict)
                    
                    except torch.cuda.OutOfMemoryError:
                        self.logger.error(f"CUDA OOM for graph {graph_idx}, performing cleanup")
                        self._cleanup_training_memory()
                        num_errors += 1
                        if num_errors > max_errors:
                            raise handle_cuda_out_of_memory(f"training epoch {self.epoch}")
                        continue
                    
                    except Exception as e:
                        self.logger.warning(f"Error processing graph {graph_idx}: {e}")
                        num_errors += 1
                        if num_errors > max_errors:
                            raise TrainingError(self.epoch, graph_idx, message=f"Too many errors: {e}")
                        continue
                
                except Exception as e:
                    self.logger.error(f"Critical error in training loop for graph {graph_idx}: {e}")
                    raise TrainingError(self.epoch, graph_idx, message=f"Critical training error: {e}")
        
            # Calculate metrics
            if num_samples == 0:
                self.logger.warning("No valid samples processed in this epoch")
                avg_loss = float('inf')
            else:
                avg_loss = total_loss / num_samples
            
            # Update learning rate
            if self.scheduler:
                try:
                    self.scheduler.step()
                    self.logger.debug(f"Learning rate updated: {self.optimizer.param_groups[0]['lr']:.6f}")
                except Exception as e:
                    self.logger.warning(f"Failed to step scheduler: {e}")
            
            # Final memory cleanup
            if self.memory_monitor:
                cleanup_result = self._cleanup_training_memory()
                self.logger.debug(f"Epoch end cleanup: {cleanup_result}")
            
            epoch_metrics = {
                "loss": avg_loss,
                "num_samples": num_samples,
                "num_errors": num_errors,
                "error_rate": num_errors / len(train_graphs),
            }
            
            self.logger.info(f"Epoch {self.epoch} completed: loss={avg_loss:.4f}, "
                           f"samples={num_samples}, errors={num_errors}")
            
            return epoch_metrics
    
    @log_function_call()
    def validate(
        self,
        val_graphs: List[TextualKnowledgeGraph],
        loss_fn: nn.Module,
        task_type: str = "node_classification",
    ) -> Dict[str, float]:
        """Validate model with comprehensive error handling and monitoring.
        
        Args:
            val_graphs: List of validation knowledge graphs
            loss_fn: Loss function
            task_type: Type of task
            
        Returns:
            Dictionary with validation metrics
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        if not isinstance(val_graphs, list) or len(val_graphs) == 0:
            raise ValidationError("val_graphs", "empty or non-list", "non-empty list of TextualKnowledgeGraph")
        
        for i, graph in enumerate(val_graphs):
            if not isinstance(graph, TextualKnowledgeGraph):
                raise ValidationError(f"val_graphs[{i}]", type(graph).__name__, "TextualKnowledgeGraph")
        
        self.logger.info(f"Starting validation with {len(val_graphs)} graphs")
        with memory_management(cleanup_on_exit=True):
            self.model.eval()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            num_valid_graphs = 0
            num_errors = 0
            
            with torch.no_grad():
                for graph_idx, graph in enumerate(tqdm(val_graphs, desc="Validation")):
                    try:
                        # Validate graph
                        if graph.num_nodes == 0:
                            self.logger.warning(f"Skipping empty validation graph at index {graph_idx}")
                            continue
                        
                        if len(graph.node_texts) != graph.num_nodes:
                            self.logger.warning(f"Skipping validation graph {graph_idx}: text count mismatch")
                            continue
                        
                        # Move data to device with error handling
                        try:
                            edge_index = safe_cuda_operation(
                                lambda: graph.edge_index.to(self.device),
                                "validation edge_index transfer"
                            )
                            
                            node_features = graph.node_features
                            if node_features is not None:
                                node_features = safe_cuda_operation(
                                    lambda: node_features.to(self.device),
                                    "validation node_features transfer"
                                )
                            else:
                                node_features = torch.randn(graph.num_nodes, 128, device=self.device)
                        
                        except GPUError as e:
                            self.logger.warning(f"GPU error in validation for graph {graph_idx}: {e}")
                            num_errors += 1
                            continue
                        
                        # Forward pass
                        predictions = self.model(edge_index, node_features, graph.node_texts)
                        
                        # Calculate loss and accuracy
                        if task_type == "node_classification":
                            if graph.node_labels is not None:
                                labels = safe_cuda_operation(
                                    lambda: graph.node_labels.to(self.device),
                                    "validation labels transfer"
                                )
                                loss = loss_fn(predictions, labels)
                                
                                # Validate loss
                                if torch.isnan(loss) or torch.isinf(loss):
                                    self.logger.warning(f"Invalid validation loss for graph {graph_idx}: {loss.item()}")
                                    continue
                                
                                # Calculate accuracy
                                pred_classes = torch.argmax(predictions, dim=1)
                                correct = (pred_classes == labels).sum().item()
                                
                                total_loss += loss.item()
                                total_correct += correct
                                total_samples += labels.size(0)
                                num_valid_graphs += 1
                            else:
                                self.logger.debug(f"Skipping validation graph {graph_idx}: no labels")
                                continue
                        elif task_type == "link_prediction":
                            loss = self._link_prediction_loss(predictions, edge_index, loss_fn)
                            
                            if torch.isnan(loss) or torch.isinf(loss):
                                self.logger.warning(f"Invalid validation loss for graph {graph_idx}: {loss.item()}")
                                continue
                            
                            total_loss += loss.item()
                            total_samples += 1
                            num_valid_graphs += 1
                        
                    except torch.cuda.OutOfMemoryError:
                        self.logger.warning(f"CUDA OOM during validation for graph {graph_idx}")
                        if self.memory_monitor:
                            self._cleanup_training_memory()
                        num_errors += 1
                        continue
                    
                    except Exception as e:
                        self.logger.warning(f"Error validating graph {graph_idx}: {e}")
                        num_errors += 1
                        continue
        
            # Calculate metrics
            if num_valid_graphs == 0:
                self.logger.warning("No valid validation graphs processed")
                avg_loss = float('inf')
                accuracy = 0.0
            else:
                avg_loss = total_loss / num_valid_graphs
                accuracy = total_correct / max(total_samples, 1) if task_type == "node_classification" else None
            
            metrics = {
                "val_loss": avg_loss,
                "num_valid_graphs": num_valid_graphs,
                "num_errors": num_errors,
                "error_rate": num_errors / len(val_graphs),
            }
            
            if accuracy is not None:
                metrics["val_accuracy"] = accuracy
            
            self.logger.info(f"Validation completed: loss={avg_loss:.4f}, "
                           f"valid_graphs={num_valid_graphs}, errors={num_errors}")
            
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
                    self.logger.info(f"Early stopping triggered at epoch {epoch} (patience: {early_stopping_patience})")
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
    
    @log_function_call()
    def save_model(self, path: str) -> None:
        """Save model checkpoint with error handling.
        
        Args:
            path: Path to save the checkpoint
            
        Raises:
            ValidationError: If path is invalid
            IOError: If saving fails
        """
        if not isinstance(path, str) or len(path.strip()) == 0:
            raise ValidationError("path", path, "non-empty string")
        
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_val_loss": self.best_val_loss,
                "global_step": self.global_step,
                "config": self.model.get_config(),
                "training_history": self.training_history,
            }
            
            if self.scheduler:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
            # Add memory info if monitoring
            if self.memory_monitor:
                memory_info = self.memory_monitor.get_memory_info()
                checkpoint["memory_info"] = {
                    "process_memory_gb": memory_info.process_memory_gb,
                    "gpu_memory_gb": memory_info.gpu_allocated_gb or 0,
                }
            
            torch.save(checkpoint, path)
            self.logger.info(f"Model checkpoint saved to {path}")
            
        except Exception as e:
            error_msg = f"Failed to save model checkpoint to {path}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise IOError(error_msg)
    
    @log_function_call()
    def load_model(self, path: str) -> None:
        """Load model checkpoint with comprehensive error handling.
        
        Args:
            path: Path to the checkpoint file
            
        Raises:
            ValidationError: If path is invalid
            IOError: If loading fails
        """
        if not isinstance(path, str) or len(path.strip()) == 0:
            raise ValidationError("path", path, "non-empty string")
        
        try:
            self.logger.info(f"Loading model checkpoint from {path}")
            checkpoint = torch.load(path, map_location=self.device)
            
            # Validate checkpoint structure
            required_keys = ["model_state_dict", "optimizer_state_dict", "epoch", "best_val_loss"]
            for key in required_keys:
                if key not in checkpoint:
                    raise KeyError(f"Missing required key in checkpoint: {key}")
            
            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load training state
            self.epoch = checkpoint["epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]
            self.global_step = checkpoint.get("global_step", 0)
            
            if "training_history" in checkpoint:
                self.training_history = checkpoint["training_history"]
            
            # Load scheduler state if available
            if self.scheduler and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # Log memory info if available
            if "memory_info" in checkpoint:
                memory_info = checkpoint["memory_info"]
                self.logger.info(f"Checkpoint memory info: {memory_info}")
            
            self.logger.info(f"Model checkpoint loaded successfully from {path}")
            self.logger.info(f"Resumed at epoch {self.epoch}, step {self.global_step}, best_val_loss={self.best_val_loss:.4f}")
            
        except Exception as e:
            error_msg = f"Failed to load model checkpoint from {path}: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise IOError(error_msg)
    
    def cleanup(self) -> None:
        """Clean up trainer resources."""
        try:
            if self.memory_monitor:
                self.memory_monitor.stop_monitoring()
                self.logger.info("Stopped memory monitoring")
            
            if self.use_wandb:
                try:
                    wandb.finish()
                    self.logger.info("Closed wandb logging")
                except Exception as e:
                    self.logger.warning(f"Error closing wandb: {e}")
            
            # Final memory cleanup
            self._cleanup_training_memory()
            
        except Exception as e:
            self.logger.error(f"Error during trainer cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid exceptions in destructor


class ZeroShotEvaluator(LoggerMixin):
    """Evaluator for zero-shot transfer capabilities with comprehensive error handling."""
    
    def __init__(self, model: HyperGNN, device: str = "auto"):
        """Initialize evaluator with validation.
        
        Args:
            model: Trained HyperGNN model
            device: Device to use
            
        Raises:
            ValidationError: If parameters are invalid
        """
        super().__init__()
        
        if not isinstance(model, HyperGNN):
            raise ValidationError("model", type(model).__name__, "HyperGNN instance")
        
        if device not in ["auto", "cuda", "cpu"]:
            raise ValidationError("device", device, "one of: auto, cuda, cpu")
        
        self.model = model
        
        try:
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"ZeroShotEvaluator initialized on device: {self.device}")
            
        except Exception as e:
            raise GPUError("evaluator_initialization", f"Failed to initialize evaluator: {e}")
    
    @log_function_call()
    def evaluate_transfer(
        self,
        source_graphs: List[TextualKnowledgeGraph],
        target_graphs: List[TextualKnowledgeGraph],
        task_type: str = "node_classification",
    ) -> Dict[str, float]:
        """Evaluate zero-shot transfer performance with comprehensive error handling.
        
        Args:
            source_graphs: Source domain graphs (for reference)
            target_graphs: Target domain graphs (for evaluation)
            task_type: Type of task
            
        Returns:
            Transfer performance metrics
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        if not isinstance(target_graphs, list) or len(target_graphs) == 0:
            raise ValidationError("target_graphs", "empty or non-list", "non-empty list")
        
        if task_type not in ["node_classification", "link_prediction"]:
            raise ValidationError("task_type", task_type, "node_classification or link_prediction")
        
        for i, graph in enumerate(target_graphs):
            if not isinstance(graph, TextualKnowledgeGraph):
                raise ValidationError(f"target_graphs[{i}]", type(graph).__name__, "TextualKnowledgeGraph")
        self.logger.info(f"Evaluating zero-shot transfer on {len(target_graphs)} target graphs...")
        
        with memory_management(cleanup_on_exit=True):
            total_correct = 0
            total_samples = 0
            graph_accuracies = []
            num_valid_graphs = 0
            num_errors = 0
            
            with torch.no_grad():
                for graph_idx, graph in enumerate(tqdm(target_graphs, desc="Zero-shot evaluation")):
                    try:
                        # Validate graph
                        if graph.num_nodes == 0:
                            self.logger.warning(f"Skipping empty graph at index {graph_idx}")
                            continue
                        
                        if len(graph.node_texts) != graph.num_nodes:
                            self.logger.warning(f"Skipping graph {graph_idx}: text count mismatch")
                            continue
                        
                        # Move data to device with error handling
                        try:
                            edge_index = safe_cuda_operation(
                                lambda: graph.edge_index.to(self.device),
                                "evaluation edge_index transfer"
                            )
                            
                            node_features = graph.node_features
                            if node_features is not None:
                                node_features = safe_cuda_operation(
                                    lambda: node_features.to(self.device),
                                    "evaluation node_features transfer"
                                )
                            else:
                                node_features = torch.randn(graph.num_nodes, 128, device=self.device)
                        
                        except GPUError as e:
                            self.logger.warning(f"GPU error in evaluation for graph {graph_idx}: {e}")
                            num_errors += 1
                            continue
                        
                        # Generate predictions
                        predictions = self.model(edge_index, node_features, graph.node_texts)
                        
                        if task_type == "node_classification" and graph.node_labels is not None:
                            labels = safe_cuda_operation(
                                lambda: graph.node_labels.to(self.device),
                                "evaluation labels transfer"
                            )
                            
                            pred_classes = torch.argmax(predictions, dim=1)
                            correct = (pred_classes == labels).sum().item()
                            
                            graph_accuracy = correct / labels.size(0)
                            graph_accuracies.append(graph_accuracy)
                            
                            total_correct += correct
                            total_samples += labels.size(0)
                            num_valid_graphs += 1
                        else:
                            if task_type == "node_classification":
                                self.logger.debug(f"Skipping graph {graph_idx}: no labels for classification")
                            continue
                    
                    except torch.cuda.OutOfMemoryError:
                        self.logger.warning(f"CUDA OOM during evaluation for graph {graph_idx}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        num_errors += 1
                        continue
                    
                    except Exception as e:
                        self.logger.warning(f"Error evaluating graph {graph_idx}: {e}")
                        num_errors += 1
                        continue
        
            # Calculate metrics
            if num_valid_graphs == 0:
                self.logger.warning("No valid graphs processed for evaluation")
                overall_accuracy = 0.0
                avg_graph_accuracy = 0.0
            else:
                overall_accuracy = total_correct / max(total_samples, 1)
                avg_graph_accuracy = sum(graph_accuracies) / max(len(graph_accuracies), 1)
            
            metrics = {
                "zero_shot_accuracy": overall_accuracy,
                "avg_graph_accuracy": avg_graph_accuracy,
                "num_target_graphs": len(target_graphs),
                "num_valid_graphs": num_valid_graphs,
                "num_errors": num_errors,
                "error_rate": num_errors / len(target_graphs),
                "total_target_nodes": total_samples,
            }
            
            self.logger.info(f"Zero-shot transfer results:")
            self.logger.info(f"  Overall accuracy: {overall_accuracy:.4f}")
            self.logger.info(f"  Average graph accuracy: {avg_graph_accuracy:.4f}")
            self.logger.info(f"  Evaluated on {num_valid_graphs}/{len(target_graphs)} graphs ({total_samples} nodes)")
            self.logger.info(f"  Error rate: {num_errors / len(target_graphs):.2%}")
            
            return metrics
    
    @log_function_call()
    def analyze_text_similarity(
        self,
        source_texts: List[str],
        target_texts: List[str],
    ) -> Dict[str, float]:
        """Analyze text similarity between source and target domains with validation.
        
        Args:
            source_texts: Source domain node texts
            target_texts: Target domain node texts
            
        Returns:
            Similarity metrics
            
        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        if not isinstance(source_texts, list) or len(source_texts) == 0:
            raise ValidationError("source_texts", "empty or non-list", "non-empty list of strings")
        
        if not isinstance(target_texts, list) or len(target_texts) == 0:
            raise ValidationError("target_texts", "empty or non-list", "non-empty list of strings")
        
        for i, text in enumerate(source_texts):
            if not isinstance(text, str):
                raise ValidationError(f"source_texts[{i}]", type(text).__name__, "string")
        
        for i, text in enumerate(target_texts):
            if not isinstance(text, str):
                raise ValidationError(f"target_texts[{i}]", type(text).__name__, "string")
        self.logger.info(f"Analyzing text similarity: {len(source_texts)} source texts, {len(target_texts)} target texts")
        
        try:
            with torch.no_grad():
                # Encode texts with error handling
                source_embeddings = safe_cuda_operation(
                    lambda: self.model.text_encoder(source_texts),
                    "source text encoding"
                )
                
                target_embeddings = safe_cuda_operation(
                    lambda: self.model.text_encoder(target_texts),
                    "target text encoding"
                )
                
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
                std_similarity = max_similarities.std().item()
                
                results = {
                    "avg_max_similarity": avg_max_similarity,
                    "min_max_similarity": min_max_similarity,
                    "similarity_std": std_similarity,
                    "num_source_texts": len(source_texts),
                    "num_target_texts": len(target_texts),
                }
                
                self.logger.info(f"Text similarity analysis completed: avg={avg_max_similarity:.4f}, "
                               f"min={min_max_similarity:.4f}, std={std_similarity:.4f}")
                
                return results
                
        except Exception as e:
            error_msg = f"Text similarity analysis failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            if isinstance(e, (ValidationError, GPUError)):
                raise
            raise RuntimeError(error_msg)