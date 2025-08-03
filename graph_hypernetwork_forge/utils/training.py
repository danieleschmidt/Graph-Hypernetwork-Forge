"""Training utilities for HyperGNN models.

Provides training loops, evaluation metrics, and optimization utilities
for hypernetwork-based graph neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import time
import logging
from tqdm import tqdm
import wandb
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for training HyperGNN models."""
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        num_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 1e-4,
        scheduler_type: str = "cosine",  # "cosine", "step", "plateau"
        scheduler_params: Dict[str, Any] = None,
        gradient_clip: Optional[float] = 1.0,
        warmup_epochs: int = 5,
        save_best: bool = True,
        save_last: bool = True,
        log_interval: int = 10,
        eval_interval: int = 1,
        use_wandb: bool = False,
        wandb_project: str = "hypergnn",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        self.gradient_clip = gradient_clip
        self.warmup_epochs = warmup_epochs
        self.save_best = save_best
        self.save_last = save_last
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.checkpoint_dir = checkpoint_dir


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricsTracker:
    """Tracks and computes training/validation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.predictions = []
        self.targets = []
        self.times = []
    
    def update(
        self,
        loss: float,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        batch_time: float = 0.0,
    ):
        self.losses.append(loss)
        self.predictions.extend(predictions.detach().cpu().numpy())
        self.targets.extend(targets.detach().cpu().numpy())
        self.times.append(batch_time)
    
    def compute_metrics(self, task_type: str = "classification") -> Dict[str, float]:
        if not self.losses:
            return {}
        
        metrics = {
            "loss": np.mean(self.losses),
            "batch_time": np.mean(self.times),
        }
        
        if task_type == "classification":
            predictions = np.array(self.predictions)
            targets = np.array(self.targets)
            
            # Binary classification
            if predictions.shape[-1] == 1 or len(predictions.shape) == 1:
                pred_labels = (predictions > 0.5).astype(int)
                metrics["accuracy"] = accuracy_score(targets, pred_labels)
                
                if len(np.unique(targets)) == 2:  # Binary case
                    try:
                        metrics["auc"] = roc_auc_score(targets, predictions)
                    except ValueError:
                        pass
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets, pred_labels, average="binary", zero_division=0
                )
                metrics.update({"precision": precision, "recall": recall, "f1": f1})
            
            # Multi-class classification
            else:
                pred_labels = np.argmax(predictions, axis=1)
                metrics["accuracy"] = accuracy_score(targets, pred_labels)
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets, pred_labels, average="macro", zero_division=0
                )
                metrics.update({"precision": precision, "recall": recall, "f1": f1})
        
        elif task_type == "regression":
            predictions = np.array(self.predictions)
            targets = np.array(self.targets)
            
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            metrics.update({"mse": mse, "mae": mae, "rmse": np.sqrt(mse)})
        
        return metrics


class HyperGNNTrainer:
    """Trainer class for HyperGNN models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Setup logging
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if config.use_wandb:
            wandb.init(project=config.wandb_project, config=vars(config))
            wandb.watch(model)
        
        self.best_val_loss = float('inf')
        self.current_epoch = 0
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                **self.config.scheduler_params
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_params.get("step_size", 30),
                gamma=self.config.scheduler_params.get("gamma", 0.1),
            )
        elif self.config.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.config.scheduler_params.get("patience", 5),
                factor=self.config.scheduler_params.get("factor", 0.5),
            )
        else:
            return None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        task_type: str = "classification",
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        progress = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress):
            start_time = time.time()
            
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if task_type == "link_prediction":
                predictions = self._forward_link_prediction(batch)
                targets = batch["labels"]
            elif task_type == "node_classification":
                predictions = self._forward_node_classification(batch)
                targets = batch["labels"]
            elif task_type == "graph_classification":
                predictions = self._forward_graph_classification(batch)
                targets = batch["labels"]
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            if self.config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            # Update metrics
            batch_time = time.time() - start_time
            self.train_metrics.update(
                loss=loss.item(),
                predictions=predictions,
                targets=targets,
                batch_time=batch_time,
            )
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{current_lr:.2e}",
                })
        
        return self.train_metrics.compute_metrics(task_type)
    
    def evaluate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        task_type: str = "classification",
    ) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                start_time = time.time()
                
                batch = self._batch_to_device(batch)
                
                if task_type == "link_prediction":
                    predictions = self._forward_link_prediction(batch)
                    targets = batch["labels"]
                elif task_type == "node_classification":
                    predictions = self._forward_node_classification(batch)
                    targets = batch["labels"]
                elif task_type == "graph_classification":
                    predictions = self._forward_graph_classification(batch)
                    targets = batch["labels"]
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
                
                loss = criterion(predictions, targets)
                
                batch_time = time.time() - start_time
                self.val_metrics.update(
                    loss=loss.item(),
                    predictions=predictions,
                    targets=targets,
                    batch_time=batch_time,
                )
        
        return self.val_metrics.compute_metrics(task_type)
    
    def _forward_link_prediction(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for link prediction."""
        # Get node embeddings
        embeddings = self.model(
            edge_index=batch["edge_index"],
            node_features=batch["node_features"],
            node_texts=batch["node_texts"],
            return_embeddings=True,
        )[1]
        
        # Get source and target embeddings
        src_embeddings = embeddings[batch["source_indices"]]
        dst_embeddings = embeddings[batch["target_indices"]]
        
        # Compute link scores (dot product)
        scores = (src_embeddings * dst_embeddings).sum(dim=1)
        return torch.sigmoid(scores)
    
    def _forward_node_classification(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for node classification."""
        predictions = self.model(
            edge_index=batch["edge_index"],
            node_features=batch["node_features"],
            node_texts=batch["node_texts"],
        )
        
        # Select target nodes
        return predictions[batch["target_node_indices"]]
    
    def _forward_graph_classification(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for graph classification."""
        embeddings = self.model(
            edge_index=batch["edge_index"],
            node_features=batch["node_features"],
            node_texts=batch["node_texts"],
            return_embeddings=True,
        )[1]
        
        # Graph-level pooling (mean pooling)
        batch_indices = batch.get("batch_indices")
        if batch_indices is not None:
            graph_embeddings = self._global_mean_pool(embeddings, batch_indices)
        else:
            graph_embeddings = embeddings.mean(dim=0, keepdim=True)
        
        return self.model.predictor(graph_embeddings)
    
    def _global_mean_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Global mean pooling for graph-level representations."""
        size = int(batch.max().item() + 1)
        return torch.zeros(size, x.size(1), device=x.device).scatter_add_(
            0, batch.unsqueeze(-1).expand(-1, x.size(1)), x
        ) / torch.bincount(batch, minlength=size).unsqueeze(-1).float()
    
    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        task_type: str = "classification",
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        if criterion is None:
            if task_type in ["classification", "link_prediction"]:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.MSELoss()
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader, criterion, task_type)
            history["train_loss"].append(train_metrics["loss"])
            
            # Validation
            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader, criterion, task_type)
                history["val_loss"].append(val_metrics["loss"])
                
                # Scheduler step
                if self.scheduler and self.config.scheduler_type == "plateau":
                    self.scheduler.step(val_metrics["loss"])
                
                # Early stopping
                if self.early_stopping(val_metrics["loss"]):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Save best model
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    if self.config.save_best:
                        self.save_checkpoint("best")
                
                # Logging
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                
                if self.config.use_wandb:
                    wandb.log(log_dict)
                
                logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}"
                )
            
            # Scheduler step (non-plateau)
            if self.scheduler and self.config.scheduler_type != "plateau":
                self.scheduler.step()
        
        # Save final model
        if self.config.save_last:
            self.save_checkpoint("last")
        
        return history
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")


def get_criterion(task_type: str, **kwargs) -> nn.Module:
    """Get appropriate loss function for task type."""
    if task_type == "link_prediction":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif task_type == "node_classification":
        return nn.CrossEntropyLoss(**kwargs)
    elif task_type == "graph_classification":
        return nn.CrossEntropyLoss(**kwargs)
    elif task_type == "regression":
        return nn.MSELoss(**kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def compute_zero_shot_metrics(
    model: nn.Module,
    target_datasets: List[DataLoader],
    device: torch.device,
    task_type: str = "link_prediction",
) -> Dict[str, float]:
    """Compute zero-shot performance metrics on target domains."""
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for dataset in target_datasets:
            metrics_tracker = MetricsTracker()
            criterion = get_criterion(task_type)
            
            for batch in dataset:
                # Move to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                if task_type == "link_prediction":
                    predictions = model.zero_shot_inference(
                        edge_index=batch["edge_index"],
                        node_features=batch["node_features"],
                        node_texts=batch["node_texts"],
                    )
                    targets = batch["labels"]
                else:
                    predictions = model.zero_shot_inference(
                        edge_index=batch["edge_index"],
                        node_features=batch["node_features"],
                        node_texts=batch["node_texts"],
                    )
                    targets = batch["labels"]
                
                loss = criterion(predictions, targets)
                metrics_tracker.update(
                    loss=loss.item(),
                    predictions=predictions,
                    targets=targets,
                )
            
            dataset_metrics = metrics_tracker.compute_metrics(task_type)
            all_metrics.append(dataset_metrics)
    
    # Aggregate metrics across datasets
    aggregated = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            aggregated[f"avg_{key}"] = np.mean(values)
            aggregated[f"std_{key}"] = np.std(values)
    
    return aggregated