"""Evaluation utilities and benchmarking for HyperGNN models.

Provides comprehensive evaluation metrics, benchmarking utilities,
and analysis tools for hypernetwork-based graph neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for different graph learning tasks."""
    
    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = "macro",
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        metrics.update({
            f"precision_{average}": precision,
            f"recall_{average}": recall,
            f"f1_{average}": f1,
        })
        
        # Per-class metrics for multiclass
        if len(np.unique(y_true)) > 2:
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
                metrics[f"precision_class_{i}"] = p
                metrics[f"recall_class_{i}"] = r
                metrics[f"f1_class_{i}"] = f
        
        # Probability-based metrics
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
                    metrics["auc_pr"] = average_precision_score(y_true, y_prob)
                else:  # Multiclass
                    metrics["auc_roc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
                    metrics["auc_roc_ovo"] = roc_auc_score(y_true, y_prob, multi_class="ovo", average=average)
            except ValueError as e:
                logger.warning(f"Could not compute AUC metrics: {e}")
        
        return metrics
    
    @staticmethod
    def link_prediction_metrics(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute link prediction specific metrics."""
        y_pred = (y_scores > threshold).astype(int)
        
        # Basic classification metrics
        metrics = EvaluationMetrics.classification_metrics(y_true, y_pred, y_scores, average="binary")
        
        # Ranking metrics
        metrics.update({
            "auc_roc": roc_auc_score(y_true, y_scores),
            "auc_pr": average_precision_score(y_true, y_scores),
        })
        
        # Hits@K metrics
        for k in [1, 3, 10]:
            metrics[f"hits@{k}"] = EvaluationMetrics._hits_at_k(y_true, y_scores, k)
        
        # Mean Reciprocal Rank
        metrics["mrr"] = EvaluationMetrics._mean_reciprocal_rank(y_true, y_scores)
        
        return metrics
    
    @staticmethod
    def _hits_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """Compute Hits@K metric."""
        # Sort by scores in descending order
        sorted_indices = np.argsort(-y_scores)
        top_k_indices = sorted_indices[:k]
        
        # Check if any of the top-k predictions are positive
        return float(np.any(y_true[top_k_indices] == 1))
    
    @staticmethod
    def _mean_reciprocal_rank(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute Mean Reciprocal Rank."""
        # Sort by scores in descending order
        sorted_indices = np.argsort(-y_scores)
        
        # Find rank of first positive example
        for rank, idx in enumerate(sorted_indices, 1):
            if y_true[idx] == 1:
                return 1.0 / rank
        
        return 0.0  # No positive examples found
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics."""
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }


class BenchmarkEvaluator:
    """Evaluator for benchmarking HyperGNN against baselines."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: Optional[Path] = None,
    ):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("./evaluation_results")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
    def evaluate_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        task_type: str = "link_prediction",
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate model on a specific dataset."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                start_time = time.time()
                
                # Move batch to device
                batch = self._batch_to_device(batch)
                
                # Forward pass
                if task_type == "link_prediction":
                    logits = self._forward_link_prediction(batch)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    targets = batch["labels"]
                elif task_type == "node_classification":
                    logits = self._forward_node_classification(batch)
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(logits, dim=-1)
                    targets = batch["labels"]
                elif task_type == "graph_classification":
                    logits = self._forward_graph_classification(batch)
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(logits, dim=-1)
                    targets = batch["labels"]
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Collect results
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        # Compute metrics
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        if task_type == "link_prediction":
            metrics = EvaluationMetrics.link_prediction_metrics(y_true, y_prob)
        elif task_type in ["node_classification", "graph_classification"]:
            metrics = EvaluationMetrics.classification_metrics(y_true, y_pred, y_prob)
        else:
            metrics = EvaluationMetrics.regression_metrics(y_true, y_pred)
        
        # Add performance metrics
        metrics.update({
            "avg_inference_time": np.mean(inference_times),
            "total_inference_time": np.sum(inference_times),
            "throughput": len(all_predictions) / np.sum(inference_times),
        })
        
        # Store results
        result = {
            "dataset": dataset_name,
            "task_type": task_type,
            "metrics": metrics,
            "num_samples": len(all_predictions),
        }
        
        if return_predictions:
            result.update({
                "predictions": y_pred,
                "targets": y_true,
                "probabilities": y_prob,
            })
        
        self.results[dataset_name] = result
        logger.info(f"Evaluated {dataset_name}: {metrics}")
        
        return result
    
    def evaluate_zero_shot_transfer(
        self,
        source_datasets: List[Tuple[torch.utils.data.DataLoader, str]],
        target_datasets: List[Tuple[torch.utils.data.DataLoader, str]],
        task_type: str = "link_prediction",
    ) -> Dict[str, Any]:
        """Evaluate zero-shot transfer performance."""
        results = {
            "source_performance": {},
            "target_performance": {},
            "transfer_metrics": {},
        }
        
        # Evaluate on source domains
        for dataloader, name in source_datasets:
            result = self.evaluate_dataset(dataloader, f"source_{name}", task_type)
            results["source_performance"][name] = result["metrics"]
        
        # Evaluate on target domains (zero-shot)
        for dataloader, name in target_datasets:
            result = self.evaluate_dataset(dataloader, f"target_{name}", task_type)
            results["target_performance"][name] = result["metrics"]
        
        # Compute transfer metrics
        source_avg = self._average_metrics(list(results["source_performance"].values()))
        target_avg = self._average_metrics(list(results["target_performance"].values()))
        
        results["transfer_metrics"] = {
            "source_average": source_avg,
            "target_average": target_avg,
            "transfer_gap": {
                k: source_avg.get(k, 0) - target_avg.get(k, 0)
                for k in source_avg.keys()
            }
        }
        
        return results
    
    def compare_with_baselines(
        self,
        baseline_results: Dict[str, Dict[str, float]],
        dataset_name: str,
        primary_metric: str = "f1_macro",
    ) -> Dict[str, Any]:
        """Compare HyperGNN results with baseline methods."""
        if dataset_name not in self.results:
            raise ValueError(f"No results found for dataset {dataset_name}")
        
        hypergnn_metrics = self.results[dataset_name]["metrics"]
        
        comparison = {
            "hypergnn": hypergnn_metrics[primary_metric],
            "baselines": {},
            "improvements": {},
        }
        
        for baseline_name, baseline_metrics in baseline_results.items():
            baseline_score = baseline_metrics.get(primary_metric, 0)
            comparison["baselines"][baseline_name] = baseline_score
            
            improvement = ((hypergnn_metrics[primary_metric] - baseline_score) / baseline_score * 100
                          if baseline_score > 0 else 0)
            comparison["improvements"][baseline_name] = improvement
        
        return comparison
    
    def generate_report(
        self,
        include_plots: bool = True,
        save_format: str = "json",
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            "model_info": {
                "model_class": self.model.__class__.__name__,
                "device": str(self.device),
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
            },
            "evaluation_results": self.results,
            "summary": self._compute_summary_statistics(),
        }
        
        # Save report
        if save_format == "json":
            report_path = self.save_dir / "evaluation_report.json"
            with open(report_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_report = self._convert_numpy_to_lists(report)
                json.dump(json_report, f, indent=2)
            logger.info(f"Report saved to {report_path}")
        
        # Generate plots
        if include_plots:
            self.generate_plots()
        
        return report
    
    def generate_plots(self):
        """Generate evaluation plots and visualizations."""
        # Performance comparison plot
        self._plot_performance_comparison()
        
        # Confusion matrices
        self._plot_confusion_matrices()
        
        # Zero-shot transfer analysis
        if "transfer_metrics" in self.results:
            self._plot_transfer_analysis()
    
    def _plot_performance_comparison(self):
        """Plot performance comparison across datasets."""
        if not self.results:
            return
        
        datasets = []
        metrics_data = {}
        
        for dataset_name, result in self.results.items():
            datasets.append(dataset_name)
            for metric_name, metric_value in result["metrics"].items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(metric_value)
        
        # Plot key metrics
        key_metrics = ["accuracy", "f1_macro", "auc_roc"]
        available_metrics = [m for m in key_metrics if m in metrics_data]
        
        if available_metrics:
            fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 4))
            if len(available_metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(available_metrics):
                axes[i].bar(datasets, metrics_data[metric])
                axes[i].set_title(f"{metric.upper()}")
                axes[i].set_ylabel("Score")
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for classification tasks."""
        for dataset_name, result in self.results.items():
            if "predictions" in result and "targets" in result:
                y_true = result["targets"]
                y_pred = result["predictions"]
                
                # Skip if not classification
                if len(np.unique(y_true)) > 10:  # Too many classes
                    continue
                
                cm = confusion_matrix(y_true, y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Confusion Matrix - {dataset_name}")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                plt.savefig(self.save_dir / f"confusion_matrix_{dataset_name}.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_transfer_analysis(self):
        """Plot zero-shot transfer analysis."""
        # Implementation for transfer analysis plots
        pass
    
    def _forward_link_prediction(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for link prediction."""
        predictions = self.model.zero_shot_inference(
            edge_index=batch["edge_index"],
            node_features=batch["node_features"],
            node_texts=batch["node_texts"],
        )
        return predictions
    
    def _forward_node_classification(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for node classification."""
        predictions = self.model.zero_shot_inference(
            edge_index=batch["edge_index"],
            node_features=batch["node_features"],
            node_texts=batch["node_texts"],
        )
        return predictions[batch["target_node_indices"]]
    
    def _forward_graph_classification(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass for graph classification."""
        embeddings = self.model.zero_shot_inference(
            edge_index=batch["edge_index"],
            node_features=batch["node_features"],
            node_texts=batch["node_texts"],
        )
        # Perform graph-level pooling
        return embeddings.mean(dim=0, keepdim=True)
    
    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across multiple evaluations."""
        if not metrics_list:
            return {}
        
        averaged = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            averaged[key] = np.mean(values)
        
        return averaged
    
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics across all evaluations."""
        if not self.results:
            return {}
        
        all_metrics = [result["metrics"] for result in self.results.values()]
        summary = {}
        
        # Compute averages and standard deviations
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                summary[f"avg_{metric_name}"] = np.mean(values)
                summary[f"std_{metric_name}"] = np.std(values)
                summary[f"min_{metric_name}"] = np.min(values)
                summary[f"max_{metric_name}"] = np.max(values)
        
        return summary
    
    def _convert_numpy_to_lists(self, obj: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_lists(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj