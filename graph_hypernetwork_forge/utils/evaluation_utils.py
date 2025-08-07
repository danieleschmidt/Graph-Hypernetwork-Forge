"""Evaluation utilities for Graph Hypernetwork Forge."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix as sklearn_confusion_matrix,
    classification_report as sklearn_classification_report,
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings


def calculate_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    task_type: str = 'classification'
) -> float:
    """
    Calculate accuracy for predictions.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        task_type: Type of task ('classification' or 'multilabel')
        
    Returns:
        Accuracy score
    """
    if task_type == 'classification':
        # For classification, predictions should be class indices
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # If predictions are logits/probabilities, get argmax
            predictions = predictions.argmax(dim=1)
        
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
    
    elif task_type == 'multilabel':
        # For multilabel, predictions should be binary (0/1)
        if predictions.dim() > 1:
            # If predictions are probabilities, threshold at 0.5
            predictions = (predictions > 0.5).float()
        
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return accuracy


def calculate_precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'binary',
    zero_division: Union[str, int] = 'warn'
) -> float:
    """
    Calculate precision score.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')
        zero_division: How to handle zero division
        
    Returns:
        Precision score
    """
    # Convert to numpy for sklearn
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Handle multi-class predictions
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(
            targets, predictions, 
            average=average, 
            zero_division=zero_division
        )
    
    return float(precision)


def calculate_recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'binary',
    zero_division: Union[str, int] = 'warn'
) -> float:
    """
    Calculate recall score.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')
        zero_division: How to handle zero division
        
    Returns:
        Recall score
    """
    # Convert to numpy for sklearn
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Handle multi-class predictions
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recall = recall_score(
            targets, predictions, 
            average=average, 
            zero_division=zero_division
        )
    
    return float(recall)


def calculate_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'binary',
    zero_division: Union[str, int] = 'warn'
) -> float:
    """
    Calculate F1 score.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')
        zero_division: How to handle zero division
        
    Returns:
        F1 score
    """
    # Convert to numpy for sklearn
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Handle multi-class predictions
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = f1_score(
            targets, predictions, 
            average=average, 
            zero_division=zero_division
        )
    
    return float(f1)


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None
) -> torch.Tensor:
    """
    Generate confusion matrix.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        num_classes: Number of classes (auto-detected if None)
        
    Returns:
        Confusion matrix as torch.Tensor
    """
    # Convert to numpy for sklearn
    if isinstance(predictions, torch.Tensor):
        pred_np = predictions.cpu().numpy()
    else:
        pred_np = predictions
        
    if isinstance(targets, torch.Tensor):
        target_np = targets.cpu().numpy()
    else:
        target_np = targets
    
    # Handle multi-class predictions
    if pred_np.ndim > 1 and pred_np.shape[1] > 1:
        pred_np = np.argmax(pred_np, axis=1)
    
    # Determine number of classes
    if num_classes is None:
        num_classes = max(int(target_np.max()) + 1, int(pred_np.max()) + 1)
    
    cm = sklearn_confusion_matrix(
        target_np, pred_np,
        labels=list(range(num_classes))
    )
    
    return torch.from_numpy(cm)


def classification_report(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_names: Optional[List[str]] = None,
    output_dict: bool = True
) -> Dict[str, Any]:
    """
    Generate classification report.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        target_names: Optional class names
        output_dict: Whether to return dict (True) or string (False)
        
    Returns:
        Classification report
    """
    # Convert to numpy for sklearn
    if isinstance(predictions, torch.Tensor):
        pred_np = predictions.cpu().numpy()
    else:
        pred_np = predictions
        
    if isinstance(targets, torch.Tensor):
        target_np = targets.cpu().numpy()
    else:
        target_np = targets
    
    # Handle multi-class predictions
    if pred_np.ndim > 1 and pred_np.shape[1] > 1:
        pred_np = np.argmax(pred_np, axis=1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sklearn_classification_report(
            target_np, pred_np,
            target_names=target_names,
            output_dict=output_dict,
            zero_division=0
        )
    
    if output_dict:
        # Add convenience keys for common metrics
        if 'macro avg' in report:
            report['precision'] = report['macro avg']['precision']
            report['recall'] = report['macro avg']['recall']
            report['f1_score'] = report['macro avg']['f1-score']
        
        return report
    else:
        return {'report': report}


def top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth targets
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy
    """
    if predictions.dim() == 1:
        # Binary case, k should be 1
        return calculate_accuracy(predictions, targets)
    
    # Get top-k predictions
    _, top_k_preds = predictions.topk(k, dim=1, largest=True, sorted=True)
    
    # Check if true labels are in top-k predictions
    targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=1)
    
    return correct.float().mean().item()


def calculate_auc(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_type: str = 'binary'
) -> float:
    """
    Calculate Area Under Curve (AUC).
    
    Args:
        predictions: Model predictions (probabilities)
        targets: Ground truth targets
        task_type: Type of task ('binary' or 'multiclass')
        
    Returns:
        AUC score
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        pred_np = predictions.cpu().numpy()
    else:
        pred_np = predictions
        
    if isinstance(targets, torch.Tensor):
        target_np = targets.cpu().numpy()
    else:
        target_np = targets
    
    try:
        if task_type == 'binary':
            # For binary classification
            if pred_np.ndim > 1:
                pred_np = pred_np[:, 1]  # Use probability of positive class
            auc = roc_auc_score(target_np, pred_np)
        
        elif task_type == 'multiclass':
            # For multiclass classification
            auc = roc_auc_score(target_np, pred_np, multi_class='ovr', average='macro')
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return float(auc)
    
    except ValueError as e:
        # Handle cases where AUC cannot be computed (e.g., only one class present)
        warnings.warn(f"Could not compute AUC: {e}")
        return 0.0


def calculate_average_precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_type: str = 'binary'
) -> float:
    """
    Calculate Average Precision (AP).
    
    Args:
        predictions: Model predictions (probabilities)
        targets: Ground truth targets
        task_type: Type of task ('binary' or 'multilabel')
        
    Returns:
        Average precision score
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        pred_np = predictions.cpu().numpy()
    else:
        pred_np = predictions
        
    if isinstance(targets, torch.Tensor):
        target_np = targets.cpu().numpy()
    else:
        target_np = targets
    
    try:
        if task_type == 'binary':
            if pred_np.ndim > 1:
                pred_np = pred_np[:, 1]  # Use probability of positive class
            ap = average_precision_score(target_np, pred_np)
        
        elif task_type == 'multilabel':
            ap = average_precision_score(target_np, pred_np, average='macro')
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return float(ap)
    
    except ValueError as e:
        warnings.warn(f"Could not compute Average Precision: {e}")
        return 0.0


def calculate_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of regression metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        pred_np = predictions.cpu().numpy().flatten()
    else:
        pred_np = predictions.flatten()
        
    if isinstance(targets, torch.Tensor):
        target_np = targets.cpu().numpy().flatten()
    else:
        target_np = targets.flatten()
    
    metrics = {}
    
    # Mean Squared Error
    metrics['mse'] = mean_squared_error(target_np, pred_np)
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(target_np, pred_np)
    
    # R-squared
    metrics['r2'] = r2_score(target_np, pred_np)
    
    # Mean Absolute Percentage Error
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((target_np - pred_np) / target_np)) * 100
        metrics['mape'] = mape if np.isfinite(mape) else float('inf')
    
    return metrics


def evaluate_model_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_type: str = 'classification',
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        task_type: Type of task ('classification', 'regression', 'multilabel')
        num_classes: Number of classes for classification
        class_names: Optional class names
        
    Returns:
        Dictionary containing all relevant metrics
    """
    results = {'task_type': task_type}
    
    if task_type == 'classification':
        # Classification metrics
        results['accuracy'] = calculate_accuracy(predictions, targets)
        results['precision'] = calculate_precision(predictions, targets, average='macro')
        results['recall'] = calculate_recall(predictions, targets, average='macro')
        results['f1_score'] = calculate_f1_score(predictions, targets, average='macro')
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(predictions, targets, num_classes)
        
        # Classification report
        results['classification_report'] = classification_report(
            predictions, targets, target_names=class_names
        )
        
        # AUC if probabilities are provided
        if predictions.dim() > 1 and predictions.size(1) > 1:
            results['auc'] = calculate_auc(predictions, targets, 'multiclass' if num_classes > 2 else 'binary')
        
        # Top-k accuracy for multi-class
        if predictions.dim() > 1 and predictions.size(1) > 2:
            results['top_3_accuracy'] = top_k_accuracy(predictions, targets, k=3)
            results['top_5_accuracy'] = top_k_accuracy(predictions, targets, k=5)
    
    elif task_type == 'regression':
        # Regression metrics
        results.update(calculate_regression_metrics(predictions, targets))
    
    elif task_type == 'multilabel':
        # Multi-label classification metrics
        results['accuracy'] = calculate_accuracy(predictions, targets, task_type='multilabel')
        results['precision'] = calculate_precision(predictions, targets, average='macro')
        results['recall'] = calculate_recall(predictions, targets, average='macro')
        results['f1_score'] = calculate_f1_score(predictions, targets, average='macro')
        
        # Average precision for multi-label
        if predictions.dim() > 1:
            results['average_precision'] = calculate_average_precision(
                predictions, targets, task_type='multilabel'
            )
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return results


def compute_metric_confidence_interval(
    metric_values: List[float],
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for metric values.
    
    Args:
        metric_values: List of metric values from different runs
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(metric_values) < 2:
        return (0.0, 0.0)
    
    alpha = 1 - confidence_level
    sorted_values = np.sort(metric_values)
    n = len(sorted_values)
    
    lower_idx = int(np.floor((alpha/2) * n))
    upper_idx = int(np.ceil((1 - alpha/2) * n)) - 1
    
    lower_bound = sorted_values[max(0, lower_idx)]
    upper_bound = sorted_values[min(n-1, upper_idx)]
    
    return (float(lower_bound), float(upper_bound))