"""Comprehensive tests for evaluation utilities and metrics."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import json

from graph_hypernetwork_forge.utils.evaluation import (
    EvaluationMetrics,
    BenchmarkEvaluator,
)
from graph_hypernetwork_forge.models import HyperGNN
from graph_hypernetwork_forge.data import create_synthetic_kg, LinkPredictionDataset, create_dataloader


class TestEvaluationMetrics:
    """Test evaluation metrics computation."""
    
    def test_classification_metrics_binary(self):
        """Test binary classification metrics."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
        
        metrics = EvaluationMetrics.classification_metrics(y_true, y_pred, y_prob, average="binary")
        
        assert "accuracy" in metrics
        assert "precision_binary" in metrics
        assert "recall_binary" in metrics
        assert "f1_binary" in metrics
        assert "auc_roc" in metrics
        assert "auc_pr" in metrics
        
        # Check accuracy calculation
        expected_accuracy = 4/5  # 4 correct out of 5
        assert metrics["accuracy"] == expected_accuracy
        
        # Check that AUC is reasonable
        assert 0 <= metrics["auc_roc"] <= 1
        assert 0 <= metrics["auc_pr"] <= 1
    
    def test_classification_metrics_multiclass(self):
        """Test multiclass classification metrics."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.3, 0.5, 0.2],
            [0.9, 0.05, 0.05],
            [0.1, 0.3, 0.6],
            [0.1, 0.2, 0.7]
        ])
        
        metrics = EvaluationMetrics.classification_metrics(y_true, y_pred, y_prob, average="macro")
        
        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "auc_roc_ovr" in metrics
        assert "auc_roc_ovo" in metrics
        
        # Check per-class metrics exist
        for i in range(3):
            assert f"precision_class_{i}" in metrics
            assert f"recall_class_{i}" in metrics
            assert f"f1_class_{i}" in metrics
    
    def test_link_prediction_metrics(self):
        """Test link prediction specific metrics."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_scores = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
        
        metrics = EvaluationMetrics.link_prediction_metrics(y_true, y_scores, threshold=0.5)
        
        assert "auc_roc" in metrics
        assert "auc_pr" in metrics
        assert "hits@1" in metrics
        assert "hits@3" in metrics
        assert "hits@10" in metrics
        assert "mrr" in metrics
        
        # Check hits@k values are reasonable
        assert 0 <= metrics["hits@1"] <= 1
        assert 0 <= metrics["hits@3"] <= 1
        assert 0 <= metrics["hits@10"] <= 1
        
        # Check MRR is reasonable
        assert 0 <= metrics["mrr"] <= 1
    
    def test_regression_metrics(self):
        """Test regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = EvaluationMetrics.regression_metrics(y_true, y_pred)
        
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        
        # Check that metrics are reasonable
        assert metrics["mse"] > 0
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert -1 <= metrics["r2"] <= 1
        
        # Check RMSE is sqrt of MSE
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-6
    
    def test_hits_at_k_metric(self):
        """Test hits@k metric calculation."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        
        # Manually calculate hits@1
        # Sorted indices by score: [1, 3, 2, 4, 0]
        # Top-1 prediction: index 1, which has y_true=1, so hits@1 = 1.0
        hits_1 = EvaluationMetrics._hits_at_k(y_true, y_scores, 1)
        assert hits_1 == 1.0
        
        # Top-3 predictions: indices [1, 3, 2], which have y_true=[1, 1, 0]
        # At least one positive, so hits@3 = 1.0
        hits_3 = EvaluationMetrics._hits_at_k(y_true, y_scores, 3)
        assert hits_3 == 1.0
    
    def test_mean_reciprocal_rank(self):
        """Test mean reciprocal rank calculation."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        
        # Sorted indices by score: [1, 3, 2, 4, 0]
        # First positive example is at rank 1, so MRR = 1/1 = 1.0
        mrr = EvaluationMetrics._mean_reciprocal_rank(y_true, y_scores)
        assert mrr == 1.0
        
        # Test case with no positive examples
        y_true_no_pos = np.array([0, 0, 0, 0, 0])
        mrr_no_pos = EvaluationMetrics._mean_reciprocal_rank(y_true_no_pos, y_scores)
        assert mrr_no_pos == 0.0
    
    def test_metrics_with_edge_cases(self):
        """Test metrics with edge cases."""
        # All same predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1])
        
        metrics = EvaluationMetrics.classification_metrics(y_true, y_pred, average="binary")
        assert "accuracy" in metrics
        assert "precision_binary" in metrics
        
        # Empty arrays
        y_true_empty = np.array([])
        y_pred_empty = np.array([])
        
        # Should handle gracefully (might raise warnings)
        try:
            metrics_empty = EvaluationMetrics.classification_metrics(y_true_empty, y_pred_empty)
            assert isinstance(metrics_empty, dict)
        except ValueError:
            # Some metrics might not be computable with empty arrays
            pass


class TestBenchmarkEvaluator:
    """Test benchmark evaluator functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for evaluation."""
        model = Mock()
        model.eval.return_value = None
        model.zero_shot_inference.return_value = torch.sigmoid(torch.randn(5))
        return model
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader for evaluation."""
        batch = {
            "edge_index": torch.randint(0, 5, (2, 10)),
            "node_features": torch.randn(5, 64),
            "node_texts": ["text1", "text2", "text3", "text4", "text5"],
            "labels": torch.randint(0, 2, (5,)).float(),
        }
        return [batch]  # Single batch
    
    @pytest.fixture
    def evaluator(self, mock_model):
        """Create benchmark evaluator."""
        device = torch.device("cpu")
        return BenchmarkEvaluator(mock_model, device)
    
    def test_evaluator_initialization(self, mock_model):
        """Test evaluator initialization."""
        device = torch.device("cpu")
        evaluator = BenchmarkEvaluator(mock_model, device)
        
        assert evaluator.model == mock_model
        assert evaluator.device == device
        assert isinstance(evaluator.results, dict)
        assert evaluator.save_dir.exists()
    
    def test_evaluate_dataset_link_prediction(self, evaluator, mock_dataloader):
        """Test dataset evaluation for link prediction."""
        with patch.object(evaluator, '_forward_link_prediction') as mock_forward:
            mock_forward.return_value = torch.sigmoid(torch.randn(5))
            
            result = evaluator.evaluate_dataset(
                dataloader=mock_dataloader,
                dataset_name="test_dataset",
                task_type="link_prediction"
            )
            
            assert isinstance(result, dict)
            assert "dataset" in result
            assert "task_type" in result
            assert "metrics" in result
            assert "num_samples" in result
            
            assert result["dataset"] == "test_dataset"
            assert result["task_type"] == "link_prediction"
            
            metrics = result["metrics"]
            assert "auc_roc" in metrics
            assert "avg_inference_time" in metrics
            assert "throughput" in metrics
    
    def test_evaluate_dataset_node_classification(self, evaluator, mock_dataloader):
        """Test dataset evaluation for node classification."""
        with patch.object(evaluator, '_forward_node_classification') as mock_forward:
            mock_forward.return_value = torch.randn(5, 3)  # 3 classes
            
            result = evaluator.evaluate_dataset(
                dataloader=mock_dataloader,
                dataset_name="test_dataset",
                task_type="node_classification"
            )
            
            assert result["task_type"] == "node_classification"
            metrics = result["metrics"]
            assert "accuracy" in metrics
            assert "precision_macro" in metrics
    
    def test_evaluate_dataset_with_predictions(self, evaluator, mock_dataloader):
        """Test dataset evaluation with prediction return."""
        with patch.object(evaluator, '_forward_link_prediction') as mock_forward:
            mock_forward.return_value = torch.sigmoid(torch.randn(5))
            
            result = evaluator.evaluate_dataset(
                dataloader=mock_dataloader,
                dataset_name="test_dataset",
                task_type="link_prediction",
                return_predictions=True
            )
            
            assert "predictions" in result
            assert "targets" in result
            assert "probabilities" in result
            
            assert isinstance(result["predictions"], np.ndarray)
            assert isinstance(result["targets"], np.ndarray)
            assert isinstance(result["probabilities"], np.ndarray)
    
    def test_evaluate_zero_shot_transfer(self, evaluator):
        """Test zero-shot transfer evaluation."""
        # Mock dataloaders
        source_dataloaders = [Mock(), Mock()]
        target_dataloaders = [Mock()]
        
        for mock_loader in source_dataloaders + target_dataloaders:
            mock_loader.__iter__.return_value = iter([{
                "edge_index": torch.randint(0, 5, (2, 10)),
                "node_features": torch.randn(5, 64),
                "node_texts": ["text"] * 5,
                "labels": torch.randint(0, 2, (5,)).float(),
            }])
        
        source_datasets = [(loader, f"source_{i}") for i, loader in enumerate(source_dataloaders)]
        target_datasets = [(loader, f"target_{i}") for i, loader in enumerate(target_dataloaders)]
        
        with patch.object(evaluator, 'evaluate_dataset') as mock_eval:
            mock_eval.return_value = {
                "metrics": {"accuracy": 0.8, "f1_macro": 0.75}
            }
            
            result = evaluator.evaluate_zero_shot_transfer(
                source_datasets=source_datasets,
                target_datasets=target_datasets,
                task_type="link_prediction"
            )
            
            assert "source_performance" in result
            assert "target_performance" in result
            assert "transfer_metrics" in result
            
            assert len(result["source_performance"]) == 2
            assert len(result["target_performance"]) == 1
            
            transfer_metrics = result["transfer_metrics"]
            assert "source_average" in transfer_metrics
            assert "target_average" in transfer_metrics
            assert "transfer_gap" in transfer_metrics
    
    def test_compare_with_baselines(self, evaluator, mock_dataloader):
        """Test comparison with baseline methods."""
        # First evaluate with the model
        with patch.object(evaluator, '_forward_link_prediction') as mock_forward:
            mock_forward.return_value = torch.sigmoid(torch.randn(5))
            
            evaluator.evaluate_dataset(
                dataloader=mock_dataloader,
                dataset_name="test_dataset",
                task_type="link_prediction"
            )
        
        # Define baseline results
        baseline_results = {
            "GCN": {"f1_macro": 0.70, "accuracy": 0.75},
            "GAT": {"f1_macro": 0.72, "accuracy": 0.77},
            "GraphSAGE": {"f1_macro": 0.68, "accuracy": 0.73},
        }
        
        comparison = evaluator.compare_with_baselines(
            baseline_results=baseline_results,
            dataset_name="test_dataset",
            primary_metric="f1_macro"
        )
        
        assert "hypergnn" in comparison
        assert "baselines" in comparison
        assert "improvements" in comparison
        
        assert len(comparison["baselines"]) == 3
        assert len(comparison["improvements"]) == 3
        
        # Check that improvements are calculated
        for baseline_name in ["GCN", "GAT", "GraphSAGE"]:
            assert baseline_name in comparison["baselines"]
            assert baseline_name in comparison["improvements"]
    
    def test_generate_report(self, evaluator, mock_dataloader):
        """Test report generation."""
        # Add some evaluation results
        with patch.object(evaluator, '_forward_link_prediction') as mock_forward:
            mock_forward.return_value = torch.sigmoid(torch.randn(5))
            
            evaluator.evaluate_dataset(
                dataloader=mock_dataloader,
                dataset_name="test_dataset",
                task_type="link_prediction"
            )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluator.save_dir = Path(temp_dir)
            
            report = evaluator.generate_report(include_plots=False, save_format="json")
            
            assert "model_info" in report
            assert "evaluation_results" in report
            assert "summary" in report
            
            # Check that report file was saved
            report_file = Path(temp_dir) / "evaluation_report.json"
            assert report_file.exists()
            
            # Load and verify JSON structure
            with open(report_file, 'r') as f:
                saved_report = json.load(f)
            
            assert "model_info" in saved_report
            assert "evaluation_results" in saved_report
    
    def test_batch_to_device(self, evaluator):
        """Test moving batch to device."""
        batch = {
            "tensor_data": torch.randn(5, 10),
            "list_data": ["text1", "text2"],
            "scalar_data": 42
        }
        
        device_batch = evaluator._batch_to_device(batch)
        
        assert "tensor_data" in device_batch
        assert "list_data" in device_batch
        assert "scalar_data" in device_batch
        
        # Tensor should be moved to device
        assert device_batch["tensor_data"].device == evaluator.device
        
        # Non-tensors should remain unchanged
        assert device_batch["list_data"] == ["text1", "text2"]
        assert device_batch["scalar_data"] == 42
    
    def test_average_metrics(self, evaluator):
        """Test metrics averaging."""
        metrics_list = [
            {"accuracy": 0.8, "f1": 0.75, "precision": 0.7},
            {"accuracy": 0.85, "f1": 0.8, "precision": 0.75},
            {"accuracy": 0.82, "f1": 0.77, "precision": 0.72},
        ]
        
        averaged = evaluator._average_metrics(metrics_list)
        
        assert "accuracy" in averaged
        assert "f1" in averaged
        assert "precision" in averaged
        
        # Check averages
        expected_accuracy = (0.8 + 0.85 + 0.82) / 3
        assert abs(averaged["accuracy"] - expected_accuracy) < 1e-6
        
        expected_f1 = (0.75 + 0.8 + 0.77) / 3
        assert abs(averaged["f1"] - expected_f1) < 1e-6
    
    def test_compute_summary_statistics(self, evaluator, mock_dataloader):
        """Test summary statistics computation."""
        # Add multiple evaluation results
        with patch.object(evaluator, '_forward_link_prediction') as mock_forward:
            mock_forward.return_value = torch.sigmoid(torch.randn(5))
            
            for i in range(3):
                evaluator.evaluate_dataset(
                    dataloader=mock_dataloader,
                    dataset_name=f"dataset_{i}",
                    task_type="link_prediction"
                )
        
        summary = evaluator._compute_summary_statistics()
        
        assert isinstance(summary, dict)
        
        # Should have avg, std, min, max for each metric
        for key in summary.keys():
            assert any(prefix in key for prefix in ["avg_", "std_", "min_", "max_"])


class TestEvaluationIntegration:
    """Integration tests for evaluation components."""
    
    def test_end_to_end_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Create synthetic data
        kg = create_synthetic_kg(num_nodes=20, num_edges=30, random_seed=42)
        dataset = LinkPredictionDataset(kg, mode="test")
        dataloader = create_dataloader(dataset, batch_size=4)
        
        # Create mock model
        model = Mock()
        model.eval.return_value = None
        
        def mock_zero_shot_inference(edge_index, node_features, node_texts, **kwargs):
            return torch.sigmoid(torch.randn(edge_index.shape[1]))
        
        model.zero_shot_inference = mock_zero_shot_inference
        
        # Create evaluator
        device = torch.device("cpu")
        evaluator = BenchmarkEvaluator(model, device)
        
        # Evaluate
        result = evaluator.evaluate_dataset(
            dataloader=dataloader,
            dataset_name="synthetic_dataset",
            task_type="link_prediction"
        )
        
        assert isinstance(result, dict)
        assert "metrics" in result
        assert "dataset" in result
        assert result["dataset"] == "synthetic_dataset"
    
    def test_multi_dataset_evaluation(self):
        """Test evaluation across multiple datasets."""
        # Create multiple datasets
        datasets = []
        dataloaders = []
        
        for i in range(3):
            kg = create_synthetic_kg(num_nodes=15, num_edges=20, random_seed=i)
            dataset = LinkPredictionDataset(kg, mode="test")
            dataloader = create_dataloader(dataset, batch_size=4)
            
            datasets.append(dataset)
            dataloaders.append(dataloader)
        
        # Mock model
        model = Mock()
        model.eval.return_value = None
        model.zero_shot_inference.return_value = torch.sigmoid(torch.randn(5))
        
        # Create evaluator
        device = torch.device("cpu")
        evaluator = BenchmarkEvaluator(model, device)
        
        # Evaluate all datasets
        results = []
        for i, dataloader in enumerate(dataloaders):
            with patch.object(evaluator, '_forward_link_prediction') as mock_forward:
                mock_forward.return_value = torch.sigmoid(torch.randn(4))  # batch_size=4
                
                result = evaluator.evaluate_dataset(
                    dataloader=dataloader,
                    dataset_name=f"dataset_{i}",
                    task_type="link_prediction"
                )
                results.append(result)
        
        assert len(results) == 3
        assert all("metrics" in result for result in results)
        assert all("dataset" in result for result in results)


@pytest.mark.performance
class TestEvaluationPerformance:
    """Performance tests for evaluation components."""
    
    def test_metrics_computation_performance(self, benchmark):
        """Benchmark metrics computation performance."""
        # Large arrays for performance testing
        y_true = np.random.randint(0, 2, 10000)
        y_pred = np.random.randint(0, 2, 10000)
        y_prob = np.random.rand(10000)
        
        def compute_metrics():
            return EvaluationMetrics.classification_metrics(y_true, y_pred, y_prob)
        
        metrics = benchmark(compute_metrics)
        assert isinstance(metrics, dict)
    
    def test_large_dataset_evaluation_performance(self, benchmark):
        """Benchmark evaluation on large dataset."""
        # Create large synthetic dataset
        kg = create_synthetic_kg(num_nodes=500, num_edges=1000, random_seed=42)
        dataset = LinkPredictionDataset(kg, mode="test")
        dataloader = create_dataloader(dataset, batch_size=32)
        
        # Mock model for performance testing
        model = Mock()
        model.eval.return_value = None
        
        def mock_inference(*args, **kwargs):
            return torch.sigmoid(torch.randn(32))
        
        model.zero_shot_inference = mock_inference
        
        device = torch.device("cpu")
        evaluator = BenchmarkEvaluator(model, device)
        
        def evaluate_large_dataset():
            with patch.object(evaluator, '_forward_link_prediction') as mock_forward:
                mock_forward.return_value = torch.sigmoid(torch.randn(32))
                
                return evaluator.evaluate_dataset(
                    dataloader=dataloader,
                    dataset_name="large_dataset",
                    task_type="link_prediction"
                )
        
        result = benchmark(evaluate_large_dataset)
        assert isinstance(result, dict)


@pytest.mark.memory_intensive
class TestEvaluationMemory:
    """Memory usage tests for evaluation components."""
    
    def test_large_evaluation_memory_usage(self):
        """Test memory usage during large evaluations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large dataset
        kg = create_synthetic_kg(num_nodes=1000, num_edges=2000, random_seed=42)
        dataset = LinkPredictionDataset(kg, mode="test")
        dataloader = create_dataloader(dataset, batch_size=64)
        
        # Mock model
        model = Mock()
        model.eval.return_value = None
        model.zero_shot_inference.return_value = torch.sigmoid(torch.randn(64))
        
        device = torch.device("cpu")
        evaluator = BenchmarkEvaluator(model, device)
        
        # Evaluate
        with patch.object(evaluator, '_forward_link_prediction') as mock_forward:
            mock_forward.return_value = torch.sigmoid(torch.randn(64))
            
            result = evaluator.evaluate_dataset(
                dataloader=dataloader,
                dataset_name="memory_test",
                task_type="link_prediction",
                return_predictions=True
            )
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable
        assert memory_increase < 200  # Less than 200MB
        assert isinstance(result, dict)


class TestEvaluationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataloader_evaluation(self):
        """Test evaluation with empty dataloader."""
        model = Mock()
        model.eval.return_value = None
        
        device = torch.device("cpu")
        evaluator = BenchmarkEvaluator(model, device)
        
        empty_dataloader = []
        
        result = evaluator.evaluate_dataset(
            dataloader=empty_dataloader,
            dataset_name="empty_dataset",
            task_type="link_prediction"
        )
        
        assert isinstance(result, dict)
        assert result["num_samples"] == 0
    
    def test_invalid_task_type_evaluation(self):
        """Test evaluation with invalid task type."""
        model = Mock()
        device = torch.device("cpu")
        evaluator = BenchmarkEvaluator(model, device)
        
        batch = {
            "edge_index": torch.randint(0, 5, (2, 10)),
            "node_features": torch.randn(5, 64),
            "node_texts": ["text"] * 5,
            "labels": torch.randint(0, 2, (5,)).float(),
        }
        dataloader = [batch]
        
        with pytest.raises(ValueError):
            evaluator.evaluate_dataset(
                dataloader=dataloader,
                dataset_name="test_dataset",
                task_type="invalid_task_type"
            )
    
    def test_missing_dataset_comparison(self):
        """Test comparison when dataset not evaluated."""
        model = Mock()
        device = torch.device("cpu")
        evaluator = BenchmarkEvaluator(model, device)
        
        baseline_results = {"GCN": {"f1_macro": 0.7}}
        
        with pytest.raises(ValueError):
            evaluator.compare_with_baselines(
                baseline_results=baseline_results,
                dataset_name="non_existent_dataset",
                primary_metric="f1_macro"
            )
    
    def test_nan_predictions_handling(self):
        """Test handling of NaN predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, np.nan, 1, 0])
        y_prob = np.array([0.1, np.nan, 0.9, 0.2])
        
        # Should handle NaN gracefully
        try:
            metrics = EvaluationMetrics.classification_metrics(y_true, y_pred, y_prob)
            assert isinstance(metrics, dict)
        except (ValueError, RuntimeError):
            # Some metrics might not be computable with NaN values
            pass
    
    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1])  # Different length
        
        with pytest.raises((ValueError, IndexError)):
            EvaluationMetrics.classification_metrics(y_true, y_pred)