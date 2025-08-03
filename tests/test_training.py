"""Comprehensive tests for training utilities and components."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import wandb

from graph_hypernetwork_forge.utils import (
    TrainingConfig,
    EarlyStopping,
    MetricsTracker,
    HyperGNNTrainer,
    get_criterion,
    compute_zero_shot_metrics,
)
from graph_hypernetwork_forge.models import HyperGNN
from graph_hypernetwork_forge.data import create_synthetic_kg, LinkPredictionDataset, create_dataloader


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.patience == 10
        assert config.scheduler_type == "cosine"
        assert config.checkpoint_dir == "./checkpoints"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            learning_rate=5e-4,
            batch_size=64,
            num_epochs=50,
            patience=15,
            scheduler_type="step",
            use_wandb=True,
        )
        
        assert config.learning_rate == 5e-4
        assert config.batch_size == 64
        assert config.num_epochs == 50
        assert config.patience == 15
        assert config.scheduler_type == "step"
        assert config.use_wandb is True


class TestEarlyStopping:
    """Test early stopping functionality."""
    
    def test_early_stopping_initialization(self):
        """Test early stopping initialization."""
        early_stopping = EarlyStopping(patience=5, min_delta=1e-3, mode="min")
        
        assert early_stopping.patience == 5
        assert early_stopping.min_delta == 1e-3
        assert early_stopping.mode == "min"
        assert early_stopping.counter == 0
        assert early_stopping.best_score is None
        assert early_stopping.early_stop is False
    
    def test_early_stopping_min_mode(self):
        """Test early stopping in minimization mode."""
        early_stopping = EarlyStopping(patience=3, min_delta=1e-3, mode="min")
        
        # First score should be accepted
        assert early_stopping(1.0) is False
        assert early_stopping.best_score == 1.0
        assert early_stopping.counter == 0
        
        # Better score should be accepted
        assert early_stopping(0.8) is False
        assert early_stopping.best_score == 0.8
        assert early_stopping.counter == 0
        
        # Worse scores should increment counter
        assert early_stopping(0.9) is False
        assert early_stopping.counter == 1
        
        assert early_stopping(1.0) is False
        assert early_stopping.counter == 2
        
        assert early_stopping(1.1) is False
        assert early_stopping.counter == 3
        
        # Should trigger early stopping
        assert early_stopping(1.2) is True
        assert early_stopping.early_stop is True
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in maximization mode."""
        early_stopping = EarlyStopping(patience=2, min_delta=1e-3, mode="max")
        
        # First score
        assert early_stopping(0.5) is False
        assert early_stopping.best_score == 0.5
        
        # Better score
        assert early_stopping(0.7) is False
        assert early_stopping.best_score == 0.7
        assert early_stopping.counter == 0
        
        # Worse scores
        assert early_stopping(0.6) is False
        assert early_stopping.counter == 1
        
        assert early_stopping(0.5) is False
        assert early_stopping.counter == 2
        
        # Should trigger early stopping
        assert early_stopping(0.4) is True
    
    def test_early_stopping_no_improvement_threshold(self):
        """Test early stopping with improvement threshold."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.1, mode="min")
        
        # Set initial score
        assert early_stopping(1.0) is False
        
        # Small improvement (less than min_delta) should not reset counter
        assert early_stopping(0.95) is False  # Improvement of 0.05 < 0.1
        assert early_stopping.counter == 1
        
        # Significant improvement should reset counter
        assert early_stopping(0.8) is False  # Improvement of 0.2 > 0.1
        assert early_stopping.counter == 0
        assert early_stopping.best_score == 0.8


class TestMetricsTracker:
    """Test metrics tracking functionality."""
    
    def test_metrics_tracker_initialization(self):
        """Test metrics tracker initialization."""
        tracker = MetricsTracker()
        
        assert tracker.losses == []
        assert tracker.predictions == []
        assert tracker.targets == []
        assert tracker.times == []
    
    def test_metrics_update(self):
        """Test updating metrics."""
        tracker = MetricsTracker()
        
        loss = 0.5
        predictions = torch.tensor([0.8, 0.3, 0.9])
        targets = torch.tensor([1.0, 0.0, 1.0])
        batch_time = 0.1
        
        tracker.update(loss, predictions, targets, batch_time)
        
        assert len(tracker.losses) == 1
        assert tracker.losses[0] == 0.5
        assert len(tracker.predictions) == 3
        assert len(tracker.targets) == 3
        assert len(tracker.times) == 1
    
    def test_classification_metrics_computation(self):
        """Test classification metrics computation."""
        tracker = MetricsTracker()
        
        # Add binary classification data
        predictions = torch.tensor([0.8, 0.3, 0.9, 0.1])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        
        tracker.update(0.3, predictions, targets)
        
        metrics = tracker.compute_metrics(task_type="classification")
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["loss"] == 0.3
    
    def test_regression_metrics_computation(self):
        """Test regression metrics computation."""
        tracker = MetricsTracker()
        
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 1.9, 3.2])
        
        tracker.update(0.1, predictions, targets)
        
        metrics = tracker.compute_metrics(task_type="regression")
        
        assert "loss" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["loss"] == 0.1
    
    def test_metrics_reset(self):
        """Test metrics tracker reset."""
        tracker = MetricsTracker()
        
        # Add some data
        tracker.update(0.5, torch.tensor([0.8]), torch.tensor([1.0]))
        assert len(tracker.losses) == 1
        
        # Reset
        tracker.reset()
        assert len(tracker.losses) == 0
        assert len(tracker.predictions) == 0
        assert len(tracker.targets) == 0
        assert len(tracker.times) == 0


class TestHyperGNNTrainer:
    """Test HyperGNN trainer functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock HyperGNN model."""
        model = Mock(spec=HyperGNN)
        model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        model.to.return_value = model
        model.train.return_value = None
        model.eval.return_value = None
        model.state_dict.return_value = {"param": torch.randn(10, 10)}
        model.load_state_dict.return_value = None
        return model
    
    @pytest.fixture
    def training_config(self):
        """Create training configuration."""
        return TrainingConfig(
            learning_rate=1e-3,
            num_epochs=5,
            patience=3,
            log_interval=1,
            eval_interval=1,
            checkpoint_dir="./test_checkpoints",
            use_wandb=False,
        )
    
    @pytest.fixture
    def sample_dataloader(self):
        """Create sample dataloader for testing."""
        kg = create_synthetic_kg(num_nodes=10, num_edges=15, random_seed=42)
        dataset = LinkPredictionDataset(kg, mode="train")
        return create_dataloader(dataset, batch_size=4, shuffle=False)
    
    def test_trainer_initialization(self, mock_model, training_config):
        """Test trainer initialization."""
        trainer = HyperGNNTrainer(mock_model, training_config)
        
        assert trainer.model == mock_model
        assert trainer.config == training_config
        assert trainer.current_epoch == 0
        assert trainer.best_val_loss == float('inf')
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'early_stopping')
    
    def test_scheduler_creation(self, mock_model, training_config):
        """Test learning rate scheduler creation."""
        # Test cosine scheduler
        training_config.scheduler_type = "cosine"
        trainer = HyperGNNTrainer(mock_model, training_config)
        assert trainer.scheduler is not None
        
        # Test step scheduler
        training_config.scheduler_type = "step"
        trainer = HyperGNNTrainer(mock_model, training_config)
        assert trainer.scheduler is not None
        
        # Test plateau scheduler
        training_config.scheduler_type = "plateau"
        trainer = HyperGNNTrainer(mock_model, training_config)
        assert trainer.scheduler is not None
        
        # Test no scheduler
        training_config.scheduler_type = None
        trainer = HyperGNNTrainer(mock_model, training_config)
        assert trainer.scheduler is None
    
    @patch('torch.save')
    def test_save_checkpoint(self, mock_save, mock_model, training_config):
        """Test checkpoint saving."""
        trainer = HyperGNNTrainer(mock_model, training_config)
        trainer.current_epoch = 10
        trainer.best_val_loss = 0.5
        
        trainer.save_checkpoint("test")
        
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        checkpoint_data = args[0]
        
        assert "epoch" in checkpoint_data
        assert "model_state_dict" in checkpoint_data
        assert "optimizer_state_dict" in checkpoint_data
        assert "best_val_loss" in checkpoint_data
        assert checkpoint_data["epoch"] == 10
        assert checkpoint_data["best_val_loss"] == 0.5
    
    @patch('torch.load')
    def test_load_checkpoint(self, mock_load, mock_model, training_config):
        """Test checkpoint loading."""
        # Mock checkpoint data
        checkpoint_data = {
            "epoch": 10,
            "model_state_dict": {"param": torch.randn(10, 10)},
            "optimizer_state_dict": {"param": torch.randn(10)},
            "scheduler_state_dict": {"param": torch.randn(5)},
            "best_val_loss": 0.5,
            "config": training_config,
        }
        mock_load.return_value = checkpoint_data
        
        trainer = HyperGNNTrainer(mock_model, training_config)
        trainer.load_checkpoint("test")
        
        assert trainer.current_epoch == 10
        assert trainer.best_val_loss == 0.5
        mock_model.load_state_dict.assert_called_once()
    
    def test_batch_to_device(self, mock_model, training_config):
        """Test moving batch to device."""
        trainer = HyperGNNTrainer(mock_model, training_config)
        
        batch = {
            "tensor_data": torch.randn(5, 10),
            "list_data": ["text1", "text2"],
            "scalar_data": 42
        }
        
        device_batch = trainer._batch_to_device(batch)
        
        assert "tensor_data" in device_batch
        assert "list_data" in device_batch
        assert "scalar_data" in device_batch
        assert device_batch["list_data"] == ["text1", "text2"]
        assert device_batch["scalar_data"] == 42
    
    @patch('wandb.init')
    @patch('wandb.watch')
    def test_wandb_integration(self, mock_watch, mock_init, mock_model):
        """Test Weights & Biases integration."""
        config = TrainingConfig(use_wandb=True, wandb_project="test_project")
        trainer = HyperGNNTrainer(mock_model, config)
        
        mock_init.assert_called_once()
        mock_watch.assert_called_once_with(mock_model)


class TestCriterionFactory:
    """Test loss function factory."""
    
    def test_link_prediction_criterion(self):
        """Test link prediction criterion."""
        criterion = get_criterion("link_prediction")
        assert isinstance(criterion, nn.BCEWithLogitsLoss)
    
    def test_node_classification_criterion(self):
        """Test node classification criterion."""
        criterion = get_criterion("node_classification")
        assert isinstance(criterion, nn.CrossEntropyLoss)
    
    def test_graph_classification_criterion(self):
        """Test graph classification criterion."""
        criterion = get_criterion("graph_classification")
        assert isinstance(criterion, nn.CrossEntropyLoss)
    
    def test_regression_criterion(self):
        """Test regression criterion."""
        criterion = get_criterion("regression")
        assert isinstance(criterion, nn.MSELoss)
    
    def test_unknown_task_type(self):
        """Test unknown task type raises error."""
        with pytest.raises(ValueError):
            get_criterion("unknown_task")


class TestZeroShotMetrics:
    """Test zero-shot evaluation metrics."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for zero-shot testing."""
        model = Mock()
        model.eval.return_value = None
        model.zero_shot_inference.return_value = torch.sigmoid(torch.randn(5))
        return model
    
    @pytest.fixture
    def mock_dataloader(self):
        """Create mock dataloader."""
        batch = {
            "edge_index": torch.randint(0, 5, (2, 10)),
            "node_features": torch.randn(5, 64),
            "node_texts": ["text1", "text2", "text3", "text4", "text5"],
            "labels": torch.randint(0, 2, (5,)).float(),
        }
        
        dataloader = [batch, batch]  # Two batches
        return dataloader
    
    def test_compute_zero_shot_metrics(self, mock_model, mock_dataloader):
        """Test zero-shot metrics computation."""
        device = torch.device("cpu")
        
        with patch('graph_hypernetwork_forge.utils.training.get_criterion') as mock_get_criterion:
            mock_criterion = Mock()
            mock_criterion.return_value = torch.tensor(0.5)
            mock_get_criterion.return_value = mock_criterion
            
            metrics = compute_zero_shot_metrics(
                model=mock_model,
                target_datasets=[mock_dataloader],
                device=device,
                task_type="link_prediction"
            )
            
            assert isinstance(metrics, dict)
            assert "avg_loss" in metrics
            mock_model.zero_shot_inference.assert_called()


class TestTrainingIntegration:
    """Integration tests for training components."""
    
    @pytest.mark.slow
    def test_end_to_end_training_loop(self):
        """Test complete training loop (mocked)."""
        # Create synthetic data
        kg = create_synthetic_kg(num_nodes=20, num_edges=30, random_seed=42)
        train_dataset = LinkPredictionDataset(kg, mode="train")
        val_dataset = LinkPredictionDataset(kg, mode="val")
        
        train_loader = create_dataloader(train_dataset, batch_size=4)
        val_loader = create_dataloader(val_dataset, batch_size=4)
        
        # Create model (simplified for testing)
        try:
            model = HyperGNN(
                text_encoder="sentence-transformers/all-MiniLM-L6-v2",
                gnn_backbone="GCN",
                hidden_dim=64,
                num_layers=1,
                dropout=0.1,
            )
        except ImportError:
            pytest.skip("sentence-transformers not available")
        
        # Training configuration
        config = TrainingConfig(
            learning_rate=1e-2,
            num_epochs=2,  # Short for testing
            patience=5,
            log_interval=1,
            eval_interval=1,
            use_wandb=False,
        )
        
        # Create trainer
        trainer = HyperGNNTrainer(model, config)
        
        # Mock the forward methods to avoid complexity
        with patch.object(trainer, '_forward_link_prediction') as mock_forward:
            mock_forward.return_value = torch.randn(4)  # Batch size 4
            
            # Train for a few epochs
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                task_type="link_prediction"
            )
            
            assert isinstance(history, dict)
            assert "train_loss" in history
            assert "val_loss" in history
            assert len(history["train_loss"]) <= 2  # Max 2 epochs
    
    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Valid configuration
        config = TrainingConfig(learning_rate=1e-3, num_epochs=10)
        assert config.learning_rate == 1e-3
        assert config.num_epochs == 10
        
        # Test default values
        config = TrainingConfig()
        assert config.learning_rate > 0
        assert config.num_epochs > 0
        assert config.patience > 0
    
    @patch('torch.save')
    def test_checkpoint_directory_creation(self, mock_save):
        """Test that checkpoint directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "new_checkpoints"
            
            config = TrainingConfig(checkpoint_dir=str(checkpoint_dir))
            
            # Mock model
            model = Mock()
            model.parameters.return_value = [torch.randn(5, 5, requires_grad=True)]
            model.to.return_value = model
            model.state_dict.return_value = {}
            
            trainer = HyperGNNTrainer(model, config)
            
            # Directory should be created
            assert checkpoint_dir.exists()
            assert checkpoint_dir.is_dir()


@pytest.mark.performance
class TestTrainingPerformance:
    """Performance tests for training components."""
    
    def test_metrics_tracker_performance(self, benchmark):
        """Benchmark metrics tracker performance."""
        tracker = MetricsTracker()
        
        def update_metrics():
            for _ in range(100):
                predictions = torch.randn(32)
                targets = torch.randint(0, 2, (32,)).float()
                tracker.update(0.5, predictions, targets, 0.1)
        
        benchmark(update_metrics)
        
        # Should handle 100 updates efficiently
        assert len(tracker.losses) == 100
    
    def test_early_stopping_performance(self, benchmark):
        """Benchmark early stopping performance."""
        early_stopping = EarlyStopping(patience=10)
        
        def check_early_stopping():
            scores = np.random.randn(1000)
            for score in scores:
                if early_stopping(score):
                    break
        
        benchmark(check_early_stopping)


@pytest.mark.memory_intensive
class TestTrainingMemory:
    """Memory usage tests for training components."""
    
    def test_metrics_tracker_memory_usage(self):
        """Test metrics tracker memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        tracker = MetricsTracker()
        
        # Add large amount of data
        for _ in range(1000):
            predictions = torch.randn(1000)
            targets = torch.randint(0, 2, (1000,)).float()
            tracker.update(0.5, predictions, targets, 0.1)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Should not use excessive memory
        assert memory_increase < 100  # Less than 100MB
    
    def test_trainer_memory_cleanup(self):
        """Test trainer memory cleanup."""
        # Create multiple trainers to test memory cleanup
        models = []
        trainers = []
        
        config = TrainingConfig(num_epochs=1, use_wandb=False)
        
        for i in range(5):
            model = Mock()
            model.parameters.return_value = [torch.randn(100, 100, requires_grad=True)]
            model.to.return_value = model
            model.state_dict.return_value = {}
            
            trainer = HyperGNNTrainer(model, config)
            
            models.append(model)
            trainers.append(trainer)
        
        # Clear references
        del models
        del trainers
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Test passes if no memory errors occur


class TestTrainingEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataloader(self):
        """Test handling of empty dataloader."""
        model = Mock()
        model.parameters.return_value = [torch.randn(5, 5, requires_grad=True)]
        model.to.return_value = model
        model.train.return_value = None
        
        config = TrainingConfig(num_epochs=1, use_wandb=False)
        trainer = HyperGNNTrainer(model, config)
        
        # Empty dataloader
        empty_loader = []
        
        # Should handle gracefully
        metrics = trainer.train_epoch(empty_loader, nn.MSELoss())
        assert isinstance(metrics, dict)
    
    def test_invalid_task_type(self):
        """Test invalid task type in training."""
        model = Mock()
        model.parameters.return_value = [torch.randn(5, 5, requires_grad=True)]
        model.to.return_value = model
        
        config = TrainingConfig(use_wandb=False)
        trainer = HyperGNNTrainer(model, config)
        
        # Mock dataloader
        batch = {"data": torch.randn(5, 10)}
        dataloader = [batch]
        
        with pytest.raises(ValueError):
            trainer.train_epoch(dataloader, nn.MSELoss(), task_type="invalid_task")
    
    def test_scheduler_step_with_none_scheduler(self):
        """Test scheduler step when scheduler is None."""
        model = Mock()
        model.parameters.return_value = [torch.randn(5, 5, requires_grad=True)]
        model.to.return_value = model
        
        config = TrainingConfig(scheduler_type=None, use_wandb=False)
        trainer = HyperGNNTrainer(model, config)
        
        # Should not raise error when scheduler is None
        assert trainer.scheduler is None
        # This should not crash
        if trainer.scheduler:
            trainer.scheduler.step()
    
    def test_nan_loss_handling(self):
        """Test handling of NaN losses."""
        tracker = MetricsTracker()
        
        # Add NaN loss
        predictions = torch.tensor([float('nan'), 0.5, 0.8])
        targets = torch.tensor([1.0, 0.0, 1.0])
        
        # Should handle NaN gracefully
        tracker.update(float('nan'), predictions, targets)
        
        metrics = tracker.compute_metrics("classification")
        # Loss should be NaN, but other metrics might still be computable
        assert "loss" in metrics