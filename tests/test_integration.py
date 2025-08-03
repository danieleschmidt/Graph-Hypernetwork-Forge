"""Integration tests for the complete Graph Hypernetwork Forge system.

These tests verify that all components work together correctly in realistic scenarios.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path
import json
from unittest.mock import patch, Mock

from graph_hypernetwork_forge import HyperGNN, TextualKnowledgeGraph
from graph_hypernetwork_forge.data import (
    create_synthetic_kg,
    LinkPredictionDataset,
    NodeClassificationDataset,
    ZeroShotDataset,
    create_dataloader,
)
from graph_hypernetwork_forge.utils import (
    TrainingConfig,
    HyperGNNTrainer,
    BenchmarkEvaluator,
    get_criterion,
)
from graph_hypernetwork_forge.database import (
    DatabaseManager,
    DatabaseConfig,
    GraphRepository,
    ExperimentRepository,
)
from graph_hypernetwork_forge.cache import (
    CacheConfig,
    MultiLevelCache,
    EmbeddingCache,
)


class TestCompleteWorkflow:
    """Test complete workflow from data creation to evaluation."""
    
    @pytest.mark.slow
    def test_end_to_end_link_prediction_workflow(self):
        """Test complete link prediction workflow."""
        try:
            # Step 1: Create knowledge graphs
            source_kg = create_synthetic_kg(
                num_nodes=50,
                num_edges=80,
                relations=["related_to", "part_of"],
                random_seed=42
            )
            
            target_kg = create_synthetic_kg(
                num_nodes=40,
                num_edges=60,
                relations=["similar_to", "connected_to"],
                random_seed=123
            )
            
            # Step 2: Create datasets
            train_dataset = LinkPredictionDataset(source_kg, mode="train")
            val_dataset = LinkPredictionDataset(source_kg, mode="val")
            test_dataset = LinkPredictionDataset(target_kg, mode="test")
            
            # Step 3: Create data loaders
            train_loader = create_dataloader(train_dataset, batch_size=8, shuffle=True)
            val_loader = create_dataloader(val_dataset, batch_size=8, shuffle=False)
            test_loader = create_dataloader(test_dataset, batch_size=8, shuffle=False)
            
            # Step 4: Initialize model
            model = HyperGNN(
                text_encoder="sentence-transformers/all-MiniLM-L6-v2",
                gnn_backbone="GAT",
                hidden_dim=64,
                num_layers=2,
                num_heads=4,
                dropout=0.1,
            )
            
            # Step 5: Setup training
            config = TrainingConfig(
                learning_rate=1e-2,
                num_epochs=3,  # Short for testing
                patience=5,
                batch_size=8,
                log_interval=1,
                eval_interval=1,
                use_wandb=False,
            )
            
            trainer = HyperGNNTrainer(model, config)
            
            # Step 6: Mock training (to avoid long computation)
            with patch.object(trainer, '_forward_link_prediction') as mock_forward:
                mock_forward.return_value = torch.randn(8)  # Batch size 8
                
                # Train model
                history = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=get_criterion("link_prediction"),
                    task_type="link_prediction"
                )
                
                assert isinstance(history, dict)
                assert "train_loss" in history
                assert "val_loss" in history
            
            # Step 7: Evaluate on target domain (zero-shot)
            evaluator = BenchmarkEvaluator(model, torch.device("cpu"))
            
            with patch.object(evaluator, '_forward_link_prediction') as mock_eval:
                mock_eval.return_value = torch.sigmoid(torch.randn(8))
                
                results = evaluator.evaluate_dataset(
                    dataloader=test_loader,
                    dataset_name="target_domain",
                    task_type="link_prediction"
                )
                
                assert isinstance(results, dict)
                assert "metrics" in results
                assert "auc_roc" in results["metrics"]
            
            # Step 8: Generate evaluation report
            report = evaluator.generate_report(include_plots=False)
            assert "evaluation_results" in report
            assert "summary" in report
            
        except ImportError:
            pytest.skip("sentence-transformers not available")
    
    def test_multi_domain_zero_shot_evaluation(self):
        """Test zero-shot evaluation across multiple domains."""
        try:
            # Create multiple source domains
            source_graphs = [
                create_synthetic_kg(
                    num_nodes=30,
                    num_edges=45,
                    relations=["type1_rel", "type1_part"],
                    random_seed=i
                )
                for i in range(3)
            ]
            
            # Create multiple target domains
            target_graphs = [
                create_synthetic_kg(
                    num_nodes=25,
                    num_edges=35,
                    relations=["type2_rel", "type2_part"],
                    random_seed=i+100
                )
                for i in range(2)
            ]
            
            # Create zero-shot dataset
            zero_shot_dataset = ZeroShotDataset(
                source_graphs=source_graphs,
                target_graphs=target_graphs,
                task_type="link_prediction"
            )
            
            source_datasets = zero_shot_dataset.get_source_data()
            target_datasets = zero_shot_dataset.get_target_data()
            
            assert len(source_datasets) == 3
            assert len(target_datasets) == 2
            
            # Test that datasets are usable
            for dataset in source_datasets + target_datasets:
                assert len(dataset) > 0
                sample = dataset[0]
                assert "graph" in sample
                assert "label" in sample
                
        except ImportError:
            pytest.skip("sentence-transformers not available")
    
    def test_model_serialization_and_loading(self):
        """Test complete model save/load cycle."""
        try:
            # Create and configure model
            original_model = HyperGNN(
                text_encoder="sentence-transformers/all-MiniLM-L6-v2",
                gnn_backbone="GCN",
                hidden_dim=32,
                num_layers=2,
                dropout=0.1,
            )
            
            # Test inference with original model
            kg = create_synthetic_kg(num_nodes=5, num_edges=8, random_seed=42)
            edge_index = kg.edge_index
            node_features = kg.node_features[:, :32]  # Match hidden_dim
            node_texts = kg.node_texts
            
            original_output = original_model.zero_shot_inference(
                edge_index, node_features, node_texts
            )
            
            # Save model
            with tempfile.TemporaryDirectory() as temp_dir:
                save_path = Path(temp_dir) / "test_model"
                save_path.mkdir()
                
                original_model.save_pretrained(str(save_path))
                
                # Verify files exist
                assert (save_path / "model.pt").exists()
                
                # Load model
                loaded_model = HyperGNN.load_pretrained(str(save_path))
                
                # Test inference with loaded model
                loaded_output = loaded_model.zero_shot_inference(
                    edge_index, node_features, node_texts
                )
                
                # Outputs should be very close (allowing for small numerical differences)
                assert torch.allclose(original_output, loaded_output, atol=1e-5)
                
        except ImportError:
            pytest.skip("sentence-transformers not available")


class TestDatabaseIntegration:
    """Test database integration with the main workflow."""
    
    def test_graph_storage_and_retrieval(self):
        """Test storing and retrieving graphs from database."""
        # Create temporary database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_config = DatabaseConfig(
                db_type="sqlite",
                db_path=str(Path(temp_dir) / "test.db")
            )
            
            db_manager = DatabaseManager(db_config)
            db_manager.create_tables()
            
            graph_repo = GraphRepository(db_manager)
            
            # Create and save graph
            kg = create_synthetic_kg(num_nodes=20, num_edges=30, random_seed=42)
            graph_id = graph_repo.save_graph(kg, "test_graph_1")
            
            # Load graph
            loaded_kg = graph_repo.load_graph(graph_id)
            
            # Verify loaded graph matches original
            assert loaded_kg.name == kg.name
            assert len(loaded_kg.nodes) == len(kg.nodes)
            assert len(loaded_kg.edges) == len(kg.edges)
            
            # Test graph listing
            graph_list = graph_repo.list_graphs()
            assert len(graph_list) == 1
            assert graph_list[0]["id"] == graph_id
            
            db_manager.close()
    
    def test_experiment_tracking(self):
        """Test experiment tracking functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_config = DatabaseConfig(
                db_type="sqlite",
                db_path=str(Path(temp_dir) / "experiments.db")
            )
            
            db_manager = DatabaseManager(db_config)
            db_manager.create_tables()
            
            exp_repo = ExperimentRepository(db_manager)
            
            # Save experiment
            exp_config = {
                "learning_rate": 1e-3,
                "batch_size": 32,
                "num_epochs": 10,
                "model_type": "HyperGNN",
            }
            
            exp_id = exp_repo.save_experiment(
                experiment_id="exp_001",
                name="Test Experiment",
                config=exp_config,
                description="Integration test experiment"
            )
            
            # Update experiment with results
            metrics = {
                "final_loss": 0.25,
                "best_accuracy": 0.85,
                "training_time": 120.5
            }
            
            exp_repo.update_experiment_status(exp_id, "completed", metrics)
            
            # Retrieve experiment
            experiment = exp_repo.get_experiment(exp_id)
            
            assert experiment is not None
            assert experiment["name"] == "Test Experiment"
            assert experiment["status"] == "completed"
            assert experiment["config"]["learning_rate"] == 1e-3
            assert experiment["metrics"]["final_loss"] == 0.25
            
            # List experiments
            experiments = exp_repo.list_experiments()
            assert len(experiments) == 1
            assert experiments[0]["id"] == exp_id
            
            db_manager.close()


class TestCacheIntegration:
    """Test caching integration with the main workflow."""
    
    def test_embedding_cache_integration(self):
        """Test embedding caching during text encoding."""
        cache_config = CacheConfig(
            backend="memory",
            max_size=100,
            ttl=3600
        )
        
        cache_manager = MultiLevelCache(cache_config)
        embedding_cache = EmbeddingCache(cache_manager)
        
        # Mock text encoder for testing
        model_name = "test-encoder"
        texts = ["text1", "text2", "text3"]
        
        # First call - cache miss
        cached_embeddings, missing_texts = embedding_cache.get_batch_embeddings(texts, model_name)
        assert all(emb is None for emb in cached_embeddings)
        assert missing_texts == texts
        
        # Simulate encoding and caching
        fake_embeddings = [torch.randn(128) for _ in texts]
        embedding_cache.set_batch_embeddings(texts, model_name, fake_embeddings)
        
        # Second call - cache hit
        cached_embeddings, missing_texts = embedding_cache.get_batch_embeddings(texts, model_name)
        assert all(emb is not None for emb in cached_embeddings)
        assert len(missing_texts) == 0
        
        # Verify cached embeddings match
        for i, cached_emb in enumerate(cached_embeddings):
            assert torch.allclose(cached_emb, fake_embeddings[i])
    
    def test_weight_generation_caching(self):
        """Test caching of generated weights."""
        from graph_hypernetwork_forge.cache import cached, get_cache_manager
        
        # Initialize cache
        cache_config = CacheConfig(backend="memory", max_size=50)
        cache_manager = get_cache_manager(cache_config)
        
        # Mock weight generator function
        @cached(cache_manager=cache_manager, ttl=3600, key_prefix="weights_")
        def generate_weights_cached(text_embeddings):
            # Simulate expensive weight generation
            return {
                "layer_0_weight": torch.randn(64, 64),
                "layer_0_bias": torch.randn(64),
            }
        
        # First call - computation
        text_emb = torch.randn(10, 384)
        weights1 = generate_weights_cached(text_emb)
        
        # Second call with same input - should be cached
        weights2 = generate_weights_cached(text_emb)
        
        # Verify same results (from cache)
        assert torch.allclose(weights1["layer_0_weight"], weights2["layer_0_weight"])
        assert torch.allclose(weights1["layer_0_bias"], weights2["layer_0_bias"])


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    def test_training_interruption_recovery(self):
        """Test recovery from training interruption."""
        try:
            # Setup training
            kg = create_synthetic_kg(num_nodes=30, num_edges=45, random_seed=42)
            train_dataset = LinkPredictionDataset(kg, mode="train")
            train_loader = create_dataloader(train_dataset, batch_size=4)
            
            model = HyperGNN(
                text_encoder="sentence-transformers/all-MiniLM-L6-v2",
                gnn_backbone="GCN",
                hidden_dim=32,
                num_layers=1,
            )
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config = TrainingConfig(
                    num_epochs=5,
                    checkpoint_dir=temp_dir,
                    save_best=True,
                    use_wandb=False,
                )
                
                trainer = HyperGNNTrainer(model, config)
                
                # Simulate training interruption by saving checkpoint
                trainer.current_epoch = 3
                trainer.best_val_loss = 0.5
                trainer.save_checkpoint("best")
                
                # Create new trainer and load checkpoint
                new_trainer = HyperGNNTrainer(model, config)
                new_trainer.load_checkpoint("best")
                
                # Verify state restoration
                assert new_trainer.current_epoch == 3
                assert new_trainer.best_val_loss == 0.5
                
        except ImportError:
            pytest.skip("sentence-transformers not available")
    
    def test_invalid_data_handling(self):
        """Test handling of invalid or corrupted data."""
        # Test with empty knowledge graph
        empty_kg = TextualKnowledgeGraph(nodes=[], edges=[], name="Empty")
        
        # Should handle gracefully
        assert len(empty_kg.nodes) == 0
        assert len(empty_kg.edges) == 0
        assert empty_kg.edge_index.shape == (2, 0)
        
        # Test with single node (no edges)
        from graph_hypernetwork_forge.data import NodeInfo
        single_node_kg = TextualKnowledgeGraph(
            nodes=[NodeInfo(id="n1", text="Single node")],
            edges=[],
            name="Single"
        )
        
        assert len(single_node_kg.nodes) == 1
        assert len(single_node_kg.edges) == 0
        assert single_node_kg.node_features.shape[0] == 1
    
    def test_device_mismatch_handling(self):
        """Test handling of device mismatches."""
        try:
            model = HyperGNN(
                text_encoder="sentence-transformers/all-MiniLM-L6-v2",
                gnn_backbone="GCN",
                hidden_dim=32,
                num_layers=1,
            )
            
            # Test with different device tensors
            device = torch.device("cpu")
            model = model.to(device)
            
            edge_index = torch.randint(0, 3, (2, 5))
            node_features = torch.randn(3, 32)
            node_texts = ["Node 1", "Node 2", "Node 3"]
            
            # Should work with proper device placement
            output = model.zero_shot_inference(
                edge_index.to(device),
                node_features.to(device),
                node_texts
            )
            
            assert isinstance(output, torch.Tensor)
            assert output.device == device
            
        except ImportError:
            pytest.skip("sentence-transformers not available")


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""
    
    @pytest.mark.performance
    def test_end_to_end_inference_performance(self, benchmark):
        """Benchmark end-to-end inference performance."""
        try:
            # Setup model and data
            model = HyperGNN(
                text_encoder="sentence-transformers/all-MiniLM-L6-v2",
                gnn_backbone="GCN",
                hidden_dim=64,
                num_layers=2,
            )
            
            kg = create_synthetic_kg(num_nodes=100, num_edges=200, random_seed=42)
            edge_index = kg.edge_index
            node_features = kg.node_features[:, :64]
            node_texts = kg.node_texts
            
            def inference():
                model.eval()
                with torch.no_grad():
                    return model.zero_shot_inference(edge_index, node_features, node_texts)
            
            result = benchmark(inference)
            assert isinstance(result, torch.Tensor)
            
        except ImportError:
            pytest.skip("sentence-transformers not available")
    
    @pytest.mark.performance
    def test_batch_processing_performance(self, benchmark):
        """Benchmark batch processing performance."""
        kg = create_synthetic_kg(num_nodes=200, num_edges=400, random_seed=42)
        dataset = LinkPredictionDataset(kg, mode="train")
        
        def process_batch():
            dataloader = create_dataloader(dataset, batch_size=32, shuffle=False)
            batch = next(iter(dataloader))
            return batch
        
        result = benchmark(process_batch)
        assert isinstance(result, dict)
    
    @pytest.mark.slow
    def test_large_scale_integration(self):
        """Test integration with larger scale data."""
        # Create larger knowledge graphs
        large_kg = create_synthetic_kg(
            num_nodes=1000,
            num_edges=2000,
            random_seed=42
        )
        
        # Test dataset creation
        dataset = LinkPredictionDataset(large_kg, mode="train")
        assert len(dataset) > 0
        
        # Test dataloader
        dataloader = create_dataloader(dataset, batch_size=64)
        batch = next(iter(dataloader))
        
        assert isinstance(batch, dict)
        assert "edge_index" in batch
        assert "node_texts" in batch
        
        # Test that processing doesn't exceed memory limits
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            if batch_count >= 10:  # Process a reasonable number of batches
                break
        
        assert batch_count > 0


class TestConfigurationIntegration:
    """Test integration of different configuration options."""
    
    def test_different_gnn_backends(self):
        """Test integration with different GNN backends."""
        kg = create_synthetic_kg(num_nodes=20, num_edges=30, random_seed=42)
        edge_index = kg.edge_index
        node_features = kg.node_features[:, :32]
        node_texts = kg.node_texts
        
        backends = ["GCN", "GAT", "GraphSAGE"]
        
        for backend in backends:
            try:
                model = HyperGNN(
                    text_encoder="sentence-transformers/all-MiniLM-L6-v2",
                    gnn_backbone=backend,
                    hidden_dim=32,
                    num_layers=1,
                )
                
                # Test inference
                output = model.zero_shot_inference(edge_index, node_features, node_texts)
                assert isinstance(output, torch.Tensor)
                assert output.shape[0] == len(node_texts)
                
            except ImportError:
                pytest.skip("sentence-transformers not available")
    
    def test_different_text_encoders(self):
        """Test integration with different text encoders."""
        kg = create_synthetic_kg(num_nodes=15, num_edges=25, random_seed=42)
        
        # Test with different encoder models
        encoders = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ]
        
        for encoder in encoders:
            try:
                model = HyperGNN(
                    text_encoder=encoder,
                    gnn_backbone="GCN",
                    hidden_dim=32,
                    num_layers=1,
                )
                
                # Test weight generation
                weights = model.generate_weights(kg.node_texts)
                assert isinstance(weights, dict)
                assert len(weights) > 0
                
            except ImportError:
                pytest.skip("sentence-transformers not available")
    
    def test_training_configuration_variants(self):
        """Test different training configurations."""
        try:
            kg = create_synthetic_kg(num_nodes=25, num_edges=35, random_seed=42)
            dataset = LinkPredictionDataset(kg, mode="train")
            dataloader = create_dataloader(dataset, batch_size=4)
            
            model = HyperGNN(
                text_encoder="sentence-transformers/all-MiniLM-L6-v2",
                gnn_backbone="GCN",
                hidden_dim=32,
                num_layers=1,
            )
            
            # Test different scheduler types
            scheduler_types = ["cosine", "step", "plateau", None]
            
            for scheduler_type in scheduler_types:
                config = TrainingConfig(
                    num_epochs=1,  # Short for testing
                    scheduler_type=scheduler_type,
                    use_wandb=False,
                )
                
                trainer = HyperGNNTrainer(model, config)
                
                # Test that trainer initializes correctly
                assert trainer.model == model
                assert trainer.config.scheduler_type == scheduler_type
                
                if scheduler_type is None:
                    assert trainer.scheduler is None
                else:
                    assert trainer.scheduler is not None
                    
        except ImportError:
            pytest.skip("sentence-transformers not available")


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test realistic usage scenarios."""
    
    def test_research_workflow_simulation(self):
        """Simulate a realistic research workflow."""
        try:
            # 1. Data preparation
            domains = ["academic", "industrial", "medical"]
            graphs = {}
            
            for domain in domains:
                kg = create_synthetic_kg(
                    num_nodes=100 + len(domain) * 10,  # Slightly different sizes
                    num_edges=150 + len(domain) * 15,
                    relations=[f"{domain}_rel1", f"{domain}_rel2"],
                    random_seed=hash(domain) % 1000
                )
                graphs[domain] = kg
            
            # 2. Model development
            model = HyperGNN(
                text_encoder="sentence-transformers/all-MiniLM-L6-v2",
                gnn_backbone="GAT",
                hidden_dim=128,
                num_layers=3,
                num_heads=8,
                dropout=0.1,
            )
            
            # 3. Training setup
            train_kg = graphs["academic"]
            train_dataset = LinkPredictionDataset(train_kg, mode="train")
            val_dataset = LinkPredictionDataset(train_kg, mode="val")
            
            train_loader = create_dataloader(train_dataset, batch_size=16)
            val_loader = create_dataloader(val_dataset, batch_size=16)
            
            # 4. Mock training
            config = TrainingConfig(
                learning_rate=1e-3,
                num_epochs=2,  # Short for testing
                patience=5,
                use_wandb=False,
            )
            
            trainer = HyperGNNTrainer(model, config)
            
            with patch.object(trainer, '_forward_link_prediction') as mock_forward:
                mock_forward.return_value = torch.randn(16)
                
                history = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    task_type="link_prediction"
                )
                
                assert isinstance(history, dict)
            
            # 5. Zero-shot evaluation on other domains
            evaluator = BenchmarkEvaluator(model, torch.device("cpu"))
            
            results = {}
            for domain in ["industrial", "medical"]:
                test_kg = graphs[domain]
                test_dataset = LinkPredictionDataset(test_kg, mode="test")
                test_loader = create_dataloader(test_dataset, batch_size=16)
                
                with patch.object(evaluator, '_forward_link_prediction') as mock_eval:
                    mock_eval.return_value = torch.sigmoid(torch.randn(16))
                    
                    result = evaluator.evaluate_dataset(
                        dataloader=test_loader,
                        dataset_name=f"{domain}_domain",
                        task_type="link_prediction"
                    )
                    
                    results[domain] = result
            
            # 6. Compare results across domains
            for domain, result in results.items():
                assert "metrics" in result
                assert "auc_roc" in result["metrics"]
                
        except ImportError:
            pytest.skip("sentence-transformers not available")
    
    def test_production_deployment_simulation(self):
        """Simulate production deployment scenario."""
        try:
            # 1. Load pre-trained model (simulated)
            model = HyperGNN(
                text_encoder="sentence-transformers/all-MiniLM-L6-v2",
                gnn_backbone="GAT",
                hidden_dim=64,
                num_layers=2,
            )
            
            # 2. Setup inference pipeline
            def inference_pipeline(node_texts, edge_index, node_features):
                model.eval()
                with torch.no_grad():
                    return model.zero_shot_inference(edge_index, node_features, node_texts)
            
            # 3. Simulate incoming data
            incoming_kg = create_synthetic_kg(num_nodes=50, num_edges=75, random_seed=999)
            
            # 4. Run inference
            predictions = inference_pipeline(
                node_texts=incoming_kg.node_texts,
                edge_index=incoming_kg.edge_index,
                node_features=incoming_kg.node_features[:, :64]
            )
            
            # 5. Validate output
            assert isinstance(predictions, torch.Tensor)
            assert predictions.shape[0] == len(incoming_kg.node_texts)
            assert torch.all(torch.isfinite(predictions))  # No NaN or Inf values
            
            # 6. Performance monitoring
            import time
            
            start_time = time.time()
            for _ in range(10):  # Simulate multiple requests
                _ = inference_pipeline(
                    node_texts=incoming_kg.node_texts[:10],  # Smaller batches
                    edge_index=incoming_kg.edge_index[:, :20],
                    node_features=incoming_kg.node_features[:10, :64]
                )
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            assert avg_time < 1.0  # Should be fast enough for production
            
        except ImportError:
            pytest.skip("sentence-transformers not available")