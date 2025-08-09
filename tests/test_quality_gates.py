"""Quality gate tests for production readiness."""

import torch
import pytest
import warnings
from graph_hypernetwork_forge import HyperGNN, TextualKnowledgeGraph
from graph_hypernetwork_forge.utils import SyntheticDataGenerator


class TestProductionReadiness:
    """Tests that must pass for production deployment."""

    def test_basic_functionality(self):
        """Test basic model functionality works."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        gen = SyntheticDataGenerator(seed=42)
        graph = gen.generate_social_network(num_nodes=5)
        
        node_features = torch.randn(graph.num_nodes, 32)
        
        model.eval()
        with torch.no_grad():
            predictions = model(graph.edge_index, node_features, graph.node_texts)
        
        # Validate basic properties
        assert predictions.shape == (5, 32)
        assert not torch.isnan(predictions).any()
        assert torch.isfinite(predictions).all()

    def test_no_runtime_warnings(self):
        """Test that no warnings are raised during inference."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            model = HyperGNN(hidden_dim=32, num_layers=2)
            gen = SyntheticDataGenerator(seed=42)
            graph = gen.generate_social_network(num_nodes=5)
            
            node_features = torch.randn(graph.num_nodes, 32)
            
            model.eval()
            with torch.no_grad():
                _ = model(graph.edge_index, node_features, graph.node_texts)
        
        # Check for warnings (filter out known harmless warnings)
        problematic_warnings = [
            w for w in warning_list 
            if not any(ignore in str(w.message).lower() for ignore in [
                "torch.utils.checkpoint",
                "deprecated",  # Allow some deprecation warnings for now
            ])
        ]
        
        assert len(problematic_warnings) == 0, f"Warnings detected: {[str(w.message) for w in problematic_warnings]}"

    def test_deterministic_behavior(self):
        """Test that model produces consistent results."""
        torch.manual_seed(42)
        model1 = HyperGNN(hidden_dim=32, num_layers=2)
        gen = SyntheticDataGenerator(seed=42)
        graph = gen.generate_social_network(num_nodes=5)
        
        node_features = torch.randn(graph.num_nodes, 32)
        
        model1.eval()
        with torch.no_grad():
            pred1 = model1(graph.edge_index, node_features, graph.node_texts)
        
        # Create identical model and test
        torch.manual_seed(42)
        model2 = HyperGNN(hidden_dim=32, num_layers=2)
        
        model2.eval()
        with torch.no_grad():
            pred2 = model2(graph.edge_index, node_features, graph.node_texts)
        
        assert torch.allclose(pred1, pred2, atol=1e-5)

    def test_memory_efficiency(self):
        """Test that model doesn't leak memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and use model multiple times
        for _ in range(3):
            model = HyperGNN(hidden_dim=64, num_layers=2)
            gen = SyntheticDataGenerator()
            graph = gen.generate_social_network(num_nodes=10)
            
            node_features = torch.randn(graph.num_nodes, 64)
            
            model.eval()
            with torch.no_grad():
                _ = model(graph.edge_index, node_features, graph.node_texts)
            
            del model
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Should not increase memory significantly
        assert memory_increase < 100, f"Memory leak detected: {memory_increase:.1f} MB increase"

    def test_error_handling(self):
        """Test graceful error handling."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Test with empty inputs
        with pytest.raises(Exception):  # Should raise some kind of error
            model(torch.empty(2, 0), torch.empty(0, 32), [])
        
        # Test with mismatched dimensions  
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        wrong_features = torch.randn(2, 16)  # Wrong dimension
        texts = ["text1", "text2"]
        
        with pytest.raises(Exception):  # Should raise dimension error
            model(edge_index, wrong_features, texts)

    def test_performance_requirements(self):
        """Test that performance meets minimum requirements."""
        model = HyperGNN(hidden_dim=64, num_layers=2)
        gen = SyntheticDataGenerator()
        graph = gen.generate_social_network(num_nodes=20)
        
        node_features = torch.randn(graph.num_nodes, 64)
        
        # Warmup
        model.eval()
        with torch.no_grad():
            _ = model(graph.edge_index, node_features, graph.node_texts)
        
        # Time multiple runs
        import time
        times = []
        
        for _ in range(3):
            start = time.time()
            with torch.no_grad():
                _ = model(graph.edge_index, node_features, graph.node_texts)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        throughput = graph.num_nodes / avg_time
        
        # Performance requirements
        assert avg_time < 2.0, f"Inference too slow: {avg_time:.3f}s"
        assert throughput > 10.0, f"Throughput too low: {throughput:.1f} nodes/sec"

    def test_caching_functionality(self):
        """Test that caching works if enabled."""
        model = HyperGNN(hidden_dim=32, num_layers=2, enable_caching=True)
        
        if model.weight_cache is None:
            pytest.skip("Caching not available")
        
        gen = SyntheticDataGenerator(seed=42)
        graph = gen.generate_social_network(num_nodes=5)
        node_features = torch.randn(graph.num_nodes, 32)
        
        # Clear cache
        model.weight_cache.clear()
        
        model.eval()
        with torch.no_grad():
            pred1 = model(graph.edge_index, node_features, graph.node_texts)
            pred2 = model(graph.edge_index, node_features, graph.node_texts)
        
        # Results should be identical
        assert torch.allclose(pred1, pred2, atol=1e-6)
        assert model.weight_cache.size() > 0

    def test_gradient_flow(self):
        """Test that gradients flow correctly in training mode."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        gen = SyntheticDataGenerator(seed=42)
        graph = gen.generate_social_network(num_nodes=5)
        
        node_features = torch.randn(graph.num_nodes, 32)
        target = torch.randn(graph.num_nodes, 32)
        
        # Training mode
        model.train()
        
        predictions = model(graph.edge_index, node_features, graph.node_texts)
        loss = torch.nn.MSELoss()(predictions, target)
        loss.backward()
        
        # Check that some parameters have gradients
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.norm() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients found in model parameters"

    def test_model_serialization(self, tmp_path):
        """Test that model can be saved and loaded correctly."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Save model
        model_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Load into new model
        model2 = HyperGNN(hidden_dim=32, num_layers=2)
        model2.load_state_dict(torch.load(model_path))
        
        # Test they produce same outputs
        gen = SyntheticDataGenerator(seed=42)
        graph = gen.generate_social_network(num_nodes=5)
        node_features = torch.randn(graph.num_nodes, 32)
        
        model.eval()
        model2.eval()
        with torch.no_grad():
            pred1 = model(graph.edge_index, node_features, graph.node_texts)
            pred2 = model2(graph.edge_index, node_features, graph.node_texts)
        
        assert torch.allclose(pred1, pred2, atol=1e-6)


class TestScalability:
    """Test model scalability across different configurations."""

    def test_different_model_sizes(self):
        """Test model works with different sizes."""
        configs = [
            {"hidden_dim": 16, "num_layers": 1},
            {"hidden_dim": 64, "num_layers": 2}, 
            {"hidden_dim": 128, "num_layers": 3},
        ]
        
        gen = SyntheticDataGenerator()
        
        for config in configs:
            model = HyperGNN(**config)
            graph = gen.generate_social_network(num_nodes=8)
            node_features = torch.randn(graph.num_nodes, config["hidden_dim"])
            
            model.eval()
            with torch.no_grad():
                predictions = model(graph.edge_index, node_features, graph.node_texts)
            
            assert predictions.shape == (8, config["hidden_dim"])
            assert not torch.isnan(predictions).any()

    def test_different_graph_sizes(self):
        """Test model scales with different graph sizes."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        gen = SyntheticDataGenerator()
        
        sizes = [5, 20, 50]
        
        for size in sizes:
            graph = gen.generate_social_network(num_nodes=size)
            node_features = torch.randn(graph.num_nodes, 32)
            
            model.eval()
            with torch.no_grad():
                predictions = model(graph.edge_index, node_features, graph.node_texts)
            
            assert predictions.shape == (size, 32)
            assert not torch.isnan(predictions).any()

    def test_all_gnn_backbones(self):
        """Test all supported GNN backbones work."""
        backbones = ["GCN", "GAT", "SAGE"]
        gen = SyntheticDataGenerator(seed=42)
        graph = gen.generate_social_network(num_nodes=8)
        
        for backbone in backbones:
            model = HyperGNN(gnn_backbone=backbone, hidden_dim=32, num_layers=2)
            node_features = torch.randn(graph.num_nodes, 32)
            
            model.eval()
            with torch.no_grad():
                predictions = model(graph.edge_index, node_features, graph.node_texts)
            
            assert predictions.shape == (8, 32)
            assert not torch.isnan(predictions).any(), f"NaN outputs with {backbone}"