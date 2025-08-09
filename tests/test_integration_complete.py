"""Comprehensive integration tests for HyperGNN."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

from graph_hypernetwork_forge import HyperGNN, TextualKnowledgeGraph
from graph_hypernetwork_forge.utils import SyntheticDataGenerator
from graph_hypernetwork_forge.utils.optimization import global_profiler
from graph_hypernetwork_forge.utils.monitoring import ModelAnalyzer


class TestHyperGNNIntegration:
    """Integration tests for complete HyperGNN functionality."""

    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        return HyperGNN(
            gnn_backbone="GAT",
            hidden_dim=32,
            num_layers=2,
            enable_caching=True
        )

    @pytest.fixture 
    def sample_graph(self):
        """Create a sample knowledge graph."""
        gen = SyntheticDataGenerator(seed=42)
        return gen.generate_social_network(num_nodes=10)

    def test_end_to_end_inference(self, small_model, sample_graph):
        """Test complete inference pipeline."""
        # Prepare inputs
        node_features = torch.randn(sample_graph.num_nodes, small_model.hidden_dim)
        
        # Forward pass
        small_model.eval()
        with torch.no_grad():
            predictions = small_model(
                sample_graph.edge_index, 
                node_features, 
                sample_graph.node_texts
            )
        
        # Validate outputs
        assert predictions.shape == (sample_graph.num_nodes, small_model.hidden_dim)
        assert not torch.isnan(predictions).any(), "Predictions contain NaN values"
        assert torch.isfinite(predictions).all(), "Predictions contain infinite values"
        
        # Check reasonable output range
        assert predictions.abs().max() < 1.0, "Predictions have unreasonable magnitude"

    def test_zero_shot_transfer(self, small_model):
        """Test zero-shot transfer between domains."""
        gen = SyntheticDataGenerator(seed=42)
        
        # Create graphs from different domains
        social_graph = gen.generate_social_network(num_nodes=8)
        citation_graph = gen.generate_citation_network(num_nodes=8)
        
        small_model.eval()
        with torch.no_grad():
            # Social domain
            social_features = torch.randn(social_graph.num_nodes, small_model.hidden_dim)
            social_preds = small_model(
                social_graph.edge_index, 
                social_features, 
                social_graph.node_texts
            )
            
            # Citation domain (zero-shot)
            citation_features = torch.randn(citation_graph.num_nodes, small_model.hidden_dim)
            citation_preds = small_model(
                citation_graph.edge_index,
                citation_features, 
                citation_graph.node_texts
            )
        
        # Both should produce valid outputs
        assert social_preds.shape[1] == citation_preds.shape[1]
        assert not torch.isnan(social_preds).any()
        assert not torch.isnan(citation_preds).any()
        
        # Predictions should be different due to different text inputs
        # (but this requires different texts, which our synthetic data provides)
        assert social_preds.shape[0] == 8
        assert citation_preds.shape[0] == 8

    def test_gradient_flow(self, small_model, sample_graph):
        """Test that gradients flow correctly through the model."""
        # Enable training mode
        small_model.train()
        
        # Create synthetic target
        node_features = torch.randn(sample_graph.num_nodes, small_model.hidden_dim)
        target = torch.randn(sample_graph.num_nodes, small_model.hidden_dim)
        
        # Forward pass
        predictions = small_model(
            sample_graph.edge_index,
            node_features,
            sample_graph.node_texts
        )
        
        # Compute loss and backward pass
        loss = nn.MSELoss()(predictions, target)
        loss.backward()
        
        # Check gradients exist and are reasonable
        has_gradients = False
        max_grad_norm = 0.0
        
        for name, param in small_model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
                
                # Check for NaN/inf gradients
                assert not torch.isnan(param.grad).any(), f"NaN gradients in {name}"
                assert torch.isfinite(param.grad).all(), f"Infinite gradients in {name}"
        
        assert has_gradients, "No gradients found in model parameters"
        assert max_grad_norm > 0, "All gradients are zero"
        assert max_grad_norm < 100, f"Gradients too large: {max_grad_norm}"

    def test_caching_consistency(self, small_model, sample_graph):
        """Test that caching produces identical results."""
        if small_model.weight_cache is None:
            pytest.skip("Caching not enabled")
        
        # Clear cache
        small_model.weight_cache.clear()
        
        node_features = torch.randn(sample_graph.num_nodes, small_model.hidden_dim)
        
        small_model.eval()
        with torch.no_grad():
            # First run (cache miss)
            pred1 = small_model(
                sample_graph.edge_index,
                node_features,
                sample_graph.node_texts
            )
            
            # Second run (cache hit)  
            pred2 = small_model(
                sample_graph.edge_index,
                node_features,
                sample_graph.node_texts
            )
        
        # Results should be identical
        assert torch.allclose(pred1, pred2, atol=1e-6), "Cached results are not identical"
        assert small_model.weight_cache.size() > 0, "Cache was not used"

    def test_different_gnn_backbones(self, sample_graph):
        """Test all supported GNN backbones."""
        backbones = ["GCN", "GAT", "SAGE"]
        
        for backbone in backbones:
            model = HyperGNN(
                gnn_backbone=backbone,
                hidden_dim=32,
                num_layers=2
            )
            
            node_features = torch.randn(sample_graph.num_nodes, model.hidden_dim)
            
            model.eval()
            with torch.no_grad():
                predictions = model(
                    sample_graph.edge_index,
                    node_features,
                    sample_graph.node_texts
                )
            
            # All backbones should produce valid outputs
            assert predictions.shape == (sample_graph.num_nodes, model.hidden_dim)
            assert not torch.isnan(predictions).any(), f"NaN outputs with {backbone}"
            assert torch.isfinite(predictions).all(), f"Infinite outputs with {backbone}"

    def test_model_scaling(self):
        """Test model behavior with different scales."""
        configs = [
            {"hidden_dim": 16, "num_layers": 1, "nodes": 5},
            {"hidden_dim": 64, "num_layers": 2, "nodes": 20},
            {"hidden_dim": 128, "num_layers": 3, "nodes": 50}
        ]
        
        gen = SyntheticDataGenerator()
        
        for config in configs:
            model = HyperGNN(
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                enable_caching=True
            )
            
            graph = gen.generate_social_network(num_nodes=config["nodes"])
            node_features = torch.randn(graph.num_nodes, config["hidden_dim"])
            
            model.eval()
            with torch.no_grad():
                predictions = model(graph.edge_index, node_features, graph.node_texts)
            
            expected_shape = (config["nodes"], config["hidden_dim"])
            assert predictions.shape == expected_shape, f"Wrong shape for config {config}"
            assert not torch.isnan(predictions).any()

    def test_weight_generation_consistency(self, small_model):
        """Test weight generation produces consistent results."""
        texts = [
            "A software engineer working on AI projects.",
            "A researcher studying machine learning algorithms.",
            "A product manager focused on user experience."
        ]
        
        small_model.eval()
        with torch.no_grad():
            # Generate weights multiple times
            weights1 = small_model.generate_weights(texts)
            weights2 = small_model.generate_weights(texts)
        
        # Should be identical (deterministic generation)
        assert len(weights1) == len(weights2)
        
        for layer_idx in range(len(weights1)):
            for weight_name in weights1[layer_idx]:
                w1 = weights1[layer_idx][weight_name]
                w2 = weights2[layer_idx][weight_name]
                assert torch.allclose(w1, w2, atol=1e-6), f"Inconsistent weights at layer {layer_idx}, {weight_name}"

    def test_textual_knowledge_graph_integration(self):
        """Test integration with TextualKnowledgeGraph."""
        # Create graph directly
        kg = TextualKnowledgeGraph()
        
        # Add nodes with text
        kg.add_node(0, text="Machine learning researcher", features=torch.randn(32))
        kg.add_node(1, text="Data scientist with expertise in NLP", features=torch.randn(32))
        kg.add_node(2, text="Software engineer building AI systems", features=torch.randn(32))
        
        # Add edges
        kg.add_edge(0, 1, relation="collaborates_with")
        kg.add_edge(1, 2, relation="mentors") 
        kg.add_edge(2, 0, relation="learns_from")
        
        # Test with HyperGNN
        model = HyperGNN(hidden_dim=32, num_layers=2)
        node_features = torch.stack([kg.nodes[i]["features"] for i in range(3)])
        
        model.eval()
        with torch.no_grad():
            predictions = model(kg.edge_index, node_features, kg.node_texts)
        
        assert predictions.shape == (3, 32)
        assert not torch.isnan(predictions).any()

    def test_performance_benchmarks(self, small_model, sample_graph):
        """Test performance meets minimum requirements."""
        node_features = torch.randn(sample_graph.num_nodes, small_model.hidden_dim)
        
        # Warmup
        small_model.eval()
        with torch.no_grad():
            _ = small_model(sample_graph.edge_index, node_features, sample_graph.node_texts)
        
        # Benchmark
        import time
        times = []
        
        for _ in range(5):
            start = time.time()
            with torch.no_grad():
                _ = small_model(sample_graph.edge_index, node_features, sample_graph.node_texts)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        throughput = sample_graph.num_nodes / avg_time
        
        # Performance requirements
        assert avg_time < 1.0, f"Inference too slow: {avg_time:.3f}s for {sample_graph.num_nodes} nodes"
        assert throughput > 5.0, f"Throughput too low: {throughput:.1f} nodes/sec"

    def test_memory_efficiency(self, sample_graph):
        """Test memory usage is reasonable."""
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and use model
        model = HyperGNN(hidden_dim=64, num_layers=2)
        node_features = torch.randn(sample_graph.num_nodes, 64)
        
        model.eval()
        with torch.no_grad():
            predictions = model(sample_graph.edge_index, node_features, sample_graph.node_texts)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Memory requirements (should be reasonable)
        assert memory_used < 500, f"Memory usage too high: {memory_used:.1f} MB"
        
        # Cleanup
        del model
        del predictions
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class TestQualityGates:
    """Quality gate tests that must pass for production."""

    def test_no_warnings_during_inference(self, capfd):
        """Test that inference produces no warnings."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        gen = SyntheticDataGenerator()
        graph = gen.generate_social_network(num_nodes=5)
        node_features = torch.randn(graph.num_nodes, 32)
        
        model.eval()
        with torch.no_grad():
            _ = model(graph.edge_index, node_features, graph.node_texts)
        
        # Check for warnings
        captured = capfd.readouterr()
        assert "warning" not in captured.err.lower(), f"Warnings detected: {captured.err}"
        assert "deprecated" not in captured.err.lower(), f"Deprecation warnings: {captured.err}"

    def test_model_determinism(self):
        """Test that model is deterministic given same inputs and seed."""
        torch.manual_seed(42)
        
        model1 = HyperGNN(hidden_dim=32, num_layers=2)
        gen = SyntheticDataGenerator(seed=42)
        graph = gen.generate_social_network(num_nodes=5)
        node_features = torch.randn(graph.num_nodes, 32)
        
        model1.eval()
        with torch.no_grad():
            pred1 = model1(graph.edge_index, node_features, graph.node_texts)
        
        # Reset and recreate
        torch.manual_seed(42)
        model2 = HyperGNN(hidden_dim=32, num_layers=2) 
        
        model2.eval()
        with torch.no_grad():
            pred2 = model2(graph.edge_index, node_features, graph.node_texts)
        
        # Should be identical
        assert torch.allclose(pred1, pred2, atol=1e-6), "Model is not deterministic"

    def test_input_validation(self):
        """Test input validation and error handling."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Test empty inputs
        with pytest.raises(Exception):
            model(torch.empty(2, 0), torch.empty(0, 32), [])
        
        # Test mismatched dimensions
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_features = torch.randn(2, 64)  # Wrong feature dim
        node_texts = ["text1", "text2"]
        
        with pytest.raises(Exception):
            model(edge_index, node_features, node_texts)

    def test_model_serialization(self, tmp_path):
        """Test model can be saved and loaded."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Save model
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Load model
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
        
        assert torch.allclose(pred1, pred2, atol=1e-6), "Loaded model produces different outputs"

    def test_code_quality_metrics(self):
        """Test code quality metrics meet standards."""
        # Test model parameter count is reasonable
        model = HyperGNN(hidden_dim=128, num_layers=3)
        param_count = ModelAnalyzer.count_parameters(model)
        
        assert param_count['total'] < 100_000_000, f"Too many parameters: {param_count['total']:,}"
        assert param_count['trainable'] > 0, "No trainable parameters"
        
        # Test model size is reasonable
        model_size = ModelAnalyzer.get_model_size(model)
        assert model_size['total_mb'] < 500, f"Model too large: {model_size['total_mb']:.1f} MB"


# Benchmark tests
class TestBenchmarks:
    """Performance benchmark tests."""

    def test_benchmark_inference_speed(self, benchmark):
        """Benchmark inference speed."""
        model = HyperGNN(hidden_dim=64, num_layers=2)
        gen = SyntheticDataGenerator()
        graph = gen.generate_social_network(num_nodes=20)
        node_features = torch.randn(graph.num_nodes, 64)
        
        model.eval()
        
        def inference():
            with torch.no_grad():
                return model(graph.edge_index, node_features, graph.node_texts)
        
        result = benchmark(inference)
        assert result is not None

    def test_benchmark_weight_generation(self, benchmark):
        """Benchmark weight generation speed."""
        model = HyperGNN(hidden_dim=64, num_layers=2)
        texts = [
            f"Person {i} working in technology" 
            for i in range(10)
        ]
        
        model.eval()
        
        def generate_weights():
            with torch.no_grad():
                return model.generate_weights(texts)
        
        result = benchmark(generate_weights)
        assert result is not None