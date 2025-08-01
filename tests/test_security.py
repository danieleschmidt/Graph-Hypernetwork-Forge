"""Security tests for Graph Hypernetwork Forge."""
import pytest
import torch
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from graph_hypernetwork_forge import HyperGNN
from graph_hypernetwork_forge.models.encoders import TextEncoder


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""

    def test_malicious_text_input(self):
        """Test handling of potentially malicious text inputs."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        malicious_inputs = [
            # Very long text
            "A" * 10000,
            # Special characters and encoding attempts
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "\x00\x01\x02\x03",  # Control characters
            "π∑∞∂∆∇",  # Unicode characters
            "",  # Empty string
            " " * 1000,  # Whitespace only
        ]
        
        for malicious_input in malicious_inputs:
            try:
                weights = model.generate_weights([malicious_input])
                assert isinstance(weights, dict)
                # Should handle gracefully without crashing
            except Exception as e:
                # If an exception occurs, it should be a controlled one
                assert "validation" in str(e).lower() or "invalid" in str(e).lower()

    def test_tensor_dimension_validation(self):
        """Test validation of tensor dimensions to prevent buffer overflows."""
        model = HyperGNN(hidden_dim=64, num_layers=2)
        
        # Test with invalid edge indices
        invalid_edge_indices = [
            torch.tensor([[-1, 0], [0, 1]], dtype=torch.long),  # Negative indices
            torch.tensor([[0, 1], [1, 1000]], dtype=torch.long),  # Out of bounds
            torch.tensor([[0, 1, 2]], dtype=torch.long),  # Wrong shape
            torch.tensor([], dtype=torch.long),  # Empty tensor
        ]
        
        node_features = torch.randn(3, 64)
        texts = ["Node 1", "Node 2", "Node 3"]
        weights = model.generate_weights(texts)
        
        for invalid_edge_index in invalid_edge_indices:
            with pytest.raises((ValueError, IndexError, RuntimeError)):
                model.forward(invalid_edge_index, node_features, weights)

    def test_node_feature_validation(self):
        """Test validation of node features."""
        model = HyperGNN(hidden_dim=64, num_layers=2)
        
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        texts = ["Node 1", "Node 2"]
        weights = model.generate_weights(texts)
        
        invalid_node_features = [
            torch.tensor([]),  # Empty tensor
            torch.randn(2, 32),  # Wrong feature dimension
            torch.randn(5, 64),  # Wrong number of nodes
            torch.tensor([[float('inf'), 1.0], [2.0, float('nan')]], dtype=torch.float),  # Invalid values
        ]
        
        for invalid_features in invalid_node_features:
            with pytest.raises((ValueError, RuntimeError)):
                model.forward(edge_index, invalid_features, weights)

    def test_text_list_validation(self):
        """Test validation of text input lists."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        invalid_text_inputs = [
            None,  # None input
            [],  # Empty list
            [None, "Valid text"],  # List with None
            ["", ""],  # List of empty strings
            123,  # Non-list input
            ["Text"] * 100000,  # Extremely large list
        ]
        
        for invalid_input in invalid_text_inputs:
            try:
                weights = model.generate_weights(invalid_input)
                if invalid_input in [None, 123]:
                    pytest.fail("Should have raised an exception for invalid input type")
            except (TypeError, ValueError, AttributeError):
                # Expected behavior for invalid inputs
                pass


@pytest.mark.security
class TestModelSafety:
    """Test model safety and secure operations."""

    def test_weight_tampering_detection(self):
        """Test detection of tampered model weights."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        texts = ["Test node 1", "Test node 2"]
        
        # Generate legitimate weights
        original_weights = model.generate_weights(texts)
        
        # Tamper with weights
        tampered_weights = original_weights.copy()
        for key in tampered_weights:
            if tampered_weights[key].numel() > 0:
                tampered_weights[key] = torch.full_like(tampered_weights[key], 999.0)
        
        # Test with tampered weights
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_features = torch.randn(2, 32)
        
        # Should handle gracefully or detect anomalies
        predictions = model.forward(edge_index, node_features, tampered_weights)
        
        # Check for numerical instability
        assert torch.all(torch.isfinite(predictions)), "Predictions should be finite"
        assert not torch.all(predictions == 999.0), "Should not directly use tampered values"

    def test_memory_exhaustion_protection(self):
        """Test protection against memory exhaustion attacks."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Test with reasonable limits
        large_but_reasonable_texts = ["Large text description"] * 1000
        
        try:
            weights = model.generate_weights(large_but_reasonable_texts)
            assert isinstance(weights, dict)
        except Exception as e:
            # Should fail gracefully with memory error, not crash
            assert "memory" in str(e).lower() or "tensor" in str(e).lower()

    def test_model_serialization_safety(self, tmp_path):
        """Test safe model serialization and deserialization."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Save model safely
        model_path = tmp_path / "safe_model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Verify file was created and has reasonable size
        assert model_path.exists()
        file_size = model_path.stat().st_size
        assert 1000 < file_size < 10**8  # Reasonable size bounds
        
        # Load model safely
        loaded_state = torch.load(model_path, map_location='cpu')
        assert isinstance(loaded_state, dict)
        
        # Verify loaded model works
        new_model = HyperGNN(hidden_dim=32, num_layers=2)
        new_model.load_state_dict(loaded_state)
        
        test_weights = new_model.generate_weights(["Test text"])
        assert isinstance(test_weights, dict)


@pytest.mark.security
class TestDataPrivacy:
    """Test data privacy and information leakage prevention."""

    def test_text_information_leakage(self):
        """Test that model doesn't leak original text information."""
        model = HyperGNN(hidden_dim=64, num_layers=2)
        
        sensitive_texts = [
            "Patient John Doe has confidential medical condition XYZ",
            "API key: sk-1234567890abcdef",
            "Password: secret123!@#"
        ]
        
        weights = model.generate_weights(sensitive_texts)
        
        # Convert weights to strings and check for leakage
        weight_str = str(weights)
        for sensitive_text in sensitive_texts:
            # Check that no part of sensitive text appears in weights
            words = sensitive_text.split()
            for word in words:
                if len(word) > 3:  # Skip short words that might appear by chance
                    assert word.lower() not in weight_str.lower(), \
                        f"Sensitive word '{word}' found in model weights"

    def test_model_weight_determinism(self):
        """Test that identical inputs produce identical outputs."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Set deterministic behavior
        torch.manual_seed(42)
        test_texts = ["Deterministic test input"]
        weights_1 = model.generate_weights(test_texts)
        
        torch.manual_seed(42)
        weights_2 = model.generate_weights(test_texts)
        
        # Weights should be identical for same inputs
        for key in weights_1:
            assert torch.allclose(weights_1[key], weights_2[key], atol=1e-8), \
                f"Non-deterministic behavior detected in {key}"

    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_file_access_protection(self, mock_file_open):
        """Test protection against unauthorized file access."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Model should work without file access
        weights = model.generate_weights(["Test without file access"])
        assert isinstance(weights, dict)

    def test_gradient_information_leakage(self):
        """Test that gradients don't leak input information."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        model.train()
        
        sensitive_text = ["Confidential business strategy document"]
        
        # Forward pass with gradient computation
        weights = model.generate_weights(sensitive_text)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        node_features = torch.randn(1, 32, requires_grad=True)
        
        predictions = model.forward(edge_index, node_features, weights)
        loss = predictions.sum()
        loss.backward()
        
        # Check that gradients don't contain sensitive information
        if node_features.grad is not None:
            grad_str = str(node_features.grad.data)
            assert "confidential" not in grad_str.lower()
            assert "business" not in grad_str.lower()
            assert "strategy" not in grad_str.lower()


@pytest.mark.security
class TestAdversarialRobustness:
    """Test robustness against adversarial inputs."""

    def test_adversarial_text_inputs(self):
        """Test robustness against adversarial text modifications."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        base_text = "Normal graph node description"
        
        # Generate adversarial variants
        adversarial_variants = [
            base_text + " " + "adversarial" * 100,  # Repeated words
            base_text.replace("graph", "gr@ph"),  # Character substitution
            base_text.upper(),  # Case change
            base_text + "\n" * 100,  # Newline injection
            base_text + " 🚀" * 50,  # Emoji flooding
        ]
        
        base_weights = model.generate_weights([base_text])
        
        for adversarial_text in adversarial_variants:
            try:
                adv_weights = model.generate_weights([adversarial_text])
                
                # Weights should be similar but not identical
                for key in base_weights:
                    if key in adv_weights:
                        similarity = torch.nn.functional.cosine_similarity(
                            base_weights[key].flatten(),
                            adv_weights[key].flatten(),
                            dim=0
                        )
                        # Should be robust (similar) but not identical
                        assert 0.5 < similarity < 0.99, \
                            f"Adversarial robustness issue with {key}"
                            
            except Exception as e:
                # Should handle gracefully
                assert "validation" in str(e).lower()

    def test_numerical_adversarial_inputs(self):
        """Test robustness against adversarial numerical inputs."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Normal inputs
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_features = torch.randn(2, 32)
        texts = ["Node 1", "Node 2"]
        weights = model.generate_weights(texts)
        
        # Adversarial numerical modifications
        adversarial_features = [
            node_features + torch.randn_like(node_features) * 0.1,  # Small noise
            node_features * 1.1,  # Scaling
            torch.clamp(node_features + 0.01, -10, 10),  # Bounded perturbation
        ]
        
        base_predictions = model.forward(edge_index, node_features, weights)
        
        for adv_features in adversarial_features:
            adv_predictions = model.forward(edge_index, adv_features, weights)
            
            # Predictions should be similar for small perturbations
            similarity = torch.nn.functional.cosine_similarity(
                base_predictions.flatten(),
                adv_predictions.flatten(),
                dim=0
            )
            assert similarity > 0.8, "Model not robust to small numerical perturbations"

    def test_denial_of_service_protection(self):
        """Test protection against denial of service attacks."""
        model = HyperGNN(hidden_dim=16, num_layers=2)  # Small model for testing
        
        # Test rapid successive calls
        texts = ["DoS test text"]
        
        import time
        start_time = time.time()
        
        for _ in range(100):  # Rapid requests
            weights = model.generate_weights(texts)
            assert isinstance(weights, dict)
        
        total_time = time.time() - start_time
        
        # Should complete within reasonable time (no hanging)
        assert total_time < 30, "Model appears to be vulnerable to DoS attacks"
        
        # Memory should not grow excessively
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 1000, "Excessive memory usage detected"