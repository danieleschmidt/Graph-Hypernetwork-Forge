"""Tests for utility functions and helpers."""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from graph_hypernetwork_forge.utils import (
    graph_utils, text_utils, model_utils, evaluation_utils
)


class TestGraphUtils:
    """Test graph utility functions."""

    def test_edge_index_validation(self):
        """Test edge index validation utility."""
        valid_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        assert graph_utils.validate_edge_index(valid_edge_index, num_nodes=3)
        
        # Test invalid edge index (out of bounds)
        invalid_edge_index = torch.tensor([[0, 1, 5], [1, 2, 0]], dtype=torch.long)
        assert not graph_utils.validate_edge_index(invalid_edge_index, num_nodes=3)

    def test_adjacency_matrix_conversion(self):
        """Test conversion between edge_index and adjacency matrix."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        adj_matrix = graph_utils.edge_index_to_adjacency(edge_index, num_nodes=3)
        
        assert adj_matrix.shape == (3, 3)
        assert adj_matrix[0, 1] == 1
        assert adj_matrix[1, 2] == 1
        assert adj_matrix[2, 0] == 1
        
        # Test reverse conversion
        recovered_edge_index = graph_utils.adjacency_to_edge_index(adj_matrix)
        assert recovered_edge_index.shape[0] == 2
        assert recovered_edge_index.shape[1] == 3

    def test_degree_calculation(self):
        """Test node degree calculation."""
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        degrees = graph_utils.calculate_degrees(edge_index, num_nodes=3)
        
        assert degrees[0] == 1  # Node 0 has degree 1
        assert degrees[1] == 3  # Node 1 has degree 3
        assert degrees[2] == 1  # Node 2 has degree 1

    def test_graph_connectivity(self):
        """Test graph connectivity checking."""
        # Connected graph
        connected_edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        assert graph_utils.is_connected(connected_edges, num_nodes=3)
        
        # Disconnected graph
        disconnected_edges = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        assert not graph_utils.is_connected(disconnected_edges, num_nodes=4)

    def test_subgraph_extraction(self):
        """Test subgraph extraction utilities."""
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        node_subset = [1, 2]
        
        sub_edge_index, node_mapping = graph_utils.extract_subgraph(
            edge_index, node_subset, num_nodes=4
        )
        
        assert sub_edge_index.shape[1] == 2  # Only edges within subset
        assert 0 in node_mapping.values()  # Remapped indices


class TestTextUtils:
    """Test text processing utilities."""

    def test_text_preprocessing(self):
        """Test text preprocessing functions."""
        raw_text = "  This is a TEST text with CAPS and   spaces  "
        processed = text_utils.preprocess_text(raw_text)
        
        assert processed == "this is a test text with caps and spaces"
        assert processed.strip() == processed
        assert "  " not in processed

    def test_text_chunking(self):
        """Test text chunking for long sequences."""
        long_text = "word " * 1000  # 1000 words
        chunks = text_utils.chunk_text(long_text, max_length=512, overlap=50)
        
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 512 for chunk in chunks)
        
        # Test overlap functionality
        if len(chunks) > 1:
            first_chunk_end = chunks[0].split()[-50:]
            second_chunk_start = chunks[1].split()[:50]
            overlap_exists = any(word in second_chunk_start for word in first_chunk_end)
            assert overlap_exists

    def test_text_similarity(self):
        """Test text similarity calculations."""
        text1 = "The quick brown fox jumps"
        text2 = "A quick brown fox jumps"
        text3 = "Completely different content"
        
        sim_high = text_utils.calculate_similarity(text1, text2)
        sim_low = text_utils.calculate_similarity(text1, text3)
        
        assert sim_high > sim_low
        assert 0 <= sim_high <= 1
        assert 0 <= sim_low <= 1

    def test_keyword_extraction(self):
        """Test keyword extraction from text."""
        text = "Machine learning and artificial intelligence are transforming technology"
        keywords = text_utils.extract_keywords(text, num_keywords=3)
        
        assert len(keywords) <= 3
        assert all(isinstance(kw, str) for kw in keywords)
        assert all(len(kw) > 1 for kw in keywords)  # No single characters

    @pytest.mark.parametrize("text,expected_tokens", [
        ("Hello world", 2),
        ("", 0),
        ("Single", 1),
        ("Multiple words in sentence", 4)
    ])
    def test_tokenization(self, text, expected_tokens):
        """Test text tokenization with various inputs."""
        tokens = text_utils.tokenize(text)
        assert len(tokens) == expected_tokens


class TestModelUtils:
    """Test model utility functions."""

    def test_parameter_counting(self):
        """Test model parameter counting utility."""
        # Create a simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.Linear(5, 1)
        )
        
        total_params = model_utils.count_parameters(model)
        trainable_params = model_utils.count_parameters(model, trainable_only=True)
        
        expected_params = (10 * 5 + 5) + (5 * 1 + 1)  # weights + biases
        assert total_params == expected_params
        assert trainable_params == expected_params

    def test_model_size_calculation(self):
        """Test model memory size calculation."""
        model = torch.nn.Linear(100, 50)
        size_mb = model_utils.calculate_model_size(model)
        
        assert size_mb > 0
        assert isinstance(size_mb, float)

    def test_gradient_clipping(self):
        """Test gradient clipping utility."""
        model = torch.nn.Linear(10, 5)
        
        # Create some gradients
        dummy_loss = model(torch.randn(1, 10)).sum()
        dummy_loss.backward()
        
        # Test gradient clipping
        original_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        clipped_grad_norm = model_utils.clip_gradients(model, max_norm=1.0)
        
        assert clipped_grad_norm <= 1.0
        assert isinstance(clipped_grad_norm, float)

    def test_model_device_management(self):
        """Test model device placement utilities."""
        model = torch.nn.Linear(10, 5)
        
        # Test device detection
        device = model_utils.get_model_device(model)
        assert device.type in ['cpu', 'cuda']
        
        # Test moving model to device
        target_device = torch.device('cpu')
        moved_model = model_utils.move_to_device(model, target_device)
        
        for param in moved_model.parameters():
            assert param.device.type == target_device.type

    def test_model_state_management(self):
        """Test model state save/load utilities."""
        model = torch.nn.Linear(10, 5)
        original_state = model.state_dict()
        
        # Save state
        state_dict = model_utils.save_model_state(model)
        assert isinstance(state_dict, dict)
        
        # Modify model
        with torch.no_grad():
            model.weight.fill_(0)
        
        # Load original state
        model_utils.load_model_state(model, original_state)
        
        # Verify restoration
        for name, param in model.named_parameters():
            assert torch.equal(param, original_state[name])


class TestEvaluationUtils:
    """Test evaluation and metrics utilities."""

    def test_accuracy_calculation(self):
        """Test accuracy metric calculation."""
        predictions = torch.tensor([0, 1, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 1])
        
        accuracy = evaluation_utils.calculate_accuracy(predictions, targets)
        expected_accuracy = 4/5  # 4 correct out of 5
        
        assert abs(accuracy - expected_accuracy) < 1e-6

    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        predictions = torch.tensor([1, 1, 0, 1, 0])
        targets = torch.tensor([1, 0, 0, 1, 1])
        
        f1 = evaluation_utils.calculate_f1_score(predictions, targets)
        
        assert 0 <= f1 <= 1
        assert isinstance(f1, float)

    def test_confusion_matrix(self):
        """Test confusion matrix generation."""
        predictions = torch.tensor([0, 1, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 1])
        
        cm = evaluation_utils.confusion_matrix(predictions, targets, num_classes=2)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(predictions)
        assert torch.all(cm >= 0)

    def test_classification_report(self):
        """Test classification report generation."""
        predictions = torch.tensor([0, 1, 1, 0, 1, 0, 1, 0])
        targets = torch.tensor([0, 1, 0, 0, 1, 1, 1, 0])
        
        report = evaluation_utils.classification_report(predictions, targets)
        
        assert isinstance(report, dict)
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1_score' in report

    @pytest.mark.parametrize("metric_name,expected_range", [
        ("accuracy", (0, 1)),
        ("precision", (0, 1)),
        ("recall", (0, 1)),
        ("f1_score", (0, 1))
    ])
    def test_metric_ranges(self, metric_name, expected_range):
        """Test that metrics are within expected ranges."""
        predictions = torch.randint(0, 2, (100,))
        targets = torch.randint(0, 2, (100,))
        
        metric_func = getattr(evaluation_utils, f"calculate_{metric_name}")
        metric_value = metric_func(predictions, targets)
        
        assert expected_range[0] <= metric_value <= expected_range[1]

    def test_top_k_accuracy(self):
        """Test top-k accuracy calculation."""
        # Predictions: probabilities for 3 classes, 5 samples
        predictions = torch.tensor([
            [0.1, 0.8, 0.1],  # Predicted class 1, actual 1 ✓
            [0.3, 0.6, 0.1],  # Predicted class 1, actual 0, but 0 is in top-2 ✓
            [0.2, 0.1, 0.7],  # Predicted class 2, actual 2 ✓
            [0.4, 0.3, 0.3],  # Predicted class 0, actual 1, but 1 is in top-2 ✓
            [0.1, 0.1, 0.8],  # Predicted class 2, actual 0, 0 not in top-2 ✗
        ])
        targets = torch.tensor([1, 0, 2, 1, 0])
        
        top1_acc = evaluation_utils.top_k_accuracy(predictions, targets, k=1)
        top2_acc = evaluation_utils.top_k_accuracy(predictions, targets, k=2)
        
        assert top1_acc == 0.6  # 3/5 correct
        assert top2_acc == 0.8  # 4/5 correct