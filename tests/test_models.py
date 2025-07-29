"""Tests for model implementations."""

import pytest
import torch
from graph_hypernetwork_forge.models import HyperGNN


class TestHyperGNN:
    """Test cases for HyperGNN model."""
    
    def test_model_initialization(self):
        """Test model can be initialized with default parameters."""
        model = HyperGNN()
        assert model.text_encoder == "sentence-transformers/all-MiniLM-L6-v2"
        assert model.gnn_backbone == "GAT"
        assert model.hidden_dim == 256
        assert model.num_layers == 3
    
    def test_model_custom_parameters(self):
        """Test model initialization with custom parameters."""
        model = HyperGNN(
            text_encoder="bert-base-uncased",
            gnn_backbone="GCN",
            hidden_dim=128,
            num_layers=2
        )
        assert model.text_encoder == "bert-base-uncased"
        assert model.gnn_backbone == "GCN"
        assert model.hidden_dim == 128
        assert model.num_layers == 2
    
    def test_generate_weights_not_implemented(self):
        """Test that generate_weights raises NotImplementedError."""
        model = HyperGNN()
        with pytest.raises(NotImplementedError):
            model.generate_weights({0: "test node"})
    
    def test_forward_not_implemented(self):
        """Test that forward pass raises NotImplementedError."""
        model = HyperGNN()
        edge_index = torch.tensor([[0, 1], [1, 0]])
        node_features = torch.randn(2, 128)
        
        with pytest.raises(NotImplementedError):
            model.forward(edge_index, node_features)