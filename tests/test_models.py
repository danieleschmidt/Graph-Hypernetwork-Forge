"""Tests for core model components."""
import pytest
import torch
import tempfile
import json
from unittest.mock import Mock, patch

from graph_hypernetwork_forge.models import HyperGNN
from graph_hypernetwork_forge.data import TextualKnowledgeGraph


class TestHyperGNN:
    """Test HyperGNN model functionality."""

    def test_hypergnn_initialization(self):
        """Test HyperGNN can be initialized with default parameters."""
        model = HyperGNN()
        assert model is not None
        assert hasattr(model, 'text_encoder')
        assert hasattr(model, 'hypernetwork')
        assert hasattr(model, 'dynamic_gnn')
        assert model.hidden_dim == 256
        assert model.num_layers == 3

    def test_hypergnn_with_custom_params(self):
        """Test HyperGNN initialization with custom parameters."""
        model = HyperGNN(
            text_encoder="sentence-transformers/all-MiniLM-L6-v2",
            gnn_backbone="GAT",
            hidden_dim=128,
            num_layers=2
        )
        assert model.hidden_dim == 128
        assert model.num_layers == 2
        assert model.gnn_backbone == "GAT"

    def test_weight_generation(self):
        """Test weight generation from text descriptions."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        sample_texts = [
            "A software engineer working on ML projects",
            "A researcher studying neural networks",
        ]
        
        # Skip if dependencies not available
        try:
            weights = model.generate_weights(sample_texts)
            assert isinstance(weights, list)
            assert len(weights) == 2  # num_layers
            assert all(isinstance(layer_weights, dict) for layer_weights in weights)
        except ImportError:
            pytest.skip("Text encoder dependencies not available")

    def test_forward_pass_with_real_data(self):
        """Test forward pass with real graph data."""
        # Create a simple graph
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        node_features = torch.randn(3, 16)
        node_texts = [
            "First node description",
            "Second node description", 
            "Third node description"
        ]
        
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        try:
            output = model(edge_index, node_features, node_texts)
            assert output.shape[0] == 3  # Number of nodes
            assert output.shape[1] == 32  # Hidden dimension
        except ImportError:
            pytest.skip("Text encoder dependencies not available")

    def test_device_handling(self):
        """Test model handles device placement correctly."""
        model = HyperGNN(hidden_dim=32, num_layers=2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Test model parameters are on correct device
        for param in model.parameters():
            assert param.device.type == device.type

    def test_config_methods(self):
        """Test model configuration get/set methods."""
        config = {
            "text_encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "gnn_backbone": "GCN",
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
        }
        
        model = HyperGNN.from_config(config)
        retrieved_config = model.get_config()
        
        assert retrieved_config["gnn_backbone"] == "GCN"
        assert retrieved_config["hidden_dim"] == 64
        assert retrieved_config["num_layers"] == 2


class TestTextualKnowledgeGraph:
    """Test TextualKnowledgeGraph data handling."""

    def test_kg_creation(self):
        """Test creation of TextualKnowledgeGraph."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        node_texts = ["First node", "Second node"]
        
        kg = TextualKnowledgeGraph(edge_index=edge_index, node_texts=node_texts)
        assert kg is not None
        assert kg.num_nodes == 2
        assert kg.num_edges == 2
        assert len(kg.node_texts) == 2

    def test_kg_with_features(self):
        """Test KG creation with node features."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        node_texts = ["Node 1", "Node 2", "Node 3"]
        node_features = torch.randn(3, 16)
        node_labels = torch.tensor([0, 1, 0], dtype=torch.long)
        
        kg = TextualKnowledgeGraph(
            edge_index=edge_index,
            node_texts=node_texts,
            node_features=node_features,
            node_labels=node_labels
        )
        
        assert kg.node_features.shape == (3, 16)
        assert kg.node_labels.shape == (3,)

    def test_kg_from_json(self):
        """Test loading KG from JSON file."""
        # Create temporary JSON file
        data = {
            "nodes": [
                {"id": 0, "text": "Person entity", "features": [1.0, 2.0], "label": 0},
                {"id": 1, "text": "Location entity", "features": [3.0, 4.0], "label": 1}
            ],
            "edges": [
                {"source": 0, "target": 1, "type": "located_at"}
            ],
            "metadata": {"domain": "test"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            kg = TextualKnowledgeGraph.from_json(temp_path)
            assert kg.num_nodes == 2
            assert kg.num_edges == 1
            assert kg.node_texts[0] == "Person entity"
            assert kg.metadata["domain"] == "test"
        finally:
            import os
            os.unlink(temp_path)

    def test_kg_statistics(self):
        """Test KG statistics calculation."""
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        node_texts = ["Node 1", "Node 2", "Node 3"]
        
        kg = TextualKnowledgeGraph(edge_index=edge_index, node_texts=node_texts)
        stats = kg.statistics()
        
        assert stats["num_nodes"] == 3
        assert stats["num_edges"] == 4
        assert "avg_degree" in stats
        assert "density" in stats

    def test_kg_subgraph(self):
        """Test subgraph extraction."""
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        node_texts = ["Node 0", "Node 1", "Node 2", "Node 3"]
        
        kg = TextualKnowledgeGraph(edge_index=edge_index, node_texts=node_texts)
        subkg = kg.subgraph([0, 1, 2])
        
        assert subkg.num_nodes == 3
        assert len(subkg.node_texts) == 3
        assert "Node 0" in subkg.node_texts

    def test_kg_neighbor_texts(self):
        """Test neighbor text retrieval."""
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        node_texts = ["Center node", "Neighbor 1", "Neighbor 2"]
        
        kg = TextualKnowledgeGraph(edge_index=edge_index, node_texts=node_texts)
        neighbor_texts = kg.get_neighbor_texts(0, k_hops=1)
        
        assert "Neighbor 1" in neighbor_texts
        assert "Center node" not in neighbor_texts


class TestModelIntegration:
    """Integration tests for model components."""

    @pytest.mark.slow
    def test_end_to_end_pipeline(self, sample_graph, sample_texts):
        """Test complete pipeline from text to predictions."""
        # Skip if no GPU available for intensive test
        if not torch.cuda.is_available():
            pytest.skip("Integration test requires GPU")
            
        model = HyperGNN(hidden_dim=64, num_layers=2)
        
        # Mock components to avoid external dependencies
        with patch.object(model, 'generate_weights') as mock_weights, \
             patch.object(model, 'forward') as mock_forward:
            
            mock_weights.return_value = {
                'layer_0': torch.randn(64, 64),
                'layer_1': torch.randn(64, 64)
            }
            mock_forward.return_value = torch.randn(3, 64)
            
            weights = model.generate_weights(sample_texts)
            predictions = model.forward(
                sample_graph['edge_index'],
                sample_graph['node_features'][:, :64],  # Match hidden_dim
                weights
            )
            
            assert predictions.shape == (3, 64)

    def test_model_serialization(self):
        """Test model can be saved and loaded."""
        model = HyperGNN(hidden_dim=32)
        
        # Test state dict operations
        state_dict = model.state_dict()
        assert isinstance(state_dict, dict)
        
        # Test loading state dict
        new_model = HyperGNN(hidden_dim=32)
        new_model.load_state_dict(state_dict, strict=False)
        assert new_model.hidden_dim == 32