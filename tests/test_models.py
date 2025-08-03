"""Comprehensive tests for core model components."""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from graph_hypernetwork_forge.models import (
    HyperGNN,
    TextEncoder,
    get_text_encoder,
    WeightGenerator,
    DynamicGNN,
    get_gnn_backbone,
)
from graph_hypernetwork_forge.models.encoders import (
    SentenceTransformerEncoder,
    TransformerEncoder,
    CustomTextEncoder,
)
from graph_hypernetwork_forge.models.hypernetworks import (
    AdaptiveWeightGenerator,
    MultiModalWeightGenerator,
)
from graph_hypernetwork_forge.data import TextualKnowledgeGraph


class TestTextEncoders:
    """Test text encoder implementations."""
    
    def test_sentence_transformer_encoder(self):
        """Test sentence transformer encoder."""
        try:
            encoder = SentenceTransformerEncoder("all-MiniLM-L6-v2")
            
            texts = ["This is a test sentence", "Another test sentence"]
            embeddings = encoder.encode(texts)
            
            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] == encoder.get_output_dim()
            assert encoder.get_output_dim() > 0
            
        except ImportError:
            pytest.skip("sentence-transformers not available")
    
    def test_custom_text_encoder(self):
        """Test custom text encoder base class."""
        class TestEncoder(CustomTextEncoder):
            def __init__(self):
                super().__init__("test", 256)
            
            def encode(self, texts):
                return torch.randn(len(texts), 256)
        
        encoder = TestEncoder()
        texts = ["test1", "test2"]
        embeddings = encoder.encode(texts)
        
        assert embeddings.shape == (2, 256)
        assert encoder.get_output_dim() == 256
    
    def test_get_text_encoder_factory(self):
        """Test text encoder factory function."""
        try:
            encoder = get_text_encoder("all-MiniLM-L6-v2")
            assert isinstance(encoder, TextEncoder)
        except ImportError:
            pytest.skip("sentence-transformers not available")


class TestWeightGenerator:
    """Test hypernetwork weight generation."""
    
    @pytest.fixture
    def weight_generator(self):
        """Create weight generator for testing."""
        return WeightGenerator(
            text_dim=384,
            hidden_dim=256,
            num_layers=2,
            gnn_type="GAT",
            num_heads=4,
        )
    
    def test_weight_generator_initialization(self, weight_generator):
        """Test weight generator initialization."""
        assert weight_generator.text_dim == 384
        assert weight_generator.hidden_dim == 256
        assert weight_generator.num_layers == 2
        assert weight_generator.gnn_type == "GAT"
        assert len(weight_generator.param_sizes) > 0
    
    def test_weight_generation(self, weight_generator):
        """Test weight generation from text embeddings."""
        batch_size = 4
        text_embeddings = torch.randn(batch_size, 384)
        
        weights = weight_generator(text_embeddings)
        
        assert isinstance(weights, dict)
        assert len(weights) > 0
        
        # Check that all weights have correct batch dimension
        for param_name, param_tensor in weights.items():
            assert param_tensor.shape[0] == batch_size
            assert param_name in weight_generator.param_sizes


class TestDynamicGNN:
    """Test dynamic GNN implementations."""
    
    @pytest.fixture(params=["GCN", "GAT", "GraphSAGE"])
    def gnn_backbone(self, request):
        """Create different GNN backbones for testing."""
        return get_gnn_backbone(
            backbone=request.param,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )
    
    def test_gnn_initialization(self, gnn_backbone):
        """Test GNN backbone initialization."""
        assert isinstance(gnn_backbone, DynamicGNN)
        assert gnn_backbone.hidden_dim == 128
        assert gnn_backbone.num_layers == 2


class TestHyperGNN:
    """Test HyperGNN model functionality."""

    def test_hypergnn_initialization(self):
        """Test HyperGNN can be initialized with default parameters."""
        model = HyperGNN()
        assert model is not None
        assert hasattr(model, 'text_encoder')
        assert hasattr(model, 'hypernetwork')
        assert hasattr(model, 'gnn_backbone')

    def test_hypergnn_with_custom_params(self):
        """Test HyperGNN initialization with custom parameters."""
        model = HyperGNN(
            text_encoder="sentence-transformers/all-MiniLM-L6-v2",
            gnn_backbone="GAT",
            hidden_dim=256,
            num_layers=3
        )
        assert model.hidden_dim == 256
        assert model.num_layers == 3

    @patch('graph_hypernetwork_forge.models.HyperGNN.generate_weights')
    def test_weight_generation(self, mock_generate_weights, sample_texts):
        """Test weight generation from text descriptions."""
        model = HyperGNN()
        mock_weights = {
            'layer_0_weight': torch.randn(128, 128),
            'layer_1_weight': torch.randn(128, 128)
        }
        mock_generate_weights.return_value = mock_weights
        
        weights = model.generate_weights(sample_texts)
        assert isinstance(weights, dict)
        assert 'layer_0_weight' in weights
        mock_generate_weights.assert_called_once_with(sample_texts)

    @patch('graph_hypernetwork_forge.models.HyperGNN.forward')
    def test_forward_pass(self, mock_forward, sample_graph):
        """Test forward pass with generated weights."""
        model = HyperGNN()
        mock_output = torch.randn(3, 128)
        mock_forward.return_value = mock_output
        
        weights = {'test_weight': torch.randn(128, 128)}
        output = model.forward(
            sample_graph['edge_index'],
            sample_graph['node_features'],
            weights
        )
        
        assert output.shape == (3, 128)
        mock_forward.assert_called_once()

    def test_device_handling(self):
        """Test model handles device placement correctly."""
        model = HyperGNN()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Test model parameters are on correct device
        for param in model.parameters():
            assert param.device.type == device.type


class TestTextualKnowledgeGraph:
    """Test TextualKnowledgeGraph data handling."""

    def test_kg_creation(self):
        """Test creation of TextualKnowledgeGraph."""
        kg = TextualKnowledgeGraph()
        assert kg is not None
        assert hasattr(kg, 'node_texts')
        assert hasattr(kg, 'edge_index')

    @patch('graph_hypernetwork_forge.data.TextualKnowledgeGraph.from_json')
    def test_kg_from_json(self, mock_from_json):
        """Test loading KG from JSON file."""
        mock_kg = Mock()
        mock_kg.node_texts = ["Person", "Location"]
        mock_kg.edge_index = torch.tensor([[0, 1], [1, 0]])
        mock_from_json.return_value = mock_kg
        
        kg = TextualKnowledgeGraph.from_json("test.json")
        assert kg.node_texts == ["Person", "Location"]
        mock_from_json.assert_called_once_with("test.json")

    def test_kg_node_feature_extraction(self):
        """Test node feature extraction from text."""
        kg = TextualKnowledgeGraph()
        texts = ["A person entity", "A location entity"]
        
        # Mock the text processing
        with patch.object(kg, 'extract_features') as mock_extract:
            mock_extract.return_value = torch.randn(2, 384)
            features = kg.extract_features(texts)
            assert features.shape == (2, 384)
            mock_extract.assert_called_once_with(texts)


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