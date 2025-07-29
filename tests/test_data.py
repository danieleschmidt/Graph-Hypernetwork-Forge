"""Tests for data loading and processing."""

import pytest
import torch
from graph_hypernetwork_forge.data import TextualKnowledgeGraph


class TestTextualKnowledgeGraph:
    """Test cases for TextualKnowledgeGraph."""
    
    def test_initialization(self):
        """Test knowledge graph initialization."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        node_features = torch.randn(3, 128)
        node_texts = {0: "node zero", 1: "node one", 2: "node two"}
        
        kg = TextualKnowledgeGraph(edge_index, node_features, node_texts)
        
        assert torch.equal(kg.edge_index, edge_index)
        assert torch.equal(kg.node_features, node_features)
        assert kg.node_texts == node_texts
        assert kg.num_nodes == 3
        assert kg.num_edges == 3
    
    def test_from_json_not_implemented(self):
        """Test that from_json raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            TextualKnowledgeGraph.from_json("test.json")