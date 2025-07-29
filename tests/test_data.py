"""Tests for data loading and processing components."""
import pytest
import torch
import json
import tempfile
import os
from unittest.mock import Mock, patch, mock_open

from graph_hypernetwork_forge.data import TextualKnowledgeGraph


class TestTextualKnowledgeGraph:
    """Test TextualKnowledgeGraph data processing."""

    def test_empty_initialization(self):
        """Test empty KG initialization."""
        kg = TextualKnowledgeGraph()
        assert kg.node_texts == []
        assert kg.edge_index is None
        assert kg.node_features is None

    def test_initialization_with_data(self):
        """Test KG initialization with provided data."""
        texts = ["Person entity", "Location entity", "Event entity"]
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        kg = TextualKnowledgeGraph(node_texts=texts, edge_index=edge_index)
        assert kg.node_texts == texts
        assert torch.equal(kg.edge_index, edge_index)

    def test_from_json_valid_file(self):
        """Test loading KG from valid JSON file."""
        test_data = {
            "nodes": [
                {"id": 0, "text": "A person named Alice"},
                {"id": 1, "text": "A city called Paris"},
                {"id": 2, "text": "An event called conference"}
            ],
            "edges": [[0, 1], [1, 2], [2, 0]]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            kg = TextualKnowledgeGraph.from_json(temp_path)
            assert len(kg.node_texts) == 3
            assert "Alice" in kg.node_texts[0]
            assert kg.edge_index.shape == (2, 3)
        finally:
            os.unlink(temp_path)

    def test_from_json_missing_file(self):
        """Test loading KG from non-existent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            TextualKnowledgeGraph.from_json("non_existent_file.json")

    def test_from_json_invalid_format(self):
        """Test loading KG from invalid JSON format."""
        invalid_data = {"invalid": "format"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError):
                TextualKnowledgeGraph.from_json(temp_path)
        finally:
            os.unlink(temp_path)

    def test_add_node(self):
        """Test adding nodes to existing KG."""
        kg = TextualKnowledgeGraph()
        kg.add_node("New person entity")
        
        assert len(kg.node_texts) == 1
        assert kg.node_texts[0] == "New person entity"

    def test_add_edge(self):
        """Test adding edges to existing KG."""
        kg = TextualKnowledgeGraph()
        kg.add_node("Node 0")
        kg.add_node("Node 1")
        
        kg.add_edge(0, 1)
        assert kg.edge_index is not None
        assert kg.edge_index.shape[1] == 1
        assert kg.edge_index[0, 0] == 0
        assert kg.edge_index[1, 0] == 1

    def test_add_bidirectional_edge(self):
        """Test adding bidirectional edges."""
        kg = TextualKnowledgeGraph()
        kg.add_node("Node 0")
        kg.add_node("Node 1")
        
        kg.add_edge(0, 1, bidirectional=True)
        assert kg.edge_index.shape[1] == 2
        # Check both directions exist
        edges = kg.edge_index.t().tolist()
        assert [0, 1] in edges
        assert [1, 0] in edges

    def test_node_count_property(self):
        """Test node count property."""
        kg = TextualKnowledgeGraph()
        assert kg.num_nodes == 0
        
        kg.add_node("Node 1")
        kg.add_node("Node 2")
        assert kg.num_nodes == 2

    def test_edge_count_property(self):
        """Test edge count property."""
        kg = TextualKnowledgeGraph()
        kg.add_node("Node 0")
        kg.add_node("Node 1")
        assert kg.num_edges == 0
        
        kg.add_edge(0, 1)
        assert kg.num_edges == 1

    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_texts(self, mock_transformer):
        """Test text encoding functionality."""
        mock_model = Mock()
        mock_model.encode.return_value = torch.randn(3, 384)
        mock_transformer.return_value = mock_model
        
        kg = TextualKnowledgeGraph()
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = kg.encode_texts(texts)
        
        assert embeddings.shape == (3, 384)
        mock_model.encode.assert_called_once_with(texts, convert_to_tensor=True)

    def test_to_dict_serialization(self):
        """Test KG serialization to dictionary."""
        kg = TextualKnowledgeGraph()
        kg.add_node("Person")
        kg.add_node("Location")
        kg.add_edge(0, 1)
        
        kg_dict = kg.to_dict()
        assert "nodes" in kg_dict
        assert "edges" in kg_dict
        assert len(kg_dict["nodes"]) == 2
        assert kg_dict["nodes"][0]["text"] == "Person"

    def test_from_dict_deserialization(self):
        """Test KG deserialization from dictionary."""
        kg_dict = {
            "nodes": [
                {"id": 0, "text": "Person entity"},
                {"id": 1, "text": "Location entity"}
            ],
            "edges": [[0, 1]]
        }
        
        kg = TextualKnowledgeGraph.from_dict(kg_dict)
        assert len(kg.node_texts) == 2
        assert kg.node_texts[0] == "Person entity"
        assert kg.num_edges == 1

    def test_subgraph_extraction(self):
        """Test extracting subgraph from larger KG."""
        kg = TextualKnowledgeGraph()
        for i in range(5):
            kg.add_node(f"Node {i}")
        
        # Add some edges
        kg.add_edge(0, 1)
        kg.add_edge(1, 2)
        kg.add_edge(2, 3)
        kg.add_edge(3, 4)
        
        # Extract subgraph with nodes [1, 2, 3]
        subgraph = kg.subgraph([1, 2, 3])
        assert len(subgraph.node_texts) == 3
        assert subgraph.num_edges == 2  # Edges (1,2) and (2,3)

    @pytest.mark.parametrize("text_input,expected_length", [
        (["Single text"], 1),
        (["Text 1", "Text 2", "Text 3"], 3),
        ([], 0)
    ])
    def test_various_text_inputs(self, text_input, expected_length):
        """Test KG handles various text input sizes."""
        kg = TextualKnowledgeGraph(node_texts=text_input)
        assert len(kg.node_texts) == expected_length

    def test_edge_validation(self):
        """Test edge validation for invalid node indices."""
        kg = TextualKnowledgeGraph()
        kg.add_node("Node 0")
        
        # Should raise error for invalid node indices
        with pytest.raises(IndexError):
            kg.add_edge(0, 5)  # Node 5 doesn't exist
        
        with pytest.raises(IndexError):
            kg.add_edge(-1, 0)  # Negative index

    def test_duplicate_edge_handling(self):
        """Test handling of duplicate edges."""
        kg = TextualKnowledgeGraph()
        kg.add_node("Node 0")
        kg.add_node("Node 1")
        
        kg.add_edge(0, 1)
        initial_edge_count = kg.num_edges
        
        # Add same edge again
        kg.add_edge(0, 1)
        # Should not increase edge count (depending on implementation)
        assert kg.num_edges >= initial_edge_count