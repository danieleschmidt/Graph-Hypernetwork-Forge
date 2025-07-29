"""Test configuration and fixtures."""
import pytest
import torch


@pytest.fixture
def sample_graph():
    """Sample graph data for testing."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_features = torch.randn(3, 128)
    return {"edge_index": edge_index, "node_features": node_features}


@pytest.fixture
def sample_texts():
    """Sample text descriptions for testing."""
    return ["Node representing a person", "Node representing a location", "Node representing an event"]