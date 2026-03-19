"""Shared pytest fixtures for Graph Hypernetwork Forge tests."""

import pytest
import torch

from graph_hypernetwork_forge import WeightGenerator, HyperGNN, ToyKnowledgeGraph


@pytest.fixture(scope="session")
def toy_kg():
    """Return a ToyKnowledgeGraph instance (session-scoped for speed)."""
    return ToyKnowledgeGraph(feat_dim=16)


@pytest.fixture
def small_model():
    """Tiny HyperGNN for fast tests."""
    return HyperGNN(
        text_dim=32,
        node_feat_dim=16,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )


@pytest.fixture
def weight_gen():
    """WeightGenerator with small dims for fast tests."""
    return WeightGenerator(text_dim=32, d_in=16, d_out=16, hidden_dim=64)
