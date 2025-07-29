"""Graph Hypernetwork Forge: Zero-Shot GNN Weight Generation from Text."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "hypernetwork-forge@yourdomain.com"

# Core API exports
from .models import HyperGNN
from .data import TextualKnowledgeGraph

__all__ = [
    "HyperGNN",
    "TextualKnowledgeGraph",
]