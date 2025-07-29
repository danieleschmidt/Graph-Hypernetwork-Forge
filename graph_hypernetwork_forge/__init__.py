"""Graph Hypernetwork Forge

A hypernetwork that generates GNN weights on-the-fly from textual metadata,
enabling zero-shot reasoning on unseen knowledge graphs.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .models import HyperGNN
from .data import TextualKnowledgeGraph

__all__ = ["HyperGNN", "TextualKnowledgeGraph"]