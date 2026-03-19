"""Graph Hypernetwork Forge

A hypernetwork that generates GNN weights on-the-fly from text embeddings of
relation types, enabling zero-shot reasoning on unseen knowledge graphs.

Core idea
---------
Instead of learning fixed per-relation weight matrices (which can't generalize
to unseen relations), a *hypernetwork* reads a **text description** of each
relation type and outputs the weight matrices that the GNN should use for that
relation.  At inference time you can handle any relation you can describe in
text — no retraining needed.

Quick start
-----------
>>> from graph_hypernetwork_forge import WeightGenerator, HyperGNN
>>> from graph_hypernetwork_forge.data import ToyKnowledgeGraph
>>> kg = ToyKnowledgeGraph()
>>> model = HyperGNN(text_dim=64, node_feat_dim=16, hidden_dim=32, num_layers=2)
>>> out = model(kg.node_features, kg.edge_index, kg.edge_texts)
>>> out.shape  # (num_nodes, hidden_dim)
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"

from .models.weight_generator import WeightGenerator
from .models.hypergnn import HyperGNN
from .data.knowledge_graph import ToyKnowledgeGraph

__all__ = ["WeightGenerator", "HyperGNN", "ToyKnowledgeGraph"]
