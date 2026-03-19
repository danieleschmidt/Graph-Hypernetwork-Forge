"""Toy knowledge graph for demos and tests.

A small 8-node KG covering a mini family + colleague network::

    Nodes
    -----
    0  Alice    (person)
    1  Bob      (person)
    2  Carol    (person)
    3  Dave     (person)
    4  Eve      (person)
    5  Acme Corp  (organisation)
    6  London     (city)
    7  Python     (skill)

    Edges (directed)
    ----------------
    (0, 1)  "is spouse of"
    (1, 0)  "is spouse of"
    (0, 2)  "knows"
    (1, 3)  "works with"
    (2, 3)  "knows"
    (3, 5)  "works at"
    (0, 5)  "works at"
    (5, 6)  "located in"
    (0, 7)  "has skill"
    (3, 7)  "has skill"
    (2, 4)  "is parent of"

Node features are random fixed vectors (seeded for reproducibility).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class ToyKnowledgeGraph:
    """A small in-memory knowledge graph for experimentation.

    Attributes:
        node_names:    Human-readable node labels.
        node_features: ``[N, feat_dim]`` feature matrix.
        edge_index:    ``[2, E]`` edge connectivity (source, target).
        edge_texts:    Relation type string for each edge.
        feat_dim:      Feature dimension (default 16).
    """

    feat_dim: int = 16

    node_names: List[str] = field(default_factory=lambda: [
        "Alice", "Bob", "Carol", "Dave", "Eve", "Acme Corp", "London", "Python",
    ])

    edge_data: List[tuple] = field(default_factory=lambda: [
        (0, 1, "is spouse of"),
        (1, 0, "is spouse of"),
        (0, 2, "knows"),
        (1, 3, "works with"),
        (2, 3, "knows"),
        (3, 5, "works at"),
        (0, 5, "works at"),
        (5, 6, "located in"),
        (0, 7, "has skill"),
        (3, 7, "has skill"),
        (2, 4, "is parent of"),
    ])

    def __post_init__(self) -> None:
        # Fixed random features (seeded for reproducibility)
        gen = torch.Generator()
        gen.manual_seed(42)
        self.node_features: torch.Tensor = torch.randn(
            len(self.node_names), self.feat_dim, generator=gen
        )

        srcs = [e[0] for e in self.edge_data]
        dsts = [e[1] for e in self.edge_data]
        self.edge_index: torch.Tensor = torch.tensor(
            [srcs, dsts], dtype=torch.long
        )
        self.edge_texts: List[str] = [e[2] for e in self.edge_data]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)

    @property
    def relation_types(self) -> List[str]:
        return list(dict.fromkeys(self.edge_texts))

    def __repr__(self) -> str:
        return (
            f"ToyKnowledgeGraph(nodes={self.num_nodes}, "
            f"edges={self.num_edges}, "
            f"relation_types={len(self.relation_types)})"
        )
