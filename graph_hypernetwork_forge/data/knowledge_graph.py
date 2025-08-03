"""Textual Knowledge Graph data structure and utilities."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class TextualKnowledgeGraph:
    """A knowledge graph with textual metadata for each node.
    
    This class represents a knowledge graph where each node has associated
    textual descriptions that can be used for dynamic weight generation.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        node_texts: List[str],
        edge_types: Optional[List[str]] = None,
        node_features: Optional[torch.Tensor] = None,
        edge_attributes: Optional[torch.Tensor] = None,
        node_labels: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
    ):
        """Initialize the textual knowledge graph.
        
        Args:
            edge_index: Edge connectivity in COO format [2, num_edges]
            node_texts: List of textual descriptions for each node
            edge_types: Optional edge type labels
            node_features: Optional node feature matrix [num_nodes, feature_dim]
            edge_attributes: Optional edge attributes [num_edges, edge_attr_dim]
            node_labels: Optional node labels for supervised learning
            metadata: Additional metadata dictionary
        """
        self.edge_index = edge_index
        self.node_texts = node_texts
        self.num_nodes = len(node_texts)
        self.num_edges = edge_index.size(1)
        
        # Optional attributes
        self.edge_types = edge_types or [f"edge_{i}" for i in range(self.num_edges)]
        self.node_features = node_features
        self.edge_attributes = edge_attributes
        self.node_labels = node_labels
        self.metadata = metadata or {}
        
        # Validate consistency
        self._validate()

    def _validate(self) -> None:
        """Validate the consistency of the knowledge graph data."""
        # Check edge index bounds
        max_node_idx = self.edge_index.max().item()
        if max_node_idx >= self.num_nodes:
            raise ValueError(
                f"Edge index contains node {max_node_idx} but only "
                f"{self.num_nodes} nodes exist"
            )
        
        # Check optional tensors dimensions
        if self.node_features is not None:
            if self.node_features.size(0) != self.num_nodes:
                raise ValueError("Node features size mismatch")
        
        if self.edge_attributes is not None:
            if self.edge_attributes.size(0) != self.num_edges:
                raise ValueError("Edge attributes size mismatch")
        
        if self.node_labels is not None:
            if self.node_labels.size(0) != self.num_nodes:
                raise ValueError("Node labels size mismatch")
        
        if len(self.edge_types) != self.num_edges:
            raise ValueError("Edge types count mismatch")

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "TextualKnowledgeGraph":
        """Load knowledge graph from JSON file.
        
        Expected JSON format:
        {
            "nodes": [
                {"id": 0, "text": "Description of node 0", "features": [...], "label": 0},
                ...
            ],
            "edges": [
                {"source": 0, "target": 1, "type": "relation_type", "attributes": [...]},
                ...
            ],
            "metadata": {"domain": "example", "version": "1.0"}
        }
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            TextualKnowledgeGraph instance
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract nodes
        nodes = sorted(data["nodes"], key=lambda x: x["id"])
        node_texts = [node["text"] for node in nodes]
        
        # Extract node features if present
        node_features = None
        if "features" in nodes[0]:
            features_list = [node.get("features", []) for node in nodes]
            if any(features_list):
                node_features = torch.tensor(features_list, dtype=torch.float32)
        
        # Extract node labels if present
        node_labels = None
        if "label" in nodes[0]:
            labels_list = [node.get("label", -1) for node in nodes]
            node_labels = torch.tensor(labels_list, dtype=torch.long)
        
        # Extract edges
        edges = data["edges"]
        edge_list = [[edge["source"], edge["target"]] for edge in edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Extract edge types
        edge_types = [edge.get("type", "unknown") for edge in edges]
        
        # Extract edge attributes if present
        edge_attributes = None
        if "attributes" in edges[0]:
            attr_list = [edge.get("attributes", []) for edge in edges]
            if any(attr_list):
                edge_attributes = torch.tensor(attr_list, dtype=torch.float32)
        
        # Extract metadata
        metadata = data.get("metadata", {})
        
        return cls(
            edge_index=edge_index,
            node_texts=node_texts,
            edge_types=edge_types,
            node_features=node_features,
            edge_attributes=edge_attributes,
            node_labels=node_labels,
            metadata=metadata,
        )

    @classmethod
    def from_networkx(
        cls, 
        graph: nx.Graph, 
        text_attr: str = "text",
        feature_attrs: Optional[List[str]] = None,
        label_attr: Optional[str] = None,
    ) -> "TextualKnowledgeGraph":
        """Create knowledge graph from NetworkX graph.
        
        Args:
            graph: NetworkX graph with node attributes
            text_attr: Node attribute containing text descriptions
            feature_attrs: List of node attributes to use as features
            label_attr: Node attribute containing labels
            
        Returns:
            TextualKnowledgeGraph instance
        """
        # Ensure nodes are integer indexed starting from 0
        node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        relabeled_graph = nx.relabel_nodes(graph, node_mapping)
        
        # Extract node texts
        node_texts = []
        for i in range(len(relabeled_graph.nodes())):
            node_data = relabeled_graph.nodes[i]
            node_texts.append(node_data.get(text_attr, f"Node {i}"))
        
        # Extract node features
        node_features = None
        if feature_attrs:
            features_matrix = []
            for i in range(len(relabeled_graph.nodes())):
                node_data = relabeled_graph.nodes[i]
                features = [node_data.get(attr, 0.0) for attr in feature_attrs]
                features_matrix.append(features)
            node_features = torch.tensor(features_matrix, dtype=torch.float32)
        
        # Extract node labels
        node_labels = None
        if label_attr:
            labels = []
            for i in range(len(relabeled_graph.nodes())):
                node_data = relabeled_graph.nodes[i]
                labels.append(node_data.get(label_attr, -1))
            node_labels = torch.tensor(labels, dtype=torch.long)
        
        # Extract edges
        edge_list = list(relabeled_graph.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Extract edge types
        edge_types = []
        for edge in relabeled_graph.edges():
            edge_data = relabeled_graph.edges[edge]
            edge_types.append(edge_data.get("type", "unknown"))
        
        return cls(
            edge_index=edge_index,
            node_texts=node_texts,
            edge_types=edge_types,
            node_features=node_features,
            node_labels=node_labels,
        )

    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object.
        
        Returns:
            PyTorch Geometric Data object
        """
        data = Data(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
        )
        
        if self.node_features is not None:
            data.x = self.node_features
        
        if self.edge_attributes is not None:
            data.edge_attr = self.edge_attributes
        
        if self.node_labels is not None:
            data.y = self.node_labels
        
        return data

    def subgraph(self, node_indices: List[int]) -> "TextualKnowledgeGraph":
        """Extract subgraph containing specified nodes.
        
        Args:
            node_indices: List of node indices to include
            
        Returns:
            New TextualKnowledgeGraph containing subgraph
        """
        node_set = set(node_indices)
        
        # Create node mapping
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Extract subgraph edges
        edge_mask = torch.tensor([
            (edge[0].item() in node_set and edge[1].item() in node_set)
            for edge in self.edge_index.t()
        ])
        
        sub_edges = self.edge_index[:, edge_mask]
        # Remap edge indices
        remapped_edges = torch.tensor([
            [old_to_new[edge[0].item()], old_to_new[edge[1].item()]]
            for edge in sub_edges.t()
        ]).t().contiguous()
        
        # Extract corresponding data
        sub_node_texts = [self.node_texts[i] for i in node_indices]
        sub_edge_types = [self.edge_types[i] for i, mask in enumerate(edge_mask) if mask]
        
        sub_node_features = None
        if self.node_features is not None:
            sub_node_features = self.node_features[node_indices]
        
        sub_edge_attributes = None
        if self.edge_attributes is not None:
            sub_edge_attributes = self.edge_attributes[edge_mask]
        
        sub_node_labels = None
        if self.node_labels is not None:
            sub_node_labels = self.node_labels[node_indices]
        
        return TextualKnowledgeGraph(
            edge_index=remapped_edges,
            node_texts=sub_node_texts,
            edge_types=sub_edge_types,
            node_features=sub_node_features,
            edge_attributes=sub_edge_attributes,
            node_labels=sub_node_labels,
            metadata=self.metadata.copy(),
        )

    def get_neighbor_texts(self, node_idx: int, k_hops: int = 1) -> List[str]:
        """Get text descriptions of k-hop neighbors.
        
        Args:
            node_idx: Target node index
            k_hops: Number of hops to consider
            
        Returns:
            List of text descriptions from neighbors
        """
        # Convert to NetworkX for easy traversal
        nx_graph = nx.Graph()
        edges = self.edge_index.t().tolist()
        nx_graph.add_edges_from(edges)
        
        # Find k-hop neighbors
        neighbors = set([node_idx])
        current_nodes = set([node_idx])
        
        for _ in range(k_hops):
            next_nodes = set()
            for node in current_nodes:
                if node in nx_graph:
                    next_nodes.update(nx_graph.neighbors(node))
            neighbors.update(next_nodes)
            current_nodes = next_nodes
        
        # Remove the original node
        neighbors.discard(node_idx)
        
        return [self.node_texts[i] for i in sorted(neighbors)]

    def save(self, file_path: Union[str, Path]) -> None:
        """Save knowledge graph to pickle file.
        
        Args:
            file_path: Output file path
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "TextualKnowledgeGraph":
        """Load knowledge graph from pickle file.
        
        Args:
            file_path: Input file path
            
        Returns:
            TextualKnowledgeGraph instance
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def statistics(self) -> Dict:
        """Get basic statistics of the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        # Convert to networkx for analysis
        nx_graph = nx.Graph()
        edges = self.edge_index.t().tolist()
        nx_graph.add_edges_from(edges)
        
        # Calculate statistics
        stats = {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "avg_degree": 2 * self.num_edges / self.num_nodes if self.num_nodes > 0 else 0,
            "density": nx.density(nx_graph) if self.num_nodes > 1 else 0,
            "num_connected_components": nx.number_connected_components(nx_graph),
            "avg_text_length": np.mean([len(text.split()) for text in self.node_texts]),
            "unique_edge_types": len(set(self.edge_types)),
            "metadata": self.metadata,
        }
        
        if self.node_features is not None:
            stats["node_feature_dim"] = self.node_features.size(1)
            
        if self.edge_attributes is not None:
            stats["edge_attr_dim"] = self.edge_attributes.size(1)
            
        if self.node_labels is not None:
            stats["num_classes"] = len(torch.unique(self.node_labels))
        
        return stats

    def __repr__(self) -> str:
        """String representation of the knowledge graph."""
        return (
            f"TextualKnowledgeGraph("
            f"num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"edge_types={len(set(self.edge_types))}, "
            f"domain={self.metadata.get('domain', 'unknown')}"
            f")"
        )