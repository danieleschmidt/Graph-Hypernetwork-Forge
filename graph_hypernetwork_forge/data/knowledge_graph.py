"""Knowledge Graph data structures and processing utilities.

Handles loading, processing, and managing textual knowledge graphs
for zero-shot hypernetwork training and inference.
"""

import json
import torch
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a knowledge graph node."""
    id: Union[str, int]
    text: str
    features: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EdgeInfo:
    """Information about a knowledge graph edge."""
    source: Union[str, int]
    target: Union[str, int]
    relation: str
    features: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class TextualKnowledgeGraph:
    """Knowledge graph with textual node descriptions.
    
    Supports various input formats and provides utilities for
    processing graphs for hypernetwork training and inference.
    """
    
    def __init__(
        self,
        nodes: List[NodeInfo],
        edges: List[EdgeInfo],
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.nodes = {node.id: node for node in nodes}
        self.edges = edges
        self.name = name or "Unnamed Graph"
        self.metadata = metadata or {}
        
        # Build NetworkX graph for analysis
        self._build_networkx_graph()
        
        # Cache computed properties
        self._edge_index = None
        self._node_texts = None
        self._node_features = None
        self._edge_features = None
        
        logger.info(f"Created KG '{self.name}' with {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    def _build_networkx_graph(self):
        """Build NetworkX graph for analysis."""
        self.nx_graph = nx.DiGraph()
        
        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            self.nx_graph.add_node(
                node_id,
                text=node.text,
                metadata=node.metadata or {}
            )
        
        # Add edges with attributes
        for edge in self.edges:
            self.nx_graph.add_edge(
                edge.source,
                edge.target,
                relation=edge.relation,
                metadata=edge.metadata or {}
            )
    
    @property
    def edge_index(self) -> torch.Tensor:
        """Get edge index tensor for PyTorch Geometric."""
        if self._edge_index is None:
            # Create node ID to index mapping
            node_ids = list(self.nodes.keys())
            id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
            
            # Convert edges to tensor format
            edge_list = []
            for edge in self.edges:
                src_idx = id_to_idx[edge.source]
                dst_idx = id_to_idx[edge.target]
                edge_list.append([src_idx, dst_idx])
            
            if edge_list:
                self._edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                self._edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return self._edge_index
    
    @property
    def node_texts(self) -> List[str]:
        """Get ordered list of node texts."""
        if self._node_texts is None:
            node_ids = sorted(self.nodes.keys())
            self._node_texts = [self.nodes[node_id].text for node_id in node_ids]
        return self._node_texts
    
    @property
    def node_features(self) -> torch.Tensor:
        """Get node feature matrix."""
        if self._node_features is None:
            node_ids = sorted(self.nodes.keys())
            features = []
            
            for node_id in node_ids:
                node = self.nodes[node_id]
                if node.features is not None:
                    features.append(node.features)
                else:
                    # Create default features (one-hot or random)
                    features.append(torch.randn(256))  # Default dimension
            
            self._node_features = torch.stack(features)
        
        return self._node_features
    
    @property
    def edge_features(self) -> Optional[torch.Tensor]:
        """Get edge feature matrix if available."""
        if self._edge_features is None and self.edges:
            features = []
            for edge in self.edges:
                if edge.features is not None:
                    features.append(edge.features)
                else:
                    # Create default edge features
                    features.append(torch.zeros(64))  # Default edge dimension
            
            if features:
                self._edge_features = torch.stack(features)
        
        return self._edge_features
    
    def get_subgraph(
        self,
        node_ids: List[Union[str, int]],
        include_neighbors: bool = False,
        hop_distance: int = 1,
    ) -> 'TextualKnowledgeGraph':
        """Extract subgraph containing specified nodes.
        
        Args:
            node_ids: List of node IDs to include
            include_neighbors: Whether to include neighboring nodes
            hop_distance: Distance for neighbor inclusion
            
        Returns:
            New TextualKnowledgeGraph with subgraph
        """
        if include_neighbors:
            # Expand to include neighbors
            expanded_nodes = set(node_ids)
            for _ in range(hop_distance):
                new_nodes = set()
                for node_id in expanded_nodes:
                    if node_id in self.nx_graph:
                        new_nodes.update(self.nx_graph.neighbors(node_id))
                        new_nodes.update(self.nx_graph.predecessors(node_id))
                expanded_nodes.update(new_nodes)
            node_ids = list(expanded_nodes)
        
        # Filter nodes and edges
        filtered_nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        filtered_edges = [
            edge for edge in self.edges 
            if edge.source in node_ids and edge.target in node_ids
        ]
        
        return TextualKnowledgeGraph(
            nodes=filtered_nodes,
            edges=filtered_edges,
            name=f"{self.name}_subgraph",
            metadata=self.metadata.copy(),
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'avg_degree': len(self.edges) * 2 / len(self.nodes) if self.nodes else 0,
            'is_connected': nx.is_connected(self.nx_graph.to_undirected()),
            'num_components': nx.number_connected_components(self.nx_graph.to_undirected()),
            'avg_text_length': np.mean([len(text.split()) for text in self.node_texts]),
            'relations': list(set(edge.relation for edge in self.edges)),
        }
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save knowledge graph to JSON format."""
        data = {
            'name': self.name,
            'metadata': self.metadata,
            'nodes': [
                {
                    'id': node.id,
                    'text': node.text,
                    'features': node.features.tolist() if node.features is not None else None,
                    'metadata': node.metadata,
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'relation': edge.relation,
                    'features': edge.features.tolist() if edge.features is not None else None,
                    'metadata': edge.metadata,
                }
                for edge in self.edges
            ]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved knowledge graph to {path}")
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'TextualKnowledgeGraph':
        """Load knowledge graph from JSON format."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse nodes
        nodes = []
        for node_data in data['nodes']:
            features = None
            if node_data.get('features'):
                features = torch.tensor(node_data['features'])
            
            nodes.append(NodeInfo(
                id=node_data['id'],
                text=node_data['text'],
                features=features,
                metadata=node_data.get('metadata'),
            ))
        
        # Parse edges
        edges = []
        for edge_data in data['edges']:
            features = None
            if edge_data.get('features'):
                features = torch.tensor(edge_data['features'])
            
            edges.append(EdgeInfo(
                source=edge_data['source'],
                target=edge_data['target'],
                relation=edge_data['relation'],
                features=features,
                metadata=edge_data.get('metadata'),
            ))
        
        return cls(
            nodes=nodes,
            edges=edges,
            name=data.get('name'),
            metadata=data.get('metadata'),
        )
    
    @classmethod
    def from_triples(
        cls,
        triples: List[Tuple[str, str, str]],
        node_texts: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> 'TextualKnowledgeGraph':
        """Create knowledge graph from triples format.
        
        Args:
            triples: List of (subject, predicate, object) triples
            node_texts: Optional mapping of entity IDs to text descriptions
            name: Graph name
            
        Returns:
            TextualKnowledgeGraph instance
        """
        # Extract unique entities
        entities = set()
        for subj, pred, obj in triples:
            entities.add(subj)
            entities.add(obj)
        
        # Create nodes
        nodes = []
        for entity in entities:
            text = node_texts.get(entity, entity) if node_texts else entity
            nodes.append(NodeInfo(id=entity, text=text))
        
        # Create edges
        edges = []
        for subj, pred, obj in triples:
            edges.append(EdgeInfo(source=subj, target=obj, relation=pred))
        
        return cls(nodes=nodes, edges=edges, name=name)
    
    @classmethod
    def from_networkx(
        cls,
        G: nx.Graph,
        text_attr: str = 'text',
        relation_attr: str = 'relation',
        name: Optional[str] = None,
    ) -> 'TextualKnowledgeGraph':
        """Create knowledge graph from NetworkX graph.
        
        Args:
            G: NetworkX graph
            text_attr: Node attribute containing text
            relation_attr: Edge attribute containing relation
            name: Graph name
            
        Returns:
            TextualKnowledgeGraph instance
        """
        # Extract nodes
        nodes = []
        for node_id, node_data in G.nodes(data=True):
            text = node_data.get(text_attr, str(node_id))
            metadata = {k: v for k, v in node_data.items() if k != text_attr}
            nodes.append(NodeInfo(
                id=node_id,
                text=text,
                metadata=metadata if metadata else None,
            ))
        
        # Extract edges
        edges = []
        for src, dst, edge_data in G.edges(data=True):
            relation = edge_data.get(relation_attr, 'connected')
            metadata = {k: v for k, v in edge_data.items() if k != relation_attr}
            edges.append(EdgeInfo(
                source=src,
                target=dst,
                relation=relation,
                metadata=metadata if metadata else None,
            ))
        
        return cls(nodes=nodes, edges=edges, name=name)
    
    @classmethod
    def from_csv(
        cls,
        nodes_file: Union[str, Path],
        edges_file: Union[str, Path],
        node_id_col: str = 'id',
        node_text_col: str = 'text',
        edge_src_col: str = 'source',
        edge_dst_col: str = 'target',
        edge_rel_col: str = 'relation',
        name: Optional[str] = None,
    ) -> 'TextualKnowledgeGraph':
        """Create knowledge graph from CSV files.
        
        Args:
            nodes_file: Path to nodes CSV file
            edges_file: Path to edges CSV file
            node_id_col: Column name for node IDs
            node_text_col: Column name for node text
            edge_src_col: Column name for edge source
            edge_dst_col: Column name for edge target
            edge_rel_col: Column name for edge relation
            name: Graph name
            
        Returns:
            TextualKnowledgeGraph instance
        """
        # Load nodes
        nodes_df = pd.read_csv(nodes_file)
        nodes = []
        for _, row in nodes_df.iterrows():
            metadata = {
                col: row[col] for col in nodes_df.columns 
                if col not in [node_id_col, node_text_col]
            }
            nodes.append(NodeInfo(
                id=row[node_id_col],
                text=row[node_text_col],
                metadata=metadata if metadata else None,
            ))
        
        # Load edges
        edges_df = pd.read_csv(edges_file)
        edges = []
        for _, row in edges_df.iterrows():
            metadata = {
                col: row[col] for col in edges_df.columns 
                if col not in [edge_src_col, edge_dst_col, edge_rel_col]
            }
            edges.append(EdgeInfo(
                source=row[edge_src_col],
                target=row[edge_dst_col],
                relation=row[edge_rel_col],
                metadata=metadata if metadata else None,
            ))
        
        return cls(nodes=nodes, edges=edges, name=name)


def create_synthetic_kg(
    num_nodes: int = 100,
    num_edges: int = 200,
    relations: List[str] = None,
    random_seed: int = 42,
) -> TextualKnowledgeGraph:
    """Create synthetic knowledge graph for testing.
    
    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges
        relations: List of possible relations
        random_seed: Random seed for reproducibility
        
    Returns:
        Synthetic TextualKnowledgeGraph
    """
    if relations is None:
        relations = ['related_to', 'part_of', 'instance_of', 'similar_to']
    
    np.random.seed(random_seed)
    
    # Generate nodes with synthetic text
    nodes = []
    for i in range(num_nodes):
        # Create synthetic text description
        topics = ['technology', 'science', 'art', 'business', 'education']
        adjectives = ['innovative', 'advanced', 'creative', 'important', 'fundamental']
        
        topic = np.random.choice(topics)
        adj = np.random.choice(adjectives)
        text = f"This is an {adj} concept in {topic} with identifier {i}"
        
        nodes.append(NodeInfo(id=f"node_{i}", text=text))
    
    # Generate edges
    edges = []
    for _ in range(num_edges):
        src = f"node_{np.random.randint(0, num_nodes)}"
        dst = f"node_{np.random.randint(0, num_nodes)}"
        rel = np.random.choice(relations)
        
        if src != dst:  # Avoid self-loops
            edges.append(EdgeInfo(source=src, target=dst, relation=rel))
    
    return TextualKnowledgeGraph(
        nodes=nodes,
        edges=edges,
        name="Synthetic Knowledge Graph",
        metadata={'synthetic': True, 'seed': random_seed}
    )