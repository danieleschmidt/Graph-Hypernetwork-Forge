"""Graph utility functions for Graph Hypernetwork Forge."""

import torch
import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Union
import torch_geometric
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import networkx as nx


def validate_edge_index(
    edge_index: torch.Tensor, 
    num_nodes: int, 
    allow_self_loops: bool = True
) -> bool:
    """
    Validate edge index tensor format and bounds.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        num_nodes: Total number of nodes in the graph
        allow_self_loops: Whether to allow self-loops (edges from node to itself)
        
    Returns:
        True if edge index is valid, False otherwise
    """
    if not isinstance(edge_index, torch.Tensor):
        return False
    
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        return False
    
    if edge_index.size(1) == 0:  # Empty graph is valid
        return True
    
    # Check if all node indices are within bounds
    if torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes):
        return False
    
    # Check self-loops if not allowed
    if not allow_self_loops:
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        if torch.any(source_nodes == target_nodes):
            return False
    
    return True


def edge_index_to_adjacency(
    edge_index: torch.Tensor, 
    num_nodes: int, 
    edge_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Convert edge index format to dense adjacency matrix.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        num_nodes: Total number of nodes
        edge_weights: Optional edge weights of shape [num_edges]
        
    Returns:
        Dense adjacency matrix of shape [num_nodes, num_nodes]
    """
    if edge_weights is None:
        edge_weights = torch.ones(edge_index.size(1), dtype=torch.float)
    
    adj_matrix = to_dense_adj(
        edge_index, 
        edge_attr=edge_weights,
        max_num_nodes=num_nodes
    ).squeeze(0)
    
    return adj_matrix


def adjacency_to_edge_index(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert dense adjacency matrix to edge index format.
    
    Args:
        adj_matrix: Dense adjacency matrix of shape [num_nodes, num_nodes]
        
    Returns:
        Edge index tensor of shape [2, num_edges]
    """
    edge_index, _ = dense_to_sparse(adj_matrix)
    return edge_index


def calculate_degrees(
    edge_index: torch.Tensor, 
    num_nodes: int, 
    directed: bool = False
) -> torch.Tensor:
    """
    Calculate node degrees from edge index.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        num_nodes: Total number of nodes
        directed: Whether the graph is directed
        
    Returns:
        Node degrees tensor of shape [num_nodes]
    """
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    
    if edge_index.size(1) == 0:
        return degrees
    
    if directed:
        # For directed graphs, count out-degrees
        source_nodes = edge_index[0]
        degrees = degrees.scatter_add_(0, source_nodes, torch.ones_like(source_nodes))
    else:
        # For undirected graphs, count both directions
        all_nodes = edge_index.view(-1)
        degrees = degrees.scatter_add_(0, all_nodes, torch.ones_like(all_nodes))
    
    return degrees


def is_connected(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """
    Check if graph is connected using BFS.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        num_nodes: Total number of nodes
        
    Returns:
        True if graph is connected, False otherwise
    """
    if num_nodes <= 1:
        return True
    
    if edge_index.size(1) == 0:
        return num_nodes <= 1
    
    # Build adjacency list
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[src].append(tgt)
        adj_list[tgt].append(src)  # Treat as undirected
    
    # BFS from node 0
    visited = set()
    queue = [0]
    visited.add(0)
    
    while queue:
        node = queue.pop(0)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return len(visited) == num_nodes


def extract_subgraph(
    edge_index: torch.Tensor,
    node_subset: List[int],
    num_nodes: int,
    relabel_nodes: bool = True
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Extract subgraph containing only specified nodes.
    
    Args:
        edge_index: Original edge index tensor of shape [2, num_edges]
        node_subset: List of node indices to include in subgraph
        num_nodes: Total number of nodes in original graph
        relabel_nodes: Whether to relabel nodes to consecutive integers
        
    Returns:
        Tuple of (subgraph_edge_index, node_mapping)
        node_mapping maps original indices to new indices if relabel_nodes=True
    """
    node_subset_set = set(node_subset)
    
    # Find edges within the subset
    edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        if src in node_subset_set and tgt in node_subset_set:
            edge_mask[i] = True
    
    subgraph_edges = edge_index[:, edge_mask]
    
    if relabel_nodes:
        # Create mapping from old to new indices
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(node_subset))}
        
        # Relabel the edges
        if subgraph_edges.size(1) > 0:
            relabeled_edges = torch.zeros_like(subgraph_edges)
            for i in range(subgraph_edges.size(1)):
                src, tgt = subgraph_edges[0, i].item(), subgraph_edges[1, i].item()
                relabeled_edges[0, i] = node_mapping[src]
                relabeled_edges[1, i] = node_mapping[tgt]
            subgraph_edges = relabeled_edges
    else:
        node_mapping = {idx: idx for idx in node_subset}
    
    return subgraph_edges, node_mapping


def get_node_neighbors(
    edge_index: torch.Tensor, 
    node_idx: int, 
    num_hops: int = 1
) -> Set[int]:
    """
    Get k-hop neighbors of a node.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        node_idx: Target node index
        num_hops: Number of hops for neighborhood (default: 1)
        
    Returns:
        Set of neighbor node indices
    """
    if edge_index.size(1) == 0:
        return set()
    
    # Build adjacency list
    adj_list = {}
    for i in range(edge_index.size(1)):
        src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
        if src not in adj_list:
            adj_list[src] = set()
        if tgt not in adj_list:
            adj_list[tgt] = set()
        adj_list[src].add(tgt)
        adj_list[tgt].add(src)  # Treat as undirected
    
    if node_idx not in adj_list:
        return set()
    
    # BFS for k-hop neighbors
    neighbors = set()
    current_level = {node_idx}
    
    for _ in range(num_hops):
        next_level = set()
        for node in current_level:
            if node in adj_list:
                for neighbor in adj_list[node]:
                    if neighbor != node_idx:  # Exclude the original node
                        neighbors.add(neighbor)
                        next_level.add(neighbor)
        current_level = next_level
    
    return neighbors


def compute_graph_statistics(
    edge_index: torch.Tensor, 
    num_nodes: int
) -> Dict[str, Union[int, float]]:
    """
    Compute various graph statistics.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        num_nodes: Total number of nodes
        
    Returns:
        Dictionary containing graph statistics
    """
    stats = {}
    
    # Basic statistics
    stats['num_nodes'] = num_nodes
    stats['num_edges'] = edge_index.size(1)
    
    if num_nodes > 0:
        stats['density'] = (2 * edge_index.size(1)) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
    else:
        stats['density'] = 0.0
    
    # Degree statistics
    if edge_index.size(1) > 0:
        degrees = calculate_degrees(edge_index, num_nodes)
        stats['avg_degree'] = degrees.float().mean().item()
        stats['max_degree'] = degrees.max().item()
        stats['min_degree'] = degrees.min().item()
        stats['degree_std'] = degrees.float().std().item()
    else:
        stats['avg_degree'] = 0.0
        stats['max_degree'] = 0
        stats['min_degree'] = 0
        stats['degree_std'] = 0.0
    
    # Connectivity
    stats['is_connected'] = is_connected(edge_index, num_nodes)
    
    return stats


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Add self-loops to all nodes in the graph.
    
    Args:
        edge_index: Original edge index tensor of shape [2, num_edges]
        num_nodes: Total number of nodes
        
    Returns:
        Edge index with self-loops added
    """
    # Create self-loop edges
    self_loops = torch.arange(num_nodes, dtype=edge_index.dtype, device=edge_index.device)
    self_loop_edges = torch.stack([self_loops, self_loops], dim=0)
    
    # Remove existing self-loops to avoid duplicates
    edge_mask = edge_index[0] != edge_index[1]
    filtered_edges = edge_index[:, edge_mask]
    
    # Concatenate original edges (without self-loops) with new self-loops
    edge_index_with_loops = torch.cat([filtered_edges, self_loop_edges], dim=1)
    
    return edge_index_with_loops


def remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Remove self-loops from the graph.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        
    Returns:
        Edge index without self-loops
    """
    edge_mask = edge_index[0] != edge_index[1]
    return edge_index[:, edge_mask]


def to_networkx(edge_index: torch.Tensor, num_nodes: int, directed: bool = False) -> nx.Graph:
    """
    Convert edge index to NetworkX graph.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        num_nodes: Total number of nodes
        directed: Whether to create directed graph
        
    Returns:
        NetworkX graph object
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Add all nodes
    G.add_nodes_from(range(num_nodes))
    
    # Add edges
    if edge_index.size(1) > 0:
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
    
    return G


def from_networkx(G: nx.Graph) -> Tuple[torch.Tensor, int]:
    """
    Convert NetworkX graph to edge index format.
    
    Args:
        G: NetworkX graph object
        
    Returns:
        Tuple of (edge_index, num_nodes)
    """
    num_nodes = G.number_of_nodes()
    
    if G.number_of_edges() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edges = list(G.edges())
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    return edge_index, num_nodes