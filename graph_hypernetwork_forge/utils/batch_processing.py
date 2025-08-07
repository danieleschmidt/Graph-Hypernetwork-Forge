"""Efficient batch processing utilities for large graphs."""

import torch
import numpy as np
from typing import List, Tuple, Dict, Iterator, Optional, Any, Union
from dataclasses import dataclass
import math
from torch_geometric.data import Data, Batch
from ..data.knowledge_graph import TextualKnowledgeGraph


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 32
    max_nodes_per_batch: int = 1000
    max_edges_per_batch: int = 5000
    memory_limit_mb: float = 1024.0
    overlap_ratio: float = 0.1  # For sliding window batching


class GraphBatcher:
    """Efficient batching for large knowledge graphs."""
    
    def __init__(self, config: BatchConfig = None):
        """Initialize graph batcher.
        
        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig()
    
    def create_node_batches(
        self, 
        kg: TextualKnowledgeGraph,
        strategy: str = "sequential"
    ) -> Iterator[Tuple[List[int], TextualKnowledgeGraph]]:
        """Create batches of nodes with their local subgraphs.
        
        Args:
            kg: Knowledge graph to batch
            strategy: Batching strategy ('sequential', 'random', 'community')
            
        Yields:
            Tuple of (node_indices, subgraph)
        """
        num_nodes = kg.num_nodes
        batch_size = min(self.config.batch_size, num_nodes)
        
        if strategy == "sequential":
            for i in range(0, num_nodes, batch_size):
                end_idx = min(i + batch_size, num_nodes)
                node_indices = list(range(i, end_idx))
                
                # Create subgraph with k-hop neighborhood
                expanded_indices = self._expand_with_neighbors(kg, node_indices)
                subgraph = kg.subgraph(expanded_indices)
                
                yield node_indices, subgraph
        
        elif strategy == "random":
            node_indices = np.random.permutation(num_nodes).tolist()
            for i in range(0, num_nodes, batch_size):
                end_idx = min(i + batch_size, num_nodes)
                batch_nodes = node_indices[i:end_idx]
                
                expanded_indices = self._expand_with_neighbors(kg, batch_nodes)
                subgraph = kg.subgraph(expanded_indices)
                
                yield batch_nodes, subgraph
        
        elif strategy == "community":
            # Simple community-based batching using graph structure
            communities = self._detect_communities(kg)
            
            for community in communities:
                if len(community) <= batch_size:
                    expanded_indices = self._expand_with_neighbors(kg, community)
                    subgraph = kg.subgraph(expanded_indices)
                    yield community, subgraph
                else:
                    # Split large communities
                    for i in range(0, len(community), batch_size):
                        end_idx = min(i + batch_size, len(community))
                        batch_nodes = community[i:end_idx]
                        
                        expanded_indices = self._expand_with_neighbors(kg, batch_nodes)
                        subgraph = kg.subgraph(expanded_indices)
                        yield batch_nodes, subgraph
    
    def create_sliding_window_batches(
        self,
        kg: TextualKnowledgeGraph,
        window_size: int = None
    ) -> Iterator[TextualKnowledgeGraph]:
        """Create overlapping sliding window batches.
        
        Args:
            kg: Knowledge graph to batch
            window_size: Size of each window
            
        Yields:
            Subgraph for each window
        """
        if window_size is None:
            window_size = self.config.max_nodes_per_batch
        
        num_nodes = kg.num_nodes
        step_size = int(window_size * (1 - self.config.overlap_ratio))
        
        for start in range(0, num_nodes, step_size):
            end = min(start + window_size, num_nodes)
            node_indices = list(range(start, end))
            
            # Expand with neighbors for context
            expanded_indices = self._expand_with_neighbors(kg, node_indices)
            subgraph = kg.subgraph(expanded_indices)
            
            yield subgraph
    
    def create_memory_aware_batches(
        self,
        kg: TextualKnowledgeGraph,
        estimate_memory_fn: Optional[callable] = None
    ) -> Iterator[TextualKnowledgeGraph]:
        """Create batches based on memory constraints.
        
        Args:
            kg: Knowledge graph to batch
            estimate_memory_fn: Function to estimate memory usage
            
        Yields:
            Memory-constrained subgraphs
        """
        if estimate_memory_fn is None:
            estimate_memory_fn = self._estimate_memory_usage
        
        current_batch_nodes = []
        current_memory_mb = 0
        
        for node_idx in range(kg.num_nodes):
            # Estimate memory for adding this node
            node_memory = estimate_memory_fn(kg, [node_idx])
            
            if (current_memory_mb + node_memory > self.config.memory_limit_mb 
                and current_batch_nodes):
                # Yield current batch
                expanded_indices = self._expand_with_neighbors(kg, current_batch_nodes)
                subgraph = kg.subgraph(expanded_indices)
                yield subgraph
                
                # Start new batch
                current_batch_nodes = [node_idx]
                current_memory_mb = node_memory
            else:
                current_batch_nodes.append(node_idx)
                current_memory_mb += node_memory
        
        # Yield final batch
        if current_batch_nodes:
            expanded_indices = self._expand_with_neighbors(kg, current_batch_nodes)
            subgraph = kg.subgraph(expanded_indices)
            yield subgraph
    
    def _expand_with_neighbors(
        self, 
        kg: TextualKnowledgeGraph, 
        node_indices: List[int],
        k_hops: int = 1
    ) -> List[int]:
        """Expand node set with k-hop neighbors.
        
        Args:
            kg: Knowledge graph
            node_indices: Core node indices
            k_hops: Number of hops to expand
            
        Returns:
            Expanded list of node indices
        """
        expanded_set = set(node_indices)
        current_frontier = set(node_indices)
        
        for _ in range(k_hops):
            next_frontier = set()
            
            if kg.edge_index is not None:
                # Find neighbors of current frontier
                edges = kg.edge_index.t()
                for edge in edges:
                    src, tgt = edge[0].item(), edge[1].item()
                    if src in current_frontier:
                        next_frontier.add(tgt)
                    if tgt in current_frontier:
                        next_frontier.add(src)
            
            expanded_set.update(next_frontier)
            current_frontier = next_frontier
        
        return sorted(list(expanded_set))
    
    def _detect_communities(
        self, 
        kg: TextualKnowledgeGraph,
        method: str = "simple"
    ) -> List[List[int]]:
        """Detect graph communities for batching.
        
        Args:
            kg: Knowledge graph
            method: Community detection method
            
        Returns:
            List of communities (each is a list of node indices)
        """
        if method == "simple":
            # Simple connected component detection
            visited = set()
            communities = []
            
            for node in range(kg.num_nodes):
                if node in visited:
                    continue
                
                # BFS to find connected component
                community = []
                queue = [node]
                visited.add(node)
                
                while queue:
                    current = queue.pop(0)
                    community.append(current)
                    
                    if kg.edge_index is not None:
                        # Find neighbors
                        edges = kg.edge_index.t()
                        for edge in edges:
                            src, tgt = edge[0].item(), edge[1].item()
                            if src == current and tgt not in visited:
                                visited.add(tgt)
                                queue.append(tgt)
                            elif tgt == current and src not in visited:
                                visited.add(src)
                                queue.append(src)
                
                communities.append(community)
            
            return communities
        
        else:
            # Fallback to simple sequential batching
            batch_size = self.config.batch_size
            return [
                list(range(i, min(i + batch_size, kg.num_nodes)))
                for i in range(0, kg.num_nodes, batch_size)
            ]
    
    def _estimate_memory_usage(
        self, 
        kg: TextualKnowledgeGraph, 
        node_indices: List[int]
    ) -> float:
        """Estimate memory usage for a set of nodes.
        
        Args:
            kg: Knowledge graph
            node_indices: Node indices to estimate
            
        Returns:
            Estimated memory usage in MB
        """
        num_nodes = len(node_indices)
        
        # Estimate based on typical tensor sizes
        base_memory_per_node = 0.1  # MB per node (features + embeddings)
        text_memory = sum(len(kg.node_texts[i]) for i in node_indices) * 0.001  # Text memory
        
        # Edge memory (rough estimate)
        edge_memory = 0
        if kg.edge_index is not None:
            node_set = set(node_indices)
            relevant_edges = 0
            edges = kg.edge_index.t()
            for edge in edges:
                if edge[0].item() in node_set or edge[1].item() in node_set:
                    relevant_edges += 1
            edge_memory = relevant_edges * 0.01  # MB per edge
        
        return base_memory_per_node * num_nodes + text_memory + edge_memory


class TextBatcher:
    """Efficient batching for text processing."""
    
    def __init__(self, max_batch_size: int = 32, max_sequence_length: int = 512):
        """Initialize text batcher.
        
        Args:
            max_batch_size: Maximum batch size
            max_sequence_length: Maximum sequence length
        """
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
    
    def create_batches(
        self, 
        texts: List[str],
        sort_by_length: bool = True
    ) -> Iterator[List[str]]:
        """Create efficient text batches.
        
        Args:
            texts: List of texts to batch
            sort_by_length: Whether to sort by length for efficiency
            
        Yields:
            Batches of texts
        """
        if sort_by_length:
            # Sort by length to minimize padding
            indexed_texts = [(i, text, len(text)) for i, text in enumerate(texts)]
            indexed_texts.sort(key=lambda x: x[2])
            sorted_texts = [text for _, text, _ in indexed_texts]
        else:
            sorted_texts = texts
        
        for i in range(0, len(sorted_texts), self.max_batch_size):
            batch = sorted_texts[i:i + self.max_batch_size]
            yield batch
    
    def create_dynamic_batches(
        self, 
        texts: List[str],
        max_tokens: int = 8192
    ) -> Iterator[List[str]]:
        """Create batches with dynamic sizing based on total tokens.
        
        Args:
            texts: List of texts to batch
            max_tokens: Maximum tokens per batch
            
        Yields:
            Token-aware batches
        """
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            # Rough token estimate (words * 1.3)
            text_tokens = int(len(text.split()) * 1.3)
            
            if current_tokens + text_tokens > max_tokens and current_batch:
                yield current_batch
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens
        
        if current_batch:
            yield current_batch


class BatchProcessor:
    """High-level batch processor combining graph and text batching."""
    
    def __init__(
        self, 
        batch_config: BatchConfig = None,
        text_batch_size: int = 32
    ):
        """Initialize batch processor.
        
        Args:
            batch_config: Graph batching configuration
            text_batch_size: Text batching size
        """
        self.graph_batcher = GraphBatcher(batch_config)
        self.text_batcher = TextBatcher(text_batch_size)
    
    def process_large_graph(
        self,
        kg: TextualKnowledgeGraph,
        process_fn: callable,
        strategy: str = "sequential",
        **kwargs
    ) -> List[Any]:
        """Process large graph in batches.
        
        Args:
            kg: Knowledge graph to process
            process_fn: Function to apply to each batch
            strategy: Batching strategy
            **kwargs: Additional arguments for process_fn
            
        Returns:
            List of batch results
        """
        results = []
        
        for node_indices, subgraph in self.graph_batcher.create_node_batches(kg, strategy):
            try:
                batch_result = process_fn(subgraph, node_indices=node_indices, **kwargs)
                results.append(batch_result)
            except Exception as e:
                print(f"Error processing batch with {len(node_indices)} nodes: {e}")
                # Continue with next batch
                continue
        
        return results
    
    def process_texts_efficiently(
        self,
        texts: List[str],
        process_fn: callable,
        **kwargs
    ) -> List[Any]:
        """Process texts in efficient batches.
        
        Args:
            texts: List of texts to process
            process_fn: Function to apply to each text batch
            **kwargs: Additional arguments for process_fn
            
        Returns:
            Concatenated results from all batches
        """
        all_results = []
        
        for text_batch in self.text_batcher.create_batches(texts):
            try:
                batch_results = process_fn(text_batch, **kwargs)
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Error processing text batch of size {len(text_batch)}: {e}")
                continue
        
        return all_results


def auto_batch_size(
    memory_limit_mb: float,
    feature_dim: int,
    estimate_overhead: float = 2.0
) -> int:
    """Automatically determine optimal batch size based on memory constraints.
    
    Args:
        memory_limit_mb: Available memory in MB
        feature_dim: Feature dimension per item
        estimate_overhead: Overhead multiplier for safety
        
    Returns:
        Recommended batch size
    """
    # Rough estimate: each item uses feature_dim * 4 bytes (float32)
    # Plus overhead for gradients, activations, etc.
    bytes_per_item = feature_dim * 4 * estimate_overhead
    mb_per_item = bytes_per_item / (1024 * 1024)
    
    recommended_batch_size = int(memory_limit_mb / mb_per_item)
    
    # Reasonable bounds
    return max(1, min(recommended_batch_size, 1024))