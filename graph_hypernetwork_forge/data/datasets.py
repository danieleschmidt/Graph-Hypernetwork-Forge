"""Dataset classes for knowledge graph training and evaluation.

Provides PyTorch Dataset implementations for various KG learning tasks
including link prediction, node classification, and graph classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import numpy as np
import random
from .knowledge_graph import TextualKnowledgeGraph, NodeInfo, EdgeInfo
import logging

logger = logging.getLogger(__name__)


class KGDataset(Dataset):
    """Base dataset class for knowledge graph tasks."""
    
    def __init__(
        self,
        graphs: List[TextualKnowledgeGraph],
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.graphs = graphs
        self.transform = transform
        self.pre_transform = pre_transform
        
        if pre_transform is not None:
            self.graphs = [pre_transform(g) for g in self.graphs]
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        graph = self.graphs[idx]
        
        if self.transform is not None:
            graph = self.transform(graph)
        
        return {
            'graph': graph,
            'edge_index': graph.edge_index,
            'node_texts': graph.node_texts,
            'node_features': graph.node_features,
            'edge_features': graph.edge_features,
        }


class LinkPredictionDataset(Dataset):
    """Dataset for link prediction tasks.
    
    Creates positive and negative edge samples for training
    link prediction models.
    """
    
    def __init__(
        self,
        graph: TextualKnowledgeGraph,
        negative_sampling_ratio: float = 1.0,
        split_ratio: float = 0.8,
        random_seed: int = 42,
        mode: str = 'train',  # 'train', 'val', 'test'
    ):
        self.graph = graph
        self.negative_sampling_ratio = negative_sampling_ratio
        self.random_seed = random_seed
        self.mode = mode
        
        # Split edges
        random.seed(random_seed)
        edges = list(graph.edges)
        random.shuffle(edges)
        
        train_size = int(len(edges) * split_ratio)
        val_size = int(len(edges) * 0.1)
        
        if mode == 'train':
            self.edges = edges[:train_size]
        elif mode == 'val':
            self.edges = edges[train_size:train_size + val_size]
        else:  # test
            self.edges = edges[train_size + val_size:]
        
        # Create negative samples
        self._create_negative_samples()
        
        logger.info(f"LinkPrediction {mode}: {len(self.positive_samples)} pos, {len(self.negative_samples)} neg")
    
    def _create_negative_samples(self):
        """Create negative edge samples."""
        # Get existing edges for filtering
        existing_edges = set()
        for edge in self.graph.edges:
            existing_edges.add((edge.source, edge.target))
            existing_edges.add((edge.target, edge.source))  # Undirected
        
        # Get all nodes
        all_nodes = list(self.graph.nodes.keys())
        
        # Create positive samples
        self.positive_samples = []
        for edge in self.edges:
            self.positive_samples.append({
                'source': edge.source,
                'target': edge.target,
                'relation': edge.relation,
                'label': 1,
            })
        
        # Create negative samples
        self.negative_samples = []
        num_negative = int(len(self.positive_samples) * self.negative_sampling_ratio)
        
        attempts = 0
        while len(self.negative_samples) < num_negative and attempts < num_negative * 10:
            src = random.choice(all_nodes)
            dst = random.choice(all_nodes)
            
            if src != dst and (src, dst) not in existing_edges:
                # Random relation for negative sample
                relation = random.choice(self.edges).relation if self.edges else 'negative'
                
                self.negative_samples.append({
                    'source': src,
                    'target': dst,
                    'relation': relation,
                    'label': 0,
                })
            
            attempts += 1
        
        # Combine positive and negative samples
        self.samples = self.positive_samples + self.negative_samples
        random.shuffle(self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        return {
            'graph': self.graph,
            'edge_index': self.graph.edge_index,
            'node_texts': self.graph.node_texts,
            'node_features': self.graph.node_features,
            'source_node': sample['source'],
            'target_node': sample['target'],
            'relation': sample['relation'],
            'label': torch.tensor(sample['label'], dtype=torch.float),
        }


class NodeClassificationDataset(Dataset):
    """Dataset for node classification tasks."""
    
    def __init__(
        self,
        graphs: List[TextualKnowledgeGraph],
        node_labels: Dict[str, Dict[str, int]],  # graph_name -> {node_id: label}
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = 42,
        mode: str = 'train',
    ):
        self.graphs = graphs
        self.node_labels = node_labels
        self.mode = mode
        
        # Create samples from all graphs
        self.samples = []
        for graph in graphs:
            if graph.name in node_labels:
                labels = node_labels[graph.name]
                for node_id, label in labels.items():
                    if node_id in graph.nodes:
                        self.samples.append({
                            'graph': graph,
                            'node_id': node_id,
                            'label': label,
                        })
        
        # Split samples
        random.seed(random_seed)
        random.shuffle(self.samples)
        
        train_size = int(len(self.samples) * split_ratio[0])
        val_size = int(len(self.samples) * split_ratio[1])
        
        if mode == 'train':
            self.samples = self.samples[:train_size]
        elif mode == 'val':
            self.samples = self.samples[train_size:train_size + val_size]
        else:  # test
            self.samples = self.samples[train_size + val_size:]
        
        logger.info(f"NodeClassification {mode}: {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        graph = sample['graph']
        
        # Get node index
        node_ids = sorted(graph.nodes.keys())
        node_idx = node_ids.index(sample['node_id'])
        
        return {
            'graph': graph,
            'edge_index': graph.edge_index,
            'node_texts': graph.node_texts,
            'node_features': graph.node_features,
            'target_node_idx': torch.tensor(node_idx, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long),
        }


class GraphClassificationDataset(Dataset):
    """Dataset for graph-level classification tasks."""
    
    def __init__(
        self,
        graphs: List[TextualKnowledgeGraph],
        graph_labels: Dict[str, int],  # graph_name -> label
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = 42,
        mode: str = 'train',
    ):
        self.graph_labels = graph_labels
        self.mode = mode
        
        # Filter graphs with labels
        labeled_graphs = [g for g in graphs if g.name in graph_labels]
        
        # Split graphs
        random.seed(random_seed)
        random.shuffle(labeled_graphs)
        
        train_size = int(len(labeled_graphs) * split_ratio[0])
        val_size = int(len(labeled_graphs) * split_ratio[1])
        
        if mode == 'train':
            self.graphs = labeled_graphs[:train_size]
        elif mode == 'val':
            self.graphs = labeled_graphs[train_size:train_size + val_size]
        else:  # test
            self.graphs = labeled_graphs[train_size + val_size:]
        
        logger.info(f"GraphClassification {mode}: {len(self.graphs)} graphs")
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        graph = self.graphs[idx]
        label = self.graph_labels[graph.name]
        
        return {
            'graph': graph,
            'edge_index': graph.edge_index,
            'node_texts': graph.node_texts,
            'node_features': graph.node_features,
            'edge_features': graph.edge_features,
            'label': torch.tensor(label, dtype=torch.long),
        }


class ZeroShotDataset(Dataset):
    """Dataset for zero-shot evaluation across different domains."""
    
    def __init__(
        self,
        source_graphs: List[TextualKnowledgeGraph],
        target_graphs: List[TextualKnowledgeGraph],
        task_type: str = 'link_prediction',  # 'link_prediction', 'node_classification'
        **task_kwargs
    ):
        self.source_graphs = source_graphs
        self.target_graphs = target_graphs
        self.task_type = task_type
        
        # Create task-specific datasets
        if task_type == 'link_prediction':
            self.source_datasets = [
                LinkPredictionDataset(graph, mode='train', **task_kwargs)
                for graph in source_graphs
            ]
            self.target_datasets = [
                LinkPredictionDataset(graph, mode='test', **task_kwargs)
                for graph in target_graphs
            ]
        elif task_type == 'node_classification':
            # For zero-shot node classification, we need labels
            node_labels = task_kwargs.get('node_labels', {})
            self.source_datasets = [
                NodeClassificationDataset([graph], {graph.name: node_labels.get(graph.name, {})}, mode='train')
                for graph in source_graphs if graph.name in node_labels
            ]
            self.target_datasets = [
                NodeClassificationDataset([graph], {graph.name: node_labels.get(graph.name, {})}, mode='test')
                for graph in target_graphs if graph.name in node_labels
            ]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def get_source_data(self) -> List[Dataset]:
        """Get source domain datasets for training."""
        return self.source_datasets
    
    def get_target_data(self) -> List[Dataset]:
        """Get target domain datasets for evaluation."""
        return self.target_datasets


def collate_kg_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for knowledge graph batches."""
    # Handle different batch structures
    if 'edge_index' in batch[0]:
        # Standard graph batch
        batch_graphs = []
        batch_edge_indices = []
        batch_node_texts = []
        batch_node_features = []
        batch_labels = []
        
        node_offset = 0
        for item in batch:
            batch_graphs.append(item['graph'])
            
            # Adjust edge indices for batching
            edge_index = item['edge_index'] + node_offset
            batch_edge_indices.append(edge_index)
            
            batch_node_texts.extend(item['node_texts'])
            batch_node_features.append(item['node_features'])
            
            if 'label' in item:
                batch_labels.append(item['label'])
            
            node_offset += len(item['node_texts'])
        
        # Combine batches
        result = {
            'graphs': batch_graphs,
            'edge_index': torch.cat(batch_edge_indices, dim=1),
            'node_texts': batch_node_texts,
            'node_features': torch.cat(batch_node_features, dim=0),
        }
        
        if batch_labels:
            result['labels'] = torch.stack(batch_labels)
        
        return result
    
    else:
        # Default collate for other structures
        return {key: [item[key] for item in batch] for key in batch[0].keys()}


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """Create DataLoader with appropriate collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_kg_batch,
        **kwargs
    )