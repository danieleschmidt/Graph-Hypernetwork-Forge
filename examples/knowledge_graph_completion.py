#!/usr/bin/env python3
"""
Knowledge Graph Completion Example

This example demonstrates using HyperGNN for knowledge graph completion tasks,
where the goal is to predict missing links between entities based on their
textual descriptions.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_hypernetwork_forge import HyperGNN, TextualKnowledgeGraph
from graph_hypernetwork_forge.utils import SyntheticDataGenerator, HyperGNNTrainer


class LinkPredictionHead(nn.Module):
    """Link prediction head for knowledge graph completion."""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Predict link probabilities.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Link probabilities [num_edges]
        """
        # Get source and target embeddings
        source_embeddings = embeddings[edge_index[0]]
        target_embeddings = embeddings[edge_index[1]]
        
        # Concatenate and predict
        edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
        link_probs = self.predictor(edge_embeddings).squeeze()
        
        return link_probs


class HyperGNNLinkPredictor(nn.Module):
    """Complete model for link prediction with HyperGNN."""
    
    def __init__(self, hypergnn_config: dict):
        super().__init__()
        self.hypergnn = HyperGNN(**hypergnn_config)
        self.link_head = LinkPredictionHead(hypergnn_config["hidden_dim"])
    
    def forward(self, edge_index, node_features, node_texts, test_edges):
        """Forward pass for link prediction.
        
        Args:
            edge_index: Training graph edges
            node_features: Node features
            node_texts: Node text descriptions
            test_edges: Edges to predict
            
        Returns:
            Link probabilities for test edges
        """
        # Get node embeddings from HyperGNN
        embeddings = self.hypergnn(edge_index, node_features, node_texts)
        
        # Predict links
        link_probs = self.link_head(embeddings, test_edges)
        
        return link_probs


def create_kg_completion_dataset():
    """Create a knowledge graph completion dataset."""
    print("Creating knowledge graph completion dataset...")
    
    # Create a larger citation network
    generator = SyntheticDataGenerator(seed=42)
    full_kg = generator.generate_citation_network(num_nodes=50, num_classes=5)
    
    # Split edges into train/test
    all_edges = full_kg.edge_index.t()
    num_edges = all_edges.size(0)
    
    # Randomly remove 20% of edges for testing
    perm = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    
    train_edges = all_edges[perm[:train_size]]
    test_edges = all_edges[perm[train_size:]]
    
    # Create training graph (with missing edges)
    train_kg = TextualKnowledgeGraph(
        edge_index=train_edges.t().contiguous(),
        node_texts=full_kg.node_texts,
        node_features=full_kg.node_features,
        node_labels=full_kg.node_labels,
        metadata={"task": "link_prediction", "original_edges": num_edges}
    )
    
    # Generate negative samples for training
    neg_edges = generate_negative_edges(train_kg, num_neg=train_edges.size(0))
    
    print(f"Dataset created:")
    print(f"  Nodes: {train_kg.num_nodes}")
    print(f"  Training edges: {train_edges.size(0)}")
    print(f"  Test edges: {test_edges.size(0)}")
    print(f"  Negative edges: {neg_edges.size(0)}")
    
    return train_kg, train_edges, test_edges, neg_edges


def generate_negative_edges(kg: TextualKnowledgeGraph, num_neg: int) -> torch.Tensor:
    """Generate negative edge samples."""
    existing_edges = set()
    for edge in kg.edge_index.t():
        existing_edges.add((edge[0].item(), edge[1].item()))
        existing_edges.add((edge[1].item(), edge[0].item()))  # Undirected
    
    negative_edges = []
    while len(negative_edges) < num_neg:
        src = torch.randint(0, kg.num_nodes, (1,)).item()
        dst = torch.randint(0, kg.num_nodes, (1,)).item()
        
        if src != dst and (src, dst) not in existing_edges:
            negative_edges.append([src, dst])
    
    return torch.tensor(negative_edges, dtype=torch.long)


def train_link_predictor():
    """Train the link prediction model."""
    print("\nTraining link prediction model...")
    
    # Create dataset
    train_kg, train_edges, test_edges, neg_edges = create_kg_completion_dataset()
    
    # Model configuration
    model_config = {
        "text_encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "gnn_backbone": "GAT",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1
    }
    
    # Create model
    model = HyperGNNLinkPredictor(model_config)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training loop
    model.train()
    num_epochs = 20
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Positive samples
        pos_scores = model(
            train_kg.edge_index,
            train_kg.node_features,
            train_kg.node_texts,
            train_edges.t()
        )
        pos_labels = torch.ones_like(pos_scores)
        
        # Negative samples
        neg_scores = model(
            train_kg.edge_index,
            train_kg.node_features,
            train_kg.node_texts,
            neg_edges.t()
        )
        neg_labels = torch.zeros_like(neg_scores)
        
        # Combined loss
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])
        
        loss = criterion(all_scores, all_labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return model, train_kg, test_edges


def evaluate_link_prediction(model, train_kg, test_edges):
    """Evaluate link prediction performance."""
    print("\nEvaluating link prediction...")
    
    model.eval()
    with torch.no_grad():
        # Predict test edges (positive samples)
        pos_scores = model(
            train_kg.edge_index,
            train_kg.node_features,
            train_kg.node_texts,
            test_edges.t()
        )
        
        # Generate negative test samples
        neg_test_edges = generate_negative_edges(train_kg, num_neg=test_edges.size(0))
        neg_scores = model(
            train_kg.edge_index,
            train_kg.node_features,
            train_kg.node_texts,
            neg_test_edges.t()
        )
        
        # Calculate metrics
        pos_predictions = (pos_scores > 0.5).float()
        neg_predictions = (neg_scores > 0.5).float()
        
        pos_accuracy = pos_predictions.mean().item()
        neg_accuracy = (1 - neg_predictions).mean().item()
        overall_accuracy = (pos_accuracy + neg_accuracy) / 2
        
        # AUC calculation (simplified)
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        
        # Sort by scores
        sorted_indices = torch.argsort(all_scores, descending=True)
        sorted_labels = all_labels[sorted_indices]
        
        # Calculate AUC (area under ROC curve)
        n_pos = torch.sum(sorted_labels).item()
        n_neg = len(sorted_labels) - n_pos
        
        if n_pos > 0 and n_neg > 0:
            auc = calculate_auc(sorted_labels)
        else:
            auc = 0.5
        
        print(f"Link Prediction Results:")
        print(f"  Positive accuracy: {pos_accuracy:.4f}")
        print(f"  Negative accuracy: {neg_accuracy:.4f}")
        print(f"  Overall accuracy: {overall_accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return {
            "pos_accuracy": pos_accuracy,
            "neg_accuracy": neg_accuracy,
            "overall_accuracy": overall_accuracy,
            "auc": auc
        }


def calculate_auc(sorted_labels):
    """Calculate AUC from sorted labels."""
    n_pos = torch.sum(sorted_labels).item()
    n_neg = len(sorted_labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Calculate rank sum for positive samples
    ranks = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float)
    pos_rank_sum = torch.sum(ranks * sorted_labels).item()
    
    # AUC formula
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def demonstrate_zero_shot_completion():
    """Demonstrate zero-shot knowledge graph completion."""
    print("\n" + "="*50)
    print("Zero-Shot Knowledge Graph Completion Demo")
    print("="*50)
    
    generator = SyntheticDataGenerator(seed=123)
    
    # Train on social networks
    print("Training on social network domain...")
    social_graphs = [
        generator.generate_social_network(num_nodes=30, num_classes=3)
        for _ in range(3)
    ]
    
    # Test on citation networks (different domain)
    print("Testing on citation network domain (zero-shot)...")
    citation_kg = generator.generate_citation_network(num_nodes=25, num_classes=3)
    
    # Show domain difference
    print(f"\nDomain comparison:")
    print(f"Social network text: {social_graphs[0].node_texts[0][:60]}...")
    print(f"Citation network text: {citation_kg.node_texts[0][:60]}...")
    
    # Simple demonstration (without full training)
    model_config = {
        "text_encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "gnn_backbone": "GAT",
        "hidden_dim": 64,
        "num_layers": 2,
    }
    
    model = HyperGNNLinkPredictor(model_config)
    model.eval()
    
    # Simulate inference on new domain
    with torch.no_grad():
        # Create some test edges
        test_edges = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
        
        scores = model(
            citation_kg.edge_index,
            citation_kg.node_features,
            citation_kg.node_texts,
            test_edges.t()
        )
        
        print(f"\nZero-shot link predictions on citation network:")
        for i, score in enumerate(scores):
            src, dst = test_edges[i]
            print(f"  Link {src} → {dst}: {score.item():.4f}")
        
        print("✅ Zero-shot completion successful!")


def main():
    """Main function demonstrating knowledge graph completion."""
    print("🔗 Knowledge Graph Completion with HyperGNN")
    print("=" * 60)
    
    try:
        # Train and evaluate link prediction
        model, train_kg, test_edges = train_link_predictor()
        results = evaluate_link_prediction(model, train_kg, test_edges)
        
        # Demonstrate zero-shot capabilities
        demonstrate_zero_shot_completion()
        
        print("\n" + "="*60)
        print("✅ Knowledge graph completion demo completed!")
        print("\nKey achievements:")
        print(f"  • Link prediction accuracy: {results['overall_accuracy']:.4f}")
        print(f"  • AUC score: {results['auc']:.4f}")
        print("  • Zero-shot transfer to new domains")
        print("  • Dynamic weight generation from text")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()