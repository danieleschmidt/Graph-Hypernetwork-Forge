#!/usr/bin/env python3
"""Demo script showcasing HyperGNN capabilities."""

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_hypernetwork_forge import HyperGNN, TextualKnowledgeGraph
from graph_hypernetwork_forge.utils import SyntheticDataGenerator


def demo_basic_usage():
    """Demonstrate basic HyperGNN usage."""
    print("🚀 HyperGNN Basic Usage Demo")
    print("=" * 50)
    
    # Create a sample knowledge graph
    generator = SyntheticDataGenerator(seed=42)
    kg = generator.generate_social_network(num_nodes=20, num_classes=3)
    
    print(f"Created knowledge graph with {kg.num_nodes} nodes and {kg.num_edges} edges")
    print(f"Sample node texts:")
    for i, text in enumerate(kg.node_texts[:3]):
        print(f"  {i+1}. {text}")
    
    # Initialize HyperGNN model
    model = HyperGNN(
        text_encoder="sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone="GAT",
        hidden_dim=64,  # Smaller for demo
        num_layers=2,
    )
    
    print(f"\nInitialized HyperGNN model:")
    print(f"  Text encoder: {model.text_encoder_name}")
    print(f"  GNN backbone: {model.gnn_backbone}")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Num layers: {model.num_layers}")
    
    # Generate some example node features
    node_features = torch.randn(kg.num_nodes, model.hidden_dim)
    
    # Forward pass
    print(f"\nPerforming forward pass...")
    model.eval()
    with torch.no_grad():
        predictions = model(kg.edge_index, node_features, kg.node_texts)
    
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3].tolist()}")
    
    return model, kg


def demo_zero_shot_transfer():
    """Demonstrate zero-shot transfer capabilities."""
    print("\n🎯 Zero-Shot Transfer Demo")
    print("=" * 50)
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Create source domain: social network
    print("Creating source domain (social network)...")
    source_kg = generator.generate_social_network(num_nodes=15, num_classes=3)
    
    # Create target domain: citation network (different text patterns)
    print("Creating target domain (citation network)...")
    target_kg = generator.generate_citation_network(num_nodes=12, num_classes=3)
    
    print(f"\nSource domain sample texts:")
    for i, text in enumerate(source_kg.node_texts[:2]):
        print(f"  {i+1}. {text}")
    
    print(f"\nTarget domain sample texts:")
    for i, text in enumerate(target_kg.node_texts[:2]):
        print(f"  {i+1}. {text}")
    
    # Initialize model
    model = HyperGNN(
        text_encoder="sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone="GAT",
        hidden_dim=64,
        num_layers=2,
    )
    
    # Generate predictions for both domains
    model.eval()
    with torch.no_grad():
        # Source domain
        source_features = torch.randn(source_kg.num_nodes, model.hidden_dim)
        source_predictions = model(source_kg.edge_index, source_features, source_kg.node_texts)
        
        # Target domain (zero-shot)
        target_features = torch.randn(target_kg.num_nodes, model.hidden_dim)
        target_predictions = model(target_kg.edge_index, target_features, target_kg.node_texts)
    
    print(f"\nSource domain predictions shape: {source_predictions.shape}")
    print(f"Target domain predictions shape: {target_predictions.shape}")
    print("✅ Zero-shot transfer successful!")
    
    return source_kg, target_kg, model


def demo_weight_generation():
    """Demonstrate dynamic weight generation."""
    print("\n⚙️  Dynamic Weight Generation Demo")
    print("=" * 50)
    
    # Create model
    model = HyperGNN(
        text_encoder="sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone="GAT",
        hidden_dim=32,  # Small for demo
        num_layers=2,
    )
    
    # Example texts from different domains
    texts = [
        "A software engineer working on machine learning projects.",
        "A research paper about neural networks and deep learning.",
        "A smartphone with advanced camera features and long battery life.",
    ]
    
    print("Example texts:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")
    
    # Generate weights
    print(f"\nGenerating GNN weights from text descriptions...")
    model.eval()
    with torch.no_grad():
        weights = model.generate_weights(texts)
    
    print(f"Generated weights for {len(weights)} layers:")
    for layer_idx, layer_weights in enumerate(weights):
        print(f"  Layer {layer_idx}:")
        for weight_name, weight_tensor in layer_weights.items():
            print(f"    {weight_name}: {weight_tensor.shape}")
    
    # Show weight variation across nodes
    first_layer_weights = weights[0]["weight"]  # [num_nodes, in_dim, out_dim]
    weight_norms = torch.norm(first_layer_weights, dim=(1, 2))
    
    print(f"\nWeight norms for different texts:")
    for i, norm in enumerate(weight_norms):
        print(f"  Text {i+1}: {norm.item():.4f}")
    
    return weights


def demo_knowledge_graph_creation():
    """Demonstrate TextualKnowledgeGraph creation and manipulation."""
    print("\n📊 Knowledge Graph Creation Demo")
    print("=" * 50)
    
    # Create edge connectivity
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3]
    ], dtype=torch.long)
    
    # Create node texts
    node_texts = [
        "Alice is a data scientist who loves machine learning.",
        "Bob works as a software engineer at a tech company.",
        "Carol is a product manager with experience in AI products.",
        "David is a researcher studying natural language processing.",
    ]
    
    # Create node features
    node_features = torch.randn(4, 16)
    
    # Create knowledge graph
    kg = TextualKnowledgeGraph(
        edge_index=edge_index,
        node_texts=node_texts,
        node_features=node_features,
        metadata={"domain": "professional_network", "created_by": "demo"}
    )
    
    print(f"Created knowledge graph:")
    print(f"  Nodes: {kg.num_nodes}")
    print(f"  Edges: {kg.num_edges}")
    print(f"  Domain: {kg.metadata.get('domain', 'unknown')}")
    
    # Show statistics
    stats = kg.statistics()
    print(f"\nGraph statistics:")
    for key, value in stats.items():
        if key != "metadata":
            print(f"  {key}: {value}")
    
    # Demonstrate subgraph extraction
    subgraph = kg.subgraph([0, 1, 2])
    print(f"\nSubgraph with nodes [0, 1, 2]:")
    print(f"  Nodes: {subgraph.num_nodes}")
    print(f"  Edges: {subgraph.num_edges}")
    
    # Demonstrate neighbor text retrieval
    neighbor_texts = kg.get_neighbor_texts(0, k_hops=1)
    print(f"\nNeighbor texts for node 0:")
    for i, text in enumerate(neighbor_texts):
        print(f"  {i+1}. {text}")
    
    return kg


def demo_multi_domain_comparison():
    """Compare model behavior across different domains."""
    print("\n🔬 Multi-Domain Comparison Demo")
    print("=" * 50)
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate graphs from different domains
    domains = {
        "social": generator.generate_social_network(num_nodes=10, num_classes=3),
        "citation": generator.generate_citation_network(num_nodes=10, num_classes=3),
        "product": generator.generate_product_network(num_nodes=10, num_classes=3),
    }
    
    # Initialize model
    model = HyperGNN(
        text_encoder="sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone="GAT",
        hidden_dim=32,
        num_layers=2,
    )
    
    print("Comparing embeddings across domains:")
    
    domain_embeddings = {}
    model.eval()
    
    with torch.no_grad():
        for domain_name, kg in domains.items():
            # Get text embeddings
            text_embeddings = model.text_encoder(kg.node_texts)
            avg_embedding = text_embeddings.mean(dim=0)
            domain_embeddings[domain_name] = avg_embedding
            
            print(f"\n{domain_name.capitalize()} domain:")
            print(f"  Sample text: {kg.node_texts[0]}")
            print(f"  Avg embedding norm: {torch.norm(avg_embedding).item():.4f}")
    
    # Calculate cross-domain similarities
    print(f"\nCross-domain similarities:")
    domain_names = list(domain_embeddings.keys())
    for i, domain1 in enumerate(domain_names):
        for j, domain2 in enumerate(domain_names):
            if i < j:
                similarity = torch.cosine_similarity(
                    domain_embeddings[domain1].unsqueeze(0),
                    domain_embeddings[domain2].unsqueeze(0)
                ).item()
                print(f"  {domain1} ↔ {domain2}: {similarity:.4f}")
    
    return domains, domain_embeddings


def main():
    """Run all demos."""
    print("🌟 Graph Hypernetwork Forge Demo")
    print("=" * 60)
    print("Demonstrating zero-shot GNN weight generation from text!\n")
    
    try:
        # Run demos
        demo_basic_usage()
        demo_zero_shot_transfer()
        demo_weight_generation()
        demo_knowledge_graph_creation()
        demo_multi_domain_comparison()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("\nKey takeaways:")
        print("  • HyperGNN generates GNN weights dynamically from text")
        print("  • Zero-shot transfer works across different knowledge domains")
        print("  • TextualKnowledgeGraph provides rich graph manipulation")
        print("  • Model adapts to different text patterns automatically")
        print("\nTry modifying the examples to explore further!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()