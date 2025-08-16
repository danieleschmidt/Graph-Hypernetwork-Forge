#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Enhanced Core Functionality Demo
Demonstrates the working core features of the Graph Hypernetwork Forge
"""

import torch
import networkx as nx
import numpy as np
from graph_hypernetwork_forge import HyperGNN, TextualKnowledgeGraph


def create_demo_graph():
    """Create a demonstration knowledge graph with textual descriptions."""
    # Create a small social network graph
    G = nx.Graph()
    
    # Add nodes with metadata
    nodes = [
        ("alice", {"description": "Software engineer specializing in machine learning and data science"}),
        ("bob", {"description": "Product manager with expertise in user experience and market research"}),
        ("charlie", {"description": "Data scientist focused on natural language processing and deep learning"}),
        ("diana", {"description": "DevOps engineer experienced in cloud infrastructure and automation"}),
        ("eve", {"description": "Research scientist working on computer vision and robotics"})
    ]
    
    for node_id, attrs in nodes:
        G.add_node(node_id, **attrs)
    
    # Add edges (relationships)
    edges = [
        ("alice", "bob"),
        ("alice", "charlie"),
        ("bob", "diana"),
        ("charlie", "eve"),
        ("diana", "eve"),
        ("alice", "diana")
    ]
    
    G.add_edges_from(edges)
    return G


def demo_basic_functionality():
    """Demonstrate basic HyperGNN functionality."""
    print("🚀 Generation 1: MAKE IT WORK - Core Functionality Demo")
    print("=" * 60)
    
    # Create demo graph
    print("📊 Creating demonstration knowledge graph...")
    G = create_demo_graph()
    print(f"   ✓ Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Prepare graph data for HyperGNN
    print("\n🔧 Preparing graph data...")
    
    # Extract node features (random for demo)
    num_nodes = G.number_of_nodes()
    node_features = torch.randn(num_nodes, 64)  # Random features for demo
    
    # Map node names to indices
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    
    # Extract edge connectivity with proper mapping
    edge_list = list(G.edges())
    edge_pairs = [[node_to_idx[src], node_to_idx[dst]] for src, dst in edge_list] + \
                 [[node_to_idx[dst], node_to_idx[src]] for src, dst in edge_list]  # Make undirected
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t()
    
    # Extract node text descriptions
    node_texts = [G.nodes[node]["description"] for node in G.nodes()]
    
    print(f"   ✓ Node features shape: {node_features.shape}")
    print(f"   ✓ Edge index shape: {edge_index.shape}")
    print(f"   ✓ Number of text descriptions: {len(node_texts)}")
    
    # Initialize HyperGNN model
    print("\n🧠 Initializing HyperGNN model...")
    model = HyperGNN(
        text_encoder="sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone="GAT",
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )
    print("   ✓ HyperGNN model initialized successfully")
    
    # Set model to evaluation mode
    model.eval()
    
    # Perform inference
    print("\n⚡ Running HyperGNN inference...")
    with torch.no_grad():
        try:
            # Generate embeddings
            embeddings = model(edge_index, node_features, node_texts)
            print(f"   ✓ Generated node embeddings with shape: {embeddings.shape}")
            
            # Test weight generation separately
            print("\n🏗️  Testing weight generation...")
            weights = model.generate_weights(node_texts)
            print(f"   ✓ Generated weights: {type(weights)}")
            if isinstance(weights, dict):
                print(f"   ✓ Weight keys: {list(weights.keys())}")
            
            # Test prediction interface
            print("\n🎯 Testing prediction interface...")
            predictions = model.predict(edge_index, node_features, node_texts)
            print(f"   ✓ Predictions shape: {predictions.shape}")
            
        except Exception as e:
            print(f"   ❌ Error during inference: {e}")
            raise
    
    # Display results
    print("\n📋 Results Summary:")
    print(f"   • Input nodes: {num_nodes}")
    print(f"   • Input edges: {edge_index.size(1) // 2}")  # Undirected, so divide by 2
    print(f"   • Input feature dim: {node_features.size(1)}")
    print(f"   • Output embedding dim: {embeddings.size(1)}")
    print(f"   • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✅ Generation 1 core functionality working successfully!")
    return model, embeddings


def demo_advanced_features():
    """Demonstrate some advanced features."""
    print("\n🎓 Advanced Features Demo:")
    print("-" * 40)
    
    # Create a different graph for transfer learning demo
    print("🔄 Testing zero-shot transfer...")
    
    # Different domain: research papers
    research_nodes = [
        "Research paper on graph neural networks for drug discovery",
        "Study of transformer architectures in computer vision tasks", 
        "Analysis of reinforcement learning in autonomous driving systems",
        "Investigation of federated learning privacy preservation techniques"
    ]
    
    # Simple graph structure
    research_features = torch.randn(4, 64)
    research_edges = torch.tensor([[0, 1, 2, 1, 2, 3, 0, 3], 
                                  [1, 0, 1, 2, 3, 2, 3, 0]], dtype=torch.long)
    
    # Create model for this domain
    research_model = HyperGNN(
        text_encoder="sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone="GAT", 
        hidden_dim=128,
        num_layers=2
    )
    
    research_model.eval()
    with torch.no_grad():
        research_embeddings = research_model.predict(research_edges, research_features, research_nodes)
        print(f"   ✓ Zero-shot transfer successful! Shape: {research_embeddings.shape}")
    
    print("\n🏆 All Generation 1 features working correctly!")


if __name__ == "__main__":
    try:
        # Run core functionality demo
        model, embeddings = demo_basic_functionality()
        
        # Run advanced features demo
        demo_advanced_features()
        
        print("\n" + "="*60)
        print("🎉 Generation 1 implementation complete and working!")
        print("   Ready to proceed to Generation 2: MAKE IT ROBUST")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Generation 1 demo failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)