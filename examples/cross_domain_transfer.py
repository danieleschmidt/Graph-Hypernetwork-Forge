#!/usr/bin/env python3
"""
Cross-Domain Transfer Learning Example

This example demonstrates the zero-shot transfer capabilities of HyperGNN
across different knowledge graph domains without any retraining.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_hypernetwork_forge import HyperGNN, TextualKnowledgeGraph
from graph_hypernetwork_forge.utils import (
    SyntheticDataGenerator, 
    HyperGNNTrainer, 
    ZeroShotEvaluator
)


def create_multi_domain_datasets():
    """Create datasets from multiple domains for transfer learning."""
    print("Creating multi-domain datasets...")
    
    generator = SyntheticDataGenerator(seed=42)
    
    domains = {
        "social": {
            "train": [generator.generate_social_network(num_nodes=40, num_classes=3) for _ in range(5)],
            "test": [generator.generate_social_network(num_nodes=30, num_classes=3) for _ in range(2)]
        },
        "citation": {
            "train": [generator.generate_citation_network(num_nodes=50, num_classes=4) for _ in range(5)],
            "test": [generator.generate_citation_network(num_nodes=35, num_classes=4) for _ in range(2)]
        },
        "product": {
            "train": [generator.generate_product_network(num_nodes=35, num_classes=3) for _ in range(5)],
            "test": [generator.generate_product_network(num_nodes=25, num_classes=3) for _ in range(2)]
        },
        "biomedical": create_biomedical_dataset(generator),
        "financial": create_financial_dataset(generator),
    }
    
    print("Domain datasets created:")
    for domain, data in domains.items():
        train_nodes = sum(kg.num_nodes for kg in data["train"])
        test_nodes = sum(kg.num_nodes for kg in data["test"])
        print(f"  {domain}: {len(data['train'])} train graphs ({train_nodes} nodes), "
              f"{len(data['test'])} test graphs ({test_nodes} nodes)")
    
    return domains


def create_biomedical_dataset(generator):
    """Create synthetic biomedical knowledge graphs."""
    # Create graphs with biomedical-style text
    graphs = {"train": [], "test": []}
    
    for split, count in [("train", 4), ("test", 2)]:
        for i in range(count):
            # Generate base graph structure
            kg = generator.generate_citation_network(num_nodes=30, num_classes=3)
            
            # Replace with biomedical texts
            biomedical_texts = generate_biomedical_texts(kg.num_nodes)
            
            # Create new KG with biomedical texts
            bio_kg = TextualKnowledgeGraph(
                edge_index=kg.edge_index,
                node_texts=biomedical_texts,
                node_features=kg.node_features,
                node_labels=kg.node_labels,
                metadata={"domain": "biomedical", "split": split}
            )
            
            graphs[split].append(bio_kg)
    
    return graphs


def create_financial_dataset(generator):
    """Create synthetic financial knowledge graphs."""
    graphs = {"train": [], "test": []}
    
    for split, count in [("train", 4), ("test", 2)]:
        for i in range(count):
            # Generate base graph structure
            kg = generator.generate_product_network(num_nodes=25, num_classes=3)
            
            # Replace with financial texts
            financial_texts = generate_financial_texts(kg.num_nodes)
            
            # Create new KG with financial texts
            fin_kg = TextualKnowledgeGraph(
                edge_index=kg.edge_index,
                node_texts=financial_texts,
                node_features=kg.node_features,
                node_labels=kg.node_labels,
                metadata={"domain": "financial", "split": split}
            )
            
            graphs[split].append(fin_kg)
    
    return graphs


def generate_biomedical_texts(num_nodes):
    """Generate biomedical-style text descriptions."""
    proteins = ["p53", "BRCA1", "EGFR", "TP53", "KRAS", "PIK3CA", "PTEN", "MYC"]
    diseases = ["cancer", "diabetes", "Alzheimer's", "Parkinson's", "hypertension"]
    tissues = ["liver", "brain", "heart", "lung", "kidney", "muscle"]
    functions = ["transcription", "metabolism", "apoptosis", "proliferation", "differentiation"]
    
    import random
    random.seed(42)
    
    texts = []
    for i in range(num_nodes):
        protein = random.choice(proteins)
        disease = random.choice(diseases)
        tissue = random.choice(tissues)
        function = random.choice(functions)
        
        text = (f"Protein {protein} is involved in {function} and has been "
                f"associated with {disease} in {tissue} tissue. "
                f"Studies show significant expression changes in pathological conditions.")
        texts.append(text)
    
    return texts


def generate_financial_texts(num_nodes):
    """Generate financial-style text descriptions."""
    companies = ["Apple Inc", "Microsoft Corp", "Google LLC", "Amazon.com", "Tesla Inc"]
    sectors = ["technology", "healthcare", "finance", "energy", "retail", "manufacturing"]
    metrics = ["revenue growth", "profit margin", "debt ratio", "ROE", "market cap"]
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
    
    import random
    random.seed(42)
    
    texts = []
    for i in range(num_nodes):
        company = random.choice(companies)
        sector = random.choice(sectors)
        metric = random.choice(metrics)
        region = random.choice(regions)
        value = round(random.uniform(5, 25), 1)
        
        text = (f"{company} operates in the {sector} sector with strong presence "
                f"in {region}. The company shows {value}% {metric} and maintains "
                f"competitive positioning in the global market.")
        texts.append(text)
    
    return texts


def train_source_domain_model(source_data, domain_name):
    """Train a model on the source domain."""
    print(f"\nTraining model on {domain_name} domain...")
    
    # Model configuration
    model = HyperGNN(
        text_encoder="sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone="GAT",
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )
    
    # Setup trainer
    trainer = HyperGNNTrainer(
        model=model,
        device="cpu"  # Use CPU for demo
    )
    
    # Train model
    history = trainer.train(
        train_graphs=source_data["train"],
        val_graphs=source_data["test"],
        num_epochs=10,  # Quick training for demo
        task_type="node_classification",
        early_stopping_patience=5
    )
    
    print(f"Training completed. Final validation loss: {history['val_loss'][-1]:.4f}")
    
    return trainer.model


def evaluate_cross_domain_transfer(model, domains, source_domain):
    """Evaluate zero-shot transfer to all other domains."""
    print(f"\nEvaluating zero-shot transfer from {source_domain} to other domains...")
    
    evaluator = ZeroShotEvaluator(model, device="cpu")
    results = {}
    
    for target_domain, target_data in domains.items():
        if target_domain == source_domain:
            continue
        
        print(f"\nTransferring to {target_domain} domain...")
        
        # Evaluate transfer
        transfer_metrics = evaluator.evaluate_transfer(
            source_graphs=domains[source_domain]["train"],
            target_graphs=target_data["test"],
            task_type="node_classification"
        )
        
        # Analyze text similarity
        source_texts = []
        for kg in domains[source_domain]["train"]:
            source_texts.extend(kg.node_texts)
        
        target_texts = []
        for kg in target_data["test"]:
            target_texts.extend(kg.node_texts)
        
        similarity_metrics = evaluator.analyze_text_similarity(
            source_texts[:100],  # Limit for demo
            target_texts[:100]
        )
        
        results[target_domain] = {
            "transfer": transfer_metrics,
            "similarity": similarity_metrics
        }
        
        print(f"  Zero-shot accuracy: {transfer_metrics['zero_shot_accuracy']:.4f}")
        print(f"  Text similarity: {similarity_metrics['avg_max_similarity']:.4f}")
    
    return results


def analyze_domain_characteristics(domains):
    """Analyze characteristics of different domains."""
    print("\nAnalyzing domain characteristics...")
    
    # Create a small model for text analysis
    model = HyperGNN(
        text_encoder="sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone="GAT",
        hidden_dim=64,
        num_layers=1
    )
    model.eval()
    
    domain_embeddings = {}
    domain_stats = {}
    
    with torch.no_grad():
        for domain_name, domain_data in domains.items():
            # Collect all texts from this domain
            all_texts = []
            for kg in domain_data["train"] + domain_data["test"]:
                all_texts.extend(kg.node_texts)
            
            # Sample texts for analysis
            sample_texts = all_texts[:50] if len(all_texts) > 50 else all_texts
            
            # Get embeddings
            embeddings = model.text_encoder(sample_texts)
            mean_embedding = embeddings.mean(dim=0)
            
            domain_embeddings[domain_name] = mean_embedding
            
            # Calculate statistics
            stats = {
                "num_texts": len(all_texts),
                "avg_text_length": np.mean([len(text.split()) for text in sample_texts]),
                "embedding_norm": torch.norm(mean_embedding).item(),
                "sample_text": sample_texts[0][:100] + "..."
            }
            domain_stats[domain_name] = stats
            
            print(f"\n{domain_name.upper()} Domain:")
            print(f"  Texts: {stats['num_texts']}")
            print(f"  Avg length: {stats['avg_text_length']:.1f} words")
            print(f"  Embedding norm: {stats['embedding_norm']:.4f}")
            print(f"  Sample: {stats['sample_text']}")
    
    # Calculate pairwise domain similarities
    print(f"\nDomain Similarities:")
    domain_names = list(domain_embeddings.keys())
    similarity_matrix = np.zeros((len(domain_names), len(domain_names)))
    
    for i, domain1 in enumerate(domain_names):
        for j, domain2 in enumerate(domain_names):
            similarity = torch.cosine_similarity(
                domain_embeddings[domain1].unsqueeze(0),
                domain_embeddings[domain2].unsqueeze(0)
            ).item()
            similarity_matrix[i, j] = similarity
            
            if i < j:  # Only print upper triangle
                print(f"  {domain1} ↔ {domain2}: {similarity:.4f}")
    
    return domain_stats, similarity_matrix, domain_names


def visualize_transfer_results(results, similarity_matrix, domain_names, source_domain):
    """Visualize transfer learning results."""
    print(f"\nVisualizing transfer results...")
    
    try:
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Transfer accuracy vs text similarity
        target_domains = list(results.keys())
        accuracies = [results[domain]["transfer"]["zero_shot_accuracy"] for domain in target_domains]
        similarities = [results[domain]["similarity"]["avg_max_similarity"] for domain in target_domains]
        
        ax1.scatter(similarities, accuracies, s=100, alpha=0.7)
        for i, domain in enumerate(target_domains):
            ax1.annotate(domain, (similarities[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('Text Similarity')
        ax1.set_ylabel('Zero-Shot Accuracy')
        ax1.set_title('Transfer Accuracy vs Text Similarity')
        ax1.grid(True, alpha=0.3)
        
        # 2. Domain similarity heatmap
        im = ax2.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
        ax2.set_xticks(range(len(domain_names)))
        ax2.set_yticks(range(len(domain_names)))
        ax2.set_xticklabels(domain_names, rotation=45, ha='right')
        ax2.set_yticklabels(domain_names)
        ax2.set_title('Domain Similarity Matrix')
        
        # Add text annotations
        for i in range(len(domain_names)):
            for j in range(len(domain_names)):
                ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha="center", va="center", color="white" if similarity_matrix[i, j] < 0.5 else "black")
        
        plt.colorbar(im, ax=ax2)
        
        # 3. Transfer accuracy by domain
        ax3.bar(target_domains, accuracies, alpha=0.7, color='skyblue')
        ax3.set_ylabel('Zero-Shot Accuracy')
        ax3.set_title(f'Transfer Performance from {source_domain}')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Text similarity distribution
        all_similarities = [results[domain]["similarity"]["avg_max_similarity"] for domain in target_domains]
        ax4.hist(all_similarities, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
        ax4.set_xlabel('Text Similarity')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Text Similarities')
        
        plt.tight_layout()
        plt.savefig('cross_domain_transfer_analysis.png', dpi=150, bbox_inches='tight')
        print("  Visualization saved as 'cross_domain_transfer_analysis.png'")
        
    except Exception as e:
        print(f"  Visualization error (continuing without plots): {e}")


def main():
    """Main function for cross-domain transfer demonstration."""
    print("🌐 Cross-Domain Transfer Learning with HyperGNN")
    print("=" * 60)
    
    try:
        # Create multi-domain datasets
        domains = create_multi_domain_datasets()
        
        # Analyze domain characteristics
        domain_stats, similarity_matrix, domain_names = analyze_domain_characteristics(domains)
        
        # Choose source domain for training
        source_domain = "social"
        print(f"\nUsing {source_domain} as source domain for training...")
        
        # Train model on source domain
        trained_model = train_source_domain_model(domains[source_domain], source_domain)
        
        # Evaluate cross-domain transfer
        transfer_results = evaluate_cross_domain_transfer(trained_model, domains, source_domain)
        
        # Visualize results
        visualize_transfer_results(transfer_results, similarity_matrix, domain_names, source_domain)
        
        # Summary
        print("\n" + "="*60)
        print("✅ Cross-domain transfer analysis completed!")
        
        print(f"\nKey findings:")
        print(f"  • Source domain: {source_domain}")
        print(f"  • Target domains: {len(transfer_results)}")
        
        # Best and worst transfer
        accuracies = {domain: results["transfer"]["zero_shot_accuracy"] 
                     for domain, results in transfer_results.items()}
        best_domain = max(accuracies, key=accuracies.get)
        worst_domain = min(accuracies, key=accuracies.get)
        
        print(f"  • Best transfer: {best_domain} ({accuracies[best_domain]:.4f})")
        print(f"  • Worst transfer: {worst_domain} ({accuracies[worst_domain]:.4f})")
        
        # Text similarity correlation
        similarities = {domain: results["similarity"]["avg_max_similarity"] 
                       for domain, results in transfer_results.items()}
        print(f"  • Text similarities range: {min(similarities.values()):.4f} - {max(similarities.values()):.4f}")
        
        print("\n🚀 HyperGNN successfully demonstrated zero-shot transfer across 5 domains!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()