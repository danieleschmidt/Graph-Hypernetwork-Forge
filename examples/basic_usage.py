#!/usr/bin/env python3
"""Basic usage example for Graph Hypernetwork Forge.

This script demonstrates how to use the HyperGNN for zero-shot 
knowledge graph reasoning with textual node descriptions.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

# Import Graph Hypernetwork Forge
from graph_hypernetwork_forge import HyperGNN, TextualKnowledgeGraph
from graph_hypernetwork_forge.data import create_synthetic_kg, LinkPredictionDataset, create_dataloader
from graph_hypernetwork_forge.utils import TrainingConfig, HyperGNNTrainer, BenchmarkEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating HyperGNN usage."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Step 1: Create or load knowledge graphs
    logger.info("Creating synthetic knowledge graphs...")
    
    # Create source domain graph (for training)
    source_kg = create_synthetic_kg(
        num_nodes=200,
        num_edges=400,
        relations=['related_to', 'part_of', 'instance_of'],
        random_seed=42
    )
    
    # Create target domain graph (for zero-shot evaluation)
    target_kg = create_synthetic_kg(
        num_nodes=150,
        num_edges=300,
        relations=['similar_to', 'connected_to', 'belongs_to'],
        random_seed=123
    )
    
    logger.info(f"Source KG: {source_kg.get_statistics()}")
    logger.info(f"Target KG: {target_kg.get_statistics()}")
    
    # Step 2: Initialize HyperGNN model
    logger.info("Initializing HyperGNN model...")
    
    model = HyperGNN(
        text_encoder="sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone="GAT",  # Can be "GCN", "GAT", or "GraphSAGE"
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
    )
    
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 3: Prepare datasets
    logger.info("Preparing datasets...")
    
    # Create link prediction datasets
    train_dataset = LinkPredictionDataset(
        graph=source_kg,
        negative_sampling_ratio=1.0,
        mode='train'
    )
    
    val_dataset = LinkPredictionDataset(
        graph=source_kg,
        mode='val'
    )
    
    test_dataset = LinkPredictionDataset(
        graph=target_kg,  # Zero-shot evaluation on different domain
        mode='test'
    )
    
    # Create data loaders
    train_loader = create_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)
    
    # Step 4: Training configuration
    logger.info("Setting up training...")
    
    config = TrainingConfig(
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=32,
        num_epochs=50,
        patience=10,
        scheduler_type="cosine",
        gradient_clip=1.0,
        save_best=True,
        log_interval=5,
        eval_interval=2,
        checkpoint_dir="./checkpoints/basic_example",
    )
    
    # Step 5: Train the model
    logger.info("Starting training...")
    
    trainer = HyperGNNTrainer(model, config, device)
    
    # Define loss function for link prediction
    criterion = nn.BCEWithLogitsLoss()
    
    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        task_type="link_prediction"
    )
    
    logger.info("Training completed!")
    
    # Step 6: Zero-shot evaluation
    logger.info("Performing zero-shot evaluation...")
    
    # Load best model
    trainer.load_checkpoint("best")
    
    # Evaluate on target domain
    evaluator = BenchmarkEvaluator(model, device, save_dir="./evaluation_results")
    
    results = evaluator.evaluate_dataset(
        dataloader=test_loader,
        dataset_name="target_domain",
        task_type="link_prediction",
        return_predictions=True
    )
    
    logger.info("Zero-shot evaluation results:")
    for metric, value in results["metrics"].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Step 7: Generate comprehensive report
    logger.info("Generating evaluation report...")
    
    report = evaluator.generate_report(include_plots=True)
    
    # Step 8: Demonstrate direct inference
    logger.info("Demonstrating direct zero-shot inference...")
    
    # Example of direct inference on new graph
    sample_texts = [
        "This is a concept related to artificial intelligence and machine learning",
        "A fundamental principle in computer science and algorithms",
        "An important topic in data analysis and statistics",
        "A research area in natural language processing"
    ]
    
    # Create simple test graph
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_features = torch.randn(4, 256)  # 4 nodes, 256 features
    
    # Perform zero-shot inference
    model.eval()
    with torch.no_grad():
        predictions = model.zero_shot_inference(
            edge_index=edge_index.to(device),
            node_features=node_features.to(device),
            node_texts=sample_texts
        )
    
    logger.info(f"Sample predictions: {predictions.cpu().numpy()}")
    
    # Step 9: Save model for later use
    logger.info("Saving trained model...")
    
    save_path = Path("./saved_models/hypergnn_example")
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    
    logger.info(f"Model saved to {save_path}")
    
    # Step 10: Demonstrate loading saved model
    logger.info("Demonstrating model loading...")
    
    loaded_model = HyperGNN.load_pretrained(str(save_path), device=device)
    logger.info("Model loaded successfully!")
    
    # Verify loaded model works
    with torch.no_grad():
        test_predictions = loaded_model.zero_shot_inference(
            edge_index=edge_index.to(device),
            node_features=node_features.to(device),
            node_texts=sample_texts
        )
    
    # Check if predictions match
    predictions_match = torch.allclose(predictions, test_predictions, atol=1e-6)
    logger.info(f"Loaded model predictions match: {predictions_match}")
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()