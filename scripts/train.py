#!/usr/bin/env python3
"""Training script for HyperGNN models."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_hypernetwork_forge import HyperGNN
from graph_hypernetwork_forge.utils import (
    HyperGNNTrainer,
    SyntheticDataGenerator,
    DatasetSplitter,
    create_sample_datasets,
)
from graph_hypernetwork_forge.data import TextualKnowledgeGraph


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log"),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HyperGNN model")
    
    # Model arguments
    parser.add_argument(
        "--text-encoder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Text encoder model name",
    )
    parser.add_argument(
        "--gnn-backbone",
        type=str,
        default="GAT",
        choices=["GCN", "GAT", "SAGE"],
        help="GNN backbone architecture",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )
    
    # Training arguments
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (number of graphs per batch)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset (if not provided, synthetic data will be used)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="social",
        choices=["social", "citation", "product"],
        help="Domain for synthetic data generation",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=10,
        help="Number of synthetic graphs to generate",
    )
    parser.add_argument(
        "--graph-size",
        type=int,
        default=100,
        help="Size of synthetic graphs",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name for logging",
    )
    
    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    return parser.parse_args()


def load_data(args):
    """Load or generate training data."""
    if args.data_path:
        # Load data from file
        logging.info(f"Loading data from {args.data_path}")
        if args.data_path.endswith('.json'):
            graphs = [TextualKnowledgeGraph.from_json(args.data_path)]
        else:
            raise ValueError(f"Unsupported data format: {args.data_path}")
    else:
        # Generate synthetic data
        logging.info(f"Generating synthetic {args.domain} network data")
        generator = SyntheticDataGenerator(seed=args.seed)
        
        graphs = []
        for i in range(args.num_graphs):
            if args.domain == "social":
                graph = generator.generate_social_network(
                    num_nodes=args.graph_size,
                    num_classes=3,
                )
            elif args.domain == "citation":
                graph = generator.generate_citation_network(
                    num_nodes=args.graph_size,
                    num_classes=5,
                )
            elif args.domain == "product":
                graph = generator.generate_product_network(
                    num_nodes=args.graph_size,
                    num_classes=4,
                )
            else:
                raise ValueError(f"Unknown domain: {args.domain}")
            
            graphs.append(graph)
    
    # Split data
    train_graphs, val_graphs, test_graphs = DatasetSplitter.train_val_test_split(
        graphs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=args.seed
    )
    
    logging.info(f"Data split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")
    
    return train_graphs, val_graphs, test_graphs


def create_model(args):
    """Create HyperGNN model."""
    logging.info("Creating HyperGNN model")
    
    model = HyperGNN(
        text_encoder=args.text_encoder,
        gnn_backbone=args.gnn_backbone,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    return model


def create_trainer(model, args):
    """Create trainer and optimizer."""
    logging.info("Setting up trainer")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Create trainer
    trainer = HyperGNNTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        wandb_project=args.wandb_project,
    )
    
    return trainer


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup
    setup_logging(args.log_level)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logging.info(f"Configuration saved to {config_path}")
    
    try:
        # Load data
        train_graphs, val_graphs, test_graphs = load_data(args)
        
        # Create model
        model = create_model(args)
        
        # Create trainer
        trainer = create_trainer(model, args)
        
        # Train model
        logging.info("Starting training")
        save_path = str(output_dir / "best_model.pt") if args.save_model else None
        
        history = trainer.train(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            num_epochs=args.num_epochs,
            task_type="node_classification",
            early_stopping_patience=args.early_stopping_patience,
            save_path=save_path,
        )
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logging.info(f"Training history saved to {history_path}")
        
        # Final evaluation
        if test_graphs:
            logging.info("Performing final evaluation on test set")
            test_metrics = trainer.validate(
                val_graphs=test_graphs,
                loss_fn=torch.nn.CrossEntropyLoss(),
                task_type="node_classification",
            )
            
            logging.info(f"Test metrics: {test_metrics}")
            
            # Save test metrics
            metrics_path = output_dir / "test_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()