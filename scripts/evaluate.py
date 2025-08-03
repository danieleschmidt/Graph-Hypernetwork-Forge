#!/usr/bin/env python3
"""Evaluation script for HyperGNN models."""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph_hypernetwork_forge import HyperGNN
from graph_hypernetwork_forge.utils import (
    ZeroShotEvaluator,
    SyntheticDataGenerator,
    create_sample_datasets,
)
from graph_hypernetwork_forge.data import TextualKnowledgeGraph


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate HyperGNN model")
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to model configuration (if not in checkpoint)",
    )
    
    # Data arguments
    parser.add_argument(
        "--test-data-path",
        type=str,
        default=None,
        help="Path to test dataset",
    )
    parser.add_argument(
        "--source-domain",
        type=str,
        default="social",
        choices=["social", "citation", "product"],
        help="Source domain for synthetic data",
    )
    parser.add_argument(
        "--target-domain",
        type=str,
        default="citation",
        choices=["social", "citation", "product"],
        help="Target domain for zero-shot evaluation",
    )
    parser.add_argument(
        "--num-test-graphs",
        type=int,
        default=5,
        help="Number of test graphs to generate",
    )
    parser.add_argument(
        "--test-graph-size",
        type=int,
        default=100,
        help="Size of test graphs",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--task-type",
        type=str,
        default="node_classification",
        choices=["node_classification", "link_prediction"],
        help="Type of evaluation task",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Perform zero-shot transfer evaluation",
    )
    parser.add_argument(
        "--analyze-similarity",
        action="store_true",
        help="Analyze text similarity between domains",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    
    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for evaluation",
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


def load_model(model_path: str, config_path: str = None, device: str = "auto"):
    """Load trained model from checkpoint."""
    logging.info(f"Loading model from {model_path}")
    
    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    if "config" in checkpoint:
        config = checkpoint["config"]
    elif config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Use default configuration
        logging.warning("No configuration found, using defaults")
        config = {
            "text_encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "gnn_backbone": "GAT",
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.1,
        }
    
    # Create model
    model = HyperGNN.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logging.info(f"Model loaded successfully on {device}")
    
    return model, config


def load_test_data(args):
    """Load or generate test data."""
    if args.test_data_path:
        logging.info(f"Loading test data from {args.test_data_path}")
        if args.test_data_path.endswith('.json'):
            test_graphs = [TextualKnowledgeGraph.from_json(args.test_data_path)]
        else:
            raise ValueError(f"Unsupported data format: {args.test_data_path}")
    else:
        logging.info("Generating synthetic test data")
        generator = SyntheticDataGenerator(seed=args.seed)
        
        test_graphs = []
        for i in range(args.num_test_graphs):
            if args.target_domain == "social":
                graph = generator.generate_social_network(
                    num_nodes=args.test_graph_size,
                    num_classes=3,
                )
            elif args.target_domain == "citation":
                graph = generator.generate_citation_network(
                    num_nodes=args.test_graph_size,
                    num_classes=5,
                )
            elif args.target_domain == "product":
                graph = generator.generate_product_network(
                    num_nodes=args.test_graph_size,
                    num_classes=4,
                )
            else:
                raise ValueError(f"Unknown domain: {args.target_domain}")
            
            test_graphs.append(graph)
    
    logging.info(f"Loaded {len(test_graphs)} test graphs")
    
    return test_graphs


def evaluate_standard(model, test_graphs, task_type, device):
    """Perform standard evaluation on test graphs."""
    logging.info("Performing standard evaluation")
    
    model.eval()
    total_correct = 0
    total_samples = 0
    graph_accuracies = []
    
    with torch.no_grad():
        for i, graph in enumerate(test_graphs):
            # Move data to device
            edge_index = graph.edge_index.to(device)
            node_features = graph.node_features
            if node_features is not None:
                node_features = node_features.to(device)
            else:
                node_features = torch.randn(graph.num_nodes, 128, device=device)
            
            try:
                # Generate predictions
                predictions = model(edge_index, node_features, graph.node_texts)
                
                if task_type == "node_classification" and graph.node_labels is not None:
                    labels = graph.node_labels.to(device)
                    pred_classes = torch.argmax(predictions, dim=1)
                    correct = (pred_classes == labels).sum().item()
                    
                    graph_accuracy = correct / labels.size(0)
                    graph_accuracies.append(graph_accuracy)
                    
                    total_correct += correct
                    total_samples += labels.size(0)
                    
                    logging.info(f"Graph {i}: Accuracy = {graph_accuracy:.4f}")
            
            except Exception as e:
                logging.error(f"Error evaluating graph {i}: {e}")
                continue
    
    overall_accuracy = total_correct / max(total_samples, 1)
    avg_graph_accuracy = sum(graph_accuracies) / max(len(graph_accuracies), 1)
    
    results = {
        "overall_accuracy": overall_accuracy,
        "avg_graph_accuracy": avg_graph_accuracy,
        "graph_accuracies": graph_accuracies,
        "num_graphs": len(test_graphs),
        "total_nodes": total_samples,
    }
    
    logging.info(f"Standard evaluation results:")
    logging.info(f"  Overall accuracy: {overall_accuracy:.4f}")
    logging.info(f"  Average graph accuracy: {avg_graph_accuracy:.4f}")
    
    return results


def evaluate_zero_shot(model, args):
    """Perform zero-shot transfer evaluation."""
    logging.info("Performing zero-shot transfer evaluation")
    
    generator = SyntheticDataGenerator(seed=args.seed)
    
    # Generate source domain data (for reference)
    source_graphs = []
    for i in range(3):
        if args.source_domain == "social":
            graph = generator.generate_social_network(num_nodes=80, num_classes=3)
        elif args.source_domain == "citation":
            graph = generator.generate_citation_network(num_nodes=100, num_classes=5)
        elif args.source_domain == "product":
            graph = generator.generate_product_network(num_nodes=90, num_classes=4)
        source_graphs.append(graph)
    
    # Generate target domain data
    target_graphs = []
    for i in range(args.num_test_graphs):
        if args.target_domain == "social":
            graph = generator.generate_social_network(
                num_nodes=args.test_graph_size, num_classes=3
            )
        elif args.target_domain == "citation":
            graph = generator.generate_citation_network(
                num_nodes=args.test_graph_size, num_classes=5
            )
        elif args.target_domain == "product":
            graph = generator.generate_product_network(
                num_nodes=args.test_graph_size, num_classes=4
            )
        target_graphs.append(graph)
    
    # Create evaluator
    evaluator = ZeroShotEvaluator(model, device=args.device)
    
    # Evaluate transfer
    transfer_results = evaluator.evaluate_transfer(
        source_graphs=source_graphs,
        target_graphs=target_graphs,
        task_type=args.task_type,
    )
    
    results = {
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "transfer_results": transfer_results,
    }
    
    # Analyze text similarity if requested
    if args.analyze_similarity:
        logging.info("Analyzing text similarity between domains")
        
        # Collect all texts from source and target
        source_texts = []
        for graph in source_graphs:
            source_texts.extend(graph.node_texts)
        
        target_texts = []
        for graph in target_graphs:
            target_texts.extend(graph.node_texts)
        
        # Analyze similarity
        similarity_results = evaluator.analyze_text_similarity(source_texts, target_texts)
        results["text_similarity"] = similarity_results
        
        logging.info(f"Text similarity analysis:")
        logging.info(f"  Average max similarity: {similarity_results['avg_max_similarity']:.4f}")
        logging.info(f"  Min max similarity: {similarity_results['min_max_similarity']:.4f}")
        logging.info(f"  Similarity std: {similarity_results['similarity_std']:.4f}")
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup
    setup_logging(args.log_level)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        model, config = load_model(args.model_path, args.config_path, args.device)
        
        # Save model configuration
        config_path = output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        results = {}
        
        # Standard evaluation
        if not args.zero_shot:
            test_graphs = load_test_data(args)
            standard_results = evaluate_standard(
                model, test_graphs, args.task_type, model.device
            )
            results["standard_evaluation"] = standard_results
        
        # Zero-shot evaluation
        if args.zero_shot:
            zero_shot_results = evaluate_zero_shot(model, args)
            results["zero_shot_evaluation"] = zero_shot_results
        
        # Save results
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Evaluation results saved to {results_path}")
        logging.info("Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()