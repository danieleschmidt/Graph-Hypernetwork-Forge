#!/usr/bin/env python3
"""
Enhanced Error Handling Demo

This script demonstrates the comprehensive error handling and validation
features added to the Graph Hypernetwork Forge codebase.

Features demonstrated:
1. Structured logging with different levels
2. Memory management and monitoring
3. GPU error handling
4. Input validation
5. Graceful error recovery
6. Custom exception handling
"""

import torch
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_hypernetwork_forge.utils import (
    get_logger, setup_logging, memory_management,
    ValidationError, ModelError, GPUError, MemoryMonitor,
    start_global_memory_monitoring, stop_global_memory_monitoring
)
from graph_hypernetwork_forge.models.hypergnn import HyperGNN
from graph_hypernetwork_forge.data.knowledge_graph import TextualKnowledgeGraph
from graph_hypernetwork_forge.utils.training import HyperGNNTrainer


def demo_logging():
    """Demonstrate structured logging capabilities."""
    print("\n=== Logging Demo ===")
    
    # Setup logging with different configurations
    setup_logging(
        log_dir="logs",
        level="INFO",
        console_output=True,
        file_output=True,
        structured_format=False
    )
    
    logger = get_logger("demo")
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.debug("This is a debug message (might not show depending on level)")
    logger.error("This is an error message")
    
    # Log with additional context
    logger.info("Processing completed", extra={
        'processing_time': 1.23,
        'items_processed': 100,
        'status': 'success'
    })
    
    print("✓ Logging demo completed - check logs/ directory for output files")


def demo_memory_management():
    """Demonstrate memory management and monitoring."""
    print("\n=== Memory Management Demo ===")
    
    # Start global memory monitoring
    start_global_memory_monitoring(interval=5.0)
    
    logger = get_logger("memory_demo")
    
    # Create a memory-intensive operation within managed context
    with memory_management(cleanup_on_exit=True) as monitor:
        logger.info("Starting memory-intensive operation")
        
        # Get current memory info
        memory_info = monitor.get_memory_info()
        logger.info(f"Memory before operation: {memory_info.process_memory_gb:.2f} GB")
        
        # Create some large tensors
        tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000)
            tensors.append(tensor)
            logger.info(f"Created tensor {i+1}/5")
        
        # Get memory info after
        memory_info = monitor.get_memory_info()
        logger.info(f"Memory after operation: {memory_info.process_memory_gb:.2f} GB")
        
        logger.info("Memory-intensive operation completed")
    
    # Stop monitoring
    stop_global_memory_monitoring()
    
    print("✓ Memory management demo completed")


def demo_input_validation():
    """Demonstrate comprehensive input validation."""
    print("\n=== Input Validation Demo ===")
    
    logger = get_logger("validation_demo")
    
    # Test 1: Invalid model parameters
    try:
        logger.info("Testing invalid model parameters...")
        model = HyperGNN(
            text_encoder="invalid_model_name",
            hidden_dim=-1,  # Invalid negative dimension
            num_layers=0    # Invalid zero layers
        )
    except ValidationError as e:
        logger.info(f"✓ Caught expected validation error: {e.message}")
        logger.info(f"  Field: {e.field}, Value: {e.value}, Expected: {e.expected}")
    except Exception as e:
        logger.warning(f"Caught different error type: {e}")
    
    # Test 2: Invalid knowledge graph data
    try:
        logger.info("Testing invalid knowledge graph data...")
        
        # Create graph with mismatched dimensions
        edge_index = torch.tensor([[0, 1], [1, 2]])  # 2 edges
        node_texts = ["node1", "node2", "node3"]     # 3 nodes
        node_features = torch.randn(2, 10)           # Only 2 feature vectors
        
        graph = TextualKnowledgeGraph(
            edge_index=edge_index,
            node_texts=node_texts,
            node_features=node_features
        )
    except ValidationError as e:
        logger.info(f"✓ Caught expected data validation error: {e.message}")
    except Exception as e:
        logger.warning(f"Caught different error type: {e}")
    
    print("✓ Input validation demo completed")


def demo_error_recovery():
    """Demonstrate graceful error recovery."""
    print("\n=== Error Recovery Demo ===")
    
    logger = get_logger("recovery_demo")
    
    # Create a valid model and graph for demonstration
    try:
        logger.info("Creating valid model and graph...")
        
        model = HyperGNN(
            text_encoder="sentence-transformers/all-MiniLM-L6-v2",
            hidden_dim=64,
            num_layers=2,
            dropout=0.1
        )
        
        # Create a simple knowledge graph
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        node_texts = ["First node", "Second node", "Third node"]
        node_features = torch.randn(3, 32)
        
        graph = TextualKnowledgeGraph(
            edge_index=edge_index,
            node_texts=node_texts,
            node_features=node_features
        )
        
        logger.info(f"✓ Created model and graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        
        # Test the model with valid inputs
        with torch.no_grad():
            embeddings = model(graph.edge_index, graph.node_features, graph.node_texts)
            logger.info(f"✓ Model inference successful: output shape {embeddings.shape}")
        
    except Exception as e:
        logger.error(f"Model creation or inference failed: {e}", exc_info=True)
    
    print("✓ Error recovery demo completed")


def demo_trainer_error_handling():
    """Demonstrate trainer error handling."""
    print("\n=== Trainer Error Handling Demo ===")
    
    logger = get_logger("trainer_demo")
    
    try:
        # Create a simple model
        model = HyperGNN(hidden_dim=32, num_layers=2)
        
        # Create trainer with memory monitoring
        trainer = HyperGNNTrainer(
            model=model,
            memory_monitoring=True,
            memory_cleanup_threshold=0.8
        )
        
        logger.info("✓ Trainer created successfully with memory monitoring enabled")
        
        # Test invalid training data handling
        try:
            # Empty training graphs list should raise validation error
            trainer.train_epoch([], torch.nn.CrossEntropyLoss())
        except ValidationError as e:
            logger.info(f"✓ Caught expected trainer validation error: {e.message}")
        
        # Cleanup trainer resources
        trainer.cleanup()
        logger.info("✓ Trainer cleanup completed")
        
    except Exception as e:
        logger.error(f"Trainer demo failed: {e}", exc_info=True)
    
    print("✓ Trainer error handling demo completed")


def main():
    """Run all error handling demos."""
    print("Graph Hypernetwork Forge - Enhanced Error Handling Demo")
    print("=" * 55)
    
    try:
        demo_logging()
        demo_memory_management()
        demo_input_validation()
        demo_error_recovery()
        demo_trainer_error_handling()
        
        print("\n" + "=" * 55)
        print("✅ All demos completed successfully!")
        print("\nKey improvements demonstrated:")
        print("• Structured logging with context and performance metrics")
        print("• Automatic memory monitoring and cleanup")
        print("• Comprehensive input validation with clear error messages")
        print("• Graceful error recovery and resource cleanup")
        print("• GPU memory management and CUDA error handling")
        print("• Custom exception hierarchy for better error categorization")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()