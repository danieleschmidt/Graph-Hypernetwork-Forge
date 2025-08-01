"""End-to-end integration tests for the complete pipeline."""
import pytest
import torch
import numpy as np
from pathlib import Path
import json
import tempfile
from typing import Dict, List, Any

from graph_hypernetwork_forge import HyperGNN, TextualKnowledgeGraph
from graph_hypernetwork_forge.models.encoders import TextEncoder
from graph_hypernetwork_forge.models.hypernetworks import WeightGenerator
from graph_hypernetwork_forge.models.gnns import DynamicGNN


@pytest.mark.integration
class TestCompleteWorkflow:
    """Test complete workflow from text to predictions."""

    def test_complete_pipeline_small_graph(self, tmp_path):
        """Test complete pipeline with small synthetic graph."""
        # Create test data
        graph_data = self._create_test_knowledge_graph(tmp_path)
        
        # Initialize model
        model = HyperGNN(
            text_encoder="sentence-transformers/all-MiniLM-L6-v2",
            gnn_backbone="gcn",
            hidden_dim=64,
            num_layers=2
        )
        
        # Load knowledge graph
        kg = TextualKnowledgeGraph.from_json(graph_data["path"])
        
        # Generate weights from text
        weights = model.generate_weights(kg.node_texts)
        assert isinstance(weights, dict)
        assert len(weights) > 0
        
        # Forward pass
        predictions = model.forward(kg.edge_index, kg.node_features, weights)
        assert predictions.shape[0] == kg.num_nodes
        assert predictions.shape[1] == model.output_dim

    def test_zero_shot_transfer(self, tmp_path):
        """Test zero-shot transfer between different domains."""
        # Create source domain graph (scientific papers)
        source_graph = self._create_scientific_graph(tmp_path / "source.json")
        
        # Create target domain graph (social network)
        target_graph = self._create_social_graph(tmp_path / "target.json")
        
        model = HyperGNN(hidden_dim=64, num_layers=2)
        
        # Train on source domain (mock training)
        source_kg = TextualKnowledgeGraph.from_json(source_graph)
        source_weights = model.generate_weights(source_kg.node_texts)
        source_predictions = model.forward(
            source_kg.edge_index, source_kg.node_features, source_weights
        )
        
        # Apply to target domain without retraining
        target_kg = TextualKnowledgeGraph.from_json(target_graph)
        target_weights = model.generate_weights(target_kg.node_texts)
        target_predictions = model.forward(
            target_kg.edge_index, target_kg.node_features, target_weights
        )
        
        # Verify predictions have correct shapes
        assert source_predictions.shape[0] == source_kg.num_nodes
        assert target_predictions.shape[0] == target_kg.num_nodes
        assert source_predictions.shape[1] == target_predictions.shape[1]

    @pytest.mark.slow
    def test_large_graph_scalability(self, tmp_path):
        """Test scalability with larger graphs."""
        # Create large graph
        large_graph = self._create_large_graph(tmp_path / "large.json", num_nodes=1000)
        
        model = HyperGNN(hidden_dim=128, num_layers=3)
        kg = TextualKnowledgeGraph.from_json(large_graph)
        
        # Test batch processing
        batch_size = 100
        all_predictions = []
        
        for i in range(0, kg.num_nodes, batch_size):
            end_idx = min(i + batch_size, kg.num_nodes)
            batch_texts = kg.node_texts[i:end_idx]
            batch_features = kg.node_features[i:end_idx]
            
            # Generate weights for batch
            batch_weights = model.generate_weights(batch_texts)
            
            # Create subgraph for batch (simplified)
            batch_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            
            batch_predictions = model.forward(
                batch_edge_index, batch_features[:2], batch_weights
            )
            all_predictions.append(batch_predictions)
        
        assert len(all_predictions) > 0

    def test_model_persistence(self, tmp_path):
        """Test model saving and loading."""
        model = HyperGNN(hidden_dim=64, num_layers=2)
        
        # Save model
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Load model
        loaded_model = HyperGNN(hidden_dim=64, num_layers=2)
        loaded_model.load_state_dict(torch.load(model_path))
        
        # Test equivalence
        test_texts = ["Test node 1", "Test node 2"]
        original_weights = model.generate_weights(test_texts)
        loaded_weights = loaded_model.generate_weights(test_texts)
        
        for key in original_weights:
            assert torch.allclose(original_weights[key], loaded_weights[key], atol=1e-6)

    def test_different_gnn_backends(self):
        """Test different GNN backend integrations."""
        backends = ["gcn", "gat", "sage"]
        test_texts = ["Node 1 description", "Node 2 description", "Node 3 description"]
        
        for backend in backends:
            model = HyperGNN(
                gnn_backend=backend,
                hidden_dim=32,
                num_layers=2
            )
            
            # Generate weights
            weights = model.generate_weights(test_texts)
            assert isinstance(weights, dict)
            assert len(weights) > 0
            
            # Test forward pass
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
            node_features = torch.randn(3, 32)
            
            predictions = model.forward(edge_index, node_features, weights)
            assert predictions.shape == (3, model.output_dim)

    def test_text_encoder_variations(self):
        """Test different text encoder configurations."""
        encoders = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]
        
        test_texts = ["Scientific paper about neural networks", "Social media post"]
        
        for encoder_name in encoders:
            model = HyperGNN(
                text_encoder=encoder_name,
                hidden_dim=64,
                num_layers=2
            )
            
            weights = model.generate_weights(test_texts)
            assert isinstance(weights, dict)
            
            # Test that different encoders produce different weights
            if len(encoders) > 1 and encoder_name == encoders[1]:
                # Compare with first encoder
                model_1 = HyperGNN(
                    text_encoder=encoders[0],
                    hidden_dim=64,
                    num_layers=2
                )
                weights_1 = model_1.generate_weights(test_texts)
                
                # Weights should be different (not exactly equal)
                for key in weights:
                    if key in weights_1:
                        assert not torch.allclose(weights[key], weights_1[key], atol=1e-3)

    def _create_test_knowledge_graph(self, tmp_path: Path) -> Dict[str, Any]:
        """Create a small test knowledge graph."""
        graph_data = {
            "nodes": [
                {"id": 0, "text": "Person: Alice is a researcher in machine learning"},
                {"id": 1, "text": "Institution: Stanford University is a research university"},
                {"id": 2, "text": "Paper: Neural networks for graph representation learning"}
            ],
            "edges": [
                {"source": 0, "target": 1, "relation": "affiliated_with"},
                {"source": 0, "target": 2, "relation": "authored"},
                {"source": 1, "target": 2, "relation": "published_at"}
            ],
            "node_features": torch.randn(3, 128).tolist(),
            "metadata": {
                "domain": "academic",
                "num_nodes": 3,
                "num_edges": 3
            }
        }
        
        file_path = tmp_path / "test_graph.json"
        with open(file_path, 'w') as f:
            json.dump(graph_data, f)
        
        return {"path": file_path, "data": graph_data}

    def _create_scientific_graph(self, file_path: Path) -> Path:
        """Create scientific domain knowledge graph."""
        graph_data = {
            "nodes": [
                {"id": 0, "text": "Research paper on deep learning architectures"},
                {"id": 1, "text": "Author specializing in computer vision"},
                {"id": 2, "text": "Conference venue for machine learning research"},
                {"id": 3, "text": "Dataset for image classification tasks"}
            ],
            "edges": [
                {"source": 1, "target": 0, "relation": "authored"},
                {"source": 0, "target": 2, "relation": "published_at"},
                {"source": 0, "target": 3, "relation": "uses_dataset"}
            ],
            "node_features": torch.randn(4, 128).tolist(),
            "metadata": {"domain": "scientific"}
        }
        
        with open(file_path, 'w') as f:
            json.dump(graph_data, f)
        
        return file_path

    def _create_social_graph(self, file_path: Path) -> Path:
        """Create social network domain knowledge graph."""
        graph_data = {
            "nodes": [
                {"id": 0, "text": "User who posts about technology trends"},
                {"id": 1, "text": "Social media influencer in tech space"},
                {"id": 2, "text": "Tech company with social media presence"},
                {"id": 3, "text": "Trending hashtag about artificial intelligence"}
            ],
            "edges": [
                {"source": 0, "target": 1, "relation": "follows"},
                {"source": 1, "target": 2, "relation": "mentions"},
                {"source": 0, "target": 3, "relation": "uses_hashtag"}
            ],
            "node_features": torch.randn(4, 128).tolist(),
            "metadata": {"domain": "social"}
        }
        
        with open(file_path, 'w') as f:
            json.dump(graph_data, f)
        
        return file_path

    def _create_large_graph(self, file_path: Path, num_nodes: int = 1000) -> Path:
        """Create large synthetic graph for scalability testing."""
        nodes = []
        edges = []
        
        # Generate nodes with diverse text descriptions
        domains = ["technology", "science", "business", "arts", "sports"]
        for i in range(num_nodes):
            domain = domains[i % len(domains)]
            nodes.append({
                "id": i,
                "text": f"{domain.capitalize()} entity {i} with descriptive metadata and context"
            })
        
        # Generate random edges
        num_edges = min(num_nodes * 2, 5000)  # Limit edges for performance
        edge_pairs = set()
        while len(edge_pairs) < num_edges:
            src = np.random.randint(0, num_nodes)
            tgt = np.random.randint(0, num_nodes)
            if src != tgt:
                edge_pairs.add((src, tgt))
        
        edges = [{"source": src, "target": tgt, "relation": "related"} 
                for src, tgt in edge_pairs]
        
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "node_features": torch.randn(num_nodes, 128).tolist(),
            "metadata": {
                "domain": "synthetic_large",
                "num_nodes": num_nodes,
                "num_edges": len(edges)
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(graph_data, f)
        
        return file_path


@pytest.mark.integration
class TestComponentIntegration:
    """Test integration between different components."""

    def test_text_encoder_hypernetwork_integration(self):
        """Test integration between text encoder and hypernetwork."""
        from graph_hypernetwork_forge.models.encoders import SentenceTransformerEncoder
        from graph_hypernetwork_forge.models.hypernetworks import MLPHypernetwork
        
        # Create components
        text_encoder = SentenceTransformerEncoder("all-MiniLM-L6-v2")
        hypernetwork = MLPHypernetwork(
            text_dim=text_encoder.embedding_dim,
            hidden_dim=64,
            num_layers=2
        )
        
        # Test integration
        texts = ["Graph neural network", "Deep learning model"]
        embeddings = text_encoder.encode(texts)
        weights = hypernetwork.generate_weights(embeddings)
        
        assert embeddings.shape[0] == len(texts)
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_hypernetwork_gnn_integration(self):
        """Test integration between hypernetwork and GNN."""
        from graph_hypernetwork_forge.models.hypernetworks import WeightGenerator
        from graph_hypernetwork_forge.models.gnns import GCNBackend
        
        # Create components
        weight_specs = {
            "layer_0_weight": (128, 64),
            "layer_0_bias": (64,),
            "layer_1_weight": (64, 32),
            "layer_1_bias": (32,)
        }
        
        weight_generator = WeightGenerator(384, weight_specs)  # 384 = text embedding dim
        gnn_backend = GCNBackend()
        
        # Test integration
        text_embeddings = torch.randn(3, 384)
        generated_weights = weight_generator(text_embeddings[0])  # Single embedding
        
        # Test GNN forward pass
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        node_features = torch.randn(3, 128)
        
        predictions = gnn_backend.forward(edge_index, node_features, generated_weights)
        assert predictions.shape[0] == 3
        assert predictions.shape[1] == 32  # output_dim from weight specs

    @pytest.mark.gpu
    def test_gpu_integration(self, device):
        """Test GPU integration and memory management."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = HyperGNN(hidden_dim=256, num_layers=3).to(device)
        
        # Create GPU tensors
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long).to(device)
        node_features = torch.randn(3, 256, device=device)
        texts = ["GPU tensor processing", "CUDA memory management", "Parallel computation"]
        
        # Test forward pass on GPU
        weights = model.generate_weights(texts)
        predictions = model.forward(edge_index, node_features, weights)
        
        assert predictions.device == device
        assert torch.cuda.memory_allocated() > 0

    def test_batch_processing_consistency(self):
        """Test that batch processing gives consistent results."""
        model = HyperGNN(hidden_dim=64, num_layers=2)
        
        texts = [
            "First text description",
            "Second text description", 
            "Third text description",
            "Fourth text description"
        ]
        
        # Process all at once
        batch_weights = model.generate_weights(texts)
        
        # Process individually
        individual_weights = []
        for text in texts:
            weights = model.generate_weights([text])
            individual_weights.append(weights)
        
        # Results should be consistent (allowing for small numerical differences)
        for i, weights in enumerate(individual_weights):
            for key in weights:
                batch_weight = batch_weights[key][i] if batch_weights[key].dim() > 2 else batch_weights[key]
                individual_weight = weights[key][0] if weights[key].dim() > 2 else weights[key]
                
                # Allow small differences due to batch processing
                assert torch.allclose(batch_weight, individual_weight, atol=1e-4)