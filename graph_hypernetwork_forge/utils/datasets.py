"""Dataset utilities and sample data generation."""

import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from sklearn.datasets import make_classification

from ..data.knowledge_graph import TextualKnowledgeGraph


class SyntheticDataGenerator:
    """Generate synthetic knowledge graphs for testing and benchmarking."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_social_network(
        self,
        num_nodes: int = 100,
        avg_degree: float = 5.0,
        num_classes: int = 3,
    ) -> TextualKnowledgeGraph:
        """Generate a synthetic social network knowledge graph.
        
        Args:
            num_nodes: Number of nodes
            avg_degree: Average node degree
            num_classes: Number of node classes
            
        Returns:
            Synthetic social network graph
        """
        # Generate graph structure
        p = avg_degree / (num_nodes - 1)
        graph = nx.erdos_renyi_graph(num_nodes, p, seed=self.seed)
        
        # Ensure connected graph
        if not nx.is_connected(graph):
            # Connect components
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                graph.add_edge(node1, node2)
        
        # Generate node features and labels
        node_features, node_labels = make_classification(
            n_samples=num_nodes,
            n_features=64,
            n_classes=num_classes,
            n_informative=32,
            random_state=self.seed,
        )
        
        # Generate textual descriptions
        professions = [
            "Software Engineer", "Data Scientist", "Product Manager", "Designer",
            "Marketing Specialist", "Sales Representative", "Teacher", "Doctor",
            "Lawyer", "Journalist", "Artist", "Musician", "Chef", "Accountant",
            "Consultant", "Entrepreneur", "Researcher", "Engineer", "Architect",
        ]
        
        hobbies = [
            "reading", "cooking", "traveling", "photography", "painting", "music",
            "sports", "gaming", "hiking", "dancing", "writing", "gardening",
            "yoga", "meditation", "programming", "learning languages", "volunteering",
        ]
        
        locations = [
            "New York", "San Francisco", "London", "Tokyo", "Berlin", "Paris",
            "Toronto", "Sydney", "Singapore", "Amsterdam", "Stockholm", "Zurich",
        ]
        
        node_texts = []
        for i in range(num_nodes):
            profession = random.choice(professions)
            hobby = random.choice(hobbies)
            location = random.choice(locations)
            age = random.randint(22, 65)
            
            text = f"A {age}-year-old {profession} based in {location} who enjoys {hobby}."
            node_texts.append(text)
        
        # Convert to edge index format
        edge_list = list(graph.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create knowledge graph
        kg = TextualKnowledgeGraph(
            edge_index=edge_index,
            node_texts=node_texts,
            node_features=torch.tensor(node_features, dtype=torch.float32),
            node_labels=torch.tensor(node_labels, dtype=torch.long),
            metadata={"domain": "social_network", "type": "synthetic"},
        )
        
        return kg
    
    def generate_citation_network(
        self,
        num_nodes: int = 200,
        avg_degree: float = 3.0,
        num_classes: int = 5,
    ) -> TextualKnowledgeGraph:
        """Generate a synthetic citation network.
        
        Args:
            num_nodes: Number of papers
            avg_degree: Average citations per paper
            num_classes: Number of research areas
            
        Returns:
            Synthetic citation network
        """
        # Generate directed graph (citation network)
        p = avg_degree / (num_nodes - 1)
        graph = nx.erdos_renyi_graph(num_nodes, p, directed=True, seed=self.seed)
        
        # Generate node features and labels
        node_features, node_labels = make_classification(
            n_samples=num_nodes,
            n_features=128,
            n_classes=num_classes,
            n_informative=64,
            random_state=self.seed,
        )
        
        # Research areas and keywords
        research_areas = [
            "Machine Learning", "Computer Vision", "Natural Language Processing",
            "Robotics", "Database Systems", "Human-Computer Interaction",
            "Computer Graphics", "Network Security", "Distributed Systems",
        ]
        
        ml_keywords = [
            "neural networks", "deep learning", "reinforcement learning", "supervised learning",
            "unsupervised learning", "optimization", "gradient descent", "backpropagation",
        ]
        
        cv_keywords = [
            "image recognition", "object detection", "semantic segmentation", "image classification",
            "convolutional neural networks", "computer vision", "image processing",
        ]
        
        nlp_keywords = [
            "natural language processing", "language models", "text classification", "sentiment analysis",
            "machine translation", "named entity recognition", "text summarization",
        ]
        
        all_keywords = {
            0: ml_keywords,
            1: cv_keywords,
            2: nlp_keywords,
            3: ml_keywords + cv_keywords,
            4: nlp_keywords + ml_keywords,
        }
        
        # Generate paper titles and abstracts
        node_texts = []
        for i in range(num_nodes):
            label = node_labels[i]
            area = research_areas[label % len(research_areas)]
            keywords = random.choices(
                all_keywords.get(label, ml_keywords), k=random.randint(2, 4)
            )
            
            title = f"A Study on {random.choice(keywords).title()} in {area}"
            abstract = (
                f"This paper presents a novel approach to {random.choice(keywords)} "
                f"in the context of {area.lower()}. We propose a method that combines "
                f"{random.choice(keywords)} with {random.choice(keywords)} to achieve "
                f"improved performance on benchmark datasets."
            )
            
            text = f"Title: {title}. Abstract: {abstract}"
            node_texts.append(text)
        
        # Convert to edge index format (undirected for simplicity)
        edge_list = [(u, v) for u, v in graph.edges()]
        edge_list.extend([(v, u) for u, v in graph.edges()])  # Make undirected
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create knowledge graph
        kg = TextualKnowledgeGraph(
            edge_index=edge_index,
            node_texts=node_texts,
            node_features=torch.tensor(node_features, dtype=torch.float32),
            node_labels=torch.tensor(node_labels, dtype=torch.long),
            metadata={"domain": "citation_network", "type": "synthetic"},
        )
        
        return kg
    
    def generate_product_network(
        self,
        num_nodes: int = 150,
        avg_degree: float = 4.0,
        num_classes: int = 4,
    ) -> TextualKnowledgeGraph:
        """Generate a synthetic product recommendation network.
        
        Args:
            num_nodes: Number of products
            avg_degree: Average connections per product
            num_classes: Number of product categories
            
        Returns:
            Synthetic product network
        """
        # Generate graph structure
        p = avg_degree / (num_nodes - 1)
        graph = nx.erdos_renyi_graph(num_nodes, p, seed=self.seed)
        
        # Generate node features and labels
        node_features, node_labels = make_classification(
            n_samples=num_nodes,
            n_features=32,
            n_classes=num_classes,
            n_informative=16,
            random_state=self.seed,
        )
        
        # Product categories and attributes
        categories = ["Electronics", "Clothing", "Books", "Home & Garden"]
        
        electronics = ["smartphone", "laptop", "headphones", "camera", "tablet", "smartwatch"]
        clothing = ["t-shirt", "jeans", "sneakers", "jacket", "dress", "hat"]
        books = ["fiction novel", "biography", "cookbook", "textbook", "manual", "poetry"]
        home_garden = ["chair", "table", "lamp", "plant", "tool set", "decoration"]
        
        products_by_category = {
            0: electronics,
            1: clothing,
            2: books,
            3: home_garden,
        }
        
        brands = ["TechCorp", "StyleCo", "ReadMore", "HomelyHome", "Universal", "Premium"]
        colors = ["black", "white", "blue", "red", "green", "silver", "gray"]
        
        # Generate product descriptions
        node_texts = []
        for i in range(num_nodes):
            label = node_labels[i]
            category = categories[label]
            product_type = random.choice(products_by_category[label])
            brand = random.choice(brands)
            color = random.choice(colors)
            price = random.randint(10, 500)
            rating = round(random.uniform(3.0, 5.0), 1)
            
            text = (
                f"A high-quality {color} {product_type} from {brand}. "
                f"Price: ${price}. Category: {category}. "
                f"Customer rating: {rating}/5 stars. "
                f"Perfect for everyday use with excellent durability."
            )
            node_texts.append(text)
        
        # Convert to edge index format
        edge_list = list(graph.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create knowledge graph
        kg = TextualKnowledgeGraph(
            edge_index=edge_index,
            node_texts=node_texts,
            node_features=torch.tensor(node_features, dtype=torch.float32),
            node_labels=torch.tensor(node_labels, dtype=torch.long),
            metadata={"domain": "product_network", "type": "synthetic"},
        )
        
        return kg
    
    def generate_multi_domain_datasets(
        self,
        domains: List[str] = ["social", "citation", "product"],
        sizes: List[int] = [100, 150, 120],
    ) -> Dict[str, List[TextualKnowledgeGraph]]:
        """Generate multiple domain datasets for transfer learning evaluation.
        
        Args:
            domains: List of domain types to generate
            sizes: List of graph sizes for each domain
            
        Returns:
            Dictionary mapping domain names to lists of graphs
        """
        datasets = {}
        
        for domain, size in zip(domains, sizes):
            graphs = []
            
            # Generate multiple graphs per domain with variations
            for i in range(3):  # 3 graphs per domain
                if domain == "social":
                    graph = self.generate_social_network(
                        num_nodes=size + random.randint(-20, 20),
                        avg_degree=random.uniform(3.0, 7.0),
                        num_classes=random.choice([3, 4, 5]),
                    )
                elif domain == "citation":
                    graph = self.generate_citation_network(
                        num_nodes=size + random.randint(-30, 30),
                        avg_degree=random.uniform(2.0, 5.0),
                        num_classes=random.choice([4, 5, 6]),
                    )
                elif domain == "product":
                    graph = self.generate_product_network(
                        num_nodes=size + random.randint(-25, 25),
                        avg_degree=random.uniform(3.0, 6.0),
                        num_classes=random.choice([3, 4, 5]),
                    )
                else:
                    raise ValueError(f"Unknown domain: {domain}")
                
                # Add domain-specific metadata
                graph.metadata.update({
                    "domain": domain,
                    "split": "train" if i < 2 else "test",
                    "graph_id": i,
                })
                
                graphs.append(graph)
            
            datasets[domain] = graphs
        
        return datasets


class DatasetSplitter:
    """Utilities for splitting datasets."""
    
    @staticmethod
    def train_val_test_split(
        graphs: List[TextualKnowledgeGraph],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[TextualKnowledgeGraph], List[TextualKnowledgeGraph], List[TextualKnowledgeGraph]]:
        """Split graphs into train/validation/test sets.
        
        Args:
            graphs: List of knowledge graphs
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed
            
        Returns:
            Tuple of (train_graphs, val_graphs, test_graphs)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        random.seed(seed)
        shuffled_graphs = graphs.copy()
        random.shuffle(shuffled_graphs)
        
        n = len(graphs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_graphs = shuffled_graphs[:train_end]
        val_graphs = shuffled_graphs[train_end:val_end]
        test_graphs = shuffled_graphs[val_end:]
        
        return train_graphs, val_graphs, test_graphs
    
    @staticmethod
    def node_split(
        graph: TextualKnowledgeGraph,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split nodes into train/validation/test masks.
        
        Args:
            graph: Knowledge graph
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed
            
        Returns:
            Tuple of (train_mask, val_mask, test_mask)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        random.seed(seed)
        indices = list(range(graph.num_nodes))
        random.shuffle(indices)
        
        train_end = int(graph.num_nodes * train_ratio)
        val_end = train_end + int(graph.num_nodes * val_ratio)
        
        train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True
        
        return train_mask, val_mask, test_mask


# Example usage and testing functions
def create_sample_datasets() -> Dict[str, List[TextualKnowledgeGraph]]:
    """Create sample datasets for testing and demonstration."""
    generator = SyntheticDataGenerator(seed=42)
    
    datasets = generator.generate_multi_domain_datasets(
        domains=["social", "citation", "product"],
        sizes=[80, 120, 100],
    )
    
    return datasets


def demonstrate_zero_shot_transfer():
    """Demonstrate zero-shot transfer capabilities."""
    print("Generating sample datasets for zero-shot transfer demonstration...")
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Source domain: social networks
    source_graphs = [
        generator.generate_social_network(num_nodes=100, num_classes=3)
        for _ in range(5)
    ]
    
    # Target domain: citation networks (different structure and text)
    target_graphs = [
        generator.generate_citation_network(num_nodes=120, num_classes=5)
        for _ in range(3)
    ]
    
    print(f"Generated {len(source_graphs)} source graphs (social network)")
    print(f"Generated {len(target_graphs)} target graphs (citation network)")
    
    # Print sample texts to show domain difference
    print("\nSample source domain texts:")
    for i, text in enumerate(source_graphs[0].node_texts[:3]):
        print(f"  {i+1}. {text}")
    
    print("\nSample target domain texts:")
    for i, text in enumerate(target_graphs[0].node_texts[:3]):
        print(f"  {i+1}. {text}")
    
    return source_graphs, target_graphs