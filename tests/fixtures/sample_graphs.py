"""Sample graph fixtures for testing."""
import torch
import json
from pathlib import Path
from typing import Dict, List, Any


def create_academic_graph() -> Dict[str, Any]:
    """Create a sample academic knowledge graph."""
    return {
        "nodes": [
            {
                "id": 0,
                "text": "Dr. Sarah Chen is a professor of computer science specializing in machine learning and neural networks at Stanford University",
                "type": "person",
                "attributes": {"department": "CS", "role": "professor"}
            },
            {
                "id": 1,
                "text": "Stanford University is a prestigious research institution located in California, known for computer science and engineering programs",
                "type": "institution",
                "attributes": {"location": "California", "type": "university"}
            },
            {
                "id": 2,
                "text": "Graph Neural Networks: A Comprehensive Survey is a seminal paper published in 2020 reviewing GNN architectures and applications",
                "type": "paper",
                "attributes": {"year": 2020, "venue": "JMLR"}
            },
            {
                "id": 3,
                "text": "PyTorch Geometric is a library for deep learning on graphs and other irregular structures built on PyTorch",
                "type": "software",
                "attributes": {"language": "Python", "framework": "PyTorch"}
            }
        ],
        "edges": [
            {"source": 0, "target": 1, "relation": "affiliated_with", "weight": 1.0},
            {"source": 0, "target": 2, "relation": "authored", "weight": 1.0},
            {"source": 2, "target": 1, "relation": "published_at", "weight": 0.8},
            {"source": 2, "target": 3, "relation": "uses", "weight": 0.6},
            {"source": 0, "target": 3, "relation": "contributes_to", "weight": 0.7}
        ],
        "metadata": {
            "domain": "academic",
            "description": "Academic knowledge graph with researchers, institutions, papers, and software",
            "num_nodes": 4,
            "num_edges": 5
        }
    }


def create_social_network_graph() -> Dict[str, Any]:
    """Create a sample social network graph."""
    return {
        "nodes": [
            {
                "id": 0,
                "text": "Alex Johnson is a tech enthusiast who regularly posts about AI developments and startup news on social media",
                "type": "user",
                "attributes": {"followers": 15000, "interests": ["AI", "startups"]}
            },
            {
                "id": 1,
                "text": "TechCorp is a Fortune 500 technology company specializing in cloud computing and artificial intelligence solutions",
                "type": "company",
                "attributes": {"employees": 50000, "sector": "technology"}
            },
            {
                "id": 2,
                "text": "#MachineLearning is a trending hashtag used to discuss ML research, applications, and industry developments",
                "type": "hashtag",
                "attributes": {"usage_count": 1000000, "category": "technology"}
            },
            {
                "id": 3,
                "text": "The AI Revolution: Transforming Industries is a viral post about AI impact across different business sectors",
                "type": "post",
                "attributes": {"likes": 5000, "shares": 1200, "comments": 800}
            }
        ],
        "edges": [
            {"source": 0, "target": 1, "relation": "follows", "weight": 1.0},
            {"source": 0, "target": 3, "relation": "liked", "weight": 1.0},
            {"source": 3, "target": 2, "relation": "tagged_with", "weight": 1.0},
            {"source": 1, "target": 3, "relation": "shared", "weight": 0.9},
            {"source": 0, "target": 2, "relation": "uses", "weight": 0.8}
        ],
        "metadata": {
            "domain": "social",
            "description": "Social network graph with users, companies, hashtags, and posts",
            "num_nodes": 4,
            "num_edges": 5
        }
    }


def create_biomedical_graph() -> Dict[str, Any]:
    """Create a sample biomedical knowledge graph."""
    return {
        "nodes": [
            {
                "id": 0,
                "text": "Diabetes mellitus type 2 is a metabolic disorder characterized by high blood sugar and insulin resistance",
                "type": "disease",
                "attributes": {"icd10": "E11", "prevalence": "global"}
            },
            {
                "id": 1,
                "text": "Metformin is a medication used to treat type 2 diabetes by decreasing glucose production in the liver",
                "type": "drug",
                "attributes": {"class": "biguanide", "route": "oral"}
            },
            {
                "id": 2,
                "text": "Insulin is a hormone produced by pancreatic beta cells that regulates glucose metabolism",
                "type": "protein",
                "attributes": {"function": "hormone", "location": "pancreas"}
            },
            {
                "id": 3,
                "text": "Pancreas is an organ that produces digestive enzymes and hormones including insulin and glucagon",
                "type": "organ",
                "attributes": {"system": "endocrine", "location": "abdomen"}
            }
        ],
        "edges": [
            {"source": 0, "target": 1, "relation": "treated_by", "weight": 0.9},
            {"source": 0, "target": 2, "relation": "involves", "weight": 1.0},
            {"source": 2, "target": 3, "relation": "produced_by", "weight": 1.0},
            {"source": 1, "target": 2, "relation": "affects", "weight": 0.7},
            {"source": 0, "target": 3, "relation": "affects", "weight": 0.8}
        ],
        "metadata": {
            "domain": "biomedical",
            "description": "Biomedical knowledge graph with diseases, drugs, proteins, and organs",
            "num_nodes": 4,
            "num_edges": 5
        }
    }


def create_synthetic_large_graph(num_nodes: int, num_edges: int) -> Dict[str, Any]:
    """Create a large synthetic graph for scalability testing."""
    import random
    import numpy as np
    
    # Generate diverse text patterns
    templates = [
        "Entity {id} is a {category} with properties {prop1} and {prop2}",
        "Node {id} represents a {category} in the {domain} domain with characteristics {prop1}",
        "Item {id} is classified as {category} and exhibits behavior {prop1} and {prop2}",
        "Object {id} belongs to category {category} and has attributes {prop1}, {prop2}"
    ]
    
    categories = ["research", "business", "social", "technical", "academic", "industrial"]
    domains = ["science", "technology", "healthcare", "finance", "education", "entertainment"]
    properties = ["efficient", "innovative", "robust", "scalable", "secure", "reliable", 
                 "fast", "accurate", "comprehensive", "flexible", "modern", "advanced"]
    
    nodes = []
    for i in range(num_nodes):
        template = random.choice(templates)
        category = random.choice(categories)
        domain = random.choice(domains)
        prop1 = random.choice(properties)
        prop2 = random.choice(properties)
        
        text = template.format(
            id=i, 
            category=category, 
            domain=domain, 
            prop1=prop1, 
            prop2=prop2
        )
        
        nodes.append({
            "id": i,
            "text": text,
            "type": category,
            "attributes": {"domain": domain, "synthetic": True}
        })
    
    # Generate random edges
    edges = []
    edge_set = set()
    relations = ["related_to", "connects_to", "influences", "depends_on", "similar_to"]
    
    for _ in range(num_edges):
        source = random.randint(0, num_nodes - 1)
        target = random.randint(0, num_nodes - 1)
        
        if source != target and (source, target) not in edge_set:
            edge_set.add((source, target))
            edges.append({
                "source": source,
                "target": target,
                "relation": random.choice(relations),
                "weight": random.uniform(0.1, 1.0)
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "domain": "synthetic",
            "description": f"Large synthetic graph with {num_nodes} nodes and {len(edges)} edges",
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "synthetic": True
        }
    }


def save_graph_to_file(graph_data: Dict[str, Any], file_path: Path) -> Path:
    """Save graph data to JSON file."""
    # Convert any tensors to lists for JSON serialization
    serializable_data = _make_json_serializable(graph_data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    return file_path


def _make_json_serializable(obj):
    """Convert PyTorch tensors and other objects to JSON-serializable format."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def create_cross_domain_test_set(tmp_dir: Path) -> Dict[str, Path]:
    """Create a set of graphs from different domains for cross-domain testing."""
    graphs = {
        "academic": create_academic_graph(),
        "social": create_social_network_graph(),
        "biomedical": create_biomedical_graph(),
        "synthetic_small": create_synthetic_large_graph(50, 100),
        "synthetic_medium": create_synthetic_large_graph(200, 500),
        "synthetic_large": create_synthetic_large_graph(1000, 2000)
    }
    
    file_paths = {}
    for domain, graph_data in graphs.items():
        file_path = tmp_dir / f"{domain}_graph.json"
        save_graph_to_file(graph_data, file_path)
        file_paths[domain] = file_path
    
    return file_paths