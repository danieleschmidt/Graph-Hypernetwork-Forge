"""Test configuration and fixtures."""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Generator, Dict, Any

# Test configuration
pytest_plugins = ["pytest_benchmark"]


@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp(prefix="hypergnn_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_graph():
    """Sample graph data for testing."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_features = torch.randn(3, 128)
    return {"edge_index": edge_index, "node_features": node_features}


@pytest.fixture
def sample_texts():
    """Sample text descriptions for testing."""
    return ["Node representing a person", "Node representing a location", "Node representing an event"]


@pytest.fixture
def large_graph_data() -> Dict[str, Any]:
    """Generate large graph data for performance testing."""
    num_nodes = 1000
    num_edges = 2000
    
    # Generate random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Generate node features
    node_features = torch.randn(num_nodes, 128)
    
    # Generate node texts
    node_texts = [f"Complex text description for node {i} with metadata" for i in range(num_nodes)]
    
    return {
        "edge_index": edge_index,
        "node_features": node_features,
        "node_texts": node_texts,
        "num_nodes": num_nodes,
        "num_edges": num_edges
    }


@pytest.fixture
def performance_config() -> Dict[str, Any]:
    """Configuration for performance testing."""
    return {
        "benchmark_rounds": 10,
        "warmup_rounds": 3,
        "timeout": 60,
        "memory_limit_mb": 1024
    }


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def benchmark_data_sizes() -> list:
    """Different data sizes for benchmarking."""
    return [10, 50, 100, 500, 1000]


# Performance test markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: mark test as memory intensive"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        # Skip slow tests unless explicitly requested
        if "slow" in item.keywords and not config.getoption("--slow", default=False):
            item.add_marker(skip_slow)
            
        # Skip GPU tests if CUDA not available
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--slow",
        action="store_true", 
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--benchmark-only",
        action="store_true",
        default=False,
        help="only run benchmark tests"
    )