"""Performance and benchmark tests for graph hypernetwork forge."""
import pytest
import torch
import time
import psutil
import os
from typing import Dict, Any
from memory_profiler import profile


class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""

    @pytest.mark.performance
    @pytest.mark.benchmark(group="model_initialization")
    def test_model_initialization_benchmark(self, benchmark, device):
        """Benchmark model initialization time."""
        def init_model():
            # Mock model initialization
            model = torch.nn.Sequential(
                torch.nn.Linear(128, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 64)
            ).to(device)
            return model
        
        result = benchmark(init_model)
        assert result is not None
        
    @pytest.mark.performance
    @pytest.mark.benchmark(group="forward_pass")
    def test_forward_pass_benchmark(self, benchmark, large_graph_data, device):
        """Benchmark forward pass performance."""
        # Mock model for benchmarking
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64)
        ).to(device)
        
        node_features = large_graph_data["node_features"].to(device)
        
        def forward_pass():
            with torch.no_grad():
                return model(node_features)
        
        result = benchmark(forward_pass)
        assert result.shape[0] == large_graph_data["num_nodes"]
        
    @pytest.mark.performance
    @pytest.mark.benchmark(group="text_encoding")
    def test_text_encoding_benchmark(self, benchmark, large_graph_data):
        """Benchmark text encoding performance."""
        texts = large_graph_data["node_texts"]
        
        def encode_texts():
            # Mock text encoding
            encoded = []
            for text in texts:
                # Simulate encoding computation
                encoded.append(torch.randn(384))  # Typical sentence transformer size
            return torch.stack(encoded)
        
        result = benchmark(encode_texts)
        assert result.shape[0] == len(texts)
        
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.parametrize("num_nodes", [100, 500, 1000, 5000])
    def test_scalability_benchmark(self, benchmark, num_nodes, device):
        """Test performance scalability with different graph sizes."""
        # Generate graph of specified size
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        node_features = torch.randn(num_nodes, 128).to(device)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64)
        ).to(device)
        
        def process_graph():
            with torch.no_grad():
                return model(node_features)
        
        result = benchmark.pedantic(process_graph, rounds=5, iterations=1)
        assert result.shape[0] == num_nodes
        
    @pytest.mark.performance
    @pytest.mark.memory_intensive
    def test_memory_usage(self, large_graph_data, device):
        """Test memory usage during model operations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model and data
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64)
        ).to(device)
        
        node_features = large_graph_data["node_features"].to(device)
        
        # Process data
        with torch.no_grad():
            result = model(node_features)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Assert reasonable memory usage (less than 500MB increase)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.2f}MB"
        assert result is not None
        
    @pytest.mark.performance
    @pytest.mark.gpu
    def test_gpu_performance(self, benchmark, large_graph_data):
        """Test GPU performance if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = torch.device("cuda")
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64)
        ).to(device)
        
        node_features = large_graph_data["node_features"].to(device)
        
        def gpu_forward():
            with torch.no_grad():
                result = model(node_features)
                torch.cuda.synchronize()  # Ensure computation completes
                return result
        
        result = benchmark(gpu_forward)
        assert result.device.type == "cuda"
        
    @pytest.mark.performance
    @pytest.mark.benchmark(group="batch_processing")
    def test_batch_processing_benchmark(self, benchmark, benchmark_data_sizes, device):
        """Benchmark batch processing performance."""
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64)
        ).to(device)
        
        # Use largest batch size for benchmark
        batch_size = max(benchmark_data_sizes)
        batch_data = torch.randn(batch_size, 128).to(device)
        
        def process_batch():
            with torch.no_grad():
                return model(batch_data)
        
        result = benchmark(process_batch)
        assert result.shape[0] == batch_size


class TestMemoryProfiling:
    """Memory profiling tests."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_profile_large_model(self, large_graph_data, device):
        """Profile memory usage of large model operations."""
        
        @profile
        def memory_intensive_operation():
            # Create large model
            model = torch.nn.Sequential(
                torch.nn.Linear(128, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 64)
            ).to(device)
            
            node_features = large_graph_data["node_features"].to(device)
            
            # Process multiple times to observe memory patterns
            results = []
            for _ in range(10):
                with torch.no_grad():
                    result = model(node_features)
                    results.append(result)
            
            return results
        
        results = memory_intensive_operation()
        assert len(results) == 10
        

class TestConcurrencyPerformance:
    """Test concurrent operations performance."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_concurrent_processing(self, large_graph_data, device):
        """Test performance of concurrent graph processing."""
        import concurrent.futures
        import threading
        
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64)
        ).to(device)
        
        def process_chunk(chunk_data):
            with torch.no_grad():
                return model(chunk_data)
        
        # Split data into chunks
        node_features = large_graph_data["node_features"].to(device)
        chunk_size = len(node_features) // 4
        chunks = [node_features[i:i+chunk_size] for i in range(0, len(node_features), chunk_size)]
        
        start_time = time.time()
        
        # Process chunks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(results) == len(chunks)
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        # Verify results
        total_processed = sum(len(result) for result in results)
        assert total_processed <= len(node_features)  # Allow for remainder


@pytest.mark.performance
class TestResourceUtilization:
    """Test system resource utilization during operations."""
    
    def test_cpu_utilization(self, large_graph_data, device):
        """Monitor CPU utilization during processing."""
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64)
        ).to(device)
        
        node_features = large_graph_data["node_features"].to(device)
        
        # Monitor CPU usage
        cpu_before = psutil.cpu_percent(interval=1)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):  # Multiple iterations
                result = model(node_features)
        end_time = time.time()
        
        cpu_after = psutil.cpu_percent(interval=1)
        processing_time = end_time - start_time
        
        # Verify reasonable performance
        assert processing_time < 30.0  # Should complete within 30 seconds
        assert result is not None
        
        print(f"CPU usage: {cpu_before}% -> {cpu_after}%")
        print(f"Processing time: {processing_time:.2f}s")


# Performance test configuration
def pytest_benchmark_update_json(config, benchmarks, output_json):
    """Add custom fields to benchmark JSON output."""
    output_json["system_info"] = {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3)
    }
    
    if torch.cuda.is_available():
        output_json["gpu_info"] = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }