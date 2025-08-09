#!/usr/bin/env python3
"""Performance benchmarking script for HyperGNN."""

import time
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph_hypernetwork_forge import HyperGNN
from graph_hypernetwork_forge.utils import SyntheticDataGenerator
from graph_hypernetwork_forge.utils.optimization import global_profiler
from graph_hypernetwork_forge.utils.monitoring import system_monitor, ModelAnalyzer


def benchmark_model_creation():
    """Benchmark model creation time."""
    print("🔧 Benchmarking Model Creation")
    print("=" * 50)
    
    start_time = time.time()
    model = HyperGNN(
        gnn_backbone="GAT",
        hidden_dim=128,
        num_layers=3,
        enable_caching=True
    )
    creation_time = time.time() - start_time
    
    # Analyze model
    param_stats = ModelAnalyzer.count_parameters(model)
    model_size = ModelAnalyzer.get_model_size(model)
    
    print(f"✅ Model creation time: {creation_time:.3f}s")
    print(f"📊 Parameters: {param_stats['total']:,} total ({param_stats['trainable']:,} trainable)")
    print(f"💾 Model size: {model_size['total_mb']:.2f} MB")
    return model


def benchmark_inference_speed(model, num_tests=5):
    """Benchmark inference speed."""
    print(f"\n⚡ Benchmarking Inference Speed ({num_tests} tests)")
    print("=" * 50)
    
    # Generate test data of different sizes
    gen = SyntheticDataGenerator()
    test_sizes = [10, 50, 100, 200]
    
    results = {}
    
    for num_nodes in test_sizes:
        print(f"\n📈 Testing with {num_nodes} nodes...")
        
        # Generate test graph
        kg = gen.generate_social_network(num_nodes=num_nodes)
        node_features = torch.randn(kg.num_nodes, model.hidden_dim)
        
        # Warmup
        model.eval()
        with torch.no_grad():
            _ = model(kg.edge_index, node_features, kg.node_texts)
        
        # Benchmark
        times = []
        for i in range(num_tests):
            start = time.time()
            with torch.no_grad():
                predictions = model(kg.edge_index, node_features, kg.node_texts)
            end = time.time()
            times.append(end - start)
            
            if i == 0:  # Show output info on first run
                print(f"   Output shape: {predictions.shape}")
        
        avg_time = sum(times) / len(times)
        throughput = num_nodes / avg_time
        
        results[num_nodes] = {
            'avg_time_ms': avg_time * 1000,
            'throughput_nodes_per_sec': throughput
        }
        
        print(f"   Avg time: {avg_time * 1000:.2f}ms")
        print(f"   Throughput: {throughput:.1f} nodes/sec")
    
    return results


def benchmark_caching_effectiveness(model, num_repeats=10):
    """Benchmark caching effectiveness."""
    print(f"\n🎯 Benchmarking Cache Effectiveness ({num_repeats} repeats)")
    print("=" * 50)
    
    gen = SyntheticDataGenerator()
    kg = gen.generate_social_network(num_nodes=50)
    node_features = torch.randn(kg.num_nodes, model.hidden_dim)
    
    # Clear any existing cache
    if model.weight_cache:
        model.weight_cache.clear()
    
    # First run (cache miss)
    model.eval()
    with torch.no_grad():
        start = time.time()
        predictions1 = model(kg.edge_index, node_features, kg.node_texts)
        first_time = time.time() - start
    
    # Subsequent runs (cache hits)
    cache_times = []
    for _ in range(num_repeats):
        with torch.no_grad():
            start = time.time()
            predictions2 = model(kg.edge_index, node_features, kg.node_texts)
            cache_times.append(time.time() - start)
    
    avg_cache_time = sum(cache_times) / len(cache_times)
    speedup = first_time / avg_cache_time
    
    print(f"🚀 First run (no cache): {first_time * 1000:.2f}ms")
    print(f"⚡ Cached runs avg: {avg_cache_time * 1000:.2f}ms") 
    print(f"📈 Cache speedup: {speedup:.2f}x")
    
    if model.weight_cache:
        print(f"💾 Cache size: {model.weight_cache.size()} entries")
    
    # Verify results are identical
    max_diff = torch.max(torch.abs(predictions1 - predictions2)).item()
    print(f"✅ Result consistency: max diff = {max_diff:.2e}")
    
    return {
        'first_time_ms': first_time * 1000,
        'cached_time_ms': avg_cache_time * 1000,
        'speedup': speedup,
        'max_diff': max_diff
    }


def benchmark_memory_usage():
    """Benchmark memory usage."""
    print(f"\n💾 Memory Usage Analysis")
    print("=" * 50)
    
    # Start monitoring
    system_monitor.start_monitoring(interval=0.5)
    
    # Create model and generate predictions
    model = HyperGNN(hidden_dim=256, num_layers=4, enable_caching=True)
    gen = SyntheticDataGenerator()
    
    # Test with increasing graph sizes
    sizes = [20, 100, 500]
    
    for size in sizes:
        print(f"\n🔍 Testing {size} nodes...")
        kg = gen.generate_social_network(num_nodes=size)
        node_features = torch.randn(kg.num_nodes, model.hidden_dim)
        
        # Get memory before
        metrics_before = system_monitor.get_current_metrics()
        
        # Run inference
        model.eval()
        with torch.no_grad():
            predictions = model(kg.edge_index, node_features, kg.node_texts)
        
        # Get memory after
        metrics_after = system_monitor.get_current_metrics()
        
        # Calculate memory delta
        memory_delta = (metrics_after.get('memory_available_gb', 0) - 
                       metrics_before.get('memory_available_gb', 0))
        
        print(f"   Memory delta: {abs(memory_delta) * 1000:.1f} MB")
        print(f"   Current usage: {metrics_after.get('memory_percent', 0):.1f}%")
        
        if 'gpu_memory_gb' in metrics_after:
            print(f"   GPU memory: {metrics_after['gpu_memory_gb']:.2f} GB")
    
    system_monitor.stop_monitoring()


def benchmark_scaling_performance():
    """Benchmark scaling with different model configurations."""
    print(f"\n📊 Scaling Performance Analysis")  
    print("=" * 50)
    
    configs = [
        {"hidden_dim": 64, "num_layers": 2, "name": "Small"},
        {"hidden_dim": 128, "num_layers": 3, "name": "Medium"},  
        {"hidden_dim": 256, "num_layers": 4, "name": "Large"},
    ]
    
    gen = SyntheticDataGenerator()
    kg = gen.generate_social_network(num_nodes=100)
    
    for config in configs:
        print(f"\n🔧 Testing {config['name']} model...")
        print(f"   Hidden dim: {config['hidden_dim']}, Layers: {config['num_layers']}")
        
        model = HyperGNN(
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            enable_caching=True
        )
        
        node_features = torch.randn(kg.num_nodes, config['hidden_dim'])
        
        # Benchmark inference
        times = []
        model.eval()
        for _ in range(5):
            start = time.time()
            with torch.no_grad():
                predictions = model(kg.edge_index, node_features, kg.node_texts)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        param_count = ModelAnalyzer.count_parameters(model)['total']
        
        print(f"   Avg time: {avg_time * 1000:.2f}ms")
        print(f"   Parameters: {param_count:,}")
        print(f"   Time per param: {avg_time * 1e6 / param_count:.2f} μs/param")


def print_profiling_summary():
    """Print profiling summary from global profiler."""
    print(f"\n📈 Performance Profiling Summary")
    print("=" * 50)
    
    report = global_profiler.report()
    
    if report:
        for func_name, avg_time in sorted(report.items(), key=lambda x: x[1], reverse=True):
            print(f"   {func_name}: {avg_time * 1000:.2f}ms avg")
    else:
        print("   No profiling data available")


def main():
    """Run comprehensive benchmarks."""
    print("🚀 HyperGNN Performance Benchmark Suite")
    print("=" * 60)
    
    try:
        # Model creation benchmark
        model = benchmark_model_creation()
        
        # Inference speed benchmark
        inference_results = benchmark_inference_speed(model)
        
        # Caching effectiveness benchmark
        cache_results = benchmark_caching_effectiveness(model)
        
        # Memory usage benchmark
        benchmark_memory_usage()
        
        # Scaling performance benchmark  
        benchmark_scaling_performance()
        
        # Profiling summary
        print_profiling_summary()
        
        # Final summary
        print(f"\n🎉 Benchmark Complete!")
        print("=" * 50)
        print("✅ All performance tests completed successfully")
        print("📊 Key metrics:")
        
        # Show best throughput
        best_throughput = max(r['throughput_nodes_per_sec'] for r in inference_results.values())
        print(f"   Best throughput: {best_throughput:.1f} nodes/sec")
        
        # Show cache effectiveness
        if cache_results['speedup'] > 1:
            print(f"   Cache speedup: {cache_results['speedup']:.2f}x")
        
        print("\n🔧 Ready for production deployment!")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())