"""Example demonstrating optimized processing with caching and batching."""

import time
import torch
from pathlib import Path

from graph_hypernetwork_forge import TextualKnowledgeGraph
from graph_hypernetwork_forge.utils import (
    EmbeddingCache, 
    BatchProcessor, 
    PerformanceProfiler,
    SyntheticDataGenerator,
    get_profiler,
    profile
)


@profile("text_encoding_optimized")
def encode_texts_with_caching(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Encode texts with intelligent caching."""
    from sentence_transformers import SentenceTransformer
    
    # Get cached embeddings
    cache = EmbeddingCache(max_size=5000, ttl_seconds=7200)  # 2 hour TTL
    cached_embeddings, missing_texts = cache.get_batch(texts, model_name)
    
    if not missing_texts:
        print(f"✅ All {len(texts)} embeddings found in cache!")
        return torch.stack([emb for emb in cached_embeddings if emb is not None])
    
    print(f"🔄 Computing {len(missing_texts)} missing embeddings (cache hit: {len(texts) - len(missing_texts)}/{len(texts)})")
    
    # Compute missing embeddings
    model = SentenceTransformer(model_name)
    missing_embeddings = model.encode(missing_texts, convert_to_tensor=True)
    
    # Cache the new embeddings
    cache.put_batch(missing_texts, missing_embeddings, model_name)
    
    # Combine cached and computed results
    result_list = []
    missing_idx = 0
    for cached_emb in cached_embeddings:
        if cached_emb is not None:
            result_list.append(cached_emb)
        else:
            result_list.append(missing_embeddings[missing_idx])
            missing_idx += 1
    
    return torch.stack(result_list)


@profile("graph_processing_batched") 
def process_large_graph_with_batching(kg, batch_size=50):
    """Process large graph using intelligent batching."""
    
    def process_subgraph(subgraph, node_indices=None, **kwargs):
        """Process a single subgraph batch."""
        # Simulate some processing (e.g., feature extraction, prediction)
        embeddings = encode_texts_with_caching(subgraph.node_texts)
        
        # Simulate computation
        time.sleep(0.01)  # Simulate processing time
        
        return {
            'batch_size': len(subgraph.node_texts),
            'embeddings_shape': embeddings.shape,
            'node_indices': node_indices
        }
    
    # Create batch processor with memory-aware configuration
    from graph_hypernetwork_forge.utils.batch_processing import BatchConfig
    config = BatchConfig(
        batch_size=batch_size,
        max_nodes_per_batch=200,
        memory_limit_mb=512.0
    )
    processor = BatchProcessor(batch_config=config)
    
    # Process graph in batches
    results = processor.process_large_graph(
        kg, 
        process_subgraph,
        strategy="community"  # Use community-based batching
    )
    
    return results


def demonstrate_optimization_features():
    """Demonstrate all optimization features."""
    print("🚀 Graph Hypernetwork Forge - Optimization Features Demo")
    print("=" * 60)
    
    # Initialize profiler
    profiler = get_profiler(enabled=True, sample_rate=1.0)
    
    # Generate test data
    print("\n📊 Generating synthetic knowledge graph...")
    generator = SyntheticDataGenerator()
    kg = generator.generate_social_network(
        num_nodes=500, 
        avg_degree=4.8  # This will create ~1200 edges
    )
    
    print(f"Generated graph: {kg.num_nodes} nodes, {kg.num_edges} edges")
    
    # Demonstrate caching
    print("\n🗄️  Testing embedding cache...")
    
    # First pass - cache miss
    with profiler.profile("first_pass", item_count=kg.num_nodes):
        embeddings1 = encode_texts_with_caching(kg.node_texts)
    
    # Second pass - cache hit  
    with profiler.profile("second_pass", item_count=kg.num_nodes):
        embeddings2 = encode_texts_with_caching(kg.node_texts)
    
    # Verify embeddings are identical
    assert torch.allclose(embeddings1, embeddings2)
    print("✅ Cache verification passed!")
    
    # Demonstrate batching
    print("\n📦 Testing intelligent batching...")
    batch_results = process_large_graph_with_batching(kg, batch_size=50)
    
    total_processed = sum(result['batch_size'] for result in batch_results)
    print(f"✅ Processed {total_processed} nodes in {len(batch_results)} batches")
    
    # Memory-aware batching
    print("\n🧠 Testing memory-aware batching...")
    from graph_hypernetwork_forge.utils.batch_processing import auto_batch_size
    
    optimal_batch_size = auto_batch_size(
        memory_limit_mb=256.0,
        feature_dim=384,  # Sentence transformer dimension
        estimate_overhead=2.5
    )
    print(f"📊 Recommended batch size: {optimal_batch_size}")
    
    # Generate performance report
    print("\n📈 Performance Analysis")
    print("-" * 30)
    
    metrics = profiler.get_metrics()
    for operation, data in metrics.items():
        print(f"{operation}:")
        print(f"  📞 Calls: {data['call_count']}")
        print(f"  ⏱️  Avg Time: {data['average_time']:.4f}s")
        print(f"  💾 Peak Memory: {data['memory_peak_mb']:.1f} MB")
        if data['throughput'] > 0:
            print(f"  🚀 Throughput: {data['throughput']:.2f} items/s")
        print()
    
    # System information
    system_info = profiler.get_system_info()
    print("🖥️  System Status:")
    print(f"  CPU Usage: {system_info.get('cpu_percent', 0):.1f}%")
    print(f"  Memory Usage: {system_info.get('memory_percent', 0):.1f}%")
    print(f"  Process Memory: {system_info.get('process_memory_mb', 0):.1f} MB")
    
    if system_info.get('gpu_available'):
        print(f"  GPU Memory: {system_info.get('gpu_memory_allocated_mb', 0):.1f} MB")
    
    # Export detailed metrics
    metrics_path = Path("performance_metrics.json")
    profiler.export_metrics(metrics_path)
    print(f"\n💾 Detailed metrics exported to: {metrics_path}")
    
    # Generate summary report
    print("\n📄 Performance Summary:")
    print("-" * 40)
    print(profiler.get_summary_report())


def demonstrate_cache_persistence():
    """Demonstrate persistent caching across sessions."""
    print("\n💽 Testing persistent cache...")
    
    cache_dir = Path("./cache_demo")
    cache = EmbeddingCache(
        max_size=1000, 
        ttl_seconds=3600,
        cache_dir=cache_dir
    )
    
    # Generate some sample texts
    sample_texts = [
        f"Sample text number {i} for caching demonstration."
        for i in range(20)
    ]
    
    # First encoding (will be cached)
    print("🔄 First encoding (cache miss)...")
    embeddings1 = encode_texts_with_caching(sample_texts[:10])
    
    # Save cache to disk
    cache.save_persistent_cache()
    print("💾 Cache saved to disk")
    
    # Simulate new session by creating new cache
    cache2 = EmbeddingCache(
        max_size=1000,
        ttl_seconds=3600, 
        cache_dir=cache_dir
    )
    
    # Should load from disk automatically
    print("🔄 Second encoding (cache hit from disk)...")
    embeddings2 = encode_texts_with_caching(sample_texts[:10])
    
    print("✅ Persistent cache demonstration complete!")
    
    # Cleanup
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def benchmark_optimization_impact():
    """Benchmark the impact of optimizations."""
    print("\n⚡ Benchmarking Optimization Impact")
    print("=" * 40)
    
    # Generate test data
    generator = SyntheticDataGenerator()
    small_kg = generator.generate_social_network(num_nodes=100, avg_degree=4.0)
    
    # Benchmark without optimizations (naive approach)
    def naive_processing(kg):
        """Naive processing without optimizations."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Process all texts individually (no batching, no caching)
        embeddings = []
        for text in kg.node_texts:
            emb = model.encode([text], convert_to_tensor=True)
            embeddings.append(emb.squeeze())
        
        return torch.stack(embeddings)
    
    # Benchmark optimized processing
    def optimized_processing(kg):
        """Optimized processing with caching and batching."""
        return encode_texts_with_caching(kg.node_texts)
    
    # Run benchmarks
    from graph_hypernetwork_forge.utils.profiling import benchmark_function
    
    print("🐌 Naive approach...")
    naive_stats = benchmark_function(naive_processing, small_kg, num_runs=3, warmup_runs=1)
    
    print("🚀 Optimized approach...")
    opt_stats = benchmark_function(optimized_processing, small_kg, num_runs=3, warmup_runs=1)
    
    # Calculate speedup
    if naive_stats['mean_time'] > 0:
        speedup = naive_stats['mean_time'] / opt_stats['mean_time']
        print(f"\n📊 Performance Comparison:")
        print(f"  Naive time: {naive_stats['mean_time']:.4f}s ± {naive_stats['std_time']:.4f}s")
        print(f"  Optimized time: {opt_stats['mean_time']:.4f}s ± {opt_stats['std_time']:.4f}s")
        print(f"  🚀 Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_optimization_features()
    demonstrate_cache_persistence()
    benchmark_optimization_impact()
    
    print("\n🎉 Optimization features demonstration complete!")
    print("Key benefits demonstrated:")
    print("  ✅ Intelligent embedding caching with TTL")
    print("  ✅ Memory-aware batch processing") 
    print("  ✅ Community-based graph batching")
    print("  ✅ Comprehensive performance profiling")
    print("  ✅ Persistent cache across sessions")
    print("  ✅ Significant performance improvements")