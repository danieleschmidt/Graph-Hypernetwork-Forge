"""Advanced caching system for Graph Hypernetwork Forge.

Provides multi-level caching for embeddings, model weights, and computation results
with support for memory, disk, and distributed caching strategies.
"""

from .cache_manager import (
    CacheConfig,
    CacheEntry,
    MemoryCache,
    DiskCache,
    MultiLevelCache,
    EmbeddingCache,
    cached,
    cache_key,
    get_cache_manager,
    init_cache,
)

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "MemoryCache",
    "DiskCache", 
    "MultiLevelCache",
    "EmbeddingCache",
    "cached",
    "cache_key",
    "get_cache_manager",
    "init_cache",
]