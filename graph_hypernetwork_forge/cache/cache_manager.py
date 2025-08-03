"""Advanced caching system for Graph Hypernetwork Forge.

Provides multi-level caching for embeddings, model weights, and computation results
with support for memory, disk, and distributed caching strategies.
"""

import os
import pickle
import hashlib
import json
import time
import threading
from typing import Any, Optional, Dict, List, Callable, Union
from pathlib import Path
import logging
from functools import wraps
from dataclasses import dataclass, asdict
import torch
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import memcache
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    backend: str = "memory"  # memory, disk, redis, memcached
    max_size: int = 1000  # Maximum number of items
    ttl: int = 3600  # Time to live in seconds
    disk_cache_dir: str = ".cache"
    compression: bool = True
    redis_url: str = "redis://localhost:6379/0"
    memcached_servers: List[str] = None
    
    def __post_init__(self):
        if self.memcached_servers is None:
            self.memcached_servers = ["127.0.0.1:11211"]


class CacheEntry:
    """Cache entry with metadata."""
    
    def __init__(self, data: Any, timestamp: float, ttl: int):
        self.data = data
        self.timestamp = timestamp
        self.ttl = ttl
        self.access_count = 0
        self.last_access = timestamp
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl <= 0:  # Never expire
            return False
        return time.time() - self.timestamp > self.ttl
    
    def access(self):
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_access = time.time()


class MemoryCache:
    """In-memory LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
    
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        if len(self.cache) <= self.max_size:
            return
        
        # Sort by last access time
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_access
        )
        
        # Remove oldest entries
        num_to_remove = len(self.cache) - self.max_size + 1
        for key, _ in sorted_entries[:num_to_remove]:
            del self.cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            self._evict_expired()
            
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired():
                    entry.access()
                    return entry.data
                else:
                    del self.cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        with self.lock:
            cache_ttl = ttl if ttl is not None else self.ttl
            entry = CacheEntry(value, time.time(), cache_ttl)
            self.cache[key] = entry
            
            self._evict_expired()
            self._evict_lru()
    
    def delete(self, key: str):
        """Delete value from cache."""
        with self.lock:
            self.cache.pop(key, None)
    
    def clear(self):
        """Clear all cached values."""
        with self.lock:
            self.cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_entries = len(self.cache)
            total_access = sum(entry.access_count for entry in self.cache.values())
            
            return {
                "total_entries": total_entries,
                "max_size": self.max_size,
                "total_access": total_access,
                "memory_usage": self._estimate_memory_usage(),
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        total_size = 0
        for entry in self.cache.values():
            try:
                total_size += len(pickle.dumps(entry.data))
            except:
                total_size += 1000  # Rough estimate
        return total_size


class DiskCache:
    """Disk-based cache with compression support."""
    
    def __init__(self, cache_dir: str = ".cache", compression: bool = True, ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.ttl = ttl
        self.lock = threading.RLock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self.lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)
            
            if not file_path.exists() or not meta_path.exists():
                return None
            
            try:
                # Check metadata for expiration
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                if self.ttl > 0 and time.time() - metadata['timestamp'] > self.ttl:
                    # Expired, remove files
                    file_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    return None
                
                # Load data
                with open(file_path, 'rb') as f:
                    if self.compression:
                        import gzip
                        data = pickle.load(gzip.GzipFile(fileobj=f))
                    else:
                        data = pickle.load(f)
                
                # Update access time
                metadata['last_access'] = time.time()
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                
                return data
                
            except Exception as e:
                logger.warning(f"Error reading cache file {file_path}: {e}")
                # Clean up corrupted files
                file_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in disk cache."""
        with self.lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)
            
            try:
                # Save data
                with open(file_path, 'wb') as f:
                    if self.compression:
                        import gzip
                        with gzip.GzipFile(fileobj=f) as gz:
                            pickle.dump(value, gz)
                    else:
                        pickle.dump(value, f)
                
                # Save metadata
                metadata = {
                    'timestamp': time.time(),
                    'ttl': ttl if ttl is not None else self.ttl,
                    'size': file_path.stat().st_size,
                    'access_count': 0,
                    'last_access': time.time(),
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                    
            except Exception as e:
                logger.error(f"Error writing cache file {file_path}: {e}")
                # Clean up partial files
                file_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
    
    def delete(self, key: str):
        """Delete value from disk cache."""
        with self.lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_metadata_path(key)
            file_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
    
    def clear(self):
        """Clear all cached files."""
        with self.lock:
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink(missing_ok=True)
            for file_path in self.cache_dir.glob("*.meta"):
                file_path.unlink(missing_ok=True)
    
    def cleanup_expired(self):
        """Clean up expired cache files."""
        with self.lock:
            current_time = time.time()
            
            for meta_path in self.cache_dir.glob("*.meta"):
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if self.ttl > 0 and current_time - metadata['timestamp'] > metadata.get('ttl', self.ttl):
                        # Remove expired files
                        cache_file = meta_path.with_suffix('.cache')
                        meta_path.unlink(missing_ok=True)
                        cache_file.unlink(missing_ok=True)
                        
                except Exception as e:
                    logger.warning(f"Error processing metadata file {meta_path}: {e}")


class MultiLevelCache:
    """Multi-level cache combining memory and disk caching."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCache(max_size=config.max_size // 2, ttl=config.ttl)
        self.disk_cache = DiskCache(
            cache_dir=config.disk_cache_dir,
            compression=config.compression,
            ttl=config.ttl
        )
        
        # Optional distributed cache backends
        self.redis_cache = None
        self.memcached_cache = None
        
        if config.backend == "redis" and REDIS_AVAILABLE:
            self.redis_cache = redis.from_url(config.redis_url)
        elif config.backend == "memcached" and MEMCACHED_AVAILABLE:
            self.memcached_cache = memcache.Client(config.memcached_servers)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        # Try memory cache first (fastest)
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try distributed cache
        if self.redis_cache:
            try:
                cached = self.redis_cache.get(key)
                if cached:
                    value = pickle.loads(cached)
                    # Promote to memory cache
                    self.memory_cache.set(key, value)
                    return value
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        if self.memcached_cache:
            try:
                value = self.memcached_cache.get(key)
                if value is not None:
                    # Promote to memory cache
                    self.memory_cache.set(key, value)
                    return value
            except Exception as e:
                logger.warning(f"Memcached cache error: {e}")
        
        # Try disk cache (slowest but persistent)
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in multi-level cache."""
        cache_ttl = ttl if ttl is not None else self.config.ttl
        
        # Set in memory cache
        self.memory_cache.set(key, value, cache_ttl)
        
        # Set in distributed cache
        if self.redis_cache:
            try:
                serialized = pickle.dumps(value)
                if cache_ttl > 0:
                    self.redis_cache.setex(key, cache_ttl, serialized)
                else:
                    self.redis_cache.set(key, serialized)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        if self.memcached_cache:
            try:
                self.memcached_cache.set(key, value, time=cache_ttl if cache_ttl > 0 else 0)
            except Exception as e:
                logger.warning(f"Memcached cache error: {e}")
        
        # Set in disk cache for persistence
        self.disk_cache.set(key, value, cache_ttl)
    
    def delete(self, key: str):
        """Delete value from all cache levels."""
        self.memory_cache.delete(key)
        self.disk_cache.delete(key)
        
        if self.redis_cache:
            try:
                self.redis_cache.delete(key)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        if self.memcached_cache:
            try:
                self.memcached_cache.delete(key)
            except Exception as e:
                logger.warning(f"Memcached cache error: {e}")
    
    def clear(self):
        """Clear all cache levels."""
        self.memory_cache.clear()
        self.disk_cache.clear()
        
        if self.redis_cache:
            try:
                self.redis_cache.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        if self.memcached_cache:
            try:
                self.memcached_cache.flush_all()
            except Exception as e:
                logger.warning(f"Memcached cache error: {e}")


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Create a deterministic key from arguments
    key_data = {
        'args': [str(arg) for arg in args],
        'kwargs': {k: str(v) for k, v in sorted(kwargs.items())}
    }
    
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


def cached(
    cache_manager: Optional[MultiLevelCache] = None,
    ttl: Optional[int] = None,
    key_prefix: str = "",
) -> Callable:
    """Decorator for caching function results."""
    
    def decorator(func: Callable) -> Callable:
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip caching if no cache manager
            if cache_manager is None:
                return func(*args, **kwargs)
            
            # Generate cache key
            func_key = f"{key_prefix}{func.__name__}"
            arg_key = cache_key(*args, **kwargs)
            full_key = f"{func_key}:{arg_key}"
            
            # Try to get from cache
            cached_result = cache_manager.get(full_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache_manager.set(full_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class EmbeddingCache:
    """Specialized cache for text embeddings."""
    
    def __init__(self, cache_manager: MultiLevelCache):
        self.cache = cache_manager
        self.prefix = "embedding:"
    
    def get_embedding(self, text: str, model_name: str) -> Optional[torch.Tensor]:
        """Get cached embedding for text and model."""
        key = f"{self.prefix}{model_name}:{hashlib.sha256(text.encode()).hexdigest()}"
        
        cached = self.cache.get(key)
        if cached is not None:
            if isinstance(cached, np.ndarray):
                return torch.from_numpy(cached)
            return cached
        
        return None
    
    def set_embedding(self, text: str, model_name: str, embedding: torch.Tensor, ttl: Optional[int] = None):
        """Cache embedding for text and model."""
        key = f"{self.prefix}{model_name}:{hashlib.sha256(text.encode()).hexdigest()}"
        
        # Convert to numpy for better serialization
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        
        self.cache.set(key, embedding, ttl)
    
    def get_batch_embeddings(
        self, 
        texts: List[str], 
        model_name: str
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Get cached embeddings for batch of texts."""
        cached_embeddings = []
        missing_texts = []
        
        for text in texts:
            embedding = self.get_embedding(text, model_name)
            if embedding is not None:
                cached_embeddings.append(embedding)
            else:
                cached_embeddings.append(None)
                missing_texts.append(text)
        
        return cached_embeddings, missing_texts
    
    def set_batch_embeddings(
        self,
        texts: List[str],
        model_name: str,
        embeddings: List[torch.Tensor],
        ttl: Optional[int] = None,
    ):
        """Cache batch of embeddings."""
        for text, embedding in zip(texts, embeddings):
            if embedding is not None:
                self.set_embedding(text, model_name, embedding, ttl)


# Global cache instance
_global_cache: Optional[MultiLevelCache] = None


def get_cache_manager(config: Optional[CacheConfig] = None) -> MultiLevelCache:
    """Get global cache manager instance."""
    global _global_cache
    
    if _global_cache is None:
        if config is None:
            config = CacheConfig()
        _global_cache = MultiLevelCache(config)
    
    return _global_cache


def init_cache(config: CacheConfig):
    """Initialize global cache with configuration."""
    global _global_cache
    _global_cache = MultiLevelCache(config)