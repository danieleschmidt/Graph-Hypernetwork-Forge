"""Intelligent caching utilities for Graph Hypernetwork Forge."""

import hashlib
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import threading
import time
from functools import wraps
import logging


logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Thread-safe cache for text embeddings with LRU eviction."""
    
    def __init__(
        self, 
        max_size: int = 10000, 
        ttl_seconds: int = 3600,
        cache_dir: Optional[Path] = None
    ):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time to live for cached items
            cache_dir: Optional persistent cache directory
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order = []
        self._lock = threading.RLock()
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
    
    def _get_text_hash(self, text: str, model_name: str) -> str:
        """Generate hash for text and model combination."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _is_expired(self, item: Dict[str, Any]) -> bool:
        """Check if cache item has expired."""
        return time.time() - item['timestamp'] > self.ttl_seconds
    
    def _evict_lru(self):
        """Remove least recently used item."""
        with self._lock:
            if not self._access_order:
                return
            
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
    
    def _cleanup_expired(self):
        """Remove expired items from cache."""
        with self._lock:
            expired_keys = [
                key for key, item in self._cache.items()
                if self._is_expired(item)
            ]
            
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
    
    def get(
        self, 
        text: str, 
        model_name: str = "default"
    ) -> Optional[torch.Tensor]:
        """Get cached embedding for text.
        
        Args:
            text: Text to lookup
            model_name: Model name used for embedding
            
        Returns:
            Cached embedding tensor or None if not found
        """
        key = self._get_text_hash(text, model_name)
        
        with self._lock:
            if key not in self._cache:
                return None
            
            item = self._cache[key]
            if self._is_expired(item):
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return item['embedding'].clone()
    
    def put(
        self, 
        text: str, 
        embedding: torch.Tensor, 
        model_name: str = "default"
    ):
        """Cache embedding for text.
        
        Args:
            text: Text being cached
            embedding: Embedding tensor to cache
            model_name: Model name used for embedding
        """
        key = self._get_text_hash(text, model_name)
        
        with self._lock:
            # Remove expired items
            self._cleanup_expired()
            
            # Evict LRU if at capacity
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            self._cache[key] = {
                'embedding': embedding.detach().cpu(),
                'timestamp': time.time(),
                'text': text,
                'model_name': model_name
            }
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
    
    def get_batch(
        self, 
        texts: list, 
        model_name: str = "default"
    ) -> Tuple[list, list]:
        """Get batch of cached embeddings.
        
        Args:
            texts: List of texts to lookup
            model_name: Model name used for embedding
            
        Returns:
            Tuple of (cached_embeddings, missing_texts)
        """
        cached_embeddings = []
        missing_texts = []
        
        for text in texts:
            embedding = self.get(text, model_name)
            if embedding is not None:
                cached_embeddings.append(embedding)
            else:
                missing_texts.append(text)
                cached_embeddings.append(None)
        
        return cached_embeddings, missing_texts
    
    def put_batch(
        self, 
        texts: list, 
        embeddings: torch.Tensor, 
        model_name: str = "default"
    ):
        """Cache batch of embeddings.
        
        Args:
            texts: List of texts
            embeddings: Batch of embedding tensors [batch_size, embed_dim]
            model_name: Model name used for embedding
        """
        for i, text in enumerate(texts):
            self.put(text, embeddings[i], model_name)
    
    def _load_persistent_cache(self):
        """Load persistent cache from disk."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if not cache_file.exists():
            return
        
        try:
            with open(cache_file, 'rb') as f:
                persistent_data = pickle.load(f)
            
            # Load non-expired items
            current_time = time.time()
            for key, item in persistent_data.items():
                if current_time - item['timestamp'] < self.ttl_seconds:
                    self._cache[key] = item
                    self._access_order.append(key)
            
            logger.info(f"Loaded {len(self._cache)} items from persistent cache")
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
    
    def save_persistent_cache(self):
        """Save cache to disk."""
        if not self.cache_dir:
            return
        
        try:
            cache_file = self.cache_dir / "embedding_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
            logger.info(f"Saved {len(self._cache)} items to persistent cache")
        except Exception as e:
            logger.warning(f"Failed to save persistent cache: {e}")
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for item in self._cache.values()
                if current_time - item['timestamp'] > self.ttl_seconds
            )
            
            return {
                'total_items': len(self._cache),
                'expired_items': expired_count,
                'max_size': self.max_size,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_request_count', 1), 1)
            }


class WeightCache:
    """Cache for generated GNN weights."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize weight cache.
        
        Args:
            max_size: Maximum number of cached weight sets
        """
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order = []
        self._lock = threading.RLock()
    
    def _get_weights_hash(
        self, 
        texts: list, 
        model_config: Dict[str, Any]
    ) -> str:
        """Generate hash for text list and model configuration."""
        # Sort texts for consistent hashing
        sorted_texts = sorted(texts)
        config_str = str(sorted(model_config.items()))
        content = f"{config_str}:{':'.join(sorted_texts)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]  # Shorter hash
    
    def get(
        self, 
        texts: list, 
        model_config: Dict[str, Any]
    ) -> Optional[list]:
        """Get cached weights for text list and model config.
        
        Args:
            texts: List of node texts
            model_config: Model configuration dictionary
            
        Returns:
            Cached weights or None if not found
        """
        key = self._get_weights_hash(texts, model_config)
        
        with self._lock:
            if key not in self._cache:
                return None
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            # Deep copy tensors to avoid reference issues
            cached_weights = self._cache[key]['weights']
            return [
                {k: v.clone() for k, v in layer.items()}
                for layer in cached_weights
            ]
    
    def put(
        self, 
        texts: list, 
        model_config: Dict[str, Any], 
        weights: list
    ):
        """Cache generated weights.
        
        Args:
            texts: List of node texts
            model_config: Model configuration dictionary  
            weights: Generated weights to cache
        """
        key = self._get_weights_hash(texts, model_config)
        
        with self._lock:
            # Evict LRU if at capacity
            while len(self._cache) >= self.max_size:
                if not self._access_order:
                    break
                lru_key = self._access_order.pop(0)
                if lru_key in self._cache:
                    del self._cache[lru_key]
            
            # Deep copy tensors for caching
            cached_weights = [
                {k: v.detach().cpu() for k, v in layer.items()}
                for layer in weights
            ]
            
            self._cache[key] = {
                'weights': cached_weights,
                'timestamp': time.time()
            }
            
            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)


def cached_embedding(cache: EmbeddingCache):
    """Decorator for caching text embeddings."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, texts, *args, **kwargs):
            if not isinstance(texts, (list, tuple)):
                texts = [texts]
            
            model_name = getattr(self, 'model_name', 'default')
            
            # Check cache for batch
            cached_embeddings, missing_texts = cache.get_batch(texts, model_name)
            
            if not missing_texts:
                # All embeddings cached
                result = torch.stack([emb for emb in cached_embeddings if emb is not None])
                return result
            
            # Compute missing embeddings
            if missing_texts:
                missing_result = func(self, missing_texts, *args, **kwargs)
                cache.put_batch(missing_texts, missing_result, model_name)
                
                # Combine cached and computed results
                result_list = []
                missing_idx = 0
                for cached_emb in cached_embeddings:
                    if cached_emb is not None:
                        result_list.append(cached_emb)
                    else:
                        result_list.append(missing_result[missing_idx])
                        missing_idx += 1
                
                result = torch.stack(result_list)
                return result
            
            return func(self, texts, *args, **kwargs)
        return wrapper
    return decorator


# Global cache instances
_embedding_cache = None
_weight_cache = None


def get_embedding_cache(
    max_size: int = 10000,
    ttl_seconds: int = 3600,
    cache_dir: Optional[Union[str, Path]] = None
) -> EmbeddingCache:
    """Get global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(max_size, ttl_seconds, cache_dir)
    return _embedding_cache


def get_weight_cache(max_size: int = 1000) -> WeightCache:
    """Get global weight cache instance."""
    global _weight_cache
    if _weight_cache is None:
        _weight_cache = WeightCache(max_size)
    return _weight_cache


def clear_all_caches():
    """Clear all global caches."""
    global _embedding_cache, _weight_cache
    if _embedding_cache:
        _embedding_cache.clear()
    if _weight_cache:
        _weight_cache.clear()