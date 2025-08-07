"""Utility functions and classes."""

from .training import HyperGNNTrainer, ZeroShotEvaluator
from .datasets import SyntheticDataGenerator, DatasetSplitter, create_sample_datasets
from . import graph_utils, text_utils, model_utils, evaluation_utils
from . import caching, batch_processing, profiling

# Import key optimization classes
from .caching import EmbeddingCache, WeightCache, get_embedding_cache, get_weight_cache
from .batch_processing import BatchProcessor, GraphBatcher, TextBatcher, auto_batch_size
from .profiling import PerformanceProfiler, ModelProfiler, get_profiler, profile

__all__ = [
    "HyperGNNTrainer",
    "ZeroShotEvaluator", 
    "SyntheticDataGenerator",
    "DatasetSplitter",
    "create_sample_datasets",
    "graph_utils",
    "text_utils", 
    "model_utils",
    "evaluation_utils",
    "caching",
    "batch_processing", 
    "profiling",
    "EmbeddingCache",
    "WeightCache",
    "get_embedding_cache",
    "get_weight_cache",
    "BatchProcessor",
    "GraphBatcher", 
    "TextBatcher",
    "auto_batch_size",
    "PerformanceProfiler",
    "ModelProfiler",
    "get_profiler",
    "profile",
]