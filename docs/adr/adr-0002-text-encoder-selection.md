# ADR-0002: Text Encoder Framework Selection

**Date**: 2025-08-01  
**Status**: Accepted  
**Deciders**: Daniel Schmidt, Core Development Team  

## Context and Problem Statement

The hypernetwork architecture requires high-quality text embeddings from node descriptions to generate effective GNN weights. The choice of text encoder significantly impacts the system's ability to understand semantic relationships and generate appropriate neural network parameters.

We need to select a text encoding framework that balances semantic understanding, computational efficiency, and integration simplicity while supporting diverse domains and languages.

## Decision Drivers

- **Semantic Quality**: Rich semantic representations for diverse text descriptions
- **Computational Efficiency**: Fast encoding for large-scale graphs
- **Domain Adaptability**: Performance across different knowledge domains
- **Integration Ease**: Simple Python API and PyTorch compatibility
- **Model Variety**: Support for different encoder architectures
- **Community Support**: Active development and maintenance

## Considered Options

### Option 1: Raw Transformer Models (Hugging Face)
- Use BERT/RoBERTa/DeBERTa directly via Hugging Face Transformers
- **Pros**: Maximum flexibility, latest models, fine-tuning control
- **Cons**: Complex preprocessing, requires custom pooling strategies

### Option 2: Sentence Transformers (Chosen)
- Use Sentence Transformers library with pre-trained sentence encoders
- **Pros**: Optimized for sentence-level embeddings, easy integration, good performance
- **Cons**: Limited fine-tuning options, dependency on external library

### Option 3: OpenAI Embeddings API
- Use OpenAI's text-embedding-ada-002 or similar models
- **Pros**: State-of-the-art quality, no local compute requirements
- **Cons**: API dependency, cost implications, privacy concerns

### Option 4: Custom Sentence Encoder
- Build custom text encoder from scratch using PyTorch
- **Pros**: Full control, domain-specific optimization
- **Cons**: High development effort, limited pretrained knowledge

## Decision Outcome

**Chosen option**: Sentence Transformers (Option 2) with Hugging Face fallback

**Rationale**: Sentence Transformers provides the optimal balance of performance, ease of use, and flexibility. It offers high-quality sentence embeddings out-of-the-box while maintaining the option to use any Hugging Face model as a backend.

### Positive Consequences
- **Easy Integration**: Simple API with minimal preprocessing required
- **High Performance**: Optimized for sentence-level semantic similarity
- **Model Variety**: Access to 100+ pre-trained models
- **Batch Processing**: Efficient batch encoding for large graphs
- **Domain Coverage**: Models trained on diverse text corpora
- **Flexible Backends**: Can use BERT, RoBERTa, MPNet, and others

### Negative Consequences
- **External Dependency**: Reliance on sentence-transformers package
- **Limited Fine-Tuning**: Less control over training process
- **Model Size**: Some models are large (400MB+)
- **Version Compatibility**: Need to manage version dependencies

## Implementation Notes

### Primary Models
```python
# Default models by use case
DEFAULT_MODELS = {
    "fast": "all-MiniLM-L6-v2",        # 80MB, good performance
    "balanced": "all-mpnet-base-v2",    # 420MB, better performance  
    "multilingual": "paraphrase-multilingual-mpnet-base-v2",
    "domain_specific": "allenai-specter"  # Scientific domains
}
```

### Architecture Integration
```python
from sentence_transformers import SentenceTransformer

class TextEncoder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        return embeddings
```

### Fallback Strategy
- Primary: Sentence Transformers with specified model
- Fallback 1: Hugging Face Transformers with mean pooling
- Fallback 2: TF-IDF vectorization for basic functionality

### Model Selection Criteria
- **Small Graphs (<1K nodes)**: all-mpnet-base-v2 (best quality)
- **Large Graphs (>10K nodes)**: all-MiniLM-L6-v2 (speed optimized)
- **Multilingual**: paraphrase-multilingual-mpnet-base-v2
- **Scientific Domains**: allenai-specter

## Performance Benchmarks

### Encoding Speed (1000 sentences)
- all-MiniLM-L6-v2: ~2 seconds (GPU), ~8 seconds (CPU)
- all-mpnet-base-v2: ~5 seconds (GPU), ~20 seconds (CPU)
- BERT-base-uncased: ~8 seconds (GPU), ~35 seconds (CPU)

### Memory Requirements
- all-MiniLM-L6-v2: 80MB model + embedding memory
- all-mpnet-base-v2: 420MB model + embedding memory
- Embedding memory: batch_size × 384/768 × 4 bytes

### Quality Metrics (STS Benchmark)
- all-mpnet-base-v2: 86.9 Spearman correlation
- all-MiniLM-L6-v2: 82.1 Spearman correlation
- BERT-base-uncased: 77.8 Spearman correlation

## Configuration Management

### Environment Variables
```bash
# Model selection
HYPERGNN_TEXT_MODEL=all-mpnet-base-v2
HYPERGNN_CACHE_DIR=~/.cache/sentence_transformers
HYPERGNN_DEVICE=cuda

# Performance tuning
HYPERGNN_BATCH_SIZE=32
HYPERGNN_MAX_LENGTH=512
```

### Runtime Configuration
```python
encoder_config = {
    "model_name": "all-mpnet-base-v2",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "normalize_embeddings": True,
    "show_progress_bar": False
}
```

## Extension Points

### Custom Domain Encoders
```python
class DomainSpecificEncoder(TextEncoder):
    def __init__(self, base_model: str, domain_vocab: Dict[str, int]):
        super().__init__(base_model)
        self.domain_embeddings = nn.Embedding(len(domain_vocab), 768)
        
    def encode(self, texts: List[str]) -> torch.Tensor:
        base_embeddings = super().encode(texts)
        domain_embeddings = self.get_domain_embeddings(texts)
        return torch.cat([base_embeddings, domain_embeddings], dim=-1)
```

### Multi-Modal Integration
```python
class MultiModalEncoder(TextEncoder):
    def encode_multimodal(self, 
                         texts: List[str],
                         images: Optional[List[PIL.Image]] = None,
                         numerical: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Implementation for multi-modal encoding
        pass
```

## Validation Strategy

### Unit Tests
- Text encoding correctness
- Batch processing consistency
- Memory usage validation
- Device placement verification

### Integration Tests
- End-to-end hypernetwork training
- Cross-domain transfer evaluation
- Performance regression testing

### Benchmark Evaluation
- Semantic similarity tasks (STS)
- Knowledge graph completion tasks
- Zero-shot transfer performance

## Links and References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [MPNet Paper](https://arxiv.org/abs/2004.09297)
- [Text Embeddings Benchmark](https://github.com/embeddings-benchmark/mteb)

## Future Considerations

- **Custom Fine-Tuning**: Domain-specific encoder training
- **Multilingual Support**: Enhanced support for non-English texts
- **Compression Techniques**: Model distillation for mobile deployment
- **Dynamic Model Selection**: Automatic model selection based on text characteristics