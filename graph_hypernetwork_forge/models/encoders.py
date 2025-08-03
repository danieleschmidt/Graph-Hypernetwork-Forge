"""Text encoders for converting textual descriptions to embeddings.

Supports various pre-trained language models and sentence transformers
for encoding node descriptions into semantic embeddings.
"""

import torch
import torch.nn as nn
from typing import List, Union, Optional, Dict, Any
from abc import ABC, abstractmethod
import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextEncoder(ABC):
    """Abstract base class for text encoders."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode list of texts into embeddings."""
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        pass


class SentenceTransformerEncoder(TextEncoder):
    """Text encoder using sentence-transformers library."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
        max_length: int = 512,
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required but not installed")
            
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or torch.device('cpu')
        
        logger.info(f"Loading SentenceTransformer: {model_name}")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self._output_dim = self.model.get_sentence_embedding_dimension()
        
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using sentence transformer."""
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )
        return embeddings
    
    def get_output_dim(self) -> int:
        return self._output_dim


class TransformerEncoder(TextEncoder):
    """Text encoder using HuggingFace transformers with mean pooling."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[torch.device] = None,
        max_length: int = 512,
        pooling: str = "mean",  # 'mean', 'cls', 'max'
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required but not installed")
            
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = device or torch.device('cpu')
        
        logger.info(f"Loading Transformer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._output_dim = self.model.config.hidden_size
        
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using transformer with pooling."""
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
        # Apply pooling
        if self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        elif self.pooling == "mean":
            # Mean pooling with attention mask
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling == "max":
            # Max pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            embeddings = torch.max(token_embeddings, 1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
            
        return embeddings
    
    def get_output_dim(self) -> int:
        return self._output_dim


class CustomTextEncoder(TextEncoder):
    """Base class for custom text encoders."""
    
    def __init__(self, model_name: str, output_dim: int):
        self.model_name = model_name
        self._output_dim = output_dim
        
    def get_output_dim(self) -> int:
        return self._output_dim


class DomainSpecificEncoder(CustomTextEncoder):
    """Example domain-specific encoder with custom vocabulary embeddings."""
    
    def __init__(
        self,
        base_model: str = "bert-base-uncased",
        domain_vocab: Dict[str, int] = None,
        domain_dim: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device('cpu')
        
        # Initialize base transformer
        self.base_encoder = TransformerEncoder(base_model, device=device)
        base_dim = self.base_encoder.get_output_dim()
        
        # Domain-specific embeddings
        if domain_vocab is None:
            domain_vocab = {}
        self.domain_vocab = domain_vocab
        self.domain_embeddings = nn.Embedding(len(domain_vocab), domain_dim)
        
        # Fusion layer
        self.fusion = nn.Linear(base_dim + domain_dim, base_dim)
        
        super().__init__(base_model, base_dim)
        
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts with domain-specific enhancement."""
        # Get base embeddings
        base_embs = self.base_encoder.encode(texts)
        
        # Extract domain-specific features
        domain_features = []
        for text in texts:
            # Simple keyword matching for domain features
            domain_vec = torch.zeros(len(self.domain_vocab))
            for word, idx in self.domain_vocab.items():
                if word.lower() in text.lower():
                    domain_vec[idx] = 1.0
            domain_features.append(domain_vec)
        
        if domain_features:
            domain_tensor = torch.stack(domain_features).to(self.device)
            domain_embs = self.domain_embeddings(domain_tensor.long().argmax(dim=-1))
            
            # Fuse base and domain embeddings
            combined = torch.cat([base_embs, domain_embs], dim=-1)
            embeddings = self.fusion(combined)
        else:
            embeddings = base_embs
            
        return embeddings


def get_text_encoder(
    encoder_name: str,
    device: Optional[torch.device] = None,
    **kwargs
) -> TextEncoder:
    """Factory function to create text encoders.
    
    Args:
        encoder_name: Name of the encoder or model
        device: Target device
        **kwargs: Additional arguments for encoder
        
    Returns:
        TextEncoder instance
    """
    device = device or torch.device('cpu')
    
    # Sentence Transformers models
    sentence_transformer_models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2", 
        "multi-qa-MiniLM-L6-cos-v1",
        "paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    
    if any(model in encoder_name for model in sentence_transformer_models):
        return SentenceTransformerEncoder(encoder_name, device=device, **kwargs)
    
    # Standard transformer models
    transformer_models = [
        "bert-base-uncased",
        "bert-large-uncased", 
        "roberta-base",
        "roberta-large",
        "distilbert-base-uncased",
        "albert-base-v2",
    ]
    
    if any(model in encoder_name for model in transformer_models):
        return TransformerEncoder(encoder_name, device=device, **kwargs)
    
    # Default to sentence transformer
    logger.warning(f"Unknown encoder {encoder_name}, defaulting to SentenceTransformer")
    return SentenceTransformerEncoder(encoder_name, device=device, **kwargs)


# Export commonly used encoders
AVAILABLE_ENCODERS = {
    "sentence-transformer": SentenceTransformerEncoder,
    "transformer": TransformerEncoder,
    "custom": CustomTextEncoder,
    "domain-specific": DomainSpecificEncoder,
}