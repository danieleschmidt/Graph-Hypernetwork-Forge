"""Text encoder modules for the Graph Hypernetwork Forge.

This module contains the TextEncoder class that was originally part of hypergnn.py
but has been extracted for better modularity and testing.
"""

from typing import List
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """Encodes text descriptions into embeddings."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        freeze_encoder: bool = False,
    ):
        """Initialize text encoder.
        
        Args:
            model_name: Pre-trained model name or path
            embedding_dim: Output embedding dimension
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.freeze_encoder = freeze_encoder
        
        # Initialize encoder based on model type
        if "sentence-transformers" in model_name:
            self.encoder = SentenceTransformer(model_name)
            self.is_sentence_transformer = True
            # Get actual embedding dimension
            self.input_dim = self.encoder.get_sentence_embedding_dimension()
        else:
            # Use Hugging Face transformers
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
            self.is_sentence_transformer = False
            self.input_dim = self.encoder.config.hidden_size
        
        # Projection layer if dimensions don't match
        if self.input_dim != embedding_dim:
            self.projection = nn.Linear(self.input_dim, embedding_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        if self.is_sentence_transformer:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into embeddings.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            Text embeddings [batch_size, embedding_dim]
        """
        if self.is_sentence_transformer:
            # Use sentence-transformers
            if self.freeze_encoder:
                with torch.no_grad():
                    embeddings = self.encoder.encode(
                        texts, convert_to_tensor=True, show_progress_bar=False
                    )
                    # Clone to make it compatible with autograd
                    embeddings = embeddings.clone().detach().requires_grad_(False)
            else:
                embeddings = self.encoder.encode(
                    texts, convert_to_tensor=True, show_progress_bar=False
                )
                # Ensure autograd compatibility
                if not embeddings.requires_grad:
                    embeddings = embeddings.clone().requires_grad_(True)
        else:
            # Use transformers
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(next(self.encoder.parameters()).device) for k, v in inputs.items()}
            
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                outputs = self.encoder(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Apply projection
        embeddings = self.projection(embeddings)
        return embeddings


class SentenceTransformerEncoder(TextEncoder):
    """Specialized sentence transformer encoder."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = None,
        freeze_encoder: bool = False,
    ):
        """Initialize sentence transformer encoder.
        
        Args:
            model_name: Sentence transformer model name
            embedding_dim: Output embedding dimension (None for default)
            freeze_encoder: Whether to freeze encoder weights
        """
        # Force sentence transformer usage
        if not model_name.startswith("sentence-transformers/"):
            model_name = f"sentence-transformers/{model_name}"
        
        # Get default embedding dimension if not specified
        if embedding_dim is None:
            temp_encoder = SentenceTransformer(model_name)
            embedding_dim = temp_encoder.get_sentence_embedding_dimension()
            del temp_encoder
        
        super().__init__(model_name, embedding_dim, freeze_encoder)