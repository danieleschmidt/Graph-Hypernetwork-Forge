"""HyperGNN: relation-conditioned message passing on knowledge graphs.

Architecture overview
---------------------

1. **Text encoder** – a tiny learned bag-of-characters embedding that maps a
   relation-type string to a fixed-size vector.  This is intentionally simple
   so the whole model runs without any pretrained models or internet access.
   You can swap in a real sentence-transformer by subclassing ``TextEncoder``.

2. **WeightGenerator** – one per GNN layer.  Reads the relation embedding and
   outputs ``(W_msg, W_self, bias)`` for that layer.

3. **Message passing** – for each edge ``(u, v)`` with relation ``r``:
       message = h_u @ W_msg[r]
   then aggregated at v by mean-pooling.  Self-features are updated with W_self.

Because the *weights* come from the relation text rather than being fixed
parameters, the model can reason over relation types it has never seen during
training — zero-shot generalisation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_generator import WeightGenerator


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """Map a relation-type string to a dense vector.

    Implements a simple character-level bag-of-embeddings encoder.  Each
    character is looked up in a learnable table; the result is mean-pooled and
    projected to ``text_dim``.

    This requires no pretrained weights and runs fully offline.  Replace with
    a sentence-transformer for production use.

    Args:
        text_dim: Output embedding dimension.
        vocab_size: Character vocabulary size (128 covers printable ASCII).
        char_emb_dim: Per-character embedding dimension.
    """

    ASCII_VOCAB = 128  # printable ASCII range

    def __init__(self, text_dim: int, char_emb_dim: int = 32) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.char_emb = nn.Embedding(self.ASCII_VOCAB, char_emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(char_emb_dim, text_dim),
            nn.Tanh(),
        )

    def _tokenize(self, text: str, device: torch.device) -> torch.Tensor:
        """Convert string to ASCII code tensor, clamp to vocab range."""
        codes = [min(ord(c), self.ASCII_VOCAB - 1) for c in text]
        if not codes:
            codes = [0]
        return torch.tensor(codes, dtype=torch.long, device=device)

    def encode_one(self, text: str, device: torch.device) -> torch.Tensor:
        """Encode a single string → ``[text_dim]`` tensor."""
        ids = self._tokenize(text, device)
        emb = self.char_emb(ids).mean(dim=0)  # [char_emb_dim]
        return self.proj(emb)                  # [text_dim]

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode a list of strings → ``[len(texts), text_dim]``."""
        return torch.stack([self.encode_one(t, device) for t in texts], dim=0)


# ---------------------------------------------------------------------------
# HyperGNN
# ---------------------------------------------------------------------------

class HyperGNN(nn.Module):
    """Hypernetwork-conditioned GNN for zero-shot KG reasoning.

    Args:
        text_dim:      Dimension of relation-type text embeddings.
        node_feat_dim: Dimension of input node feature vectors.
        hidden_dim:    Hidden (and output) dimension of node embeddings.
        num_layers:    Number of message-passing layers.
        dropout:       Dropout rate on node embeddings between layers.
        char_emb_dim:  Character embedding size in the text encoder.

    Forward signature::

        out = model(node_features, edge_index, edge_texts)

    where
        ``node_features`` : FloatTensor  ``[N, node_feat_dim]``
        ``edge_index``    : LongTensor   ``[2, E]``  (source, target)
        ``edge_texts``    : List[str]    length E – relation type for each edge

    Returns
        FloatTensor ``[N, hidden_dim]`` – contextualised node embeddings.
    """

    def __init__(
        self,
        text_dim: int,
        node_feat_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        char_emb_dim: int = 32,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.text_dim = text_dim
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Text encoder (shared across all layers)
        self.text_encoder = TextEncoder(text_dim=text_dim, char_emb_dim=char_emb_dim)

        # Input projection: node_feat_dim → hidden_dim
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # One WeightGenerator per layer; all layers use hidden_dim → hidden_dim
        self.weight_generators = nn.ModuleList([
            WeightGenerator(
                text_dim=text_dim,
                d_in=hidden_dim,
                d_out=hidden_dim,
                hidden_dim=max(64, text_dim * 2),
                num_hidden=2,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Layer norms for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    # ------------------------------------------------------------------
    # message passing
    # ------------------------------------------------------------------

    def _message_passing(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        rel_weights: dict,
    ) -> torch.Tensor:
        """One round of relation-conditioned message passing.

        For each edge (u→v) with relation r we compute:
            m_{u→v} = h_u @ W_msg[r]

        Messages at each target node v are mean-pooled:
            agg_v = mean_{u: (u,v)∈E} m_{u→v}

        Final update:
            h'_v = agg_v  +  h_v @ W_self[r_dominant]

        Because each node can receive edges of different relation types, we use
        the relation weights of each *individual edge* for its message, and for
        the self-loop we use the average W_self across all incoming relations.

        Args:
            h:           ``[N, d]`` node embeddings.
            edge_index:  ``[2, E]`` (source, target) edge indices.
            rel_weights: dict with keys ``W_msg [E, d, d]``, ``W_self [E, d, d]``,
                         ``bias [E, d]``.

        Returns:
            ``[N, d]`` updated node embeddings.
        """
        N, d = h.shape
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)
        device = h.device

        W_msg  = rel_weights["W_msg"]   # [E, d_in, d_out]
        W_self = rel_weights["W_self"]  # [E, d_in, d_out]
        bias   = rel_weights["bias"]    # [E, d_out]
        d_out  = W_msg.size(-1)

        # --- messages ---
        h_src = h[src]                              # [E, d_in]
        msg = torch.bmm(h_src.unsqueeze(1), W_msg)  # [E, 1, d_out]
        msg = msg.squeeze(1)                         # [E, d_out]
        msg = msg + bias                             # [E, d_out]

        # mean aggregation at each target node
        agg = torch.zeros(N, d_out, device=device)
        cnt = torch.zeros(N, 1, device=device)
        idx = dst.unsqueeze(1).expand(-1, d_out)
        agg.scatter_add_(0, idx, msg)
        cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(E, 1, device=device))
        cnt = cnt.clamp(min=1.0)
        agg = agg / cnt                              # [N, d_out]

        # --- self-loop using per-node average W_self ---
        # For each node v: average W_self over all incoming edges
        W_self_agg = torch.zeros(N, d, d_out, device=device)
        idx3 = dst.view(-1, 1, 1).expand(-1, d, d_out)
        W_self_agg.scatter_add_(0, idx3, W_self)
        W_self_agg = W_self_agg / cnt.unsqueeze(-1)  # [N, d, d_out]

        # For isolated nodes (no incoming edges) just use identity-like
        isolated = (cnt.squeeze(1) == 0)
        if isolated.any():
            # use zero self-weight → they only get bias
            pass

        self_out = torch.bmm(h.unsqueeze(1), W_self_agg).squeeze(1)  # [N, d_out]

        return agg + self_out

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_texts: List[str],
    ) -> torch.Tensor:
        """Run HyperGNN forward pass.

        Args:
            node_features: ``[N, node_feat_dim]`` initial node features.
            edge_index:    ``[2, E]`` edge connectivity (source, target).
            edge_texts:    Length-E list of relation-type strings.

        Returns:
            ``[N, hidden_dim]`` node embeddings.
        """
        if edge_index.size(1) != len(edge_texts):
            raise ValueError(
                f"edge_index has {edge_index.size(1)} edges but "
                f"edge_texts has {len(edge_texts)} entries"
            )

        device = node_features.device

        # Project input features to hidden_dim
        h = F.relu(self.input_proj(node_features))   # [N, hidden_dim]

        # Encode all relation texts at once (deduplicated for efficiency)
        unique_texts = list(dict.fromkeys(edge_texts))  # preserves order, dedup
        text_to_idx = {t: i for i, t in enumerate(unique_texts)}
        edge_rel_ids = torch.tensor(
            [text_to_idx[t] for t in edge_texts], dtype=torch.long, device=device
        )

        text_embs = self.text_encoder(unique_texts, device)  # [U, text_dim]

        for layer_idx in range(self.num_layers):
            gen = self.weight_generators[layer_idx]
            norm = self.layer_norms[layer_idx]

            # Generate weights for each unique relation → then gather for edges
            # unique_weights: {key: [U, ...]}
            unique_weights = gen(text_embs)  # batch forward: [U, ...]

            # Gather per-edge weights
            rel_weights = {
                k: v[edge_rel_ids] for k, v in unique_weights.items()
            }

            # Message passing
            h_new = self._message_passing(h, edge_index, rel_weights)

            # Residual + activation + norm
            if h_new.shape == h.shape:
                h_new = h_new + h   # residual
            h_new = F.relu(h_new)

            if self.training and self.dropout > 0.0:
                h_new = F.dropout(h_new, p=self.dropout)

            h = norm(h_new)

        return h

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    def score_triple(
        self,
        head_emb: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Dot-product score for a (head, tail) pair (link prediction).

        Args:
            head_emb: ``[hidden_dim]`` or ``[B, hidden_dim]``.
            tail_emb: ``[hidden_dim]`` or ``[B, hidden_dim]``.

        Returns:
            Scalar or ``[B]`` score tensor.
        """
        return (head_emb * tail_emb).sum(dim=-1)

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
