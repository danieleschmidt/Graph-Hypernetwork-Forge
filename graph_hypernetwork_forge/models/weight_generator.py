"""WeightGenerator: maps text embeddings → GNN weight matrices.

A hypernetwork is a neural network that *generates the weights* of another
network.  Here the input is a text embedding describing a relation type and
the output is a set of weight matrices for one GNN message-passing layer.

Design
------
For a GNN layer with input dim `d_in` and output dim `d_out`, we need:

    W_msg  : (d_in,  d_out)   message transformation
    W_self : (d_in,  d_out)   self-loop transformation
    bias   : (d_out,)

WeightGenerator contains one small MLP per weight tensor.  Each MLP reads the
text embedding (shape `[text_dim]`) and outputs a flat vector that we reshape
into the required matrix.

Parameters are shared across relation types (only the *input text* differs),
so the total parameter count is independent of the number of relation types.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightGenerator(nn.Module):
    """Generate GNN weight matrices from a text embedding.

    One instance handles one GNN layer (fixed ``d_in`` / ``d_out``).  For a
    multi-layer GNN, instantiate one ``WeightGenerator`` per layer.

    Args:
        text_dim:   Dimension of the input text embedding.
        d_in:       Input feature dimension of the GNN layer.
        d_out:      Output feature dimension of the GNN layer.
        hidden_dim: Hidden dimension of the internal MLP.
        num_hidden: Number of hidden layers in the MLP (depth).
        dropout:    Dropout rate applied between hidden layers.
        init_scale: Scale factor applied to generated weights at init.
                    Smaller values keep activations stable early in training.
    """

    def __init__(
        self,
        text_dim: int,
        d_in: int,
        d_out: int,
        hidden_dim: int = 128,
        num_hidden: int = 2,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()

        if text_dim <= 0 or d_in <= 0 or d_out <= 0:
            raise ValueError("text_dim, d_in, d_out must all be positive integers")

        self.text_dim = text_dim
        self.d_in = d_in
        self.d_out = d_out
        self.init_scale = init_scale

        # ----- define the shapes we need to generate -----
        # (name, shape, num_elements)
        self._weight_specs: List[Tuple[str, Tuple[int, ...]]] = [
            ("W_msg",  (d_in, d_out)),
            ("W_self", (d_in, d_out)),
            ("bias",   (d_out,)),
        ]

        # One small MLP per weight tensor
        self.generators = nn.ModuleDict()
        for name, shape in self._weight_specs:
            n_out = math.prod(shape)
            self.generators[name] = self._build_mlp(text_dim, hidden_dim, n_out, num_hidden, dropout)

        # Learnable per-weight scale (log-space for positivity)
        self.log_scales = nn.ParameterDict({
            name: nn.Parameter(torch.full((1,), math.log(init_scale)))
            for name, _ in self._weight_specs
        })

        self._reset_parameters()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mlp(in_dim: int, hidden: int, out_dim: int, depth: int, dropout: float) -> nn.Sequential:
        """Build a depth-layer MLP: Linear → ReLU → Dropout (×depth) → Linear."""
        layers: list = []
        prev = in_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, hidden), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def _reset_parameters(self) -> None:
        """Small init so generated weights start near zero."""
        for mod in self.generators.values():
            last_linear = [m for m in mod.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last_linear.bias)
            nn.init.normal_(last_linear.weight, std=0.01)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def forward(self, text_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate weight matrices from a text embedding.

        Args:
            text_emb: Text embedding of shape ``[text_dim]`` (single relation)
                      or ``[batch, text_dim]`` (batch of relations).

        Returns:
            Dictionary with keys ``W_msg``, ``W_self``, ``bias``.
            Shapes when unbatched: ``(d_in, d_out)``, ``(d_in, d_out)``, ``(d_out,)``.
            Shapes when batched:   ``(B, d_in, d_out)``, …
        """
        single = text_emb.dim() == 1
        if single:
            text_emb = text_emb.unsqueeze(0)  # [1, text_dim]

        weights: Dict[str, torch.Tensor] = {}
        for name, shape in self._weight_specs:
            flat = self.generators[name](text_emb)  # [B, prod(shape)]
            scale = self.log_scales[name].exp()
            w = flat.view(text_emb.size(0), *shape) * scale
            weights[name] = w.squeeze(0) if single else w

        return weights
