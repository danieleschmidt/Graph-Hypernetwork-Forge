"""Tests for WeightGenerator.

Covers:
- Output shapes (single and batched)
- Determinism (same input → same output)
- Different inputs → different outputs
- Gradient flow
- Invalid construction
- Learnable log_scales
"""

import math

import pytest
import torch

from graph_hypernetwork_forge import WeightGenerator


class TestWeightGeneratorShapes:
    """Output shape contracts."""

    def test_single_input_shapes(self, weight_gen):
        emb = torch.randn(32)
        out = weight_gen(emb)
        assert "W_msg" in out
        assert "W_self" in out
        assert "bias" in out
        assert out["W_msg"].shape == (16, 16), f"W_msg shape wrong: {out['W_msg'].shape}"
        assert out["W_self"].shape == (16, 16)
        assert out["bias"].shape == (16,)

    def test_batched_input_shapes(self, weight_gen):
        embs = torch.randn(5, 32)
        out = weight_gen(embs)
        assert out["W_msg"].shape == (5, 16, 16)
        assert out["W_self"].shape == (5, 16, 16)
        assert out["bias"].shape == (5, 16)

    def test_batch_size_1(self, weight_gen):
        embs = torch.randn(1, 32)
        out = weight_gen(embs)
        assert out["W_msg"].shape == (1, 16, 16)

    def test_non_square_weights(self):
        gen = WeightGenerator(text_dim=32, d_in=8, d_out=24, hidden_dim=64)
        emb = torch.randn(32)
        out = gen(emb)
        assert out["W_msg"].shape == (8, 24)
        assert out["W_self"].shape == (8, 24)
        assert out["bias"].shape == (24,)

    def test_batched_non_square(self):
        gen = WeightGenerator(text_dim=16, d_in=4, d_out=8, hidden_dim=32)
        embs = torch.randn(3, 16)
        out = gen(embs)
        assert out["W_msg"].shape == (3, 4, 8)


class TestWeightGeneratorDeterminism:
    """Same input must always yield the same output."""

    def test_deterministic_eval(self, weight_gen):
        weight_gen.eval()
        emb = torch.randn(32)
        with torch.no_grad():
            out1 = weight_gen(emb)
            out2 = weight_gen(emb)
        for k in out1:
            assert torch.allclose(out1[k], out2[k]), f"Non-deterministic output for {k}"

    def test_different_inputs_different_outputs(self, weight_gen):
        weight_gen.eval()
        emb1 = torch.randn(32)
        emb2 = torch.randn(32)
        with torch.no_grad():
            out1 = weight_gen(emb1)
            out2 = weight_gen(emb2)
        # At least W_msg should differ
        assert not torch.allclose(out1["W_msg"], out2["W_msg"])


class TestWeightGeneratorGradients:
    """Gradients should flow through the generator."""

    def test_gradients_flow(self, weight_gen):
        emb = torch.randn(32, requires_grad=True)
        out = weight_gen(emb)
        loss = out["W_msg"].sum() + out["W_self"].sum() + out["bias"].sum()
        loss.backward()
        assert emb.grad is not None, "No gradient on input embedding"
        assert emb.grad.shape == emb.shape

    def test_log_scales_are_parameters(self, weight_gen):
        param_names = [n for n, _ in weight_gen.named_parameters()]
        scale_names = [n for n in param_names if "log_scale" in n]
        assert len(scale_names) == 3, f"Expected 3 log_scales, got {len(scale_names)}"

    def test_scales_appear_in_optimizer(self, weight_gen):
        opt = torch.optim.Adam(weight_gen.parameters(), lr=1e-3)
        # Just check that no error is raised
        emb = torch.randn(32)
        out = weight_gen(emb)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        opt.step()


class TestWeightGeneratorConstruction:
    """Constructor validation."""

    def test_basic_construction(self):
        gen = WeightGenerator(text_dim=64, d_in=32, d_out=32)
        assert gen.text_dim == 64
        assert gen.d_in == 32
        assert gen.d_out == 32

    def test_invalid_dims_raise(self):
        with pytest.raises(ValueError):
            WeightGenerator(text_dim=0, d_in=32, d_out=32)
        with pytest.raises(ValueError):
            WeightGenerator(text_dim=32, d_in=0, d_out=32)
        with pytest.raises(ValueError):
            WeightGenerator(text_dim=32, d_in=32, d_out=-1)

    def test_custom_hidden_dim(self):
        gen = WeightGenerator(text_dim=32, d_in=8, d_out=8, hidden_dim=256)
        emb = torch.randn(32)
        out = gen(emb)
        assert out["W_msg"].shape == (8, 8)

    def test_no_hidden_layers(self):
        gen = WeightGenerator(text_dim=32, d_in=8, d_out=8, num_hidden=0, hidden_dim=64)
        emb = torch.randn(32)
        out = gen(emb)
        assert out["W_msg"].shape == (8, 8)


class TestWeightGeneratorScale:
    """Init scale controls weight magnitude."""

    def test_small_init_scale_gives_small_outputs(self):
        gen = WeightGenerator(text_dim=32, d_in=8, d_out=8, init_scale=1e-4)
        gen.eval()
        with torch.no_grad():
            emb = torch.randn(32)
            out = gen(emb)
        # With init_scale=1e-4 the outputs should be small
        assert out["W_msg"].abs().max().item() < 1.0, "Weights too large for small init_scale"
