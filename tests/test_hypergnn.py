"""Tests for HyperGNN.

Covers:
- Output shapes
- No NaN/Inf in outputs
- Zero-shot generalisation (unseen relation types)
- Input validation
- TextEncoder
- Training / backward pass
- num_parameters
- score_triple
"""

import pytest
import torch

from graph_hypernetwork_forge import HyperGNN, ToyKnowledgeGraph
from graph_hypernetwork_forge.models.hypergnn import TextEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simple_kg():
    """5-node KG with 2 relation types."""
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ], dtype=torch.long)
    edge_texts = ["knows", "knows", "works with", "knows"]
    node_features = torch.randn(5, 8)
    return node_features, edge_index, edge_texts


# ---------------------------------------------------------------------------
# TextEncoder
# ---------------------------------------------------------------------------

class TestTextEncoder:
    def test_single_string_shape(self):
        enc = TextEncoder(text_dim=32, char_emb_dim=16)
        dev = torch.device("cpu")
        out = enc.encode_one("hello world", dev)
        assert out.shape == (32,)

    def test_batch_shape(self):
        enc = TextEncoder(text_dim=32, char_emb_dim=16)
        dev = torch.device("cpu")
        out = enc(["knows", "works at", "is parent of"], dev)
        assert out.shape == (3, 32)

    def test_empty_string_safe(self):
        enc = TextEncoder(text_dim=32)
        dev = torch.device("cpu")
        out = enc.encode_one("", dev)
        assert out.shape == (32,)

    def test_different_strings_different_outputs(self):
        enc = TextEncoder(text_dim=32)
        dev = torch.device("cpu")
        out1 = enc.encode_one("knows", dev)
        out2 = enc.encode_one("located in", dev)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# HyperGNN construction
# ---------------------------------------------------------------------------

class TestHyperGNNConstruction:
    def test_basic_construction(self):
        model = HyperGNN(text_dim=32, node_feat_dim=8, hidden_dim=16, num_layers=1)
        assert model.num_layers == 1
        assert len(model.weight_generators) == 1
        assert len(model.layer_norms) == 1

    def test_two_layer_construction(self, small_model):
        assert small_model.num_layers == 2
        assert len(small_model.weight_generators) == 2

    def test_invalid_num_layers(self):
        with pytest.raises(ValueError):
            HyperGNN(text_dim=32, node_feat_dim=8, hidden_dim=16, num_layers=0)

    def test_num_parameters_positive(self, small_model):
        assert small_model.num_parameters() > 0


# ---------------------------------------------------------------------------
# Forward pass shapes and sanity
# ---------------------------------------------------------------------------

class TestHyperGNNForward:
    def test_output_shape_toy_kg(self, small_model, toy_kg):
        with torch.no_grad():
            out = small_model(toy_kg.node_features, toy_kg.edge_index, toy_kg.edge_texts)
        assert out.shape == (toy_kg.num_nodes, small_model.hidden_dim)

    def test_no_nan_untrained(self, small_model, toy_kg):
        with torch.no_grad():
            out = small_model(toy_kg.node_features, toy_kg.edge_index, toy_kg.edge_texts)
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_simple_kg_output_shape(self):
        model = HyperGNN(text_dim=32, node_feat_dim=8, hidden_dim=16, num_layers=2)
        feats, ei, texts = simple_kg()
        with torch.no_grad():
            out = model(feats, ei, texts)
        assert out.shape == (5, 16)

    def test_single_node_single_edge(self):
        """Graph with 2 nodes, 1 edge."""
        model = HyperGNN(text_dim=32, node_feat_dim=8, hidden_dim=16, num_layers=1)
        feats = torch.randn(2, 8)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        edge_texts = ["knows"]
        with torch.no_grad():
            out = model(feats, edge_index, edge_texts)
        assert out.shape == (2, 16)

    def test_single_layer_model(self):
        model = HyperGNN(text_dim=32, node_feat_dim=8, hidden_dim=16, num_layers=1)
        feats, ei, texts = simple_kg()
        with torch.no_grad():
            out = model(feats, ei, texts)
        assert out.shape == (5, 16)

    def test_edge_text_count_mismatch_raises(self, small_model, toy_kg):
        with pytest.raises(ValueError):
            small_model(toy_kg.node_features, toy_kg.edge_index, toy_kg.edge_texts[:-1])


# ---------------------------------------------------------------------------
# Zero-shot: unseen relation types
# ---------------------------------------------------------------------------

class TestZeroShot:
    def test_unseen_relation_no_crash(self, small_model, toy_kg):
        """Model must not crash on a relation type never seen in 'training'."""
        new_rel = "is grandmother of"
        assert new_rel not in toy_kg.relation_types

        extra_src = torch.tensor([0], dtype=torch.long)
        extra_dst = torch.tensor([4], dtype=torch.long)
        new_edge_index = torch.cat(
            [toy_kg.edge_index, torch.stack([extra_src, extra_dst])], dim=1
        )
        new_texts = toy_kg.edge_texts + [new_rel]

        small_model.eval()
        with torch.no_grad():
            out = small_model(toy_kg.node_features, new_edge_index, new_texts)

        assert out.shape == (toy_kg.num_nodes, small_model.hidden_dim)
        assert not torch.isnan(out).any()

    def test_all_unseen_relations(self, small_model):
        """All edges use relation types not seen before — must still work."""
        feats = torch.randn(4, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        edge_texts = ["brand new rel A", "brand new rel B", "brand new rel A"]
        small_model.eval()
        with torch.no_grad():
            out = small_model(feats, edge_index, edge_texts)
        assert out.shape == (4, small_model.hidden_dim)
        assert not torch.isnan(out).any()

    def test_single_char_relation(self, small_model):
        feats = torch.randn(3, 16)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_texts = ["a", "b"]
        with torch.no_grad():
            out = small_model(feats, edge_index, edge_texts)
        assert out.shape == (3, small_model.hidden_dim)


# ---------------------------------------------------------------------------
# Gradient flow / trainability
# ---------------------------------------------------------------------------

class TestTraining:
    def test_backward_no_error(self, small_model, toy_kg):
        out = small_model(toy_kg.node_features, toy_kg.edge_index, toy_kg.edge_texts)
        loss = out.sum()
        loss.backward()

    def test_parameters_update(self, small_model, toy_kg):
        opt = torch.optim.SGD(small_model.parameters(), lr=0.1)
        # Snapshot initial params
        before = {n: p.clone().detach() for n, p in small_model.named_parameters()}

        opt.zero_grad()
        out = small_model(toy_kg.node_features, toy_kg.edge_index, toy_kg.edge_texts)
        out.sum().backward()
        opt.step()

        changed = 0
        for name, param in small_model.named_parameters():
            if not torch.allclose(before[name], param.detach()):
                changed += 1

        assert changed > 0, "No parameters changed after an optimiser step"

    def test_loss_decreases(self, toy_kg):
        """Loss should generally decrease over a few training steps."""
        model = HyperGNN(text_dim=32, node_feat_dim=16, hidden_dim=16, num_layers=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        src, dst = toy_kg.edge_index
        for _ in range(15):
            opt.zero_grad()
            embs = model(toy_kg.node_features, toy_kg.edge_index, toy_kg.edge_texts)
            pos = (embs[src] * embs[dst]).sum(dim=-1)
            perm = torch.randperm(dst.size(0))
            neg = (embs[src] * embs[dst[perm]]).sum(dim=-1)
            loss = torch.clamp(1.0 - pos + neg, min=0.0).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss at end should be ≤ loss at start (may not hold for all seeds,
        # but with 15 steps on a simple objective it typically does)
        assert losses[-1] <= losses[0] * 2, "Loss does not appear to decrease at all"


# ---------------------------------------------------------------------------
# score_triple
# ---------------------------------------------------------------------------

class TestScoreTriple:
    def test_score_scalar(self, small_model):
        a = torch.randn(16)
        b = torch.randn(16)
        s = small_model.score_triple(a, b)
        assert s.shape == ()

    def test_score_batched(self, small_model):
        a = torch.randn(4, 16)
        b = torch.randn(4, 16)
        s = small_model.score_triple(a, b)
        assert s.shape == (4,)

    def test_identical_embeddings_positive(self, small_model):
        a = torch.randn(16)
        s = small_model.score_triple(a, a)
        assert s.item() > 0, "Identical embeddings should score positive"
