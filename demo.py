#!/usr/bin/env python3
"""Graph Hypernetwork Forge — runnable demo.

Demonstrates the core idea: a hypernetwork that generates GNN weight matrices
on-the-fly from text descriptions of relation types, enabling zero-shot
reasoning on knowledge graphs.

Run::

    python demo.py

Expected output: node embeddings + a link-prediction example using a *new*
relation type that the model has never seen during any training.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from graph_hypernetwork_forge import WeightGenerator, HyperGNN, ToyKnowledgeGraph


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1.  Build the toy KG
# ---------------------------------------------------------------------------
print_section("Knowledge Graph")

kg = ToyKnowledgeGraph(feat_dim=16)
print(kg)
print(f"\nNodes: {kg.node_names}")
print(f"\nEdges (src, dst, relation):")
for (src, dst, rel) in kg.edge_data:
    print(f"  {kg.node_names[src]:12s} --[{rel}]--> {kg.node_names[dst]}")

print(f"\nUnique relation types: {kg.relation_types}")


# ---------------------------------------------------------------------------
# 2.  Build the model
# ---------------------------------------------------------------------------
print_section("Model")

model = HyperGNN(
    text_dim=64,
    node_feat_dim=kg.feat_dim,
    hidden_dim=32,
    num_layers=2,
    dropout=0.0,
)
print(model)
print(f"\nTotal parameters: {model.num_parameters():,}")


# ---------------------------------------------------------------------------
# 3.  Forward pass (no training, random weights)
# ---------------------------------------------------------------------------
print_section("Forward pass (untrained)")

with torch.no_grad():
    node_embs = model(kg.node_features, kg.edge_index, kg.edge_texts)

print(f"Output shape: {node_embs.shape}  (expected [{kg.num_nodes}, 32])")
print(f"No NaNs: {not torch.isnan(node_embs).any()}")
print(f"Output norms: {node_embs.norm(dim=1).tolist()}")


# ---------------------------------------------------------------------------
# 4.  Quick training loop: self-supervised — node embeddings should be
#     similar for nodes that share the same relation type.
# ---------------------------------------------------------------------------
print_section("Quick training demo (20 steps)")

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for step in range(20):
    model.train()
    optimizer.zero_grad()

    embs = model(kg.node_features, kg.edge_index, kg.edge_texts)

    # Simple contrastive-style objective:
    # connected nodes should have higher dot-product than random pairs.
    src, dst = kg.edge_index[0], kg.edge_index[1]
    pos_scores = model.score_triple(embs[src], embs[dst])

    # Random negatives (shuffle dst)
    perm = torch.randperm(dst.size(0))
    neg_scores = model.score_triple(embs[src], embs[dst[perm]])

    loss = torch.clamp(1.0 - pos_scores + neg_scores, min=0.0).mean()
    loss.backward()
    optimizer.step()

    if (step + 1) % 5 == 0:
        print(f"  Step {step+1:3d}  loss={loss.item():.4f}")


# ---------------------------------------------------------------------------
# 5.  Zero-shot: new relation type never seen during training
# ---------------------------------------------------------------------------
print_section("Zero-shot: unseen relation type")

# Add an edge with a brand-new relation type not in the original KG.
# The hypernetwork generates appropriate weights purely from the text.
new_relation = "is colleague of"
print(f"New relation: '{new_relation}'")
print(f"Was in training set: {new_relation in kg.relation_types}")

new_src = torch.tensor([1, 2], dtype=torch.long)   # Bob, Carol
new_dst = torch.tensor([2, 0], dtype=torch.long)   # Carol, Alice
new_edge_index = torch.cat([kg.edge_index,
                             torch.stack([new_src, new_dst])], dim=1)
new_edge_texts = kg.edge_texts + [new_relation, new_relation]

model.eval()
with torch.no_grad():
    zs_embs = model(kg.node_features, new_edge_index, new_edge_texts)

print(f"Output shape with new relation: {zs_embs.shape}")
print(f"No NaNs: {not torch.isnan(zs_embs).any()}")

# Score the two new edges
bob_emb   = zs_embs[1]
carol_emb = zs_embs[2]
score = model.score_triple(bob_emb, carol_emb).item()
print(f"Link score Bob→Carol ('{new_relation}'): {score:.4f}")


# ---------------------------------------------------------------------------
# 6.  WeightGenerator standalone demo
# ---------------------------------------------------------------------------
print_section("WeightGenerator standalone")

gen = WeightGenerator(text_dim=64, d_in=32, d_out=32, hidden_dim=128)
print(gen)

# Single relation
dummy_emb = torch.randn(64)
weights = gen(dummy_emb)
print(f"\nSingle-relation output keys: {list(weights.keys())}")
for k, v in weights.items():
    print(f"  {k}: {tuple(v.shape)}")

# Batched relations
batch_embs = torch.randn(5, 64)
batch_weights = gen(batch_embs)
print(f"\nBatched output (batch=5):")
for k, v in batch_weights.items():
    print(f"  {k}: {tuple(v.shape)}")


print_section("Done ✓")
print("All demos ran successfully.\n")
