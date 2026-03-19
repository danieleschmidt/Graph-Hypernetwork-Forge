# Graph Hypernetwork Forge

A hypernetwork that generates GNN weight matrices **on-the-fly from text
descriptions of relation types**, enabling zero-shot reasoning on knowledge
graphs.

## The Idea

Standard GNNs learn a fixed set of weight matrices — one per relation type.
That works well when all relation types are seen during training, but fails
completely on any new relation type at inference time.

**Graph Hypernetwork Forge** breaks that limitation:

```
relation text  →  TextEncoder  →  text embedding
text embedding  →  WeightGenerator  →  (W_msg, W_self, bias)
(W_msg, W_self, bias)  →  message passing on the KG  →  node embeddings
```

Because the weights are *generated from text*, the model can handle any
relation you can describe in English — no retraining, no fine-tuning.

## Installation

```bash
git clone https://github.com/danieleschmidt/Graph-Hypernetwork-Forge
cd Graph-Hypernetwork-Forge
pip install torch numpy
```

Python ≥ 3.10, PyTorch ≥ 2.0. No other dependencies required.

## Quick Start

```python
from graph_hypernetwork_forge import WeightGenerator, HyperGNN, ToyKnowledgeGraph

# Build a toy 8-node knowledge graph
kg = ToyKnowledgeGraph(feat_dim=16)
print(kg)  # ToyKnowledgeGraph(nodes=8, edges=11, relation_types=7)

# Create the model
model = HyperGNN(
    text_dim=64,       # dimension of relation text embeddings
    node_feat_dim=16,  # dimension of input node features
    hidden_dim=32,     # GNN hidden / output dimension
    num_layers=2,
)

# Forward pass — edge_texts is a list of relation-type strings, one per edge
node_embeddings = model(kg.node_features, kg.edge_index, kg.edge_texts)
print(node_embeddings.shape)  # torch.Size([8, 32])
```

## Runnable Demo

```bash
python demo.py
```

The demo:
1. Prints the toy KG structure
2. Runs a forward pass (untrained)
3. Trains for 20 steps with a margin-ranking objective
4. Demonstrates zero-shot inference with a **brand-new relation type**
   (`"is colleague of"`) that was never in the training set
5. Shows the `WeightGenerator` standalone API

## Architecture

### `TextEncoder`

A character-level bag-of-embeddings encoder: each character → learnable
embedding → mean-pool → linear projection → `[text_dim]`.

Fully self-contained; no pretrained weights, no downloads, works offline.
Swap in a sentence-transformer for production use.

### `WeightGenerator`

One instance per GNN layer. Contains three small MLPs:

| output    | shape             | purpose                         |
|-----------|-------------------|---------------------------------|
| `W_msg`   | `(d_in, d_out)`   | transform sender node features  |
| `W_self`  | `(d_in, d_out)`   | transform own node features     |
| `bias`    | `(d_out,)`        | additive bias                   |

Each MLP: `Linear → ReLU → … → Linear` with a learnable log-scale on the
output.  Batched input `[B, text_dim]` → batched output `[B, d_in, d_out]`.

### `HyperGNN`

The main model.  For each message-passing layer:

1. Encode all unique relation texts (deduplication for efficiency)
2. Call `WeightGenerator` to get per-relation weight matrices
3. For each edge `(u → v)` with relation `r`:
   `message = h_u @ W_msg[r] + bias[r]`
4. Mean-pool messages at each target node
5. Add self-loop: `h_v @ mean(W_self[r] for incoming r)`
6. Residual connection + ReLU + LayerNorm

### `ToyKnowledgeGraph`

A small 8-node KG (people, an org, a city, a skill) with 7 relation types,
used for demos and tests.

## Testing

```bash
pytest tests/
```

38 tests covering shapes, determinism, gradient flow, zero-shot inference, and
training convergence.

## Project Layout

```
graph_hypernetwork_forge/
  __init__.py
  models/
    weight_generator.py   # WeightGenerator
    hypergnn.py           # TextEncoder + HyperGNN
  data/
    knowledge_graph.py    # ToyKnowledgeGraph
demo.py                   # runnable demonstration
tests/
  test_weight_generator.py
  test_hypergnn.py
```

## License

MIT
