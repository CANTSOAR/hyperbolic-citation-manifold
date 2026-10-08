# Hyperbolic Citation Manifold

The goal is to embed the arXiv math and physics citation graph in hyperbolic space (a Poincaré ball) and use the geometry to find papers that should be connected but aren't. Each paper gets a point in the ball. Citations pull papers together, unrelated papers are pushed apart, and older foundational work is pushed toward the center. If the geometry works, two papers that sit close together with no citation between them are candidates for a missing link. To test this, the model is trained only on pre-2024 papers and asked to predict which older papers the 2024+ papers actually cite.

**Status: early prototype.** The data download, the Rust parser, and the PyTorch model, loss, training, and evaluation code are written. The model has not been trained yet and there are no results. The main blocker is citation data (see [Status](#status-and-next-steps)).

## Why hyperbolic space

Citation graphs look a lot like trees. A small number of foundational papers are cited by many papers, each of which is cited by more specialized work, and so on. The number of papers grows roughly exponentially as you move from the foundations out to the frontier.

Euclidean space cannot hold a tree like that without distortion, because the volume of a Euclidean ball only grows polynomially with its radius. In hyperbolic space volume grows exponentially with radius, so a tree embeds with low distortion even in a few dimensions. In the Poincaré ball this gives a useful reading of the coordinates:

- **Distance from the origin** shows how general or how specialized a paper is (foundations near the center, frontier work near the boundary).
- **Direction** shows the subfield.

The design intentionally leaves out a global uniformity loss. The idea is that without one, empty regions open up between clusters that are only weakly connected, and new papers that bridge those clusters should land in those gaps.

## Pipeline

```
Kaggle arXiv metadata (JSON, ~5 GB)
        |  download_data.py
        v
Rust parser (data_parser/)        filter to math/physics, write nodes.csv + edges.csv
        |
        v
dataset.py                        TF-IDF features from abstracts, PyG graph, temporal split
        |
        v
model.py + losses.py + train.py   HGCN on the Poincare ball, Riemannian Adam
        |
        v
evaluate.py                       Hits@10 citation prediction for 2024+ papers
```

### 1. Data (`download_data.py`, `data_parser/`)

- `download_data.py` uses the Kaggle API to download the `Cornell-University/arxiv` metadata snapshot into `data/`.
- `data_parser/src/main.rs` streams the JSON lines file in two passes:
  1. Keeps papers whose category string matches math or physics archives (`math`, `physics`, `astro-ph`, `cond-mat`, `gr-qc`, `hep-*`, `nlin`, `nucl-*`, `quant-ph`) and writes `id, categories, date, abstract` to `data/nodes.csv`.
  2. Reads a `references` field, if present, and keeps citation edges where both ends are in the filtered set, writing `data/edges.csv`.

On the local snapshot (3,028,029 records), pass 1 kept **2,196,659 math/physics papers**. Pass 2 produced **0 edges**, because the Kaggle metadata snapshot has no `references` field. The parser assumes a citation source that has not been added yet.

### 2. Features and split (`hyperbolic_model/dataset.py`)

- Node features: 128-dimensional TF-IDF vectors of each abstract (scikit-learn, English stop words removed).
- Graph: a PyTorch Geometric `Data` object with a directed `edge_index` (citing paper to cited paper).
- Temporal split: papers dated before 2024 are training nodes; papers from 2024 on are test nodes.

### 3. Model (`hyperbolic_model/model.py`)

A two-layer hyperbolic graph convolutional network (HGCN) on a `geoopt.PoincareBall` with curvature c = 1:

1. Linear layer (128 to 64) and ReLU in Euclidean space.
2. `expmap0` onto the Poincaré ball.
3. Neighbor aggregation: `logmap0` back to the tangent space at the origin, then a mean over neighbors. This is a tangent-space approximation of the Fréchet mean, not an exact one.
4. Linear layer (64 to 32) on the node feature plus the aggregated neighbor feature, then `expmap0` again.

Output: a 32-dimensional point in the Poincaré ball for each paper.

### 4. Loss (`hyperbolic_model/losses.py`)

The total loss is the sum of two terms:

- **Contrastive graph loss.** Squared hyperbolic distance between cited pairs (pull together), plus a squared hinge `relu(1 - d)^2` on randomly sampled negative pairs (push apart up to a margin of 1).
- **Hierarchy (norm ordering) loss.** For each citation where the citing paper is newer than the cited paper, penalize the cited paper having a larger norm than the citing one. This pushes older, cited papers toward the center. It is a simple norm-ordering constraint, not a full entailment-cone loss.

There is deliberately no uniformity term.

### 5. Training (`hyperbolic_model/train.py`)

- PyG `NeighborLoader` mini-batches (15 neighbors at hop 1, 10 at hop 2, batch size 256), seeded only from pre-2024 nodes.
- `geoopt.optim.RiemannianAdam`, learning rate 0.01, 10 epochs.
- Saves `hyperbolic_model.pt` and, if the full graph fits in memory, `embeddings.pt`.

### 6. Evaluation (`hyperbolic_model/evaluate.py`)

The "time machine" test: sample up to 100 papers from 2024 on. For each one that cites at least one pre-2024 paper, rank all pre-2024 papers by hyperbolic distance and check whether a true citation is among the 10 nearest. The reported metric is **Hits@10**.

A dedicated "missing link" scorer (ranking close but uncited pairs) is not written yet. Today, nearest neighbors in the ball that are not cited are the intended candidates, but nothing outputs them.

## Results

None yet. The model has not been trained because the edge list is empty, and there are no logs, metrics, figures, or example "should be connected" pairs in this repo.

## How to run

Requirements: Python 3 with the packages in `hyperbolic_model/requirements.txt`, a Rust toolchain, and Kaggle API credentials (`~/.kaggle/kaggle.json`). The raw snapshot is about 5 GB and `nodes.csv` is about 2 GB.

```bash
# 1. Download the arXiv metadata snapshot into ./data
pip install -r hyperbolic_model/requirements.txt
python download_data.py

# 2. Parse to nodes.csv / edges.csv (paths are relative, so run from data_parser/)
cd data_parser
cargo run --release
cd ..

# 3. Train and evaluate (scripts read ../data, so run from hyperbolic_model/)
cd hyperbolic_model
python train.py
python evaluate.py
```

Step 3 needs a non-empty `data/edges.csv` to do anything useful.

## Repo layout

```
download_data.py            Kaggle download of the arXiv metadata snapshot
data_parser/                Rust parser: JSON -> data/nodes.csv, data/edges.csv
  src/main.rs
hyperbolic_model/
  dataset.py                TF-IDF features, PyG graph, pre-2024 / 2024+ split
  model.py                  HGCN on the Poincare ball (geoopt)
  losses.py                 contrastive loss + norm-ordering hierarchy loss
  train.py                  NeighborLoader training with Riemannian Adam
  evaluate.py               Hits@10 citation prediction for 2024+ papers
  requirements.txt
scope.txt                   original design notes
data/                       (gitignored) raw snapshot and parsed CSVs
```

## Status and next steps

Done:

- Kaggle download script.
- Rust parser that filters arXiv to about 2.2M math/physics papers.
- HGCN model, losses, mini-batch training loop, and Hits@10 evaluation, written but not yet run end to end.

Known gaps and next steps:

1. **Add a citation source.** The Kaggle snapshot has no references, so the graph currently has no edges. Options include Semantic Scholar or OpenAlex reference lists joined on arXiv ID. This is the main blocker.
2. **Fix the date field.** The parser uses `update_date` (date of the latest version), not the first submission date from `versions`. For a temporal split this can leak, because an old paper revised in 2024 counts as "new".
3. **Batched inference.** `train.py` and `evaluate.py` try a full-graph forward pass over about 2.2M nodes. This needs to use `NeighborLoader` for inference too.
4. **Missing-link scoring.** Add a script that outputs nearby but uncited pairs, filtered by year and subfield, and check a sample by hand.
5. **Baselines.** Compare Hits@10 against a Euclidean GCN of the same size and against TF-IDF cosine nearest neighbors, so a hyperbolic gain can be measured rather than assumed.
6. **Better text features.** Swap TF-IDF for a sentence embedding of the abstract.
