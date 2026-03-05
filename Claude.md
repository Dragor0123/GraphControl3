# Claude.md — K-hop Mask Connectivity Analysis

## 1. Objective

Before running the Structure-Aware Similarity Graph experiment, verify that the k-hop masked similarity graph remains well-connected enough to produce meaningful Laplacian PE. If the graph is too sparse or fragmented at certain k values, the resulting PE will be uninformative, and any performance drop would reflect broken graph structure rather than a flaw in the idea itself.

## 2. What to Measure

For each dataset × each k value, compute the following on the **masked similarity graph** (after both k-hop masking and thresholding):

```python
adj_sim = (sim_matrix * khop_mask > threshold).float()
```

### Metric 1: Average Node Degree
```python
avg_degree = adj_sim.sum() / num_nodes
```
How many similarity edges each node has on average. If this is very low (e.g., < 1), most nodes are isolated or near-isolated, and the PE will be meaningless.

### Metric 2: Number of Connected Components
```python
import scipy.sparse.csgraph as csgraph
n_components, labels = csgraph.connected_components(sp.csr_matrix(adj_sim.numpy()), directed=False)
```
How many disconnected pieces the graph has. Laplacian PE is computed independently within each component, so many small components → PE captures only trivial local structure.

### Metric 3: Largest Component Ratio
```python
largest_component_size = np.bincount(labels).max()
largest_ratio = largest_component_size / num_nodes
```
What fraction of nodes belong to the largest connected component. If this is close to 1.0, the graph is essentially connected despite some isolated nodes. If it's low (e.g., 0.3), the graph is severely fragmented.

### Metric 4: Isolated Node Count
```python
degrees = adj_sim.sum(dim=1)
n_isolated = (degrees == 0).sum().item()
isolated_ratio = n_isolated / num_nodes
```
Nodes with zero similarity edges. These nodes will get arbitrary/zero PE values.

## 3. Experimental Setup

### Datasets
Cora_ML, DBLP, Photo, Chameleon, Squirrel, Actor

### K values
k = 0 (baseline, no mask), 1, 2, 3

### Thresholds
Same as the main experiment: Squirrel = 0.15, all others = 0.17

### Important Notes

- This is a **static analysis** — no training involved.
- Compute on the **full graph** (not subgraphs). The similarity graph in `obtain_attributes()` is built on the full graph before subgraph sampling.
- Use the same `similarity()` function and `_build_khop_mask()` from the main codebase for consistency.
- Self-loop: `_build_khop_mask` includes self-loop via `fill_diagonal_(1.0)`. For the baseline (k=0), the raw similarity matrix naturally has diagonal = 1.0 (cosine similarity with self). Both cases preserve self-loops, so they are comparable.

## 4. Expected Output

### Console Table (per dataset)
```
Dataset: Cora_ML (N=2995, threshold=0.17)
--------------------------------------------------
k  | Avg Degree | Components | Largest Ratio | Isolated
0  |    45.2    |      1     |    1.000      |    0
1  |     3.8    |     12     |    0.953      |   28
2  |    11.4    |      3     |    0.991      |    5
3  |    18.7    |      1     |    1.000      |    0
```

### JSON Output
```json
{
  "dataset": "Cora_ML",
  "num_nodes": 2995,
  "threshold": 0.17,
  "results": {
    "k0": {"avg_degree": ..., "n_components": ..., "largest_ratio": ..., "n_isolated": ..., "isolated_ratio": ...},
    "k1": {"avg_degree": ..., "n_components": ..., "largest_ratio": ..., "n_isolated": ..., "isolated_ratio": ...},
    "k2": {...},
    "k3": {...}
  }
}
```

Save to `./results/khop_connectivity/connectivity_{dataset}.json`

## 5. Interpretation Guide

| Observation | Implication |
|-------------|------------|
| avg_degree >= 5 and largest_ratio > 0.9 | Graph is healthy — safe to use this k value |
| avg_degree < 2 or isolated_ratio > 0.1 | Graph is too sparse — PE quality will degrade, skip this k value in main experiment |
| n_components >> 1 but largest_ratio > 0.8 | A few small fragments but most nodes in one component — likely still usable |
| Results similar across k=1,2,3 | Original graph is dense, so even 1-hop covers most similarity edges — k=1 sufficient |
| k=1 too sparse but k=2 healthy | Minimum viable k for this dataset is 2 |

## 6. Execution

```bash
python khop_connectivity.py --datasets Cora_ML DBLP Photo Chameleon Squirrel Actor
```
