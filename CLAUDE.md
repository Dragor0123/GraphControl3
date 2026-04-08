# CLAUDE.md — Diagnostic Experiments for GraphControl Branch Complementarity Analysis

## Project Context

This codebase implements **GraphControl** (WWW '24), a graph domain transfer learning framework that fine-tunes a GCC pre-trained GIN encoder using a ControlNet-style architecture. The goal of these diagnostic experiments is to analyze whether the two branches (frozen encoder and trainable copy) produce **complementary or redundant** representations, especially under varying homophily conditions.

### Architecture Summary

In `models/gcc_graphcontrol.py`, the forward pass (`forward_subgraph`) works as follows:

```python
# Frozen branch: pre-trained GIN encoder (parameters frozen)
h_frozen = self.encoder.forward_subgraph(x, edge_index, batch, root_n_id)

# ControlNet branch: trainable copy of the same GIN encoder
x_down = self.zero_conv1(x_sim)      # Z1(P') — zero-initialized linear on condition
x_down = x_down + x                   # P + Z1(P')
h_tc_raw = self.trainable_copy.forward_subgraph(x_down, edge_index, batch, root_n_id)
h_control = self.zero_conv2(h_tc_raw) # Z2(g_c(...)) — zero-initialized linear

# Final output: simple addition
out = h_frozen + h_control
logits = self.linear_classifier(out)
```

Key observations:
- Both branches use the **same `edge_index`** (same subgraph topology) for message passing.
- `x` is the Laplacian PE from the original adjacency A (computed per subgraph in `process_attributes`).
- `x_sim` is the Laplacian PE from the feature-similarity-based adjacency A' (computed on full graph in `obtain_attributes`).
- Both are smallest-32 eigenvectors of the respective normalized Laplacians.
- The encoder outputs are **graph-level pooled representations** (score_over_layer from UnsupervisedGIN).

### File Structure

```
.
├── graphcontrol.py          # Main fine-tuning script (entry point)
├── gcc.py                   # GCC baseline script
├── checkpoint/gcc.pth       # Pre-trained GCC weights
├── models/
│   ├── gcc_graphcontrol.py  # GCC_GraphControl model (KEY FILE TO MODIFY)
│   ├── gcc.py               # GCC model (UnsupervisedGIN)
│   ├── encoder.py           # GIN_Encoder, GCN_Encoder, etc.
│   ├── mlp.py               # MLP utilities
│   ├── model_manager.py     # load_model() — loads pre-trained weights
│   └── pooler.py            # Subgraph pooling utilities
├── utils/
│   ├── transforms.py        # process_attributes(), obtain_attributes() — PE computation
│   ├── sampling.py          # collect_subgraphs() — RWR-based subgraph sampling
│   ├── normalize.py         # get_laplacian_matrix(), similarity()
│   ├── args.py              # Argument parser
│   └── random.py            # Seed utilities
├── datasets/
│   └── __init__.py          # NodeDataset class, dataset loading
└── optimizers/
    └── __init__.py          # Optimizer factory
```

---

## Task: Implement 4 Diagnostic Experiments

Create a single script `analyze.py` at the project root. It should:
1. Load a dataset and fine-tune GraphControl normally (reuse logic from `graphcontrol.py`).
2. After fine-tuning, run the 4 diagnostic analyses below on the **test set**.
3. Save results (metrics + plots) to an `analysis_results/` directory.

### Important Implementation Notes

- **Do NOT modify** the original `graphcontrol.py`, `gcc.py`, or any existing files. All new code goes into `analyze.py` (and optionally a modified model file if needed).
- You may create a subclass or a wrapper around `GCC_GraphControl` to extract intermediate representations.
- The model must be **fully fine-tuned first** before running diagnostics (use the same fine-tuning logic as `graphcontrol.py`).
- Use the same preprocessing pipeline: `collect_subgraphs()`, `process_attributes()`, `obtain_attributes()`.
- Datasets to analyze: `Cora_ML`, `DBLP`, `Photo` (homophilic), `Squirrel`, `Actor` (heterophilic).
  - Note: `Squirrel` may require `--threshold 0.15`; others use `0.17`.
  - If `Squirrel` or `Actor` fail to load or have issues, proceed with the remaining datasets.
- Run with default args from `args.py` unless specified otherwise.
- Use `--seeds 0` for a single run during development; final runs use `--seeds 0 1 2 3 4`.

---

### Experiment 1: Per-Branch Classification Performance

**Goal**: Measure classification accuracy when using only one branch vs. the combined output.

**Method**: After fine-tuning the full GraphControl model, evaluate test accuracy under 3 conditions:
- **(A) Frozen only**: `logits = classifier(h_frozen)`
- **(B) ControlNet only**: `logits = classifier(h_control)` (after zero_conv2)
- **(B') ControlNet raw**: `logits = classifier(h_tc_raw)` (before zero_conv2, requires a separate classifier head or retraining — alternatively, just report this as a representation quality metric without classification)
- **(C) Combined (original)**: `logits = classifier(h_frozen + h_control)`

**Implementation approach**:
```python
# In the analysis forward pass, extract all intermediate outputs:
with torch.no_grad():
    encoder.eval()
    h_frozen = self.encoder.forward_subgraph(x, edge_index, batch, root_n_id)

x_down = self.zero_conv1(x_sim) + x
h_tc_raw = self.trainable_copy.forward_subgraph(x_down, edge_index, batch, root_n_id)
h_control = self.zero_conv2(h_tc_raw)
h_combined = h_frozen + h_control

# Evaluate each through the same linear_classifier
logits_frozen = self.linear_classifier(h_frozen)
logits_control = self.linear_classifier(h_control)
logits_combined = self.linear_classifier(h_combined)
```

Note: Using the same `linear_classifier` (trained on combined representations) for individual branches is intentional — it reveals how well each branch alone can drive the classifier that was optimized for their sum.

**Output**: A table (printed + saved as CSV):
```
Dataset     | Homophily Ratio | Frozen_Acc | Control_Acc | Combined_Acc
Cora_ML     | ...     | ...        | ...         | ...
DBLP        | ...      | ...        | ...         | ...
Photo       | ...      | ...        | ...         | ...
Squirrel    | ...      | ...        | ...         | ...
Actor       | ...      | ...        | ...         | ...
```
Homophily Ratio : The edge homophily ratio of the dataset.

---

### Experiment 2: Representation Redundancy Metrics

**Goal**: Quantify how similar/redundant the two branch outputs are.

**Method**: Collect `h_frozen` and `h_control` (post zero_conv2) for all test nodes. Compute:

1. **Per-node cosine similarity**: `cos(h_frozen_i, h_control_i)` for each test node i. Report mean, std, and plot the distribution (histogram).

2. **CKA (Linear Centered Kernel Alignment)**: Between the full `H_frozen` (N×D) and `H_control` (N×D) matrices. CKA ∈ [0,1]; 1 means identical representational structure.
   ```python
   def linear_CKA(X, Y):
       # X, Y: (N, D) matrices, centered
       X = X - X.mean(0)
       Y = Y - Y.mean(0)
       hsic_xy = torch.norm(Y.T @ X, 'fro') ** 2
       hsic_xx = torch.norm(X.T @ X, 'fro') ** 2
       hsic_yy = torch.norm(Y.T @ Y, 'fro') ** 2
       return (hsic_xy / (torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy))).item()
   ```

3. **Effective rank ratio**: Compute effective rank of `H_frozen`, `H_control`, and `[H_frozen; H_control]` (concatenated along feature dim). If complementary, `erank([H_frozen; H_control])` should be significantly larger than `max(erank(H_frozen), erank(H_control))`.
   ```python
   def effective_rank(X):
       # X: (N, D)
       s = torch.linalg.svdvals(X - X.mean(0))
       p = s / s.sum()
       p = p[p > 1e-10]
       return torch.exp(-torch.sum(p * torch.log(p))).item()
   ```

**Output**: A table (printed + saved as CSV):
```
Dataset     | Cosine_Mean | Cosine_Std | CKA   | ERank_Frozen | ERank_Control | ERank_Concat | ERank_Ratio
```
Where `ERank_Ratio = ERank_Concat / max(ERank_Frozen, ERank_Control)`.

Also save histogram plots of cosine similarity distributions per dataset.

---

### Experiment 3: Local Homophily vs. Branch Contribution

**Goal**: Analyze whether the ControlNet branch contributes more for nodes with low local homophily.

**Method**:

**Step 3a — Compute local homophily for each node** (on the full graph, not subgraphs):
```python
def compute_local_homophily(edge_index, y, num_nodes):
    """Returns tensor of shape (num_nodes,) with local homophily per node."""
    h = torch.zeros(num_nodes)
    deg = torch.zeros(num_nodes)
    row, col = edge_index
    same_label = (y[row] == y[col]).float()
    # scatter add
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
    h.scatter_add_(0, row, same_label)
    h = h / deg.clamp(min=1)
    return h
```

**Step 3b — Compute branch contribution metrics per test node**:
For each test node i, from the fine-tuned model extract `h_frozen_i`, `h_control_i`, `h_combined_i = h_frozen_i + h_control_i`:

- **Norm ratio**: `ratio_i = ||h_control_i|| / (||h_frozen_i|| + ||h_control_i||)`
- **Cosine with final**: `cos_frozen_i = cos(h_combined_i, h_frozen_i)` and `cos_control_i = cos(h_combined_i, h_control_i)`
- **Correct classification indicator**: whether `h_combined`, `h_frozen` alone, `h_control` alone each predict correctly.

**Step 3c — Group by local homophily quartiles** (Q1=top 25% homophily, Q4=bottom 25%):
For each quartile, report:
- Mean norm ratio
- Mean cosine_frozen, mean cosine_control
- Accuracy of each variant (frozen-only, control-only, combined)

**Output**:
- A table per dataset showing metrics across Q1–Q4.
- A grouped bar chart or line plot per dataset: x-axis = quartile, y-axis = norm_ratio (and/or accuracy of each variant).

---

### Experiment 4: Spectral Energy (Rayleigh Quotient) Analysis

**Goal**: Compare the spectral frequency characteristics of the two branch outputs.

**Method**: For each test subgraph, compute the Rayleigh Quotient of each branch's **node-level representations before pooling**.

This requires extracting node-level hidden representations from the GIN encoder before the graph pooling step. The `UnsupervisedGIN.forward()` returns `(score_over_layer, all_outputs)` where `all_outputs` is a list of pooled hidden states. We need the **unpooled** node-level representations.

**Approach**: Modify or wrap `UnsupervisedGIN.forward()` to also return the last hidden layer `h` (before pooling). Specifically, in `models/gcc.py`, the `UnsupervisedGIN.forward()` method has:
```python
hidden_rep = [x]
h = x
for i in range(self.num_layers - 1):
    h = self.ginlayers[i](h, edge_index)
    h = self.batch_norms[i](h)
    h = F.relu(h)
    hidden_rep.append(h)
```
The last `h` (i.e., `hidden_rep[-1]`) is the node-level representation before pooling.

For each test subgraph with `n_sub` nodes:
1. Extract node-level `h_frozen_nodes` (n_sub × D) and `h_control_nodes` (n_sub × D) from each branch.
2. Compute the subgraph's normalized Laplacian `L_sub` from its `edge_index`.
3. For each feature dimension d, compute `RQ_d = h_d^T L_sub h_d / h_d^T h_d`, then average over dimensions:
   ```python
   def compute_avg_rayleigh_quotient(H_nodes, edge_index, num_nodes):
       """
       H_nodes: (num_nodes, D) node representations
       edge_index: subgraph edges
       Returns: scalar average RQ across feature dimensions
       """
       L = compute_normalized_laplacian(edge_index, num_nodes)  # dense (num_nodes, num_nodes)
       rqs = []
       for d in range(H_nodes.shape[1]):
           h = H_nodes[:, d]
           numerator = h @ L @ h
           denominator = h @ h
           if denominator > 1e-10:
               rqs.append((numerator / denominator).item())
       return np.mean(rqs) if rqs else 0.0
   ```
4. Aggregate RQ values across all test subgraphs.

**Output**:
- A table per dataset: `Dataset | RQ_Frozen_Mean | RQ_Frozen_Std | RQ_Control_Mean | RQ_Control_Std | RQ_Diff`
- Histogram plots comparing RQ distributions of the two branches per dataset.
- Optionally: scatter plot of (local_homophily_i, RQ_frozen_i) and (local_homophily_i, RQ_control_i) to see if spectral characteristics vary with homophily.

---

## Output Structure

```
analysis_results/
├── exp1_branch_accuracy.csv
├── exp2_redundancy_metrics.csv
├── exp2_cosine_histograms/
│   ├── Cora_ML.png
│   ├── DBLP.png
│   └── ...
├── exp3_homophily_contribution/
│   ├── Cora_ML_table.csv
│   ├── Cora_ML_plot.png
│   └── ...
├── exp4_spectral_analysis/
│   ├── rayleigh_quotient_table.csv
│   ├── Cora_ML_rq_histogram.png
│   └── ...
└── summary.txt   # Overall findings in plain text
```

---

## Running the Experiments

```bash
# Single seed, single dataset (for development/debugging)
python analyze.py --dataset Cora_ML --lr 0.5 --optimizer adamw --use_adj --seeds 0 --model GCC_GraphControl

# Full run across all target datasets (use threshold = 0.15 for Squirrel)
python analyze.py --dataset Cora_ML --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj  --seeds 0 1 2 3 4 --model GCC_GraphControl
python analyze.py --dataset DBLP --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj  --seeds 0 1 2 3 4 --model GCC_GraphControl
python analyze.py --dataset Photo --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj  --seeds 0 1 2 3 4 --model GCC_GraphControl
python analyze.py --dataset Chameleon --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj  --seeds 0 1 2 3 4 --model GCC_GraphControl
python analyze.py --dataset Squirrel --lr 0.5 --optimizer adamw --threshold 0.15 --use_adj  --seeds 0 1 2 3 4 --model GCC_GraphControl
python analyze.py --dataset Actor --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj  --seeds 0 1 2 3 4 --model GCC_GraphControl
```

---

## Important Constraints

1. **Do not modify existing source files.** Create new files only.
2. The fine-tuning must complete successfully before any analysis runs. Use early stopping with patience=15 (same as original).
3. For Experiment 4, extracting node-level representations requires accessing GIN internals. Create a wrapper/subclass rather than editing `models/gcc.py` directly.
4. All plots should use matplotlib with clear labels, titles, and legends.
5. Handle edge cases: datasets with very few test nodes, subgraphs with isolated nodes, numerical stability in RQ computation.
6. Print progress and intermediate results to stdout for monitoring.
