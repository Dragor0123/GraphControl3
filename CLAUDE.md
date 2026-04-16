# CLAUDE.md — Dual-Track Experiment: Condition Redesign + Operator Probe

## Context & Motivation

This codebase implements **GraphControl** (WWW '24), a graph domain transfer learning framework that fine-tunes a GCC pre-trained GIN encoder using a ControlNet-style architecture.

Previous diagnostic experiments revealed:
- **Frozen branch is useless**: standalone accuracy is 15–41% (near random guess) across all datasets.
- **Control branch dominates**: Combined ≈ Control-only, with only 0.3–2%p difference.
- **No node-adaptive behavior**: branch contributions are uniform across local homophily quartiles.
- **Spectral redundancy**: RQ_Control > RQ_Frozen but this is an artifact of condition injection, not intentional complementarity.
- **L_gain (objective-level redesign) failed**: no accuracy improvement; heterophilic datasets degraded.

**Root cause**: The condition A' (feature-similarity-based adjacency) has strong homophilic bias. Oracle experiments showed that replacing A' with ground-truth-filtered A yields massive gains (e.g., Squirrel: 26.6% → 90.8%).

**This experiment tests two orthogonal axes**:
1. **Condition Redesign (primary)**: Change *what structural signal* the trainable branch receives.
2. **Operator Probe (secondary)**: Change *how* the trainable branch propagates signals.

The 2×2 design disentangles whether the bottleneck is the input signal quality or the propagation mechanism.

---

## Architecture Summary

```python
# Current GraphControl forward pass (models/gcc_graphcontrol.py)

# Frozen branch
h_frozen = self.encoder.forward_subgraph(x, edge_index, batch, root_n_id)

# ControlNet branch
x_down = self.zero_conv1(x_sim)      # Z1(P')
x_down = x_down + x                   # P + Z1(P')
h_tc_raw = self.trainable_copy.forward_subgraph(x_down, edge_index, batch, root_n_id)
h_control = self.zero_conv2(h_tc_raw) # Z2(g_c(...))

# Final output
out = h_frozen + h_control
logits = self.linear_classifier(out)
```

Key facts:
- `x` = Laplacian PE from subgraph adjacency A (smallest 32 eigenvectors), computed in `process_attributes()`.
- `x_sim` = Laplacian PE from feature-similarity-based A' (smallest 32 eigenvectors), computed in `obtain_attributes()`.
- Both branches use the **same `edge_index`** (same subgraph topology).
- The GIN encoder uses sum aggregation: `h_v = MLP((1+ε)·h_v + Σ_{u∈N(v)} h_u)`.

### File Structure

```
.
├── graphcontrol.py          # Main fine-tuning script (entry point)
├── gcc.py                   # GCC baseline script
├── checkpoint/gcc.pth       # Pre-trained GCC weights
├── models/
│   ├── gcc_graphcontrol.py  # GCC_GraphControl model (KEY FILE)
│   ├── gcc.py               # GCC model (UnsupervisedGIN)
│   ├── encoder.py           # GIN_Encoder, etc.
│   ├── mlp.py               # MLP utilities
│   ├── model_manager.py     # load_model()
│   └── pooler.py            # Subgraph pooling utilities
├── utils/
│   ├── transforms.py        # process_attributes(), obtain_attributes() — PE computation (KEY FILE)
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

## Experimental Design: 2×2 Matrix

We test **4 configurations** by crossing two axes:

| Config | Condition A' | Propagation Rule | Description |
|--------|-------------|-----------------|-------------|
| **Baseline** | Similarity A' (original) | Standard GIN (sum) | Current GraphControl |
| **C1** | Dissimilarity A' (new) | Standard GIN (sum) | Condition redesign only |
| **O1** | Similarity A' (original) | Anti-smoothing GIN (subtract) | Operator probe only |
| **C1+O1** | Dissimilarity A' (new) | Anti-smoothing GIN (subtract) | Both combined |

### Datasets
- **Homophilic**: Cora_ML, DBLP, Photo
- **Heterophilic**: Chameleon, Squirrel, Actor

### Seeds
- Development: seed 0 only
- Final comparison: seeds 0, 1, 2, 3, 4

---

## Track 1: Condition Redesign (C1)

### What to change

Modify `obtain_attributes()` in `utils/transforms.py` to produce a **dissimilarity-based A'** instead of the current similarity-based A'.

### Current implementation (in `obtain_attributes()`)

The current flow is:
1. Compute pairwise cosine similarity matrix K from node features X.
2. Discretize: `A'[i,j] = 1 if K[i,j] > threshold, else 0`.
3. Compute normalized Laplacian of A'.
4. Extract smallest 32 eigenvectors → P' (condition PE).

### New implementation: Dissimilarity A' (intersection with original A)

The new condition topology A'_dissim should contain edges that:
- **Exist in the original adjacency A** (i.e., are real edges), AND
- Have **low feature similarity** (i.e., `K[i,j] < dissim_threshold`).

This selects "real heterophilic edges" — edges in the original graph where connected nodes have dissimilar features. This is the unsupervised approximation of the oracle experiment's ground-truth heterophilic edge selection.

**Implementation steps:**

```python
def obtain_attributes_dissimilarity(data, num_dim, threshold, dissim_threshold=None):
    """
    Produce dissimilarity-based condition PE.
    
    Args:
        data: PyG data object with data.x (node features) and data.edge_index (original adjacency)
        num_dim: number of eigenvectors to extract (default 32)
        threshold: NOT used for dissimilarity, kept for API compatibility
        dissim_threshold: cosine similarity threshold below which an edge is considered dissimilar.
                          If None, use (1.0 - threshold) as default or a sensible value like 0.5.
    """
    # Step 1: Compute cosine similarity matrix
    X = data.x  # (N, d)
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    K = X_norm @ X_norm.T  # (N, N) cosine similarity matrix
    
    # Step 2: Get original adjacency as dense matrix
    N = X.size(0)
    A_orig = torch.zeros(N, N, device=X.device)
    A_orig[data.edge_index[0], data.edge_index[1]] = 1.0
    
    # Step 3: Dissimilarity condition = low similarity AND exists in original A
    if dissim_threshold is None:
        dissim_threshold = 0.5  # default; sweep this
    A_dissim = ((K < dissim_threshold).float()) * A_orig
    
    # Step 4: Remove self-loops, ensure symmetry
    A_dissim.fill_diagonal_(0)
    A_dissim = (A_dissim + A_dissim.T).clamp(max=1.0)
    
    # Step 5: Handle edge case — if A_dissim is empty (no dissimilar edges), 
    # fall back to original A' or return zero PE
    if A_dissim.sum() == 0:
        print(f"WARNING: A_dissim is empty for dissim_threshold={dissim_threshold}. Falling back to original A'.")
        # Fall back to original obtain_attributes logic
        return obtain_attributes_original(data, num_dim, threshold)
    
    # Step 6: Compute normalized Laplacian of A_dissim
    # Use the same Laplacian computation as in the original code
    D = A_dissim.sum(dim=1)
    D_inv_sqrt = torch.where(D > 0, D.pow(-0.5), torch.zeros_like(D))
    D_inv_sqrt_mat = torch.diag(D_inv_sqrt)
    L_norm = torch.eye(N, device=X.device) - D_inv_sqrt_mat @ A_dissim @ D_inv_sqrt_mat
    
    # Step 7: Eigen-decomposition, extract smallest eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
    P_dissim = eigenvectors[:, :num_dim]  # smallest num_dim eigenvectors
    
    return P_dissim
```

### Important details

1. **`dissim_threshold` is a key hyperparameter.** Start with values in {0.3, 0.5, 0.7}. Lower values select only very dissimilar edges (sparse A_dissim), higher values include more edges.

2. **Per-subgraph vs full-graph condition**: The original `obtain_attributes()` computes A' on the **full graph** and then indexes into it per subgraph. The new implementation should follow the same pattern: compute A'_dissim on the full graph, then for each subgraph, extract the corresponding sub-matrix of P'_dissim.

3. **Compatibility**: The output P'_dissim must have the same shape and role as the original P' (x_sim). It feeds into `self.zero_conv1(x_sim)` in the model's forward pass. No model code changes needed for this track.

4. **The PE extraction uses smallest eigenvectors** (same as baseline). This is intentional — we are changing only the topology, not the spectral selection, to isolate the effect of topology change.

### CLI interface

Add a command-line flag to select the condition mode:

```
--condition_mode similarity    # default, original behavior
--condition_mode dissimilarity # new: dissimilarity ∩ A
--dissim_threshold 0.5         # threshold for dissimilarity condition
```

---

## Track 2: Operator Probe (O1)

### What to change

Modify the **trainable copy's GIN aggregation** from sum (low-pass) to subtraction (high-pass).

### Current GIN aggregation (in `encoder.py`, GIN_Encoder)

The GIN layer computes:
```
h_v^{(l+1)} = MLP^{(l)}( (1 + ε) · h_v^{(l)} + Σ_{u ∈ N(v)} h_u^{(l)} )
```

This is a low-pass filter: it averages/sums neighbor features, smoothing the signal.

### Anti-smoothing GIN (probe variant)

Replace sum with subtraction:
```
h_v^{(l+1)} = MLP^{(l)}( (1 + ε) · h_v^{(l)} - Σ_{u ∈ N(v)} h_u^{(l)} )
```

This is a high-pass filter: it computes the difference between a node and its neighbors, preserving heterophilic signals.

### Implementation

**Option A (recommended): Minimal code change in GIN layer**

Locate the GIN convolution layer used by the trainable copy. In PyG's GINConv or the custom implementation, the aggregation is typically:

```python
# Inside GINConv or equivalent
out = (1 + self.eps) * x + self.aggr(x, edge_index)  # aggr = sum
```

For the anti-smoothing variant, change the `+` to `-`:

```python
out = (1 + self.eps) * x - self.aggr(x, edge_index)
```

**Option B: Create a separate encoder class**

Create `AntiSmoothingGIN_Encoder` that inherits from `GIN_Encoder` but overrides the aggregation sign. This avoids modifying the original encoder code.

### Important details

1. **Only the trainable copy is modified.** The frozen encoder remains unchanged (standard GIN with sum aggregation). This is critical — the frozen encoder's weights were pre-trained with sum aggregation.

2. **Pre-trained weight initialization**: The trainable copy currently loads GCC pre-trained weights. When the aggregation rule changes (sum → subtract), the MLP weights from pre-training may still be usable as initialization since MLP weights are independent of the aggregation rule. However, the behavior will differ from pre-training, so the model needs to learn to adapt. **Do NOT use random initialization** — keep pre-trained MLP weights.

3. **Zero-init still applies**: `zero_conv1` and `zero_conv2` are still zero-initialized, so at the start of training, the anti-smoothing branch contributes nothing (same as baseline). The anti-smoothing signal grows gradually during training.

### CLI interface

```
--operator_mode standard       # default, original GIN (sum)
--operator_mode anti_smoothing # new: GIN with subtraction
```

---

## Experiment Execution Plan

### Phase 1: Implementation (Day 1)

1. Implement `--condition_mode dissimilarity` in `utils/transforms.py`.
   - Keep the original `obtain_attributes()` intact; add a new function or branch.
   - Log statistics: number of edges in A'_dissim, density, and overlap ratio with original A.

2. Implement `--operator_mode anti_smoothing` in the model.
   - Create a minimal variant of the GIN encoder with sign-flipped aggregation.
   - Ensure the trainable copy uses this variant while the frozen encoder stays unchanged.

3. Verify both modifications run without errors on Cora_ML (1 seed).

### Phase 2: Main Experiments (Day 2–3)

Run the 2×2 matrix on all 6 datasets:

```bash
# Baseline (already have these numbers)
python graphcontrol.py --dataset Cora_ML --condition_mode similarity --operator_mode standard --seeds 0

# C1: Condition redesign only
python graphcontrol.py --dataset Cora_ML --condition_mode dissimilarity --operator_mode standard --dissim_threshold 0.5 --seeds 0

# O1: Operator probe only
python graphcontrol.py --dataset Cora_ML --condition_mode similarity --operator_mode anti_smoothing --seeds 0

# C1+O1: Both
python graphcontrol.py --dataset Cora_ML --condition_mode dissimilarity --operator_mode anti_smoothing --dissim_threshold 0.5 --seeds 0
```

Repeat for: DBLP, Photo, Chameleon, Squirrel, Actor.

For C1, also sweep `dissim_threshold` ∈ {0.3, 0.5, 0.7} on Cora_ML and Squirrel first to find a reasonable value before running all datasets.

### Phase 3: Diagnostic Measurements (Day 3–4)

For each configuration, record:

1. **Accuracy**: combined_acc, frozen_acc, control_acc (reuse Exp1 logic from previous `analyze.py`)
2. **Condition topology statistics**:
   - Number of edges in A'
   - Edge homophily ratio of A' (what fraction of edges in A' connect same-class nodes)
   - Overlap ratio: |A' ∩ A| / |A'|
3. **Spectral diagnostic** (if time permits):
   - RQ of frozen and control branches (reuse Exp4 logic)

### Phase 4: Analysis (Day 4–5)

Produce the following output tables:

**Table 1: 2×2 Combined Accuracy**

```
Dataset     | Baseline | C1 only | O1 only | C1+O1 | Oracle (reference)
Cora_ML     | ...      | ...     | ...     | ...   | 0.8632
DBLP        | ...      | ...     | ...     | ...   | 0.9120
Photo       | ...      | ...     | ...     | ...   | 0.9482
Chameleon   | ...      | ...     | ...     | ...   | 0.6833
Squirrel    | ...      | ...     | ...     | ...   | 0.9079
Actor       | ...      | ...     | ...     | ...   | 0.9289
```

**Table 2: Branch Accuracy (Frozen / Control / Combined) per Config**

**Table 3: Condition Topology Statistics**

```
Dataset     | Condition | #Edges | Density | Homophily_Ratio | Overlap_with_A
Cora_ML     | sim       | ...    | ...     | ...             | ...
Cora_ML     | dissim    | ...    | ...     | ...             | ...
...
```

**Table 4: dissim_threshold Sweep (Cora_ML, Squirrel only)**

```
Dataset   | dissim_threshold | #Edges | Acc    | Frozen_Acc | Control_Acc
Cora_ML   | 0.3              | ...    | ...    | ...        | ...
Cora_ML   | 0.5              | ...    | ...    | ...        | ...
Cora_ML   | 0.7              | ...    | ...    | ...        | ...
Squirrel  | 0.3              | ...    | ...    | ...        | ...
...
```

---

## Decision Criteria After Experiments

After completing the 2×2 matrix, answer these questions:

1. **Does C1 improve heterophilic datasets without hurting homophilic ones?**
   - If yes → condition redesign is a valid direction, proceed to refine.
   - If no → the bottleneck may not be condition quality alone.

2. **Does O1 improve anything?**
   - If yes → propagation mechanism matters, proceed to more sophisticated operator redesign.
   - If no → operator-level changes are deprioritized.

3. **Is C1+O1 > max(C1, O1)?**
   - If yes → the two axes are complementary and both should be pursued.
   - If no → one axis dominates; focus on that.

4. **How does the best config compare to the oracle?**
   - If close → the proposed unsupervised method is effective.
   - If far → there's room for improvement, consider learnable condition generation.

---

## Critical Implementation Notes

1. **Do NOT modify** the frozen encoder or its PE computation (`process_attributes()`). Only the condition (x_sim / P') and the trainable copy's propagation are being changed.

2. **Subgraph-level computation**: Remember that GraphControl operates on subgraphs, not the full graph. The condition PE (P') is computed on the full graph in `obtain_attributes()` and then indexed per subgraph. The dissimilarity A' should be computed on the full graph as well, and the resulting P'_dissim should be indexed in the same way.

3. **Memory**: Computing the full N×N cosine similarity matrix may be expensive for large graphs. For Actor (N≈7,600) and Squirrel (N≈5,200) this should be fine. If memory is an issue, use sparse computation.

4. **Edge case handling**: If A'_dissim has disconnected components (likely for low dissim_threshold), the Laplacian eigen-decomposition will have multiple zero eigenvalues. This is fine — the smallest eigenvectors will capture connected component structure, which is still informative.

5. **Logging**: For every run, log and save:
   - All accuracies (frozen, control, combined)
   - Condition topology stats (edge count, homophily ratio, overlap with A)
   - Training loss curves
   - Hyperparameters used

6. **Results directory**: Save all results to `experiment_results/dual_track/` with subdirectories per config:
   ```
   experiment_results/dual_track/
   ├── baseline/
   ├── C1_dissim/
   ├── O1_antismooth/
   ├── C1_O1_combined/
   └── threshold_sweep/
   ```
