# AGENT.md — Fast Pre-Meeting Experiments for GraphControl / SSLC

## Objective

You are working in a GraphControl codebase that already contains a standard fine-tuning pipeline and a completed diagnostic analysis pipeline. Your job is to implement **two independent, minimal experiments** that can be run separately before a meeting.

These experiments are **not** the full SSLC method. They are intended to answer two narrow questions:

1. **Experiment 1 — Fusion Ablation:**  
   Is the limitation of current GraphControl mainly due to the simple additive fusion `h = h_frozen + h_control`?  
   In other words, can we get a meaningful gain by changing only the fusion mechanism while keeping the two branches unchanged?

2. **Experiment 2 — Mini-SSLC Operator Separation:**  
   Does a minimal operator-level separation in the trainable branch already change the behavior of GraphControl in a useful way?  
   In other words, if the trainable branch no longer uses exactly the same structural operator as the frozen branch, do we observe a better branch role separation?

The two experiments must be implemented as **two independent Python scripts**, so that Experiment 1 can be completed and summarized first, and Experiment 2 can be run afterward.

---

## Critical Constraints

1. **Do not modify existing source files.**
   - Do **not** edit `graphcontrol.py`, `gcc.py`, existing model files, or existing utility files.
   - Create new files only.

2. **Experiment 1 and Experiment 2 must be independent scripts.**
   - Example:
     - `run_fusion_ablation.py`
     - `run_operator_separation.py`

3. Each script must:
   - fine-tune the model from scratch using the existing GraphControl pipeline logic,
   - evaluate on the test set,
   - save a compact CSV / TXT summary into a new output directory.

4. Keep implementation minimal and robust.
   - No large framework refactor.
   - No dependency on unfinished SSLC modules.
   - Reuse as much existing pipeline logic as possible.

5. Use the original GraphControl training setup as the default.
   - Early stopping / patience should match the original setup as closely as possible.
   - Use the same preprocessing pipeline and subgraph sampling pipeline.

---

## Codebase Context

The current GraphControl forward structure is approximately:

```python
# frozen branch
h_frozen = self.encoder.forward_subgraph(x, edge_index, batch, root_n_id)

# trainable branch
x_down = self.zero_conv1(x_sim)
x_down = x_down + x
h_tc_raw = self.trainable_copy.forward_subgraph(x_down, edge_index, batch, root_n_id)
h_control = self.zero_conv2(h_tc_raw)

# current fusion
out = h_frozen + h_control
logits = self.linear_classifier(out)
```

Key properties:
- Both branches currently use the **same `edge_index`** for message passing.
- `x` is the Laplacian PE from the original adjacency.
- `x_sim` is the condition PE from the feature-similarity-based adjacency.
- The frozen branch is fixed; the trainable branch is optimized during downstream fine-tuning.

---

# Experiment 1 — Fusion Ablation

## Scientific Question

If we keep both branches unchanged and only replace the fusion rule, can we significantly improve the downstream performance?

This experiment is meant to test whether the current limitation is **just a fusion problem** or whether the branches themselves are poorly structured for complementarity.

## Required Script

Create:

```text
run_fusion_ablation.py
```

## Required Variants

Implement the following fusion variants:

### Variant A — Baseline Additive Fusion
```python
out = h_frozen + h_control
```

### Variant B — Learnable Scalar Gate
```python
alpha = sigmoid(a)   # a is a learnable scalar parameter
out = alpha * h_frozen + (1 - alpha) * h_control
```

Requirements:
- `a` is a trainable scalar parameter.
- Initialize it so that `alpha` starts near 0.5.

### Variant C — Node-wise Gate
```python
g = sigmoid(MLP(concat(h_frozen, h_control)))   # shape: (N, 1)
out = g * h_frozen + (1 - g) * h_control
```

Requirements:
- Gate MLP should be lightweight.
- Suggested structure: Linear(2d -> d), ReLU, Linear(d -> 1)
- `d` = branch representation dimension.

### Optional Variant D — Dimension-wise Gate
Only implement this if time is short-friendly and clean:
```python
g = sigmoid(MLP(concat(h_frozen, h_control)))   # shape: (N, d)
out = g ⊙ h_frozen + (1 - g) ⊙ h_control
```

If this adds too much complexity, omit it.

## Implementation Notes

- Do not modify the existing GraphControl class directly.
- Create a wrapper or subclass for analysis/training with custom fusion.
- The branch computation should remain identical to the original GraphControl.
- Only the fusion rule should change.

## Datasets to Prioritize

Run at least:
- `Cora_ML`
- `Chameleon`
- `Squirrel`

If time permits, additional datasets can be added later.

## Metrics to Save

For each dataset and fusion variant, save:
- test accuracy
- validation accuracy at the selected checkpoint
- best epoch

Additionally, for the gated variants save:
- scalar gate value (for scalar gate)
- mean gate value over test nodes (for node-wise gate)
- std of gate value over test nodes (for node-wise gate)

## Output Format

Create an output directory such as:

```text
fusion_ablation_results/
```

and save:

```text
fusion_ablation_results/
├── summary.csv
├── Cora_ML_baseline.txt
├── Cora_ML_scalar_gate.txt
├── Cora_ML_node_gate.txt
├── Chameleon_baseline.txt
├── ...
```

Suggested `summary.csv` columns:

```text
Dataset,Variant,ValAcc,TestAcc,BestEpoch,GateMean,GateStd,ScalarAlpha
```

If a field is not applicable, leave it blank.

## Desired Interpretation

This experiment should make it easy to answer:

- Does fusion-only redesign significantly improve over baseline?
- Or does the trainable branch still dominate even with smarter fusion?

---

# Experiment 2 — Mini-SSLC Operator Separation

## Scientific Question

If we minimally change the operator used by the trainable branch, does GraphControl begin to behave more like a genuinely complementary dual-branch model?

This experiment is meant to test whether **operator-level separation** matters, even before building the full SSLC design.

## Required Script

Create:

```text
run_operator_separation.py
```

This script should be independent from `run_fusion_ablation.py`.

## Core Idea

Keep the frozen branch unchanged.

Change only the trainable branch so that it does **not** use exactly the same topology as the frozen branch.

The simplest and most realistic version is:

- Frozen branch: uses the original sampled subgraph `edge_index`
- Trainable branch: uses a topology derived from the **condition graph** rather than the original graph

## Minimal Required Variant

### Variant A — Baseline GraphControl
Trainable branch uses the same original `edge_index`.

### Variant B — Condition-Topology Trainable Branch
Trainable branch uses an alternative edge index derived from the condition graph / feature-similarity graph.

The exact implementation may depend on what is easiest in the current codebase, but the intended meaning is:

- the frozen branch propagates on the original sampled subgraph structure,
- the trainable branch propagates on a structurally different graph induced by the condition.

## Acceptable Practical Implementations

You may choose **one** of the following depending on code feasibility:

### Option 1 — Condition-only topology
Build trainable-branch `edge_index_control` purely from the condition adjacency / feature-similarity graph.

### Option 2 — Mixed topology
Build:
```python
A_control = A ∪ A_condition
```
and use its edge index for the trainable branch.

### Option 3 — Complementary topology
Build a trainable branch graph that keeps edges emphasized by the condition but not already dominant in the original graph.

If full complementary extraction is too time-consuming, use Option 1 or 2.

## Important Implementation Principle

This is **not** a full new method. It is only a minimal test of whether changing the structural operator for the trainable branch changes behavior in a useful way.

Do not add extra regularizers, spectral losses, or complex gating here.

Keep it simple:
- same classifier,
- same optimization,
- same training loop,
- only operator/topology for the trainable branch changes.

## Datasets to Prioritize

Run at least:
- `Cora_ML`
- `Chameleon`
- `Squirrel`

## Metrics to Save

For each dataset and variant, save:
- validation accuracy
- test accuracy
- best epoch

If feasible, also save:
- frozen-only test accuracy
- control-only test accuracy
- combined test accuracy

This is very helpful because it lets us see whether operator separation changes branch role distribution.

## Output Format

Create:

```text
operator_separation_results/
```

and save:

```text
operator_separation_results/
├── summary.csv
├── Cora_ML_baseline.txt
├── Cora_ML_condition_operator.txt
├── Chameleon_baseline.txt
├── ...
```

Suggested `summary.csv` columns:

```text
Dataset,Variant,ValAcc,TestAcc,BestEpoch,FrozenOnlyAcc,ControlOnlyAcc,CombinedAcc
```

---

# Shared Implementation Guidance

## Reuse Existing Training Logic

Both scripts should reuse the GraphControl fine-tuning logic from the original code as much as possible:
- dataset loading
- preprocessing
- subgraph collection
- PE construction
- optimizer setup
- early stopping

Do not rewrite the pipeline from scratch unless necessary.

## Prefer Wrappers / Lightweight Subclasses

If you need branch access or custom fusion/operator handling:
- create a wrapper model,
- or create a lightweight subclass in the new script itself,
- or create a new helper file if absolutely necessary.

But avoid editing original source files.

## Reproducibility

Each script should accept:
- `--dataset`
- `--seeds`
- `--lr`
- `--optimizer`
- `--threshold`
- `--use_adj`
- `--model`

At minimum, mimic the current command-line interface enough to run easily on existing datasets.

## Recommended Initial Runs

For quick screening:

```bash
python run_fusion_ablation.py --dataset Cora_ML --seeds 0 1 --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj --model GCC_GraphControl
python run_fusion_ablation.py --dataset Chameleon --seeds 0 1 --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj --model GCC_GraphControl
python run_fusion_ablation.py --dataset Squirrel --seeds 0 1 --lr 0.5 --optimizer adamw --threshold 0.15 --use_adj --model GCC_GraphControl

python run_operator_separation.py --dataset Cora_ML --seeds 0 1 --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj --model GCC_GraphControl
python run_operator_separation.py --dataset Chameleon --seeds 0 1 --lr 0.5 --optimizer adamw --threshold 0.17 --use_adj --model GCC_GraphControl
python run_operator_separation.py --dataset Squirrel --seeds 0 1 --lr 0.5 --optimizer adamw --threshold 0.15 --use_adj --model GCC_GraphControl
```

If stable and useful, expand to more seeds later.

---

# Expected Deliverables

At the end, you should provide:

1. `run_fusion_ablation.py`
2. `run_operator_separation.py`
3. a short note or terminal summary explaining:
   - what was implemented,
   - how to run each script,
   - where results are saved.

---

# Success Criteria

## Experiment 1 succeeds if it clearly answers:
- Is fusion-only redesign enough?

## Experiment 2 succeeds if it clearly answers:
- Does minimal operator separation already matter?

Even negative results are valuable, as long as the scripts run cleanly and produce interpretable outputs.

---

# Final Reminder

Do not chase a full SSLC implementation here.

The goal is to obtain **fast, decision-quality evidence** before the meeting:
- whether fusion-only is insufficient,
- and whether operator-level separation is promising.
