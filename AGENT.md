# AGENT.md

## Purpose

This document defines the implementation task for extending the original GraphControl codebase so that we can:

1. keep the current GraphControl baseline fully reproducible,
2. preserve the already completed diagnostics:
   - Experiment 1: branch-wise classification accuracy,
   - Experiment 4: Rayleigh Quotient analysis,
3. add the remaining instrumentation needed for the next research step:
   - sample-wise margin,
   - sample-wise margin gain,
   - infrastructure for future `L_gain` insertion.


The purpose of introducing the margin gain objective is to address a fundamental limitation of the current GraphControl architecture. Empirical analysis shows that the trainable control branch dominates prediction, while the frozen branch contributes negligibly, indicating that the model does not function as a true residual system.

To correct this, we aim to encourage the trainable branch to improve predictions relative to the frozen branch, rather than independently solving the entire task. The margin gain formulation explicitly measures how much the combined model improves over the frozen baseline at the sample level, and is designed to guide the trainable branch toward learning complementary, residual information instead of duplicating or overriding the frozen representation.

Note: The priority is **clean implementation, minimal disruption to the original code, and reproducible experimentation**.
---

## Current Research Status

The following parts are already completed and should be treated as verified baseline outputs:

### Completed
- Baseline GraphControl run is fixed.
- Experiment 1 output pipeline is implemented:
  - Frozen branch accuracy
  - Control branch accuracy
  - Combined branch accuracy
- Experiment 4 output pipeline is implemented:
  - Rayleigh Quotient for frozen branch
  - Rayleigh Quotient for control branch

### Not yet completed
- Sample-wise margin extraction
- Sample-wise margin gain extraction
- `L_gain`-ready forward/loss design
- Unified metric export format for these new outputs

---

## High-Level Goal

The next implementation step is **not yet to redesign the model**.

The immediate goal is:

> instrument the original GraphControl pipeline so that we can observe how the current baseline behaves at the sample level, especially in terms of margin and margin gain, before introducing `L_gain`.

This means:
- do **not** change the core learning logic yet,
- do **not** introduce new training objectives yet,
- do **prepare the code so that `L_gain` can be inserted cleanly later**.

---

## Important Conceptual Clarification

The current GraphControl formulation should be treated as:

- frozen branch:
  - input: original positional encoding `P`
  - output: `h_f`
- control branch:
  - input: `P + Z1(P')`
  - trainable copy output passed through `Z2`
  - output: `h_c = Z2(f_control(P + Z1(P')))`
- final representation:
  - `h = h_f + h_c`

The agent must not rewrite this logic unless explicitly instructed later.

---

## Implementation Principles

### 1. Preserve baseline behavior
Any new code must preserve the current baseline training result when no new loss is activated.

### 2. Prefer additive instrumentation
Add logging, extraction, and helper functions in a way that does not alter optimization unless explicitly enabled.

### 3. Keep the code easy to diff
Avoid large refactors unless absolutely necessary. Prefer:
- wrapper methods,
- helper utilities,
- clearly isolated metric functions,
- optional flags.

### 4. Make future `L_gain` insertion easy
Even if `L_gain` is not activated now, the forward pass and outputs should expose the tensors needed later.

---

## Required Outputs for the Next Step

The implementation must enable extraction of the following quantities for each evaluation sample:

### A. Frozen logits
- `z_f`

### B. Combined logits
- `z_fc`

### C. True-class margin
For logits `z` and label `y`, define:

`m(z, y) = z_y - max_{k != y} z_k`

Need:
- `margin_f = m(z_f, y)`
- `margin_fc = m(z_fc, y)`

### D. Margin gain
- `margin_gain = margin_fc - margin_f`

These should be computed sample-wise and then exportable as:
- raw per-sample values,
- summary statistics.

---

## Concrete Task List

## Task 1. Inspect and isolate forward outputs

### Goal
Identify where to cleanly extract:
- frozen branch representation `h_f`,
- control branch representation `h_c`,
- combined representation `h_f + h_c`,
- frozen logits `z_f`,
- combined logits `z_fc`.

### Requirements
- Do not duplicate unnecessary forward computation.
- If needed, add a new analysis/eval forward method that returns intermediate tensors.
- Keep the original training forward intact if possible.

### Preferred design
Implement something like:

```python
forward_with_diagnostics(...)
    -> {
        "h_f": ...,
        "h_c": ...,
        "h_fc": ...,
        "z_f": ...,
        "z_c": ... optional,
        "z_fc": ...,
    }
```

If `z_c` is already easy to compute, include it as well.

---

## Task 2. Implement margin computation utilities

### Goal
Create stable helper functions for margin extraction.

### Required function behavior
Given:
- logits tensor of shape `[N, C]`
- label tensor of shape `[N]`

compute:
- true class logit
- largest non-true-class logit
- margin per sample

### Notes
- Must work on batched data.
- Must avoid Python loops when possible.
- Should be numerically straightforward and easy to test.

### Recommended API
```python
compute_true_class_margin(logits, labels) -> Tensor[N]
```

and optionally

```python
compute_margin_gain(z_f, z_fc, labels) -> Tensor[N]
```

---

## Task 3. Add metric collection pipeline

### Goal
At evaluation time, collect per-sample values for:
- `margin_f`
- `margin_fc`
- `margin_gain`

### Output requirements
For each dataset / seed run, export:
- dataset name
- seed
- split name if relevant
- sample index or node id if available
- label
- prediction from frozen branch
- prediction from combined branch
- `margin_f`
- `margin_fc`
- `margin_gain`

### File format
Use CSV unless there is a strong reason not to.

Recommended filename pattern:
- `margin_metrics_<dataset>_seed<k>.csv`

---

## Task 4. Add summary export

### Goal
Also export aggregate summaries, such as:
- mean margin_f
- mean margin_fc
- mean margin_gain
- std of margin_gain
- fraction of samples with positive margin_gain
- fraction of samples with negative margin_gain

Recommended filename pattern:
- `margin_summary_<dataset>.csv`

---

## Task 5. Prepare for future `L_gain` insertion

### Goal
Restructure the loss computation minimally so that `L_gain` can later be activated with a flag.

### Current phase requirement
Do not train with `L_gain` yet unless explicitly requested later.

### Required preparation
The code should make it obvious where future logic would go:

```python
loss = ce_loss
if use_l_gain:
    loss = loss + lambda_gain * l_gain
```

You do not need to finalize the training behavior now, but the forward path must already expose:
- `z_f`
- `z_fc`

during training-time computation if future insertion requires it.

---

## Scope Boundaries

## In scope
- Adding diagnostic forward outputs
- Adding margin utilities
- Adding CSV export for raw/sample-wise margin metrics
- Adding CSV export for summary margin metrics
- Preparing `L_gain` insertion points

## Out of scope for this stage
- Changing the GraphControl architecture
- Changing condition generation
- Changing adjacency
- Adding fusion modules
- Introducing anti-smoothing propagation
- Running large hyperparameter searches
- Replacing the original training pipeline entirely

---

## Code Organization Guidance

The agent should prefer one of the following approaches:

### Option A. Add a dedicated analysis script
Example:
- `analyze_margin.py`

This is good if the original training/eval script is messy and should remain untouched.

### Option B. Add a lightweight wrapper around the current model/eval logic
This is good if the existing code already exposes most tensors.

### Option C. Add optional diagnostic mode into the current evaluation path
Only if this can be done cleanly without damaging readability.

### Strong preference
Prefer **isolated additions** over invasive rewrites.

---

## Recommended Output Structure

Suggested directory:

```text
analysis_results/
├──{dataset_name}
│   ├── exp1_branch_accuracy.csv # existing file
│   ├── exp2_redundancy_metrics.csv # existing file
│   ├── exp2_cosine_histograms # existing file
│   ├── exp3_homophily_contribution # existing file
│   ├── exp4_spectral_analysis # existing file
│   ├── margin_raw/
|   │   ├── margin_metrics_Cora_ML_seed0.csv
│   |   ├── margin_metrics_Cora_ML_seed1.csv
│   |   └── ...
|   ├── margin_summary/
│   |   ├── margin_summary_Cora_ML.csv
│   |   ├── margin_summary_Chameleon.csv
│   |   └── ...
│   └── ...
```

If the project already has a result directory convention, follow that instead.

---

## Validation Checklist

Before considering the implementation complete, verify the following:

### Functional validation
- Baseline accuracy remains unchanged when no new loss is enabled.
- Margin CSVs are generated successfully.
- Margin values are not all zero or constant.
- `margin_gain = margin_fc - margin_f` is computed correctly.
- Summary statistics match the raw exported values.

### Structural validation
- New code paths are optional and do not break the original baseline.
- Intermediate outputs are clearly named.
- Future `L_gain` insertion is obvious from the code structure.

### Minimal sanity checks
For at least one dataset:
- run one seed,
- export raw margin metrics,
- export summary CSV,
- inspect a few rows manually.

---

## Expected Research Use of the Output

The newly exported metrics will be used to answer the following question:

> Is the current GraphControl baseline behaving like a true residual helper for the frozen branch, or is the trainable branch simply dominating prediction across most samples?

This margin-level analysis is intended to provide the empirical justification for introducing `L_gain` in the next step.

---

## Final Instruction to the Agent

Implement only what is necessary to make the current GraphControl baseline observable at the sample level in terms of margin and margin gain.

Do not over-engineer.
Do not redesign the model yet.
Make the next research step easy, clean, and reproducible.
