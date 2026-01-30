# QAMP Project: Local and Multi-Scale Strategies to Mitigate Exponential Concentration in Quantum Kernels

---

**Goal:** Build and evaluate quantum kernels for SVMs and test whether **local (patch-wise)** and **multi-scale** kernels mitigate **exponential concentration** as qubit count (dimension `d`) and/or depth increase.  
**Status:** Active research codebase with reproducible benchmarks, summaries, and plots (see checkpoint docs).

---

## 1) Overview

Quantum fidelity kernels tend to **concentrate** as `d` grows, pushing off-diagonal kernel entries toward 0 (kernel -> identity), which can reduce separability. This repo implements and benchmarks three kernel families under a **unified API**:

1. **Baseline (global fidelity):** overlap of full quantum states.  
2. **Local (patch-wise):** compare **subsystems** via subcircuits or reduced density matrices (RDMs), then aggregate.  
3. **Multi-scale:** **non-negative mix** of kernels computed at multiple granularities (local + global).

We evaluate kernel geometry (off-diagonal statistics, effective rank, alignment) and downstream SVM accuracy, with sweep studies over **d = 4..20** on multiple datasets. See `docs/checkpoints/` for detailed results and plots.

---

## 2) Repository layout

```
project/
  qkernels/                  # kernel implementations + feature maps
  analysis/                  # diagnostics, plotting, summaries
  scripts/                   # benchmark runners + demos
  configs/                   # TOML experiment configs
  datasets/                  # local CSV datasets (large)
  outputs/                   # kernels, metadata, summaries
  figs/                      # generated figures
  docs/                      # checkpoint notes and recipes
  tests/                     # pytest unit tests
  PLAN.md                    # project plan
  README.md                  # this file
```

---

## 3) Installation

**Python:** 3.12 recommended (tested).

### 3.1 Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 3.2 Git LFS (large `.npy` artifacts)
This repo tracks large kernel matrices and outputs with **Git LFS**.  
If you need the full artifacts, install LFS and pull:

```bash
git lfs install
git lfs pull
```

**Clone without downloading large `.npy` files (LFS pointers only):**
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <repo_url>
git lfs install
```

You can later fetch artifacts selectively:
```bash
git lfs fetch --include "outputs/**,figs/**"
git lfs checkout --include "outputs/**,figs/**"
```

---

## 4) Configuration with TOML

All experiments are controlled by **TOML configs** in `configs/`. These define:
- dataset + preprocessing options,
- seed grid + train/val/test splits,
- feature-map depth + entanglement,
- kernel variants (baseline/local/multiscale),
- optional Nyström settings.

Example (abbreviated):
```toml
[run]
dataset = "breast_cancer"
seed_grid = [42, 43]
n_features = 8
pca = true
val_size = 0.2
test_size = 0.2

[feature_map]
name = "zz_qiskit"
depth_grid = [1]
entanglement = "linear"
backend = "statevector"

[post]
normalize = true
center_grid = [false]
report_rank = true

[svm]
C_grid = [0.1, 1.0, 10.0]

[[kernels]]
name = "baseline"
enabled = true

[[kernels]]
name = "local"
enabled = true
partitions = [[0,1],[2,3]]
method = "rdm"
agg = "mean"

[[kernels]]
name = "multiscale"
enabled = true
scales = [ [[0,1],[2,3]], [[0,1,2,3]] ]
weights_grid = [[0.5, 0.5]]

[nystrom]
enabled = false
```

Notes:
- `run_experiment.py` resolves config paths relative to repo root.
- `seed_grid`, `depth_grid`, `center_grid`, and `weights_grid` define sweeps.
- Local datasets (CSV) live in `datasets/`.

---

## 5) Unified Kernel API

Each kernel module exposes:

```python
def build_kernel(X, feature_map="zz", depth=1, backend="statevector", seed=42, **kwargs):
    """
    Returns:
      K    : np.ndarray (n, n)  # symmetric, ~PSD, float64
      meta : dict               # full config (for logging)
    """
```

Kernel-specific kwargs:
- **Baseline:** `entanglement`  
- **Local:** `partitions`, `method` ("subcircuits" | "rdm"), `agg`, `weights`, `rdm_metric`  
- **Multi-scale:** `scales`, `weights`, `normalize`

---

## 6) End-to-end flow (scripts and outputs)

This is the **recommended flow** to generate all tables and plots:

More runnable examples are collected in **[docs/RECIPES.md](docs/RECIPES.md)**.

1) **Run benchmarks (build kernels + diagnostics + SVM metrics)**  
```bash
python -m scripts.run_experiment --config configs/breast_cancer_d8.toml
```

2) **Summarize outputs into CSV/Markdown**  
```bash
python -m analysis.summarize_benchmarks \
  --roots outputs/benchmarks/breast_cancer_d4 outputs/benchmarks/breast_cancer_d6 \
  --out outputs/benchmarks/summary_all.csv \
  --md outputs/benchmarks/summary_all.md
```

3) **Plot vs d curves (concentration / effective rank / accuracy)**  
```bash
python -m analysis.plot_vs_d \
  --summary outputs/benchmarks/summary_all.md \
  --out figs/checkpoint3/vs_d \
  --also-p95
```

4) **Delta analysis (vs baseline) + heatmaps + tradeoffs**  
```bash
python -m analysis.plot_deltas \
  --summary outputs/benchmarks/summary_all.csv \
  --out figs/checkpoint3/deltas
```

5) **Checkpoint-specific comparison figures (optional)**  
```bash
python -m analysis.make_checkpoint2_figure \
  --baseline <baseline_K.npy> \
  --local <local_K.npy> \
  --multiscale <multiscale_K.npy> \
  --out figs/checkpoint2/breast_cancer_compare.png
```

You can also run multiple configs and build per-dataset + global summaries:
```bash
python -m scripts.run_all_benchmarks --configs configs/breast_cancer_d4.toml configs/breast_cancer_d6.toml
```

---

## 7) Outputs and figures

### 7.1 Raw artifacts (per run)
- `outputs/benchmarks/<dataset>_d*/<case>_K.npy`  
- `outputs/benchmarks/<dataset>_d*/<case>_K_centered.npy` (optional)
- `outputs/benchmarks/<dataset>_d*/<case>_meta.json`
- `outputs/benchmarks/<dataset>_d*/<case>_splits.json`
- `outputs/benchmarks/<dataset>_d*/metrics.csv`

### 7.2 Diagnostics figures (per run)
Generated by `analysis.diagnostics` or `run_experiment.py`:
- `*_matrix.png` (kernel heatmap)
- `*_offdiag_hist.png` (off-diagonal histogram)
- `*_spectrum.png` (eigen-spectrum)

### 7.3 Summary tables
Generated by `analysis.summarize_benchmarks`:
- `summary.csv`
- `summary.md`
- (global) `summary_all.csv` / `summary_all.md`

### 7.4 vs-d plots
Generated by `analysis.plot_vs_d`:
- `<dataset>_concentration_p50_vs_d.png`
- `<dataset>_concentration_p95_vs_d.png`
- `<dataset>_effrank_vs_d.png`
- `<dataset>_testacc_vs_d.png`
- `<dataset>_vs_d_curves.csv`

### 7.5 Delta plots (vs baseline)
Generated by `analysis.plot_deltas`:
- `delta_by_d.csv`
- `delta_by_dataset.csv`
- `delta_delta_test_acc_<dataset>.png`
- `delta_delta_offdiag_p50_<dataset>.png`
- `delta_delta_eff_rank_<dataset>.png`
- `tradeoff_<dataset>.png`
- `bar_mean_delta_test_acc.png`
- `heatmap_delta_test_acc_local.png`
- `heatmap_delta_test_acc_multiscale.png`

Checkpoint-specific figures and narratives live in `docs/checkpoints/`.

---

## 8) Nyström option for large datasets

For large datasets, `run_experiment.py` supports **Nyström/landmark** approximation to avoid full $O(n^2)$ kernel matrices. Enable in TOML:

```toml
[nystrom]
enabled = true
datasets = ["star_classification", "exam_score_prediction"]
n_landmarks = 1000
diag_samples = 2000
chunk_size = 256
```

In Nyström mode, the pipeline builds cross-kernels and evaluates a linear SVM on explicit features (`eval_linear_features`). This is optional and can be skipped if the full kernel is feasible.

---

## 9) Datasets

Supported datasets in `scripts/demo_common.py`:
- `make_circles`, `iris`
- `breast_cancer`, `parkinsons`
- `exam_score_prediction`, `star_classification`
- `ionosphere`, `heart_disease`

Some large datasets are run as **subsets** for tractability; see `configs/` and checkpoint notes for details.

---

## 10) Troubleshooting

- **Kernel not PSD:** use `float64`, enforce symmetry `(K + K.T)/2`, add small diagonal reg.
- **Slow runs:** keep depth 1-2, reduce `n`, prefer statevector first.
- **Mismatched splits:** always reuse the `*_splits.json` produced by each run.
- **Windows paths:** all scripts resolve config/output paths relative to repo root.

---

## 11) License & citation

**License:** TBD (MIT suggested).  
If you use this code, please cite the project and relevant QML/quantum-kernel references.

---

## 12) Glossary (quick)

- **MCC:** Matthews Correlation Coefficient (robust classification metric).
- **Nyström:** Low-rank kernel approximation using landmarks.
- **PSD:** Positive Semi-Definite.
- **RDM:** Reduced Density Matrix.
- **TOML:** Simple, human-readable configuration format.

---

> **Status notice:** This README reflects the current state of the project and will continue evolving as results and experiments expand.

---

## Project plan

See **[PLAN.md](./PLAN.md)** for scope, shared interfaces, artifacts, and milestones.
