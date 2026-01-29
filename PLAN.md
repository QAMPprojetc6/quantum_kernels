# Project Plan - Local & Multi-Scale Strategies to Mitigate Exponential Concentration in Quantum Kernels

**Work mode:** async  
**Proposed split:** 1 person per kernel (Global, Local, Multi-Scale)  
**Goal:** show that Local and/or Multi-Scale kernels mitigate exponential concentration and provide better or more robust performance than the Global (fidelity) baseline, with clear diagnostics.

---

## 1) Scope (what we will build)

* **Three kernels in Qiskit**
  * **Global (baseline):** standard fidelity kernel.
  * **Local (patch-wise):** similarities via subcircuits or reduced density matrices (RDMs, Reduced Density Matrices), aggregated across patches.
  * **Multi-Scale:** weighted combination of kernels computed at multiple granularities (e.g., pairs -> all qubits).
* **Experiment pipeline** with TOML-driven sweeps (seeds, depth, `d`, weights) and reproducible outputs.
* **Diagnostics & plots:** kernel heatmaps, off-diagonal histograms, eigen-spectra, off-diag p50/p95 vs d, effective rank, test accuracy, and delta-vs-baseline analyses.
* **Reproducible artifacts:** kernels, splits, metrics, and figures stored with consistent names.

Non-goals: large-scale hardware runs, asymptotic theory.

---

## 2) Async team split (1 person = 1 kernel)

* **Global (baseline):** fidelity kernel with shared feature map.
* **Local (patch-wise):** per-patch kernels (subcircuits or RDM), aggregated (mean/weighted).
* **Multi-Scale:** cross-scale combination with non-negative weights; supports mixing local + global.

Each person works independently but adheres to the same **interfaces and artifact formats** below.

---

## 3) Common contracts (to avoid integration pain)

### 3.1 Unified API (same signature across kernels)

```python
def build_kernel(
    X,                      # np.ndarray (n_samples, d)
    feature_map="zz",       # shared across kernels
    depth=1,                # {1, 2} to start
    backend="statevector",  # "statevector" only in current implementation
    seed=42,
    **kwargs                # kernel-specific (see below)
):
    """
    Returns:
        K: np.ndarray (n, n)  # symmetric, ≈PSD, float64
        meta: dict            # full config used (for logging)
    """
```

Kernel-specific `**kwargs`:

* **Global:** `entanglement` (sampling backend not implemented).
* **Local:** `partitions=[(0,1),(2,3)]`, `method="subcircuits"|"rdm"`, `agg="mean"`, `weights=None`.
* **Multi-Scale:** `scales=[[(0,1),(2,3)],[(0,1,2,3)]]`, `weights=[0.5,0.5]`, `normalize=True`.

We share **one** `feature_maps.py` to avoid duplication.

### 3.2 Artifacts (names & formats)

* Kernel matrix: `outputs/benchmarks/<dataset>_d*/<case>_K.npy`
* Metadata: `outputs/benchmarks/<dataset>_d*/<case>_meta.json`
* Splits (shared by all): `outputs/benchmarks/<dataset>_d*/<case>_splits.json`
* Per-run plots (diagnostics):
  * `*_matrix.png`, `*_offdiag_hist.png`, `*_spectrum.png`
* Summaries:
  * `summary.csv`, `summary.md`, and global `summary_all.*`
* Aggregate plots:
  * vs-d curves (`*_concentration_p50_vs_d.png`, `*_effrank_vs_d.png`, `*_testacc_vs_d.png`)
  * delta analyses (`delta_*`, heatmaps, tradeoffs)

### 3.3 Quality checklist (each person runs locally)

* Shape: `K.shape == (n, n)`; diagonals reasonable (e.g., ≈1 for fidelity-style).
* Symmetry: `||K - K.T||_∞ < 1e-8`.
* PSD (Positive Semi-Definite) (numerical): `min(eigvals(K)) > -1e-8` (else regularize `K += 1e-6 I`).
* Determinism: fixed `seed` -> same `K` and plots.
* Timing: log wall-time for `n≈100` in `meta.json`.

---

## 4) Shared configuration (TOML files in `configs/`)

TOML files under `configs/` control datasets, seeds, feature map, partitions/scales, and paths.
`scripts/run_experiment.py` reads these configs to keep experiments comparable.

Note: TOML = Tom's Obvious, Minimal Language. Easy to read and write file, by being minimal and by using human-readable syntax.

---

## 5) Datasets, parameters, and baselines (common to all)

* **Datasets:** `make_circles`, `iris`, `breast_cancer`, `parkinsons`, `exam_score_prediction`, `star_classification`, `ionosphere`, `heart_disease`.
* **Splits & seeds:** 60/20/20; `seed_grid` in configs (splits saved and reused).
* **Feature map:** ZZ-style; `depth_grid` in configs (default depth = 1).
* **SVM (precomputed kernel):** `C_grid` in configs.
* **Local default:** contiguous pairs for partitions.
* **Multi-Scale default:** `scales = [pairs, all]`, `weights = [0.5, 0.5]`.

---

## 6) Minimal evaluation & diagnostics

For each kernel & dataset:

* **Metrics:** accuracy (val/test), off-diagonal stats (p50/p95), effective rank, alignment (optional).
* **Plots:** kernel heatmap; off-diagonal histogram; eigen-spectrum.
* **Aggregate plots:** vs-d curves and delta-vs-baseline summaries.

All code uses fixed seeds; all artifacts saved with the file names above.

---

## 7) Milestones (async, low interaction)

* **Milestone 1 - Sun, Nov 16, 2025 (async):**  
  Each person uploads for `make_circles` (≈150 samples):  
  `K.npy`, the 3 plots, and `meta.json`.

* **Milestone 2 - Fri, Dec 14, 2025 (async; aligns with mid-report):**  
  Repeat on **Iris**; add a small `metrics.csv` with SVM accuracy (precomputed kernel).

* **(Stretch)** Before **Thu, Jan 30, 2026** (MVP):  
  Identify one dataset where Local or Multi-Scale clearly wins or is more robust.

Status: pipeline extended to multi-dataset sweeps (`d=4..20`), with checkpoint reports under `docs/checkpoints/`.

Minimum meetings required; a short message per milestone would be enough.  
(It has been difficult to agree on a single schedule among everyone)

---

## 8) Repository layout (starting point)

```
project/
  qkernels/
    __init__.py
    feature_maps.py
    baseline_kernel.py     # Person A
    local_kernel.py        # Person B
    multiscale_kernel.py   # Person C
  analysis/
    diagnostics.py         # heatmap, off-diag histogram, spectrum
    eval_svm.py            # SVM with precomputed kernels
    summarize_benchmarks.py
    plot_vs_d.py
    plot_deltas.py
  scripts/
    run_experiment.py
    run_all_benchmarks.py
  configs/                 # TOML configs per dataset / sweep
  outputs/                 # kernels, splits, meta, metrics, summaries
  figs/                    # figures from diagnostics and aggregate plots
  docs/                    # checkpoint notes
  README.md
  PLAN.md                  # (this document)
```

---

## 9) Risks & simple mitigations

* **Incomparable results:** single `config.toml`, shared splits/seed, same feature map/depth.
* **Non-PSD kernels (numerical):** add `+1e-6 I`, use float64, center if needed.
* **Slow runs:** keep depth small (1–2), use subsets, or enable Nyström for large datasets.
* **Code divergence:** one shared `feature_maps.py`; identical API across kernels.

---

## 10) Acceptance criteria (per kernel)

* `build_kernel(...)` implemented with the common signature.
* Passes the **quality checklist** above.
* Produces the 3 plots + `K.npy` + `meta.json` for configured datasets.
* Runs with the shared TOML configs (no hard-coded params).
* Contributes to summary tables and vs-d / delta plots.

---

## 11) After approval

Once the team agrees:

1. We scaffold the repo with the structure above.
2. We add a **concise `README.md`** (how to run, dependencies, seeds, artifacts).
3. We create the initial TOML configs (datasets, feature map, partitions/scales, paths).

---

## Example for a TOML config file

```toml
# example.toml

[run]
dataset = "breast_cancer"
seed_grid = [42]
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
scales = [
  [[0,1],[2,3]],
  [[0,1,2,3]]
]
weights_grid = [[0.5, 0.5]]

[nystrom]
enabled = false
```

---
