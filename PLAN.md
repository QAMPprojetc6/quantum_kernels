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
* **Minimal experiment pipeline** on small datasets (one synthetic + one small real/tabular).
* **Diagnostics & plots:** kernel heatmaps, off-diagonal histograms, eigen-spectra, centered alignment vs. labels.
* **Reproducible artifacts:** `K.npy`, splits, metrics, and figures stored with consistent names.

Non-goals: large-scale hardware runs, asymptotic theory.

---

## 2) Async team split (1 person = 1 kernel)

* **Global (baseline) - Sarvagya Kaushik**

  * Implements the fidelity kernel with a shared feature map.
  * Produces baseline plots and metrics.

* **Local (patch-wise) - Debashis Saikia**

  * Implements per-patch kernels (subcircuits or RDM).
  * Aggregates patches (mean or weighted).

* **Multi-Scale - Claudia Zendejas-Morales**

  * Implements cross-scale combination with simple weights (equal -> tuned later).
  * Runs ablations (drop one scale).

Each person works independently but adheres to the same **interfaces and artifact formats** below.

---

## 3) Common contracts (to avoid integration pain)

### 3.1 Unified API (same signature across kernels)

```python
def build_kernel(
    X,                      # np.ndarray (n_samples, d)
    feature_map="zz",       # shared across kernels
    depth=1,                # {1, 2} to start
    backend="statevector",  # "statevector" | "sampling"
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

* **Global:** (optional) `shots`, `noise_model`.
* **Local:** `partitions=[(0,1),(2,3)]`, `method="subcircuits"|"rdm"`, `agg="mean"`, `weights=None`.
* **Multi-Scale:** `scales=[[(0,1),(2,3)],[(0,1,2,3)]]`, `weights=[0.5,0.5]`.

We share **one** `feature_maps.py` to avoid duplication.

### 3.2 Artifacts (names & formats)

* Kernel matrix: `outputs/K_{kernel}-{dataset}_{seed}.npy`
* Metadata: `outputs/meta_{kernel}-{dataset}_{seed}.json`
* Splits (shared by all): `outputs/splits_{dataset}_{seed}.json`
* Plots:
  * `figs/{kernel}-{dataset}_{seed}_matrix.png`
  * `figs/{kernel}-{dataset}_{seed}_offdiag_hist.png`
  * `figs/{kernel}-{dataset}_{seed}_spectrum.png`

### 3.3 Quality checklist (each person runs locally)

* Shape: `K.shape == (n, n)`; diagonals reasonable (e.g., ≈1 for fidelity-style).
* Symmetry: `||K - K.T||_∞ < 1e-8`.
* PSD (Positive Semi-Definite) (numerical): `min(eigvals(K)) > -1e-8` (else regularize `K += 1e-6 I`).
* Determinism: fixed `seed` -> same `K` and plots.
* Timing: log wall-time for `n≈100` in `meta.json`.

---

## 4) Shared configuration (single `config.toml`)

One TOML file controls datasets, seeds, feature map, partitions/scales, and paths.
Everyone reads from the same file to keep experiments comparable.

Note: TOML = Tom's Obvious, Minimal Language. Easy to read and write file, by being minimal and by using human-readable syntax.

---

## 5) Datasets, parameters, and baselines (common to all)

* **Datasets:** start with `make_circles` (binary) and **Iris** (3-class).
* **Splits & seeds:** 60/20/20; `seed = 42` (splits saved once and reused).
* **Feature map:** `zz`; `depth ∈ {1, 2}` (start with `1`).
* **SVM (precomputed kernel):** `C ∈ {0.1, 1, 10}`.
* **Local default:** contiguous pairs for partitions.
* **Multi-Scale default:** `scales = [pairs, all]`, `weights = [0.5, 0.5]`.

---

## 6) Minimal evaluation & diagnostics

For each kernel & dataset:

* **Metrics:** Accuracy, and MCC (Matthews Correlation Coefficient).
* **Plots:** kernel heatmap; off-diagonal histogram; eigen-spectrum.
* **Optional:** centered alignment (K vs. labels); light noise robustness.

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
  outputs/                 # K.npy, splits, meta, metrics
  figs/                    # figures from diagnostics
  config.toml              # single source of truth for runs
  README.md
  PLAN.md                  # (this document)
```

---

## 9) Risks & simple mitigations

* **Incomparable results:** single `config.toml`, shared splits/seed, same feature map/depth.
* **Non-PSD kernels (numerical):** add `+1e-6 I`, use float64, center if needed.
* **Slow runs:** keep depth small (1–2), cache circuits, use small `n` first.
* **Code divergence:** one shared `feature_maps.py`; identical API across kernels.

---

## 10) Acceptance criteria (per kernel)

* `build_kernel(...)` implemented with the common signature.
* Passes the **quality checklist** above.
* Produces the 3 plots + `K.npy` + `meta.json` for both datasets.
* Runs with the shared `config.toml` (no hard-coded params).

---

## 11) After approval

Once the team agrees:

1. We scaffold the repo with the structure above.
2. We add a **concise `README.md`** (how to run, dependencies, seeds, artifacts).
3. We create the initial **`config.toml`** (datasets, feature map, partitions/scales, paths).

---

## Example for the `config.toml` file

```toml
# example.toml

[run]
seed = 42
dataset = "make_circles"   # then change to other, e.g. "iris"
n_samples = 150
test_size = 0.2
val_size  = 0.2

[feature_map]
name = "zz"
depth = 1
backend = "statevector"    # or "sampling"
shots = 0                  # 0 = statevector; >0 = sampling

[baseline_kernel]
enabled = true

[local_kernel]
enabled = true
partitions = [[0,1],[2,3]]   # pairs of qubits
method = "subcircuits"       # or "rdm"
agg = "mean"                 # or "weighted"
weights = []                 # empty = equal weights

[multiscale_kernel]
enabled = true
scales = [
  [[0,1],[2,3]],             # scale S1: pairs
  [[0,1,2,3]]                # scale S2: pairs
]
weights = [0.5, 0.5]         # mix between scales

[svm]
C = [0.1, 1.0, 10.0]
kernel = "precomputed"       # we use K that we generate

[paths]
outputs = "outputs"
figs    = "figs"
```

---
