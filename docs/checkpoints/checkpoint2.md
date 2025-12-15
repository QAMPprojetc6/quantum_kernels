# Checkpoint 2
## Local and Multi-Scale Strategies to Mitigate Exponential Concentration in Quantum Kernels (#6)

----

## Project overview

Quantum kernels embed classical inputs into quantum states and use **state overlap** as a similarity score for kernel methods (e.g., SVMs). A key limitation is **exponential concentration**: as qubit count and/or circuit depth grows, kernel matrices can drift toward the identity (off-diagonals near zero), reducing separability.

We implement and benchmark two mitigation strategies:

* **Local (patch-wise) kernels:** compute similarity on **subsystems** (patches) via reduced density matrices (RDMs), then aggregate across patches.
* **Multi-Scale kernels:** compute kernels at multiple granularities (local + global) and combine them via a **non-negative weighted mix**.

----

## Progress made 

We built a reproducible, end-to-end benchmarking pipeline:

* Implemented **baseline**, **local**, and **multi-scale** kernels under a shared unified API.
* TOML-driven runner: `scripts/run_experiment.py` + multi-config orchestrator: `scripts/run_all_benchmarks.py`.
* Diagnostics + evaluation:

  * `analysis/diagnostics.py`: heatmap, off-diagonal histogram, eigen-spectrum
  * `analysis/eval_svm.py`: SVM with precomputed kernels (C sweep)
  * `analysis/summarize_benchmarks.py`: aggregates runs into per-dataset and global summaries
* "Recipes" for quick reproducibility/ablations: `docs/RECIPES.md`.
* Cross-platform stability (CLI + IDE) with consistent run IDs for joining artifacts + metrics.

----

## Explorations and experiments

We ran systematic sweeps on 4 datasets:

* `make_circles` (d=2), `iris` (d=4), `breast_cancer` (PCA to d=8), `parkinsons` (PCA to d=8)
* Feature maps: `zz_qiskit`, `zz_manual_canonical`; depth: 1-2
* Kernels: baseline, local-only, multi-scale (local + baseline; weight grids)


**Overall pattern so far:** local/multi-scale reduce concentration (off-diagonals shift away from ~0), but accuracy gains are dataset-dependent; weight grids can overfit validation.


----

## Key insights and learnings

### Equivalent comparisons

We only compare runs that match **dataset, feature map, depth, entanglement, backend, and preprocessing (normalize/center)**.

**Centering:** we report uncentered results for concentration interpretability; learning metrics are compared within the same centered/uncentered grouping.

---

### Best matched results per dataset (uncentered)

### `make_circles (n=150, d=2)` : `zz_qiskit`, depth=2, ent=linear, centered=False

| Kernel     | Scales / Patches |    Weights | OffDiag μ±σ | EffRank | Align |   Val |  Test |
| ---------- | ---------------- |-----------:| ----------: | ------: | ----: | ----: | ----: |
| baseline   | all qubits       |          - | 0.306±0.227 |    10.1 | 0.109 | 0.800 | 0.600 |
| local      | 1q×2             |          - | 0.616±0.177 |     3.8 | 0.015 | 0.633 | 0.433 |
| multiscale | 1q×2 + 2q×1      | [0.8, 0.2] | 0.547±0.184 |     5.3 | 0.049 | 0.800 | 0.667 |

**Reading:** Multi-scale improves **test** vs baseline (0.667 vs 0.600); local-only becomes too low-rank and underperforms.

---

### `iris (n=150, d=4)` : `zz_manual_canonical`, depth=1, ent=ring, centered=False

| Kernel     | Scales / Patches |    Weights | OffDiag μ±σ | EffRank | Align |   Val |  Test |
| ---------- | ---------------- |-----------:| ----------: | ------: | ----: | ----: | ----: |
| baseline   | all qubits       |          - | 0.116±0.162 |    40.8 | 0.391 | 0.933 | 0.933 |
| local      | 1q×4             |          - | 0.803±0.119 |     2.3 | 0.444 | 0.867 | 0.700 |
| multiscale | 1q×4 + 4q×1      | [0.8, 0.2] | 0.602±0.104 |     7.6 | 0.456 | 0.967 | 0.833 |

**Reading:** Baseline is best on **test**; multi-scale wins on **val** (possible weight overfitting), while local-only is overly uniform/low-rank.

---

### `breast_cancer (n=569, d=8)` : `zz_manual_canonical`, depth=1, ent=ring, centered=False

| Kernel     | Scales / Patches |    Weights | OffDiag μ±σ | EffRank | Align |   Val |  Test |
| ---------- | ---------------- |-----------:| ----------: | ------: | ----: | ----: | ----: |
| baseline   | all qubits       |          - | 0.012±0.030 |   475.8 | 0.073 | 0.781 | 0.675 |
| local      | 2q×4             |          - | 0.312±0.091 |    75.0 | 0.090 | 0.754 | 0.728 |
| multiscale | 2q×4 + 8q×1      | [0.5, 0.5] | 0.193±0.060 |   184.7 | 0.094 | 0.807 | 0.719 |

**Reading:** Baseline shows the strongest concentration signature (off-diags ~0); local-only gives best **test** (0.728), multi-scale is competitive.

![Breast cancer: Baseline vs Local vs Multi-Scale diagnostics](figs/checkpoint2/breast_cancer_compare.png)
*Breast Cancer (8 qubits): local and multi-scale reduce concentration vs baseline (off-diagonal mass shifts away from 0) and change the spectrum, with competitive SVM performance.*

> *Takeaway*: Local/multi-scale reduce concentration and are competitive; local improves test accuracy vs baseline.

---

### `parkinsons (n=195, d=8)` : `zz_qiskit`, depth=1, ent=linear, centered=False

| Kernel     | Scales / Patches |    Weights | OffDiag μ±σ | EffRank | Align |   Val |  Test |
| ---------- | ---------------- |-----------:| ----------: | ------: | ----: | ----: | ----: |
| baseline   | all qubits       |          - | 0.004±0.008 |   193.6 | 0.072 | 0.769 | 0.795 |
| local      | 2q×4             |          - | 0.250±0.064 |    63.2 | 0.046 | 0.769 | 0.795 |
| multiscale | 2q×4 + 8q×1      | [0.0, 1.0] | 0.004±0.008 |   193.6 | 0.072 | 0.769 | 0.795 |

**Reading:** No gains in this sweep; best multi-scale collapses to baseline-only, suggests we need different partitions/feature maps/depth or noise model.

![Parkinsons: Baseline vs Local vs Multi-Scale diagnostics](figs/checkpoint2/parkinsons_compare.png)
*Parkinsons (8 qubits): local/multi-scale strongly reduce concentration vs baseline, but this sweep shows no SVM gain, suggesting partitions/feature map/depth need retuning.*

> *Takeaway*: Local/multi-scale reduce concentration, but no SVM gain yet, partitions/feature map/depth need retuning.

----

### Technical challenges

* **Reproducible runs across setups:** CLI vs IDE changed the working directory and broke relative paths. We fixed this by resolving configs/outputs relative to the repo root.
* **Reliable aggregation at scale:** Large sweeps (seeds × depths × weights) exposed fragile joins between artifacts and `metrics.csv`. We introduced **stable run IDs** and normalized paths so summaries are complete.
* **Platform quirks:** Windows default encoding broke CSV export when using symbols (e.g., λ). We standardized all summaries to **UTF-8**.
* **Numerical stability:** Centering and floating-point noise can yield small negative eigenvalues. We standardized symmetrization and allow light diagonal regularization for stable SVM training.


----

### Current status

* All three kernel families (baseline/local/multi-scale) run end-to-end via TOML configs on **4 datasets**, producing diagnostics, SVM metrics, per-dataset summaries, and a global summary.
* Empirically, **concentration mitigation is visible** (off-diagonal statistics shift away from ~0), but **accuracy benefits are not universal**: strongest improvements so far appear on `breast_cancer` (local/multi-scale competitive), while `parkinsons` shows no improvement yet.

----

### Path forward


1. **Tight ablations per dataset:** baseline-only vs local-only vs multi-scale under fixed splits/seeds, then report a single selected model per setting.
2. **Weight selection protocol:** choose multi-scale weights via validation (and confirm once on test) to reduce "grid extreme" reporting.
3. **Partition design:** try structured patches (1q/2q/3q, overlapping vs disjoint) to avoid overly uniform local kernels.
4. **Probe concentration more directly:** increase qubits/features gradually (e.g., d=8 -> 12 where feasible) and test depth sensitivity.
5. **Robustness check:** light input noise or sampling noise to test stability trends.

**MVP target:** one dataset where local or multi-scale shows a clear advantage (accuracy and/or robustness), supported by diagnostics explaining reduced concentration and improved kernel structure.
