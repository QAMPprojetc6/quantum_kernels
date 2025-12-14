# Checkpoint 2
## Local and Multi-Scale Strategies to Mitigate Exponential Concentration in Quantum Kernels (#6)

----

## Project overview

Quantum kernels embed classical inputs into quantum states and use state overlap as a similarity score for kernel methods (e.g., SVMs). A major practical issue is **exponential concentration**: as qubit count and/or circuit depth grows, kernel matrices can collapse toward the identity (off-diagonal entries near zero), reducing the kernel’s ability to separate classes.

Our goal is to implement and evaluate two mitigation strategies:

* **Local (patch-wise) kernels:** compute similarity on **subsystems** (patches) via reduced density matrices (RDMs), then aggregate across patches.
* **Multi-Scale kernels:** compute kernels at multiple granularities (local + global), then combine them via a **non-negative weighted mix**.

We aim to deliver reproducible Qiskit implementations, diagnostic plots (histograms/spectra), and benchmarking results on small datasets (starting with `make_circles` and `iris`).

----

## Progress made 

We built an end-to-end, reproducible benchmarking pipeline:

* Implemented **baseline**, **local**, and **multi-scale** kernel modules with a shared "unified API".
* Added scripts to run benchmarks from TOML configs:

  * `scripts/run_experiment.py` (full benchmark runner)
  * `scripts/run_all_benchmarks.py` (runs multiple configs + per-dataset summary + global summary)
* Added diagnostics and evaluation:

  * `analysis/diagnostics.py`: kernel heatmap, off-diagonal histogram, eigen-spectrum
  * `analysis/eval_svm.py`: SVM evaluation with precomputed kernels (C sweep)
  * `analysis/summarize_benchmarks.py`: aggregate results into `summary.csv/.md`
* Ensured cross-platform usability (CLI + IDE) via path normalization and stable output resolution.
* Established “recipes” for quick reproduction and ablation-style comparisons (`docs/RECIPES.md`).

We can now run large sweeps automatically and summarize outcomes for reporting.

----

## Explorations and experiments

We ran systematic sweeps over:

* **Datasets:** `make_circles` (d=2) and `iris` (d=4)
* **Feature maps:** `zz_qiskit` and `zz_manual_canonical`
* **Depth:** 1 and 2 (early stage)
* **Kernel variants:**

  * Baseline (all-qubits overlap)
  * Local-only (1q or 2q patches via RDMs)
  * Multi-Scale (local + baseline mixed; weight grids)

**What worked**

* Local and multi-scale kernels **consistently reduced concentration** in the diagnostics: off-diagonal similarities shifted away from 0, and kernels showed richer structure than the baseline in many settings.
* On `iris` (with `zz_manual_canonical`), **multi-scale improved centered alignment** relative to baseline at depth 1 and remained competitive at depth 2.

**What did not (yet) work**

* On `make_circles`, baseline often remained best in SVM performance and alignment, while local-only kernels sometimes produced overly "flat/high-similarity" kernels (many off-diagonals large), which did not translate into better alignment or SVM results.

----

## Key insights and learnings

### Equivalent comparisons

To make comparisons meaningful, we compare runs that share:

* dataset, feature map, depth, entanglement, backend, centering/normalization settings, and (ideally) the same split seed.

Below are median (and best) diagnostic summaries for representative matched blocks (uncentered kernels). These focus on concentration mitigation **and** “learnability signal” via centered alignment.

---

**Note on centering:** We group results by whether kernels are centered ($K_c = HKH$) because centering changes the interpretation of kernel entries and can introduce negative values. In this report, we present **uncentered** matched comparisons because they are the most interpretable for **concentration diagnostics** (off-diagonals collapsing toward 0). We still evaluate **learning performance** (alignment/SVM) separately under the same centered/uncentered grouping in our benchmark summaries.

### Matched comparisons (uncentered)

### `make_circles` : `zz_qiskit`, depth = 1, entanglement = linear

| Kernel                        | off-diag mean | off-diag p50 | alignment (median) | alignment (best) |
| ----------------------------- | ------------: | -----------: | -----------------: | ---------------: |
| baseline                      |         0.250 |        0.191 |              0.081 |            0.107 |
| local (1q)                    |         0.681 |        0.692 |              0.022 |            0.042 |
| multiscale (local1q+baseline) |         0.428 |        0.395 |              0.065 |            0.100 |

**Reading:** Local and Multi-Scale clearly push off-diagonals away from zero (less concentration). However, baseline still has stronger median/best alignment than local-only, and multi-scale is close but not consistently above baseline.

---

### `make_circles` : `zz_qiskit`, depth = 2, entanglement = linear

| Kernel                        | off-diag mean | off-diag p50 | alignment (median) | alignment (best) |
| ----------------------------- | ------------: | -----------: | -----------------: | ---------------: |
| baseline                      |         0.306 |        0.264 |              0.082 |            0.109 |
| local (1q)                    |         0.617 |        0.624 |              0.013 |            0.021 |
| multiscale (local1q+baseline) |         0.449 |        0.419 |              0.057 |            0.099 |

**Reading:** Same pattern: reduced concentration for local/multi-scale, but baseline remains strongest in alignment; local-only is worst.

---

### `iris` : `zz_manual_canonical`, depth = 1, entanglement = ring

| Kernel                        | off-diag mean | off-diag p50 | alignment (median) | alignment (best) |
| ----------------------------- | ------------: | -----------: | -----------------: | ---------------: |
| baseline                      |         0.116 |        0.057 |              0.391 |            0.391 |
| local (1q)                    |         0.641 |        0.641 |              0.414 |            0.444 |
| multiscale (local1q+baseline) |         0.283 |        0.252 |              0.414 |        **0.456** |

**Reading:** Baseline shows much smaller off-diagonals (more concentrated). Local and Multi-Scale increase off-diagonal structure. Importantly, **Multi-Scale improves best alignment** beyond baseline, suggesting that reduced concentration can translate into a more useful kernel for learning on Iris.

---

### `iris` : `zz_manual_canonical`, depth = 2, entanglement = ring

| Kernel                        | off-diag mean | off-diag p50 | alignment (median) | alignment (best) |
| ----------------------------- | ------------: | -----------: | -----------------: | ---------------: |
| baseline                      |         0.078 |        0.047 |              0.169 |            0.169 |
| local (1q)                    |         0.643 |        0.646 |              0.147 |            0.157 |
| multiscale (local1q+baseline) |         0.250 |        0.240 |              0.172 |        **0.179** |

**Reading:** Multi-Scale remains slightly better than baseline in alignment; local-only degrades. This hints that mixing local and global information can be more stable than local-only as depth increases.

----

### Technical challenges

* **Path/cwd differences (CLI vs IDE):** running from IDE used a different working directory and initially wrote outputs under `scripts/`. We fixed this by resolving config/output paths relative to the repo root.
* **Windows encoding issue (CSV):** writing Greek symbols (e.g., `λ_min`) failed on cp1252. We fixed CSV writing by explicitly using UTF-8.
* **Metrics join mismatch:** summaries initially showed many empty fields because `metrics.csv` used Windows path formatting while artifact-derived paths used POSIX formatting. We fixed the join by normalizing run identifiers (stable join keys).
* **Kernel comparability:** Centering ($K_c = HKH$) changes the meaning of kernel entries (and may yield negative values), so we report diagnostics and performance **separately for centered vs uncentered** runs. In the main text we show uncentered tables for concentration; centered results are tracked in the benchmark summaries for learning metrics.
---

### Current status

* All three kernel families (baseline/local/multi-scale) run end-to-end via TOML-configured benchmarks.
* We can generate diagnostics, SVM metrics, per-dataset summaries, and a global summary across all runs.
* Empirically:

  * **Concentration mitigation** is visible in local and multi-scale kernels (especially in off-diagonal distributions).
  * **Learning signal improvements** appear most clearly in `iris` via multi-scale alignment gains.
  * On `make_circles`, baseline is still strongest in many sweeps; multi-scale is competitive but not consistently better yet.


### Path forward

1. **More ablations:** for each dataset and fixed feature map/depth, run:
   * baseline-only, local-only, multi-scale (local+baseline), with fixed splits/seeds.
2. **Weight tuning strategy:** select multi-scale weights using validation (or alignment proxy) rather than reporting raw grid extremes; report test once per selected model.
3. **Patch design improvements:** for Iris and upcoming higher-d feature sets, evaluate patch sizes (1q vs 2q vs 3q) and structured partitions (pairs/triplets) to avoid overly uniform "local-only" kernels.
4. **Scale up beyond toy:** add one additional small real-world tabular dataset with 20–30 features (as suggested in the program guide) to probe concentration more realistically.
5. **Robustness checks (lightweight):** optional input noise or shallow sampling noise to test stability trends.
6. **Writing cadence:** maintain a running methods/results log; for the MVP report, include:

   * one clear dataset where multi-scale or local is better (accuracy and/or robustness),
   * diagnostics (histogram + spectrum) that explain *why*.

**MVP target:** demonstrate at least one dataset where Local or Multi-Scale kernels show a clear advantage (performance or robustness) over the baseline, supported by diagnostic evidence of reduced concentration and improved kernel structure.