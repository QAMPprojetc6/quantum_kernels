# Checkpoint 3
## Scaling the Quantum-Kernel Study to More Datasets (d=4..20)

----

## Project overview

This checkpoint extends the quantum-kernel benchmark suite to **multiple real-world datasets** while keeping the same experimental protocol (baseline, local, and multi-scale kernels) and the same **dimension sweep** (**d = 4, 6, 8, 10, 12, 14, 16, 18, 20**). The goal remains the same: evaluate whether **local** and **multi-scale** kernels mitigate **exponential concentration** as `d` grows, and whether any improvement in kernel geometry translates into better SVM classification performance.

----

## What changed since Checkpoint 2

**Broader dataset coverage:**
- Added and benchmarked: `exam_score_prediction`, `star_classification`, `ionosphere`, `heart_disease`.
- Kept: `breast_cancer`, `parkinsons`.

**Consistent feature dimension across datasets:**
- Each dataset is evaluated at the same set of `d` values (4 to 20) to keep the **number of qubits/features aligned**.
- This is essential for **comparing concentration behavior across datasets**, since kernel concentration depends strongly on dimension.

**Engineered features where needed:**
- For datasets with fewer raw features, **pairwise interaction features** (simple polynomial products) were added to reach the required `d` values.
- This allowed a complete, matched sweep across datasets without changing the feature map or pipeline logic.

**Large datasets handled with subsets and optional Nystrom:**
- For `exam_score_prediction` and `star_classification`, **subset runs were used** because the full datasets are substantially larger.
- A **Nyström / landmark approximation** option was added and tested for scalability, but was **not used in the final results due to time constraints**.

----

## Experimental protocol (kept consistent)

- **Kernel families:** baseline, local (patch-based), multi-scale (local + global mix).
- **Feature map / depth / entanglement:** fixed within each sweep to ensure fair comparisons.
- **Splits and seeds:** deterministic train/val/test splits; multiple seeds per configuration.
- **Metrics:** kernel off-diagonal statistics, effective rank, alignment, and SVM accuracy (precomputed kernels).
- **Preprocessing:** standardization, optional PCA when reducing to smaller `d`, and **angle encoding** (mapping standardized features to rotation angles).

----

## Why the same number of features matters

Kernel concentration is fundamentally tied to **dimension**. If datasets are tested at different `d`, then a "better" or "worse" kernel might simply reflect a different number of qubits rather than a true method advantage. Keeping **the same d-grid across datasets** is essential for:

- fair, apples-to-apples comparisons,
- interpreting concentration vs. dimension curves,
- isolating the effect of **kernel strategy** (baseline vs local vs multiscale).

----

## Subsets for large datasets

Two datasets are much larger than the rest:

- **`exam_score_prediction`**
- **`star_classification`**

For these, **subset configurations** were used (e.g., `*_subset_d*.toml`) to keep total runtime tractable while still preserving the d=4..20 sweep. This is why the figures in this checkpoint reflect **subset results** for those datasets.

----

## Nyström approximation (optional, not used in final plots)

We added a Nyström/landmark approximation to enable approximate kernels on large datasets:

- **What it is:**
  - Approximate the full kernel using a subset of **landmark points**.
  - Reduces kernel construction from $O(n^2)$ to roughly $O(nm)$ with `m` landmarks.

- **Why it was included:**
  - Enables experiments on large datasets where full kernel construction is expensive.

- **Why it was not used in the final results:**
  - While it solved memory pressure, it was still too slow in CPU time for our workflow.
  - Subset-based runs provided comparable insights within reasonable runtime.

Nyström-style methods remain a **standard approach for large datasets** and are supported in the codebase for future scaling.

----

## Results: summary of what the plots show

Across datasets, the same general pattern appears:

- **Baseline kernels concentrate as `d` grows** (off-diagonal p50/p95 tends toward 0).
- **Local kernels consistently reduce concentration** (higher p50/p95; lower effective rank collapse).
- **Multi-scale kernels sit between baseline and local**, as expected from mixing local + global structure.
- **Accuracy changes are dataset-dependent:** reduced concentration does **not always** imply better test accuracy.
- **Delta plots** explicitly show the performance and geometry differences **relative to baseline** at each `d`.
- **Tradeoff plots** visualize the relationship between concentration (off-diag p50) and test accuracy.

----

## Plots vs d (figs/checkpoint3/vs_d)

### breast_cancer
![breast_cancer: concentration p50 vs d](../../figs/checkpoint3/vs_d/breast_cancer_concentration_p50_vs_d.png)
![breast_cancer: concentration p95 vs d](../../figs/checkpoint3/vs_d/breast_cancer_concentration_p95_vs_d.png)
![breast_cancer: effective rank vs d](../../figs/checkpoint3/vs_d/breast_cancer_effrank_vs_d.png)
![breast_cancer: test accuracy vs d](../../figs/checkpoint3/vs_d/breast_cancer_testacc_vs_d.png)

### parkinsons
![parkinsons: concentration p50 vs d](../../figs/checkpoint3/vs_d/parkinsons_concentration_p50_vs_d.png)
![parkinsons: concentration p95 vs d](../../figs/checkpoint3/vs_d/parkinsons_concentration_p95_vs_d.png)
![parkinsons: effective rank vs d](../../figs/checkpoint3/vs_d/parkinsons_effrank_vs_d.png)
![parkinsons: test accuracy vs d](../../figs/checkpoint3/vs_d/parkinsons_testacc_vs_d.png)

### exam_score_prediction (subset)
![exam_score_prediction: concentration p50 vs d](../../figs/checkpoint3/vs_d/exam_score_prediction_concentration_p50_vs_d.png)
![exam_score_prediction: concentration p95 vs d](../../figs/checkpoint3/vs_d/exam_score_prediction_concentration_p95_vs_d.png)
![exam_score_prediction: effective rank vs d](../../figs/checkpoint3/vs_d/exam_score_prediction_effrank_vs_d.png)
![exam_score_prediction: test accuracy vs d](../../figs/checkpoint3/vs_d/exam_score_prediction_testacc_vs_d.png)

### star_classification (subset)
![star_classification: concentration p50 vs d](../../figs/checkpoint3/vs_d/star_classification_concentration_p50_vs_d.png)
![star_classification: concentration p95 vs d](../../figs/checkpoint3/vs_d/star_classification_concentration_p95_vs_d.png)
![star_classification: effective rank vs d](../../figs/checkpoint3/vs_d/star_classification_effrank_vs_d.png)
![star_classification: test accuracy vs d](../../figs/checkpoint3/vs_d/star_classification_testacc_vs_d.png)

### ionosphere
![ionosphere: concentration p50 vs d](../../figs/checkpoint3/vs_d/ionosphere_concentration_p50_vs_d.png)
![ionosphere: concentration p95 vs d](../../figs/checkpoint3/vs_d/ionosphere_concentration_p95_vs_d.png)
![ionosphere: effective rank vs d](../../figs/checkpoint3/vs_d/ionosphere_effrank_vs_d.png)
![ionosphere: test accuracy vs d](../../figs/checkpoint3/vs_d/ionosphere_testacc_vs_d.png)

### heart_disease
![heart_disease: concentration p50 vs d](../../figs/checkpoint3/vs_d/heart_disease_concentration_p50_vs_d.png)
![heart_disease: concentration p95 vs d](../../figs/checkpoint3/vs_d/heart_disease_concentration_p95_vs_d.png)
![heart_disease: effective rank vs d](../../figs/checkpoint3/vs_d/heart_disease_effrank_vs_d.png)
![heart_disease: test accuracy vs d](../../figs/checkpoint3/vs_d/heart_disease_testacc_vs_d.png)

----

## Delta analyses (figs/checkpoint3/deltas)

These figures show **delta values relative to the baseline kernel** for each dataset and d.

### Global summaries

![Mean delta test acc across datasets](../../figs/checkpoint3/deltas/bar_mean_delta_test_acc.png)
![Heatmap: delta test acc (local vs baseline)](../../figs/checkpoint3/deltas/heatmap_delta_test_acc_local.png)
![Heatmap: delta test acc (multiscale vs baseline)](../../figs/checkpoint3/deltas/heatmap_delta_test_acc_multiscale.png)

### breast_cancer
![breast_cancer: delta test acc](../../figs/checkpoint3/deltas/delta_delta_test_acc_breast_cancer.png)
![breast_cancer: delta off-diag p50](../../figs/checkpoint3/deltas/delta_delta_offdiag_p50_breast_cancer.png)
![breast_cancer: delta effective rank](../../figs/checkpoint3/deltas/delta_delta_eff_rank_breast_cancer.png)
![breast_cancer: tradeoff](../../figs/checkpoint3/deltas/tradeoff_breast_cancer.png)

### parkinsons
![parkinsons: delta test acc](../../figs/checkpoint3/deltas/delta_delta_test_acc_parkinsons.png)
![parkinsons: delta off-diag p50](../../figs/checkpoint3/deltas/delta_delta_offdiag_p50_parkinsons.png)
![parkinsons: delta effective rank](../../figs/checkpoint3/deltas/delta_delta_eff_rank_parkinsons.png)
![parkinsons: tradeoff](../../figs/checkpoint3/deltas/tradeoff_parkinsons.png)

### exam_score_prediction (subset)
![exam_score_prediction: delta test acc](../../figs/checkpoint3/deltas/delta_delta_test_acc_exam_score_prediction.png)
![exam_score_prediction: delta off-diag p50](../../figs/checkpoint3/deltas/delta_delta_offdiag_p50_exam_score_prediction.png)
![exam_score_prediction: delta effective rank](../../figs/checkpoint3/deltas/delta_delta_eff_rank_exam_score_prediction.png)
![exam_score_prediction: tradeoff](../../figs/checkpoint3/deltas/tradeoff_exam_score_prediction.png)

### star_classification (subset)
![star_classification: delta test acc](../../figs/checkpoint3/deltas/delta_delta_test_acc_star_classification.png)
![star_classification: delta off-diag p50](../../figs/checkpoint3/deltas/delta_delta_offdiag_p50_star_classification.png)
![star_classification: delta effective rank](../../figs/checkpoint3/deltas/delta_delta_eff_rank_star_classification.png)
![star_classification: tradeoff](../../figs/checkpoint3/deltas/tradeoff_star_classification.png)

### ionosphere
![ionosphere: delta test acc](../../figs/checkpoint3/deltas/delta_delta_test_acc_ionosphere.png)
![ionosphere: delta off-diag p50](../../figs/checkpoint3/deltas/delta_delta_offdiag_p50_ionosphere.png)
![ionosphere: delta effective rank](../../figs/checkpoint3/deltas/delta_delta_eff_rank_ionosphere.png)
![ionosphere: tradeoff](../../figs/checkpoint3/deltas/tradeoff_ionosphere.png)

### heart_disease
![heart_disease: delta test acc](../../figs/checkpoint3/deltas/delta_delta_test_acc_heart_disease.png)
![heart_disease: delta off-diag p50](../../figs/checkpoint3/deltas/delta_delta_offdiag_p50_heart_disease.png)
![heart_disease: delta effective rank](../../figs/checkpoint3/deltas/delta_delta_eff_rank_heart_disease.png)
![heart_disease: tradeoff](../../figs/checkpoint3/deltas/tradeoff_heart_disease.png)

----

## Notes for future work

- Revisit Nyström runs on larger datasets.
- Explore richer feature maps or higher depths for datasets where accuracy remains flat.
- Try structured or overlapping patches for local kernels to see if gains translate to accuracy.
- Continue to report **delta plots** and **vs-d curves** together for interpretability.

----

## Assets

- **Deltas:** `figs/checkpoint3/deltas/`
- **vs d curves:** `figs/checkpoint3/vs_d/`
- **Summary tables:** `outputs/benchmarks/summary_all.csv` and `outputs/benchmarks/summary_all.md`
