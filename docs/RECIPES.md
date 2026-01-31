# Recipes (current code)

Run all commands from the **repo root**.  
PowerShell JSON quoting tip: use single quotes around JSON (`'[[0,1],[2,3]]'`). CMD prefers double quotes.

---

## 1) Demo scripts (single-run, local outputs)

These scripts produce `K.npy`, `meta.json`, `splits.json`, figures, and a `metrics.csv` row.

### Baseline (global fidelity)
```bash
python -m scripts.run_baseline_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --out-prefix outputs/baseline/bl_circles_baseline
python -m scripts.run_baseline_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --out-prefix outputs/baseline/bl_iris_baseline_centered --center --report-rank
```

### Local (patch-wise)
```bash
python -m scripts.run_local_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --partitions '[[0],[1]]' --method rdm --agg mean --out-prefix outputs/local/loc_circles_local1q
python -m scripts.run_local_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --partitions '[[0,1],[2,3]]' --method rdm --agg mean --out-prefix outputs/local/loc_iris_local2q --center --report-rank
```

### Multi-scale (mix local + global)
```bash
python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0],[1]], [[0,1]]]' --weights '[0.5, 0.5]' --out-prefix outputs/multiscale/ms_circles_ms_local1q_baseline
python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --out-prefix outputs/multiscale/ms_iris_default_pairs_all
```

Notes:
- Demo scripts accept datasets: `make_circles`, `iris`, `star_classification`, `exam_score_prediction`, `ionosphere`, `heart_disease`.
- Optional flags: `--normalize`, `--center`, `--report-rank`, `--rdm-metric hs`.

---

## 2) Benchmark pipeline (TOML-driven sweeps)

### Run a single config
```bash
python -m scripts.run_experiment --config configs/breast_cancer_d8.toml
```

### Run multiple configs + summarize automatically
```bash
python -m scripts.run_all_benchmarks --configs configs/breast_cancer_d4.toml configs/breast_cancer_d6.toml
```

### Nyström (for large datasets)
Enable in your TOML:
```toml
[nystrom]
enabled = true
datasets = ["star_classification", "exam_score_prediction"]
n_landmarks = 1000
diag_samples = 2000
chunk_size = 256
```

---

## 3) Summaries and plots

### Summaries (CSV/MD)
```bash
python -m analysis.summarize_benchmarks --root outputs/benchmarks/breast_cancer_d8 --out outputs/benchmarks/breast_cancer_d8/summary.csv --md outputs/benchmarks/breast_cancer_d8/summary.md
python -m analysis.summarize_benchmarks --roots outputs/benchmarks/breast_cancer_d4 outputs/benchmarks/breast_cancer_d6 --out outputs/benchmarks/summary_all.csv --md outputs/benchmarks/summary_all.md --include-paths
```

### vs-d plots
```bash
python -m analysis.plot_vs_d --summary outputs/benchmarks/summary_all.md --out figs/checkpoint3/vs_d --also-p95
```

### Best C vs d (baseline/local/multiscale in one plot)
```bash
python -m analysis.plot_c_vs_d --summary outputs/benchmarks/summary_all.csv --out figs/checkpoint3/c_vs_d
```

### Spectra comparison (baseline/local/multiscale, mean±std across seeds)
```bash
python -m analysis.plot_spectra_compare --summary outputs/benchmarks/summary_all.csv --out figs/checkpoint3/spectra --d 12 --datasets breast_cancer parkinsons exam_score_prediction star_classification ionosphere heart_disease
```
Optional flags:
- `--normalize-trace` (compare shapes, not scale)
- `--skip-first N` (drop the top N eigenvalues before aggregation)
- `--log-scale` (semilogy for wide dynamic range)

### Delta analyses (vs baseline)
```bash
python -m analysis.plot_deltas --summary outputs/benchmarks/summary_all.csv --out figs/checkpoint3/deltas
```

### Checkpoint 2 comparison figure
```bash
python -m analysis.make_checkpoint2_figure --baseline <baseline_K.npy> --local <local_K.npy> --multiscale <multiscale_K.npy> --out figs/checkpoint2/breast_cancer_compare.png
```

---

## 4) Diagnostics / evaluation utilities

### Diagnostics from an existing kernel
```bash
python -m analysis.diagnostics --kernel outputs/benchmarks/breast_cancer_d8/<case>_K.npy --save-prefix figs/diagnostics/breast_cancer_d8_case
```

### Evaluate SVM from a saved kernel
```bash
python -m analysis.eval_svm --kernel outputs/benchmarks/breast_cancer_d8/<case>_K.npy --splits outputs/benchmarks/breast_cancer_d8/<case>_splits.json --C 0.1 1 10 --out outputs/benchmarks/breast_cancer_d8/metrics.csv
```

---

## 5) Helpers

### Check missing seed coverage for delta CIs
```bash
python -m scripts.check_missing_seed_deltas --summary outputs/benchmarks/summary_all.csv
```

### Merge heart-disease sites (data prep)
```bash
python scripts/merge_heart_disease_sites.py
```
This script is optional and requires `pandas`.
