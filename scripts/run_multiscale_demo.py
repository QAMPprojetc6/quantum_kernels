"""
Run a Multi-Scale kernel demo end-to-end:
  - load dataset (make_circles | iris)
  - make fixed splits
  - build a multi-scale kernel K (optionally with custom scales/weights)
  - save artifacts (K.npy, meta.json, splits.json)
  - generate diagnostics plots (heatmap, off-diag histogram, spectrum)
  - optionally center K and report spectrum/effective-rank stats

This script can reproduce:
  - Baseline (all-qubits only): a single scale with one patch containing all qubits.
  - Local-only (1q patches): a single scale consisting of single-qubit patches.
  - Multi-Scale (Local + Baseline): mix local (1q) and all-qubits scales with non-negative weights.

Run from repo root (recommended):
  python -m scripts.run_multiscale_demo ...

CLI:
  python scripts/run_multiscale_demo.py \
    --dataset make_circles --n-samples 150 \
    --feature-map zz_qiskit --depth 1 --entanglement linear \
    --out-prefix outputs/ms_circles_zzq

Notes on --scales format:
- --scales is JSON: a list of "scales"
- each scale is a list of "patches"
- each patch is a list of qubit indices
Example (d=2):
  --scales '[[[0],[1]], [[0,1]]]'
  -> scale 1: patches [0], [1]
  -> scale 2: patch [0,1]
"""

import argparse
import csv
import os
import json
from pathlib import Path
import numpy as np

from qkernels.multiscale_kernel import build_kernel
from analysis.diagnostics import plot_heatmap, plot_offdiag_hist, plot_spectrum
from analysis.eval_svm import eval_precomputed


# ---------------------------
# data helpers
# ---------------------------
def _load_dataset(name: str, n_samples: int, seed: int):
    from sklearn.datasets import make_circles, load_iris
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    if name == "make_circles":
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.1, random_state=seed)
    elif name == "iris":
        iris = load_iris()
        X, y = iris.data, iris.target
    else:
        raise ValueError("dataset must be 'make_circles' or 'iris'.")

    # Shuffle (iris comes ordered)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X = X.astype(np.float64)
    # Standardize features
    X = StandardScaler().fit_transform(X)
    # Optional: map to radians for feature maps (simple choice)
    X = np.pi * X / 2.0
    return X, y.astype(int)


def _make_splits(n: int, seed: int, val_size: float, test_size: float):
    # We keep splits deterministic via the seed and store them alongside outputs
    # so that baseline/local/multi-scale comparisons use identical indices.

    from sklearn.model_selection import train_test_split
    idx_all = np.arange(n, dtype=int)
    idx_train, idx_tmp = train_test_split(
        idx_all,
        test_size=(val_size + test_size),
        random_state=seed,
        shuffle=True,
        stratify=None,
    )
    rel_test = test_size / (val_size + test_size)
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=rel_test,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center the kernel: Kc = H K H."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    # enforce exact symmetry
    Kc = 0.5 * (Kc + Kc.T)
    return Kc

def _spectrum_stats(K: np.ndarray, thresh: float = 1e-6) -> dict:
    """Return basic spectrum stats for symmetric K."""
    Ks = 0.5 * (K + K.T)
    w = np.linalg.eigvalsh(Ks)
    eff_rank = int(np.sum(w > thresh))
    return {
        "n": int(K.shape[0]),
        "trace": float(np.sum(w)),
        "lambda_min": float(w.min()),
        "lambda_max": float(w.max()),
        "effective_rank@{:.0e}".format(thresh): eff_rank,
        "threshold": float(thresh),
    }


# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Run multiscale kernel demo end-to-end.")
    ap.add_argument("--dataset", default="make_circles", choices=["make_circles", "iris"])
    ap.add_argument("--n-samples", type=int, default=150, help="Only used for make_circles")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.2)

    ap.add_argument("--feature-map", default="zz_qiskit")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--entanglement", default=None)

    ap.add_argument("--scales", default=None, help="JSON string: e.g. '[[[0],[1]], [[0,1]]]'")
    ap.add_argument("--weights", default=None, help="JSON string: e.g. '[0.5, 0.5]'")

    # Normalize flags (keep default=False to preserve current behavior)
    ap.add_argument("--normalize", dest="normalize", action="store_true", default=False, help="Normalize kernel to unit diagonal (recommended for RDM/HS). Default: off.")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable unit-diagonal normalization (explicitly).")
    ap.add_argument("--center", action="store_true", help="Save/use centered kernel Kc = H K H.")
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--report-rank", action="store_true", help="Report spectrum stats (effective rank, min/max eigenvalue, trace).")

    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs, e.g. outputs/multiscale/ms_circles_ms")
    ap.add_argument("--C", nargs="+", type=float, default=[0.1, 1.0, 10.0], help="SVM C grid")

    args = ap.parse_args()

    # data
    X, y = _load_dataset(args.dataset, args.n_samples, args.seed)
    n, d = X.shape
    train_idx, val_idx, test_idx = _make_splits(n, seed=args.seed, val_size=args.val_size, test_size=args.test_size)

    # kernel
    scales = None  # default: pairs + all (inside multiscale kernel)
    weights = None  # default: uniform
    if args.scales is not None:
        scales = json.loads(args.scales)
    if args.weights is not None:
        weights = json.loads(args.weights)
    kwargs = {}
    if args.entanglement is not None:
        kwargs["entanglement"] = args.entanglement

    K, meta = build_kernel(
        X,
        feature_map=args.feature_map,
        depth=args.depth,
        backend="statevector",
        seed=args.seed,
        scales=scales,
        weights=weights,
        normalize=args.normalize,
        **kwargs,
    )

    if args.center:
        K = _center_kernel(K)
        meta["center"] = True
    else:
        meta["center"] = False

    # spectrum report (optional)
    if args.report_rank:
        stats = _spectrum_stats(K, thresh=1e-6)
        meta["spectrum_stats"] = stats
        print("[SPECTRUM]", stats)

    # paths
    out_prefix = Path(args.out_prefix)
    out_dir = out_prefix.parent if str(out_prefix.parent) != "" else Path(".")
    figs_dir = Path("figs/multiscale")
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    if args.report_rank:
        with open(str(out_prefix) + "_spectrum.txt", "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

    suffix = "_centered" if args.center else ""
    kpath = str(out_prefix) + f"_K{suffix}.npy"
    mpath = str(out_prefix) + "_meta.json"
    spath = str(out_prefix) + "_splits.json"
    fprefix = str(figs_dir / (out_prefix.name + suffix))

    # save artifacts
    np.save(kpath, K)
    meta_out = dict(meta)
    meta_out.update({
        "dataset": args.dataset,
        "seed": args.seed,
        "depth": args.depth,
        "feature_map": args.feature_map,
        "entanglement": args.entanglement,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "normalize": args.normalize,
    })

    with open(mpath, "w") as f:
        json.dump(meta_out, f, indent=2)

    with open(spath, "w") as f:
        json.dump({
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "y_all": y.tolist(),
        }, f, indent=2)

    # figures
    plot_heatmap(K, f"{fprefix}_matrix.png")
    plot_offdiag_hist(K, f"{fprefix}_offdiag_hist.png")
    plot_spectrum(K, f"{fprefix}_spectrum.png")

    # metrics (SVM precomputed)
    metrics = eval_precomputed(K, y, train_idx, val_idx, test_idx, args.C)

    # append CSV
    metrics_csv = str(out_dir / "metrics.csv")
    header = ["out_prefix", "dataset", "feature_map", "depth", "best_C", "val_acc", "test_acc"]
    row = [
        str(out_prefix),
        args.dataset,
        args.feature_map,
        args.depth,
        metrics["best_C"],
        metrics["val_acc"],
        metrics["test_acc"],
    ]

    write_header = not os.path.exists(metrics_csv)
    with open(metrics_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print("[OK] Saved:")
    print("  K:", kpath)
    print("  meta:", mpath)
    print("  splits:", spath)
    print("  figs:", fprefix + "_{matrix,offdiag_hist,spectrum}.png")
    print("  metrics row ->", metrics_csv, metrics)


if __name__ == "__main__":
    main()



# ---------------------------
# Examples (one-liners)
# ---------------------------
#
#
# # Dataset: make_circles (d=2)
#
# Baseline (all-qubits only)
# python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0,1]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_circles_baseline
#
# Local-only (1q patches)
# python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0],[1]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_circles_local1q
#
# Multi-Scale (Local 1q + Baseline all-qubits)
# python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0],[1]], [[0,1]]]' --weights '[0.5, 0.5]' --out-prefix outputs/multiscale/ms_circles_ms_local1q_baseline
#
# Multi-Scale default (pairs + all)
# python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --out-prefix outputs/multiscale/ms_circles_default_pairs_all
# uses scales: '[[[0,1]], [[0,1]]]'; uniform weights
#
#
# # Dataset: iris (d=4)
#
# Baseline (all-qubits only)
# python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0,1,2,3]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_iris_baseline
#
# Local-only (1q patches)
# python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0],[1],[2],[3]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_iris_local1q
# Local-only (2q patches)
# python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0,1],[2,3]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_iris_local2q
#
# Multi-Scale (Local 1q + Baseline all-qubits)
# python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0],[1],[2],[3]], [[0,1,2,3]]]' --weights '[0.5, 0.5]' --out-prefix outputs/multiscale/ms_iris_ms_local1q_baseline --center --report-rank
# Multi-Scale (Local 2q + Baseline all-qubits)
# python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0,1],[2,3]], [[0,1,2,3]]]' --weights '[0.6, 0.4]' --out-prefix outputs/multiscale/ms_iris_ms_local2q_baseline  --center --report-rank
#
# Multi-Scale default (pairs + all)
# python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --out-prefix outputs/multiscale/ms_iris_default_pairs_all
# uses scales: '[[[0,1],[2,3]], [[0,1,2,3]]]'; uniform weights

