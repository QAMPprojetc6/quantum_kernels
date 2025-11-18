"""
Run the multiscale kernel end-to-end:
- load dataset (make_circles | iris)
- make fixed splits
- build multiscale kernel (statevector path)
- save K.npy + meta.json + splits.json
- generate figures (heatmap, offdiag hist, spectrum)
- evaluate SVM with precomputed kernel (write metrics.csv)

Example:
  python scripts/run_multiscale_demo.py \
    --dataset make_circles --n-samples 150 \
    --feature-map zz_qiskit --depth 1 --entanglement linear \
    --out-prefix outputs/ms_circles_zzq

"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import json as _json

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

    X = X.astype(np.float64)
    # Standardize features (recommended)
    X = StandardScaler().fit_transform(X)
    # Optional: map to radians for feature maps (simple choice)
    X = np.pi * X / 2.0
    return X, y.astype(int)


def _make_splits(n: int, seed: int, val_size: float, test_size: float):
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(n, dtype=int)
    idx_train, idx_tmp = train_test_split(idx_all, test_size=(val_size + test_size), random_state=seed, shuffle=True,
                                          stratify=None)
    rel_test = test_size / (val_size + test_size)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=rel_test, random_state=seed, shuffle=True, stratify=None)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["make_circles", "iris"], required=True)
    ap.add_argument("--n-samples", type=int, default=150, help="used only for make_circles")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--scales", type=str, default=None, help='JSON for scales, e.g. [[[0],[1]], [[0,1]]]')
    ap.add_argument("--weights", type=str, default=None, help='JSON list of weights, e.g. [0.5, 0.5]')

    ap.add_argument("--feature-map", default="zz_qiskit", help="zz_qiskit | zz_manual | zz_manual_canonical")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--entanglement", default=None, help="e.g., linear, ring, full (depends on map)")
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")

    ap.add_argument("--out-prefix", required=True, help="prefix for outputs, e.g., outputs/ms_circles_zzq")
    ap.add_argument("--C", nargs="+", type=float, default=[0.1, 1.0, 10.0], help="SVM Cs")

    ap.add_argument("--center", action="store_true", default=False, help="Center the kernel (K <- H K H) before plots/SVM.")
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--report-rank", action="store_true", default=False, help="Print and store kernel spectrum stats (effective rank, min/max eigenvalue, trace).")

    args = ap.parse_args()

    # ---------------- data ----------------
    X, y = _load_dataset(args.dataset, args.n_samples, args.seed)
    n, d = X.shape
    train_idx, val_idx, test_idx = _make_splits(n, seed=args.seed, val_size=args.val_size, test_size=args.test_size)

    # ---------------- kernel ----------------
    scales = None  # default: pairs + all
    weights = None  # default: uniform
    if args.scales is not None:
        scales = _json.loads(args.scales)
    if args.weights is not None:
        weights = _json.loads(args.weights)
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

    # ----- spectrum report (optional) -----
    if args.report_rank:
        stats = _spectrum_stats(K, thresh=1e-6)
        meta["spectrum_stats"] = stats
        print("[SPECTRUM]", stats)

    # ---------------- paths ----------------
    from pathlib import Path
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

    # ---------------- save artifacts ----------------
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
    import json, csv, os  # ensure imports exist
    with open(mpath, "w") as f:
        json.dump(meta_out, f, indent=2)

    with open(spath, "w") as f:
        json.dump({
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "y_all": y.tolist(),
        }, f, indent=2)

    # ---------------- figures ----------------
    plot_heatmap(K, f"{fprefix}_matrix.png")
    plot_offdiag_hist(K, f"{fprefix}_offdiag_hist.png")
    plot_spectrum(K, f"{fprefix}_spectrum.png")

    # ---------------- metrics (SVM precomputed) ----------------
    metrics = eval_precomputed(K, y, train_idx, val_idx, test_idx, args.C)

    # append CSV
    metrics_csv = str(out_dir / "metrics.csv")
    header = ["out_prefix", "dataset", "feature_map", "depth", "best_C", "val_acc", "test_acc"]
    row = [str(out_prefix), args.dataset, args.feature_map, args.depth, metrics["best_C"], metrics["val_acc"],
           metrics["test_acc"]]
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



# examples of use
#
# python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --out-prefix outputs/multiscale/ms_circles_zzq
# # only local (S1)
# python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0],[1]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_circles_local
# # only global (S2)
# python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0,1]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_circles_global
# # local + global (multi-scale)
# python -m scripts.run_multiscale_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --scales '[[[0],[1]], [[0,1]]]' --weights '[0.5, 0.5]' --out-prefix outputs/multiscale/ms_circles_ms2q
#
#
# python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --out-prefix outputs/multiscale/ms_iris_manualcanon
# # only local (S1)
# python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0,1],[2,3]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_iris_local_pairs
# # only global (S2)
# python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0,1,2,3]]]' --weights '[1.0]' --out-prefix outputs/multiscale/ms_iris_global_all
# # local + global (multi-scale)
# python -m scripts.run_multiscale_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --scales '[[[0,1],[2,3]], [[0,1,2,3]]]' --weights '[0.6, 0.4]' --out-prefix outputs/multiscale/ms_iris_ms_pairs_all  --center --report-rank
