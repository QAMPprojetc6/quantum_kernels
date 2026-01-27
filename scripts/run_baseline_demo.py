"""
Run a Baseline (global fidelity) kernel demo end-to-end:
  - load dataset (make_circles | iris)
  - make fixed splits
  - build a baseline kernel K (global fidelity)
  - save artifacts (K.npy, meta.json, splits.json)
  - generate diagnostics plots (heatmap, off-diag histogram, spectrum)
  - optionally normalize (unit diagonal), center K, and report spectrum/effective-rank stats

Run from repo root (recommended):
  python -m scripts.run_baseline_demo ...

CLI:
  python -m scripts.run_baseline_demo \
    --dataset make_circles --n-samples 150 \
    --feature-map zz_qiskit --depth 1 --entanglement linear \
    --out-prefix outputs/baseline/bl_circles_baseline
"""

import argparse
from pathlib import Path
import numpy as np

from qkernels.baseline_kernel import build_kernel
from analysis.eval_svm import eval_precomputed

from scripts.demo_common import (
    load_dataset,
    make_splits,
    normalize_unit_diag,
    center_kernel,
    spectrum_stats,
    prepare_dirs,
    artifact_paths,
    save_json,
    save_splits,
    write_spectrum_txt,
    append_metrics_row,
    plot_all,
)


def main():
    ap = argparse.ArgumentParser(description="Run baseline kernel demo end-to-end.")
    ap.add_argument(
        "--dataset",
        default="make_circles",
        choices=["make_circles", "iris", "star_classification", "exam_score_prediction", "ionosphere"],
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=150,
        help="Used for make_circles, star_classification, exam_score_prediction, and ionosphere",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.2)

    ap.add_argument("--feature-map", default="zz_qiskit")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--entanglement", default=None)

    # Normalize flags (same convention as run_multiscale_demo.py)
    ap.add_argument("--normalize", dest="normalize", action="store_true", default=False, help="Normalize kernel to unit diagonal. Default: off.")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable unit-diagonal normalization (explicitly).")

    ap.add_argument("--center", action="store_true", help="Save/use centered kernel Kc = H K H.")
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--report-rank", action="store_true", help="Report spectrum stats (effective rank, min/max eigenvalue, trace).")

    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs, e.g. outputs/baseline/bl_circles_baseline")
    ap.add_argument("--C", nargs="+", type=float, default=[0.1, 1.0, 10.0], help="SVM C grid")

    args = ap.parse_args()

    # data
    X, y = load_dataset(args.dataset, args.n_samples, args.seed)
    n, d = X.shape
    train_idx, val_idx, test_idx = make_splits(n, seed=args.seed, val_size=args.val_size, test_size=args.test_size)

    # kernel (baseline fidelity)
    kwargs = {}
    if args.entanglement is not None:
        kwargs["entanglement"] = args.entanglement

    K, meta = build_kernel(
        X,
        feature_map=args.feature_map,
        depth=args.depth,
        backend="statevector",
        seed=args.seed,
        **kwargs,
    )

    # optional normalize (script-level, to match CLI semantics)
    if args.normalize:
        K = normalize_unit_diag(K)

    if args.center:
        K = center_kernel(K)
        meta["center"] = True
    else:
        meta["center"] = False

    # spectrum report (optional)
    stats = None
    if args.report_rank:
        stats = spectrum_stats(K, thresh=1e-6)
        meta["spectrum_stats"] = stats
        print("[SPECTRUM]", stats)

    # paths (same naming scheme as run_multiscale_demo.py)
    out_prefix = Path(args.out_prefix)
    out_dir, figs_dir = prepare_dirs(out_prefix, figs_subdir="figs/baseline")
    paths = artifact_paths(out_prefix, figs_dir, centered=bool(args.center))

    if args.report_rank and stats is not None:
        write_spectrum_txt(paths["spectrum_path"], stats)

    # save artifacts
    np.save(paths["kpath"], K)

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
    save_json(paths["mpath"], meta_out)
    save_splits(paths["spath"], train_idx, val_idx, test_idx, y)

    # figures
    plot_all(K, paths["fprefix"])

    # metrics (SVM precomputed) + append CSV (same schema as run_multiscale_demo.py)
    metrics = eval_precomputed(K, y, train_idx, val_idx, test_idx, args.C)

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
    append_metrics_row(metrics_csv, header, row)

    print("[OK] Saved:")
    print("  K:", paths["kpath"])
    print("  meta:", paths["mpath"])
    print("  splits:", paths["spath"])
    print("  figs:", paths["fprefix"] + "_{matrix,offdiag_hist,spectrum}.png")
    print("  metrics row ->", metrics_csv, metrics)


if __name__ == "__main__":
    main()


# ---------------------------
# Examples (one-liners)
# ---------------------------
#
# # Dataset: make_circles
# python -m scripts.run_baseline_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --out-prefix outputs/baseline/bl_circles_baseline
# python -m scripts.run_baseline_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 2 --entanglement linear --out-prefix outputs/baseline/bl_circles_baseline_d2 --report-rank
# python -m scripts.run_baseline_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --out-prefix outputs/baseline/bl_circles_baseline_centered --center --report-rank
#
# # Dataset: iris
# python -m scripts.run_baseline_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --out-prefix outputs/baseline/bl_iris_baseline
# python -m scripts.run_baseline_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --out-prefix outputs/baseline/bl_iris_baseline_centered --center --report-rank
