"""
Run a Local (patch-wise) kernel demo end-to-end:
  - load dataset (make_circles | iris)
  - make fixed splits
  - build a local kernel K (RDM or subcircuits)
  - save artifacts (K.npy, meta.json, splits.json)
  - generate diagnostics plots (heatmap, off-diag histogram, spectrum)
  - optionally normalize (unit diagonal), center K, and report spectrum/effective-rank stats

Run from repo root (recommended):
  python -m scripts.run_local_demo ...

Notes on --partitions format:
- --partitions is JSON: a list of "patches"
- each patch is a list of qubit indices
Examples:
  d=2:  --partitions '[[0],[1]]'     (1q patches)
  d=4:  --partitions '[[0,1],[2,3]]' (pairs)
"""

import argparse
import json
from pathlib import Path
import numpy as np

from qkernels.local_kernel import build_kernel
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
    ap = argparse.ArgumentParser(description="Run local kernel demo end-to-end.")
    ap.add_argument(
        "--dataset",
        default="make_circles",
        choices=["make_circles", "iris", "star_classification", "exam_score_prediction"],
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=150,
        help="Used for make_circles, star_classification, and exam_score_prediction",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--test-size", type=float, default=0.2)

    ap.add_argument("--feature-map", default="zz_qiskit")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--entanglement", default=None)

    ap.add_argument("--partitions", default=None, help="JSON string: e.g. '[[0,1],[2,3]]' or '[[0],[1]]'")
    ap.add_argument("--method", default="rdm", choices=["rdm", "subcircuits"])
    ap.add_argument("--agg", default="mean", choices=["mean", "weighted"])
    ap.add_argument("--weights", default=None, help="JSON string: e.g. '[0.5, 0.5]' (only for agg=weighted)")
    ap.add_argument("--rdm-metric", default="fidelity", choices=["fidelity", "hs"], help="RDM similarity: 'fidelity' or Hilbert–Schmidt inner product 'hs'.")

    # Normalize flags (same convention as other demos)
    ap.add_argument("--normalize", dest="normalize", action="store_true", default=False, help="Normalize kernel to unit diagonal. Default: off.")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable unit-diagonal normalization (explicitly).")

    ap.add_argument("--center", action="store_true", help="Save/use centered kernel Kc = H K H.")
    ap.add_argument("--no-center", dest="center", action="store_false")
    ap.add_argument("--report-rank", action="store_true", help="Report spectrum stats (effective rank, min/max eigenvalue, trace).")

    ap.add_argument("--out-prefix", required=True, help="Prefix for outputs, e.g. outputs/local/loc_circles_local1q")
    ap.add_argument("--C", nargs="+", type=float, default=[0.1, 1.0, 10.0], help="SVM C grid")

    args = ap.parse_args()

    # data
    X, y = load_dataset(args.dataset, args.n_samples, args.seed)
    n, d = X.shape
    train_idx, val_idx, test_idx = make_splits(n, seed=args.seed, val_size=args.val_size, test_size=args.test_size)

    # parse partitions/weights
    partitions = None
    if args.partitions is not None:
        partitions = json.loads(args.partitions)

    weights = None
    if args.weights is not None:
        weights = json.loads(args.weights)

    # kernel
    kwargs = {"rdm_metric": args.rdm_metric}
    if args.entanglement is not None:
        kwargs["entanglement"] = args.entanglement

    K, meta = build_kernel(
        X,
        feature_map=args.feature_map,
        depth=args.depth,
        backend="statevector",
        seed=args.seed,
        partitions=partitions,
        method=args.method,
        agg=args.agg,
        weights=weights,
        **kwargs,
    )

    # optional normalize (script-level)
    if args.normalize:
        K = normalize_unit_diag(K)

    # optional centering
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

    # paths
    out_prefix = Path(args.out_prefix)
    out_dir, figs_dir = prepare_dirs(out_prefix, figs_subdir="figs/local")
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
        "method": args.method,
        "agg": args.agg,
        "partitions": partitions,
        "weights": weights,
        "rdm_metric": args.rdm_metric,
    })
    save_json(paths["mpath"], meta_out)
    save_splits(paths["spath"], train_idx, val_idx, test_idx, y)

    # figures
    plot_all(K, paths["fprefix"])

    # metrics (SVM precomputed)
    metrics = eval_precomputed(K, y, train_idx, val_idx, test_idx, args.C)

    # append CSV (same schema as other demos)
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
# # Dataset: make_circles (d=2)
# # Local-only (1q patches)
# python -m scripts.run_local_demo --dataset make_circles --n-samples 150 --feature-map zz_qiskit --depth 1 --entanglement linear --partitions '[[0],[1]]' --method rdm --agg mean --out-prefix outputs/local/loc_circles_local1q
#
# # Dataset: iris (d=4)
# # Local-only (1q patches)
# python -m scripts.run_local_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --partitions '[[0],[1],[2],[3]]' --method rdm --agg mean --out-prefix outputs/local/loc_iris_local1q
# # Local-only (2q patches)
# python -m scripts.run_local_demo --dataset iris --feature-map zz_manual_canonical --depth 1 --entanglement ring --partitions '[[0,1],[2,3]]' --method rdm --agg mean --out-prefix outputs/local/loc_iris_local2q --center --report-rank
