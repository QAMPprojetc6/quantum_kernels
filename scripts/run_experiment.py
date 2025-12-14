"""
Run full benchmarks (baseline + local + multiscale) from a TOML config.

Run from repo root:
  python -m scripts.run_experiment --config configs/circles.toml
  python -m scripts.run_experiment --config configs/iris.toml

Artifacts written per run (out_prefix):
  - <out_prefix>_K.npy
  - <out_prefix>_K_centered.npy (if enabled)
  - <out_prefix>_splits.json
  - <out_prefix>_meta.json
  - figs: <figs_dir>/<out_prefix.name>_matrix.png, _offdiag_hist.png, _spectrum.png
  - metrics CSV: <out_dir>/metrics.csv  (one row per run)

Config schema (expected, minimal):
  [run] dataset, n_samples (for circles), seed_grid, val_size, test_size
  [paths] out_dir, figs_dir
  [feature_map] name, entanglement, depth_grid, backend
  [post] normalize, center_grid, report_rank
  [svm] C_grid
  [[kernels]] name in {"baseline","local","multiscale"}, enabled=true/false, plus kernel-specific params
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# TOML reader: Python 3.11+ has tomllib
try:
    import tomllib  # type: ignore
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from analysis.diagnostics import plot_heatmap, plot_offdiag_hist, plot_spectrum
from analysis.eval_svm import eval_precomputed

from qkernels.baseline_kernel import build_kernel as build_baseline_kernel
from qkernels.local_kernel import build_kernel as build_local_kernel
from qkernels.multiscale_kernel import build_kernel as build_multiscale_kernel

from scripts.demo_common import load_dataset, make_splits, center_kernel, spectrum_stats, normalize_unit_diag, save_splits


# ---------------------------
# utils
# ---------------------------

def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-serializable python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    return obj


def _dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2), encoding="utf-8")


def _fmt_float_tag(x: float) -> str:
    # 0.5 -> "0p5", 0.05 -> "0p05"
    s = f"{x:g}"
    return s.replace(".", "p").replace("-", "m")


def _infer_local_tag(partitions: List[List[int]]) -> str:
    sizes = sorted({len(p) for p in partitions})
    if sizes == [1]:
        return "local1q"
    if sizes == [2]:
        return "local2q"
    if len(sizes) == 1:
        return f"local{sizes[0]}q"
    return "local_mixed"


def _case_name(
    dataset_id: str,
    kernel_name: str,
    tag: str,
    depth: int,
    seed: int,
    weights: Optional[List[float]] = None,
    normalize: bool = False,
    centered: bool = False,
) -> str:
    parts = [dataset_id, kernel_name]
    if tag:
        parts.append(tag)
    parts.append(f"d{depth}")
    parts.append(f"s{seed}")
    if weights is not None:
        parts.append("w" + "-".join(_fmt_float_tag(float(w)) for w in weights))
    if normalize:
        parts.append("norm")
    if centered:
        parts.append("centered")
    return "_".join(parts)


def _write_metrics_row(metrics_csv: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_csv.exists()
    with metrics_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        # ensure all fields exist
        w.writerow({k: _json_safe(row.get(k, "")) for k in fieldnames})


# ---------------------------
# kernel runners
# ---------------------------

def _run_one(
    *,
    out_prefix: Path,
    figs_dir: Path,
    kernel_kind: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: str,
    seed: int,
    feature_map: str,
    entanglement: Optional[str],
    depth: int,
    backend: str,
    C_grid: List[float],
    normalize: bool,
    center: bool,
    report_rank: bool,
    # kernel-specific
    method: Optional[str] = None,
    agg: Optional[str] = None,
    partitions: Optional[List[List[int]]] = None,
    rdm_metric: Optional[str] = None,
    scales: Optional[List[List[List[int]]]] = None,
    weights: Optional[List[float]] = None,
) -> None:
    # 1) build K
    if kernel_kind == "baseline":
        K, meta_k = build_baseline_kernel(
            X,
            feature_map=feature_map,
            depth=depth,
            backend=backend,
            seed=seed,
            entanglement=entanglement,
        )
    elif kernel_kind == "local":
        K, meta_k = build_local_kernel(
            X,
            feature_map=feature_map,
            depth=depth,
            backend=backend,
            seed=seed,
            entanglement=entanglement,
            partitions=partitions,
            method=method or "rdm",
            agg=agg or "mean",
            weights=None,
            rdm_metric=rdm_metric,
        )
    elif kernel_kind == "multiscale":
        K, meta_k = build_multiscale_kernel(
            X,
            feature_map=feature_map,
            depth=depth,
            backend=backend,
            seed=seed,
            entanglement=entanglement,
            scales=scales,
            weights=weights,
        )
    else:
        raise ValueError(f"Unknown kernel kind: {kernel_kind}")

    # 2) optional normalize
    if normalize:
        K = normalize_unit_diag(K)

    # 3) save kernels
    K_path = out_prefix.with_name(out_prefix.name + "_K.npy")
    np.save(K_path, K)

    K_used = K
    Kc_path = None
    if center:
        Kc = center_kernel(K)
        Kc_path = out_prefix.with_name(out_prefix.name + "_K_centered.npy")
        np.save(Kc_path, Kc)
        K_used = Kc

    # 4) save splits (per-run for easy joins)
    splits_path = out_prefix.with_name(out_prefix.name + "_splits.json")
    save_splits(splits_path.as_posix(), train_idx, val_idx, test_idx, y)

    # 5) figures (always for uncentered K, like the demos)
    fig_prefix = figs_dir / out_prefix.name
    plot_heatmap(K, str(fig_prefix) + "_matrix.png")
    plot_offdiag_hist(K, str(fig_prefix) + "_offdiag_hist.png")
    plot_spectrum(K, str(fig_prefix) + "_spectrum.png")

    # 6) meta (merge kernel meta + runner meta)
    meta_out: Dict[str, Any] = {}
    meta_out.update(meta_k if isinstance(meta_k, dict) else {})
    meta_out.update(
        {
            "dataset": dataset,
            "seed": seed,
            "n_samples": int(X.shape[0]),
            "d": int(X.shape[1]),
            "feature_map": feature_map,
            "entanglement": entanglement,
            "depth": depth,
            "backend": backend,
            "normalize": bool(normalize),
            "center": bool(center),
            "C_grid": list(map(float, C_grid)),
        }
    )

    if partitions is not None:
        meta_out["partitions"] = partitions
    if scales is not None:
        meta_out["scales"] = scales
    if weights is not None:
        meta_out["weights"] = weights
    if method is not None:
        meta_out["method"] = method
    if agg is not None:
        meta_out["agg"] = agg
    if rdm_metric is not None:
        meta_out["rdm_metric"] = rdm_metric

    # 7) optional spectrum report (use K_used like demos)
    if report_rank:
        stats = spectrum_stats(K_used)
        meta_out["spectrum_stats"] = stats
        spec_path = out_prefix.with_name(out_prefix.name + "_spectrum.txt")
        spec_path.write_text(json.dumps(_json_safe(stats), indent=2), encoding="utf-8")

    meta_path = out_prefix.with_name(out_prefix.name + "_meta.json")
    _dump_json(meta_path, meta_out)

    # 8) SVM eval (precomputed, uses K_used like demos)
    metrics = eval_precomputed(K_used, y, train_idx, val_idx, test_idx, list(map(float, C_grid)))

    # 9) append metrics row
    metrics_csv = out_prefix.parent / "metrics.csv"
    fieldnames = [
        "kernel",
        "dataset",
        "seed",
        "out_prefix",
        "feature_map",
        "depth",
        "entanglement",
        "backend",
        "normalize",
        "center",
        "method",
        "agg",
        "rdm_metric",
        "partitions",
        "scales",
        "weights",
        "best_C",
        "val_acc",
        "test_acc",
    ]
    row = {
        "kernel": kernel_kind,
        "dataset": dataset,
        "seed": seed,
        "out_prefix": str(out_prefix),
        "feature_map": feature_map,
        "depth": depth,
        "entanglement": entanglement,
        "backend": backend,
        "normalize": bool(normalize),
        "center": bool(center),
        "method": method or "",
        "agg": agg or "",
        "rdm_metric": rdm_metric or "",
        "partitions": partitions or "",
        "scales": scales or "",
        "weights": weights or "",
        "best_C": metrics["best_C"],
        "val_acc": metrics["val_acc"],
        "test_acc": metrics["test_acc"],
    }
    _write_metrics_row(metrics_csv, row, fieldnames)

    # 10) console summary
    print(f"[OK] {kernel_kind:9s} | {out_prefix.name}")
    print(f"     K:        {K_path.name}")
    if Kc_path is not None:
        print(f"     K_center: {Kc_path.name}")
    print(f"     meta:     {meta_path.name}")
    print(f"     splits:   {splits_path.name}")
    print(f"     figs:     {fig_prefix.name}_*.png")
    print(f"     metrics:  best_C={metrics['best_C']} val={metrics['val_acc']:.4f} test={metrics['test_acc']:.4f}")


# ---------------------------
# main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run full kernel benchmarks from a TOML config.")
    ap.add_argument("--config", required=True, help="Path to TOML config, e.g. configs/circles.toml")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)

    # Resolve config path relative to repo root (not current working directory)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()

    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

    # read config
    run_cfg = cfg.get("run", {})
    dataset = str(run_cfg.get("dataset", "make_circles")).strip()
    n_samples = int(run_cfg.get("n_samples", 150))
    seed_grid = list(run_cfg.get("seed_grid", [42]))
    val_size = float(run_cfg.get("val_size", 0.2))
    test_size = float(run_cfg.get("test_size", 0.2))

    paths_cfg = cfg.get("paths", {})
    out_dir = Path(paths_cfg.get("out_dir", "outputs/benchmarks"))
    figs_dir = Path(paths_cfg.get("figs_dir", "figs/benchmarks"))

    # Resolve output paths relative to repo root (not IDE working directory)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    if not figs_dir.is_absolute():
        figs_dir = (repo_root / figs_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    fmap_cfg = cfg.get("feature_map", {})
    feature_map = str(fmap_cfg.get("name", "zz_qiskit"))
    entanglement = fmap_cfg.get("entanglement", None)
    depth_grid = list(fmap_cfg.get("depth_grid", [1]))
    backend = str(fmap_cfg.get("backend", "statevector"))

    post_cfg = cfg.get("post", {})
    normalize_default = bool(post_cfg.get("normalize", False))
    center_grid = list(post_cfg.get("center_grid", [False]))
    report_rank = bool(post_cfg.get("report_rank", True))

    svm_cfg = cfg.get("svm", {})
    C_grid = list(map(float, svm_cfg.get("C_grid", [0.1, 1.0, 10.0])))

    kernels_cfg = cfg.get("kernels", [])
    if not isinstance(kernels_cfg, list) or len(kernels_cfg) == 0:
        raise ValueError("Config must include at least one [[kernels]] entry.")

    # Friendly dataset id used in filenames
    dataset_id = "circles" if dataset == "make_circles" else dataset

    print(f"[RUN] config={cfg_path} dataset={dataset} out_dir={out_dir} figs_dir={figs_dir}")
    print(f"      seeds={seed_grid} depths={depth_grid} center_grid={center_grid} normalize_default={normalize_default}")

    # sweep
    for seed in seed_grid:
        # Load dataset once per seed (matches demo behavior)
        X, y = load_dataset(dataset, n_samples=n_samples, seed=int(seed))
        train_idx, val_idx, test_idx = make_splits(len(X), seed=int(seed), val_size=val_size, test_size=test_size)

        for depth in depth_grid:
            for center in center_grid:
                for kspec in kernels_cfg:
                    if not bool(kspec.get("enabled", True)):
                        continue

                    kname = str(kspec.get("name", "")).strip().lower()
                    if kname not in {"baseline", "local", "multiscale"}:
                        raise ValueError(f"Unsupported kernel name in config: {kname!r}")

                    # kernel-level override for normalize (optional)
                    normalize = bool(kspec.get("normalize", normalize_default))

                    # ---- kernel-specific params ----
                    tag = str(kspec.get("tag", "")).strip()

                    if kname == "baseline":
                        case = _case_name(
                            dataset_id=dataset_id,
                            kernel_name="baseline",
                            tag="",
                            depth=int(depth),
                            seed=int(seed),
                            weights=None,
                            normalize=normalize,
                            centered=bool(center),
                        )
                        out_prefix = out_dir / case
                        _run_one(
                            out_prefix=out_prefix,
                            figs_dir=figs_dir,
                            kernel_kind="baseline",
                            X=X,
                            y=y,
                            train_idx=train_idx,
                            val_idx=val_idx,
                            test_idx=test_idx,
                            dataset=dataset,
                            seed=int(seed),
                            feature_map=feature_map,
                            entanglement=entanglement,
                            depth=int(depth),
                            backend=backend,
                            C_grid=C_grid,
                            normalize=normalize,
                            center=bool(center),
                            report_rank=report_rank,
                        )

                    elif kname == "local":
                        partitions = kspec.get("partitions", None)
                        if partitions is None:
                            # default: 1q partitions
                            partitions = [[i] for i in range(X.shape[1])]
                        partitions = [list(map(int, p)) for p in partitions]

                        method = str(kspec.get("method", "rdm"))
                        agg = str(kspec.get("agg", "mean"))
                        rdm_metric = kspec.get("rdm_metric", None)

                        if not tag:
                            tag = _infer_local_tag(partitions)

                        case = _case_name(
                            dataset_id=dataset_id,
                            kernel_name="local",
                            tag=tag,
                            depth=int(depth),
                            seed=int(seed),
                            weights=None,
                            normalize=normalize,
                            centered=bool(center),
                        )
                        out_prefix = out_dir / case
                        _run_one(
                            out_prefix=out_prefix,
                            figs_dir=figs_dir,
                            kernel_kind="local",
                            X=X,
                            y=y,
                            train_idx=train_idx,
                            val_idx=val_idx,
                            test_idx=test_idx,
                            dataset=dataset,
                            seed=int(seed),
                            feature_map=feature_map,
                            entanglement=entanglement,
                            depth=int(depth),
                            backend=backend,
                            C_grid=C_grid,
                            normalize=normalize,
                            center=bool(center),
                            report_rank=report_rank,
                            method=method,
                            agg=agg,
                            partitions=partitions,
                            rdm_metric=rdm_metric,
                        )

                    elif kname == "multiscale":
                        scales = kspec.get("scales", None)
                        if scales is None:
                            # Default: pairs then all (like earlier default)
                            d = int(X.shape[1])
                            pairs = [[i, i + 1] for i in range(0, d, 2) if i + 1 < d]
                            scales = [pairs, [list(range(d))]]
                        # normalize scale structure to List[List[List[int]]]
                        scales = [[[int(i) for i in patch] for patch in scale] for scale in scales]

                        weights_grid = kspec.get("weights_grid", None)
                        if weights_grid is None:
                            weights = kspec.get("weights", None)
                            if weights is None:
                                weights = [1.0 / len(scales)] * len(scales)
                            weights_grid = [weights]
                        # list of weight vectors
                        weights_grid = [list(map(float, w)) for w in weights_grid]

                        # default tag if missing
                        if not tag:
                            tag = "multiscale"

                        for weights in weights_grid:
                            case = _case_name(
                                dataset_id=dataset_id,
                                kernel_name="multiscale",
                                tag=tag,
                                depth=int(depth),
                                seed=int(seed),
                                weights=weights,
                                normalize=normalize,
                                centered=bool(center),
                            )
                            out_prefix = out_dir / case
                            _run_one(
                                out_prefix=out_prefix,
                                figs_dir=figs_dir,
                                kernel_kind="multiscale",
                                X=X,
                                y=y,
                                train_idx=train_idx,
                                val_idx=val_idx,
                                test_idx=test_idx,
                                dataset=dataset,
                                seed=int(seed),
                                feature_map=feature_map,
                                entanglement=entanglement,
                                depth=int(depth),
                                backend=backend,
                                C_grid=C_grid,
                                normalize=normalize,
                                center=bool(center),
                                report_rank=report_rank,
                                scales=scales,
                                weights=weights,
                            )

    print("[DONE] Benchmarks finished.")


if __name__ == "__main__":
    main()


# ---------------------------
# Examples
# ---------------------------
#
# python -m scripts.run_experiment --config configs/circles.toml
#
# python -m scripts.run_experiment --config configs/iris.toml
#

