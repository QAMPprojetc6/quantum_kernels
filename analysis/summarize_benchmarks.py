"""
Summarize benchmark outputs into a single table (CSV/Markdown).

It scans a root directory for kernel artifacts:
  - *_K.npy (or *_K_centered.npy)
  - corresponding *_meta.json
  - corresponding *_splits.json  (must contain y_all, train_idx, val_idx, test_idx)

It computes diagnostics:
  - Off-diagonal mean/std + percentiles (p5/p50/p95)
  - Spectrum stats: lambda_min (sym), effective rank (entropy-based)
  - Centered kernel alignment with labels

It also tries to join SVM metrics from any metrics.csv found under the root:
  - expects columns like: out_prefix, best_C, val_acc, test_acc (dataset/feature_map/depth optional)

Usage:
  python -m analysis.summarize_benchmarks --root outputs/benchmarks/circles --out outputs/benchmarks/circles/summary.csv
  python -m analysis.summarize_benchmarks --root outputs --out outputs/summary.csv --md outputs/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------
# Helpers: kernel stats
# ---------------------------

def symmetrize(K: np.ndarray) -> np.ndarray:
    return 0.5 * (K + K.T)


def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - (1.0 / n) * np.ones((n, n))
    return H @ K @ H


def offdiag_values(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return K[mask]


def offdiag_stats(K: np.ndarray) -> Dict[str, float]:
    vals = offdiag_values(K)
    return {
        "offdiag_mean": float(np.mean(vals)),
        "offdiag_std": float(np.std(vals)),
        "offdiag_p5": float(np.percentile(vals, 5)),
        "offdiag_p50": float(np.percentile(vals, 50)),
        "offdiag_p95": float(np.percentile(vals, 95)),
    }


def effective_rank_entropy(eigvals: np.ndarray, eps: float = 1e-12) -> float:
    """
    Entropy-based effective rank:
      r_eff = exp(H(p)), where p_i = lambda_i / sum(lambda)
    Requires nonnegative spectrum; we clip tiny negatives to 0.
    """
    w = np.array(eigvals, dtype=np.float64)
    w[w < 0] = 0.0
    s = float(np.sum(w))
    if s <= eps:
        return 0.0
    p = w / s
    # Avoid log(0)
    p = np.clip(p, eps, 1.0)
    H = -float(np.sum(p * np.log(p)))
    return float(np.exp(H))


def spectrum_stats(K: np.ndarray) -> Dict[str, float]:
    Ks = symmetrize(K)
    w = np.linalg.eigvalsh(Ks)
    w_sorted = np.sort(w)[::-1]
    return {
        "lambda_min_sym": float(np.min(w_sorted)),
        "lambda_max_sym": float(np.max(w_sorted)),
        "trace": float(np.sum(w_sorted)),
        "eff_rank_entropy": float(effective_rank_entropy(w_sorted)),
    }


def label_kernel_onehot(y: np.ndarray) -> np.ndarray:
    """
    Build a label Gram matrix from one-hot encoded labels:
      L = Y Y^T
    Works for binary and multi-class.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    # Map labels to [0..C-1]
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((y.shape[0], classes.shape[0]), dtype=np.float64)
    for i, yi in enumerate(y):
        Y[i, idx[yi]] = 1.0
    return Y @ Y.T


def centered_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Centered kernel alignment:
      A = <Kc, Lc>_F / (||Kc||_F ||Lc||_F)
    where L is a label-kernel built from one-hot labels.
    """
    Kc = center_kernel(K)
    L = label_kernel_onehot(y)
    Lc = center_kernel(L)

    num = float(np.sum(Kc * Lc))
    den = float(np.linalg.norm(Kc, "fro") * np.linalg.norm(Lc, "fro"))
    if den == 0.0:
        return 0.0
    return num / den


# ---------------------------
# Metrics.csv join
# ---------------------------

def _read_metrics_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


def normalize_out_prefix_key(p: str) -> str:
    """
    Normalize an out_prefix to a stable join key across OS/path styles.

    We use only the basename (final path component), so:
      D:\\...\\outputs\\benchmarks\\circles\\circles_baseline_d1_s42
      outputs/benchmarks/circles/circles_baseline_d1_s42
    both map to:
      circles_baseline_d1_s42
    """
    if not p:
        return ""
    p = p.strip().strip('"').strip("'")
    # Normalize slashes, then take last component.
    p = p.replace("\\", "/")
    return p.split("/")[-1]


def load_metrics_index(root: Path) -> Dict[str, Dict[str, str]]:
    """
    Build an index: out_prefix -> row
    by scanning all metrics.csv under root.
    """
    index: Dict[str, Dict[str, str]] = {}
    for p in root.rglob("metrics.csv"):
        try:
            rows = _read_metrics_csv(p)
        except Exception:
            continue
        for r in rows:
            key_raw = r.get("out_prefix") or r.get("kernel_path") or ""
            key = normalize_out_prefix_key(key_raw)
            if key:
                index[key] = r
    return index


# ---------------------------
# Artifact scanning + parsing
# ---------------------------

@dataclass
class ArtifactTriplet:
    k_path: Path
    meta_path: Optional[Path]
    splits_path: Optional[Path]


def infer_out_prefix_from_kpath(k_path: Path) -> str:
    """
    Turn:
      .../ms_circles_baseline_K.npy -> .../ms_circles_baseline
      .../ms_iris_ms_X_K_centered.npy -> .../ms_iris_ms_X
    """
    stem = k_path.with_suffix("").as_posix()
    if stem.endswith("_K_centered"):
        return stem[:-len("_K_centered")]
    if stem.endswith("_K"):
        return stem[:-len("_K")]
    return stem


def find_artifacts(root: Path) -> List[ArtifactTriplet]:
    trips: List[ArtifactTriplet] = []
    for k_path in root.rglob("*_K.npy"):
        base = k_path.with_suffix("").as_posix()
        # meta/splits are commonly <prefix>_meta.json and <prefix>_splits.json
        out_prefix = infer_out_prefix_from_kpath(k_path)
        meta_path = Path(out_prefix + "_meta.json")
        splits_path = Path(out_prefix + "_splits.json")

        trips.append(
            ArtifactTriplet(
                k_path=k_path,
                meta_path=meta_path if meta_path.exists() else None,
                splits_path=splits_path if splits_path.exists() else None,
            )
        )

    # Also allow *_K_centered.npy if present (but those would not match *_K.npy pattern)
    for k_path in root.rglob("*_K_centered.npy"):
        # Avoid duplicates if both patterns caught it (they won't, but keep safe)
        if any(t.k_path == k_path for t in trips):
            continue
        out_prefix = infer_out_prefix_from_kpath(k_path)
        meta_path = Path(out_prefix + "_meta.json")
        splits_path = Path(out_prefix + "_splits.json")
        trips.append(
            ArtifactTriplet(
                k_path=k_path,
                meta_path=meta_path if meta_path.exists() else None,
                splits_path=splits_path if splits_path.exists() else None,
            )
        )

    return sorted(trips, key=lambda t: t.k_path.as_posix())


def safe_load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r") as f:
        return json.load(f)


def read_splits(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    with path.open("r") as f:
        data = json.load(f)
    # minimal expected keys
    if "y_all" not in data:
        return None
    return data


def summarize_one(trip: ArtifactTriplet, metrics_index: Dict[str, Dict[str, str]]) -> Optional[Dict[str, Any]]:
    K = np.load(trip.k_path)
    meta = safe_load_json(trip.meta_path)
    splits = read_splits(trip.splits_path)

    n = int(K.shape[0])
    centered_flag = bool(meta.get("center", False))
    # If it's a centered file, mark it
    if trip.k_path.with_suffix("").name.endswith("_K_centered"):
        centered_flag = True

    # dataset/d
    dataset = meta.get("dataset", "")
    d = meta.get("n_qubits", meta.get("num_qubits", meta.get("d", None)))
    if d is None and splits is not None:
        # Can't infer d from splits; leave blank
        d = ""
    # kernel name
    kernel_name = meta.get("kernel", "")
    if not kernel_name:
        # fall back to filename prefix
        kernel_name = trip.k_path.name

    # feature map summary
    fmap = meta.get("feature_map", meta.get("feature_map_name", ""))
    depth = meta.get("depth", "")
    ent = meta.get("entanglement", "")

    fmap_desc = f"{fmap}, depth={depth}, ent={ent}".strip().strip(",")
    # patches/scales/weights
    patches = meta.get("partitions", "")
    scales = meta.get("scales", "")
    weights = meta.get("weights", "")

    if scales:
        scales_or_patches = str(scales)
    elif patches:
        scales_or_patches = str(patches)
    else:
        scales_or_patches = ""

    weights_str = str(weights) if weights else ""

    # Diagnostics: offdiag & spectrum
    od = offdiag_stats(K)
    sp = spectrum_stats(K)

    # Alignment (centered)
    align = ""
    if splits is not None:
        y = np.asarray(splits["y_all"])
        align = float(centered_alignment(K, y))

    # Join SVM metrics via out_prefix key
    out_prefix = infer_out_prefix_from_kpath(trip.k_path)
    join_key = normalize_out_prefix_key(out_prefix)
    mrow = metrics_index.get(join_key, {})
    best_C = mrow.get("best_C", "")
    val_acc = mrow.get("val_acc", "")
    test_acc = mrow.get("test_acc", "")

    notes = ""
    # PSD-ish hint
    if sp["lambda_min_sym"] < -1e-8:
        notes = "λ_min<0 (regularization may be needed)"

    return {
        "Kernel": kernel_name,
        "Dataset": dataset,
        "n": n,
        "d (qubits)": d,
        "Feature map (name, depth, entanglement)": fmap_desc,
        "Scales / Patches": scales_or_patches,
        "Weights": weights_str,
        "Centered?": bool(centered_flag),

        "Off-diag mean": od["offdiag_mean"],
        "Off-diag std": od["offdiag_std"],
        "Off-diag p5": od["offdiag_p5"],
        "Off-diag p50": od["offdiag_p50"],
        "Off-diag p95": od["offdiag_p95"],

        "Eff. rank (entropy)": sp["eff_rank_entropy"],
        "λ_min (sym)": sp["lambda_min_sym"],
        "Alignment (centered)": align,

        "SVM best C": best_C,
        "Val acc": val_acc,
        "Test acc": test_acc,
        "Notes": notes,

        "_k_path": trip.k_path.as_posix(),
        "_out_prefix": out_prefix,
        "_join_key": join_key,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_markdown(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    # header
    md = []
    md.append("| " + " | ".join(fieldnames) + " |")
    md.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for r in rows:
        md.append("| " + " | ".join(fmt(r.get(k, "")) for k in fieldnames) + " |")

    path.write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Summarize kernel benchmark artifacts into a single table.")
    ap.add_argument("--root", required=True, help="Root directory to scan (e.g., outputs/benchmarks/circles)")
    ap.add_argument("--out", required=True, help="Output CSV path (e.g., outputs/benchmarks/circles/summary.csv)")
    ap.add_argument("--md", default=None, help="Optional Markdown output path")
    ap.add_argument("--include-paths", action="store_true", help="Include artifact paths columns in output")
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out)
    out_md = Path(args.md) if args.md else None

    metrics_index = load_metrics_index(root)
    trips = find_artifacts(root)

    rows: List[Dict[str, Any]] = []
    for t in trips:
        r = summarize_one(t, metrics_index)
        if r is not None:
            rows.append(r)

    if not rows:
        print(f"[WARN] No artifacts found under: {root}")
        return

    # Column order aligned with your template (plus split mean/std into two cols)
    base_fields = [
        "Kernel",
        "Dataset",
        "n",
        "d (qubits)",
        "Feature map (name, depth, entanglement)",
        "Scales / Patches",
        "Weights",
        "Centered?",
        "Off-diag mean",
        "Off-diag std",
        "Off-diag p5",
        "Off-diag p50",
        "Off-diag p95",
        "Eff. rank (entropy)",
        "λ_min (sym)",
        "Alignment (centered)",
        "SVM best C",
        "Val acc",
        "Test acc",
        "Notes",
    ]

    if args.include_paths:
        base_fields += ["_out_prefix", "_k_path"]

    # Sort for readability: Dataset then Kernel then n/d
    rows_sorted = sorted(
        rows,
        key=lambda r: (str(r.get("Dataset", "")), str(r.get("Kernel", "")), int(r.get("n", 0))),
    )

    write_csv(out_csv, rows_sorted, base_fields)
    print(f"[OK] Wrote CSV summary: {out_csv}")

    if out_md is not None:
        write_markdown(out_md, rows_sorted, base_fields)
        print(f"[OK] Wrote Markdown summary: {out_md}")


if __name__ == "__main__":
    main()


# ---------------------------
# Examples
# ---------------------------
#
# python -m analysis.summarize_benchmarks --root outputs/benchmarks/circles --out outputs/benchmarks/circles/summary.csv --md outputs/benchmarks/circles/summary.md
#
# python -m analysis.summarize_benchmarks --root outputs/benchmarks/iris --out outputs/benchmarks/iris/summary.csv --md outputs/benchmarks/iris/summary.md
#

