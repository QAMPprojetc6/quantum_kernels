"""
Plot selected SVM C vs d (qubits) per dataset and kernel, aggregating across seeds.

Inputs:
  - summary_all.csv or summary_all.md from analysis.summarize_benchmarks

Outputs:
  - per-kernel plots (baseline/local/multiscale) for each dataset
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


# -----------------------------
# Parsing helpers
# -----------------------------

def _to_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if x == "" or x.lower() in {"none", "nan"}:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _to_int(x: str) -> Optional[int]:
    x = (x or "").strip()
    if x == "" or x.lower() in {"none", "nan"}:
        return None
    try:
        return int(x)
    except ValueError:
        return None


def _to_bool(x: str) -> Optional[bool]:
    x = (x or "").strip().lower()
    if x in {"true", "1", "yes"}:
        return True
    if x in {"false", "0", "no"}:
        return False
    return None


def _norm_weights_str(s: str) -> str:
    s = (s or "").strip()
    if s == "":
        return ""
    return re.sub(r"\s+", "", s)


_FMAP_RE = re.compile(
    r"^\s*(?P<name>[^,]+)\s*,\s*depth=(?P<depth>\d+)\s*,\s*ent=(?P<ent>[^,]+)\s*$"
)


def _parse_feature_map_cell(cell: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    cell = (cell or "").strip()
    m = _FMAP_RE.match(cell)
    if not m:
        return None, None, None
    name = m.group("name").strip()
    depth = int(m.group("depth"))
    ent = m.group("ent").strip()
    return name, depth, ent


def _parse_md_table(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") and "Kernel" in ln and "Dataset" in ln and "|" in ln:
            if i + 1 < len(lines) and re.match(r"^\s*\|\s*-", lines[i + 1]):
                start = i
                break
    if start is None:
        raise ValueError(f"No markdown table found in: {path}")

    header = [c.strip() for c in lines[start].strip().strip("|").split("|")]
    rows: List[Dict[str, str]] = []

    j = start + 2
    while j < len(lines):
        ln = lines[j].strip()
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < len(header):
            cells = cells + [""] * (len(header) - len(cells))
        if len(cells) > len(header):
            cells = cells[: len(header)]
        row = dict(zip(header, cells))
        if any(v.strip() for v in row.values()):
            rows.append(row)
        j += 1

    return rows


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".md":
        return _parse_md_table(path)
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class Row:
    kernel: str
    dataset: str
    d: int
    fmap_name: Optional[str]
    fmap_depth: Optional[int]
    fmap_ent: Optional[str]
    centered: Optional[bool]
    weights: str
    val_acc: Optional[float]
    best_c: Optional[float]


def _coerce_rows(rows: List[Dict[str, str]]) -> List[Row]:
    out: List[Row] = []
    for r in rows:
        kernel = (r.get("Kernel") or r.get("kernel") or "").strip().lower()
        dataset = (r.get("Dataset") or r.get("dataset") or "").strip()
        d = _to_int(r.get("d (qubits)", "") or r.get("d", ""))
        if kernel == "" or dataset == "" or d is None:
            continue

        fmap_cell = r.get("Feature map (name, depth, entanglement)", "") or r.get("Feature map", "")
        fmap_name, fmap_depth, fmap_ent = _parse_feature_map_cell(fmap_cell)

        centered = _to_bool(r.get("Centered?", "") or r.get("Centered", ""))
        weights = _norm_weights_str(r.get("Weights", "") or r.get("weights", ""))
        val_acc = _to_float(r.get("Val acc", "") or r.get("Val", ""))
        best_c = _to_float(r.get("SVM best C", "") or r.get("best_C", ""))

        out.append(
            Row(
                kernel=kernel,
                dataset=dataset,
                d=int(d),
                fmap_name=fmap_name,
                fmap_depth=fmap_depth,
                fmap_ent=fmap_ent,
                centered=centered,
                weights=weights,
                val_acc=val_acc,
                best_c=best_c,
            )
        )
    return out


# -----------------------------
# Aggregation
# -----------------------------

def _mode(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    counts: Dict[float, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    # Deterministic tie-break: smallest C
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

def _mean(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _aggregate_by_d(rows: List[Row]) -> Dict[int, Optional[float]]:
    by_d: Dict[int, List[float]] = {}
    for r in rows:
        if r.best_c is None:
            continue
        by_d.setdefault(r.d, []).append(float(r.best_c))
    return {d: _mode(vs) for d, vs in by_d.items()}


def _pick_multiscale_rows(rows: List[Row], weights_filter: str) -> List[Row]:
    ms = [r for r in rows if r.kernel == "multiscale"]
    if not ms:
        return []

    if weights_filter:
        wf = _norm_weights_str(weights_filter)
        return [r for r in ms if r.weights == wf]

    picked: List[Row] = []
    by_d: Dict[int, Dict[str, List[Row]]] = {}
    for r in ms:
        by_d.setdefault(r.d, {}).setdefault(r.weights, []).append(r)

    for d, groups in by_d.items():
        best_w = None
        best_val = None
        for w, rs in groups.items():
            vals = [x.val_acc for x in rs if x.val_acc is not None]
            if not vals:
                continue
            m = _mean([float(v) for v in vals])
            if m is None:
                continue
            if best_val is None or m > best_val:
                best_val = m
                best_w = w
        if best_w is None:
            for rs in groups.values():
                picked.extend(rs)
        else:
            picked.extend(groups[best_w])
    return picked


def _filter_common(
    rows: List[Row],
    dataset: str,
    feature_map: Optional[str],
    depth: Optional[int],
    entanglement: Optional[str],
    centered: Optional[bool],
) -> List[Row]:
    out = [r for r in rows if r.dataset == dataset]
    if feature_map is not None:
        out = [r for r in out if r.fmap_name == feature_map]
    if depth is not None:
        out = [r for r in out if r.fmap_depth == depth]
    if entanglement is not None:
        out = [r for r in out if r.fmap_ent == entanglement]
    if centered is not None:
        out = [r for r in out if r.centered == centered]
    return out


# -----------------------------
# Plotting
# -----------------------------

STYLE = {
    "baseline": dict(color="#000000", marker="o", linestyle="-", label="baseline"),
    "local": dict(color="#E69F00", marker="s", linestyle="--", label="local"),
    "multiscale": dict(color="#56B4E9", marker="^", linestyle="-.", label="multiscale"),
}


def _plot_combined(
    dataset: str,
    curves: Dict[str, Dict[int, Optional[float]]],
    out_dir: Path,
) -> None:
    xs_all = sorted({d for by_d in curves.values() for d in by_d.keys()})
    if not xs_all:
        return

    plt.figure(figsize=(8, 5))
    for kernel in ["baseline", "local", "multiscale"]:
        by_d = curves.get(kernel, {})
        if not by_d:
            continue

        ys = []
        for d in xs_all:
            ys.append(by_d.get(d, None))

        xs_plot = [x for x, y in zip(xs_all, ys) if y is not None]
        ys_plot = [y for y in ys if y is not None]
        if not xs_plot:
            continue

        style = STYLE[kernel]
        plt.plot(
            xs_plot,
            ys_plot,
            linewidth=2.0,
            markersize=6,
            markeredgewidth=1.4,
            **style,
        )

    plt.title(f"{dataset}: best C vs d")
    plt.xlabel("d (qubits)")
    plt.ylabel("best C")
    plt.xticks(xs_all)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend(frameon=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_bestC_vs_d.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot selected SVM C vs d per dataset and kernel.")
    ap.add_argument("--summary", required=True, help="Path to summary_all.csv or summary_all.md")
    ap.add_argument("--out", required=True, help="Output directory for plots")
    ap.add_argument("--dataset", type=str, default="", help="Dataset name to plot (default: all datasets found).")
    ap.add_argument("--feature-map", type=str, default="", help="Filter by feature map name (e.g., zz_qiskit).")
    ap.add_argument("--depth", type=int, default=-1, help="Filter by depth (e.g., 1).")
    ap.add_argument("--entanglement", type=str, default="", help="Filter by entanglement (e.g., linear/ring).")
    ap.add_argument("--centered", type=int, default=-1, help="Filter by centered: 0/1 (default: no filter).")
    ap.add_argument(
        "--multiscale-weights",
        type=str,
        default="",
        help='If set, fixes multiscale weights selection, e.g. "[0.5, 0.5]". '
             "If not set, chooses (per d) the weights group with best mean val_acc.",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out)

    fmap = args.feature_map.strip() or None
    depth = args.depth if args.depth >= 0 else None
    ent = args.entanglement.strip() or None
    centered = None if args.centered < 0 else (True if args.centered == 1 else False)

    rows_raw = _read_rows(summary_path)
    rows = _coerce_rows(rows_raw)

    if args.dataset.strip():
        datasets = [args.dataset.strip()]
    else:
        datasets = sorted({r.dataset for r in rows})

    for ds in datasets:
        block = _filter_common(rows, ds, fmap, depth, ent, centered)
        if not block:
            print(f"[WARN] No rows matched for dataset={ds} with the given filters.")
            continue

        base_rows = [r for r in block if r.kernel == "baseline"]
        loc_rows = [r for r in block if r.kernel == "local"]
        ms_rows = _pick_multiscale_rows(block, args.multiscale_weights)

        curves = {
            "baseline": _aggregate_by_d(base_rows),
            "local": _aggregate_by_d(loc_rows),
            "multiscale": _aggregate_by_d(ms_rows),
        }

        _plot_combined(ds, curves, out_dir)
        print(f"[OK] Wrote best-C plot for dataset={ds}")


if __name__ == "__main__":
    main()


# Examples:
# python -m analysis.plot_c_vs_d --summary outputs/benchmarks/summary_all.csv --out figs/checkpoint3/c_vs_d
# python -m analysis.plot_c_vs_d --summary outputs/benchmarks/summary_all.csv --out figs/checkpoint3/c_vs_d --dataset breast_cancer
