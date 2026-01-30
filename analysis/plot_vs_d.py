"""
Plot metrics vs d (qubits) from a Markdown summary table (summary_all.md / summary.md).

Generates per-dataset plots:
  - Concentration vs d: Off-diag p50 (median). Optional: also Off-diag p95.
  - Spectrum richness vs d: effective rank (entropy-based).
  - Performance vs d: test accuracy.

The script expects a Markdown table with columns like those produced by summarize_benchmarks,
e.g.:
| Kernel | Dataset | n | d (qubits) | Feature map (name, depth, entanglement) | ... | Centered? | ... |
| ...    | ...     |   |     ...    | zz_qiskit, depth=1, ent=linear          | ... | False     | ... |

Usage examples:
  python -m analysis.plot_vs_d --summary outputs/benchmarks/summary_all.md --out figs/checkpoint2/vs_d

  # Parkinsons sweep (depth=1 only, centered=False) with multiscale weight fixed to [0.5, 0.5]
  python -m analysis.plot_vs_d --summary outputs/benchmarks/summary_all.md --out figs/checkpoint2/vs_d \
    --dataset parkinsons --feature-map zz_qiskit --depth 1 --entanglement linear --centered 0 \
    --multiscale-weights "[0.5, 0.5]" --also-p95

  # Breast cancer sweep (depth=1 only, centered=False)
  python -m analysis.plot_vs_d --summary outputs/benchmarks/summary_all.md --out figs/checkpoint2/vs_d \
    --dataset breast_cancer --feature-map zz_manual_canonical --depth 1 --entanglement ring --centered 0 \
    --multiscale-weights "[0.5, 0.5]" --also-p95
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

# Color/marker styles
# - Distinct markers + line styles so plots remain readable even in grayscale.
# - Palette based on Okabe–Ito (high contrast), plus different linestyles.
STYLE_MAP = {
    "baseline":   dict(color="#000000", marker="o", linestyle="-",  markerfacecolor="#000000", markeredgecolor="#000000"),
    "local":      dict(color="#E69F00", marker="s", linestyle="--", markerfacecolor="none",     markeredgecolor="#E69F00"),
    "multiscale": dict(color="#56B4E9", marker="^", linestyle="-.", markerfacecolor="none",     markeredgecolor="#56B4E9"),
}
PLOT_ORDER = ["baseline", "local", "multiscale"]

def _style_for_series(name: str) -> dict:
    """Resolve style for a series name (robust to name decorations)."""
    key = name.strip().lower()
    for k in STYLE_MAP:
        if key == k or key.startswith(k):
            return STYLE_MAP[k].copy()
    # Fallback: still use distinct marker/linestyle
    return dict(marker="D", linestyle=":", markerfacecolor="none")


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
    """Normalize weights string like '[0.5, 0.5]' -> '[0.5,0.5]' for stable matching."""
    s = (s or "").strip()
    if s == "":
        return ""
    # remove spaces
    s = re.sub(r"\s+", "", s)
    return s


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


def _parse_md_table(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Parse the *first* markdown pipe table found in the file.
    Returns (header_fields, rows_as_str_dicts).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    start = None
    for i, ln in enumerate(lines):
        # Heuristic: the summary table header always includes these two columns
        if ln.strip().startswith("|") and "Kernel" in ln and "Dataset" in ln and "|" in ln:
            # Next line should be separator like | --- | --- |
            if i + 1 < len(lines) and re.match(r"^\s*\|\s*-", lines[i + 1]):
                start = i
                break
    if start is None:
        raise ValueError(f"No markdown table found in: {path}")

    header = [c.strip() for c in lines[start].strip().strip("|").split("|")]
    rows: List[Dict[str, str]] = []

    j = start + 2  # skip header + separator
    while j < len(lines):
        ln = lines[j].strip()
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        # tolerate short/long rows
        if len(cells) < len(header):
            cells = cells + [""] * (len(header) - len(cells))
        if len(cells) > len(header):
            cells = cells[: len(header)]
        row = dict(zip(header, cells))
        # ignore fully-empty rows
        if any(v.strip() for v in row.values()):
            rows.append(row)
        j += 1

    return header, rows


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
    weights: str  # normalized string
    off_p50: Optional[float]
    off_p95: Optional[float]
    eff_rank: Optional[float]
    val_acc: Optional[float]
    test_acc: Optional[float]


def _coerce_rows(rows: List[Dict[str, str]]) -> List[Row]:
    out: List[Row] = []
    for r in rows:
        kernel = (r.get("Kernel") or r.get("kernel") or "").strip()
        dataset = (r.get("Dataset") or r.get("dataset") or "").strip()

        d = _to_int(r.get("d (qubits)", "") or r.get("d", ""))
        if kernel == "" or dataset == "" or d is None:
            continue

        fmap_cell = r.get("Feature map (name, depth, entanglement)", "") or r.get("Feature map", "")
        fmap_name, fmap_depth, fmap_ent = _parse_feature_map_cell(fmap_cell)

        centered = _to_bool(r.get("Centered?", "") or r.get("Centered", ""))

        weights = _norm_weights_str(r.get("Weights", "") or r.get("weights", ""))

        off_p50 = _to_float(r.get("Off-diag p50", "") or r.get("Off-diag median", ""))
        off_p95 = _to_float(r.get("Off-diag p95", "") or r.get("Off-diag p95 / p50 / p5", ""))

        eff_rank = _to_float(r.get("Eff. rank (entropy)", "") or r.get("Eff. rank", "") or r.get("EffRank", ""))
        val_acc = _to_float(r.get("Val acc", "") or r.get("Val", ""))
        test_acc = _to_float(r.get("Test acc", "") or r.get("Test", ""))

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
                off_p50=off_p50,
                off_p95=off_p95,
                eff_rank=eff_rank,
                val_acc=val_acc,
                test_acc=test_acc,
            )
        )
    return out


# -----------------------------
# Aggregation
# -----------------------------

def _mean_std(vals: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, v ** 0.5


def _aggregate_by_d(rows: List[Row], metric: str) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """
    rows: already filtered to a single kernel (and optionally a single weights group)
    metric in {"off_p50", "off_p95", "eff_rank", "val_acc", "test_acc"}
    """
    by_d: Dict[int, List[float]] = {}
    for r in rows:
        v = getattr(r, metric)
        if v is None:
            continue
        by_d.setdefault(r.d, []).append(float(v))
    return {d: _mean_std(vs) for d, vs in by_d.items()}


def _pick_multiscale_rows(rows: List[Row], weights_filter: str) -> List[Row]:
    """
    If weights_filter provided (normalized string), keep only those rows.
    Otherwise: per-d, pick the weights group with best mean val_acc.
    """
    ms = [r for r in rows if r.kernel == "multiscale"]
    if not ms:
        return []

    if weights_filter:
        wf = _norm_weights_str(weights_filter)
        return [r for r in ms if r.weights == wf]

    # choose per-d based on best mean val_acc over weights groups
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
            m, _s = _mean_std([float(v) for v in vals])
            if m is None:
                continue
            if best_val is None or m > best_val:
                best_val = m
                best_w = w
        if best_w is None:
            # fallback: just take all rows for this d
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

def _plot_lines(
    xs: List[int],
    series: Dict[str, Dict[int, Tuple[Optional[float], Optional[float]]]],
    title: str,
    ylabel: str,
    out_path: Path,
):
    plt.figure(figsize=(8, 6))

    names = [n for n in PLOT_ORDER if n in series] + [n for n in series.keys() if n not in PLOT_ORDER]

    for name in names:
        by_d = series[name]

        ys: List[Optional[float]] = []
        yerr: List[Optional[float]] = []
        for d in xs:
            m, s = by_d.get(d, (None, None))
            ys.append(m)
            yerr.append(s)

        # plot only where we have values
        xs_plot = [x for x, y in zip(xs, ys) if y is not None]
        ys_plot = [y for y in ys if y is not None]
        if not xs_plot:
            continue

        yerr_plot = [e for y, e in zip(ys, yerr) if y is not None]

        style = _style_for_series(name)
        plt.errorbar(
            xs_plot,
            ys_plot,
            yerr=yerr_plot,
            label=name,
            linewidth=2.2,
            markersize=7,
            markeredgewidth=1.6,
            capsize=3,
            capthick=1.4,
            elinewidth=1.4,
            **style,
        )

    plt.title(title)
    plt.xlabel("d (qubits)")
    plt.ylabel(ylabel)
    plt.xticks(xs)
    plt.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend(frameon=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_curves_csv(out_csv: Path, xs: List[int], curves: Dict[str, Dict[str, Dict[int, Tuple[Optional[float], Optional[float]]]]]):
    """
    curves[kernel][metric][d] = (mean, std)
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = ["d"]
    for k in ["baseline", "local", "multiscale"]:
        for m in ["off_p50", "off_p95", "eff_rank", "val_acc", "test_acc"]:
            fields.append(f"{k}.{m}.mean")
            fields.append(f"{k}.{m}.std")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for d in xs:
            row: Dict[str, Any] = {"d": d}
            for k in ["baseline", "local", "multiscale"]:
                for m in ["off_p50", "off_p95", "eff_rank", "val_acc", "test_acc"]:
                    mean, std = curves.get(k, {}).get(m, {}).get(d, (None, None))
                    row[f"{k}.{m}.mean"] = "" if mean is None else f"{mean:.10g}"
                    row[f"{k}.{m}.std"] = "" if std is None else f"{std:.10g}"
            w.writerow(row)


def _run_for_dataset(
    all_rows: List[Row],
    dataset: str,
    out_dir: Path,
    feature_map: Optional[str],
    depth: Optional[int],
    entanglement: Optional[str],
    centered: Optional[bool],
    multiscale_weights: str,
    also_p95: bool,
):
    rows = _filter_common(all_rows, dataset, feature_map, depth, entanglement, centered)
    if not rows:
        print(f"[WARN] No rows matched for dataset={dataset} with the given filters.")
        return

    # Choose multiscale rows (fixed weights if provided; else best-by-val per d)
    ms_rows = _pick_multiscale_rows(rows, multiscale_weights)

    base_rows = [r for r in rows if r.kernel == "baseline"]
    loc_rows = [r for r in rows if r.kernel == "local"]

    xs = sorted({r.d for r in (base_rows + loc_rows + ms_rows)})
    if not xs:
        print(f"[WARN] No d-values found for dataset={dataset}.")
        return

    # Aggregate metrics
    curves: Dict[str, Dict[str, Dict[int, Tuple[Optional[float], Optional[float]]]]] = {
        "baseline": {
            "off_p50": _aggregate_by_d(base_rows, "off_p50"),
            "off_p95": _aggregate_by_d(base_rows, "off_p95"),
            "eff_rank": _aggregate_by_d(base_rows, "eff_rank"),
            "val_acc": _aggregate_by_d(base_rows, "val_acc"),
            "test_acc": _aggregate_by_d(base_rows, "test_acc"),
        },
        "local": {
            "off_p50": _aggregate_by_d(loc_rows, "off_p50"),
            "off_p95": _aggregate_by_d(loc_rows, "off_p95"),
            "eff_rank": _aggregate_by_d(loc_rows, "eff_rank"),
            "val_acc": _aggregate_by_d(loc_rows, "val_acc"),
            "test_acc": _aggregate_by_d(loc_rows, "test_acc"),
        },
        "multiscale": {
            "off_p50": _aggregate_by_d(ms_rows, "off_p50"),
            "off_p95": _aggregate_by_d(ms_rows, "off_p95"),
            "eff_rank": _aggregate_by_d(ms_rows, "eff_rank"),
            "val_acc": _aggregate_by_d(ms_rows, "val_acc"),
            "test_acc": _aggregate_by_d(ms_rows, "test_acc"),
        },
    }

    # Save curves CSV
    out_csv = out_dir / f"{dataset}_vs_d_curves.csv"
    _write_curves_csv(out_csv, xs, curves)

    # Plot 1: concentration p50 vs d
    _plot_lines(
        xs=xs,
        series={
            "baseline": curves["baseline"]["off_p50"],
            "local": curves["local"]["off_p50"],
            "multiscale": curves["multiscale"]["off_p50"],
        },
        title=f"{dataset}: concentration vs d (Off-diag p50)",
        ylabel="Off-diagonal median (p50)",
        out_path=out_dir / f"{dataset}_concentration_p50_vs_d.png",
    )

    # Optional: p95 vs d
    if also_p95:
        _plot_lines(
            xs=xs,
            series={
                "baseline": curves["baseline"]["off_p95"],
                "local": curves["local"]["off_p95"],
                "multiscale": curves["multiscale"]["off_p95"],
            },
            title=f"{dataset}: concentration tail vs d (Off-diag p95)",
            ylabel="Off-diagonal p95",
            out_path=out_dir / f"{dataset}_concentration_p95_vs_d.png",
        )

    # Plot 2: effective rank vs d
    _plot_lines(
        xs=xs,
        series={
            "baseline": curves["baseline"]["eff_rank"],
            "local": curves["local"]["eff_rank"],
            "multiscale": curves["multiscale"]["eff_rank"],
        },
        title=f"{dataset}: spectrum richness vs d (effective rank)",
        ylabel="Effective rank (entropy-based)",
        out_path=out_dir / f"{dataset}_effrank_vs_d.png",
    )

    # Plot 3: test acc vs d
    _plot_lines(
        xs=xs,
        series={
            "baseline": curves["baseline"]["test_acc"],
            "local": curves["local"]["test_acc"],
            "multiscale": curves["multiscale"]["test_acc"],
        },
        title=f"{dataset}: performance vs d (test accuracy)",
        ylabel="Test accuracy",
        out_path=out_dir / f"{dataset}_testacc_vs_d.png",
    )

    print(f"[OK] Wrote plots + curves CSV for dataset={dataset} into: {out_dir}")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, required=True, help="Path to summary_all.md (or summary.md).")
    ap.add_argument("--out", type=str, required=True, help="Output directory for plots.")
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
    ap.add_argument("--also-p95", action="store_true", help="Also write Off-diag p95 vs d plot.")

    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out)

    fmap = args.feature_map.strip() or None
    depth = args.depth if args.depth >= 0 else None
    ent = args.entanglement.strip() or None
    centered = None if args.centered < 0 else (True if args.centered == 1 else False)

    _header, rows_raw = _parse_md_table(summary_path)
    rows = _coerce_rows(rows_raw)

    if args.dataset.strip():
        datasets = [args.dataset.strip()]
    else:
        datasets = sorted({r.dataset for r in rows})

    for ds in datasets:
        _run_for_dataset(
            all_rows=rows,
            dataset=ds,
            out_dir=out_dir,
            feature_map=fmap,
            depth=depth,
            entanglement=ent,
            centered=centered,
            multiscale_weights=args.multiscale_weights,
            also_p95=args.also_p95,
        )


if __name__ == "__main__":
    main()



#
# Run
#
# python -m analysis.plot_vs_d --summary outputs/benchmarks/summary_all.md --out figs/checkpoint2/vs_d --also-p95
#
# Ex: only parkinsons with zz_qiskit, depth=1, centered=False, and fixed multiscale to [0.5, 0.5]
# python -m analysis.plot_vs_d --summary outputs/benchmarks/summary_all.md --out figs/checkpoint2/vs_d --dataset parkinsons --feature-map zz_qiskit --depth 1 --entanglement linear --centered 0 --multiscale-weights "[0.5, 0.5]" --also-p95
#
