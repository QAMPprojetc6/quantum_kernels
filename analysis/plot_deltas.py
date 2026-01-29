"""
Build delta tables (vs baseline) and generate comparison plots across kernels.

Inputs:
  - summary_all.csv or summary_all.md from analysis.summarize_benchmarks

Outputs (under --out directory):
  - delta_by_d.csv
  - delta_by_dataset.csv
  - per-dataset delta vs d plots
  - per-dataset tradeoff scatter plots
  - heatmaps of delta accuracy (local, multiscale)
  - barplot of mean delta accuracy across datasets

Notes:
  - If the summary file includes _out_prefix or _k_path (from summarize_benchmarks --include-paths),
    the script extracts seeds and reports 95% CI across seeds.
  - Without seed info, deltas are still computed but CI columns are left blank.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
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

    return header, rows


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".md":
        _, rows = _parse_md_table(path)
        return rows

    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


_SEED_RE = re.compile(r"_s(?P<seed>\d+)(?:_|$)")


def _extract_seed_from_str(s: str) -> Optional[int]:
    if not s:
        return None
    m = _SEED_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group("seed"))
    except ValueError:
        return None


def _extract_seed_from_row(r: Dict[str, str]) -> Optional[int]:
    for key in ("_out_prefix", "_k_path", "out_prefix", "kernel_path"):
        val = r.get(key, "")
        seed = _extract_seed_from_str(val)
        if seed is not None:
            return seed
    return None


@dataclass(frozen=True)
class Row:
    dataset: str
    kernel: str
    d: int
    fmap_name: Optional[str]
    fmap_depth: Optional[int]
    fmap_ent: Optional[str]
    centered: Optional[bool]
    seed: Optional[int]
    off_p50: Optional[float]
    off_p95: Optional[float]
    eff_rank: Optional[float]
    test_acc: Optional[float]


def _coerce_rows(rows: List[Dict[str, str]]) -> List[Row]:
    out: List[Row] = []
    for r in rows:
        dataset = (r.get("Dataset") or r.get("dataset") or "").strip()
        kernel = (r.get("Kernel") or r.get("kernel") or "").strip().lower()
        d = _to_int(r.get("d (qubits)", "") or r.get("d", ""))
        if dataset == "" or kernel == "" or d is None:
            continue

        fmap_cell = r.get("Feature map (name, depth, entanglement)", "") or r.get("Feature map", "")
        fmap_name, fmap_depth, fmap_ent = _parse_feature_map_cell(fmap_cell)
        centered = _to_bool(r.get("Centered?", "") or r.get("Centered", ""))

        off_p50 = _to_float(r.get("Off-diag p50", "") or r.get("Off-diag median", ""))
        off_p95 = _to_float(r.get("Off-diag p95", "") or r.get("Off-diag p95 / p50 / p5", ""))
        eff_rank = _to_float(r.get("Eff. rank (entropy)", "") or r.get("eff_rank_entropy", ""))
        test_acc = _to_float(r.get("Test acc", "") or r.get("test_acc", ""))

        seed = _extract_seed_from_row(r)
        out.append(
            Row(
                dataset=dataset,
                kernel=kernel,
                d=int(d),
                fmap_name=fmap_name,
                fmap_depth=fmap_depth,
                fmap_ent=fmap_ent,
                centered=centered,
                seed=seed,
                off_p50=off_p50,
                off_p95=off_p95,
                eff_rank=eff_rank,
                test_acc=test_acc,
            )
        )
    return out


def _match_kernel(name: str) -> Optional[str]:
    k = name.lower()
    if k.startswith("baseline"):
        return "baseline"
    if k.startswith("local"):
        return "local"
    if k.startswith("multiscale"):
        return "multiscale"
    return None


def _agg_mean(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return float(np.mean(xs))


def _aggregate(rows: List[Row]) -> Dict[Tuple, Dict[str, Optional[float]]]:
    groups: Dict[Tuple, Dict[str, List[Optional[float]]]] = {}
    for r in rows:
        k = _match_kernel(r.kernel)
        if k is None:
            continue
        key = (r.dataset, r.d, k, r.fmap_name, r.fmap_depth, r.fmap_ent, r.centered)
        if key not in groups:
            groups[key] = {
                "off_p50": [],
                "off_p95": [],
                "eff_rank": [],
                "test_acc": [],
            }
        groups[key]["off_p50"].append(r.off_p50)
        groups[key]["off_p95"].append(r.off_p95)
        groups[key]["eff_rank"].append(r.eff_rank)
        groups[key]["test_acc"].append(r.test_acc)

    agg: Dict[Tuple, Dict[str, Optional[float]]] = {}
    for key, vals in groups.items():
        agg[key] = {
            "off_p50": _agg_mean(vals["off_p50"]),
            "off_p95": _agg_mean(vals["off_p95"]),
            "eff_rank": _agg_mean(vals["eff_rank"]),
            "test_acc": _agg_mean(vals["test_acc"]),
        }
    return agg


def _aggregate_by_seed(rows: List[Row]) -> Dict[Tuple, Dict[str, Optional[float]]]:
    groups: Dict[Tuple, Dict[str, List[Optional[float]]]] = {}
    for r in rows:
        k = _match_kernel(r.kernel)
        if k is None:
            continue
        key = (r.dataset, r.d, k, r.fmap_name, r.fmap_depth, r.fmap_ent, r.centered, r.seed)
        if key not in groups:
            groups[key] = {
                "off_p50": [],
                "off_p95": [],
                "eff_rank": [],
                "test_acc": [],
            }
        groups[key]["off_p50"].append(r.off_p50)
        groups[key]["off_p95"].append(r.off_p95)
        groups[key]["eff_rank"].append(r.eff_rank)
        groups[key]["test_acc"].append(r.test_acc)

    agg: Dict[Tuple, Dict[str, Optional[float]]] = {}
    for key, vals in groups.items():
        agg[key] = {
            "off_p50": _agg_mean(vals["off_p50"]),
            "off_p95": _agg_mean(vals["off_p95"]),
            "eff_rank": _agg_mean(vals["eff_rank"]),
            "test_acc": _agg_mean(vals["test_acc"]),
        }
    return agg


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# -----------------------------
# Plot helpers
# -----------------------------

STYLE = {
    "baseline": dict(color="#000000", marker="o", linestyle="-"),
    "local": dict(color="#E69F00", marker="s", linestyle="--"),
    "multiscale": dict(color="#56B4E9", marker="^", linestyle="-."),
}


def _plot_delta_vs_d(
    out_dir: Path,
    dataset: str,
    d_vals: Sequence[int],
    deltas_local: Sequence[float],
    deltas_ms: Sequence[float],
    ci_local: Sequence[float],
    ci_ms: Sequence[float],
    metric_name: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        d_vals,
        deltas_local,
        yerr=ci_local,
        label="local",
        capsize=3,
        elinewidth=1.0,
        **STYLE["local"],
    )
    plt.errorbar(
        d_vals,
        deltas_ms,
        yerr=ci_ms,
        label="multiscale",
        capsize=3,
        elinewidth=1.0,
        **STYLE["multiscale"],
    )
    plt.axhline(0.0, color="#666666", linewidth=1, linestyle=":")
    plt.title(f"{dataset}: {metric_name} delta vs baseline")
    plt.xlabel("d (qubits)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"delta_{metric_name}_{dataset}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_tradeoff(
    out_dir: Path,
    dataset: str,
    points: Dict[str, List[Tuple[float, float]]],
) -> None:
    plt.figure(figsize=(6, 4))
    for k, pts in points.items():
        if not pts:
            continue
        xs, ys = zip(*pts)
        plt.scatter(xs, ys, label=k, **{**STYLE[k], "s": 30})
    plt.title(f"{dataset}: off-diag p50 vs test acc")
    plt.xlabel("off-diag p50")
    plt.ylabel("test acc")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"tradeoff_{dataset}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_bar_delta_acc(out_dir: Path, datasets: List[str], local_vals: List[float], ms_vals: List[float]) -> None:
    x = np.arange(len(datasets))
    width = 0.35
    plt.figure(figsize=(max(6, len(datasets) * 0.6), 4))
    plt.bar(x - width / 2, local_vals, width, label="local", color=STYLE["local"]["color"])
    plt.bar(x + width / 2, ms_vals, width, label="multiscale", color=STYLE["multiscale"]["color"])
    plt.axhline(0.0, color="#666666", linewidth=1, linestyle=":")
    plt.xticks(x, datasets, rotation=45, ha="right")
    plt.ylabel("mean delta test acc vs baseline")
    plt.title("Mean delta test acc by dataset")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / "bar_mean_delta_test_acc.png"
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_heatmap(
    out_dir: Path,
    dataset_names: List[str],
    d_vals: List[int],
    mat: np.ndarray,
    title: str,
    fname: str,
) -> None:
    plt.figure(figsize=(max(6, len(d_vals) * 0.5), max(4, len(dataset_names) * 0.35)))
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
    im = plt.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="delta test acc vs baseline (green=better, red=worse)")
    plt.yticks(range(len(dataset_names)), dataset_names)
    plt.xticks(range(len(d_vals)), d_vals)
    plt.title(title)
    plt.xlabel("d (qubits)")
    plt.ylabel("dataset")
    plt.tight_layout()
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=160)
    plt.close()


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build delta tables and kernel comparison plots.")
    ap.add_argument("--summary", required=True, help="Path to summary_all.csv or summary_all.md")
    ap.add_argument("--out", required=True, help="Output directory for tables and plots")
    ap.add_argument("--datasets", nargs="*", default=None, help="Optional dataset filter (exact names)")
    ap.add_argument("--feature-map", default=None, help="Filter by feature map name (e.g., zz_qiskit)")
    ap.add_argument("--depth", type=int, default=None, help="Filter by feature map depth")
    ap.add_argument("--entanglement", default=None, help="Filter by entanglement label")
    ap.add_argument("--centered", default=None, choices=["true", "false", "any"], help="Filter centered kernels")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _coerce_rows(_read_rows(summary_path))

    # Filters
    if args.datasets:
        ds_set = set(args.datasets)
        rows = [r for r in rows if r.dataset in ds_set]
    if args.feature_map is not None:
        rows = [r for r in rows if r.fmap_name == args.feature_map]
    if args.depth is not None:
        rows = [r for r in rows if r.fmap_depth == args.depth]
    if args.entanglement is not None:
        rows = [r for r in rows if r.fmap_ent == args.entanglement]
    if args.centered is not None and args.centered != "any":
        want = True if args.centered == "true" else False
        rows = [r for r in rows if r.centered == want]

    agg = _aggregate(rows)
    has_seed = any(r.seed is not None for r in rows)
    agg_seed = _aggregate_by_seed(rows) if has_seed else {}

    # Build baseline map
    baseline_map: Dict[Tuple, Dict[str, Optional[float]]] = {}
    for key, metrics in agg.items():
        dataset, d, kernel, fmap, depth, ent, centered = key
        if kernel == "baseline":
            baseline_map[(dataset, d, fmap, depth, ent, centered)] = metrics

    # Delta table by d
    delta_rows: List[Dict[str, Any]] = []
    for key, metrics in agg.items():
        dataset, d, kernel, fmap, depth, ent, centered = key
        if kernel not in {"local", "multiscale"}:
            continue
        base_key = (dataset, d, fmap, depth, ent, centered)
        base = baseline_map.get(base_key)
        if not base:
            continue
        # CI across seeds (if available)
        delta_ci = {
            "test_acc": None,
            "off_p50": None,
            "off_p95": None,
            "eff_rank": None,
        }
        if has_seed:
            deltas_by_seed = {m: [] for m in delta_ci.keys()}
            for seed_key, mvals in agg_seed.items():
                ds_s, d_s, k_s, fmap_s, depth_s, ent_s, cent_s, seed_s = seed_key
                if (ds_s, d_s, k_s, fmap_s, depth_s, ent_s, cent_s) != (dataset, d, kernel, fmap, depth, ent, centered):
                    continue
                base_s = agg_seed.get((dataset, d, "baseline", fmap, depth, ent, centered, seed_s))
                if not base_s:
                    continue
                for m in delta_ci.keys():
                    if mvals[m] is None or base_s[m] is None:
                        continue
                    deltas_by_seed[m].append(float(mvals[m] - base_s[m]))
            for m, vals in deltas_by_seed.items():
                if len(vals) >= 2:
                    delta_ci[m] = 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
                else:
                    delta_ci[m] = None
        row = {
            "Dataset": dataset,
            "d (qubits)": d,
            "Kernel": kernel,
            "Feature map": fmap,
            "Depth": depth,
            "Entanglement": ent,
            "Centered?": centered,
            "Delta test acc": None if metrics["test_acc"] is None or base["test_acc"] is None else metrics["test_acc"] - base["test_acc"],
            "Delta offdiag p50": None if metrics["off_p50"] is None or base["off_p50"] is None else metrics["off_p50"] - base["off_p50"],
            "Delta offdiag p95": None if metrics["off_p95"] is None or base["off_p95"] is None else metrics["off_p95"] - base["off_p95"],
            "Delta eff rank": None if metrics["eff_rank"] is None or base["eff_rank"] is None else metrics["eff_rank"] - base["eff_rank"],
            "CI95 Delta test acc": delta_ci["test_acc"],
            "CI95 Delta offdiag p50": delta_ci["off_p50"],
            "CI95 Delta offdiag p95": delta_ci["off_p95"],
            "CI95 Delta eff rank": delta_ci["eff_rank"],
            "Baseline test acc": base["test_acc"],
            "Baseline offdiag p50": base["off_p50"],
            "Baseline eff rank": base["eff_rank"],
        }
        delta_rows.append(row)

    delta_rows_sorted = sorted(delta_rows, key=lambda r: (r["Dataset"], r["Kernel"], int(r["d (qubits)"])))
    _write_csv(out_dir / "delta_by_d.csv", delta_rows_sorted, list(delta_rows_sorted[0].keys()) if delta_rows_sorted else [])

    # Delta by dataset (mean across d)
    by_ds: Dict[Tuple[str, str], List[float]] = {}
    for r in delta_rows:
        key = (r["Dataset"], r["Kernel"])
        val = r["Delta test acc"]
        if val is None:
            continue
        by_ds.setdefault(key, []).append(float(val))

    delta_ds_rows: List[Dict[str, Any]] = []
    datasets = sorted({r["Dataset"] for r in delta_rows})
    for ds in datasets:
        for k in ["local", "multiscale"]:
            vals = by_ds.get((ds, k), [])
            mean_val = float(np.mean(vals)) if vals else None
            delta_ds_rows.append(
                {
                    "Dataset": ds,
                    "Kernel": k,
                    "Mean delta test acc": mean_val,
                }
            )

    _write_csv(out_dir / "delta_by_dataset.csv", delta_ds_rows, ["Dataset", "Kernel", "Mean delta test acc"])

    # Plots: delta vs d per dataset
    datasets = sorted({r["Dataset"] for r in delta_rows})
    for ds in datasets:
        ds_rows = [r for r in delta_rows if r["Dataset"] == ds]
        d_vals = sorted({int(r["d (qubits)"]) for r in ds_rows})
        for metric_key, label in [
            ("Delta test acc", "delta test acc"),
            ("Delta offdiag p50", "delta offdiag p50"),
            ("Delta eff rank", "delta eff rank"),
        ]:
            local_vals = []
            ms_vals = []
            local_ci = []
            ms_ci = []
            for d in d_vals:
                local = [r for r in ds_rows if r["Kernel"] == "local" and int(r["d (qubits)"]) == d]
                ms = [r for r in ds_rows if r["Kernel"] == "multiscale" and int(r["d (qubits)"]) == d]
                local_vals.append(float(local[0][metric_key]) if local and local[0][metric_key] is not None else np.nan)
                ms_vals.append(float(ms[0][metric_key]) if ms and ms[0][metric_key] is not None else np.nan)
                local_ci.append(_safe_float(local[0].get("CI95 " + metric_key, 0.0)) if local else 0.0)
                ms_ci.append(_safe_float(ms[0].get("CI95 " + metric_key, 0.0)) if ms else 0.0)
            _plot_delta_vs_d(out_dir, ds, d_vals, local_vals, ms_vals, local_ci, ms_ci, metric_key.replace(" ", "_").lower(), label)

    # Tradeoff scatter per dataset (offdiag p50 vs test acc for each kernel)
    # Use aggregated metrics (not deltas)
    agg_rows: List[Dict[str, Any]] = []
    for key, metrics in agg.items():
        dataset, d, kernel, fmap, depth, ent, centered = key
        if kernel not in {"baseline", "local", "multiscale"}:
            continue
        agg_rows.append(
            {
                "Dataset": dataset,
                "d": d,
                "Kernel": kernel,
                "off_p50": metrics["off_p50"],
                "test_acc": metrics["test_acc"],
            }
        )
    for ds in sorted({r["Dataset"] for r in agg_rows}):
        pts = {k: [] for k in ["baseline", "local", "multiscale"]}
        for r in agg_rows:
            if r["Dataset"] != ds:
                continue
            if r["off_p50"] is None or r["test_acc"] is None:
                continue
            pts[r["Kernel"]].append((float(r["off_p50"]), float(r["test_acc"])))
        _plot_tradeoff(out_dir, ds, pts)

    # Barplot of mean delta test acc across datasets
    ds_names = sorted({r["Dataset"] for r in delta_ds_rows})
    local_vals = []
    ms_vals = []
    for ds in ds_names:
        l = [r for r in delta_ds_rows if r["Dataset"] == ds and r["Kernel"] == "local"]
        m = [r for r in delta_ds_rows if r["Dataset"] == ds and r["Kernel"] == "multiscale"]
        local_vals.append(float(l[0]["Mean delta test acc"]) if l and l[0]["Mean delta test acc"] is not None else 0.0)
        ms_vals.append(float(m[0]["Mean delta test acc"]) if m and m[0]["Mean delta test acc"] is not None else 0.0)
    if ds_names:
        _plot_bar_delta_acc(out_dir, ds_names, local_vals, ms_vals)

    # Heatmaps of delta test acc
    d_vals = sorted({int(r["d (qubits)"]) for r in delta_rows})
    ds_names = sorted({r["Dataset"] for r in delta_rows})
    if ds_names and d_vals:
        mat_local = np.full((len(ds_names), len(d_vals)), np.nan, dtype=float)
        mat_ms = np.full((len(ds_names), len(d_vals)), np.nan, dtype=float)
        for r in delta_rows:
            i = ds_names.index(r["Dataset"])
            j = d_vals.index(int(r["d (qubits)"]))
            if r["Delta test acc"] is None:
                continue
            if r["Kernel"] == "local":
                mat_local[i, j] = float(r["Delta test acc"])
            elif r["Kernel"] == "multiscale":
                mat_ms[i, j] = float(r["Delta test acc"])
        _plot_heatmap(out_dir, ds_names, d_vals, mat_local, "Delta test acc (local vs baseline)", "heatmap_delta_test_acc_local.png")
        _plot_heatmap(out_dir, ds_names, d_vals, mat_ms, "Delta test acc (multiscale vs baseline)", "heatmap_delta_test_acc_multiscale.png")


if __name__ == "__main__":
    main()

# ---------------------------
# Examples
# ---------------------------
#
# python -m analysis.plot_deltas --summary outputs/benchmarks/summary_all.csv --out figs/checkpoint3/deltas
#
# Optional filters (if needed)
# --datasets breast_cancer parkinsons
# --feature-map zz_qiskit
# --depth 1
# --entanglement linear
# --centered false
