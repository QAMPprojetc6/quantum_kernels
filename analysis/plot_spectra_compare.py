"""
Plot baseline/local/multiscale eigen-spectra for a fixed d, with mean±std bands across seeds.

Inputs:
  - summary_all.csv or summary_all.md from analysis.summarize_benchmarks (ideally with --include-paths)

Outputs:
  - one plot per dataset (baseline/local/multiscale curves with std bands)
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    k_path: str
    out_prefix: str


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

        k_path = (r.get("_k_path") or r.get("kernel_path") or "").strip()
        out_prefix = (r.get("_out_prefix") or r.get("out_prefix") or "").strip()

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
                k_path=k_path,
                out_prefix=out_prefix,
            )
        )
    return out


def _kernel_path_from_row(r: Row) -> Optional[Path]:
    if r.k_path:
        return Path(r.k_path)
    if r.out_prefix:
        suffix = "_K_centered.npy" if r.centered else "_K.npy"
        return Path(r.out_prefix + suffix)
    return None


def _pick_multiscale_rows(rows: List[Row], weights_filter: str) -> List[Row]:
    ms = [r for r in rows if r.kernel == "multiscale"]
    if not ms:
        return []

    if weights_filter:
        wf = _norm_weights_str(weights_filter)
        return [r for r in ms if r.weights == wf]

    by_w: Dict[str, List[Row]] = {}
    for r in ms:
        by_w.setdefault(r.weights, []).append(r)

    best_w = None
    best_val = None
    for w, rs in by_w.items():
        vals = [v for v in (x.val_acc for x in rs) if v is not None]
        if not vals:
            continue
        m = float(np.mean(vals))
        if best_val is None or m > best_val:
            best_val = m
            best_w = w

    if best_w is None:
        return ms
    return by_w[best_w]


def _filter_common(
    rows: List[Row],
    dataset: str,
    d: int,
    feature_map: Optional[str],
    depth: Optional[int],
    entanglement: Optional[str],
    centered: Optional[bool],
) -> List[Row]:
    out = [r for r in rows if r.dataset == dataset and r.d == d]
    if feature_map is not None:
        out = [r for r in out if r.fmap_name == feature_map]
    if depth is not None:
        out = [r for r in out if r.fmap_depth == depth]
    if entanglement is not None:
        out = [r for r in out if r.fmap_ent == entanglement]
    if centered is not None:
        out = [r for r in out if r.centered == centered]
    return out


def _spectra_stats(
    rows: List[Row],
    max_eigs: Optional[int],
    normalize_trace: bool,
    skip_first: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    spectra: List[np.ndarray] = []
    for r in rows:
        k_path = _kernel_path_from_row(r)
        if k_path is None:
            continue
        if not k_path.exists():
            continue
        K = np.load(k_path)
        Ks = 0.5 * (K + K.T)
        w = np.linalg.eigvalsh(Ks)
        w = np.sort(w)[::-1]
        if skip_first > 0:
            if skip_first >= len(w):
                continue
            w = w[skip_first:]
        if normalize_trace:
            s = float(np.sum(w))
            if s > 0.0:
                w = w / s
        spectra.append(w)

    if not spectra:
        return None

    min_len = min(len(w) for w in spectra)
    if max_eigs is not None and max_eigs > 0:
        min_len = min(min_len, int(max_eigs))

    stack = np.stack([w[:min_len] for w in spectra], axis=0)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    return mean, std


STYLE = {
    "baseline": dict(color="#000000", label="baseline", linestyle="-", marker="o", markerfacecolor="#000000"),
    "local": dict(color="#E69F00", label="local", linestyle="--", marker="s", markerfacecolor="none"),
    "multiscale": dict(color="#56B4E9", label="multiscale", linestyle="-.", marker="^", markerfacecolor="none"),
}


def _plot_one(
    dataset: str,
    d: int,
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    max_eigs: Optional[int],
    normalize_trace: bool,
    skip_first: int,
    log_scale: bool,
) -> None:
    plt.figure(figsize=(8, 5))
    for k in ["baseline", "local", "multiscale"]:
        if k not in stats:
            continue
        mean, std = stats[k]
        x = np.arange(1, len(mean) + 1)
        style = STYLE[k]
        if log_scale:
            eps = 1e-12
            mean_plot = np.clip(mean, eps, None)
            low = np.clip(mean - std, eps, None)
            high = np.clip(mean + std, eps, None)
            plt.semilogy(
                x,
                mean_plot,
                color=style["color"],
                label=style["label"],
                linewidth=2.0,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4,
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=1.0,
                markevery=max(1, len(x) // 25),
            )
            plt.fill_between(x, low, high, color=style["color"], alpha=0.2)
        else:
            plt.plot(
                x,
                mean,
                color=style["color"],
                label=style["label"],
                linewidth=2.0,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4,
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=1.0,
                markevery=max(1, len(x) // 25),
            )
            plt.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.2)

    title = f"{dataset}: eigen-spectrum (d={d})"
    if max_eigs is not None and max_eigs > 0:
        title += f", top {max_eigs}"
    if normalize_trace:
        title += " (trace-normalized)"
    if skip_first > 0:
        title += f", skip first {skip_first}"
    if log_scale:
        title += ", log-scale"
    plt.title(title)
    plt.xlabel("rank (sorted eigenvalues)")
    plt.ylabel("eigenvalue")
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_d{d}_spectrum_compare.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot baseline/local/multiscale spectra for a fixed d.")
    ap.add_argument("--summary", required=True, help="Path to summary_all.csv or summary_all.md")
    ap.add_argument("--out", required=True, help="Output directory for plots")
    ap.add_argument("--d", type=int, required=True, help="Target d (qubits)")
    ap.add_argument("--dataset", type=str, default="", help="Single dataset to plot")
    ap.add_argument("--datasets", nargs="*", default=None, help="Optional dataset list (overrides --dataset)")
    ap.add_argument("--feature-map", type=str, default="", help="Filter by feature map name (e.g., zz_qiskit)")
    ap.add_argument("--depth", type=int, default=-1, help="Filter by depth (e.g., 1)")
    ap.add_argument("--entanglement", type=str, default="", help="Filter by entanglement (e.g., linear/ring)")
    ap.add_argument("--centered", type=int, default=-1, help="Filter by centered: 0/1 (default: no filter)")
    ap.add_argument("--multiscale-weights", type=str, default="", help='Fix multiscale weights, e.g. "[0.5, 0.5]"')
    ap.add_argument("--max-eigs", type=int, default=-1, help="Optional max number of eigenvalues to plot")
    ap.add_argument("--normalize-trace", action="store_true", help="Normalize each spectrum by its trace to compare shape.")
    ap.add_argument("--skip-first", type=int, default=0, help="Skip the first N largest eigenvalues.")
    ap.add_argument("--log-scale", action="store_true", help="Use log-scale (semilogy) on the y-axis.")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out)

    fmap = args.feature_map.strip() or None
    depth = args.depth if args.depth >= 0 else None
    ent = args.entanglement.strip() or None
    centered = None if args.centered < 0 else (True if args.centered == 1 else False)
    max_eigs = args.max_eigs if args.max_eigs and args.max_eigs > 0 else None
    normalize_trace = bool(args.normalize_trace)
    skip_first = max(0, int(args.skip_first))
    log_scale = bool(args.log_scale)

    rows_raw = _read_rows(summary_path)
    rows = _coerce_rows(rows_raw)

    if args.datasets:
        datasets = list(args.datasets)
    elif args.dataset.strip():
        datasets = [args.dataset.strip()]
    else:
        datasets = sorted({r.dataset for r in rows})

    for ds in datasets:
        block = _filter_common(rows, ds, args.d, fmap, depth, ent, centered)
        if not block:
            print(f"[WARN] No rows matched for dataset={ds} d={args.d}")
            continue

        base_rows = [r for r in block if r.kernel == "baseline"]
        loc_rows = [r for r in block if r.kernel == "local"]
        ms_rows = _pick_multiscale_rows(block, args.multiscale_weights)

        stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        base_stats = _spectra_stats(base_rows, max_eigs, normalize_trace, skip_first)
        if base_stats is not None:
            stats["baseline"] = base_stats
        loc_stats = _spectra_stats(loc_rows, max_eigs, normalize_trace, skip_first)
        if loc_stats is not None:
            stats["local"] = loc_stats
        ms_stats = _spectra_stats(ms_rows, max_eigs, normalize_trace, skip_first)
        if ms_stats is not None:
            stats["multiscale"] = ms_stats

        if not stats:
            print(f"[WARN] No spectra loaded for dataset={ds} d={args.d}")
            continue

        missing = [k for k in ["baseline", "local", "multiscale"] if k not in stats]
        if missing:
            print(f"[WARN] Missing kernels for {ds} d={args.d}: {', '.join(missing)}")

        _plot_one(ds, args.d, stats, out_dir, max_eigs, normalize_trace, skip_first, log_scale)
        print(f"[OK] Wrote spectra plot for dataset={ds} d={args.d}")


if __name__ == "__main__":
    main()


# Examples:
# python -m analysis.plot_spectra_compare --summary outputs/benchmarks/summary_all.csv --out figs/checkpoint3/spectra --d 12 --dataset breast_cancer
# python -m analysis.plot_spectra_compare --summary outputs/benchmarks/summary_all.csv --out figs/checkpoint3/spectra --d 12 --datasets breast_cancer parkinsons ionosphere --max-eigs 200
