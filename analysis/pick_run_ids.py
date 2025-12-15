"""
Pick best matched runs (baseline/local/multiscale) for a dataset block and print
run_ids + K.npy paths to use in figures (Checkpoint 2).

This script expects:
- A global summary file (summary_all.md OR summary_all.csv) produced by summarize_benchmarks.py
- Per-dataset metrics.csv produced by run_experiment.py, typically at:
    outputs/benchmarks/<dataset>/metrics.csv

Example:
python -m analysis.pick_run_ids \
  --summary outputs/benchmarks/summary_all.md \
  --bench-root outputs/benchmarks \
  --dataset breast_cancer \
  --feature-map zz_manual_canonical \
  --depth 1 \
  --entanglement ring \
  --centered 0 \
  --pick-by test
"""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Parsing utilities
# ----------------------------

def _to_bool(s: str) -> bool:
    s = (s or "").strip().lower()
    return s in {"1", "true", "yes", "y"}


def _to_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _norm_feature_map_cell(cell: str) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Parses strings like: "zz_manual_canonical, depth=1, ent=ring"
    Returns: (name, depth, entanglement)
    """
    cell = (cell or "").strip()
    name = cell.split(",")[0].strip() if cell else ""
    depth_m = re.search(r"depth\s*=\s*(\d+)", cell)
    ent_m = re.search(r"ent\s*=\s*([A-Za-z0-9_\-]+)", cell)
    depth = int(depth_m.group(1)) if depth_m else None
    ent = ent_m.group(1) if ent_m else None
    return name, depth, ent


def _parse_md_table(path: Path) -> List[Dict[str, str]]:
    """
    Parses the first Markdown table in the file.

    Accepts alignment separators like:
      | --- | ---: | :---: | --- |
    """
    import re

    text = path.read_text(encoding="utf-8")
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    # A markdown table separator line: pipes + dashes + optional colons/spaces
    sep_re = re.compile(r"^\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|$")

    start = None
    for i in range(len(lines) - 1):
        header_ln = lines[i].strip()
        sep_ln = lines[i + 1].strip()
        if header_ln.startswith("|") and header_ln.count("|") >= 3 and sep_re.match(sep_ln):
            start = i
            break

    if start is None:
        raise ValueError(f"No markdown table found in: {path}")

    header = [c.strip() for c in lines[start].strip().strip("|").split("|")]
    rows: List[Dict[str, str]] = []

    for ln in lines[start + 2 :]:
        if not ln.strip().startswith("|"):
            break
        parts = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(parts) != len(header):
            continue
        rows.append({header[j]: parts[j] for j in range(len(header))})

    return rows


def _parse_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _load_summary(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".md":
        return _parse_md_table(path)
    if path.suffix.lower() == ".csv":
        return _parse_csv(path)
    raise ValueError("Unsupported summary format. Use .md or .csv")


def _metrics_row_to_kernel_path(r: Dict[str, str]) -> Optional[str]:
    """
    Convert a metrics.csv row into a concrete K.npy path.

    Supports two schemas:
      - old: has 'kernel_path'
      - current: has 'out_prefix' (prefix for artifacts), plus 'center' flag
    """
    kp = (r.get("kernel_path") or "").strip()
    if kp:
        return kp

    out_prefix = (r.get("out_prefix") or "").strip()
    if not out_prefix:
        return None

    # metrics.csv stores center as True/False (string)
    center_str = (r.get("center") or "").strip().lower()
    is_centered = center_str in {"true", "1", "yes"}

    if is_centered:
        return out_prefix + "_K_centered.npy"
    return out_prefix + "_K.npy"


# ----------------------------
# Metrics resolution
# ----------------------------

def _read_metrics(metrics_csv: Path) -> List[Dict[str, str]]:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _run_id_from_kernel_path(kernel_path: str) -> str:
    """
    kernel_path: ".../<run_id>_K.npy"
    """
    p = Path(kernel_path)
    stem = p.stem  # "<run_id>_K"
    if stem.endswith("_K"):
        return stem[:-2]
    return stem


def _float_close(a: Optional[float], b: Optional[float], tol: float = 1e-3) -> bool:
    """
    Summary tables often round (e.g., 0.781) while metrics.csv stores full precision
    (e.g., 0.780701...). Use a looser tolerance.
    """
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _resolve_kernel_path_via_metrics(
    metrics_rows: List[Dict[str, str]],
    kernel_name: str,
    val_acc: Optional[float],
    test_acc: Optional[float],
    best_c: Optional[float],
) -> Optional[str]:
    """
    Find the metrics.csv row corresponding to the selected summary row.

    Supports the current metrics schema where artifacts are referenced by:
      - out_prefix (+ '_K.npy' or '_K_centered.npy')
    """

    kname = kernel_name.strip().lower()

    # 0) Prefer explicit kernel column match if present
    by_kernel = [r for r in metrics_rows if (r.get("kernel") or "").strip().lower() == kname]
    pool = by_kernel if by_kernel else metrics_rows

    # 1) If val/test are missing in summary, pick best test_acc in pool
    if val_acc is None or test_acc is None:
        best_score = -1.0
        best_kp: Optional[str] = None
        for r in pool:
            kp = _metrics_row_to_kernel_path(r)
            if not kp:
                continue
            rt = _to_float(r.get("test_acc", ""))
            if rt is None:
                continue
            # tiny bonus if filename hints kernel (optional)
            bonus = 1e-6 if kname in Path(kp).name.lower() else 0.0
            score = rt + bonus
            if score > best_score:
                best_score = score
                best_kp = kp
        return best_kp

    # 2) Otherwise, match closest (test_acc, val_acc), with optional best_C
    best_dist = math.inf
    best_kp: Optional[str] = None

    for r in pool:
        kp = _metrics_row_to_kernel_path(r)
        if not kp:
            continue

        rv = _to_float(r.get("val_acc", ""))
        rt = _to_float(r.get("test_acc", ""))
        rc = _to_float(r.get("best_C", ""))

        if rv is None or rt is None:
            continue

        dist = abs(rt - test_acc) + 0.5 * abs(rv - val_acc)

        if best_c is not None and rc is not None:
            dist += 0.01 * abs(rc - best_c)

        # tiny preference for filename hint
        if kname in Path(kp).name.lower():
            dist -= 1e-6

        if dist < best_dist:
            best_dist = dist
            best_kp = kp

    return best_kp




# ----------------------------
# Selection logic
# ----------------------------

def _pick_best(rows: List[Dict[str, str]], pick_by: str) -> Dict[str, str]:
    """
    pick_by: "test" or "alignment"
    """
    if not rows:
        raise ValueError("No rows to pick from.")

    def key(r: Dict[str, str]) -> Tuple[float, float]:
        # Higher is better
        test = _to_float(r.get("Test acc", "")) or -1.0
        val = _to_float(r.get("Val acc", "")) or -1.0
        align = _to_float(r.get("Alignment (centered)", "")) or -1.0

        if pick_by == "alignment":
            # prioritize alignment, break ties by test then val
            return (align, test + 1e-6 * val)
        # default: prioritize test, break ties by val
        return (test, val)

    return sorted(rows, key=key, reverse=True)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick run_ids/K paths for Checkpoint 2 figures.")
    ap.add_argument("--summary", required=True, help="Path to summary_all.md or summary_all.csv")
    ap.add_argument("--bench-root", default="outputs/benchmarks", help="Root folder containing per-dataset outputs.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--feature-map", required=True, help="Feature map name (e.g., zz_qiskit, zz_manual_canonical)")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--entanglement", default=None, help="Entanglement label to match (e.g., linear, ring)")
    ap.add_argument("--centered", type=int, choices=[0, 1], default=0)
    ap.add_argument("--pick-by", choices=["test", "alignment"], default="test",
                    help="Criterion to pick the representative run per kernel.")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    bench_root = Path(args.bench_root)

    rows = _load_summary(summary_path)

    centered_bool = bool(args.centered)
    target_fm = args.feature_map.strip()

    # Filter to "apples-to-apples" block
    block: List[Dict[str, str]] = []
    for r in rows:
        if (r.get("Dataset", "") or "").strip() != args.dataset:
            continue

        fm_name, fm_depth, fm_ent = _norm_feature_map_cell(r.get("Feature map (name, depth, entanglement)", ""))
        if fm_name != target_fm:
            continue
        if fm_depth is not None and fm_depth != args.depth:
            continue
        if args.entanglement is not None:
            if (fm_ent or "").strip() != args.entanglement:
                continue

        if _to_bool(r.get("Centered?", "")) != centered_bool:
            continue

        block.append(r)

    if not block:
        raise ValueError(
            "No rows matched your block. Check --dataset/--feature-map/--depth/--entanglement/--centered."
        )

    # Split by kernel
    by_kernel: Dict[str, List[Dict[str, str]]] = {"baseline": [], "local": [], "multiscale": []}
    for r in block:
        k = (r.get("Kernel", "") or "").strip().lower()
        if k in by_kernel:
            by_kernel[k].append(r)

    # Pick best per kernel
    picks: Dict[str, Dict[str, str]] = {}
    for k in ["baseline", "local", "multiscale"]:
        if not by_kernel[k]:
            continue
        picks[k] = _pick_best(by_kernel[k], pick_by=args.pick_by)

    # Resolve K.npy paths via metrics.csv
    metrics_csv = bench_root / args.dataset / "metrics.csv"
    metrics_rows = _read_metrics(metrics_csv)

    resolved: Dict[str, Dict[str, str]] = {}
    for k, row in picks.items():
        val = _to_float(row.get("Val acc", ""))
        test = _to_float(row.get("Test acc", ""))
        best_c = _to_float(row.get("SVM best C", ""))

        kp = _resolve_kernel_path_via_metrics(metrics_rows, k, val, test, best_c)
        if not kp:
            raise RuntimeError(f"Could not resolve kernel_path for {k} via {metrics_csv}")

        run_id = _run_id_from_kernel_path(kp)
        resolved[k] = {"run_id": run_id, "kernel_path": kp}

    # Print results + ready command
    print("\n[OK] Selected representative runs:")
    for k in ["baseline", "local", "multiscale"]:
        if k not in resolved:
            continue
        row = picks[k]
        print(f"\n- {k.upper()}")
        print(f"  run_id:     {resolved[k]['run_id']}")
        print(f"  K.npy:      {resolved[k]['kernel_path']}")
        print(f"  val/test:   {row.get('Val acc','')} / {row.get('Test acc','')}")
        print(f"  best_C:     {row.get('SVM best C','')}")
        if k == "local":
            print(f"  patches:    {row.get('Scales / Patches','')}")
        if k == "multiscale":
            print(f"  scales:     {row.get('Scales / Patches','')}")
            print(f"  weights:    {row.get('Weights','')}")

    out_png = f"figs/checkpoint2/{args.dataset}_compare.png"
    title = f"{args.dataset} (d={block[0].get('d (qubits)','?')}, centered={centered_bool}): Baseline vs Local vs Multi-Scale"

    print("\nCommand to make a 2x3 comparison figure:")
    print(
        "python -m analysis.make_checkpoint2_figure "
        f"--baseline \"{resolved['baseline']['kernel_path']}\" "
        f"--local \"{resolved['local']['kernel_path']}\" "
        f"--multiscale \"{resolved['multiscale']['kernel_path']}\" "
        f"--out \"{out_png}\" "
        f"--title \"{title}\""
    )


if __name__ == "__main__":
    main()


#
# RUn
#
# python -m analysis.pick_run_ids --summary outputs/benchmarks/summary_all.md --bench-root outputs/benchmarks --dataset breast_cancer --feature-map zz_manual_canonical --depth 1 --entanglement ring --centered 0 --pick-by test
#
# python -m analysis.pick_run_ids --summary outputs/benchmarks/summary_all.md --bench-root outputs/benchmarks --dataset parkinsons --feature-map zz_qiskit --depth 1 --entanglement linear --centered 0 --pick-by test
#

