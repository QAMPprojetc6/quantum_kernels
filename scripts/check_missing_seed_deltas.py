"""
Check summary_all.csv for dataset/d/kernel groups with missing seed coverage.

If nothing is printed, every group has >=2 seeds available.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


SEED_RE = re.compile(r"_s(\d+)(?:_|$)")


def _get_seed(s: str | None) -> int | None:
    if not s:
        return None
    m = SEED_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Check missing seed coverage in summary_all.csv")
    ap.add_argument(
        "--summary",
        default="outputs/benchmarks/summary_all.csv",
        help="Path to summary_all.csv (must include _out_prefix/_k_path columns).",
    )
    args = ap.parse_args()

    path = Path(args.summary)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    seeds = defaultdict(set)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row.get("Dataset", "")
            d = row.get("d (qubits)", "")
            kernel = (row.get("Kernel", "") or "").lower()
            seed = _get_seed(row.get("_out_prefix") or row.get("_k_path"))
            if seed is None:
                continue
            key = (ds, d, kernel)
            seeds[key].add(seed)

    missing = [(k, v) for k, v in seeds.items() if len(v) < 2]
    if not missing:
        return

    for (ds, d, kernel), seed_set in sorted(missing):
        print(f"missing seeds: dataset={ds} d={d} kernel={kernel} seeds={sorted(seed_set)}")


if __name__ == "__main__":
    main()

# Run:
# python -m scripts.check_missing_seed_deltas --summary outputs/benchmarks/summary_all.csv
