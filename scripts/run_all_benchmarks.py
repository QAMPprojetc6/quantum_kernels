"""
Run one or more benchmark configs and then summarize results.

- Runs: scripts.run_experiment for each config
- Summarizes per-config roots: analysis.summarize_benchmarks -> summary.csv + summary.md
- Additionally writes a GLOBAL summary combining all executed configs.

Example (CLI):
  python -m scripts.run_all_benchmarks --configs configs/circles.toml configs/iris.toml

IDE-friendly:
  Edit DEFAULT_CONFIGS below and hit Run.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


# Convenient IDE default: edit this list and Run.
DEFAULT_CONFIGS = [
    "configs/circles.toml",
    "configs/iris.toml",
]


def _run_module(module: str, args: List[str]) -> None:
    cmd = [sys.executable, "-m", module] + args
    print(f"\n[RUN] {' '.join(cmd)}\n")
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(cmd, check=True, cwd=str(repo_root))

def _infer_root_from_config(config_path: Path) -> Path:
    """
    Mirror the out_dir convention used in the example configs:
      outputs/benchmarks/<dataset_id>

    We derive dataset_id from filename:
      circles.toml -> outputs/benchmarks/circles
      iris.toml    -> outputs/benchmarks/iris
    """
    name = config_path.stem.lower()
    # return Path("outputs/benchmarks") / name
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "outputs" / "benchmarks" / name



def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run benchmarks (run_experiment) + summarization (summarize_benchmarks), plus a global summary."
    )
    ap.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="List of TOML configs to run. If omitted, uses DEFAULT_CONFIGS inside the script.",
    )
    ap.add_argument(
        "--skip-summary",
        action="store_true",
        help="Run benchmarks only (do not build summary tables).",
    )
    ap.add_argument(
        "--global-out",
        default="outputs/benchmarks/summary_all.csv",
        help="Path for global CSV summary (combined across executed configs).",
    )
    ap.add_argument(
        "--global-md",
        default="outputs/benchmarks/summary_all.md",
        help="Optional path for global Markdown summary.",
    )
    args = ap.parse_args()

    configs = args.configs if args.configs and len(args.configs) > 0 else DEFAULT_CONFIGS

    # 1) Run benchmarks for each config
    for cfg in configs:
        _run_module("scripts.run_experiment", ["--config", cfg])

    # 2) Summarize each dataset output root + also create a global summary
    if not args.skip_summary:
        roots: List[Path] = []
        for cfg in configs:
            root = _infer_root_from_config(Path(cfg))
            roots.append(root)

            out_csv = root / "summary.csv"
            out_md = root / "summary.md"
            _run_module(
                "analysis.summarize_benchmarks",
                ["--root", str(root), "--out", str(out_csv), "--md", str(out_md)],
            )

        # Global summary over ALL executed configs:
        # just point summarize_benchmarks at outputs/benchmarks, since it will scan recursively.
        # global_root = Path("outputs/benchmarks")
        repo_root = Path(__file__).resolve().parents[1]
        global_root = repo_root / "outputs" / "benchmarks"

        repo_root = Path(__file__).resolve().parents[1]

        global_out_csv = Path(args.global_out)
        if not global_out_csv.is_absolute():
            global_out_csv = (repo_root / global_out_csv).resolve()

        global_out_md = Path(args.global_md) if args.global_md else None
        if global_out_md is not None and not global_out_md.is_absolute():
            global_out_md = (repo_root / global_out_md).resolve()

        global_args = ["--root", str(global_root), "--out", str(global_out_csv)]
        if global_out_md is not None:
            global_args += ["--md", str(global_out_md)]

        _run_module("analysis.summarize_benchmarks", global_args)

    print("\n[DONE] All requested benchmarks completed.\n")


if __name__ == "__main__":
    main()


# Run:
#
# python -m scripts.run_all_benchmarks --configs configs/circles.toml configs/iris.toml
#
# or
#
# run this file from IDE (check DEFAULT_CONFIGS)