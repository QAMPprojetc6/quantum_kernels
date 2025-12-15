"""
Make a 2x3 figure comparing Baseline vs Local vs Multi-Scale for one dataset.

Top row: off-diagonal histogram
Bottom row: eigen-spectrum

Example:
python -m analysis.make_checkpoint2_figure \
  --baseline outputs/benchmarks/breast_cancer/<RUNID_baseline>_K.npy \
  --local outputs/benchmarks/breast_cancer/<RUNID_local>_K.npy \
  --multiscale outputs/benchmarks/breast_cancer/<RUNID_ms>_K.npy \
  --out figs/checkpoint2/breast_cancer_compare.png \
  --title "Breast Cancer (d=8, uncentered): Baseline vs Local vs Multi-Scale"
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


def _load(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    K = np.load(p)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kernel must be square, got shape {K.shape} from {p}")
    return K.astype(np.float64, copy=False)


def _offdiag_vals(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return K[mask]


def _spectrum(K: np.ndarray) -> np.ndarray:
    Ks = 0.5 * (K + K.T)
    w = np.linalg.eigvalsh(Ks)
    return np.sort(w)[::-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Make checkpoint 2 comparison figure (2x3).")
    ap.add_argument("--baseline", required=True, help="Path to baseline *_K.npy")
    ap.add_argument("--local", required=True, help="Path to local *_K.npy")
    ap.add_argument("--multiscale", required=True, help="Path to multiscale *_K.npy")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="Baseline vs Local vs Multi-Scale", help="Figure title")
    ap.add_argument("--bins", type=int, default=50, help="Histogram bins")
    args = ap.parse_args()

    paths: Dict[str, str] = {
        "Baseline": args.baseline,
        "Local": args.local,
        "Multi-Scale": args.multiscale,
    }
    Ks = {name: _load(p) for name, p in paths.items()}

    # Shared histogram range for fair visual comparison
    all_off = np.concatenate([_offdiag_vals(K) for K in Ks.values()])
    vmin, vmax = float(np.min(all_off)), float(np.max(all_off))

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(args.title, y=0.98)

    # Row 1: off-diagonal hist
    for j, (name, K) in enumerate(Ks.items(), start=1):
        ax = fig.add_subplot(2, 3, j)
        vals = _offdiag_vals(K)
        ax.hist(vals, bins=args.bins, range=(vmin, vmax))
        ax.set_title(f"{name}: off-diag hist")
        ax.set_xlabel("K_ij (i≠j)")
        ax.set_ylabel("count")

    # Row 2: spectrum
    for j, (name, K) in enumerate(Ks.items(), start=1):
        ax = fig.add_subplot(2, 3, 3 + j)
        w = _spectrum(K)
        ax.plot(w, marker="o", linewidth=1)
        ax.set_title(f"{name}: spectrum")
        ax.set_xlabel("index")
        ax.set_ylabel("eigenvalue")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
