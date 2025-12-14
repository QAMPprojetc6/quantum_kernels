"""
Diagnostics for precomputed kernel matrices (Gram matrices).

Produces three standard plots:
  - Heatmap of K (matrix structure)
  - Off-diagonal histogram (concentration / contrast)
  - Eigen-spectrum of symmetrized K (effective rank proxy)

This script is kernel-agnostic: it works for baseline, local, and multi-scale
as long as you provide a saved K.npy.

CLI:
    python analysis/diagnostics.py \
      --kernel outputs/K_baseline-make_circles_42.npy \
      --save-prefix figs/baseline-make_circles_42

Notes:
- If a centered-kernel diagnostics is needed, pass a centered K (e.g., K_centered.npy).
- Spectrum is computed on Ks = 0.5*(K + K.T) to avoid tiny numerical asymmetry.
"""


import argparse
import json
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(K: np.ndarray, out_path: str) -> None:
    plt.figure()
    plt.imshow(K, aspect="auto")
    plt.colorbar()
    plt.title("Kernel matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_offdiag_hist(K: np.ndarray, out_path: str) -> None:
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = K[mask]
    plt.figure()
    plt.hist(vals, bins=50)
    plt.title("Off-diagonal histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_spectrum(K: np.ndarray, out_path: str) -> None:
    # Use symmetric to avoid tiny asymmetry
    Ks = 0.5 * (K + K.T)
    w = np.linalg.eigvalsh(Ks)
    w_sorted = np.sort(w)[::-1]
    plt.figure()
    plt.plot(w_sorted, marker="o")
    plt.title("Eigen-spectrum")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main(kernel_path: str, save_prefix: str, meta_path: Optional[str] = None) -> None:
    K = np.load(kernel_path)
    out_dir = os.path.dirname(save_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    plot_heatmap(K, f"{save_prefix}_matrix.png")
    plot_offdiag_hist(K, f"{save_prefix}_offdiag_hist.png")
    plot_spectrum(K, f"{save_prefix}_spectrum.png")

    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        # Save a short text summary next to figures (optional)
        with open(f"{save_prefix}_meta.txt", "w") as f:
            f.write(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel diagnostics")
    parser.add_argument("--kernel", required=True, help="Path to K.npy")
    parser.add_argument("--save-prefix", required=True, help="Prefix for output figures")
    parser.add_argument("--meta", default=None, help="Optional path to meta_*.json")
    args = parser.parse_args()
    main(args.kernel, args.save_prefix, args.meta)
