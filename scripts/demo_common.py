"""
Shared helpers for demo scripts.

This module factors out the duplicated logic between:
  - scripts/run_multiscale_demo.py
  - scripts/run_baseline_demo.py

Important:
- Keep behavior identical to the original scripts: same preprocessing, splits,
  centering, spectrum reporting, artifact naming, plotting, and CSV schema.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


# ---------------------------
# data helpers
# ---------------------------
def load_dataset(
    name: str,
    n_samples: int,
    seed: int,
    n_features: Optional[int] = None,
    pca: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess a small dataset.

    Supported:
      - make_circles (synthetic, uses n_samples)
      - iris (full dataset)
      - breast_cancer (full dataset)
      - parkinsons (OpenML id 1488; downloads on first use)

    Preprocessing:
      - shuffle with RNG(seed)
      - StandardScaler
      - optional dimensionality reduction to n_features (PCA or simple truncation)
      - map to radians: X <- pi * X / 2
    """
    from sklearn.datasets import make_circles, load_iris, load_breast_cancer, fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)

    if name == "make_circles":
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.1, random_state=seed)

    elif name == "iris":
        iris = load_iris()
        X, y = iris.data, iris.target

    elif name in {"breast_cancer", "breast-cancer", "cancer"}:
        bc = load_breast_cancer()
        X, y = bc.data, bc.target  # binary

    elif name in {"parkinsons", "parkinson", "parkinson_disease", "parkinson-disease"}:
        # OpenML Parkinsons dataset (UCI voice measures), target is "status" (0/1)
        ds = fetch_openml(data_id=1488, as_frame=False)  # downloads/caches on first call
        X, y = ds.data, ds.target

    else:
        raise ValueError("dataset must be one of: make_circles, iris, breast_cancer, parkinsons.")

    # Shuffle (iris comes ordered)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X = X.astype(np.float64)

    # standardize
    X = StandardScaler().fit_transform(X)

    # optional dimensionality reduction
    if n_features is not None and int(n_features) > 0 and X.shape[1] > int(n_features):
        k = int(n_features)
        if pca:
            X = PCA(n_components=k, random_state=seed).fit_transform(X)
            # re-standardize after PCA (keeps angle scaling more consistent)
            X = StandardScaler().fit_transform(X)
        else:
            X = X[:, :k]

    # map to angles
    X = np.pi * X / 2.0

    # OpenML can return y as strings; make it int if possible
    try:
        y = y.astype(int)
    except Exception:
        y = np.array(y, dtype=int)

    return X, y


def make_splits(n: int, seed: int, val_size: float, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic train/val/test splits.

    Behavior matches the current demo scripts:
      - no stratify
      - use val_size + test_size as holdout, then split holdout into val/test.
    """
    from sklearn.model_selection import train_test_split

    idx_all = np.arange(n, dtype=int)
    idx_train, idx_tmp = train_test_split(
        idx_all,
        test_size=(val_size + test_size),
        random_state=seed,
        shuffle=True,
        stratify=None,
    )
    rel_test = test_size / (val_size + test_size)
    idx_val, idx_test = train_test_split(
        idx_tmp,
        test_size=rel_test,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


# ---------------------------
# kernel post-processing
# ---------------------------
def center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center the kernel: Kc = H K H. Enforces exact symmetry."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    return 0.5 * (Kc + Kc.T)


def spectrum_stats(K: np.ndarray, thresh: float = 1e-6) -> Dict:
    """Return basic spectrum stats for symmetric K."""
    Ks = 0.5 * (K + K.T)
    w = np.linalg.eigvalsh(Ks)
    eff_rank = int(np.sum(w > thresh))
    return {
        "n": int(K.shape[0]),
        "trace": float(np.sum(w)),
        "lambda_min": float(w.min()),
        "lambda_max": float(w.max()),
        "effective_rank@{:.0e}".format(thresh): eff_rank,
        "threshold": float(thresh),
    }


def normalize_unit_diag(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize kernel to unit diagonal: K_ij <- K_ij / sqrt(K_ii K_jj)."""
    d = np.sqrt(np.clip(np.diag(K), eps, None))
    Kn = K / (d[:, None] * d[None, :])
    Kn = 0.5 * (Kn + Kn.T)
    np.fill_diagonal(Kn, 1.0)
    return Kn.astype(np.float64)


# ---------------------------
# I/O helpers
# ---------------------------
def prepare_dirs(out_prefix: Path, figs_subdir: str) -> Tuple[Path, Path]:
    """
    Create output dir (parent of out_prefix) and figures dir.
    Returns: (out_dir, figs_dir)
    """
    out_dir = out_prefix.parent
    figs_dir = Path(figs_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, figs_dir


def artifact_paths(out_prefix: Path, figs_dir: Path, centered: bool) -> Dict[str, str]:
    """
    Match the current demo naming scheme:
      suffix = "_centered" if centered else ""
      kpath = f"{out_prefix}_K{suffix}.npy"
      mpath = f"{out_prefix}_meta.json"
      spath = f"{out_prefix}_splits.json"
      fprefix = f"{figs_dir}/{out_prefix.name}{suffix}"
    """
    suffix = "_centered" if centered else ""
    return {
        "suffix": suffix,
        "kpath": str(out_prefix) + f"_K{suffix}.npy",
        "mpath": str(out_prefix) + "_meta.json",
        "spath": str(out_prefix) + "_splits.json",
        "fprefix": str(figs_dir / (out_prefix.name + suffix)),
        "spectrum_path": str(out_prefix) + "_spectrum.txt",
    }


def save_json(path: str, obj: Dict) -> None:
    def _default(o):
        # NumPy scalars -> Python scalars
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        # NumPy arrays -> lists
        if isinstance(o, np.ndarray):
            return o.tolist()
        # Fallback
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def save_splits(path: str, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, y_all: np.ndarray) -> None:
    save_json(
        path,
        {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "y_all": y_all.tolist(),
        },
    )


def write_spectrum_txt(path: str, stats: Dict) -> None:
    with open(path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


def append_metrics_row(metrics_csv: str, header: list, row: list) -> None:
    """
    Append a row with the same CSV schema used by the demo scripts.
    """
    write_header = not os.path.exists(metrics_csv)
    with open(metrics_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def plot_all(K: np.ndarray, fprefix: str) -> None:
    """
    Generate the three standard diagnostic figures.
    """
    from analysis.diagnostics import plot_heatmap, plot_offdiag_hist, plot_spectrum

    plot_heatmap(K, f"{fprefix}_matrix.png")
    plot_offdiag_hist(K, f"{fprefix}_offdiag_hist.png")
    plot_spectrum(K, f"{fprefix}_spectrum.png")
