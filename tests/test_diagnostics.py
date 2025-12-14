import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import pytest

from analysis.diagnostics import (
    plot_heatmap,
    plot_offdiag_hist,
    plot_spectrum,
    main as diag_main,
)


def _make_psd_kernel(n: int, rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    X = rng.normal(size=(n, n // 2 + 1))
    K = X @ X.T
    # normalize diagonals to ~1 for stability
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K = (K / d[:, None]) / d[None, :]
    # symmetrize for good measure
    K = 0.5 * (K + K.T)
    return K.astype(np.float64)


def test_plot_functions_create_files(tmp_path: Path):
    K = _make_psd_kernel(10)
    # Heatmap
    out_heat = tmp_path / "hm.png"
    plot_heatmap(K, str(out_heat))
    assert out_heat.exists() and out_heat.stat().st_size > 0

    # Off-diagonal hist
    out_hist = tmp_path / "hist.png"
    plot_offdiag_hist(K, str(out_hist))
    assert out_hist.exists() and out_hist.stat().st_size > 0

    # Spectrum
    out_spec = tmp_path / "spec.png"
    plot_spectrum(K, str(out_spec))
    assert out_spec.exists() and out_spec.stat().st_size > 0


def test_cli_main_generates_all_files_and_meta(tmp_path: Path):
    # Prepare inputs
    K = _make_psd_kernel(8)
    kpath = tmp_path / "K.npy"
    np.save(kpath, K)

    meta = {"kernel": "baseline", "seed": 42}
    mpath = tmp_path / "meta.json"
    mpath.write_text(__import__("json").dumps(meta))

    # Save prefix inside a subfolder that doesn't exist yet
    save_dir = tmp_path / "figs"
    save_prefix = save_dir / "run1"
    # Run main
    diag_main(str(kpath), str(save_prefix), str(mpath))

    # Expect files
    files = [
        f"{save_prefix}_matrix.png",
        f"{save_prefix}_offdiag_hist.png",
        f"{save_prefix}_spectrum.png",
        f"{save_prefix}_meta.txt",
    ]
    for fp in files:
        assert os.path.exists(fp), f"Missing output: {fp}"
        assert os.path.getsize(fp) > 0


def test_cli_main_handles_plain_prefix_without_dir(tmp_path: Path, monkeypatch):
    # Ensure CWD is tmp_path so outputs go there
    monkeypatch.chdir(tmp_path)

    K = _make_psd_kernel(6)
    kpath = tmp_path / "K.npy"
    np.save(kpath, K)

    # save_prefix without directory (should default to ".")
    save_prefix = "plainprefix"
    diag_main(str(kpath), save_prefix, None)

    for suffix in ("_matrix.png", "_offdiag_hist.png", "_spectrum.png"):
        p = tmp_path / f"{save_prefix}{suffix}"
        assert p.exists() and p.stat().st_size > 0
