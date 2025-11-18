from pathlib import Path
import json
import os

import numpy as np
import pytest

from analysis.eval_svm import read_splits, eval_precomputed, main as eval_main


def _make_psd_kernel(n: int, d: int = 3, seed: int = 0) -> np.ndarray:
    """Small PSD kernel via X @ X.T, normalized to diag ~ 1."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    K = X @ X.T
    dnorm = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K = (K / dnorm[:, None]) / dnorm[None, :]
    K = 0.5 * (K + K.T)
    return K.astype(np.float64)


def _write_splits(tmp_path: Path, n: int):
    """Create a simple split and labels JSON."""
    # train 0..(n//2 - 1), val next quarter, test rest
    t_end = n // 2
    v_end = t_end + n // 4
    train_idx = list(range(0, t_end))
    val_idx = list(range(t_end, v_end))
    test_idx = list(range(v_end, n))
    # binary labels alternating (just deterministic)
    y_all = [i % 2 for i in range(n)]
    sp = tmp_path / "splits.json"
    sp.write_text(json.dumps({
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "y_all": y_all,
    }))
    return sp


def test_read_splits_roundtrip(tmp_path: Path):
    n = 12
    sp = _write_splits(tmp_path, n)
    train_idx, val_idx, test_idx, y_all = read_splits(str(sp))
    assert train_idx.ndim == 1 and val_idx.ndim == 1 and test_idx.ndim == 1
    assert y_all.shape == (n,)
    # No overlaps sanity (not strictly required, just a basic check)
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(val_idx).isdisjoint(set(test_idx))


def test_eval_precomputed_returns_metrics(tmp_path: Path):
    n = 20
    K = _make_psd_kernel(n, d=5)
    sp = _write_splits(tmp_path, n)
    train_idx, val_idx, test_idx, y_all = read_splits(str(sp))
    Cs = [0.1, 1.0, 10.0]
    metrics = eval_precomputed(K, y_all, train_idx, val_idx, test_idx, Cs)
    # keys present
    assert set(metrics.keys()) == {"best_C", "val_acc", "test_acc"}
    # best_C ∈ Cs
    assert metrics["best_C"] in Cs
    # accuracies are floats in [0,1]
    assert 0.0 <= metrics["val_acc"] <= 1.0
    assert 0.0 <= metrics["test_acc"] <= 1.0


def test_main_writes_csv_with_dir(tmp_path: Path):
    n = 18
    K = _make_psd_kernel(n, d=4)
    kpath = tmp_path / "K.npy"
    np.save(kpath, K)

    sp = _write_splits(tmp_path, n)

    out_dir = tmp_path / "outputs"
    out_csv = out_dir / "metrics.csv"
    eval_main(str(kpath), str(sp), [0.1, 1.0, 10.0], str(out_csv))

    assert out_csv.exists() and out_csv.stat().st_size > 0

    # Run again to ensure it appends (no duplicate header)
    size_before = out_csv.stat().st_size
    eval_main(str(kpath), str(sp), [0.1, 1.0, 10.0], str(out_csv))
    assert out_csv.stat().st_size > size_before


def test_main_writes_csv_with_plain_filename(tmp_path: Path, monkeypatch):
    # ensure CWD = tmp_path, and use an out path without directory
    monkeypatch.chdir(tmp_path)

    n = 16
    K = _make_psd_kernel(n, d=4)
    kpath = tmp_path / "K.npy"
    np.save(kpath, K)
    sp = _write_splits(tmp_path, n)

    out_csv = "metrics.csv"  # no directory
    eval_main(str(kpath), str(sp), [0.1, 1.0], out_csv)
    p = tmp_path / out_csv
    assert p.exists() and p.stat().st_size > 0
