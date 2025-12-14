"""
Basic behavioral tests for kernel implementations.

These tests go beyond API-signature checks:
- output shape and dtype
- symmetry
- diagonal sanity (no NaNs/Infs; baseline should be ~1)
- kernel depends on X (not always identity / constant)
- determinism for fixed inputs (same call -> same K)
"""

import numpy as np
import pytest


# Ensure Qiskit is available (repo requirement); skip gracefully otherwise.
pytest.importorskip("qiskit")

from qkernels.baseline_kernel import build_kernel as build_baseline
from qkernels.local_kernel import build_kernel as build_local
from qkernels.multiscale_kernel import build_kernel as build_multiscale


def _make_toy_X(n: int = 6, d: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float64)
    # Keep angles modest; this is just to exercise the pipelines.
    X = (np.pi / 4.0) * X
    return X


def _assert_kernel_basic_properties(K: np.ndarray, n: int) -> None:
    assert isinstance(K, np.ndarray)
    assert K.shape == (n, n)
    assert np.issubdtype(K.dtype, np.floating)

    # Finite values
    assert np.isfinite(K).all()

    # Symmetry (numerical tolerance)
    assert np.allclose(K, K.T, atol=1e-10, rtol=0.0)

    # Diagonal sanity
    diag = np.diag(K)
    assert np.isfinite(diag).all()


def _assert_kernel_depends_on_X(build_fn, X: np.ndarray, **kwargs) -> None:
    """Small perturbation in X should change K (not exactly identical)."""
    K1, _ = build_fn(X, **kwargs)

    X2 = X.copy()
    X2[0, 0] += 0.123  # small perturbation
    K2, _ = build_fn(X2, **kwargs)

    # The kernels should not be exactly identical.
    # We use a tiny tolerance to avoid false positives due to numerical roundoff.
    assert not np.allclose(K1, K2, atol=1e-12, rtol=0.0)


def _assert_deterministic(build_fn, X: np.ndarray, **kwargs) -> None:
    """Same call should produce the same result (for statevector demos)."""
    K1, _ = build_fn(X, **kwargs)
    K2, _ = build_fn(X, **kwargs)
    assert np.allclose(K1, K2, atol=1e-12, rtol=0.0)


def test_baseline_kernel_properties():
    X = _make_toy_X(n=6, d=2, seed=1)
    n = X.shape[0]

    K, meta = build_baseline(
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        entanglement="linear",
    )

    _assert_kernel_basic_properties(K, n)

    # Baseline (fidelity) kernel should have diagonal ~ 1.
    assert np.allclose(np.diag(K), 1.0, atol=1e-8, rtol=0.0)

    # Fidelity values should be in [0, 1] (allow tiny numerical slop).
    assert K.min() >= -1e-12
    assert K.max() <= 1.0 + 1e-12

    # Metadata sanity
    assert isinstance(meta, dict)
    assert meta.get("kernel") in {None, "baseline", "global", "fidelity"}  # tolerate legacy naming

    _assert_kernel_depends_on_X(
        build_baseline,
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        entanglement="linear",
    )
    _assert_deterministic(
        build_baseline,
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        entanglement="linear",
    )


def test_local_kernel_properties_rdm_single_qubit_patches():
    X = _make_toy_X(n=6, d=2, seed=2)
    n, d = X.shape

    K, meta = build_local(
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        entanglement="linear",
        partitions=[[0], [1]],
        method="rdm",
        agg="mean",
        # If supported by your local_kernel changes; otherwise it will be ignored/raise.
        rdm_metric="fidelity",
    )

    _assert_kernel_basic_properties(K, n)

    # Local kernels may not have diag exactly 1 depending on metric/normalization,
    # but should still be non-negative for fidelity-style similarity.
    assert np.diag(K).min() >= -1e-12

    assert isinstance(meta, dict)
    assert meta.get("kernel") in {None, "local"}

    _assert_kernel_depends_on_X(
        build_local,
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        entanglement="linear",
        partitions=[[0], [1]],
        method="rdm",
        agg="mean",
        rdm_metric="fidelity",
    )
    _assert_deterministic(
        build_local,
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        entanglement="linear",
        partitions=[[0], [1]],
        method="rdm",
        agg="mean",
        rdm_metric="fidelity",
    )


def test_multiscale_kernel_properties_default_scales():
    X = _make_toy_X(n=6, d=2, seed=3)
    n = X.shape[0]

    K, meta = build_multiscale(
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        scales=None,
        weights=None,
        # keep script-like behavior explicit:
        normalize=False,
        entanglement="linear",
    )

    _assert_kernel_basic_properties(K, n)

    assert isinstance(meta, dict)
    assert meta.get("kernel") in {None, "multiscale"}

    _assert_kernel_depends_on_X(
        build_multiscale,
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        scales=None,
        weights=None,
        normalize=False,
        entanglement="linear",
    )
    _assert_deterministic(
        build_multiscale,
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        scales=None,
        weights=None,
        normalize=False,
        entanglement="linear",
    )
