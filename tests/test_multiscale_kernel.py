import numpy as np
import pytest

from qkernels.multiscale_kernel import build_kernel


def _toy_X(n: int, d: int) -> np.ndarray:
    """Small deterministic dataset; values in [-pi/2, pi/2]."""
    base = np.linspace(-0.5, 0.5, n * d, dtype=np.float64).reshape(n, d)
    return np.pi * base


@pytest.mark.parametrize("fmap", ["zz_manual", "zz_qiskit", "zz_manual_canonical"])
def test_basic_properties_and_psd(fmap):
    n, d = 5, 4
    X = _toy_X(n, d)
    K, meta = build_kernel(
        X,
        feature_map=fmap,
        depth=1,
        backend="statevector",
        scales=None,      # defaults: pairs + all
        weights=None,     # defaults: uniform
    )

    # Shape
    assert K.shape == (n, n)

    # Symmetry
    assert np.allclose(K, K.T, atol=1e-10)

    # Unit diagonal (default normalize=True)
    assert np.allclose(np.diag(K), 1.0, rtol=1e-8, atol=1e-8)

    # PSD (up to tiny numerical noise)
    w = np.linalg.eigvalsh(0.5 * (K + K.T))
    assert w.min() >= -1e-8

    # Meta basics
    assert meta["kernel"] == "multiscale"
    assert meta["feature_map"] == fmap
    assert meta["n_samples"] == n
    assert meta["n_qubits"] == d


def test_backend_not_supported():
    X = _toy_X(3, 4)
    with pytest.raises(NotImplementedError):
        build_kernel(X, backend="sampling")


def test_invalid_scales_index_raises():
    n, d = 4, 3
    X = _toy_X(n, d)
    # Patch references out-of-range qubit index 5
    bad_scales = [[(0, 1)], [(5,)]]
    with pytest.raises(ValueError):
        build_kernel(X, scales=bad_scales)


def test_negative_weights_raise():
    n, d = 4, 4
    X = _toy_X(n, d)
    scales = [[(0, 1)], [(2, 3)]]
    weights = [0.7, -0.3]
    with pytest.raises(ValueError):
        build_kernel(X, scales=scales, weights=weights)


def test_weights_length_mismatch_raises():
    n, d = 4, 4
    X = _toy_X(n, d)
    scales = [[(0, 1)], [(2, 3)]]
    weights = [1.0]  # mismatch
    with pytest.raises(ValueError):
        build_kernel(X, scales=scales, weights=weights)


def test_normalize_flag_effect():
    n, d = 6, 4
    X = _toy_X(n, d)
    # With normalize=True (default): diag ~ 1
    K1, _ = build_kernel(X, normalize=True)
    # With normalize=False: diagonal not necessarily 1
    K2, _ = build_kernel(X, normalize=False)
    assert np.allclose(np.diag(K1), 1.0, atol=1e-8)
    # Ensure there exists at least one diagonal entry that differs from 1
    assert not np.allclose(np.diag(K2), 1.0, atol=1e-6)


def test_determinism_same_inputs_same_output():
    n, d = 5, 4
    X = _toy_X(n, d)
    K1, meta1 = build_kernel(X, feature_map="zz_manual", depth=1)
    K2, meta2 = build_kernel(X, feature_map="zz_manual", depth=1)
    assert np.allclose(K1, K2, atol=1e-12)
    assert meta1["scales"] == meta2["scales"]
    assert np.allclose(meta1["weights"], meta2["weights"])


def test_custom_scales_and_weights():
    n, d = 5, 4
    X = _toy_X(n, d)
    # Define two scales: (i) pairs (0,1) & (2,3), (ii) all qubits
    scales = [[(0, 1), (2, 3)], [tuple(range(d))]]
    weights = [0.3, 0.7]  # non-negative; will be normalized internally
    K, meta = build_kernel(X, scales=scales, weights=weights)
    # Weights should be normalized to sum 1
    assert np.isclose(np.sum(meta["weights"]), 1.0)
    assert len(meta["weights"]) == 2
    # Kernel still PSD and symmetric
    w = np.linalg.eigvalsh(0.5 * (K + K.T))
    assert w.min() >= -1e-8
    assert np.allclose(K, K.T, atol=1e-10)
