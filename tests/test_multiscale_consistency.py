"""
Multi-scale kernel consistency tests.

Key check:
- If multiscale returns K = sum_s w_s K_s, then computing each K_s separately
  and mixing them must reproduce K_mix (within numerical tolerance).
"""

import numpy as np
import pytest

pytest.importorskip("qiskit")

from qkernels.multiscale_kernel import build_kernel as build_multiscale


def _make_toy_X(n: int = 6, d: int = 2, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float64)
    X = (np.pi / 4.0) * X
    return X


def test_multiscale_is_weighted_sum_of_single_scale_kernels():
    X = _make_toy_X(n=6, d=2, seed=10)

    # Define two scales for d=2:
    # S1: local 1q patches
    # S2: all-qubits patch (baseline-like scale)
    S1 = [[[0], [1]]]
    S2 = [[[0, 1]]]

    w1, w2 = 0.3, 0.7

    K1, _ = build_multiscale(
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        scales=S1,
        weights=[1.0],
        normalize=False,
        entanglement="linear",
    )

    K2, _ = build_multiscale(
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        scales=S2,
        weights=[1.0],
        normalize=False,
        entanglement="linear",
    )

    Kmix, _ = build_multiscale(
        X,
        feature_map="zz_manual",
        depth=1,
        backend="statevector",
        seed=42,
        scales=[S1[0], S2[0]],     # list of scales (each is a partition list)
        weights=[w1, w2],
        normalize=False,
        entanglement="linear",
    )

    K_expected = w1 * K1 + w2 * K2

    # The result should match the explicit weighted sum
    assert np.allclose(Kmix, K_expected, atol=1e-10, rtol=0.0)
