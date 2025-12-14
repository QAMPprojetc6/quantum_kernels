"""
Baseline (global fidelity) kernel.

Unified API:
    build_kernel(X, feature_map="zz", depth=1, backend="statevector", seed=42, **kwargs)

Returns:
    K: (n, n) ndarray (symmetric, ~PSD)
    meta: dict (config used)

Implements:
  - statevector backend: exact fidelity kernel K_ij = |<psi(x_i) | psi(x_j)>|^2
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from qiskit.quantum_info import Statevector

from .feature_maps import get_feature_map_spec


def _statevectors_for_samples(
    X: np.ndarray,
    fmap_name: str,
    depth: int,
    entanglement: Optional[str] = None,
) -> np.ndarray:
    """
    Build feature-map circuits for each sample and return an array of statevectors.

    Returns
    -------
    S : np.ndarray (n, 2^d) complex
        Row i is the statevector |psi(x_i)> as a complex array.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, d).")

    n, d = X.shape
    spec = get_feature_map_spec(
        name=fmap_name,
        depth=depth,
        num_qubits=int(d),
        entanglement=entanglement,
    )
    builder = spec["builder"]

    dim = 2 ** int(d)
    S = np.empty((n, dim), dtype=np.complex128)

    for i in range(n):
        x_i = np.asarray(X[i], dtype=np.float64).ravel()
        qc = builder(x_i)
        sv = Statevector.from_instruction(qc)
        S[i, :] = np.asarray(sv.data, dtype=np.complex128)

    return S


def build_kernel(
    X: np.ndarray,
    feature_map: str = "zz",
    depth: int = 1,
    backend: str = "statevector",
    seed: int = 42,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Baseline fidelity kernel (global).

    Parameters
    ----------
    X : np.ndarray (n_samples, d)
    feature_map : str
        Passed to get_feature_map_spec(...).
    depth : int
        Feature map reps/layers.
    backend : str
        "statevector" | "sampling" (sampling not implemented here)
    seed : int
        (Kept for API consistency; statevector path is deterministic.)

    Other kwargs
    ------------
    entanglement : Optional[str]
        Passed through to feature map builder (Qiskit ZZ or manual variants).

    Returns
    -------
    K : np.ndarray (n, n) float64
        Fidelity kernel matrix.
    meta : dict
        Config used.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, d).")

    n, d = X.shape
    d = int(d)

    # Extract optional args (commonly passed from CLI via kwargs)
    entanglement: Optional[str] = kwargs.pop("entanglement", None)

    # Validate feature map exists / is buildable
    fmap_spec = get_feature_map_spec(
        name=feature_map,
        depth=depth,
        num_qubits=d,
        entanglement=entanglement,
    )

    meta: Dict[str, Any] = {
        "kernel": "baseline",
        "feature_map": fmap_spec["name"],
        "feature_map_impl": fmap_spec.get("impl", None),
        "depth": int(depth),
        "backend": backend,
        "seed": int(seed),
        "num_qubits": d,
        "entanglement": entanglement,
        **kwargs,
    }

    backend_norm = str(backend).strip().lower()
    if backend_norm != "statevector":
        raise NotImplementedError(
            "baseline_kernel currently supports backend='statevector' only. "
            "Use statevector for now, or implement a sampling-based estimator later."
        )

    # Compute all statevectors
    S = _statevectors_for_samples(X, fmap_name=feature_map, depth=depth, entanglement=entanglement)

    # Overlap Gram matrix G_ij = <psi_i | psi_j>
    #    If S rows are |psi_i>, then G = S * S^\dagger = S @ S.conj().T
    G = S @ S.conj().T  # (n, n) complex

    # Fidelity kernel K_ij = |G_ij|^2
    K = np.abs(G) ** 2
    K = K.astype(np.float64, copy=False)

    # Numerical hygiene: enforce symmetry + diagonal ~ 1
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)

    return K, meta
