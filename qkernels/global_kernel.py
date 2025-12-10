"""
Global (fidelity) kernel.

Unified API:
    build_kernel(X, feature_map="zz", depth=1, backend="statevector", seed=42, **kwargs)
Returns:
    K: (n, n) ndarray (symmetric, ~PSD)
    meta: dict (config used)

TODO: implement fidelity overlaps end-to-end (statevector or sampling).
"""

from typing import Tuple, Dict, Any
import numpy as np
from .feature_maps import get_feature_map_spec
from qiskit.quantum_info import Statevector


def build_kernel(
    X: np.ndarray,
    feature_map: str = "zz",
    depth: int = 1,
    backend: str = "statevector",
    seed: int = 42,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Global fidelity kernel (baseline).

    Parameters
    ----------
    X : np.ndarray (n_samples, d)
        Input features; assume preprocessing (scaling) happened upstream.
    feature_map : str
    depth : int
    backend : str
        "statevector" | "sampling" (not yet implemented)
    seed : int

    Returns
    -------
    K : np.ndarray (n, n)
    meta : dict
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape

    # --- Get feature map spec
    fmap_spec = get_feature_map_spec(
        name=feature_map,
        depth=depth,
        num_qubits=n_features,
    )

    impl = fmap_spec["impl"]
    builder = fmap_spec["builder"]
    name = fmap_spec["name"]


    # --- Build statevectors for all inputs ---
    states = []
    for x in X:
        qc = builder(x)
        psi = Statevector.from_instruction(qc)
        states.append(psi.data)

    # --- Compute kernel ---
    K = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        for j in range(i, n_samples):
            fid = np.abs(np.vdot(states[i], states[j])) ** 2
            K[i, j] = K[j, i] = fid
    np.fill_diagonal(K, 1.0)

    meta = {
        "type": "global_fidelity",
        "feature_map": name,
        "impl": impl,
        "depth": depth,
        "backend": backend,
        "n_samples": n_samples,
        "n_features": n_features,
        "seed": seed,
    }

    return K, meta