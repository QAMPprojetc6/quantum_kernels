"""
Baseline (global fidelity) kernel.

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
    backend : str
        "statevector" | "sampling" (sampling not implemented here)
    seed : int

    Returns
    -------
    K : np.ndarray (n, n)
    meta : dict
    """

    d = int(X.shape[1])
    _ = get_feature_map_spec(feature_map, depth, num_qubits=d)  # validate feature map

    meta = {
        "kernel": "baseline",
        "feature_map": feature_map,
        "depth": depth,
        "backend": backend,
        "seed": seed,
        **kwargs,
    }

    raise NotImplementedError("Global kernel not implemented yet. Use this file as a skeleton to add fidelity overlaps.")
    # return K, meta
