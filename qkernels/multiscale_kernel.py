"""
Multi-scale kernel.

Unified API:
    build_kernel(X, feature_map="zz", depth=1, backend="statevector", seed=42,
                 scales=[[(0,1),(2,3)], [(0,1,2,3)]], weights=[0.5, 0.5], **kwargs)

Returns:
    K: (n, n) ndarray (symmetric, ~PSD)
    meta: dict

TODO:
 - Compute per-scale kernels using the same local logic and combine with non-negative weights.
 - Add simple ablation support (optional).
"""

from typing import Tuple, Dict, Any, Iterable, List, Optional
import numpy as np
from .feature_maps import get_feature_map_spec


def build_kernel(
    X: np.ndarray,
    feature_map: str = "zz",
    depth: int = 1,
    backend: str = "statevector",
    seed: int = 42,
    scales: Optional[List[Iterable[Iterable[int]]]] = None,
    weights: Optional[List[float]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Multi-scale kernel.

    Parameters
    ----------
    X : np.ndarray (n_samples, d)
    scales : list of partitions lists, e.g., [ [(0,1),(2,3)], [(0,1,2,3)] ]
    weights : list of non-negative floats summing to 1 (optional)

    Returns
    -------
    K : np.ndarray (n, n)
    meta : dict
    """
    d = X.shape[1]
    if scales is None:
        # Default: pairs then all
        pairs = [(i, i + 1) for i in range(0, d, 2) if i + 1 < d]
        scales = [pairs, [tuple(range(d))]]

    if weights is None:
        weights = [1.0 / len(scales)] * len(scales)

    if len(weights) != len(scales):
        raise ValueError("weights and scales must have the same length.")

    fmap = get_feature_map_spec(feature_map, depth, num_qubits=d)
    meta = {
        "kernel": "multiscale",
        "feature_map": fmap["name"],
        "depth": depth,
        "backend": backend,
        "seed": seed,
        "scales": [[list(p) for p in s] for s in scales],
        "weights": list(weights),
        **kwargs,
    }

    # TODO: compute K = sum_s w_s * K^(s) using local-patch logic per scale.
    n = X.shape[0]
    K = np.eye(n, dtype=np.float64)

    raise NotImplementedError("Multi-scale kernel not implemented yet. Combine per-scale kernels with non-negative weights.")
    # return K, meta
