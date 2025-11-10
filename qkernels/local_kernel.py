"""
Local (patch-wise) kernel.

Unified API:
    build_kernel(X, feature_map="zz", depth=1, backend="statevector", seed=42,
                 partitions=[(0,1),(2,3)], method="subcircuits"|"rdm",
                 agg="mean", weights=None, **kwargs)

Returns:
    K: (n, n) ndarray (symmetric, ~PSD)
    meta: dict

TODO:
 - Implement per-patch overlaps (subcircuits) OR reduced density matrices (RDMs).
 - Support aggregation (mean or weighted).
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
    partitions: Optional[Iterable[Iterable[int]]] = None,
    method: str = "subcircuits",  # or "rdm"
    agg: str = "mean",
    weights: Optional[List[float]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Patch-wise kernel.

    Parameters
    ----------
    X : np.ndarray (n_samples, d)
    partitions : iterable of iterables (e.g., [(0,1), (2,3)])
    method : "subcircuits" | "rdm"
    agg : "mean" | "weighted"
    weights : list of floats (same length as number of patches) or None

    Returns
    -------
    K : np.ndarray (n, n)
    meta : dict
    """

    # meta = {...}

    raise NotImplementedError("Local kernel not implemented yet. Use this skeleton to add per-patch overlaps and aggregation.")
    # return K, meta
