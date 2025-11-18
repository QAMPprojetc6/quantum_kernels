"""
Multi-scale kernel.

Unified API:
    build_kernel(
        X,
        feature_map="zz",
        depth=1,
        backend="statevector",
        seed=42,
        scales=[[(0,1),(2,3)], [(0,1,2,3)]],
        weights=[0.5, 0.5],
        **kwargs
    )

Returns:
    K: (n, n) ndarray (symmetric, ~PSD)
    meta: dict

Notes
-----
- Each scale is a list of "patches" (tuples of qubit indices). For a scale s with patches P_s,
  we compute a per-scale kernel:
        K^(s)[i,j] = mean_{p in P_s} Tr( ρ_i^p ρ_j^p )
  where ρ_i^p is the reduced density matrix (RDM) of sample i on patch p.
- The final kernel is a non-negative mix:
        K = sum_s w_s * K^(s),     w_s >= 0, sum_s w_s = 1
- If `normalize=True` (default), we normalize K to unit diagonal:
        K_ij <- K_ij / sqrt(K_ii * K_jj)
  This preserves PSD and makes kernels comparable across scales.
"""

from typing import Tuple, Dict, Any, Iterable, List, Optional
import numpy as np

from .feature_maps import get_feature_map_spec
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace


def _ensure_scales_default(d: int) -> List[List[Iterable[int]]]:
    """Default scales: (i) contiguous pairs, (ii) all qubits."""
    pairs = [(i, i + 1) for i in range(0, d, 2) if i + 1 < d]
    return [pairs, [tuple(range(d))]]


def _validate_scales(scales: List[Iterable[Iterable[int]]], d: int) -> List[List[Tuple[int, ...]]]:
    """
    Validate and canonicalize `scales` into a list of list of integer tuples.
    Ensures indices are in-range and no duplicate indices inside a patch.
    """
    canon: List[List[Tuple[int, ...]]] = []
    for s_idx, scale in enumerate(scales):
        scale_list: List[Tuple[int, ...]] = []
        for p in scale:
            patch = tuple(int(q) for q in p)
            if len(patch) == 0:
                raise ValueError(f"Empty patch in scale {s_idx}.")
            if any((q < 0 or q >= d) for q in patch):
                raise ValueError(f"Patch {patch} in scale {s_idx} has out-of-range qubit index for d={d}.")
            if len(set(patch)) != len(patch):
                raise ValueError(f"Patch {patch} in scale {s_idx} has repeated indices.")
            scale_list.append(patch)
        if len(scale_list) == 0:
            raise ValueError(f"Scale {s_idx} has no patches.")
        canon.append(scale_list)
    return canon


def _validate_weights(weights: List[float], n_scales: int) -> np.ndarray:
    """Ensure weights are non-negative and sum to 1 (if not, renormalize)."""
    if len(weights) != n_scales:
        raise ValueError("weights and scales must have the same length.")
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("All weights must be non-negative.")
    s = float(w.sum())
    if s <= 0:
        # If all zeros, default to uniform
        w = np.ones(n_scales, dtype=np.float64) / n_scales
    else:
        w = w / s
    return w


def _statevectors_for_samples(X: np.ndarray, fmap_name: str, depth: int) -> List[Statevector]:
    """
    Build the feature-map circuit for each sample and return its statevector.
    """
    n, d = X.shape
    spec = get_feature_map_spec(name=fmap_name, depth=depth, num_qubits=d)
    builder = spec["builder"]

    svs: List[Statevector] = []
    for i in range(n):
        params = np.asarray(X[i], dtype=np.float64).ravel()
        qc = builder(params)
        svs.append(Statevector(qc))
    return svs


def _rdm_for_patch(rho_full: DensityMatrix, d: int, patch: Tuple[int, ...]) -> DensityMatrix:
    """
    Compute the reduced density matrix on a given patch by tracing out the complement.
    Qubit indices follow the circuit's ordering (0..d-1).
    """
    all_idx = set(range(d))
    trace_out = sorted(all_idx - set(patch))
    if len(trace_out) == 0:
        # No reduction needed; return the full DM
        return rho_full
    return partial_trace(rho_full, trace_out)


def _hs_inner(A: DensityMatrix, B: DensityMatrix) -> float:
    """Hilbert–Schmidt inner product Tr(A B), returned as a real float."""
    # A, B are Hermitian; numerical noise may introduce small imaginary parts.
    val = np.trace(A.data @ B.data)
    return float(np.real(val))


def _per_scale_kernel(
    svs: List[Statevector],
    d: int,
    patches: List[Tuple[int, ...]],
) -> np.ndarray:
    """
    Build K^(s) for one scale by averaging HS inner products of patch RDMs.
    """
    n = len(svs)
    # Precompute full density matrices once per sample
    rhos_full = [DensityMatrix(sv) for sv in svs]
    # Precompute RDMs per patch & sample: rdm[p][i] = rho_i^patch
    rdm_per_patch: List[List[DensityMatrix]] = []
    for patch in patches:
        rdms = [ _rdm_for_patch(rhos_full[i], d, patch) for i in range(n) ]
        rdm_per_patch.append(rdms)

    # Compute per-scale kernel matrix
    Ks = np.zeros((n, n), dtype=np.float64)
    inv_m = 1.0 / float(len(patches))
    for a in range(n):
        # Diagonal block faster
        for b in range(a, n):
            acc = 0.0
            for p_idx in range(len(patches)):
                acc += _hs_inner(rdm_per_patch[p_idx][a], rdm_per_patch[p_idx][b])
            val = acc * inv_m
            Ks[a, b] = Ks[b, a] = val
    return Ks


def _normalize_kernel(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize kernel to unit diagonal: K_ij / sqrt(K_ii K_jj).
    Keeps PSD and improves comparability across scales.
    """
    diag = np.clip(np.diag(K), eps, None)
    D_inv_sqrt = 1.0 / np.sqrt(diag)
    K_norm = (K * D_inv_sqrt[:, None]) * D_inv_sqrt[None, :]
    # Enforce exact symmetry
    K_norm = 0.5 * (K_norm + K_norm.T)
    return K_norm


def build_kernel(
    X: np.ndarray,
    feature_map: str = "zz",
    depth: int = 1,
    backend: str = "statevector",
    seed: int = 42,  # kept for API consistency; RNG not needed in this implementation
    scales: Optional[List[Iterable[Iterable[int]]]] = None,
    weights: Optional[List[float]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Multi-scale kernel.

    Parameters
    ----------
    X : np.ndarray (n_samples, d)
        Input features (assumed preprocessed/scaled upstream).
    feature_map : str
        Feature map selector passed to `get_feature_map_spec`.
    depth : int
        Number of feature-map repetitions.
    backend : str
        Only "statevector" is supported in this implementation.
    seed : int
        Unused here; present for API stability.
    scales : list[list[iterable[int]]], optional
        Scales as lists of patches (tuples of qubit indices).
    weights : list[float], optional
        Non-negative weights per scale. If None, uniform.

    Returns
    -------
    K : np.ndarray (n, n)
        Multi-scale kernel matrix (normalized to unit diagonal unless `normalize=False` in kwargs).
    meta : dict
        Configuration and bookkeeping info.
    """
    if backend != "statevector":
        raise NotImplementedError("Only backend='statevector' is supported in multiscale kernel (no Aer required).")

    n, d = X.shape
    # Scales & weights
    if scales is None:
        scales = _ensure_scales_default(d)
    scales_canon = _validate_scales(scales, d)

    if weights is None:
        weights = [1.0 / len(scales_canon)] * len(scales_canon)
    w = _validate_weights(weights, len(scales_canon))

    # Optional normalization toggle (defaults to True)
    normalize: bool = bool(kwargs.pop("normalize", True))

    # Build statevectors for all samples
    svs = _statevectors_for_samples(X, fmap_name=feature_map, depth=depth)

    # Per-scale kernels and weighted sum
    K = np.zeros((n, n), dtype=np.float64)
    per_scale_contrib = []
    for s_idx, patches in enumerate(scales_canon):
        Ks = _per_scale_kernel(svs, d=d, patches=patches)
        per_scale_contrib.append(Ks.copy())
        K += w[s_idx] * Ks

    if normalize:
        K = _normalize_kernel(K)

    # Enforce numeric symmetry (just in case)
    K = 0.5 * (K + K.T)

    meta: Dict[str, Any] = {
        "kernel": "multiscale",
        "feature_map": feature_map,
        "depth": depth,
        "backend": backend,
        "seed": seed,
        "scales": [[list(p) for p in s] for s in scales_canon],
        "weights": list(map(float, w)),
        "normalize": normalize,
        "n_samples": int(n),
        "n_qubits": int(d),
        # Optional: per-scale purity (diagonal of Ks) can be informative
        "per_scale_diag_means": [float(np.mean(np.diag(Ks))) for Ks in per_scale_contrib],
    }

    return K, meta
