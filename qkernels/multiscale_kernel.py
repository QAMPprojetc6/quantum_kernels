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
from .baseline_kernel import build_kernel_cross as _baseline_kernel_cross
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


def _statevectors_for_samples(
    X: np.ndarray,
    fmap_name: str,
    depth: int,
    entanglement: Optional[str] = None,
) -> List[Statevector]:
    """
    Build the feature-map circuit for each sample and return its statevector.
    """
    n, d = X.shape
    spec = get_feature_map_spec(
        name=fmap_name,
        depth=depth,
        num_qubits=d,
        entanglement=entanglement,
    )
    builder = spec["builder"]

    svs: List[Statevector] = []
    for i in range(n):
        params = np.asarray(X[i], dtype=np.float64).ravel()
        qc = builder(params)
        svs.append(Statevector(qc))
    return svs


def _rdm_for_patch(sv: Statevector, d: int, patch: Tuple[int, ...]) -> DensityMatrix:
    """
    Compute the reduced density matrix (RDM) on a given patch by tracing out the complement.
    IMPORTANT: we trace starting from the Statevector to avoid materializing the full 2^d x 2^d density matrix.
    """
    all_idx = set(range(d))
    trace_out = sorted(all_idx - set(patch))
    # partial_trace(Statevector, trace_out) returns a (small) DensityMatrix on the kept subsystem
    return partial_trace(sv, trace_out)


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
    Build K^(s) for one scale by averaging, per patch:
      - If patch is the full system: fidelity = |<psi_i|psi_j>|^2  (no density matrix materialization)
      - Otherwise: HS inner product of patch RDMs Tr(rho_i^p rho_j^p)
    """
    n = len(svs)
    Ks = np.zeros((n, n), dtype=np.float64)
    inv_m = 1.0 / float(len(patches))

    # Accumulate per-patch contributions
    for patch in patches:
        if len(patch) == d:
            # Full-system patch: HS(pure,pure) == fidelity
            for a in range(n):
                va = svs[a].data
                for b in range(a, n):
                    vb = svs[b].data
                    fid = float(np.abs(np.vdot(va, vb)) ** 2)
                    Ks[a, b] += fid
                    if b != a:
                        Ks[b, a] += fid
        else:
            # Local patch: compute small RDMs via partial_trace(Statevector, trace_out)
            rdms = [_rdm_for_patch(svs[i], d, patch) for i in range(n)]
            for a in range(n):
                for b in range(a, n):
                    hs = _hs_inner(rdms[a], rdms[b])
                    Ks[a, b] += hs
                    if b != a:
                        Ks[b, a] += hs

    Ks *= inv_m
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
    entanglement = kwargs.pop("entanglement", None)

    # Build statevectors for all samples
    svs = _statevectors_for_samples(X, fmap_name=feature_map, depth=depth, entanglement=entanglement)

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
        "entanglement": entanglement,
        # Optional: per-scale purity (diagonal of Ks) can be informative
        "per_scale_diag_means": [float(np.mean(np.diag(Ks))) for Ks in per_scale_contrib],
    }

    return K, meta


def build_kernel_cross(
    X: np.ndarray,
    X_ref: np.ndarray,
    feature_map: str = "zz",
    depth: int = 1,
    backend: str = "statevector",
    seed: int = 42,
    scales: Optional[List[Iterable[Iterable[int]]]] = None,
    weights: Optional[List[float]] = None,
    normalize: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """
    Cross-kernel K(X, X_ref) for the multiscale kernel (for Nystrom use).
    """
    if backend != "statevector":
        raise NotImplementedError("multiscale_kernel_cross supports backend='statevector' only.")

    X = np.asarray(X, dtype=float)
    X_ref = np.asarray(X_ref, dtype=float)
    if X.ndim != 2 or X_ref.ndim != 2:
        raise ValueError("X and X_ref must be 2D arrays.")
    n, d = X.shape
    m, d_ref = X_ref.shape
    if d_ref != d:
        raise ValueError("X and X_ref must have the same feature dimension.")

    if scales is None:
        scales = _ensure_scales_default(d)
    scales_canon = _validate_scales(scales, d)
    if weights is None:
        weights = [1.0 / len(scales_canon)] * len(scales_canon)
    w_scales = _validate_weights(weights, len(scales_canon))

    entanglement = kwargs.pop("entanglement", None)

    spec_full = get_feature_map_spec(feature_map, depth=depth, num_qubits=d, entanglement=entanglement)
    builder_full = spec_full["builder"]

    K = np.zeros((n, m), dtype=np.float64)
    diag_X = np.zeros(n, dtype=np.float64)
    diag_ref = np.zeros(m, dtype=np.float64)

    for s_idx, patches in enumerate(scales_canon):
        w_s = float(w_scales[s_idx])
        num_patches = len(patches)

        K_s = np.zeros((n, m), dtype=np.float64)
        diag_X_s = np.zeros(n, dtype=np.float64)
        diag_ref_s = np.zeros(m, dtype=np.float64)

        full_patches = [p for p in patches if len(p) == d]
        rdm_patches = [p for p in patches if len(p) != d]

        # Full-patch contributions (fidelity)
        for _ in full_patches:
            K_full = _baseline_kernel_cross(
                X,
                X_ref,
                feature_map=feature_map,
                depth=depth,
                backend=backend,
                seed=seed,
                entanglement=entanglement,
            )
            K_s += K_full
            diag_X_s += 1.0
            diag_ref_s += 1.0

        # RDM/HS contributions for local patches
        rhos_ref_by_patch = []
        diag_ref_by_patch = []
        for P in rdm_patches:
            traced_out = [q for q in range(d) if q not in P]
            rhos_ref = []
            diag_ref_patch = np.zeros(m, dtype=np.float64)
            for j in range(m):
                qc = builder_full(np.asarray(X_ref[j], dtype=np.float64).ravel())
                sv = Statevector.from_instruction(qc)
                rho = partial_trace(sv, traced_out)
                rhos_ref.append(rho.data)
                diag_ref_patch[j] = float(np.trace(rho.data @ rho.data).real)
            rhos_ref_by_patch.append(np.stack(rhos_ref, axis=0))
            diag_ref_by_patch.append(diag_ref_patch)

        for i in range(n):
            qc = builder_full(np.asarray(X[i], dtype=np.float64).ravel())
            sv = Statevector.from_instruction(qc)
            for p_idx, P in enumerate(rdm_patches):
                traced_out = [q for q in range(d) if q not in P]
                rho = partial_trace(sv, traced_out)
                diag_X_s[i] += float(np.trace(rho.data @ rho.data).real)
                rhos_ref_arr = rhos_ref_by_patch[p_idx]
                vals = np.einsum("ij,mji->m", rho.data, rhos_ref_arr, optimize=True).real
                K_s[i, :] += vals

        for diag_ref_patch in diag_ref_by_patch:
            diag_ref_s += diag_ref_patch

        if num_patches > 0:
            K_s /= float(num_patches)
            diag_X_s /= float(num_patches)
            diag_ref_s /= float(num_patches)

        K += w_s * K_s
        diag_X += w_s * diag_X_s
        diag_ref += w_s * diag_ref_s

    if normalize:
        dX = np.sqrt(np.clip(diag_X, 1e-12, None))
        dR = np.sqrt(np.clip(diag_ref, 1e-12, None))
        K = K / dX[:, None] / dR[None, :]

    return K.astype(np.float64, copy=False)
