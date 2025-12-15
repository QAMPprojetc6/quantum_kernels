"""
Local (patch-wise) kernel.

Unified API:
    build_kernel(X, feature_map="zz", depth=1, backend="statevector", seed=42,
                 partitions=[(0,1),(2,3)], method="subcircuits"|"rdm",
                 agg="mean", weights=None, **kwargs)

Returns:
    K: (n, n) ndarray (Symmetric, Positive Semi Definite)
    meta: dict

Methods:
 - Per-patch overlaps (subcircuits)
 - Reduced density matrices (RDMs)
"""

from typing import Tuple, Dict, Any, Iterable, List, Optional
import numpy as np
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity, DensityMatrix
from .feature_maps import get_feature_map_spec


def _eigenclip_psd(
    K: np.ndarray, 
    clip: float = 1e-12
) -> np.ndarray:
    """
    Ensure the kernel matrix is positive semi-definite (PSD)
    by clipping small negative eigenvalues caused by numerical noise.

    Parameters
    ----------
    K: np.ndarray
        Input kernel matrix (must be square and symmetric or nearly symmetric).
    clip: float, optional
        Threshold below which eigenvalues are set to zero (default: 1e-12).

    Returns
    -------
    np.ndarray
        A PSD-corrected version of K.
    """
    w, V = np.linalg.eigh((K + K.T) * 0.5)
    w[w < clip] = 0.0
    return (V * w) @ V.T

def _normalize_kernel(
    K: np.ndarray
)-> np.ndarray:
    """
    Normalize the kernel matrix so that all diagonal entries are 1.

    This ensures self-similarity (K[i, i]) = 1 for every sample,
    which keeps the kernel scale-consistent across different datasets.

    Parameters
    ----------
    K: np.ndarray
        Input kernel matrix (square).

    Returns
    -------
    np.ndarray
        Normalized kernel matrix with unit diagonal.
    """
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K = K / d[:, None] / d[None, :]
    np.fill_diagonal(K, 1.0)
    return K

def _aggregate(
    grams: List[np.ndarray],
    agg: str,
    weights: Optional[List[float]]
) -> np.ndarray:
    """
    Combine multiple patch-wise kernel (Gram) matrices
    into a single global kernel using mean or weighted aggregation.

    Parameters
    ----------
    grams : List[np.ndarray]
        List of kernel matrices, one for each patch or subsystem.
    agg: str
        Aggregation strategy. Options:
        - "mean"    : simple arithmetic mean of all patch kernels
        - "weighted": weighted mean based on user-supplied weights
    weights: Optional[List[float]], optional
        List of non-negative weights (one per patch). Used only if agg = "weighted".
        The weights will be normalized to sum to 1.

    Returns
    -------
    np.ndarray
        Aggregated kernel matrix combining all patches.
    """
    if agg == "weighted":
        if weights is None:
            raise ValueError("weights must be provided when agg='weighted'.")
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or len(w) != len(grams):
            raise ValueError("weights must be a 1D list with the same length as grams.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
        s = float(w.sum())
        if s <= 0:
            raise ValueError("weights must sum to a positive value.")
        w = w / s
        return sum((w[i] * grams[i] for i in range(len(grams))), start=np.zeros_like(grams[0], dtype=float))

    return sum(grams, start=np.zeros_like(grams[0], dtype=float)) / float(len(grams))


def build_kernel(
    X: np.ndarray,
    feature_map: str = "zz_manual",  # "zz_manual" | "zz_manual_canonical" | "zz_qiskit"
    depth: int = 1,
    backend: str = "statevector",    # "statevector" | "sampling"
    seed: int = 42,
    partitions: Optional[Iterable[Iterable[int]]] = None,
    method: str = "rdm",             # "subcircuits" | "rdm"
    agg: str = "mean",               # "mean" | "weighted"
    weights: Optional[List[float]] = None,
    entanglement: Optional[str] = None,  # feature-map entanglement
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:

    """
    Patch-wise (local) kernel.

    Computes a kernel by comparing *local patches* of the quantum state per sample,
    then aggregating patch-wise similarities into a single Gram matrix.

    Parameters
    ----------
    X : np.ndarray (n_samples, d)
        Input features. `d` is interpreted as the number of qubits/features used by the feature map.
    feature_map : str
        Feature map name passed to `get_feature_map_spec` (e.g., "zz_manual", "zz_manual_canonical", "zz_qiskit").
    depth : int
        Number of feature-map repetitions (layers).
    backend : str
        Backend selector for API consistency. Currently only "statevector" is supported.
    seed : int
        Random seed (kept for reproducibility / API consistency).
    partitions : iterable of iterables of int, optional
        Patch definition, e.g. [(0,1), (2,3)]. If None, a default partitioning is used.
    method : {"subcircuits", "rdm"}
        How to compute patch similarity:
          - "subcircuits": build per-patch circuits and compare full patch states
          - "rdm": compute reduced density matrices (RDMs) from the full state
    agg : {"mean", "weighted"}
        How to aggregate patch Gram matrices into the final kernel.
    weights : list[float] or None
        Patch weights for agg="weighted". Must be non-negative and match number of patches.
        If agg="mean", this is ignored.
    entanglement : str or None
        Entanglement pattern forwarded to the feature map (e.g., "linear", "ring", "full", ...).
    **kwargs : Any
        Optional extra options. Example: `rdm_metric="fidelity"` or `rdm_metric="hs"`.

    Returns
    -------
    K : np.ndarray (n, n)
        Symmetric (approximately PSD) kernel matrix.
    meta : dict
        Metadata with configuration used (for logging/reproducibility).
    """

    X = np.asarray(X, dtype=float)
    n, d = X.shape

    backend_norm = str(backend).strip().lower()
    if backend_norm != "statevector":
        raise NotImplementedError(
            "local_kernel currently supports backend='statevector' only. "
            "If you want sampling, implement a shot-based estimator for subcircuits/RDM."
        )

    # Optional: metric for RDM similarity (kept in kwargs to avoid expanding the core signature)
    rdm_metric = str(kwargs.pop("rdm_metric", "fidelity")).strip().lower()
    if rdm_metric not in {"fidelity", "hs"}:
        raise ValueError("rdm_metric must be 'fidelity' or 'hs'.")


    Q = d  # number of qubits  

    if partitions is None:
        partitions = [tuple([i, i+1]) for i in range(0, Q - 1, 2)]  
    else:
        partitions = [tuple(part) for part in partitions]
    if len(partitions) == 0:
        raise ValueError("At least one partition must be specified.")

    grams = []
    parts = partitions

    if method == "subcircuits":
        # Build per-patch specs and states
        for P in parts:
            qP = len(P)
            specP = get_feature_map_spec(feature_map, depth=depth, num_qubits=qP, entanglement=entanglement)
            states = []
            for i in range(n):
                xP = X[i, list(P)]
                qc = specP["builder"](xP)
                sv = Statevector.from_instruction(qc).data  # 1D complex array size 2^qP
                states.append(sv)
            Phi = np.stack(states, axis=0)  # (n, 2^qP)
            G = np.abs(Phi @ Phi.conj().T) ** 2
            grams.append(G)
    
    elif method == "rdm":
        # One full spec, then partial trace per patch
        spec_full = get_feature_map_spec(feature_map, depth=depth, num_qubits=Q, entanglement=entanglement)
        full_states = []
        for i in range(n):
            qc = spec_full["builder"](X[i])
            sv = Statevector.from_instruction(qc)
            full_states.append(sv)
        for P in parts:
            traced_out = [q for q in range(Q) if q not in P]
            rhos = []
            for sv in full_states:
                # Avoid building the full density matrix (2^d x 2^d). For larger d, Qiskit's
                # partial_trace on DensityMatrix can fail (einsum index explosion) and is much heavier.
                if len(traced_out) == 0:
                    rhoP = DensityMatrix(sv)  # full system (no reduction)
                else:
                    rhoP = partial_trace(sv, traced_out)  # trace directly from Statevector
                rhos.append(rhoP)

            m = len(rhos)
            G = np.empty((m, m), float)
            for i in range(m):
                G[i, i] = 1.0
                for j in range(i+1, m):
                    if rdm_metric == "fidelity":
                        v = state_fidelity(rhos[i], rhos[j], validate=False)
                    else:
                        # Hilbert–Schmidt inner product: Tr(rho_i rho_j)
                        v = np.trace(rhos[i].data @ rhos[j].data).real
                    G[i, j] = G[j, i] = float(v)
            grams.append(G)

    else:
        raise ValueError(f"Unknown method: {method}. Try using 'subcircuits' or 'rdm'")

    # Aggregate patches, then clean up
    K = _aggregate(grams, agg, weights)
    K = 0.5 * (K + K.T)
    K = _normalize_kernel(K)
    K = _eigenclip_psd(K)
    K = _normalize_kernel(K)

    evals = np.linalg.eigvalsh((K + K.T) * 0.5)
    off = K.copy(); np.fill_diagonal(off, np.nan)

    meta = dict(
        kernel="local",
        backend=backend_norm,
        seed=seed,
        feature_map=feature_map,
        depth=depth,
        entanglement=entanglement,
        rdm_metric=rdm_metric,
        method=method,
        partitions=parts,
        agg=agg,
        weights=weights,
        n=n, d=d, Q=Q,
        eigenvalues=evals,
        offdiag_mean=np.nanmean(off),
        offdiag_std=np.nanstd(off),
    )
    return K, meta
