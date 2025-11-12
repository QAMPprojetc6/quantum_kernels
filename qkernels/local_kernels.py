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
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity, DensityMatrix
from feature_maps import get_feature_map_spec


def _eigenclip_psd(K: np.ndarray, clip: float = 1e-12) -> np.ndarray:
    w, V = np.linalg.eigh((K + K.T) * 0.5)
    w[w < clip] = 0.0
    return (V * w) @ V.T

def _normalize_kernel(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K = K / d[:, None] / d[None, :]
    np.fill_diagonal(K, 1.0)
    return K

def _aggregate(grams: List[np.ndarray], agg: str, weights: Optional[List[float]]) -> np.ndarray:
    if agg == "weighted":
        w = np.asarray(weights, float)
        w = w / w.sum()
        return sum(w[i] * grams[i] for i in range(len(grams)))
    return sum(grams) / float(len(grams))



def build_kernel(
    X: np.ndarray,
    feature_map: str = "zz_manual",     # "zz_manual" | "zz_manual_canonical" | "zz_qiskit"
    depth: int = 1,
    entanglement: Optional[str] = None, # "linear" | "ring"
    partitions: Optional[Iterable[Iterable[int]]] = None,
    method: str = "rdm",        # "subcircuits" or "rdm"
    agg: str = "mean",                  # "mean" or "weighted"
    weights: Optional[List[float]] = None,
    seed: int = 42,
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

    X = np.asarray(X, dtype=float)
    n, d = X.shape

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
                rho_full = DensityMatrix(sv)
                rhoP = partial_trace(rho_full, traced_out)
                rhos.append(rhoP)
            m = len(rhos)
            G = np.empty((m, m), float)
            for i in range(m):
                G[i, i] = 1.0
                for j in range(i+1, m):
                    f = state_fidelity(rhos[i], rhos[j], validate=False)
                    G[i, j] = G[j, i] = float(f)
            grams.append(G)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Aggregate patches, then clean up
    K = _aggregate(grams, agg, weights)
    K = 0.5 * (K + K.T)
    K = _normalize_kernel(K)
    K = _eigenclip_psd(K)
    K = _normalize_kernel(K)

    evals = np.linalg.eigvalsh((K + K.T) * 0.5)
    off = K.copy(); np.fill_diagonal(off, np.nan)

    meta = dict(
        feature_map=feature_map,
        depth=depth,
        entanglement=entanglement,
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
