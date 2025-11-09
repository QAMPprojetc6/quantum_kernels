"""
Shared quantum feature maps.

Two ZZ options are provided:
  - "zz_manual": a hand-rolled ZZ-style feature map (simple Rx + CZ entanglement).
  - "zz_qiskit": Qiskit's circuit-library `zz_feature_map` function.

API
---
get_feature_map_spec(
    name: str = "zz_manual",
    depth: int = 1,
    num_qubits: int = 4,
    entanglement: str | None = None
) -> dict
    Returns a dictionary with:
      - "name": normalized name ("zz_manual" | "zz_qiskit")
      - "impl": "manual" | "qiskit"
      - "depth": int
      - "num_qubits": int
      - "entanglement": str
      - "builder": fn(params: np.ndarray) -> qiskit.QuantumCircuit

Notes
-----
- The builder expects a 1D array `params` of length == num_qubits.
- For depth > 1, both implementations reuse the same feature vector each layer.
- The manual implementation is ZZ-style but not identical to Qiskit's ZZFeatureMap.
"""

from typing import Callable, Dict, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map, RZZGate


# -------------------------------
# Manual ZZ-style feature map (simplified version)
# -------------------------------
def _zz_manual_builder_factory(
    num_qubits: int,
    depth: int,
    entanglement: str,
) -> Callable[[np.ndarray], QuantumCircuit]:
    """
    Returns a builder for a simple, hand-rolled ZZ-style feature map.

    Encoding (per layer):
      - Apply Rx(data[i]) on each qubit i
      - Apply CZ entanglement according to `entanglement`:
          * "linear": CZ on (0-1, 1-2, ..., n-2 - n-1)
          * "ring"  : same as linear + CZ(n-1, 0)
    """
    ent = entanglement.lower()
    if ent not in {"linear", "ring"}:
        raise ValueError(f"Unsupported entanglement for manual map: {entanglement!r}. Use 'linear' or 'ring'.")

    def builder(params: np.ndarray) -> QuantumCircuit:
        if params.ndim != 1:
            raise ValueError("params must be a 1D array.")
        if params.size != num_qubits:
            raise ValueError(f"params.size must be {num_qubits} (got {params.size}).")

        qc = QuantumCircuit(num_qubits)

        for _ in range(depth):
            # data re-uploading layer
            for q in range(num_qubits):
                qc.rx(float(params[q]), q)
            # entanglement pattern
            for q in range(num_qubits - 1):
                qc.cz(q, q + 1)
            if ent == "ring" and num_qubits > 2:
                qc.cz(num_qubits - 1, 0)

        return qc

    return builder


# -------------------------------
# Manual ZZ feature map (canonical version)
# -------------------------------
def zz_manual_canonical_builder_factory(num_qubits: int, depth: int, entanglement: str, alpha: float = 1.0):
    """
    Canonical-like ZZ map:
      - H on all qubits (once)
      - For each layer:
          * RZ(x_i) on each qubit i
          * RZZ(alpha * x_i * x_j) on entangled pairs
    entanglement: "linear" or "ring"
    """
    ent = entanglement.lower()
    if ent not in {"linear", "ring"}:
        raise ValueError("entanglement must be 'linear' or 'ring'.")

    def builder(x: np.ndarray) -> QuantumCircuit:
        if x.ndim != 1 or x.size != num_qubits:
            raise ValueError(f"x must be 1D with size {num_qubits}.")
        qc = QuantumCircuit(num_qubits)
        # 1) Hadamards
        qc.h(range(num_qubits))
        for _ in range(depth):
            # 2) Local Z encodings
            for q in range(num_qubits):
                qc.rz(float(x[q]), q)
            # 3) ZZ interactions
            for q in range(num_qubits - 1):
                theta = alpha * float(x[q]) * float(x[q + 1])
                qc.append(RZZGate(theta), [q, q + 1])
            if ent == "ring" and num_qubits > 2:
                theta = alpha * float(x[-1]) * float(x[0])
                qc.append(RZZGate(theta), [num_qubits - 1, 0])
        return qc

    return builder


# -------------------------------
# Qiskit zz_feature_map wrapper
# -------------------------------
def _zz_qiskit_builder_factory(
    num_qubits: int,
    depth: int,
    entanglement: str,
) -> Callable[[np.ndarray], QuantumCircuit]:
    """
    Returns a builder that wraps Qiskit's `zz_feature_map` function.

    - feature_dimension = num_qubits
    - reps = depth
    - entanglement = provided (e.g., "linear", "full", "pairwise", "circular", ...)
    """
    fmap = zz_feature_map(
        feature_dimension=num_qubits,
        reps=depth,
        entanglement=entanglement,
    )

    def builder(params: np.ndarray) -> QuantumCircuit:
        if params.ndim != 1:
            raise ValueError("params must be a 1D array.")
        if params.size != num_qubits:
            raise ValueError(f"params.size must be {num_qubits} (got {params.size}).")
        # Bind by order
        qc = fmap.assign_parameters(params, inplace=False)
        return qc

    return builder


# -------------------------------
# Public factory
# -------------------------------
def get_feature_map_spec(
    name: str = "zz_manual",
    depth: int = 1,
    num_qubits: int = 4,
    entanglement: Optional[str] = None,
) -> Dict:
    """
    Return a feature map spec (metadata + circuit builder).

    Parameters
    ----------
    name : str
        Aliases (case-insensitive):
          - qiskit: "zz_qiskit", "qiskit_zz", "zz_library", "zz"
          - manual: "zz_manual", "manual_zz", "zz-hand"
          - manual (canonical): "zz_manual_canonical", "manual_zz_canonical", "zz-hand-canonical", "canonical"
        NOTE: "zz" maps to the Qiskit variant by default.
    depth : int
        Number of layers (reps).
    num_qubits : int
        Number of qubits.
    entanglement : Optional[str]
        Manual: "linear" | "ring" (default: "ring")
        Qiskit: any string accepted by `zz_feature_map` (default: "linear")

    Returns
    -------
    dict with keys: {"name", "impl", "depth", "num_qubits", "entanglement", "builder"}
    """
    key = name.strip().lower()

    qiskit_aliases = {"zz_qiskit", "qiskit_zz", "zz_library", "zz"}
    manual_aliases = {"zz_manual", "manual_zz", "zz-hand"}
    manual_canonical_aliases = {"zz_manual_canonical", "manual_zz_canonical", "zz-hand-canonical", "canonical"}

    if key in qiskit_aliases:
        impl = "qiskit"
        norm_name = "zz_qiskit"
        ent = entanglement or "linear"
        builder = _zz_qiskit_builder_factory(num_qubits=num_qubits, depth=depth, entanglement=ent)
    elif key in manual_aliases:
        impl = "manual"
        norm_name = "zz_manual"
        ent = entanglement or "ring"
        builder = _zz_manual_builder_factory(num_qubits=num_qubits, depth=depth, entanglement=ent)
    elif key in manual_canonical_aliases:
        impl = "manual-canonical"
        norm_name = "zz_manual_canonical"
        ent = entanglement or "ring"
        builder = zz_manual_canonical_builder_factory(num_qubits=num_qubits, depth=depth, entanglement=ent)
    else:
        supported = sorted(list(qiskit_aliases | manual_aliases | manual_canonical_aliases))
        raise ValueError(f"Unknown feature map '{name}'. Supported: {supported}")

    return {
        "name": norm_name,
        "impl": impl,
        "depth": depth,
        "num_qubits": num_qubits,
        "entanglement": ent,
        "builder": builder,
    }




"""
# Examples of use

x_i = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4], dtype=np.float64)

# Manual, ring (default)
spec = get_feature_map_spec("zz_manual", depth=1, num_qubits=4)
qc1 = spec["builder"](x_i)

# Manual, lineal chain
spec = get_feature_map_spec("zz_manual", depth=1, num_qubits=4, entanglement="linear")
qc2 = spec["builder"](x_i)

# Manual canonical, lineal chain
spec = get_feature_map_spec("zz_manual_canonical", depth=1, num_qubits=4, entanglement="linear")
qc3 = spec["builder"](x_i)

# Qiskit library, entanglement "full"
spec = get_feature_map_spec("zz_qiskit", depth=1, num_qubits=4, entanglement="full")
qc4 = spec["builder"](x_i)


"""