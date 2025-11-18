import numpy as np
import pytest

from qkernels.feature_maps import get_feature_map_spec


def _mk_x(d: int):
    # fixed vector of angles (radians)
    return np.array([(i + 1) * np.pi / (2 * d) for i in range(d)], dtype=np.float64)


# -------------------------
# Alias resolution
# -------------------------
@pytest.mark.parametrize(
    "name,impl_expected",
    [
        ("zz_qiskit", "qiskit"),
        ("qiskit_zz", "qiskit"),
        ("zz_library", "qiskit"),
        ("zz", "qiskit"),  # by design, zz -> qiskit
        ("zz_manual", "manual"),
        ("manual_zz", "manual"),
        ("zz-hand", "manual"),
        ("zz_manual_canonical", "manual-canonical"),
        ("manual_zz_canonical", "manual-canonical"),
        ("zz-hand-canonical", "manual-canonical"),
        ("canonical", "manual-canonical"),
    ],
)
def test_aliases_map_to_expected_impl(name, impl_expected):
    spec = get_feature_map_spec(name=name, depth=1, num_qubits=4)
    assert spec["impl"] == impl_expected
    assert spec["num_qubits"] == 4
    assert callable(spec["builder"])


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        get_feature_map_spec(name="zz_unknown", depth=1, num_qubits=2)


# -------------------------
# Entanglement defaults
# -------------------------
def test_entanglement_defaults():
    # qiskit default -> linear
    s_q = get_feature_map_spec(name="zz_qiskit", depth=1, num_qubits=3)
    assert s_q["entanglement"] == "linear"

    # manual default -> ring
    s_m = get_feature_map_spec(name="zz_manual", depth=1, num_qubits=3)
    assert s_m["entanglement"] == "ring"

    # manual-canonical default -> ring
    s_c = get_feature_map_spec(name="zz_manual_canonical", depth=1, num_qubits=3)
    assert s_c["entanglement"] == "ring"


def test_manual_rejects_invalid_entanglement():
    with pytest.raises(ValueError):
        # Entanglement invalid in manual
        get_feature_map_spec(name="zz_manual", depth=1, num_qubits=3, entanglement="fully_connected")


# -------------------------
# Builder behavior (type/shape checks)
# -------------------------
@pytest.mark.parametrize(
    "name",
    [
        "zz_qiskit",
        "zz_manual",
        "zz_manual_canonical",
    ],
)
def test_builder_returns_bound_circuit(name):
    d = 4
    spec = get_feature_map_spec(name=name, depth=2, num_qubits=d)  # depth > 1 to cover ring
    x_i = _mk_x(d)
    qc = spec["builder"](x_i)

    # Correct type and qubits
    from qiskit import QuantumCircuit
    assert isinstance(qc, QuantumCircuit)
    assert qc.num_qubits == d

    # The builder must return a circuit with bound parameters (no free symbols)
    assert len(qc.parameters) == 0


def test_builder_rejects_wrong_shape():
    d = 3
    spec = get_feature_map_spec(name="zz_qiskit", depth=1, num_qubits=d)
    x_bad = np.array([0.1, 0.2], dtype=np.float64)  # incorrect size

    with pytest.raises(ValueError):
        spec["builder"](x_bad)

    spec_m = get_feature_map_spec(name="zz_manual", depth=1, num_qubits=d)
    with pytest.raises(ValueError):
        spec_m["builder"](np.ones((d, 1)))  # no 1D


def test_qiskit_entanglement_variants_bind_ok():
    d = 4
    x = _mk_x(d)
    # Try a couple of common variants supported by zz_feature_map
    for ent in ("linear", "full"):
        spec = get_feature_map_spec(name="zz_qiskit", depth=1, num_qubits=d, entanglement=ent)
        qc = spec["builder"](x)
        assert qc.num_qubits == d
        assert len(qc.parameters) == 0
