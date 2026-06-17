"""
Tests requiring the full dependency set from misc/requirements.txt:
  matplotlib, pyyaml, qiskit-aer, qualtran, cirq-core, ply

Run with:
  pytest tests/test_full.py
"""
import sys
import os
import pytest
import numpy as np

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit"),
    pytest.mark.filterwarnings("ignore::UserWarning:cotengra"),
]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RLSComp.regex_utils import list_to_acdfa_direct
from RLSComp.mps_utils import MPS_to_state, ACDFA_to_MPS, build_mps_from_bitstrings, is_mps
from qiskit.quantum_info import Statevector


def verify_state(target_state, circuit):
    statevector = Statevector.from_instruction(circuit)
    dimSysAnc = len(statevector.data)
    dimSys = len(target_state)
    dimAnc = dimSysAnc // dimSys
    n_ancilla = int(np.ceil(np.log2(dimAnc)))
    statevector_no_ancilla = statevector.data[0::(2 ** n_ancilla)]
    overlap = np.abs(np.dot(target_state, statevector_no_ancilla.conj()))
    return np.allclose(overlap, 1)


# ============================================================================================
# Tests for Bartschi 2019 Dicke state preparation
# ============================================================================================

def test_bartschi_2019():
    from RLSComp.bartschi2019_dicke import dicke_state, test_circuit_qasm
    dicke_circ = dicke_state(6, 2)
    counts = test_circuit_qasm(dicke_circ)
    assert len(counts) == 15
    for k in counts.keys():
        assert k.count('1') == 2
    dicke_circ = dicke_state(6, 3)
    counts = test_circuit_qasm(dicke_circ)
    assert len(counts) == 20
    for k in counts.keys():
        assert k.count('1') == 3


# ============================================================================================
# Tests for Gleinig sparse state preparation
# ============================================================================================

def test_gleinig_sparse():
    from RLSComp.benchmarking_utils import get_gleinig_sparse_stats
    n = 6
    bitstring_list = ["100110", "000010", "001011", "111111", "000001", "111000"]
    acdfa = list_to_acdfa_direct(bitstring_list)
    MPS_LIST = ACDFA_to_MPS(acdfa)
    target_state = MPS_to_state(MPS_LIST)
    depth, qubits, circ = get_gleinig_sparse_stats(n, bitstring_list)
    assert verify_state(target_state, circ)


# ============================================================================================
# Tests for run_experiment.py
# ============================================================================================

def test_run_experiment_build_mps_regex():
    from RLSComp.run_experiment import build_mps
    n = 5
    MPS_LIST = build_mps("regex", "(0)*1(0)*", [], n)
    assert is_mps(MPS_LIST)
    assert len(MPS_LIST) == n

def test_run_experiment_build_mps_regex_complement():
    from RLSComp.run_experiment import build_mps
    n = 5
    MPS_LIST = build_mps("regex", "(0)*1(0)*", [], n, complement=True)
    assert is_mps(MPS_LIST)
    assert len(MPS_LIST) == n

def test_run_experiment_build_mps_bitstring_list():
    from RLSComp.run_experiment import build_mps
    bitstrings = ["001", "010", "100"]
    MPS_LIST = build_mps("bitstring_list", None, bitstrings, None)
    assert is_mps(MPS_LIST)
    assert len(MPS_LIST) == len(bitstrings[0])

def test_run_experiment_build_mps_motzkin():
    from RLSComp.run_experiment import build_mps
    n = 4
    MPS_LIST = build_mps("motzkin", None, [], n)
    assert is_mps(MPS_LIST)
    assert len(MPS_LIST) == n

def test_run_experiment_build_mps_invalid_type():
    from RLSComp.run_experiment import build_mps
    with pytest.raises(ValueError, match="Unsupported input_type"):
        build_mps("unknown", None, [], 4)

def test_run_experiment_run_SeqRLSP():
    from RLSComp.run_experiment import run_SeqRLSP
    n = 5
    config = {"input_type": "regex", "regex": "(0)*1(0)*", "regex_complement": False}
    qubits, depth, _, all_gates, time_taken, _ = run_SeqRLSP(n, config)
    assert qubits == n
    assert depth > 0
    assert all_gates > 0
    assert time_taken >= 0

def test_run_experiment_run_SeqIsoRLSP():
    from RLSComp.run_experiment import run_SeqIsoRLSP
    n = 5
    config = {"input_type": "regex", "regex": "(0)*1(0)*", "regex_complement": False}
    qubits, depth, _, all_gates, time_taken, _ = run_SeqIsoRLSP(n, config)
    assert qubits == n
    assert depth > 0
    assert all_gates > 0
    assert time_taken >= 0

def test_run_experiment_run_TreeRLSP():
    from RLSComp.run_experiment import run_TreeRLSP
    n = 5
    config = {"input_type": "regex", "regex": "(0)*1(0)*", "regex_complement": False}
    qubits, depth, _, all_gates, time_taken, _ = run_TreeRLSP(n, config)
    assert qubits == n
    assert depth > 0
    assert all_gates > 0
    assert time_taken >= 0

def test_run_experiment_method_max_n():
    from RLSComp.run_experiment import run_experiment
    import math
    config = {
        "input_type": "regex",
        "regex": "(0)*1(0)*",
        "regex_complement": False,
        "system_sizes": [4, 6, 8],
        "methods": ["SeqRLSP", "TreeRLSP"],
        "method_max_n": {"TreeRLSP": 5},
    }
    results = run_experiment(config)
    assert all(v > 0 for v in results["SeqRLSP_depth"])
    assert results["TreeRLSP_depth"][0] > 0
    assert math.isnan(results["TreeRLSP_depth"][1])
    assert math.isnan(results["TreeRLSP_depth"][2])

def test_run_experiment_end_to_end():
    from RLSComp.run_experiment import run_experiment
    config = {
        "input_type": "regex",
        "regex": "(0)*1(0)*",
        "regex_complement": False,
        "system_sizes": [4, 6],
        "methods": ["SeqRLSP", "TreeRLSP"],
    }
    results = run_experiment(config)
    assert results["sizes"] == [4, 6]
    for method in ["SeqRLSP", "TreeRLSP"]:
        for field in ["qubits", "depth", "2gates", "gates", "time"]:
            assert len(results[f"{method}_{field}"]) == 2
        assert all(v > 0 for v in results[f"{method}_depth"])

def test_run_experiment_timeout():
    from RLSComp.run_experiment import run_experiment
    config = {
        "input_type": "regex",
        "regex": "(0)*1(0)*",
        "regex_complement": False,
        "system_sizes": [4],
        "methods": ["SeqRLSP", "TreeRLSP"],
        "timeout": 60,
    }
    results = run_experiment(config)
    for method in ["SeqRLSP", "TreeRLSP"]:
        assert all(v > 0 for v in results[f"{method}_depth"])

def test_run_experiment_timeout_skips_subsequent():
    from RLSComp.run_experiment import run_experiment, RUNNERS, _MethodTimeout
    import math
    original = RUNNERS["SeqRLSP"]
    def fake_runner(n, config):
        raise _MethodTimeout()
    RUNNERS["SeqRLSP"] = fake_runner
    try:
        config = {
            "input_type": "regex",
            "regex": "(0)*1(0)*",
            "regex_complement": False,
            "system_sizes": [4, 6, 8],
            "methods": ["SeqRLSP"],
            "timeout": 60,
        }
        results = run_experiment(config)
        assert all(math.isnan(v) for v in results["SeqRLSP_depth"])
    finally:
        RUNNERS["SeqRLSP"] = original

def test_run_experiment_readme_config():
    """Mirrors the README example config with all listed methods at small system sizes."""
    from RLSComp.run_experiment import run_experiment
    config = {
        "input_type": "regex",
        "regex": "(0)*1(0)*",
        "regex_complement": False,
        "system_sizes": [4, 5],
        "methods": ["SeqRLSP", "TreeRLSP", "qualtran", "qiskit", "gleinig_sparse", "bartschi2019_dicke"],
        "ancilla_dim": 2,
    }
    results = run_experiment(config)
    assert results["sizes"] == [4, 5]
    for method in config["methods"]:
        for field in ["qubits", "depth", "2gates", "gates", "time"]:
            assert len(results[f"{method}_{field}"]) == 2
    for method in ["SeqRLSP", "TreeRLSP"]:
        assert all(v > 0 for v in results[f"{method}_depth"])
