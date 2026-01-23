import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RLSComp.regex_utils import accepted_strings, list_to_acdfa_direct
from RLSComp.mps_utils import MPS_to_state, ACDFA_to_MPS, get_tree_decomposition, get_state_from_tree, build_mps_from_regex, build_mps_from_DFA, build_dicke_mps_from_bitstrings, build_mps_from_bitstrings
from RLSComp.circuit_utils import MPS_to_circuit_SeqRLSP, MPS_to_circuit_SeqIsoRLSP, Tree_to_circuit
from RLSComp.benchmarking_utils import get_gleinig_sparse_stats
from RLSComp.interface import build_SeqRLSP_circuit
from qiskit.quantum_info import Statevector

"""
Functions to check if we have prepared the correct state given isometries. These algorithms are inefficient
and only work for small system sizes. 
"""

def verify_state(target_state, circuit):
    statevector = Statevector.from_instruction(circuit)
    dimSysAnc = len(statevector.data)
    dimSys = len(target_state)
    dimAnc = dimSysAnc//dimSys
    n_ancilla = int(np.ceil(np.log2(dimAnc)))
    # Assuming that the ancilla at the end decouples into the all 0 state,
    # The relevant system eigenvector is the every 2**n_ancilla element
    statevector_no_ancilla = statevector.data[0::(2**n_ancilla)]
    overlap = np.abs(np.dot(target_state, statevector_no_ancilla.conj()))
    return np.allclose(overlap,1)

def dictionary_from_state(statevector):
    N = len(statevector)
    statevector_dict = {}
    for i in range(N):
        if abs(statevector[i])>1e-12:
            statevector_dict[bin(i)[2:].zfill(int(np.log2(N)))] = statevector[i]
    return statevector_dict

# ============================================================================================
# Tests for Regex
# ============================================================================================

def test_build_mps_from_regex():
    """Test for verifying that regex gives the correct strings"""
    regex = "(0)*1(0)*1(0)*"
    n = 6
    MPS_LIST, dfa = build_mps_from_regex(regex, n)
    assert dfa is not None
    strings = accepted_strings(dfa, n)
    for s in strings:
        assert s.count('1') == 2
    assert len(strings) > 0
   
def test_build_dicke_mps_from_bitstrings():
    """Test for verifying that the bitstring input method give the correct strings"""
    n, k = 5, 2
    MPS_LIST, dfa = build_dicke_mps_from_bitstrings(n, k)
    assert dfa is None
    state = MPS_to_state(MPS_LIST)
    d = dictionary_from_state(state)
    assert len(d) == n*(n-1)//2  # C(5,2) = 10
    for s in d.keys():
        assert s.count('1') == 2

# ============================================================================================
# Tests for Complements
# ============================================================================================

def test_complement_approach():
    """Test for verifying that regex gives the correct complement strings"""
    regex = "(0)*1(0)*1(0)*"
    n = 6
    MPS_LIST, dfa = build_mps_from_regex(regex, n,complement=True)
    assert dfa is not None
    strings = accepted_strings(dfa, n)
    #instead of giving the correct amount of strings, it gives a superposition of all strings
    assert len(strings) ==2**n - n*(n-1)//2
    for s in strings:
        assert s.count('1') != 2

def test_list_complement_approach():
    """Test for verifying that the list complement approach gives the correct strings"""
    bitstring_list = ["100110", "000010", "001011", "111111", "000001", "111000"]
    acdfa = list_to_acdfa_direct(bitstring_list, complement=True)
    MPS_LIST = ACDFA_to_MPS(acdfa)
    state = MPS_to_state(MPS_LIST)
    d = dictionary_from_state(state)
    assert len(d) == 2**len(bitstring_list[0]) - len(bitstring_list)  # 2^6 - 6
    for s in d.keys():
        assert s not in bitstring_list  # all strings should be different from the input list

def test_multiple_of_3_approach():
    regex = "(1(0(1)*0)*1|0)*"
    n = 6
    MPS_LIST, dfa = build_mps_from_regex(regex, n)
    state = MPS_to_state(MPS_LIST)
    d = dictionary_from_state(state)
    for s in d.keys():
        s_int = int(s, 2)
        assert s_int % 3 == 0  # all strings should be multiples of 3
    assert len(d) > 0  # there should be at least one multiple of

# ============================================================================================
# Tests for Bartschi 2019 Dicke state preparation
# ============================================================================================    

def test_bartschi_2019():
    from RLSComp.bartschi2019_dicke import dicke_state, test_circuit_qasm
    dicke_circ = dicke_state(6,2)
    counts = test_circuit_qasm(dicke_circ)
    assert len(counts)==15
    for k in counts.keys():
        assert k.count('1') == 2
    dicke_circ = dicke_state(6,3)
    counts = test_circuit_qasm(dicke_circ)
    assert len(counts)==20
    for k in counts.keys():
        assert k.count('1') == 3

# ============================================================================================
# Tests for Gleinig sparse state preparation
# ============================================================================================   

def test_gleinig_sparse():
    n=6
    bitstring_list = ["100110", "000010", "001011", "111111", "000001", "111000"]
    acdfa = list_to_acdfa_direct(bitstring_list)
    MPS_LIST = ACDFA_to_MPS(acdfa)
    target_state = MPS_to_state(MPS_LIST)
    depth, qubits, circ = get_gleinig_sparse_stats(n, bitstring_list)
    assert verify_state(target_state, circ)

# ============================================================================================
# Tests for Qualtran sparse state preparation
# ============================================================================================  

# def test_qualtran_sparse():
#     from src.benchmarking_utils import get_qualtran_sparse_stats
#     n=3
#     bitstring_list = ["100", "000", "011"]
#     # acdfa = list_to_acdfa_direct(bitstring_list)
#     # MPS_LIST = ACDFA_to_MPS(acdfa)
#     # target_state = MPS_to_state(MPS_LIST)
#     depth, qubits, circ = get_qualtran_sparse_stats(n, bitstring_list,mu=3)
#     assert True

# ============================================================================================
# Tests for SeqIsoRLSP and SeqRLSP
# ============================================================================================

def test_SeqRLSP_W():
    regex = "(0)*1(0)*"
    n = 10
    MPS_LIST, dfa = build_mps_from_regex(regex, n)
    target_state = MPS_to_state(MPS_LIST)
    #unitaries, ancilla_state = get_no_ancilla_unitaries_from_MPS(MPS_LIST)
    circ = MPS_to_circuit_SeqRLSP(MPS_LIST)
    assert verify_state(target_state, circ)

def test_SeqRLSP_dicke():
    regex = "(0)*1(0)*1(0)*1(0)*"
    n = 8
    MPS_LIST, dfa = build_mps_from_regex(regex, n)
    target_state = MPS_to_state(MPS_LIST)
    #unitaries, ancilla_state = get_no_ancilla_unitaries_from_MPS(MPS_LIST)
    circ = MPS_to_circuit_SeqRLSP(MPS_LIST)
    assert verify_state(target_state, circ)

def test_SeqIsoRLSP_dicke():
    regex = "(0)*1(0)*1(0)*1(0)*"
    n = 8
    MPS_LIST, dfa = build_mps_from_regex(regex, n)
    target_state = MPS_to_state(MPS_LIST)
    circ = MPS_to_circuit_SeqIsoRLSP(MPS_LIST)
    assert verify_state(target_state, circ)

def test_SeqIsoRLSP_product():
    bitstring_list = ["000000"]
    acdfa = list_to_acdfa_direct(bitstring_list)
    MPS_LIST = ACDFA_to_MPS(acdfa)
    target_state = MPS_to_state(MPS_LIST)
    circ = MPS_to_circuit_SeqIsoRLSP(MPS_LIST)
    assert verify_state(target_state, circ)

def test_SeqIsoRLSP_arbitrary():
    bitstring_list = ["100110", "000010", "001011", "111111", "000001", "111000"]
    acdfa = list_to_acdfa_direct(bitstring_list)
    MPS_LIST = ACDFA_to_MPS(acdfa)
    target_state = MPS_to_state(MPS_LIST)
    circ = MPS_to_circuit_SeqIsoRLSP(MPS_LIST)
    assert verify_state(target_state, circ)

# ============================================================================================
# Tests for the tree decomposition
# ============================================================================================

def test_correct_state_from_tree():
    regex = "(0)*1(0)*1(0)*"
    n = 5 #Here system size 6 fails while 4,8 etc., ones that are multiples of 4 seem to work. 
    MPS_LIST, dfa = build_mps_from_regex(regex, n)
    state_from_regex = MPS_to_state(MPS_LIST)
    tree = get_tree_decomposition(MPS_LIST)
    state_from_tree = get_state_from_tree(tree)
    assert np.allclose(state_from_regex, state_from_tree, atol=1e-12)

def test_tree_circuit_dicke_2_rectangular():
    n = 7
    regex = "(0)*1(0)*1(0)*" 
    MPS_LIST, dfa = build_mps_from_regex(regex, n)
    target_state =  MPS_to_state(MPS_LIST)
    circ = Tree_to_circuit(MPS_LIST)
    assert verify_state(target_state, circ)

def test_no_ancilla_arbitrary_rectangular():
    n=6
    bitstring_list = ["100110", "000010", "001011", "111111", "000001", "111000"]
    acdfa = list_to_acdfa_direct(bitstring_list)
    MPS_LIST = ACDFA_to_MPS(acdfa)
    target_state = MPS_to_state(MPS_LIST)
    circ = Tree_to_circuit(MPS_LIST)
    assert verify_state(target_state, circ)


# ============================================================================================
# Tests for the interface
# ============================================================================================

def test_SeqRLSP_from_regex():
    regex = "(0)*1(0)*1(0)*"
    n = 7
    MPS_LIST, _ = build_mps_from_regex(regex, n)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_SeqRLSP_circuit(
        regex,n
    )
    assert verify_state(target_state, circ)

def test_SeqRLSP_from_bitstrings():
    bitstrings = [
        "00101",
        "01001",
        "10001",
    ]
    MPS_LIST = build_mps_from_bitstrings(bitstrings)
    target_state = MPS_to_state(MPS_LIST)

    circ = build_SeqRLSP_circuit(bitstrings)
    assert verify_state(target_state, circ)

def test_SeqRLSP_from_dfa():
    regex = "(0)*1(0)*1(0)*"
    n = 6
    _, dfa = build_mps_from_regex(regex, n)
    MPS_LIST = build_mps_from_DFA(dfa)
    target_state = MPS_to_state(MPS_LIST)

    circ = build_SeqRLSP_circuit(dfa)
    assert verify_state(target_state, circ)

def test_SeqRLSP_from_mps():
    regex = "(0)*1(0)*"
    n = 5

    MPS_LIST, _ = build_mps_from_regex(regex, n)
    target_state = MPS_to_state(MPS_LIST)

    circ = build_SeqRLSP_circuit(MPS_LIST)

    assert verify_state(target_state, circ)


