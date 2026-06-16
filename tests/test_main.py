import sys
import os
import pytest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RLSComp.regex_utils import accepted_strings, list_to_acdfa_direct
from RLSComp.mps_utils import MPS_to_state, ACDFA_to_MPS, get_tree_decomposition, get_state_from_tree, build_mps_from_regex, build_mps_from_DFA, build_dicke_mps_from_bitstrings, build_mps_from_bitstrings
from RLSComp.circuit_utils import MPS_to_circuit_SeqRLSP, MPS_to_circuit_SeqIsoRLSP, Tree_to_circuit
from RLSComp.benchmarking_utils import get_gleinig_sparse_stats
from RLSComp.interface import build_SeqRLSP_circuit, build_TreeRLSP_circuit
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
    print(dimSysAnc, dimSys, dimAnc)
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

def test_SeqRLSP_from_bitstrings_no_reflection_symmetry():
    bitstrings = ["0000", "0001"]
    MPS_LIST = build_mps_from_bitstrings(bitstrings)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_SeqRLSP_circuit(bitstrings)
    assert verify_state(target_state, circ)

def test_SeqRLSP_from_dfa():
    regex = "(0)*1(0)*1(0)*"
    n = 6
    _, dfa = build_mps_from_regex(regex, n)
    #assert(dfa.is_acyclic())
    MPS_LIST = build_mps_from_DFA(dfa,system_size=n)
    target_state = MPS_to_state(MPS_LIST)

    circ = build_SeqRLSP_circuit(dfa,system_size=n)
    assert verify_state(target_state, circ)

def test_SeqRLSP_from_mps():
    regex = "(0)*1(0)*"
    n = 5

    MPS_LIST, _ = build_mps_from_regex(regex, n)
    target_state = MPS_to_state(MPS_LIST)

    circ = build_SeqRLSP_circuit(MPS_LIST)

    assert verify_state(target_state, circ)



# ============================================================================================
# Tests for the Motzkin state preparation
# ============================================================================================

def test_motzkin_strings():
    from RLSComp.motzkin_utils import get_motzkin_strings
    n = 6
    strings = get_motzkin_strings(n)
    for s in strings:
        height = 0
        for c in s:
            if c == '0':
                height += 1
            else:
                height -= 1
            assert height >= 0
        assert height == 0


def test_motzkin_state_preparation():
    from RLSComp.motzkin_utils import motzkinDFA, get_motzkin_strings
    from RLSComp.regex_utils import accepted_strings_cyclic_dfa
    n = 4
    dfa = motzkinDFA(n)
    #assert(dfa.is_acyclic())
    motzkin_strings_dfa = accepted_strings_cyclic_dfa(dfa, n)
    print(motzkin_strings_dfa)
    #print(dfa)
    strings = get_motzkin_strings(n)
    target_state = np.zeros(2**n)
    for s in strings:
        index = int(s, 2)
        target_state[index] = 1/np.sqrt(len(strings))
    
    circ_strings = build_SeqRLSP_circuit(strings)
    #verify_state(target_state, circ_strings)
    assert verify_state(target_state, circ_strings)

    circ_dfa = build_SeqRLSP_circuit(dfa, n)
    assert verify_state(target_state, circ_dfa)


# ============================================================================================
# Tests for the interface — SeqRLSP with use_isometries=True
# ============================================================================================

def test_SeqRLSP_iso_from_regex():
    regex = "(0)*1(0)*1(0)*"
    n = 6
    MPS_LIST, _ = build_mps_from_regex(regex, n)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_SeqRLSP_circuit(regex, n, use_isometries=True)
    assert verify_state(target_state, circ)

def test_SeqRLSP_iso_from_bitstrings():
    bitstrings = ["00101", "01001", "10001"]
    MPS_LIST = build_mps_from_bitstrings(bitstrings)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_SeqRLSP_circuit(bitstrings, use_isometries=True)
    assert verify_state(target_state, circ)

def test_SeqRLSP_iso_from_dfa():
    bitstrings = ["100110", "000010", "001011", "111111", "000001", "111000"]
    acdfa = list_to_acdfa_direct(bitstrings)
    MPS_LIST = build_mps_from_DFA(acdfa)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_SeqRLSP_circuit(acdfa, use_isometries=True)
    assert verify_state(target_state, circ)

def test_SeqRLSP_iso_from_mps():
    regex = "(0)*1(0)*"
    n = 5
    MPS_LIST, _ = build_mps_from_regex(regex, n)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_SeqRLSP_circuit(MPS_LIST, use_isometries=True)
    assert verify_state(target_state, circ)


# ============================================================================================
# Tests for the interface — SeqRLSP with complement=True
# ============================================================================================

def test_SeqRLSP_complement_from_regex():
    regex = "(0)*1(0)*1(0)*"
    n = 5
    MPS_LIST, _ = build_mps_from_regex(regex, n, complement=True)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_SeqRLSP_circuit(regex, n, complement=True)
    assert verify_state(target_state, circ)

def test_SeqRLSP_complement_from_bitstrings():
    bitstrings = ["00101", "01001", "10001"]
    MPS_LIST = build_mps_from_bitstrings(bitstrings, complement=True)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_SeqRLSP_circuit(bitstrings, complement=True)
    assert verify_state(target_state, circ)

def test_SeqRLSP_complement_from_acyclic_dfa():
    bitstrings = ["001", "010", "100"]
    acdfa = list_to_acdfa_direct(bitstrings)
    MPS_LIST = build_mps_from_bitstrings(bitstrings, complement=True)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_SeqRLSP_circuit(acdfa, complement=True)
    assert verify_state(target_state, circ)

def test_SeqRLSP_complement_from_cyclic_dfa():
    regex = "(0)*1(0)*1(0)*"
    n = 5
    _, dfa = build_mps_from_regex(regex, n)
    MPS_LIST, _ = build_mps_from_regex(regex, n, complement=True)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_SeqRLSP_circuit(dfa, system_size=n, complement=True)
    assert verify_state(target_state, circ)

def test_SeqRLSP_error_complement_from_mps():
    MPS_LIST, _ = build_mps_from_regex("(0)*1(0)*", 5)
    with pytest.raises(ValueError, match="complement is not supported for MPS input"):
        build_SeqRLSP_circuit(MPS_LIST, complement=True)


# ============================================================================================
# Tests for the interface — TreeRLSP with all input types
# ============================================================================================

def test_TreeRLSP_from_regex():
    regex = "(0)*1(0)*1(0)*"
    n = 6
    MPS_LIST, _ = build_mps_from_regex(regex, n)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_TreeRLSP_circuit(regex, n)
    assert verify_state(target_state, circ)

def test_TreeRLSP_from_bitstrings():
    bitstrings = ["00101", "01001", "10001", "10010"]
    MPS_LIST = build_mps_from_bitstrings(bitstrings)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_TreeRLSP_circuit(bitstrings)
    assert verify_state(target_state, circ)

def test_TreeRLSP_from_dfa():
    bitstrings = ["100110", "000010", "001011", "111111", "000001", "111000"]
    acdfa = list_to_acdfa_direct(bitstrings)
    MPS_LIST = build_mps_from_DFA(acdfa)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_TreeRLSP_circuit(acdfa)
    assert verify_state(target_state, circ)

def test_TreeRLSP_from_mps():
    regex = "(0)*1(0)*"
    n = 6
    MPS_LIST, _ = build_mps_from_regex(regex, n)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_TreeRLSP_circuit(MPS_LIST)
    assert verify_state(target_state, circ)


# ============================================================================================
# Tests for the interface — TreeRLSP with complement=True
# ============================================================================================

def test_TreeRLSP_complement_from_regex():
    regex = "(0)*1(0)*1(0)*"
    n = 5
    MPS_LIST, _ = build_mps_from_regex(regex, n, complement=True)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_TreeRLSP_circuit(regex, n, complement=True)
    assert verify_state(target_state, circ)

def test_TreeRLSP_complement_from_bitstrings():
    bitstrings = ["00101", "01001", "10001"]
    MPS_LIST = build_mps_from_bitstrings(bitstrings, complement=True)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_TreeRLSP_circuit(bitstrings, complement=True)
    assert verify_state(target_state, circ)

def test_TreeRLSP_complement_from_acyclic_dfa():
    bitstrings = ["001", "010", "100"]
    acdfa = list_to_acdfa_direct(bitstrings)
    MPS_LIST = build_mps_from_bitstrings(bitstrings, complement=True)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_TreeRLSP_circuit(acdfa, complement=True)
    assert verify_state(target_state, circ)

def test_TreeRLSP_complement_from_cyclic_dfa():
    regex = "(0)*1(0)*1(0)*"
    n = 5
    _, dfa = build_mps_from_regex(regex, n)
    MPS_LIST, _ = build_mps_from_regex(regex, n, complement=True)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_TreeRLSP_circuit(dfa, system_size=n, complement=True)
    assert verify_state(target_state, circ)

def test_TreeRLSP_error_complement_from_mps():
    MPS_LIST, _ = build_mps_from_regex("(0)*1(0)*", 5)
    with pytest.raises(ValueError, match="complement is not supported for MPS input"):
        build_TreeRLSP_circuit(MPS_LIST, complement=True)


# ============================================================================================
# Tests for the interface — error / validation cases for build_SeqRLSP_circuit
# ============================================================================================

def test_SeqRLSP_error_regex_no_system_size():
    with pytest.raises(ValueError, match="system_size must be provided"):
        build_SeqRLSP_circuit("(0)*1(0)*")

def test_SeqRLSP_error_bitstrings_with_system_size():
    with pytest.raises(ValueError, match="system_size must not be provided"):
        build_SeqRLSP_circuit(["00101", "01001"], system_size=5)

def test_SeqRLSP_error_mixed_length_bitstrings():
    with pytest.raises(ValueError, match="same length"):
        build_SeqRLSP_circuit(["001", "0100", "111"])

def test_SeqRLSP_error_acyclic_dfa_with_system_size():
    acdfa = list_to_acdfa_direct(["00101", "01001"])
    assert acdfa.is_acyclic()
    with pytest.raises(ValueError, match="system_size must not be provided"):
        build_SeqRLSP_circuit(acdfa, system_size=5)

def test_SeqRLSP_error_cyclic_dfa_no_system_size():
    from RLSComp.motzkin_utils import motzkinDFA
    dfa = motzkinDFA(4)
    assert not dfa.is_acyclic()
    with pytest.raises(ValueError, match="system_size must be provided"):
        build_SeqRLSP_circuit(dfa)

def test_SeqRLSP_error_mps_with_system_size():
    MPS_LIST, _ = build_mps_from_regex("(0)*1(0)*", 5)
    with pytest.raises(ValueError, match="system_size must not be provided"):
        build_SeqRLSP_circuit(MPS_LIST, system_size=5)

def test_SeqRLSP_error_invalid_type():
    with pytest.raises(TypeError, match="Unsupported input type"):
        build_SeqRLSP_circuit(42)


# ============================================================================================
# Tests for the interface — error / validation cases for build_TreeRLSP_circuit
# ============================================================================================

def test_TreeRLSP_error_regex_no_system_size():
    with pytest.raises(ValueError, match="system_size must be provided"):
        build_TreeRLSP_circuit("(0)*1(0)*")

def test_TreeRLSP_error_bitstrings_with_system_size():
    with pytest.raises(ValueError, match="system_size must not be provided"):
        build_TreeRLSP_circuit(["00101", "01001"], system_size=5)

def test_TreeRLSP_error_mixed_length_bitstrings():
    with pytest.raises(ValueError, match="same length"):
        build_TreeRLSP_circuit(["001", "0100", "111"])

def test_TreeRLSP_from_cyclic_dfa():
    regex = "(0)*1(0)*1(0)*"
    n = 6
    _, dfa = build_mps_from_regex(regex, n)
    MPS_LIST = build_mps_from_DFA(dfa, system_size=n)
    target_state = MPS_to_state(MPS_LIST)
    circ = build_TreeRLSP_circuit(dfa, system_size=n)
    assert verify_state(target_state, circ)

def test_TreeRLSP_error_acyclic_dfa_with_system_size():
    acdfa = list_to_acdfa_direct(["00101", "01001"])
    assert acdfa.is_acyclic()
    with pytest.raises(ValueError, match="system_size must not be provided"):
        build_TreeRLSP_circuit(acdfa, system_size=5)

def test_TreeRLSP_error_cyclic_dfa_no_system_size():
    from RLSComp.motzkin_utils import motzkinDFA
    dfa = motzkinDFA(4)
    assert not dfa.is_acyclic()
    with pytest.raises(ValueError, match="system_size must be provided"):
        build_TreeRLSP_circuit(dfa)

def test_TreeRLSP_error_mps_with_system_size():
    MPS_LIST, _ = build_mps_from_regex("(0)*1(0)*", 5)
    with pytest.raises(ValueError, match="system_size must not be provided"):
        build_TreeRLSP_circuit(MPS_LIST, system_size=5)

def test_TreeRLSP_error_invalid_type():
    with pytest.raises(TypeError, match="Unsupported input type"):
        build_TreeRLSP_circuit(42)


# ============================================================================================
# README examples — each test mirrors a code snippet from the README verbatim
# ============================================================================================

def test_readme_package_import():
    """README: from RLSComp import build_SeqRLSP_circuit, build_TreeRLSP_circuit"""
    from RLSComp import build_SeqRLSP_circuit, build_TreeRLSP_circuit
    assert callable(build_SeqRLSP_circuit)
    assert callable(build_TreeRLSP_circuit)

def test_readme_SeqRLSP_regex():
    """README regex example — W state on 5 qubits."""
    from RLSComp import build_SeqRLSP_circuit
    regex = "(0)*1(0)*"
    num_qubits = 5
    circuit = build_SeqRLSP_circuit(regex, num_qubits)
    MPS_LIST, _ = build_mps_from_regex(regex, num_qubits)
    assert verify_state(MPS_to_state(MPS_LIST), circuit)

def test_readme_SeqRLSP_bitstring_list():
    """README bitstring list example."""
    from RLSComp import build_SeqRLSP_circuit
    bitstring_list = ["000", "010", "111"]
    circuit = build_SeqRLSP_circuit(bitstring_list)
    MPS_LIST = build_mps_from_bitstrings(bitstring_list)
    assert verify_state(MPS_to_state(MPS_LIST), circuit)

def test_readme_SeqRLSP_dfa():
    """README DFA example — acyclic DFA, no system_size."""
    from RLSComp import build_SeqRLSP_circuit
    dfa = list_to_acdfa_direct(["000", "010", "111"])
    circuit = build_SeqRLSP_circuit(dfa)
    MPS_LIST = build_mps_from_bitstrings(["000", "010", "111"])
    assert verify_state(MPS_to_state(MPS_LIST), circuit)

def test_readme_SeqRLSP_mps():
    """README MPS example."""
    from RLSComp import build_SeqRLSP_circuit
    MPS_LIST, _ = build_mps_from_regex("(0)*1(0)*", 5)
    circuit = build_SeqRLSP_circuit(MPS_LIST)
    assert verify_state(MPS_to_state(MPS_LIST), circuit)

def test_readme_TreeRLSP_regex():
    """README: build_TreeRLSP_circuit admits the same inputs — regex."""
    from RLSComp import build_TreeRLSP_circuit
    regex = "(0)*1(0)*"
    num_qubits = 5
    circuit = build_TreeRLSP_circuit(regex, num_qubits)
    MPS_LIST, _ = build_mps_from_regex(regex, num_qubits)
    assert verify_state(MPS_to_state(MPS_LIST), circuit)

def test_readme_TreeRLSP_bitstring_list():
    """README: build_TreeRLSP_circuit admits the same inputs — bitstring list."""
    from RLSComp import build_TreeRLSP_circuit
    bitstring_list = ["000", "010", "111"]
    circuit = build_TreeRLSP_circuit(bitstring_list)
    MPS_LIST = build_mps_from_bitstrings(bitstring_list)
    assert verify_state(MPS_to_state(MPS_LIST), circuit)

def test_readme_TreeRLSP_dfa():
    """README: build_TreeRLSP_circuit admits the same inputs — acyclic DFA."""
    from RLSComp import build_TreeRLSP_circuit
    dfa = list_to_acdfa_direct(["000", "010", "111"])
    circuit = build_TreeRLSP_circuit(dfa)
    MPS_LIST = build_mps_from_bitstrings(["000", "010", "111"])
    assert verify_state(MPS_to_state(MPS_LIST), circuit)

def test_readme_TreeRLSP_mps():
    """README: build_TreeRLSP_circuit admits the same inputs — MPS."""
    from RLSComp import build_TreeRLSP_circuit
    MPS_LIST, _ = build_mps_from_regex("(0)*1(0)*", 5)
    circuit = build_TreeRLSP_circuit(MPS_LIST)
    assert verify_state(MPS_to_state(MPS_LIST), circuit)