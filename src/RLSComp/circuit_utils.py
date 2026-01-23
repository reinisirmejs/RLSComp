import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, Isometry
from .mps_utils import get_no_ancilla_unitaries_from_MPS, get_staircase_padded_mps, get_tree_decomposition, get_circuit_info_from_tree_rectangular

"""
Functions to convert the isometries to unitary operators and then to quantum circuits.
"""

def check_unitary(U):
    n = len(U)
    return np.allclose(np.eye(n), U@U.conj().T) and np.allclose(np.eye(n), U.conj().T@U)

def unitary_matrix_to_circuit(U):
    n_qubits = int(np.log2(U.shape[0]))
    U_ckt = QuantumCircuit(n_qubits)
    gate = UnitaryGate(U, check_input=True)
    U_ckt.append(gate, range(n_qubits))
    return U_ckt

def tree_isometries_to_circuit(isometries_rectangular, act_on_qubits):
    num_qubits =len(act_on_qubits)
    U_ckt = QuantumCircuit(num_qubits)
    gate = Isometry(isometries_rectangular, 0,0)
    U_ckt.append(gate, range(num_qubits))
    return U_ckt

"""
Functions to convert the MPS to quantum circuits. In these functions we directly feed in the specific isometries from before
"""

""" 
Option 1: This function takes the unitaries directly from get_no_ancilla_unitaries_from_MPS and creates a circuit with no ancilla qubits at all.
Here we directly exploit the MPS structure and carefully create unitary operators directly from the isometries
"""

def MPS_to_circuit_SeqRLSP(MPS_LIST):
    unitaries, _ = get_no_ancilla_unitaries_from_MPS(MPS_LIST)
    system_size = len(unitaries)
    big_ckt = QuantumCircuit(system_size)
    for i in range(system_size):
        unitary = unitaries[i]
        shape = np.shape(unitary)
        nqubits = int(np.log2(shape[0]))
        U_ckt = unitary_matrix_to_circuit(unitary)
        act_on_qubits = list(range(i, i+nqubits))
        big_ckt.append(U_ckt, act_on_qubits)
    return big_ckt

def MPS_to_circuit_SeqIsoRLSP(MPS_LIST):
    isometries, _ = get_no_ancilla_unitaries_from_MPS(MPS_LIST,rectangular=True)
    system_size = len(isometries)
    big_ckt = QuantumCircuit(system_size)
    for i in range(system_size):
        isometry = isometries[i]
        shape = np.shape(isometry)
        nqubits = int(np.log2(shape[0]))
        act_on_qubits = list(range(i, i+nqubits))
        U_ckt = tree_isometries_to_circuit(isometry, act_on_qubits)
        big_ckt.append(U_ckt, act_on_qubits)
    return big_ckt

"""
Option 2: This function prepares the tree method, which should have a log(N) depth
"""

def Tree_to_circuit(MPS_LIST):
    system_size = len(MPS_LIST)
    padded_mps = get_staircase_padded_mps(MPS_LIST)
    tree = get_tree_decomposition(padded_mps)
    circuit_layers = get_circuit_info_from_tree_rectangular(system_size,tree)
    big_ckt = QuantumCircuit(system_size)
    for circuit_layer in circuit_layers:
        acted_on_qubits = []
        for unit_tuple in circuit_layer:
            isometry = unit_tuple['matrix']
            act_on_qubits = unit_tuple['qubits']
            acted_on_qubits.extend(act_on_qubits)
            U_ckt = tree_isometries_to_circuit(isometry,act_on_qubits)
            big_ckt.append(U_ckt, act_on_qubits)
        assert len(acted_on_qubits) == len(set(acted_on_qubits)), "Some qubits are acted on more than once in the same layer!"
    return big_ckt

    



