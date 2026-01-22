import numpy as np

from src.regex_utils import *
from src.mps_utils import *

# Import Qiskit libraries for quantum circuit creation and manipulation
from qiskit import qasm2, qasm3, QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation

# Import Qualtran libraries for sparse state preparation and T-complexity analysis
from qualtran.bloqs.state_preparation import SparseStatePreparationAliasSampling
from qualtran.cirq_interop.t_complexity_protocol import t_complexity

# Import cirq functions
import cirq
from cirq import qasm
from cirq.contrib.qasm_import import circuit_from_qasm

from src.bartschi2019_dicke import dicke_state
from src.gleinig_sparse import sparse_gleinig, build_coeffs
from src.qualtran_sparse import get_qualtran_circuit_depth, list_to_coeff_map


def get_qiskit_stats(amplitudes, num_qubits):
    """
    Generate and analyze a quantum circuit using Qiskits StatePreparation.
    
    When a Statevector argument is passed the state is prepared based on the Isometry synthesis described in [1].

    [1] Iten et al., Quantum circuits for isometries (2016). Phys. Rev. A 93, 032318.
    
    Args:
        amplitudes (array): Normalized amplitudes for the quantum state to prepare
        num_qubits (int): Number of qubits in the circuit
        verbose (bool): Whether to print detailed statistics
        
    Returns:
        tuple: (circuit depth, gate counts dictionary, total gate count, nonlocal gate count)
    """

    # Create a StatePreparation circuit with the provided amplitudes
    stateprep = StatePreparation(amplitudes)
    # Initialize a quantum circuit with the specified number of qubits
    qc = QuantumCircuit(num_qubits)
    # Add the state preparation operation to the circuit
    qc.append(stateprep, range(num_qubits))
    
    # Transpile the circuit to a standard gate set for hardware implementation
    # Note: using the gate set for the Heron devices.
    qc_t = transpile(qc, basis_gates=['cz', 'rz', 'sx', 'x', 'id'])

    # Calculate statistics about the transpiled circuit
    depth = qc_t.depth()  # Circuit depth (longest path)
    gate_counts = qc_t.count_ops()  # Dictionary of gate types and their counts
    total_gates = sum(gate_counts.values())  # Total number of gates
    nonlocal_gates = qc_t.num_nonlocal_gates()  # Number of multi-qubit gates

    return depth, gate_counts, total_gates, nonlocal_gates, qc_t

def get_our_cost_from_circ(circuit):
    """
    Get the cost of our circuit from a given circuit.
    
    Args:
        circuit (qiskit.Circuit): The circuit to analyze
        
    Returns:
        tuple: (circuit depth, number of qubits)
    """
    our_circuit_t = transpile(circuit, basis_gates=['cx', 'rz', 'sx', 'x', 'id'],optimization_level=3)
    return our_circuit_t.depth(), our_circuit_t.num_qubits, our_circuit_t

def get_bartschi2019_stats(system_size, k):
    circ = dicke_state(system_size, k)
    circ_t = transpile(circ, basis_gates=['cx', 'rz', 'sx', 'x', 'id'],optimization_level=3)
    return circ_t.depth(), circ_t.num_qubits, circ_t

def get_gleinig_sparse_stats(system_size, bit_string_list):
    coeffs = np.ones(len(bit_string_list), dtype=complex)
    coeffs = build_coeffs(bit_string_list, coeffs)
    circ, count_CNOTS, count_1qb_gates = sparse_gleinig(bit_string_list, coeffs, system_size)
    circ_t = transpile(circ, basis_gates=['cx', 'rz', 'sx', 'x', 'id'],optimization_level=3)
    return circ_t.depth(), circ_t.num_qubits, circ_t

def get_qualtran_sparse_stats(system_size, bit_string_list, mu=3):
    coeff_map = list_to_coeff_map(bit_string_list, system_size)
    depth, qubits, circ = get_qualtran_circuit_depth(coeff_map, system_size, mu=mu)
    return depth, qubits, circ

    