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

from src.dicke_states_bartschi2019 import dicke_state
from src.gleinig_sparse import sparse_gleinig, build_coeffs


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

def remove_control(input_moment):
    classical_control_class = cirq.ops.classically_controlled_operation.ClassicallyControlledOperation
    input_moment = list(input_moment)
    output_moment = []
    where_classically_controlled = [moment_item.__class__ is classical_control_class for moment_item in input_moment]
    for i in range(len(where_classically_controlled)):
        if where_classically_controlled[i]:
            new_gate = input_moment[i].without_classical_controls()
            output_moment.append(new_gate)
        else:
            output_moment.append(input_moment[i])
    return cirq.Moment(output_moment)

def get_qiskit_circuit(circuit, transpile_circuit=False):
    qasm_str = circuit.to_qasm(header='')
    insert_text = ' \n\nOPENQASM 3.0;\ninclude "stdgates.inc";\ngate sxdg q1 {pow(-0.5) @ x q1;}\n\n\n'
    qasm_str_split = qasm_str.split('//')
    qasm_str_split[0] = insert_text
    qasm_str_alt = '//'.join(qasm_str_split)
    qiskit_circuit = qasm3.loads(qasm_str_alt)
    if transpile_circuit:
        qiskit_circuit = transpile(qiskit_circuit, basis_gates=['cx', 'rz', 'sx', 'x', 'id'])
    return qiskit_circuit

def decompose_cswap(moment):
    ctrl = moment.qubits[0]
    x = moment.qubits[1]
    y = moment.qubits[2]
    circuit = cirq.Circuit()
    circuit += [cirq.CNOT(y, x)]
    circuit += [cirq.CNOT(ctrl, x), cirq.H(y)]
    circuit += [cirq.T(ctrl), cirq.T(x) ** -1, cirq.T(y)]
    circuit += [cirq.CNOT(y, x)]
    circuit += [cirq.CNOT(ctrl, y), cirq.T(x)]
    circuit += [cirq.CNOT(ctrl, x), cirq.T(y) ** -1]
    circuit += [cirq.T(x) ** -1, cirq.CNOT(ctrl, y)]
    circuit += [cirq.CNOT(y, x)]
    circuit += [cirq.T(x), cirq.H(y)]
    circuit += [cirq.CNOT(y, x)]
    return circuit

def clean_circuit(op):
    cleaned_circuit = []
    depth_addition = 0
    decomp_circuit = cirq.Circuit(cirq.decompose_once(op))
    for j in range(len(decomp_circuit.moments)):
        moment_circuit = cirq.Circuit(decomp_circuit.moments[j])
        moment_circuit_decomp = cirq.Circuit(cirq.decompose(moment_circuit))
        for i in range(len(moment_circuit_decomp.moments)):
            moment_circuit_decomp_i = cirq.Circuit(moment_circuit_decomp).moments[i]
            try:
                # Try to see that the moment can be written into openqasm
                _ = cirq_to_qiskit(cirq.Circuit(moment_circuit_decomp_i))
                cleaned_circuit.append(moment_circuit_decomp_i)
            except:
                try:
                    # If it doesn't work then it could be the case that the qubits are 'CleanQubits'
                    # So just change these to NamedQubits
                    tmp = cirq.map_clean_and_borrowable_qubits(cirq.Circuit(moment_circuit_decomp_i))
                    _ = get_qiskit_circuit(cirq.Circuit(tmp))
                    cleaned_circuit.append(tmp)
                except:
                    try:
                        # If that doesn't work then it could be that we have a classical control
                        # For now, we just remove the classical control and put the gate in as if it wasn't controlled
                        # This may have an impact on the depth, but might also be fine depending on the hardware
                        tmp_decompose = cirq.decompose(tmp.moments[0])
                        tmp_new = remove_control(tmp_decompose)
                        _ = get_qiskit_circuit(cirq.Circuit(tmp_new))
                        cleaned_circuit.append(tmp_new)
                    except:
                        try:
                            # Finally, it could be a controlled swap gate, which we can just write this circuit explicitly 
                            # instead
                            cswap_circuit = decompose_cswap(list(tmp.moments[0])[0])
                            _ = get_qiskit_circuit(cswap_circuit)
                            cleaned_circuit.append(cswap_circuit.moments)
                        except:
                            print(f'Failed: Moment {j}, sub-Moment {i}')
                            print(len(tmp.moments))
                            print(tmp.moments)
                            depth_addition +=1
                            continue   
    return cirq.Circuit(cleaned_circuit)


def get_qualtran_circuit_depth(coeff_map, num_qubits, mu=3):
    N = 2**num_qubits
    # Create a sparse state preparation circuit using alias sampling technique
    # The precision parameter controls the error in the prepared state
    sparse_state_prep_alias = SparseStatePreparationAliasSampling.from_sparse_dict(
        coeff_map, N, precision=2**-mu / len(coeff_map)
    )
    
    circuit = sparse_state_prep_alias.as_composite_bloq().to_cirq_circuit()
    op = next(circuit.all_operations())
    cleaned_circuit = clean_circuit(op)
    qiskit_circuit = get_qiskit_circuit(cleaned_circuit, transpile_circuit=True)
    our_circuit_t = transpile(qiskit_circuit, basis_gates=['cx', 'rz', 'sx', 'x', 'id'],optimization_level=3)
    return our_circuit_t.depth(), our_circuit_t.num_qubits, our_circuit_t


def get_qualtran_circuit_clean(coeff_map, num_qubits, mu=3):
    N = 2**num_qubits
    # Create a sparse state preparation circuit using alias sampling technique
    # The precision parameter controls the error in the prepared state
    sparse_state_prep_alias = SparseStatePreparationAliasSampling.from_sparse_dict(
        coeff_map, N, precision=2**-mu / len(coeff_map)
    )
    circuit = sparse_state_prep_alias.as_composite_bloq().to_cirq_circuit()
    op = next(circuit.all_operations())
    cleaned_circuit = clean_circuit(op)
    qiskit_circuit = get_qiskit_circuit(cleaned_circuit, transpile_circuit=True)
    return qiskit_circuit

def get_qualtran_stats(coeff_map, n_qubits, mu=3):
    """Generate and analyze a quantum circuit using Qualtran's SparseStatePreparationAliasSampling.
    
    Initialize a $d$-sparse state over $L$ indices using coherent alias sampling.

    In particular, we take the zero state to:

    $$
        sum_{j=0}^{d-1} sqrt{p_{mathrm{ind}_j}} |mathrm{ind}_j rangle |mathrm{temp}_jrangle
    $$

    where $mathrm{ind}_j in [0, L)$ is the index of the $j$-th non-zero coefficient,
    and the probabilities $p_l$ are $mu$-bit binary approximations to the true values,
    and the register $|mathrm{temp}_j rangle$ may be entangled with the index register.

    This construction is designed to work specifically when you don't require specific phases,
    and the problem is reduced to [classical alias sampling]
    (https://en.wikipedia.org/wiki/Alias_method). We sample `l` with probability `p[l]` by first
    selecting `l` uniformly at random and then returning it with probability `keep[l] / 2**mu`;
    otherwise returning `alt[l]`.

    In comparison to the non-sparse implementation , this uses $lceil log d rceil$ extra 
    ancilla qubits, and reduces the iteration length to $d$ from $L$.

    Registers:
        selection: The input/output register $|mathrm{ind}_l rangle$ of size lg(L) where the desired
            coefficient state is prepared.
        sigma_mu: A mu-sized register containing uniform probabilities for comparison against `keep`.
        sparse_index: A lg(d)-sized register storing the sparse index $j in [0, d)$.
        alt: A lg(L)-sized register of alternate indices
        keep: a mu-sized register of probabilities of keeping the initially sampled index.
        less_than_equal: one bit for the result of the comparison.

    This gate corresponds to the following operations:
    - UNIFORM_d on the `sparse_index` register.
    - H^mu on the `sigma` register.
    - QROM addressed by the `sparse_index` register into the `selection`, `alt`, and `keep` signature.
    - LessThanEqualGate comparing the `keep` and `sigma` registers.
    - Coherent swap between the `selection` and `alt` registers if the comparison returns True.

    Total space will be $(2 log(L) + log(d) + 2 mu + 1)$ work qubits + $log(L)$ ancillas for QROM.

    References:
        [Qubitization of Arbitrary Basis Quantum Chemistry Leveraging Sparsity and Low Rank Factorization](https://arxiv.org/abs/1902.02134).
        Berry et al. (2019). Section 5, Eqs. 43, 44.

        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et al. (2018). Section III.D. and Figure 11.
    
    Args:
        coeff_map (dict): Dictionary mapping basis state indices to their coefficients
        n_qubits (int): Number of qubits in the circuit
        mu (int): Precision parameter for the state preparation, controls error
        
    Returns:
        tuple: (T gate count, Clifford gate count, rotation gate count)
    """
    # Calculate the total Hilbert space dimension
    N = 2**n_qubits
    
    # Create a sparse state preparation circuit using alias sampling technique
    # The precision parameter controls the error in the prepared state
    sparse_state_prep_alias = SparseStatePreparationAliasSampling.from_sparse_dict(
        coeff_map, N, precision=2**-mu / len(coeff_map)
    )
    
    # Calculate T-complexity metrics for resource estimation
    # This analyzes the number of T gates, Clifford gates, and rotation gates
    t_comp = t_complexity(sparse_state_prep_alias)
    return t_comp.t, t_comp.clifford, t_comp.rotations

def qiskit_to_cirq(qiskit_circuit):
    qasm_string = qasm2.dumps(qiskit_circuit)
    cirq_circuit = circuit_from_qasm(qasm_string)
    return cirq_circuit

def cirq_to_qiskit(cirq_circuit):
    qasm_str = qasm(cirq_circuit)
    qiskit_circuit = qasm2.loads(qasm_str)
    return qiskit_circuit

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

    