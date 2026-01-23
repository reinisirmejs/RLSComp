# Import Qiskit libraries for quantum circuit creation and manipulation
from qiskit import qasm2, qasm3, transpile

# Import Qualtran libraries for sparse state preparation and T-complexity analysis
from qualtran.bloqs.state_preparation import SparseStatePreparationAliasSampling

# Import cirq functions
import cirq
from cirq import qasm

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

def list_to_coeff_map(lst, n):
    """Convert a list of strings to a coefficient map for a DFA."""
    coeff_map = {}
    for s in lst:
        if len(s) != n:
            raise ValueError(f"String '{s}' does not match the expected length {n}.")
        int_s = int(s,2)
        coeff_map[int_s] = 1  # Assign coefficient 1 to each string
    return coeff_map

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

def cirq_to_qiskit(cirq_circuit):
    qasm_str = qasm(cirq_circuit)
    qiskit_circuit = qasm2.loads(qasm_str)
    return qiskit_circuit