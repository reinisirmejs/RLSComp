import numpy as np
from qiskit import QuantumCircuit

from qiskit.circuit.library import UCRYGate, UnitaryGate

import numpy as np
import scipy.linalg

from itertools import combinations

def bitstrings_with_k_ones(n: int, k: int) -> list[int]:
    """
    Return a list of integers whose n-bit binary representation
    contains exactly k ones.
    """
    if not (0 <= k <= n):
        return []

    result = []

    # choose which positions contain 1s
    # positions indexed 0..n-1 (LSB = position 0)
    for positions in combinations(range(n), k):
        value = 0
        for p in positions:
            value |= (1 << p)
        result.append(value)

    return result

def generate_column_for_dicke_final_gate(i, k, ancillary_qubits):
    current_ones = i
    ones_missing = k - current_ones
    if ones_missing > ancillary_qubits:
        raise ValueError(f"Cannot resolve state with {current_ones} ones and {ancillary_qubits} ancillary qubits.")
    strings_in_superposition = bitstrings_with_k_ones(ancillary_qubits, ones_missing)
    column = np.zeros(2**ancillary_qubits, dtype=complex)
    for s in strings_in_superposition:
        column[s] = 1/np.sqrt(len(strings_in_superposition))
    return column

def dicke_final_gate_general(k: int) -> np.ndarray:
    if k < 0:
        raise ValueError("k must be nonnegative")

    ancillary_qubits = int(np.ceil(np.log2(k + 1)))
    dim = 2 ** ancillary_qubits

    # Indices of columns you want to specify
    start = max(0, k - ancillary_qubits)
    end = min(k, dim - 1)
    specified = list(range(start, end + 1))
    specified_set = set(specified)

    U = np.zeros((dim, dim), dtype=complex)

    if len(specified) == 0:
        return np.eye(dim, dtype=complex)

    # Collect specified columns into a matrix C (dim x m)
    cols = []
    for i in specified:
        col = np.asarray(
            generate_column_for_dicke_final_gate(i, k, ancillary_qubits),
            dtype=complex
        ).reshape(-1)
        if col.shape[0] != dim:
            raise ValueError(f"Generated column has length {col.shape[0]}, expected {dim}")
        cols.append(col)
        U[:, i] = col

    C = np.stack(cols, axis=1)  # dim x m

    # Compute orthonormal basis for orthogonal complement of span(C)
    # null_space(C^H) gives vectors v s.t. C^H v = 0, i.e. v orthogonal to all columns of C
    N = scipy.linalg.null_space(C.conj().T)  # dim x (dim - rank(C))

    open_cols = [j for j in range(dim) if j not in specified_set]

    if N.shape[1] != len(open_cols):
        raise ValueError(
            f"Null space dimension {N.shape[1]} does not match number of open columns {len(open_cols)}. "
            "This usually means the specified columns are not full-rank (linearly dependent)."
        )

    # Fill remaining columns in a deterministic order
    for idx, j in enumerate(open_cols):
        U[:, j] = N[:, idx]

    qc = QuantumCircuit(ancillary_qubits)
    # If you want this U to act on (q0,q1) in the |q0 q1> sense,
    # you typically append on [q1,q0] because Qiskit is little-endian.
    qc.append(UnitaryGate(U), [k for k in range(ancillary_qubits)])

    return qc


def dicke_control_counting_gate(k, theta_arr):
    ancillary_qubits = np.ceil(np.log2(k+1)).astype(int)
    qc = QuantumCircuit(ancillary_qubits+1)  # q0,q1 are controls; q2 is target
    # angle_list is ordered by control state: 00, 01, 10, 11
    ucry = UCRYGate(theta_arr)
    act_on_qubits = [q for q in range(ancillary_qubits,-1,-1)]
    qc.append(ucry, act_on_qubits)
    return qc

def generate_theta_rotation_array(n, k):
    ancillary_qubits = np.ceil(np.log2(k+1)).astype(int)
    theta_arr = np.zeros(2**ancillary_qubits)
    for i in range(k):
        theta_arr[i] = 2*np.arcsin(np.sqrt((k-i)/(n)))
    return list(theta_arr)


def dicke_k_state_circuit(n,k):
    if n<k:
        raise ValueError(f"Dicke state with {k} excitations requires at least {k} qubits.")
    circ = QuantumCircuit(n)
    ancillary_qubits = np.ceil(np.log2(k+1)).astype(int)
    for i in range(n-ancillary_qubits):
        ### First we apply the proper rotation, controlled by the counting qubits ###
        theta_arr = generate_theta_rotation_array(n-i, k)
        counting_gate = dicke_control_counting_gate(k, theta_arr)
        acting_on_qubits = [j for j in range(ancillary_qubits+i, i-1, -1)]  # controls first, then target
        circ.append(counting_gate, acting_on_qubits)

        ### Second we apply the counting gate which updates the ancillas ###
        for j in range(ancillary_qubits):
            circ.mcx([k for k in range(i, i+ancillary_qubits-j)], i+ancillary_qubits-j, mode="noancilla")

        ### Third we SWAP the ancillas away by one spot ###
        if i<n-ancillary_qubits-1:
            for j in range(ancillary_qubits):
                circ.swap(i+ancillary_qubits-j, i+ancillary_qubits-j+1)

    ### Fourth we apply the final gate which resolves the ancillas ### 
    final_gate = dicke_final_gate_general(k)
    circ.append(final_gate, [k for k in range(n-ancillary_qubits, n)])  # controls first, then target
    return circ


def dicke_3_final_gate():
    s = 1 / np.sqrt(2)
    # U defined in the basis (|00>,|01>,|10>,|11>) for qubits ordered as |q0 q1>
    U = np.array([
        [0,   0,  0,  1],
        [s,   0,  s,  0],
        [-s,  0,  s,  0],
        [0,   1,  0,  0],
    ], dtype=complex)
    qc = QuantumCircuit(2)
    # If you want this U to act on (q0,q1) in the |q0 q1> sense,
    # you typically append on [q1,q0] because Qiskit is little-endian.
    qc.append(UnitaryGate(U), [0, 1])
    return qc


def dicke_3_state_circuit(n):
    if n < 3:
        raise ValueError("Dicke state with 3 excitations requires at least 3 qubits.")
    circ = QuantumCircuit(n)
    for i in range(n-2):
        #circ.x(1)
        #circ.x(2)
        # theta_arr = [-2*np.arcsin(np.sqrt(3/(n-i))), -2*np.arcsin(np.sqrt(2/(n-i))),-2*np.arcsin(np.sqrt(1/(n-i))),0]
        # print(np.sin(np.array(theta_arr)/2))
        theta_arr = generate_theta_rotation_array(n-i, 3)
        counting_gate = dicke_control_counting_gate(3, theta_arr)
        acting_on_qubits = [j for j in range(2+i, i-1, -1)]  # controls first, then target
        circ.append(counting_gate, acting_on_qubits)
        circ.ccx(i,i+1,i+2)
        circ.cx(i, i+1)
        if i<n-3:
            circ.swap(i+2, i+3)
            circ.swap(i+1, i+2)


    circ.append(dicke_3_final_gate(), [n-2, n-1])  # controls first, then target
    
    return circ

def w_state_circuit(n):
    circ = QuantumCircuit(n)
    for i in range(n-1):
        theta = 1/np.sqrt(n-i)
        circ.x(i+1)
        circ.cry(2*np.arcsin(theta), i+1, i)  
        circ.x(i+1)
        circ.cx(i, i+1)
        if i+2 < n:
            circ.swap(i+1, i+2)
    circ.x(n-1)
    return circ



if __name__ == "__main__":
    n = 5
    k=3
    dicke_k_state_circuit(n,k)