import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator
import emps
import scipy.sparse as sp
from scipy.linalg import qr
from qiskit.circuit.library import UnitaryGate


def make_unitary(A0, A1):
    """
    Given two X x X matrices A0 and A1 that satisfy A0^† A0 + A1^† A1 = I,
    construct a 2X x 2X unitary matrix U that extends Q = [A0 . ,A1 .].
    """
    X = A0.shape[0]
    
    # Construct Q
    Q = np.hstack([A0, A1])  # X x 2X matrix
    
    # Find an orthonormal complement to Q
    Q_full, _ = qr(Q.T, mode='full')  # QR decomposition of (2X x X) matrix Q^T
    B = Q_full[:, X:].T  # Extract the last X rows to form B (X x 2X)
    
    # Construct the unitary matrix U
    U = np.vstack([Q, B])  # 2X x 2X matrix

    return U



def make_unitary(A0, A1):
    # Check the unitarity condition A0†A0 + A1†A1 = I
    # if not np.allclose(A0 @ A0.T + A1 @ A1.T, np.eye(A0.shape[0])):
    #     raise ValueError("A0 and A1 do not satisfy the unitarity condition A0†A0 + A1†A1 = I")
    
    # Create the matrix Q with the form [A0, B0; A1, B1]
    # We need to find the correct B0 and B1 such that Q is unitary
    X = A0.shape[0]
    
    # Stack the matrix Q as [A0, B0; A1, B1]
    Q_top = np.hstack([A0, np.zeros((X, X), dtype=complex)])  # Start with A0 and zero for B0
    Q_bottom = np.hstack([A1, np.zeros((X, X), dtype=complex)])  # Start with A1 and zero for B1
    
    Q = np.vstack([Q_top, Q_bottom])  # Full matrix Q
    
    # Perform QR decomposition to ensure unitary matrix
    Q, R = np.linalg.qr(Q)  # Q will now be the unitary matrix
    
    # The B0 and B1 terms are extracted from the second half of Q
    B0 = Q[:, X:]  # The second half of Q is B0
    B1 = Q[X:, X:]  # The second half of Q for the bottom block is B1
    
    # Construct the final unitary matrix Q with the correct B0 and B1
    final_Q = np.hstack([np.hstack([A0, B0]), np.hstack([A1, B1])])
    
    return final_Q



def U_to_cir(A_0, A_1, n_ancilla):
    
    Q = make_unitary(A_0, A_1)

    # Create a circuit with 1 system and n_ancilla qubits
    U_ckt = QuantumCircuit(1 + n_ancilla)
    
    # Apply the required gate
    gate = UnitaryGate(Q, check_input=False)
    U_ckt.append(gate, range(1 + n_ancilla))

    return U_ckt


def mps_to_ckt(system_size, n_ancilla, A_list):

    # We assume we get the MPS as a list of matrices A_0^(1), A_1^(1), ..., A_0^(N), A_1^(N)
    big_ckt = QuantumCircuit(system_size + n_ancilla)

    for i in range(system_size):
        A_0 = A_list[2*i]
        A_1 = A_list[2*i + 1]

        U_ckt = U_to_cir(A_0, A_1, n_ancilla)

        big_ckt.append(U_ckt, [i] + list(range(system_size, system_size + n_ancilla)))

    return big_ckt



def create_ghz(n):  

    dim = 2**n  # Dimension of the Hilbert space
    data = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Coefficients of GHZ state
    indices = np.array([0, dim - 1])  # Basis state positions
    indptr = np.array([0, 2])  # Column pointer for CSC format
    
    return sp.csc_matrix((data, indices, indptr), shape=(dim, 1))


def pad_A(A_list, n_ancilla):
    # Function to make all MPS matrices the same size by padding with ???? TBD
    for i in range(len(A_list)):
        temp = np.eye(2**n_ancilla)
        temp[:A_list[i].shape[0], :A_list[i].shape[1]] = A_list[i]
        A_list[i] = temp
    return A_list


if __name__ == "__main__":

    system_size = 5

    state = create_ghz(system_size).toarray().reshape([2]*system_size)

    mps_site_list = emps.state2mps(state)

    flattened_mps = [mps_site[:, i, :] for mps_site in mps_site_list for i in range(2)]
    
    bond_dim = np.max([np.array(mps_site.shape) for mps_site in mps_site_list])

    n_ancilla = int(np.ceil(np.log2(bond_dim)).item())

    A_list = pad_A(flattened_mps, n_ancilla)

    big_ckt = mps_to_ckt(system_size, n_ancilla, A_list)

    print(big_ckt)