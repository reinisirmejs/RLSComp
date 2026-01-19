import circuit_utils
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit import QuantumCircuit

def fancy_isometry_padding(isometries, ancilla_dim):
    """
    this function pads isometries in the staircase way, e.g. 2, 4, 8, ..., ancilla_dim, ..., ancilla_dim,..,
      8, 4,2"""
    new_isometries = []
    n = len(isometries)//2
    for i in range(n):
        A0 = isometries[2*i]
        A1 = isometries[2*i+1]
        desired_dim = min(2**(i+1),2**(n-i),ancilla_dim)
        assert max(A0.shape)<=desired_dim, "The specified ancilla_dim is too small for the isometries"
        A0_new, A1_new = circuit_utils.reinis_padding(A0, A1, desired_dim)
        new_isometries.append(A0_new)
        new_isometries.append(A1_new)
    return new_isometries



def fancy_get_isometries_from_MPS(MPS_LIST,ancilla_dim):

    MPS_LIST = circuit_utils.normalize(MPS_LIST)
    left_mps_list, right = circuit_utils.left_canonical_form(MPS_LIST)

    isometries = circuit_utils.left_isometries(left_mps_list)
    print(isometries)

    #Here we do the padding to ensure that the isometries are of the correct size
    isometries = fancy_isometry_padding(isometries, ancilla_dim)
    print(isometries)
    ancilla_state = np.zeros(ancilla_dim)
    ancilla_state[0] = 1
    return isometries, ancilla_state


def fancy_mps_to_ckt(system_size, n_ancilla, A_list):

    # We assume we get the MPS as a list of matrices A_0^(1), A_1^(1), ..., A_0^(N), A_1^(N)
    big_ckt = QuantumCircuit(system_size + n_ancilla)
    #Here we can potentially add the initial state preparation
    if True:
        A_list = circuit_utils.reverse_ordering_of_As(A_list)
    for i in range(system_size):
        A_0 = A_list[2*i]
        A_1 = A_list[2*i + 1]

        #Here make sure the ancilla size is correct here
        n_i_ancilla = min(2**(i+1), 2**(system_size-i), 2**n_ancilla)
        U_ckt = circuit_utils.U_to_cir(A_0, A_1, n_i_ancilla)

        #make sure we apply the reduced number of ancilla qubits to the correcr registers
        big_ckt.append(U_ckt, list(range(n_i_ancilla))+[i+n_ancilla])

    return big_ckt


def fancy_MPS_to_circuit(A_list):
    system_size = len(A_list)//2
    dimA = np.shape(A_list[0])[0]
    n_ancilla = int(np.ceil(np.log2(dimA)))
    return fancy_mps_to_ckt(system_size, n_ancilla, A_list)
