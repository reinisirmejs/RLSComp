import numpy as np
from scipy.linalg import null_space

"""
Here we store the padding functions neccessary for each state of the algorithm
Generic functions we use for padding
"""

def nearest_upper_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()

def reverse_ordering_of_As(A_list):
    A_list_new=[]
    N = int(len(A_list)/2)
    for i in range(N):
        A_list_new.append(A_list[2*(N-1-i)])
        A_list_new.append(A_list[2*(N-1-i)+1])
    return A_list_new

"""
Padding functions for Option 1
"""

def pre_padding_As_to_powers_of_2(A0,A1):
    rows = A0.shape[0]
    cols = A0.shape[1]
    desired_rows = nearest_upper_power_of_2(rows)
    desired_cols = nearest_upper_power_of_2(cols)
    A0_new = np.zeros((desired_rows, cols))
    A1_new = np.zeros((desired_rows, cols))
    A0_new[:A0.shape[0], :A0.shape[1]] = A0
    A1_new[:A1.shape[0], :A1.shape[1]] = A1
    our_indep_vectors = np.vstack([A0_new, A1_new])
    Anull = null_space(our_indep_vectors.T)
    if desired_cols>cols:
        A_stacked = np.vstack([our_indep_vectors.T, Anull[:, :desired_cols - cols].T]).T
    else:
        A_stacked = our_indep_vectors
    A0_new = A_stacked[:desired_rows, :desired_cols]
    A1_new = A_stacked[desired_rows:, :desired_cols]
    return A0_new, A1_new

def padding_As_no_ancilla_qiskit_ordering(A0,A1,rectangular = False):
    A0, A1 = pre_padding_As_to_powers_of_2(A0,A1) #first pad the isometries to power of 2 dimensions
    #stack them according to the qiskit ordering
    A = np.empty((A0.shape[0] + A1.shape[0], A0.shape[1]), dtype=A0.dtype)
    A[0::2] = A0; A[1::2] = A1
    #find the orthogonal vectors
    Anull = null_space(A.T)
    dim = A.shape[0]
    avail_columns = A.shape[1]
    missing_column = dim - 2*avail_columns
    if A.shape[1]==A.shape[0]:
        return A
    #add the missing columns to make sure we have a square unitary matrix
    missing_column = dim - avail_columns
    if not rectangular:
        if missing_column>0:
            A = np.hstack([A, Anull])
    return A

def padding_isometries_list_no_ancilla(isometries,rectangular = False):
    unitary_op_list = []
    n = len(isometries) // 2
    #invert the ordering for qiskit
    isometries = reverse_ordering_of_As(isometries)
    for i in range(n):
        A0 = isometries[2*i]
        A1 = isometries[2*i+1]
        Unitary = padding_As_no_ancilla_qiskit_ordering(A0, A1, rectangular = rectangular)
        if not rectangular:
            assert np.allclose(Unitary.T @ Unitary, np.eye(Unitary.shape[0])), f'Unitary is not unitary, {Unitary}'
        unitary_op_list.append(Unitary)
    return unitary_op_list

"""
Option 2: Specific padding functions to the tree method
"""

# Maria added this function
def padding_isometries_list_to_powers_of_2(isometries):
    new_isometries = []
    for i in range(len(isometries)//2):
        A0 = isometries[2*i]
        A1 = isometries[2*i+1]
        A0_new, A1_new = pre_padding_As_to_powers_of_2(A0, A1)
        new_isometries.append(A0_new)
        new_isometries.append(A1_new)
    return new_isometries

