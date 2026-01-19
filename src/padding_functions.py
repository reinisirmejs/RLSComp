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

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def reverse_ordering_of_As(A_list):
    A_list_new=[]
    N = int(len(A_list)/2)
    for i in range(N):
        A_list_new.append(A_list[2*(N-1-i)])
        A_list_new.append(A_list[2*(N-1-i)+1])
    return A_list_new

"""
Generic padding to square matrices of the desired size
"""

def padding_As(A0,A1,desired_dim):
    A0_new = np.zeros((desired_dim, A0.shape[1]))
    A1_new = np.zeros((desired_dim, A1.shape[1]))
    #First lets pad the relevant columns with 0s to ensure the correct size
    A0_new[:A0.shape[0], :A0.shape[1]] = A0
    A1_new[:A1.shape[0], :A1.shape[1]] = A1

    #We create a column matrix of the relevant vectors that are still isometries
    #However, the column dimension is wrong so needs to be padded with (desired_dim - A0.shape[1])
    #extra vectors that are linearly independent to the A0.shape[0] vectors we already have
    our_indep_vectors = np.vstack([A0_new, A1_new])
    #print(our_indep_vectors)

    new_indep_vectors = null_space(our_indep_vectors.T)
    #print(new_indep_vectors)
    #this will be too big though
    new_isos_combined = np.hstack([our_indep_vectors, new_indep_vectors[:, :desired_dim - A0.shape[1]]])
    A0_new = new_isos_combined[:desired_dim, :desired_dim]
    A1_new = new_isos_combined[desired_dim:, :desired_dim]

    return A0_new,A1_new

"""Harcoded version of madding where the extra vectors are just the identity matrix, 
instead of taken from the null space. No noticeable improvement in performance"""

def padding_As_hardcoded(A0,A1,desired_dim):
    A0_new = np.zeros((desired_dim, desired_dim))
    A1_new = np.zeros((desired_dim, desired_dim))
    #First lets pad the relevant columns with 0s to ensure the correct size
    A0_new[:A0.shape[0], :A0.shape[1]] = A0.copy()
    A1_new[:A1.shape[0], :A1.shape[1]] = A1.copy()
    #We create a column matrix of the relevant vectors that are still isometries
    #However, the column dimension is wrong so needs to be padded with (desired_dim - A0.shape[1])
    #extra vectors that are linearly independent to the A0.shape[0] vectors we already have
    new_isos_combined = np.vstack([A0_new, A1_new])
    #print(our_indep_vectors)
    dominant_dim = A0.shape[1]

    new_isos_combined[dominant_dim:desired_dim, dominant_dim:desired_dim] = np.eye(desired_dim - dominant_dim)
    A0_new = new_isos_combined[:desired_dim, :desired_dim]
    A1_new = new_isos_combined[desired_dim:, :desired_dim]
    assert np.allclose(A0_new.T @ A0_new + A1_new.T @ A1_new, np.eye(desired_dim)), f'As are not not isometries, {A0}, {A1}, {A0_new}, {A1_new}'

    return A0_new,A1_new

"""
Specific padding function for option 1
"""

def padding_isometries_list(isometries, desired_dim):
    new_isometries = []
    for i in range(len(isometries)//2):
        A0 = isometries[2*i]
        A1 = isometries[2*i+1]
        assert max(A0.shape)<=desired_dim, "The specified ancilla_dim is too small for the isometries"
        A0_new, A1_new = padding_As_hardcoded(A0, A1, desired_dim)
        new_isometries.append(A0_new)
        new_isometries.append(A1_new)
    return new_isometries

"""
Specific padding function for option 2"""

def padding_isometries_list_ancilla_efficient(isometries, max_ancilla_dim):
    new_isometries = []
    n = len(isometries) // 2
    for i in range(n):
        desired_dim = min(2**(i+1), 2**(n-i),max_ancilla_dim) # The maximum bond dimension dim
        A0 = isometries[2*i]
        A1 = isometries[2*i+1]
        #print(i, A0.shape, desired_dim)
        assert max(A0.shape)<=desired_dim, "The specified ancilla_dim is too small for the isometries"
        A0_new, A1_new = padding_As(A0, A1, desired_dim)
        new_isometries.append(A0_new)
        new_isometries.append(A1_new)
    return new_isometries

"""
Rectangular padding for option 3
"""

def padding_As_rectangular(A0,A1,desired_dim):
    A0_new = np.zeros((desired_dim, A0.shape[1]))
    A1_new = np.zeros((desired_dim, A1.shape[1]))
    #First lets pad the relevant columns with 0s to ensure the correct size
    A0_new[:A0.shape[0], :A0.shape[1]] = A0
    A1_new[:A1.shape[0], :A1.shape[1]] = A1
    m_low = A0.shape[1]
    m = nearest_upper_power_of_2(m_low)
    #We create a column matrix of the relevant vectors that are still isometries
    #However, the column dimension is wrong so needs to be padded with (desired_dim - A0.shape[1])
    #extra vectors that are linearly independent to the A0.shape[0] vectors we already have
    our_indep_vectors = np.vstack([A0_new, A1_new])
    new_indep_vectors = null_space(our_indep_vectors.T)
    new_isos_combined = np.hstack([our_indep_vectors, new_indep_vectors[:, :m- A0.shape[1]]])
    return new_isos_combined


def padding_isometries_list_ancilla_rectangular(isometries, max_ancilla_dim):
    new_isometries = []
    n = len(isometries) // 2
    for i in range(n):
        desired_dim = min(2**(i+1), 2**(n-i),max_ancilla_dim) # The maximum bond dimension dim
        A0 = isometries[2*i]
        A1 = isometries[2*i+1]
        #print(i, A0.shape, desired_dim)
        assert max(A0.shape)<=desired_dim, "The specified ancilla_dim is too small for the isometries"
        iso_rect = padding_As_rectangular(A0, A1, desired_dim)
        new_isometries.append(iso_rect)
    return new_isometries

"""
Padding functions for Option 4
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
Option 5: Specific padding functions to the tree method
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

#This function takes a matrix that has some orthonormal rows and some rows full of 0s
#It collects the rows that are non-zero and creates an orthonormal space, to make it into a unitary?
#Test that this works

def padding_add_nonzero_rows(initial_mat):
    non_zero_row_vectors = []
    tol = 1e-12
    zero_row_indx = []
    for (idx_row,row) in enumerate(initial_mat):
        if sum(abs(row))>tol:
            non_zero_row_vectors.append(row)
        else:
            zero_row_indx.append(idx_row)
    #print(non_zero_row_vectors)
    add_vecs = null_space(non_zero_row_vectors).T
    #print(add_vecs)
    for (add_vec_idx,row_idx) in enumerate(zero_row_indx):
        initial_mat[row_idx, :] = add_vecs[add_vec_idx]
    return initial_mat


def padding_remove_zero_rows(initial_mat):
    non_zero_row_vectors = []
    tol = 1e-12
    for (idx_row,row) in enumerate(initial_mat):
        if sum(abs(row))>tol:
            non_zero_row_vectors.append(row)
    assert is_power_of_two(len(non_zero_row_vectors))
    return non_zero_row_vectors
