import numpy as np
import matplotlib.pyplot as plt

from functools import reduce
from operator import mul

import scipy

import gc
gc.enable()

def state2mps(state, max_bond_dim=None):

    site_tensor_list = []

    physical_dims = state.shape

    num_sites = len(physical_dims)

    # Step 1
    state1 = np.reshape(state, (physical_dims[0], reduce(mul, physical_dims[1:])))

    # Step 2
    U, S, V_dag = scipy.linalg.svd(state1, full_matrices=False, lapack_driver='gesvd')

    req_norm = np.sqrt(np.power(S, 2).sum())
    assert np.isclose(req_norm, np.linalg.norm(state))
    
    r_prev = U.shape[-1] # Extracting bond dimension

    if max_bond_dim is not None:
        if r_prev > max_bond_dim:
            U = U[:, :max_bond_dim]
            S = S[:max_bond_dim]
            V_dag = V_dag[:max_bond_dim, :]
            r_prev = max_bond_dim

    site_tensor_list.append(U)     # First site tensor d1 x r1


    for i in range(1, num_sites-1):

        Q = np.diag(S) @ V_dag

        U, S, V_dag = scipy.linalg.svd(Q.reshape(r_prev * physical_dims[i], reduce(mul, physical_dims[i+1:])), full_matrices=False, lapack_driver='gesvd')

        r_next = U.shape[-1]

        if max_bond_dim is not None:
            if r_next > max_bond_dim:
                U = U[:, :max_bond_dim]
                S = S[:max_bond_dim]
                V_dag = V_dag[:max_bond_dim, :]
                r_next = max_bond_dim
        
        A = U.reshape(r_prev, physical_dims[1], r_next) # Second site tensor r1 x d2 x r2

        site_tensor_list.append(A)  

        r_prev = r_next

    last_site = np.diag(S) @ V_dag


    if max_bond_dim is not None:
        if r_next > max_bond_dim:
            last_site = last_site[:max_bond_dim, :]
            r_next = max_bond_dim

    actual_norm = np.linalg.norm(last_site)
    assert np.isclose(np.sqrt(np.power(S, 2).sum()), actual_norm)

    site_tensor_list.append(last_site * req_norm / actual_norm)

    return [np.expand_dims(site_tensor_list[0], axis=0)] + site_tensor_list[1:-1] + [np.expand_dims(site_tensor_list[-1], axis=-1)]


def mps2state(site_tensor_list):

    num_sites = len(site_tensor_list)

    # Regularizing the shapes (adding dummy indices on left and right if needed)
    if len(site_tensor_list[-1].shape) == 2:
        site_tensor_list[-1] = np.expand_dims(site_tensor_list[-1], axis=-1)

    if len(site_tensor_list[0].shape) == 2:
        site_tensor_list[0] = np.expand_dims(site_tensor_list[0], axis=0)

    out = np.einsum('ijk, klm ->  ijlm', site_tensor_list[0], site_tensor_list[1], optimize=True)

    for i in range(2, num_sites):
        out = np.einsum('...j, jkl -> ...kl', out, site_tensor_list[i], optimize=True)

    return np.einsum('i...i -> ...', out, optimize=True)


def truncate(site_tensor_list, new_dim, canonize_only=False):


    num_sites = len(site_tensor_list)

    denominator = 1

    # Canonization from left to right
    for i in range(num_sites-1):
        site_tensor = site_tensor_list[i]

        # Regularizing the shapes (adding dummy indices on left and right if needed)
        if len(site_tensor.shape) == 2:
            site_tensor = np.expand_dims(site_tensor, axis=-1)
        
        flag = False
        summer = 0
        for j in range(site_tensor.shape[1]):
            summer += site_tensor[:, j, :].conj().T @ site_tensor[:, j, :]
            if not np.isclose(summer, np.eye(site_tensor.shape[-1])).all():
                flag = True
        
        if flag:
            
            old_shape = site_tensor.shape
            site_tensor = site_tensor.reshape(-1, site_tensor.shape[-1])
            U, S, V_dag = scipy.linalg.svd(site_tensor, full_matrices=False, lapack_driver='gesvd')
            
            site_tensor_list[i] = U.reshape(old_shape[0], old_shape[1], -1)

            site_tensor_list[i+1] = np.einsum('ij, jkl -> ikl', np.diag(S) @ V_dag, site_tensor_list[i+1], optimize=True)

            del site_tensor 
            del U, S, V_dag               


    if canonize_only:
        return site_tensor_list


    for i in range(num_sites-1, 0, -1):
        site_tensor = site_tensor_list[i]

        # Regularizing the shapes (adding dummy indices on left and right if needed)
        if len(site_tensor.shape) == 2 and i == num_sites-1:
            site_tensor = np.expand_dims(site_tensor, axis=-1)

        if i < num_sites-1:
            X = U_tilde @ np.diag(S_tilde)
            site_tensor = np.einsum('ijk, kl -> ijl', site_tensor, X, optimize=True)

        orig_shape = site_tensor.shape

        site_tensor = site_tensor.reshape(orig_shape[0], -1)
        U, S, V_dag = scipy.linalg.svd(site_tensor, full_matrices=False, lapack_driver='gesvd')

        del site_tensor

        if i == num_sites-1:
            numerator = np.sqrt(np.power(S, 2).sum())
        
        rank = len(S)
        
        if rank <= new_dim:
            U_tilde = U
            S_tilde = S
            V_dag_tilde = V_dag
            left_dim = rank

        else:
            U_tilde = U[:, :new_dim]
            S_tilde = S[:new_dim]
            V_dag_tilde = V_dag[:new_dim, :]
            left_dim = new_dim


        new_tensor = V_dag_tilde.reshape(left_dim, orig_shape[1], orig_shape[2])

        site_tensor_list[i] = new_tensor


    # Left most site
    site_tensor = site_tensor_list[0]

    if len(site_tensor.shape) == 2:
        site_tensor = np.expand_dims(site_tensor, axis=0)

    orig_shape = site_tensor.shape

    X = U_tilde @ np.diag(S_tilde)

    new_tensor = np.einsum('ijk, kl -> ijl', site_tensor, X, optimize=True)

    denominator = np.sqrt(np.power(S_tilde, 2).sum())

    site_tensor_list[0] = new_tensor * numerator / denominator
    
    return site_tensor_list


def contract(mps_1, mps_2):
    assert len(mps_1) == len(mps_2)

    temp = np.einsum('ljm, ijk -> ilmk', mps_1[0].conj(), mps_2[0], optimize=True)
    for i in range(1, len(mps_1)):
        temp = np.einsum('ilmk, mbn, kbp -> ilnp', temp, mps_1[i].conj(), mps_2[i], optimize=True)

    return np.einsum('illi -> ', temp, optimize=True).item()


def apply(mpo, mps):

    assert len(mpo) == len(mps)

    for i in range(len(mpo)):
        temp = np.einsum('ijk, lmnj -> ilmnk', mps[i], mpo[i], optimize=True)
        temp = temp.reshape(temp.shape[0] * temp.shape[1], temp.shape[2], temp.shape[3] * temp.shape[4])
        mps[i] = temp
    
    return mps








if __name__ == '__main__':
    state = np.random.randn(5, 5, 5, 5, 5)
    state2 = np.random.randn(5, 5, 5, 5, 5)
    # state = state/np.linalg.norm(state)

    error_list = []
    error_list_2 = []
    error_list_3 = []

    max_bond_dim = 25


    # print("State1: ", np.linalg.norm(state)**2, contract(mps_full, mps_full))
    # mps_full2 = state2mps(state2, bond_dim=max_bond_dim)
    # print("State2: ", np.linalg.norm(state2)**2, contract(mps_full2, mps_full2))

    # print("Overlap: ", state.reshape(1, -1).conj() @ state2.reshape(-1, 1), contract(mps_full, mps_full2))

    for bond_dim in range(1,max_bond_dim+6):
        mps_full = state2mps(state, max_bond_dim=max_bond_dim)
        
        print(f'Bond Dimension: {bond_dim}')

        mps = state2mps(state, max_bond_dim=bond_dim)
        state_reconstructed = mps2state(mps)
        assert np.isclose(np.linalg.norm(state), np.linalg.norm(state_reconstructed))
        error_list.append(np.linalg.norm(state - state_reconstructed))
        # error_list.append((state.reshape(1, -1).conj() @ state_reconstructed.reshape(-1, 1)).item())


        assert np.isclose(state, mps2state(mps_full)).all()

        mps_truncated = truncate(mps_full, new_dim=bond_dim)
        state_truncated = mps2state(mps_truncated)
        assert np.isclose(np.linalg.norm(state), np.linalg.norm(state_truncated))
        error_list_2.append(np.linalg.norm(state - state_truncated))
        # error_list_2.append((state.reshape(1, -1).conj() @ state_truncated.reshape(-1, 1)).item())


        # Difference between the two methods
        assert(np.isclose(np.linalg.norm(state_reconstructed), np.linalg.norm(state_truncated)))
        error_list_3.append(np.linalg.norm(state_truncated - state_reconstructed))
        # error_list_3.append((state_reconstructed.reshape(1, -1).conj() @ state_truncated.reshape(-1, 1)).item())


    plt.plot(error_list, label='Reconstructed', marker='D')
    plt.plot(error_list_2, label='Truncated', marker='o')
    plt.plot(error_list_3, label='Between Methods', marker='x')
    plt.xlabel('Bond Dimension')
    plt.ylabel("Error")
    # plt.ylabel('Overlap')
    plt.legend()
    plt.savefig('Plots/fig.png')