import numpy as np
import itertools
from math import comb
def prod_cs_red(n,k,i,m,r,c):
    res = 0
    if c==(r+m):
        #print(n-i,k-r-m,n+1-i, k-r, m)
        if (k-r)<=(n-i+1):
            res = np.sqrt(1-m+(-1)**m * (r-k)/(n-i+1))
        else:
            #res = np.sqrt(comb(n-i,k-r-m)*comb(1,m)/comb(n+1-i,k-r))
            res = 0
    if m==0:
        if i>=n-k+2:
            if r==c:
                if r<=(k+i-n-2):
                    res+=1
    return res


def build_A_dicke(n,k,i,m):
    #n is the system size
    #k is the number of 1s in the dicke state
    #i is the index of the system register (runs from 1 to n)
    #m is the qubit register in {0,1}
    n_anc = k+1
    Aim = np.zeros((n_anc, n_anc))
    for r in range(n_anc):
        for c in range(n_anc):
            Aim[r,c] = prod_cs_red(n,k,i,m,r,c)
    return Aim


def build_A_list_dicke(n,k):
    A_list = []
    for i in range(1,n+1):
        for m in [0,1]:
            A_list.append(build_A_dicke(n,k,i,m))
    return A_list


def verify_dicke_isos(n,k,A_list):

    bit_keys = generate_n_bit_strings(n)
    #r_state = np.array([0,1])
    l_state = np.array([-1,0])
    res_dict = {}
    for bit_key in bit_keys:
        r_state = np.array([1,0])
        for i in range(n):
            #print(A_list[2*i+int(bit_key[i])],r_state)
            #r_state = np.dot(A_list[2*i+int(bit_key[i])],r_state)
            r_state = A_list[2*i+int(bit_key[i])] @ r_state
        overlap = np.dot(l_state,r_state)
        res_dict[bit_key] = overlap

    return res_dict


def build_W_vector(n):
    weight = 1/np.sqrt(n)
    st0 = np.array([1,0])
    st1 = np.array([0,1])
    full_state = np.zeros(2**n)
    for i in range(n):
        state = 1
        for j in range(n):
            if j==i:
                state = np.kron(state,st1)
            else:
                state = np.kron(state,st0)
        full_state+=weight*state
    return full_state


def generate_n_bit_arrays(n):
    # Generate all n-bit strings (arrays of 0s and 1s)
    return [list(bits) for bits in itertools.product([0, 1], repeat=n)]

def generate_n_bit_strings(n):
    # Generate all n-bit strings by iterating through all combinations of 0s and 1s
    return [''.join(bits) for bits in itertools.product('01', repeat=n)]
