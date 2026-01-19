import numpy as np
import itertools
from tensornetwork import ncon
from numpy import linalg as LA
from src.padding_functions import padding_isometries_list_no_ancilla, padding_isometries_list_to_powers_of_2
from src.regex_utils import regex_to_dfa, list_to_acdfa_direct

""""
Basic MPS functions
"""
def transpose_mps(mps):
    return [A.transpose(1, 0, 2) for A in mps]

def compute_norm(mps):
    mps2 = transpose_mps(mps)
    tensor = mps2[0]
    norm_squared = ncon([tensor, np.conj(tensor)], [[-1, 1, -2], [-3, 1, -4]])
    for i in range(1, len(mps2)):
        tensor = mps2[i]
        norm_squared = ncon([norm_squared, tensor, np.conj(tensor)], [[-1, 1, -3, 3], [1, 2, -2], [3, 2, -4]])     
    return np.sqrt(norm_squared.item())

def normalize(mps):
    norm = compute_norm(mps)
    if np.isclose(norm, 0):
        raise ValueError("Cannot normalize MPS: computed norm is zero.")
    normalized_mps = [tensor / norm**(1/len(mps)) for tensor in mps]
    return normalized_mps

def get_bond_dimension(begin_state, final_state, transitions):
    layer = [begin_state]
    max_size = len(layer)
    while layer != [final_state]:
        # Build the next layer    
        next_layer = []
        for node in layer:
            if node != final_state:
                next_layer += list(transitions[node].values())
        next_layer = list(set(next_layer))
        layer = next_layer
        if len(next_layer) > max_size:
            max_size = len(next_layer)
    return max_size

"""
MPS functions to get the canonical form of the MPS and extract the isometries
"""

def build_mps_from_regex(regex, n, complement=False):
    dfa = regex_to_dfa(regex, complement=complement)
    A, v_l, v_r = DFA_to_MPS(dfa)
    MPS_LIST = MPS_to_list(A, v_l, v_r, n)
    return MPS_LIST, dfa

def build_dicke_mps_from_bitstrings(n, k):
    def bitstrings_with_k_ones(n, k):
        positions = itertools.combinations(range(n), k)
        result = []
        for pos in positions:
            bits = ['0'] * n
            for p in pos:
                bits[p] = '1'
            result.append(''.join(bits))
        return result
    
    input_list = bitstrings_with_k_ones(n, k)
    acdfa = list_to_acdfa_direct(input_list)
    MPS_LIST = ACDFA_to_MPS(acdfa)
    return MPS_LIST, None

def MPS_to_list(A, v_l, v_r, n):
    left = np.zeros((1,v_l.shape[0]))
    left[0,:] = v_l.reshape(v_l.shape[0])
    right = np.zeros((v_r.shape[0],1))
    right[:,0] = v_r.reshape(v_r.shape[0])
    MPS_list = [ncon([left, A],[[-2,1],[-1,1,-3]]).reshape(A.shape[0],1,A.shape[2])]
    for _ in range(n - 2):
        MPS_list.append(A)
    MPS_list.append(ncon([A, right],[[-1,-2,1],[1,-3]]).reshape(A.shape[0],A.shape[1],1))
    return MPS_list

def MPS_to_state(mps):
    mps = normalize(mps)
    mps2 = transpose_mps(mps)
    tensor = mps2[0]
    state = tensor
    for i in range(1, len(mps2)):
        tensor = mps2[i]
        state = ncon([state, tensor],[[-1,-2,1],[1,-3,-4]]).reshape(state.shape[0],state.shape[1]*tensor.shape[1],tensor.shape[2])
    return np.squeeze(state)

def left_canonical_form_staircase(mps):
    mps2 = transpose_mps(mps)
    L = len(mps2)
    bond_dims = []
    # Right-to-left SVD sweep
    for n in range(L - 1, 0, -1):
        tensor = mps2[n]
        chi_l, d, chi_r = tensor.shape
        U, S, V = LA.svd(tensor.reshape(chi_l, d * chi_r), full_matrices=False)
        mps2[n] = V.reshape(V.shape[0], d, chi_r)
        mps2[n - 1] = ncon([mps2[n - 1], U @ np.diag(S)], [[-1, -2, 1], [1, -3]])
    # Left-to-right SVD sweep
    for n in range(L):
        tensor = mps2[n]
        chi_l, d, chi_r = tensor.shape
        U, S, V = LA.svd(tensor.reshape(chi_l * d ,chi_r), full_matrices=False)
        mps2[n] = U.reshape(chi_l, d, U.shape[1])
        if n == L - 1:
            right = (np.diag(S) @ V @ np.array([1.0] + [0.0 for _ in range(chi_r-1)]))
        else:    
            mps2[n + 1] = ncon([np.diag(S) @ V, mps2[n + 1]], [[-1, 1], [1, -2, -3]]) 
        #mps2[n + 1] = ncon([np.diag(S) @ V, mps2[n + 1]], [[-1, 1], [1, -2, -3]])
        bond_dims.append(mps2[n].shape[2])
    #print("Bond dimensions:", bond_dims)
    mps = transpose_mps(mps2)
    return mps, right

def left_isometries(left_canonical_mps):
    isometries = [left_canonical_mps[0][0,:,:], left_canonical_mps[0][1,:,:]]
    for i in range(1, len(left_canonical_mps)):
        isometries.append(left_canonical_mps[i][0,:,:])
        isometries.append(left_canonical_mps[i][1,:,:])
    return isometries

def get_padded_mps(MPS_LIST):
    """Return an MPS with bond dimensions padded to powers of 2."""
    MPS_LIST = normalize(MPS_LIST)
    left_mps_list, _ = left_canonical_form_staircase(MPS_LIST)
    isometries = left_isometries(left_mps_list)
    isometries = padding_isometries_list_to_powers_of_2(isometries)
    mps = [np.stack([isometries[2*i], isometries[2*i+1]], axis=0)for i in range(len(isometries)//2)]
    return mps

def get_polar_decomposition(tensor1, tensor2):
    """Compute the polar decomposition of the matrix obtained by contracting two neighboring
    MPS tensors."""
    matrix = ncon([tensor1, tensor2], [[-3, -1, 1], [1, -2, -4]]).reshape(tensor1.shape[1] * tensor2.shape[1], tensor1.shape[0] * tensor2.shape[2])
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    W = U
    P = np.diag(S) @ Vt
    return W.reshape(tensor1.shape[1], tensor2.shape[1], U.shape[1]), P.reshape(P.shape[0], tensor1.shape[0], tensor2.shape[2])

def get_tree_decomposition(mps):
    """Construct a hierarchical (tree) decomposition of a given MPS by iteratively
    combining neighboring tensors through polar decompositions."""
    mps2 = transpose_mps(mps)
    L = len(mps2)
    W_tree = []
    for i in range(int(np.ceil(np.log2(L)))):
        if i == 0:
            tensors = mps2
        length = len(tensors) // 2
        rest = len(tensors) % 2
        W_layer = []
        for j in range(length):
            W, P = get_polar_decomposition(tensors[2 * j], tensors[2 * j + 1])
            W_layer.append(W)
            tensors[j] = ncon([P], [[-2, -1, -3]])
        tensors = tensors[:j + 1] + [tensors[-1]]
        if rest == 0:
            tensors = tensors[:-1]
        W_tree.append(W_layer)
    return W_tree

def get_state_from_tree(tree):
    """Reconstruct the full many-body state from its tree decomposition."""
    state = []
    if len(tree) == 1:
        state.append([tree[0][0]])
    for l in range(len(tree) - 1):
        state_layer = []
        for j in range(len(tree[l + 1])):
            if l == 0:
                layer = tree[0]
            else:
                layer = state[-1]
            if len(layer[2 * j: 2 * j + 2]) == 1:
                tensor = ncon([tree[l + 1][j], layer[2 * j]], [[1, -3, -4], [-1, -2, 1]])
                tensor = tensor.reshape(np.prod(tensor.shape[:2]), *tensor.shape[2:])
            elif len(layer[2 * j: 2 * j + 2]) == 2:
                tensor = ncon([tree[l + 1][j], layer[2 * j], layer[2 * j + 1]], [[1, 2, -5], [-1, -2, 1], [-3, -4, 2]])
                tensor = tensor.reshape(np.prod(tensor.shape[:2]), np.prod(tensor.shape[2:4]), tensor.shape[-1])
            state_layer.append(tensor)
        if len(layer[2 * j:]) > 2:
            state_layer.append(layer[-1])
        state.append(state_layer)
    state_vector = state[-1][0]
    return state_vector.reshape(np.prod(state_vector.shape))

"""
DFA related functions for MPS conversion
"""

def get_dfa_info(dfa):
    states_dictio = {state: i for i, state in enumerate(dfa.states)}
    states = [i for i in range(len(dfa.states))]
    inputs = [0, 1]
    transitions = {}
    dfa_dict = dfa.to_dict()
    for state in dfa.states:
        if state in dfa_dict:
            transitions[states_dictio[state]] = {}
            if '0' in dfa_dict[state]:
                transitions[states_dictio[state]][0] = states_dictio[dfa_dict[state]['0']]
            if '1' in dfa_dict[state]:
                transitions[states_dictio[state]][1] = states_dictio[dfa_dict[state]['1']]
    begin_state = states_dictio[dfa.start_state]
    final_states = [states_dictio[state] for state in dfa.final_states]
    return states, inputs, transitions, begin_state, final_states

def DFA_to_MPS(dfa):
    """
    Converts a finite state machine to Matrix Product State (MPS) representation.
    
    Parameters:
    -----------
    states : list or array-like
        List of all states in the finite state machine.
    inputs : list or array-like
        List of possible input values (typically [0, 1] for binary).
    transitions : dict
        Dictionary mapping states to their transitions, where each transition is a dict
        mapping input values to destination states.
    begin_state : int
        The index of the initial state.
    final_state : int
        The index of the accepting/final state.
    
    Returns:
    --------
    A : numpy.ndarray
        3D tensor of shape (len(inputs), len(states), len(states)) representing the transition matrices.
        A[i,j,k] = 1 if there's a transition from state j to state k with input i, otherwise 0.
    v_l : numpy.ndarray
        Vector of length len(states) representing the initial state distribution.
        Contains 1 at the position of begin_state and 0 elsewhere.
    v_r : numpy.ndarray
        Column vector of length len(states) representing the final state distribution.
        Contains 1 at the position of final_states and 0 elsewhere.
    """
    states, inputs, transitions, begin_state, final_states = get_dfa_info(dfa)

    # Initialize tensor A with zeros - dimensions: [input_values × states × states]
    A = np.zeros((len(inputs), len(states), len(states)), dtype=int)
    # Fill the tensor A based on the transition function
    for start_state in states:
        if start_state in transitions:
            for input_value in list(transitions[start_state].keys()):
                # Verify that input values are binary (0 or 1)
                assert input_value == 0 or input_value == 1
                # Set the corresponding tensor element to 1
                # A[input, current_tate, next_state] = 1 indicates a valid transition
                A[input_value, start_state, transitions[start_state][input_value]] = 1
    # Create the left boundary vector (initial state)
    v_l = np.zeros(len(states), dtype=int)
    v_l[begin_state] = 1  # Set 1 at the position of the initial state
    # Create the right boundary vector (final state)
    v_r = np.zeros(len(states), dtype=int)
    for state in final_states:
        v_r[state] = 1  # Set 1 at the position of the final states
    # Return the MPS representation (A, v_l, v_r)
    # Note: v_r is reshaped to a column vector using np.newaxis
    return A, v_l, v_r[:,np.newaxis]

def ACDFA_to_MPS(acdfa):
    assert(acdfa.is_deterministic())
    assert(acdfa.is_acyclic())
    states, inputs, transitions, begin_state, final_states = get_dfa_info(acdfa)
    assert(len(final_states) == 1)
    final_state = final_states[0]
    d = get_bond_dimension(begin_state, final_state, transitions)
    MPS_list = []
    layer = [begin_state]
    while layer != [final_state]:
        # Build the next layer    
        next_layer = []
        for node in layer:
            if node != final_state:
                next_layer += list(transitions[node].values())
        next_layer = list(set(next_layer))
        next_layer.sort()
        # Build the matrix for the current site
        A = np.zeros((len(inputs), d, d), dtype=int)
        for i, node in enumerate(layer):
            for input_ in transitions[node]:
                A[input_, i, next_layer.index(transitions[node][input_])] = 1
        layer = next_layer
        MPS_list.append(A)
    MPS_list[0] = MPS_list[0][:, :1]
    MPS_list[-1] = MPS_list[-1][...,:1]
    return MPS_list


"""
1) This is the most efficient method, where we do not need to use any ancilla qubits at all. Here
we directly exploit the MPS structure and carefully create unitary operators directly from the isometries"""

def get_no_ancilla_unitaries_from_MPS(MPS_LIST,rectangular = False):
    left_mps_list, _ = left_canonical_form_staircase(MPS_LIST)
    isometries = left_isometries(left_mps_list)
    # Here we do the padding to ensure that the isometries are of the correct size
    unitaries = padding_isometries_list_no_ancilla(isometries,rectangular=rectangular)
    ancilla_state = np.zeros(2)
    ancilla_state[0] = 1
    return unitaries, ancilla_state

# Maria added these functions
"""
2) Tree Decomposition of MPS via Polar Decomposition
"""

def get_circuit_info_from_tree_rectangular(system_size,tree):
    """Construct the circuit information from a tree decomposition."""
    circuit_layers = []
    for layer in tree:
        circuit_layer = []
        for tensor in layer:
            #Tensor shape is left up, right up, down
            output_dim = np.prod(tensor.shape[:2]); input_dim = tensor.shape[2]
            completed_tensor = np.zeros([output_dim, input_dim], dtype=complex)
            for col in range(input_dim):
                column_as_matrix = tensor[:, :, col]
                #Now we want to flatten it into a vector. For the ordering to be correct, 
                #we need the right indices to be more significant. 
                #since left corresponds to rows and right to columns
                #we want to flatten it as M[:,0], M[:,1], ...
                completed_tensor[:,col] = column_as_matrix.ravel(order='F')
            qubits = [] #We will update the correct qubits later
            circuit_layer.append({"matrix": completed_tensor, "qubits":qubits})
        circuit_layers.insert(0, circuit_layer)
    #Now that the tensors are correct, lets add the correct qubit counts
    circuit_layers = update_circuit_tree_dictionary(system_size, tree, circuit_layers)
    return circuit_layers

def get_correct_qubit_info_from_tree(system_size,tree):
    Nlayers = len(tree)
    non_trash_qubits = list(range(system_size))
    qubit_action_dict = {}
    for l in range(Nlayers):
        layer = tree[l]
        qubits_available_in_the_layer = non_trash_qubits
        qubit_layer_dict = {}
        for t, tensor in enumerate(layer):
            output_dim = np.prod(tensor.shape[:2])
            output_qubits = int(np.log2(output_dim))
            input_dim = tensor.shape[2]
            input_qubits = int(np.log2(input_dim))
            qubits_used_by_tensor = qubits_available_in_the_layer[0:output_qubits] #This is the result to be stored
            qubits_available_in_the_layer = qubits_available_in_the_layer[output_qubits:] #remove the qubits used by the tensor
            num_of_trash_qubits = output_qubits-input_qubits #dummy legs comming into the tensor
            if num_of_trash_qubits>0:
                trash_qubits = qubits_used_by_tensor[(output_qubits-num_of_trash_qubits):output_qubits]
                non_trash_qubits = [x for x in non_trash_qubits if x not in trash_qubits] #remove the trash qubits
            qubit_layer_dict[t] = qubits_used_by_tensor #store the result
        qubit_action_dict[Nlayers-1-l] = qubit_layer_dict #store the result for the entire layer
    return qubit_action_dict

def invert_qubit_ordering_for_qiskit(system_size, qubit_list):
    n = system_size -1
    qubit_list = [n-q for q in qubit_list]
    return qubit_list

def update_circuit_tree_dictionary(system_size,tree,circuit_layers):
    correct_qubit_info = get_correct_qubit_info_from_tree(system_size, tree)
    for (l,layer) in enumerate(circuit_layers):
        for (t,elem) in enumerate(layer):
            circuit_layers[l][t]['qubits'] = invert_qubit_ordering_for_qiskit(system_size,correct_qubit_info[l][t])
    return circuit_layers




