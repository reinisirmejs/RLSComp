import numpy as np

def FSA_to_MPS(dfa):
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
        Contains 1 at the position of final_state and 0 elsewhere.
    """
    states = dfa.states
    inputs = ['0', '1']
    transitions =  dfa.to_dict()
    begin_state = dfa.start_state
    final_state = dfa.final_states

    # Initialize tensor A with zeros - dimensions: [input_values × states × states]
    A = np.zeros((len(inputs), len(states), len(states)), dtype=int)
    
    # Fill the tensor A based on the transition function
    for start_state in states:
        for input_value in list(transitions[start_state].keys()):
            # Verify that input values are binary (0 or 1)
            assert input_value == 0 or input_value == 1
            
            # Set the corresponding tensor element to 1
            # A[input, current_state, next_state] = 1 indicates a valid transition
            A[input_value, start_state, transitions[start_state][input_value]] = 1
    
    # Create the left boundary vector (initial state)
    v_l = np.zeros(len(states), dtype=int)
    v_l[begin_state] = 1  # Set 1 at the position of the initial state
    
    # Create the right boundary vector (final state)
    v_r = np.zeros(len(states), dtype=int)
    v_r[final_state] = 1  # Set 1 at the position of the final state
    
    # Return the MPS representation (A, v_l, v_r)
    # Note: v_r is reshaped to a column vector using np.newaxis
    return A, v_l, v_r[:,np.newaxis]
