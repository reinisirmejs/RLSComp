from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import DeterministicFiniteAutomaton, State, Symbol
import random

def is_dfa(obj):
    # Best: isinstance(obj, DFA)
    return isinstance(obj, DeterministicFiniteAutomaton)

def is_regex(obj):
    return isinstance(obj, str)

def regex_fmt(user_lst):
    # Format each string in the list
    formatted_strings = [''.join(f'({char})' for char in s) for s in user_lst]
    # Join all the formatted strings with a pipe character
    return '|'.join(formatted_strings)

def regex_to_dfa(regex,complement=False):
    # Convert regex to an NFA
    nfa = Regex(regex).to_epsilon_nfa()

    # Convert NFA to DFA (deterministic finite automaton)
    dfa = nfa.to_deterministic()

    if complement:
        dfa = dfa.get_complement()

    min_dfa = dfa.minimize()

    return min_dfa

def remove_state(dfa, state):
    # Removes one state from the dfa
    transitions = dfa.to_dict()
    for tmpstate in transitions:
        if tmpstate == state:
            for sym in transitions[tmpstate]:
                dfa.remove_transition(state, sym, transitions[tmpstate][sym])
        else:
            for sym in transitions[tmpstate]:
                dest_state = transitions[tmpstate][sym]
                if dest_state == state:
                    dfa.remove_transition(tmpstate, sym, state)
    
    if state in dfa.final_states:
        dfa.remove_final_state(state)
    dfa.states.remove(state)

    return dfa

def get_finite_complement(dfa):
    # Takes the complement of an acyclic dfa, intersecting it with the finite language.
    assert(dfa.is_acyclic())
    assert(len(dfa.final_states)==1)

    dfa = dfa.copy()
    final_state = next(iter(dfa.final_states))

    depth = 1
    layer = [dfa.start_state]
    layers = [layer]
    error_states = {}
    transitions = dfa.to_dict()

    # Walk the dfa layer per layer
    while layer != [final_state]:        
        # Add an error state for the transitions in the next layer
        error_state = State(f"error_{depth}")
        error_states[depth] = error_state
        dfa.states.add(error_state)

        # Build the next layer and insert errors
        next_layer = []
        for node in layer:
            if node != final_state:
                next_layer += list(transitions[node].values())
            for sym in dfa.symbols:   
                if sym not in transitions[node]:
                    dfa.add_transition(node, sym, error_state)

        next_layer = list(set(next_layer))

        layers.append(next_layer)
        layer = next_layer
        depth += 1

    # Link all the error layers
    max_depth = depth - 1
    for i in range(1, max_depth):
        src = error_states[i]
        dst = error_states.get(i + 1)
        for sym in dfa.symbols:
            dfa.add_transition(src, sym, dst)

    dfa.add_final_state(error_states[max_depth])
    dfa = remove_state(dfa, final_state)

    return dfa

def list_to_acdfa(lst,complement=False):
    regex = regex_fmt(lst)
    dfa = regex_to_dfa(regex,complement=False)
    if complement:
        dfa = get_finite_complement(dfa)

    return 

def list_to_acdfa_direct(lst, complement=False):
    dfa = DeterministicFiniteAutomaton()
    state_counter = 0

    root = State(state_counter)
    dfa.add_start_state(root)
    state_counter += 1

    # Dictionary to track the paths from root
    transitions = {}

    for bitstring in lst:
        current = root
        for bit in bitstring:
            symbol = Symbol(bit)
            key = (current, symbol)
            if key in transitions:
                current = transitions[key]
            else:
                new_state = State(state_counter)
                dfa.add_transition(current, symbol, new_state)
                transitions[key] = new_state
                current = new_state
                state_counter += 1
        dfa.add_final_state(current)

    dfa = dfa.minimize()

    if complement:
        dfa = get_finite_complement(dfa)

    return dfa

def compare_dfa_builders(num_strings, bitstring_length):
    bitstrings = [''.join(random.choice("01") for _ in range(bitstring_length)) for _ in range(num_strings)]

    dfa_regex = list_to_acdfa(bitstrings)
    dfa_direct = list_to_acdfa_direct(bitstrings)

    accepted_by_regex = set(accepted_strings(dfa_regex))
    accepted_by_direct = set(accepted_strings(dfa_direct))

    len_regex = len(dfa_regex.states)
    len_direct = len(dfa_direct.states)

    if accepted_by_regex == accepted_by_direct:
        print("Both DFAs accept the same set of strings.")
    else:
        print("Mismatch detected!")
        print("Strings accepted by regex DFA but not by direct DFA:", accepted_by_regex - accepted_by_direct)
        print("Strings accepted by direct DFA but not by regex DFA:", accepted_by_direct - accepted_by_regex)

    if len_regex == len_direct:
        print("Both DFAs have the same state lenght.")
    else:
        print("Mismatch detected!")
        print("Lenght of regex DFA vs direct DFA:", len_regex,len_direct)

    return bitstrings, accepted_by_regex, accepted_by_direct

def accepted_strings_cyclic_dfa(dfa, system_size):
    """Generate all accepted strings of exactly a given system size from a DFA."""
    accepted_strings = set()
    queue = [( "", dfa.start_state )]   # Start BFS from the initial state
    transitions_dict = dfa.to_dict()    # Get all transitions

    while queue:
        current_string, current_state = queue.pop(0)

        # Skip if we exceed system size
        if len(current_string) > system_size:
            continue

        # If current string has exactly system_size and current state is final, add it
        if len(current_string) == system_size and current_state in dfa.final_states:
            accepted_strings.add(current_string)

        # Expand transitions from the current state
        for symbol, next_state in transitions_dict.get(current_state, {}).items():
            queue.append((current_string + str(symbol), next_state))

    return sorted(accepted_strings)

def accepted_strings_acyclic_dfa(dfa):
    """Generate all accepted strings from an acyclic DFA."""
    accepted_strings = set()
    queue = [("", dfa.start_state)]     # Start BFS from the initial state
    transitions_dict = dfa.to_dict()    # Get all transitions

    while queue:
        current_string, current_state = queue.pop(0)

        # If current state is a final state, add current string to accepted strings
        if current_state in dfa.final_states:
            accepted_strings.add(current_string)

        # Explore all transitions from the current state
        for symbol, next_state in transitions_dict.get(current_state, {}).items():
            queue.append((current_string + str(symbol), next_state))

    return sorted(accepted_strings)

def accepted_strings(dfa, system_size=None):
    """Generate accepted strings from a DFA. If the DFA is acyclic, it generates all strings. If cyclic, limits by system size."""
    # Check if the DFA is acyclic
    if dfa.is_acyclic():
        accepted_strings = accepted_strings_acyclic_dfa(dfa)  # No system size needed for acyclic DFA
    else:
        # For cyclic DFA, check if system_size is provided
        # Raise an error if no system size
        if system_size is None:
            raise ValueError("Please specify system size for cyclic DFA.")  
        accepted_strings = accepted_strings_cyclic_dfa(dfa, system_size)  # Use system size for cyclic DFA
    return accepted_strings

