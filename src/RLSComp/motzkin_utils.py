from pyformlang.finite_automaton import DeterministicFiniteAutomaton, State, Symbol
from .regex_utils import accepted_strings_cyclic_dfa
import math

# Given a system size n, it builds the DFA for a Motzkin state 
# By default, 0 goes up and 1 goes down. 
# You can switch the zeroup flag to False if you want 1 to go up and 0 down.
def motzkinDFA(n, zeroup=True):
	if n%2:
		raise ValueError("Motzkin states with two symbols only exist for even n.")

	dfa = DeterministicFiniteAutomaton()

	max_height = int(n/2)
	# Initiate the root state
	height = 0
	root = State(height)
	dfa.add_start_state(root)
	dfa.add_final_state(root)
	prev_state = root

	# Build the other states
	for height in range(1, max_height + 1):
		next_state = State(height)
		if zeroup:
			# 0 goes up
			upsymbol = Symbol("0")
			# 1 goes down
			downsymbol = Symbol("1")
		else:
			# 0 goes up
			upsymbol = Symbol("1")
			# 1 goes down
			downsymbol = Symbol("0")
			
		dfa.add_transition(prev_state, upsymbol, next_state)
		dfa.add_transition(next_state, downsymbol, prev_state)

		prev_state = next_state

	return dfa

# Given a system size n, gives you the strings that compose the Motzkin 
def get_motzkin_strings(n):
	dfa = motzkinDFA(n)
	return accepted_strings_cyclic_dfa(dfa, n)
    
def catalan(n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    return math.comb(2*n, n) // (n + 1)

def even_motzkin_count(n):
	if n % 2 != 0:
		raise ValueError("n must be even")
	k = n // 2
	return catalan(k)

