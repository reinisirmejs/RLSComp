import numpy as np
from qiskit.quantum_info import Statevector

"""
Functions to check if we have prepared the correct state given isometries. These algorithms are inefficient
and only work for small system sizes. MOVE TO TESTING FILE LATER
"""

def verify_state(target_state, circuit):
    statevector = Statevector.from_instruction(circuit)
    dimSysAnc = len(statevector.data)
    dimSys = len(target_state)
    dimAnc = dimSysAnc//dimSys
    n_ancilla = int(np.ceil(np.log2(dimAnc)))
    # Assuming that the ancilla at the end decouples into the all 0 state,
    # The relevant system eigenvector is the every 2**n_ancilla element
    statevector_no_ancilla = statevector.data[0::(2**n_ancilla)]
    overlap = np.abs(np.dot(target_state, statevector_no_ancilla.conj()))
    return np.allclose(overlap,1)

def dictionary_from_state(statevector):
    N = len(statevector)
    statevector_dict = {}
    for i in range(N):
        if abs(statevector[i])>1e-12:
            statevector_dict[bin(i)[2:].zfill(int(np.log2(N)))] = statevector[i]
    return statevector_dict

