# Import necessary libraries
import numpy as np
from benchmarking_utils import get_qualtran_circuit_depth, get_our_cost, get_qiskit_stats, get_our_cost_dicke
import matplotlib.pyplot as plt

from math import comb
import itertools


def dicke_state_vector(n, k):
    """
    Generate the statevector of an n-qubit Dicke state with Hamming weight k.
    The Dicke state is the equal superposition of all computational basis states with exactly k ones.
    """
    # Number of basis states
    dim = 2**n
    # Initialize statevector with zeros
    statevector = np.zeros(dim, dtype=complex)

    # Iterate over all basis states
    for i in range(dim):
        # Count the number of ones in the binary representation of i
        if bin(i).count('1') == k:
            # Set amplitude to 1/sqrt(comb(n, k)) for equal superposition
            statevector[i] = 1 / np.sqrt(comb(n, k))

    return statevector


def dicke_state_coeff_map(n, k):
    """
    Generate all n-bit bitstrings with exactly k ones and create a dictionary
    where keys are integer representations of these bitstrings and values are 1.
    Combines bitstring generation and dictionary creation in one function.
    """
    positions = itertools.combinations(range(n), k)
    coeff_map = {}
    bitstrings = []
    
    for pos in positions:
        bits = ['0'] * n
        for p in pos:
            bits[p] = '1'
        bitstring = ''.join(bits)
        bitstrings.append(''.join(bits))
        integer_val = int(bitstring, 2)
        coeff_map[integer_val] = 1
    
    return coeff_map, bitstrings

if __name__ == '__main__':
    sizes = range(4,18,2)
    qiskit_sizes = range(4,18,2)

    k = 3

    qualtran_depths = np.zeros(len(sizes))
    qualtran_qubits = np.zeros(len(sizes))
    our_depths = np.zeros(len(sizes))
    our_qubits = np.zeros(len(sizes))

    qiskit_depths = np.zeros(len(qiskit_sizes))
    qiskit_qubits = np.zeros(len(qiskit_sizes))
    

    # Collect statistics for each system size
    print('Runnning the Qiskit Circuits')
    for i, num_qubits in enumerate(qiskit_sizes):
        print(num_qubits)
        normalized_amplitudes = dicke_state_vector(num_qubits, k)

        # Calculate the circuit statistics for the qiskit implementation
        qiskit_depths[i], _, _, _ = get_qiskit_stats(normalized_amplitudes, num_qubits)

    print('Runnning the Our Circuits and Qualtran Circuits')
    for i, num_qubits in enumerate(sizes):
        regex = "(0)*1(0)*1(0)*1(0)*"
        print(num_qubits)
        coeff_map, bitstrings = dicke_state_coeff_map(num_qubits, k)

        qualtran_depths[i], qualtran_qubits[i] = get_qualtran_circuit_depth(coeff_map, num_qubits, mu=3)
        our_depths[i], our_qubits[i] = get_our_cost_dicke(regex,num_qubits, 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*4.25,3.5), sharex=True)
    ax1.plot(sizes, qualtran_depths, label='Qualtran')
    ax1.plot(sizes, our_depths, label='Us')
    ax1.plot(qiskit_sizes, qiskit_depths, label='Qiskit')
    ax2.plot(sizes, qualtran_qubits, label='Qualtran')
    ax2.plot(sizes, our_qubits, label='Us')
    ax2.plot(sizes, sizes, label='Qiskit')
    ax1.set_xlabel("Dicke State Size")
    ax2.set_xlabel("Dicke State Size")
    ax1.set_ylabel("Circuit Depth")
    ax2.set_ylabel("No. of Total Qubits")
    ax1.set_yscale('log')
    ax1.set_xticks(sizes)
    ax1.legend()
    plt.savefig('figs/dicke_state_cost_comparison_us_vs_qualtran_vs_qiskit_v2.pdf', bbox_inches='tight')
    plt.show()