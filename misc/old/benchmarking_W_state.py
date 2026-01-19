# Import necessary libraries
import numpy as np
from benchmarking_utils import get_qualtran_circuit_depth, get_our_cost, get_qiskit_stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sizes = range(4,36,4)
    qiskit_sizes = range(4,14,2)

    qualtran_depths = np.zeros(len(sizes))
    qualtran_qubits = np.zeros(len(sizes))
    our_depths = np.zeros(len(sizes))
    our_qubits = np.zeros(len(sizes))

    qiskit_depths = np.zeros(len(qiskit_sizes))
    qiskit_qubits = np.zeros(len(qiskit_sizes))
    

    # Collect statistics for each system size
    for i, num_qubits in enumerate(qiskit_sizes):
        # Create an amplitude vector with 1s at powers of 2 positions
        # This is the state_vector for the W state
        amplitudes = np.zeros(2**num_qubits)
        for j in range(num_qubits):
            amplitudes[2**j] = 1

        # Normalize the amplitudes (required for valid quantum state)
        normalized_amplitudes = np.array(amplitudes) / np.linalg.norm(amplitudes)

        # Calculate the circuit statistics for the qiskit implementation
        qiskit_depths[i], _, _, _ = get_qiskit_stats(normalized_amplitudes, num_qubits)

    for i, num_qubits in enumerate(sizes):
        coeff_map = {2**j: 1 for j in range(num_qubits)}

        qualtran_depths[i], qualtran_qubits[i] = get_qualtran_circuit_depth(coeff_map, num_qubits, mu=3)
        our_depths[i], our_qubits[i] = get_our_cost(num_qubits)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*4.25,3.5), sharex=True)
    ax1.plot(sizes, qualtran_depths, label='Qualtran')
    ax1.plot(sizes, our_depths, label='Us')
    ax1.plot(qiskit_sizes, qiskit_depths, label='Qiskit')
    ax2.plot(sizes, qualtran_qubits, label='Qualtran')
    ax2.plot(sizes, our_qubits, label='Us')
    ax2.plot(sizes, sizes, label='Qiskit')
    ax1.set_xlabel("W State Size")
    ax2.set_xlabel("W State Size")
    ax1.set_ylabel("Circuit Depth")
    ax2.set_ylabel("No. of Total qubits")
    ax1.set_yscale('log')
    ax1.set_xticks(sizes)
    ax1.legend()
    plt.savefig('figs/cost_comparison_us_vs_qualtran_vs_qiskit.pdf', bbox_inches='tight')
    plt.show()