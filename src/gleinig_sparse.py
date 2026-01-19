import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector,Operator
from qiskit.circuit import Gate
from typing import Set, Dict, Tuple, List
import copy
from itertools import combinations
import time

def most_imbalanced_split(strings: Set[str], n: int) -> Tuple[int, Set[str], Set[str]]:
    best_b, best_imbalance, best_T0, best_T1 = -1, -1, set(), set()
    for b in range(n):
        T0 = {x for x in strings if x[b] == '0'}
        T1 = {x for x in strings if x[b] == '1'}
        if T0 and T1:
            imbalance = abs(len(T0) - len(T1))
            if imbalance > best_imbalance:
                best_b, best_T0, best_T1 = b, T0, T1
                best_imbalance = imbalance

    return best_b, best_T0, best_T1

# Returns True if the bitstring x has the values `vals` at the positions `bits`
def matches(x, bits, vals):
    return all(int(x[bit]) == val for bit, val in zip(bits, vals))


def get_M_params(alpha, beta):
    # Normalize:
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha / norm, beta / norm

    # Compute omega:
    omega = np.arctan2(abs(alpha), abs(beta))

    # Compute phi:
    phi = -np.angle((beta / alpha) * np.tan(omega))

    return omega, phi


def build_M(alpha, beta):
    omega, phi = get_M_params(alpha, beta)
    s = np.sin(omega)
    c = np.cos(omega)
    eiφ = np.exp(1j * phi)
    e_minus_iφ = np.exp(-1j * phi)
    M = np.array([
        [s, c * eiφ],
        [c * e_minus_iφ, -s]
    ])
    
    return M


def multi_controlled_M_gate(alpha: float, beta: float, num_controls: int, name: str = "CM") -> Gate:
    
    M = build_M(alpha, beta)
    M_op = Operator(M)

    M_instr = M_op.to_instruction()
    if num_controls == 0:
        CM = M_instr
    else:
        CM = M_instr.control(num_controls, label=name)

    return CM


def subroutine_gleinig(S: Set[str], coeffs: Dict[str, complex], n: int, previous_count_CNOTS, previous_count_1qb_gates) -> QuantumCircuit:
    C = QuantumCircuit(n)
    dif_qubits, dif_values = [], []
    dif_qubits_qiskit = []
    T = set(S)

    count_CNOTS = 0
    count_1qb_gates = 0

    S_prime = S.copy()
    coeffs_prime = coeffs.copy()

    # WHILE loop 1
    while len(T) > 1:
        b, T0, T1 = most_imbalanced_split(T, n)
        dif_qubits.append(b)
        dif_qubits_qiskit.append(n-1-b)
        if len(T0) < len(T1):
            T = T0
            dif_values.append(0)
        else:
            T = T1
            dif_values.append(1)

    x1 = next(iter(T)) # There is only one element
    dif = dif_qubits.pop()
    dif_qiskit = dif_qubits_qiskit.pop()
    dif_values.pop()

    # Build T' (second branch)
    T_prime = {x for x in S if matches(x, dif_qubits, dif_values)}
    T_prime.remove(x1)

    # WHILE loop 2
    while len(T_prime) > 1:
        b, T0, T1 = most_imbalanced_split(T_prime, n)
        dif_qubits.append(b)
        dif_qubits_qiskit.append(n-1-b)
        if len(T0) < len(T1):
            T_prime = T0
            dif_values.append(0)
        else:
            T_prime = T1
            dif_values.append(1)

    x2 = next(iter(T_prime))

    # Apply NOT to flip x1[dif] to 1
    if x1[dif] == '0':
        C.x(dif_qiskit)
        count_1qb_gates += 1
        x1 = x1[:dif] + '1' + x1[dif+1:]
        x2 = x2[:dif] + ('1' if x2[dif] == '0' else '0') + x2[dif+1:]
        S_new = set()
        coeffs_new = {}
        for s in S_prime:
            new_element = s[:dif] + ('1' if s[dif] == '0' else '0') + s[dif+1:]
            S_new.add(new_element)
            # Accumulate the contribution of s in the target new_element
            coeffs_new[new_element] = coeffs_new.get(new_element, 0+0j) + coeffs_prime.get(s, 0+0j)
        S_prime = S_new
        coeffs_prime = coeffs_new

    # Apply CNOTs to make x1 and x2 differ only in 'dif'
    for b in range(n):
        if b != dif and x1[b] != x2[b]:
            C.cx(dif_qiskit, n-1-b)
            count_CNOTS += 1
            new_S = set()
            new_coeffs = {}
            for s in S_prime:
                if s[dif] == '1':
                    # Flip the target bit at position b in the *string* representation
                    flipped_bit = '1' if s[b] == '0' else '0'
                    flipped_s = s[:b] + flipped_bit + s[b+1:]
                    new_S.add(flipped_s)
                    # Accumulate the original amplitude coeffs_prime[s] in flipped_s
                    new_coeffs[flipped_s] = new_coeffs.get(flipped_s, 0+0j) + coeffs_prime.get(s, 0+0j)
                    if s == x1:
                        x1 = x1[:b] + ('1' if x1[b] == '0' else '0') + x1[b+1:]
                    if s == x2:
                        x2 = x2[:b] + ('1' if x2[b] == '0' else '0') + x2[b+1:]
                else:
                    new_S.add(s)
                    # si s no cambia, su amplitud pasa tal cual
                    new_coeffs[s] = new_coeffs.get(s, 0+0j) + coeffs_prime.get(s, 0+0j)
            S_prime = new_S
            coeffs_prime = new_coeffs

    # Apply NOTs on dif_qubits to set them to 1:
    for b in dif_qubits:
        if x2[b] == '0':
            C.x(n-1-b)
            count_1qb_gates += 1
            x1 = x1[:b] + ('1' if x1[b] == '0' else '0') + x1[b+1:]
            x2 = x2[:b] + ('1' if x2[b] == '0' else '0') + x2[b+1:]
            S_new = set()
            coeffs_new = {}
            for s in S_prime:
                new_element = s[:b] + ('1' if s[b] == '0' else '0') + s[b+1:]
                S_new.add(new_element)
                coeffs_new[new_element] = coeffs_new.get(new_element, 0+0j) + coeffs_prime.get(s, 0+0j)
            S_prime = S_new
            coeffs_prime = coeffs_new

    # Apply the controlled G gate:
    norm = np.sqrt(np.abs(coeffs_prime.get(x1, 0+0j))**2 + np.abs(coeffs_prime.get(x2, 0+0j))**2)
    if norm == 0:
        raise RuntimeError("Norm zero para (x1,x2) — algo fue mal antes.")
    alpha = coeffs_prime[x2] / norm
    beta = coeffs_prime[x1] / norm
    CM_gate = multi_controlled_M_gate(alpha, beta, len(dif_qubits))
    C.append(CM_gate, dif_qubits_qiskit + [dif_qiskit])
    #print('alpha =', alpha, ', beta =', beta, ', controlled on', dif_qubits_qiskit + [dif_qiskit])
    count_1qb_gates += 2 # According to the paper it can be implemented as 2 rotations (does qiskit do it this way?)
    count_CNOTS += len(dif_qubits_qiskit) + 1 # This is probably wrong, I don't know exactly how many CNOTS are taken to implement multicontrolled

    # Update S_prime by eliminating x1 and just keeping x2:
    S_new = set()
    coeffs_new = {}
    for s in S_prime:
        if s != x1 and s != x2:
            S_new.add(s)
            coeffs_new[s] = coeffs_prime.get(s, 0+0j)
        elif s == x2:
            S_new.add(s)
            coeffs_new[s] = np.sqrt(np.abs(coeffs_prime.get(x1, 0+0j))**2 + np.abs(coeffs_prime.get(x2, 0+0j))**2)
    S_prime = S_new
    coeffs_prime = coeffs_new

    total_count_CNOTS = previous_count_CNOTS + count_CNOTS
    total_count_1qb_gates = previous_count_1qb_gates + count_1qb_gates

    return S_prime, coeffs_prime, C, total_count_CNOTS, total_count_1qb_gates



def sparse_gleinig(S: Set[str], coeffs: Dict[str, complex], n: int) -> QuantumCircuit:
    """
    Given a sparse state |psi> = sum_{x in S} c_x |x⟩, builds a circuit C such that C|0^n⟩ = |psi>.
    """
    remaining_S = set(S)
    remaining_coeffs = copy.deepcopy(coeffs)
    circuit_stack = []
    count_CNOTS = 0
    count_1qb_gates = 0

    # While more than one basis state remains, apply Algorithm 1
    while len(remaining_S) > 1:
        # Reduce one pair using Algorithm 1
        remaining_S, remaining_coeffs, C_partial, count_CNOTS, count_1qb_gates = subroutine_gleinig(remaining_S, remaining_coeffs, n, count_CNOTS, count_1qb_gates)
        circuit_stack.append(C_partial)

    # Now reduce the remaining state to |0...0⟩ with NOTs if needed
    last_state = next(iter(remaining_S))
    final_cleanup = QuantumCircuit(n)
    for i, bit in enumerate(last_state):
        if bit == '1':
            final_cleanup.x(n-1-i)
            count_1qb_gates += 1
            #print('Add a NOT gate on qubit', n-1-i, '\n')

    # Compose full circuit (apply inverses in reverse order)
    full = QuantumCircuit(n)
    full.compose(final_cleanup, inplace=True)
    for C_partial in reversed(circuit_stack):
        full.compose(C_partial.inverse(), inplace=True)

    return full, count_CNOTS, count_1qb_gates


def build_coeffs(S, coeff_vector):
    S = list(S)
    coeff_vector = np.array(coeff_vector, dtype=complex)    
    norm = np.linalg.norm(coeff_vector)
    coeffs_normalized = coeff_vector / norm
    return {s: c for s, c in zip(S, coeffs_normalized)}


def generate_w_state(n):
    S = ['0'*i + '1' + '0'*(n - i - 1) for i in range(n)]
    coeffs = np.ones(n)
    return S, coeffs

def generate_dicke_state(n, k):
    # All bitstrings of length n with Hamming weight k
    S = [''.join('1' if i in ones else '0' for i in range(n))
         for ones in combinations(range(n), k)]
    amplitude = 1 / np.sqrt(len(S))
    coeffs = np.full(len(S), amplitude)
    return S, coeffs

def generate_w_3banded(n): # Given n >= 3
    S = ['0'*i + '111' + '0'*(n - i - 3) for i in range(n-2)]
    coeffs = np.ones(len(S))
    return S, coeffs

def generate_INC(n): # Given n >= 3
    S = ['0'*i + '1'*(n - i) for i in range(n+1)]
    coeffs = np.ones(len(S))
    return S, coeffs


if __name__ == "__main__":
    option = 'D'

    if option == 'A':
        n = 3
        S = ['001', '100', '111']
        coeff_vector = [2, 8, 10]
    elif option == 'B':
        n = 4
        S = ['1001', '0010', '1000', '0111', '0110'] # They need to be all different
        coeff_vector = [1, 4, 3, 2, 10]
    elif option == 'C':
        n = 100
        print('\nW-state with N =', n)
        S, coeff_vector = generate_w_state(n)
    elif option == 'D':
        n = 20
        k = 2
        print('\nDicke state on N =', n, 'with', k, 'excitations')
        S, coeff_vector = generate_dicke_state(n,k)
    elif option == 'E':
        n = 100
        print('\n3-banded W-state with N =', n)
        S, coeff_vector = generate_w_3banded(n)
    elif option == 'F':
        n = 99
        print('\nINC (domain wall) state on N =', n)
        S, coeff_vector = generate_INC(n)

    coeffs = build_coeffs(S, coeff_vector)

    start = time.time()
    qc, count_CNOTS, count_1qb_gates = sparse_gleinig(S, coeffs, n)
    end = time.time()

    check = False

    if check:
        initial_state = Statevector.from_label('0'*n)
        final_state = initial_state.evolve(qc)
        qr = QuantumRegister(n)
        print("\nNonzero amplitudes:")
        all_good = True
        tol = 1e-8
        for i, amp in enumerate(final_state.data):
            if np.abs(amp) > tol:
                b = format(i, f'0{len(qr)}b')  # binary string
                if b not in coeffs or np.abs(coeffs[b] - amp) > tol:
                    all_good = False
                print(f"|{b}> : {amp}")

        if not all_good:
            print('We are obtaining the incorrect state.')
        else:
            print('We get the correct state!')

    print('Time elapsed:', end-start)
    print('Number of CNOT gates:', count_CNOTS) # (Probably wrong counting)
    print('Number of one-qubit gates:', count_1qb_gates,'\n')
