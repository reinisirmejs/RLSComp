from RLSComp.interface import build_SeqRLSP_circuit
from RLSComp.benchmarking_utils import get_our_cost_from_circ
from RLSComp.bartschi2019_dicke import dicke_state
from qiskit import transpile

regex = "(0)*1(0)*"
n=30

bitstrings = ["0"*n, "1"*n]
circ_GHZ = build_SeqRLSP_circuit(bitstrings,use_isometries=True)
depth_ghz, qubits_ghz, our_circ_transpiled_ghz = get_our_cost_from_circ(circ_GHZ)
print("GHZ: ",our_circ_transpiled_ghz.count_ops())

# circ = build_SeqRLSP_circuit(regex, n,use_isometries=True)
# fig_iso = circ.draw("mpl")
# fig_iso.savefig("w_4_isos.pdf", bbox_inches="tight")
# depth, qubits, our_circ_transpiled = get_our_cost_from_circ(circ)
# all_gates = our_circ_transpiled.size()
# two_gates = our_circ_transpiled.num_nonlocal_gates()
# print(two_gates)
# fig = our_circ_transpiled.draw("mpl",fold=-1)
# fig.savefig("w_4_circuit.pdf", bbox_inches="tight")

# circ_bartschi = dicke_state(n,1)
# circ_bartschi = transpile(circ_bartschi, basis_gates=['cx', 'rz', 'sx', 'x', 'id'],optimization_level=3)
# print(circ_bartschi.count_ops())
# fig_bartschi = circ_bartschi.draw("mpl",fold=-1)
# fig_bartschi.savefig("w_4_bartschi.pdf")

