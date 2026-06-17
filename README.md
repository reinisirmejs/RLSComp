# RLSComp

**RLSComp** is a Python package for compiling quantum circuits for Regular Language States  
It provides both **core functionality** for general use and **full reproducibility** for research experiments, including test suites. The package is based on [preprint:arxiv/2602.02698](https://arxiv.org/abs/2602.02698).

<p align="center">
  <a href="#features">Features</a> -
  <a href="#installation">Installation</a> -
  <a href="#citation">Cite Us</a>
</p>

---

## Features

- Compile quantum circuits for preparing Regular Language States using the Sequential (SeqRLSP) or Tree (TreeRLSP) approach.
- Reproduce experimental scripts and tests (available via Git clone).
- Lightweight core installation for ease of use.

---

# Installation

## 1. Core functionality (pip install)

If you only need the main functions (e.g., `build_SeqRLSP_circuit` and `build_TreeRLSP_circuit`), you can install directly from GitHub:

```
pip install git+https://github.com/reinisirmejs/RLSComp.git
```

### Using the Core Functions

Both `build_SeqRLSP_circuit` and `build_TreeRLSP_circuit` return a `qiskit.circuit.QuantumCircuit` that prepares the equal superposition over all strings in the specified language. They are imported as:

```python
from RLSComp import build_SeqRLSP_circuit, build_TreeRLSP_circuit
```

Both functions accept the same types of input.

#### Regex and system size

```python
regex = "(0)*1(0)*" 
num_qubits = 5 
circuit = build_SeqRLSP_circuit(regex, num_qubits)
```

The `complement=True` keyword prepares instead the equal superposition over all strings that do **not** match the regex:

```python
circuit = build_SeqRLSP_circuit(regex, num_qubits, complement=True)
```

#### Bitstring list

All bitstrings must have the same length. The system size is inferred automatically.

```python
bitstring_list = ["000", "010", "111"] 
circuit = build_SeqRLSP_circuit(bitstring_list) 
```

The `complement=True` keyword is also supported here and prepares the superposition over all same-length bitstrings **not** in the list:

```python
circuit = build_SeqRLSP_circuit(bitstring_list, complement=True)
```

#### DFA

The DFA must be a `pyformlang.finite_automaton.DeterministicFiniteAutomaton` object over the binary alphabet `{0, 1}`.

For an **acyclic DFA** (finite language), the system size is inferred from the DFA structure:

```python
circuit = build_SeqRLSP_circuit(dfa) 
```

For a **cyclic DFA** (infinite language, e.g., built from a regex), a `system_size` must be provided:

```python
circuit = build_SeqRLSP_circuit(dfa, system_size=6)
```

The `complement=True` keyword is also supported for both acyclic and cyclic DFAs:

```python
circuit = build_SeqRLSP_circuit(dfa, complement=True)             # acyclic
circuit = build_SeqRLSP_circuit(dfa, system_size=6, complement=True)  # cyclic
```

#### MPS

```python
circuit = build_SeqRLSP_circuit(MPS) 
```

where the MPS over a sequence of symbols $x_1, x_2, \dots, x_N \in \Sigma$ is represented as a `List` of rank-3 tensors $\{A^{[1]}, A^{[2]}, \dots , A^{[N]}\}$. Each local tensor is stored as a `numpy.ndarray` and indexed as $A[x,i,j]$, where
- $x$ is the physical index corresponding to the input symbol,
- $i$ is the left virtual (bond) index,
- $j$ is the right virtual (bond) index.

#### Isometry decomposition

`build_SeqRLSP_circuit` uses `use_isometries=True` by default, which keeps the rectangular isometries and decomposes them using Qiskit's `Isometry` gate synthesis. Setting `use_isometries=False` instead pads each isometry to a full square unitary and decomposes it using Qiskit's `UnitaryGate` synthesis:

```python
circuit = build_SeqRLSP_circuit(regex, num_qubits, use_isometries=False)
```

---

## 2. Full functionality and reproducibility

Clone the repository and install the dependences from `misc/requirements.txt`.

#### Running Tests

Run the tests to verify the correct installation

```
pytest tests/ 
```

### Running experiments

We can run an experiment that compares the resources for different methods using the `run_experiment.py`. We include an example config in `configs/my_experiment.yaml`.

```
python -m RLSComp.run_experiment --config=configs/my_experiment.yaml
```

Any experiment can be run by creating an arbitrary config  `my_experiment.yaml` file:

```
experiment: my_experiment

input_type: regex          # Options: "regex" or "bitstring_list"
# If using regex
regex: "(0)*1(0)*"
regex_complement: False  # If True, prepares all bitstrings that do not match the regex

# If using input_type: bitstring_list
# Provide bitstring_list = ["0001", "0010", "1000"]

system_sizes: [4, 8]  # System sizes to benchmark. For input_type: regex it is possible to provide multiple
methods: [SeqRLSP, SeqUnitRLSP, TreeRLSP, qualtran, qiskit, gleinig_sparse, bartschi2019_dicke]
# These are all available methods. Note that some methods might not be suitable for some inputs/large system sizes     

plotting: True
plot_output: figs/my_experiment_plot.pdf

# Optional: per-method timeout in seconds (default: 36000 = 10h).
# If a method exceeds this, it is skipped for all subsequent system sizes.
timeout: 300

# Optional: per-method maximum system size. Methods not listed have no limit.
method_max_n:
  qiskit: 14
  qualtran: 24
  gleinig_sparse: 64
```

For large-scale experiments, the `timeout` and `method_max_n` options help manage resources. The timeout applies per method per system size — if a method exceeds it, that method is automatically skipped for all remaining (larger) sizes. The `method_max_n` dictionary lets you set hard system-size caps for individual methods, overriding the built-in defaults. Results are saved incrementally after each system size, so partial results are preserved even if the process is interrupted.

## Citation

If you use this code, please cite this paper using the following bibtex entry:

```
@misc{RLSComp,
      title={Compiling Quantum Regular Language States},
      author={Armando Bellante and Reinis Irmejs and Marta Florido-Llinàs and María Cea Fernández and Marianna Crupi and Matthew Kiser and J. Ignacio Cirac},
      year={2026},
      eprint={2602.02698},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2602.02698},
}
```


