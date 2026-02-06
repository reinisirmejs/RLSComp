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

Use the circuit builders directly:

`from RLSComp import build_SeqRLSP_circuit, build_TreeRLSP_circuit`

which both admit the following possible inputs:

#### Regex or system size

```
regex = "0*10*" 
num_qubits = 5 
circuit = build_SeqRLSP_circuit(regex, num_qubits)
```

#### Bitstring list

```
bitstring_list = ["000", "010", "111"] 
circuit = build_SeqRLSP_circuit(bitstring_list) 
```

#### DFA

The DFA is given as a `pyformlang.finite_automaton.DeterministicFiniteAutomaton` object

```
circuit = build_SeqRLSP_circuit(dfa) 
```

#### MPS

```
circuit = build_SeqRLSP_circuit(MPS) 
```
where the MPS over a sequence of symbols $x_1, x_2, \dots, x_N \in \Sigma$ is represented as a `List` of rank-3 tensors $\{A^[1], A^[2], \dots , A^[N]\}$. Each local tensor is stored as a `numpy.ndarray` and indexed as $A[x,i,j]$, where
- $x$ is the physical index corresponding to the input symbol,
- $i$ is the left virtual (bond) index,
- $j$ is the right virtual (bond) index.



## 2. Full functionality and reproducibility

Clone the repository and install the dependences from `misc/requirements.txt`.

#### Running Tests

Run the tests to verify the correct installation

```
pytest tests/ 
```

### Running experiments

We can run an experiment that compares the resources for different methods using the `run_experiment.py`. We include an example config for benchmarking the **W-state** preparation.

```
python src/RLSComp/run_experiment.py --config=configs/w_state_config.yaml
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

system_sizes: [4]  # System sizes to benchmark. For input_type: regex it is possible to provide multiple
methods: [SeqRLSP, TreeRLSP, qualtran, qiskit, gleinig_sparse, bartschi2019_dicke]
# These are all available methods. Note that some methods might not be suitable for some inputs/large system sizes     

plotting: True
plot_output: figs/my_experiment_plot.pdf
```

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



