# RLSComp

**RLSComp** is a Python package for compiling quantum circuits for Regular Language States  
It provides both **core functionality** for general use and **full reproducibility** for research experiments, including test suites. The package is based on preprint:arxiv.org/abs/2601.xxxxx

---

## Features

- Compile quantum circuits for preparing Regular Language States using the Sequential (SeqRLSP) or Tree (TreeRLSP) approach.
- Reproduce experimental scripts and tests (available via Git clone).
- Lightweight core installation for ease of use.
- Full functionality with development environment for testing and experiments.

---

## Installation

### 1. Core functionality (pip install)

If you only need the main functions (e.g., `build_SeqRLSP_circuit` and `build_TreeRLSP_circuit`), you can install directly from GitHub:

`pip install git+https://github.com/reinisirmejs/RLSComp.git`

# RLSComp

RLSComp is a Python package for compiling quantum circuits for Regular Language States. It provides both core functionality for general use and full reproducibility for research experiments, including test suites.

The package is based on the preprint: https://arxiv.org/abs/2601.xxxxx

## Features

- Compile quantum circuits for preparing Regular Language States using the Sequential (SeqRLSP) or Tree (TreeRLSP) approach.
- Reproduce experimental scripts and tests (available via Git clone).
- Lightweight core installation for ease of use.
- Full functionality with a development environment for testing and experiments.

## Installation

### Core functionality

Install directly from the GitHub repository using pip: `pip install git+https://github.com/reinisirmejs/RLSComp.git`

### Full functionality and reproducibility

Clone the repository and install in a virtual environment using the repository requirements, then install the package in editable mode. (See `misc/requirements.txt` and the repository root for the recommended workflow.)

## Using the Core Functions

### Import

Use the circuit builders directly:

`from RLSComp import build_SeqRLSP_circuit, build_TreeRLSP_circuit`

### Regex or system size

```
regex = "0*(10*)*" 
num_qubits = 5 
circuit = build_SeqRLSP_circuit(regex, num_qubits) 
print(circuit)
```

### Bitstring list

```
bitstring_list = ["000", "010", "111"] 
circuit = build_SeqRLSP_circuit(bitstring_list) 
print(circuit)
```

### DFA

The DFA is given as a `` object

```
circuit = build_SeqRLSP_circuit(dfa) 
print(circuit)
```

### MPS

The MPS is given as a object
```
circuit = build_SeqRLSP_circuit(MPS) 
print(circuit)
```


## Running Tests

Run the test suite with pytest using the repository setup. Optional coverage can be enabled via pytest coverage flags.

## Example Workflow

`from RLSComp import build_SeqRLSP_circuit`  
`circuit = build_SeqRLSP_circuit(regex="0*(10*)*", num_qubits=5)`  
`result = simulate(circuit)  # example function`  
`print(result)`

## Project Structure

- `RLSComp/` - package source
- `tests/` - test suite
- `requirements.txt` - development and reproducibility dependencies
- `setup.py` or `pyproject.toml` - packaging configuration
- `README.md` - project documentation

## Contributing

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Install the package in editable mode.
5. Run the test suite and ensure it passes.

## License

Include your license here.
```



