import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import pickle
import numpy as np
import time
import signal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#plt.style.use('misc/plot_style.mplstyle')

from .regex_utils import regex_to_dfa, list_to_acdfa_direct, accepted_strings
from .mps_utils import DFA_to_MPS, MPS_to_list, MPS_to_state, ACDFA_to_MPS
from .circuit_utils import  MPS_to_circuit_SeqUnitRLSP, MPS_to_circuit_SeqIsoRLSP, Tree_to_circuit
from .benchmarking_utils import get_our_cost_from_circ, get_qiskit_stats, get_bartschi2019_stats, get_gleinig_sparse_stats, get_qualtran_sparse_stats
from .motzkin_utils import get_motzkin_strings, motzkin_dfa, even_motzkin_count


def build_mps(input_type, regex, bitstrings, system_size,complement=False):
    if input_type == "regex":
        dfa = regex_to_dfa(regex,complement=complement)
        A, v_l, v_r = DFA_to_MPS(dfa)
        return MPS_to_list(A, v_l, v_r, system_size)
    elif input_type == "bitstring_list":
        acdfa = list_to_acdfa_direct(bitstrings, complement=complement)
        return ACDFA_to_MPS(acdfa)
    elif input_type == "motzkin":
        dfa = motzkin_dfa(system_size)
        A, v_l, v_r = DFA_to_MPS(dfa)
        return MPS_to_list(A, v_l, v_r, system_size)
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

"""
Run our Option 1 - SeqRLSP with unitaries constructed from unitary matrices or isometries respectively
"""

def run_SeqRLSP(n,config):
    start = time.perf_counter()
    MPS_LIST = build_mps(config["input_type"], config.get("regex", None), config.get("bitstrings", []), n, complement=config.get("regex_complement", False))
    our_circ = MPS_to_circuit_SeqIsoRLSP(MPS_LIST)
    depth, qubits, our_circ_transpiled = get_our_cost_from_circ(our_circ)
    end = time.perf_counter()
    our_time = end - start
    all_gates = our_circ_transpiled.size()
    two_gates = our_circ_transpiled.num_nonlocal_gates()
    print("SeqRLSP time was: ", our_time, flush=True)
    return qubits, depth, two_gates, all_gates, our_time, our_circ_transpiled

def run_SeqUnitRLSP(n,config):
    start = time.perf_counter()
    MPS_LIST = build_mps(config["input_type"], config.get("regex", None), config.get("bitstrings", []), n, complement=config.get("regex_complement", False))
    our_circ = MPS_to_circuit_SeqUnitRLSP(MPS_LIST)
    depth, qubits, our_circ_transpiled = get_our_cost_from_circ(our_circ)
    end = time.perf_counter()
    our_time = end - start
    all_gates = our_circ_transpiled.size()
    two_gates = our_circ_transpiled.num_nonlocal_gates()
    print("SeqUnitRLSP time was: ", our_time,flush=True)
    return qubits, depth, two_gates, all_gates, our_time, our_circ_transpiled

"""
Run our option 2: The tree method with log(N) depth, no ancillas but non-local connectivity
"""

def run_TreeRLSP(n,config):
    start = time.perf_counter()
    MPS_LIST = build_mps(config["input_type"], config.get("regex", None), config.get("bitstrings", []), n, complement=config.get("regex_complement", False))
    our_circ = Tree_to_circuit(MPS_LIST)
    depth, qubits, our_circ_transpiled = get_our_cost_from_circ(our_circ)
    end = time.perf_counter()
    our_time = end - start
    all_gates = our_circ_transpiled.size()
    two_gates = our_circ_transpiled.num_nonlocal_gates()
    print("TreeRLSP time was: ", our_time,flush=True)
    return qubits, depth, two_gates, all_gates, our_time, our_circ_transpiled

"""
Run the Qiskit method for comparison. Since it is completely general it is also inefficient and 
thus works for small system sizes only.
"""

def run_qiskit(n,config):
    if n<=14:
        MPS_LIST = build_mps(config["input_type"], config.get("regex", None), config.get("bitstrings", []), n, complement=config.get("regex_complement", False))
        state = MPS_to_state(MPS_LIST)
        start = time.perf_counter()
        depth, _, _, _, qiskit_circ = get_qiskit_stats(state, n)
        end = time.perf_counter()
        qiskit_time = end-start
        qubits = n
        all_gates = qiskit_circ.size()
        two_gates = qiskit_circ.num_nonlocal_gates()
    else:
        print("Qiskit is not supported for n>14, skipping...")
        depth, qubits, all_gates, two_gates, qiskit_time, qiskit_circ = (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    print("Qiskit time was: ", qiskit_time,flush=True)
    return qubits, depth, two_gates, all_gates, qiskit_time, qiskit_circ

"""
Run the Qualtran method for comparison. It is a method for sparse states only, but not very efficient.Limited
to system sizes <32.
"""

def run_qualtran(n,config):
    #fix the coefficient map for the general case
    # Here is a simple fix to avoid the coefficient map for n>16 being calculated since it is too large in the case of the complements
    if n>32:
        print("Too large system size for Qualtran, skipping...")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    if config["input_type"] == "regex" and config.get("regex_complement", False) and n>14:
        print("Too large coefficient map for Qualtran, skipping...")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    if config["input_type"] =="regex":
        ancilla_dim = config.get("ancilla_dim", 2)
        approx_dicke_list_length = n**(ancilla_dim/2)
    #This block ensures that we do not run out of memory by calculating the Gleinig list strings unnecessairly
    if config["input_type"]=="regex" and approx_dicke_list_length>10**6:
        print("Too large coefficient map for Qualtran, skipping...")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, None)

    if config["input_type"] == "regex":
        bitstrings = accepted_strings(regex_to_dfa(config.get("regex", None),complement=config.get("regex_complement", False)), n)
    elif config["input_type"] == "motzkin":
        if even_motzkin_count(n)>5000:
            print("Too large coefficient map for Qualtran, skipping...")
            return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
        bitstrings = get_motzkin_strings(n)
    else:
        bitstrings = config.get("bitstrings", [])

    # if config["input_type"] != "bitstring_list":
    #     bitstrings = accepted_strings(regex_to_dfa(config.get("regex", None),complement=config.get("regex_complement", False)), n)
    # else:
    #     bitstrings = config.get("bitstrings", [])    
    if len(bitstrings)<5000:
        start = time.perf_counter()
        depth, qubits, circ = get_qualtran_sparse_stats(n,bitstrings, mu=3)
        end = time.perf_counter()
        qualtran_time = end-start
        all_gates = circ.size()
        two_gates = circ.num_nonlocal_gates()
    else:
        print("Too many coefficients for Qualtran, skipping...")
        depth, qubits, all_gates, two_gates, qualtran_time, circ = (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    print("Qualtran time was: ", qualtran_time,flush=True)
    return qubits, depth, two_gates, all_gates, qualtran_time, circ

"""
Run the Bärtschi 2019 Dicke state generation method for comparison. Only works for Dicke states.
"""

def run_bartschi2019_dicke(n,config):
    if config.get("input_type") != "regex":
        print("Dicke state generation only supports regex input type, please set input_type to 'regex'.")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    if config.get("regex_complement", False)==True:
        print("Dicke state generation does not support regex complement, please set regex_complement to False.")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    regex = config.get("regex", None)
    k = regex.count('1')
    start = time.perf_counter()
    depth, qubits, circ = get_bartschi2019_stats(n,k)
    end = time.perf_counter()
    time_taken = end - start
    all_gates = circ.size()
    two_gates = circ.num_nonlocal_gates()
    print("Bartschi time was: ", time_taken,flush=True)
    return qubits, depth, two_gates, all_gates, time_taken, circ

"""
Run the Gleinig sparse method for comparison. Only works for sparse states."""

def run_gleinig_sparse(n, config):
    """
    Runs the Gleinig sparse method for a given system size n.
    """
    if n>256:
        print("Maybe too large system size for Sparse Gleinig, skipping...")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    if config["input_type"] == "regex" and config.get("regex_complement", False) and n>=12:
        print("Too large coefficient map for Sparse Gleinig, skipping...")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    if config["input_type"]=="regex":
        ancilla_dim = config.get("ancilla_dim", 2)
        approx_dicke_list_length = n**(ancilla_dim/2)
    #This block ensures that we do not run out of memory by calculating the Gleinig list strings unnecessairly
    if config["input_type"]=="regex" and approx_dicke_list_length>10**6:
        print("Too large coefficient map for Sparse Gleinig, skipping...")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    
    if config["input_type"] == "regex":
        bitstrings = accepted_strings(regex_to_dfa(config.get("regex", None),complement=config.get("regex_complement", False)), n)
    elif config["input_type"] == "motzkin":
        if even_motzkin_count(n)>10000:
            print("Too large coefficient map for Sparse Gleinig, skipping...")
            return (np.nan, np.nan, np.nan, np.nan, np.nan, None)
        bitstrings = get_motzkin_strings(n)
    else:
        bitstrings = config.get("bitstrings", [])

    if len(bitstrings)<10000:
        start = time.perf_counter()
        depth, qubits, circ = get_gleinig_sparse_stats(n, bitstrings)
        end = time.perf_counter()
        gleinig_time = end-start
        all_gates = circ.size()
        two_gates = circ.num_nonlocal_gates()
    else:
        print("Too many coefficients for Gleinig, skipping...")
        depth, qubits, all_gates, two_gates, gleinig_time, circ = (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    print("Gleinig time was: ", gleinig_time, flush=True)
    return qubits, depth, two_gates, all_gates, gleinig_time, circ

RUNNERS = {
    "SeqUnitRLSP": run_SeqUnitRLSP,
    "SeqRLSP": run_SeqRLSP,
    "TreeRLSP": run_TreeRLSP,
    "qiskit": run_qiskit,
    "qualtran": run_qualtran,
    "bartschi2019_dicke": run_bartschi2019_dicke,
    "gleinig_sparse": run_gleinig_sparse,
}

class _MethodTimeout(Exception):
    pass

def run_method(method, n, config):
    if method not in RUNNERS:
        print(f"Unknown method: {method}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, None

    timeout = config.get("timeout", 36000)
    timed_out = False
    def _handler(signum, frame):
        nonlocal timed_out
        timed_out = True
        raise _MethodTimeout()

    try:
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
        result = RUNNERS[method](n, config)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return result
    except _MethodTimeout:
        signal.signal(signal.SIGALRM, old_handler)
        print(f"Timeout: {method} at n={n} exceeded {timeout}s, skipping...", flush=True)
        raise
    except BaseException as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if timed_out:
            print(f"Timeout: {method} at n={n} exceeded {timeout}s, skipping...", flush=True)
            raise _MethodTimeout()
        print(f"Failed running {method} at n={n}")
        print(e)
        return np.nan, np.nan, np.nan, np.nan, np.nan, None


def run_experiment(config, output_path=None):
    sizes = config["system_sizes"]
    methods = config["methods"]

    #Initialize the results dictionary
    RESULT_FIELDS = ["qubits", "depth", "2gates", "gates", "time"]
    results = {
        "config": config,
        "sizes": sizes,
    }
    for method in methods:
        for field in RESULT_FIELDS:
            results[f"{method}_{field}"] = []
        results[f"{method}_circ"] = {}
    SKIP_RESULT = (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    timed_out = set()
    method_max_n = config.get("method_max_n", {})
    #Run the experiment for all system sizes and methods
    for n in sizes:
        sys.stdout.flush()
        print(f"\n--- Running for system size n={n} ---")
        start_time = time.perf_counter()
        for method in methods:
            if method in timed_out:
                print(f"Skipping {method} at n={n} (timed out at smaller size)", flush=True)
                qubits, depth, two_gates, all_gates, run_time, circ = SKIP_RESULT
            elif method in method_max_n and n > method_max_n[method]:
                print(f"Skipping {method} at n={n} (exceeds max_n={method_max_n[method]})", flush=True)
                qubits, depth, two_gates, all_gates, run_time, circ = SKIP_RESULT
            else:
                try:
                    qubits, depth, two_gates, all_gates, run_time, circ = run_method(method, n, config)
                except _MethodTimeout:
                    timed_out.add(method)
                    qubits, depth, two_gates, all_gates, run_time, circ = SKIP_RESULT
            results[f"{method}_qubits"].append(qubits)
            results[f"{method}_depth"].append(depth)
            results[f"{method}_2gates"].append(two_gates)
            results[f"{method}_gates"].append(all_gates)
            results[f"{method}_time"].append(run_time)
            if config.get("save_circuit", False):
                results[f"{method}_circ"][n] = circ
        end_time = time.perf_counter()
        print(f'System size {n} took {end_time-start_time} seconds')
        if output_path is not None:
            with open(output_path, "wb") as f:
                pickle.dump(results, f)
    return results


def plot_results(config, results):

    plot_quantities = {
        "depth": ("Depth", "log"),
        "gates": ("Gates", "log"),
        "time": ("Compilation Time (seconds)", "log"),
    }
    method_res_key = {
        "SeqRLSP": ("SeqRLSP", "SeqRLSP", "-"),
        "SeqUnitRLSP": ("SeqUnitRLSP", "SeqUnitRLSP", "-"),
        "TreeRLSP": ("TreeRLSP", "TreeRLSP", "-"),
        "qiskit": ("qiskit", "Qiskit", "--"),
        "qualtran": ("qualtran", "Qualtran", "--"),
        "bartschi2019_dicke": ("bartschi2019_dicke", "Bärtschi 2019", "--"),
        "gleinig_sparse": ("gleinig_sparse", "Gleinig 2021", "--"),
    }

    ncols = len(plot_quantities)
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.7 * nrows), sharex=True, layout="constrained")
    axes = axes.flatten()

    chosen_methods = config.get("methods", list(method_res_key.keys()))
    x_vals = results.get("sizes")

    ax_idx = 0
    for ax, (quantity, (ylabel, yscale)) in zip(axes, plot_quantities.items()):
        for method in chosen_methods:
            entry = method_res_key.get(method)
            if entry is None:
                continue
            method_key, method_name, method_linestyle = entry
            y_vals = results.get(f"{method_key}_{quantity}", [np.nan] * len(x_vals))
            ax.plot(x_vals, y_vals, label=method_name, linestyle=method_linestyle, marker="x", markersize=3.5, linewidth=1)
            ax.set_xlim([x_vals[0], x_vals[-1]])
        ax.set_ylabel(ylabel)
        if x_vals[-1] >= 250:
            ax.set_xticks([0, 50, 100, 150, 200, 250])
        elif x_vals[-1] >= 100:
            ax.set_xticks([0, 25, 50, 75, 100])
        elif x_vals[-1] >= 64:
            ax.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 64])
        if yscale == "log":
            ax.set_yscale("log")
        ax.text(
            -0.11, 0.94,
            f"({chr(97 + ax_idx)})",
            transform=ax.transAxes,
            fontsize=12, fontweight="bold",
            va="bottom", ha="right",
        )
        ax_idx += 1

    if "qualtran" in chosen_methods:
        qualtran_qubits = results.get("qualtran_qubits")
        qualtran_ancillas = [(anc - qub) if anc is not None else None for (anc, qub) in zip(qualtran_qubits, x_vals)]

        if ncols == 2:
            inset_loc = "upper right"
            inset_pad = 0.3
        else:
            inset_loc = "lower right"
            inset_pad = 1.5
        ax_inset = inset_axes(axes[1], width="40%", height="30%", loc=inset_loc, borderpad=inset_pad)
        ax_inset.plot(x_vals, qualtran_ancillas, color='C3', linestyle="--", linewidth=1, marker="x", markersize="3.5")
        ax_inset.set_ylabel("Qualtran Anc.", fontsize='7')
        ax_inset.tick_params(labelsize=7)
        if ncols == 3:
            axes[1].set_xlabel("System Size, $N$")
        else:
            axes[0].set_xlabel("System Size, $N$")
            axes[1].set_xlabel("System Size, $N$")

    axes[0].legend(loc='lower right', fontsize='7', ncol=1, frameon=False)
    path = config.get("plot_output", "results_plot.pdf")
    plt.savefig(path, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    output_dict_name = "outputs/" + config.get("experiment", "results") + ".pkl"
    results = run_experiment(config, output_path=output_dict_name)
    #Plot the results if needed
    if config["plotting"]:
        plot_results(config, results)


if __name__ == "__main__":
    main()