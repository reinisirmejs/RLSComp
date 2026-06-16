from .mps_utils import build_mps_from_DFA, build_mps_from_regex, build_mps_from_bitstrings, is_mps, is_bitstring_list, validate_bitstring_list
from .circuit_utils import MPS_to_circuit_SeqRLSP, MPS_to_circuit_SeqIsoRLSP, Tree_to_circuit
from .regex_utils import is_dfa, is_regex, get_finite_complement

def build_SeqRLSP_circuit(input_obj, system_size=None, *, complement=False, use_isometries=False):
    """
    Build a SeqRLSP circuit from various input types.

    Positional usage:
      - (regex, system_size)

    Supported inputs:
      - regex (str)
      - list[str] of bitstrings
      - DFA
      - MPS

    system_size is required for regex and cyclic DFA inputs.
    """

    # --------------------------------------------------
    # Regex input (string)
    # --------------------------------------------------
    if is_regex(input_obj):
        if system_size is None:
            raise ValueError(
                "system_size must be provided when input is a regex string"
            )

        MPS_LIST, _ = build_mps_from_regex(
            input_obj, system_size, complement=complement
        )

    # --------------------------------------------------
    # Bitstring list
    # --------------------------------------------------
    elif is_bitstring_list(input_obj):
        if system_size is not None:
            raise ValueError(
                "system_size must not be provided for bitstring input, it is inferred from the length of the bitstrings"
            )

        validate_bitstring_list(input_obj)

        MPS_LIST = build_mps_from_bitstrings(
            input_obj, complement=complement
        )

    # --------------------------------------------------
    # DFA
    # --------------------------------------------------
    elif is_dfa(input_obj):
        if input_obj.is_acyclic():
            if system_size is not None:
                raise ValueError(
                    "system_size must not be provided for Acyclic DFA input"
                )
            if complement:
                input_obj = get_finite_complement(input_obj)
        else:
            if system_size is None:
                raise ValueError(
                    "system_size must be provided for a DFA input with cycles"
                )
            if complement:
                input_obj = input_obj.get_complement().minimize()

        MPS_LIST = build_mps_from_DFA(input_obj, system_size=system_size)

    # --------------------------------------------------
    # MPS
    # --------------------------------------------------
    elif is_mps(input_obj):
        if system_size is not None:
            raise ValueError(
                "system_size must not be provided for MPS input"
            )
        if complement:
            raise ValueError(
                "complement is not supported for MPS input"
            )

        MPS_LIST = input_obj

    else:
        raise TypeError(
            f"Unsupported input type: {type(input_obj)}"
        )

    #print("Original MPS_LIST: ", MPS_LIST)
    if use_isometries:
        return MPS_to_circuit_SeqIsoRLSP(MPS_LIST)
    else:
        return MPS_to_circuit_SeqRLSP(MPS_LIST)
    

def build_TreeRLSP_circuit(input_obj, system_size=None, *, complement=False):
    """
    Build a TreeRLSP circuit from various input types.

    Positional usage:
      - (regex, system_size)

    Supported inputs:
      - regex (str)
      - list[str] of bitstrings
      - DFA
      - MPS

    system_size is required for regex and cyclic DFA inputs.
    """

    # --------------------------------------------------
    # Regex input (string)
    # --------------------------------------------------
    if is_regex(input_obj):
        if system_size is None:
            raise ValueError(
                "system_size must be provided when input is a regex string"
            )

        MPS_LIST, _ = build_mps_from_regex(
            input_obj, system_size, complement=complement
        )

    # --------------------------------------------------
    # Bitstring list
    # --------------------------------------------------
    elif is_bitstring_list(input_obj):
        if system_size is not None:
            raise ValueError(
                "system_size must not be provided for bitstring input, it is inferred from the length of the bitstrings"
            )

        validate_bitstring_list(input_obj)

        MPS_LIST = build_mps_from_bitstrings(
            input_obj, complement=complement
        )

    # --------------------------------------------------
    # DFA
    # --------------------------------------------------
    elif is_dfa(input_obj):
        if input_obj.is_acyclic():
            if system_size is not None:
                raise ValueError(
                    "system_size must not be provided for Acyclic DFA input"
                )
            if complement:
                input_obj = get_finite_complement(input_obj)
        else:
            if system_size is None:
                raise ValueError(
                    "system_size must be provided for a DFA input with cycles"
                )
            if complement:
                input_obj = input_obj.get_complement().minimize()

        MPS_LIST = build_mps_from_DFA(input_obj, system_size=system_size)

    # --------------------------------------------------
    # MPS
    # --------------------------------------------------
    elif is_mps(input_obj):
        if system_size is not None:
            raise ValueError(
                "system_size must not be provided for MPS input"
            )
        if complement:
            raise ValueError(
                "complement is not supported for MPS input"
            )

        MPS_LIST = input_obj

    else:
        raise TypeError(
            f"Unsupported input type: {type(input_obj)}"
        )
    return Tree_to_circuit(MPS_LIST)