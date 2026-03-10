from typing import Dict, List, Tuple

import numpy as np

from qldpc_sim.data_structure.pauli import PauliChar, PauliString
from qldpc_sim.data_structure.logical_operator import LogicalOperator
from qldpc_sim.data_structure.logical_qubit import LogicalQubit
from qldpc_sim.data_structure.tanner_graph import (
    CheckNode,
    TannerEdge,
    VariableNode,
    TannerGraph,
)

from .ec_code import ErrorCorrectionCode


class CSSCode(ErrorCorrectionCode):
    """Subclass of ErrorCorrectionCode for CSS (Calderbank-Shor-Steane) codes.

    CSS codes are defined by separate X and Z parity check matrices Hx and Hz
    satisfying Hx @ Hz.T = 0 (mod 2).
    """

    @classmethod
    def from_css_pcm(
        cls,
        code_name: str,
        hx: np.ndarray,
        hz: np.ndarray,
        logical_qubits: List[Tuple[List[int], List[int]]],
        var_coordinate: Dict[int, Tuple[int, ...]] = None,
        check_coordinate: Dict[int, Tuple[int, ...]] = None,
    ) -> "CSSCode":
        """Construct a CSS code from separate Hx and Hz parity check matrices.

        Parameters
        ----------
        code_name : str
            Name of the error correction code.
        hx : np.ndarray
            X-type parity check matrix (binary).
        hz : np.ndarray
            Z-type parity check matrix (binary).
        logical_qubits : List[Tuple[List[int], List[int]]]
            List of tuples representing the logical X and Z operators for each logical qubit.
            Each logical operator is a binary vector mapping to variable node indices.
        var_coordinate : Dict[int, Tuple[int, ...]], optional
            Mapping from variable node index to coordinates.
        check_coordinate : Dict[int, Tuple[int, ...]], optional
            Mapping from check node index to coordinates.

        Returns
        -------
        CSSCode
            Constructed CSS error correction code instance.
        """
        hx = np.asarray(hx, dtype=np.uint8)
        hz = np.asarray(hz, dtype=np.uint8)

        if hx.shape[1] != hz.shape[1]:
            raise ValueError(
                "Hx and Hz must have the same number of columns (physical qubits)."
            )

        n = hx.shape[1]
        n_x_checks = hx.shape[0]
        n_z_checks = hz.shape[0]
        k = len(logical_qubits)

        d = min(
            min(sum(1 for p in lop[0] if p != 0) for lop in logical_qubits),
            min(sum(1 for p in lop[1] if p != 0) for lop in logical_qubits),
        )

        if var_coordinate is not None and len(var_coordinate) != n:
            raise ValueError(
                "Variable coordinate dictionary length must match the number of physical qubits."
            )
        if var_coordinate is not None:
            var_nodes = [
                VariableNode(tag=f"v_{i}_{code_name}", coordinates=var_coordinate[i])
                for i in range(n)
            ]
        else:
            var_nodes = [VariableNode(tag=f"v_{i}_{code_name}") for i in range(n)]

        # Create logical qubits
        logical_qubits_list = []
        for i, lq in enumerate(logical_qubits):
            lx_target = []
            lz_target = []
            for j in range(n):
                if lq[0][j] == 1:
                    lx_target.append(var_nodes[j])
                if lq[1][j] == 1:
                    lz_target.append(var_nodes[j])
            l = LogicalQubit(
                logical_x=LogicalOperator(
                    operator=PauliString(string=tuple([PauliChar.X] * len(lx_target))),
                    target_nodes=tuple(lx_target),
                    logical_type=PauliChar.X,
                ),
                logical_z=LogicalOperator(
                    operator=PauliString(string=tuple([PauliChar.Z] * len(lz_target))),
                    target_nodes=tuple(lz_target),
                    logical_type=PauliChar.Z,
                ),
                name=f"{code_name}_lq_{i}",
            )
            logical_qubits_list.append(l)

        # Create check nodes with appropriate types
        check_nodes = []
        check_id_to_node = {}

        # X-type checks from Hx
        for i in range(n_x_checks):
            check_coord = check_coordinate[i] if check_coordinate is not None else None
            check_node = CheckNode(
                tag=f"c_x_{i}_{code_name}",
                coordinates=check_coord,
                check_type=PauliChar.X,
            )
            check_nodes.append(check_node)
            check_id_to_node[i] = check_node

        # Z-type checks from Hz
        for i in range(n_z_checks):
            check_coord = (
                check_coordinate[n_x_checks + i]
                if check_coordinate is not None
                else None
            )
            check_node = CheckNode(
                tag=f"c_z_{i}_{code_name}",
                coordinates=check_coord,
                check_type=PauliChar.Z,
            )
            check_nodes.append(check_node)
            check_id_to_node[n_x_checks + i] = check_node

        # Build edges from Hx (X-type checks)
        edges = []
        for i in range(n_x_checks):
            for j in range(n):
                if hx[i, j] == 1:
                    edges.append(
                        TannerEdge(
                            check_node=check_id_to_node[i],
                            variable_node=var_nodes[j],
                            pauli_checked=PauliChar.X,
                        )
                    )

        # Build edges from Hz (Z-type checks)
        for i in range(n_z_checks):
            for j in range(n):
                if hz[i, j] == 1:
                    edges.append(
                        TannerEdge(
                            check_node=check_id_to_node[n_x_checks + i],
                            variable_node=var_nodes[j],
                            pauli_checked=PauliChar.Z,
                        )
                    )

        tanner_graph = TannerGraph(
            variable_nodes=set(var_nodes),
            check_nodes=set(check_nodes),
            edges=set(edges),
        )

        return cls(
            n=n,
            k=k,
            d=d,
            name=code_name,
            tanner_graph=tanner_graph,
            logical_qubits=logical_qubits_list,
            validate_algebraic_properties=False,
        )
