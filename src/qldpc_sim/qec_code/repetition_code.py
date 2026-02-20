from typing import List
import numpy as np
from pydantic import Field, model_validator
from scipy.sparse import csr_matrix

from qldpc_sim.data_structure.logical_operator import LogicalOperator
from qldpc_sim.data_structure.logical_qubit import LogicalQubit
from qldpc_sim.data_structure.pauli import PauliChar, PauliString
from qldpc_sim.data_structure.tanner_graph import TannerGraph
from qldpc_sim.qec_code.ec_code import ErrorCorrectionCode


class RepetitionCode(ErrorCorrectionCode):
    name: str = "RepetitionCode"
    n: int = Field(
        default=1, init=False, description="Number of physical qubits in the code."
    )
    k: int = Field(
        default=1, init=False, description="Number of logical qubits in the code."
    )
    d: int = 3
    stabiliser_type: PauliChar = Field(
        default=PauliChar.Z, description="Type of stabilisers in the repetition code."
    )
    validate_algebraic_properties: bool = False
    tanner_graph: TannerGraph | None = None

    logical_qubits: List[LogicalQubit] = []

    @model_validator(mode="after")
    def compute_tanner_graph(self) -> "RepetitionCode":
        self.d = self.n
        checks = np.zeros((self.d - 1, self.d), dtype=int)
        for i in range(self.n - 1):
            checks[i, i] = 1
            checks[i, i + 1] = 1

        if self.stabiliser_type == PauliChar.X:
            self.tanner_graph = TannerGraph.from_pcm(csr_matrix(checks), None)
        else:
            self.tanner_graph = TannerGraph.from_pcm(None, csr_matrix(checks))
        return self

    @model_validator(mode="after")
    def compute_logical_qubits(self) -> "RepetitionCode":
        if self.stabiliser_type == PauliChar.Z:
            logical_z = LogicalOperator(
                operator=PauliString(string=(PauliChar.Z,)),
                target_nodes=(tuple(self.tanner_graph.variable_nodes)[0],),
                logical_type=PauliChar.Z,
            )
            logical_x = LogicalOperator(
                operator=PauliString(string=(PauliChar.X,) * self.n),
                target_nodes=tuple(self.tanner_graph.variable_nodes),
                logical_type=PauliChar.X,
            )
        else:
            logical_z = LogicalOperator(
                operator=PauliString(string=(PauliChar.Z,) * self.n),
                target_nodes=tuple(self.tanner_graph.variable_nodes),
                logical_type=PauliChar.Z,
            )
            logical_x = LogicalOperator(
                operator=PauliString(string=(PauliChar.X,)),
                target_nodes=(tuple(self.tanner_graph.variable_nodes)[0],),
                logical_type=PauliChar.X,
            )
        self.logical_qubits = [
            LogicalQubit(
                name="lq0",
                logical_x=logical_x,
                logical_z=logical_z,
            )
        ]
        return self
