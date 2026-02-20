from typing import List, Tuple
from uuid import UUID, uuid4
from scipy.sparse import csr_matrix

from pydantic import BaseModel, Field, field_validator, model_validator
from ..data_structure import LogicalQubit, TannerGraph


class ErrorCorrectionCode(BaseModel):
    """Class representing an error correction code defined by its Tanner graph and logical operators."""

    class Config:
        arbitrary_types_allowed = True

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the error correction code.",
    )

    name: str = Field(
        default_factory=lambda: f"ECC_{uuid4().hex[:8]}",
        description="Human-readable name for the error correction code.",
    )

    n: int = Field(
        description="Number of physical qubits (variable nodes) in the code.",
        default=None,
    )
    k: int = Field(
        description="Number of logical qubits in the code.",
        default=None,
    )
    d: int = Field(
        description="Distance of the code.",
        default=None,
    )

    tanner_graph: TannerGraph
    logical_qubits: List[LogicalQubit]

    validate_algebraic_properties: bool = Field(
        default=True,
        description="Whether to validate algebraic properties of the code upon initialization.",
    )

    @field_validator("n", "k", "d")
    def validate_non_negative_parameters(cls, value):
        if value < 0:
            raise ValueError("Parameters n, k, and d must be non-negative.")
        return value

    @model_validator(mode="after")
    def validate_logical_operators_count(self) -> "ErrorCorrectionCode":
        if not self.validate_algebraic_properties:
            return self
        if len(self.logical_qubits) != self.k:
            raise ValueError("Number of logical operators must equal k.")
        return self

    @model_validator(mode="after")
    def validate_logical_operators_distance(self) -> "ErrorCorrectionCode":
        if not self.validate_algebraic_properties:
            return self
        for lq in self.logical_qubits:
            if lq.logical_x.operator.weight < self.d:
                raise ValueError(
                    f"Logical qubit {lq} has logical X operator with weight less than d."
                )
            if lq.logical_z.operator.weight < self.d:
                raise ValueError(
                    f"Logical qubit {lq} has logical Z operator with weight less than d."
                )
        return self

    @model_validator(mode="after")
    def validate_logical_operators_commutation(self) -> "ErrorCorrectionCode":
        if not self.validate_algebraic_properties:
            return self
        pcm, var, check = self.tanner_graph.get_parity_check_matrix()
        var_to_col_index = {v.id: i for i, v in enumerate(var)}
        logical_operators = []
        for lq in self.logical_qubits:
            for lop in [lq.logical_x, lq.logical_z]:
                op_simplectic_vector = [0] * (2 * self.n)
                for pauli, var_node in zip(lop.operator.string, lop.target_nodes):
                    match pauli:
                        case "X":
                            op_simplectic_vector[var_to_col_index[var_node.id]] = 1
                        case "Z":
                            op_simplectic_vector[
                                var_to_col_index[var_node.id] + self.n
                            ] = 1
                        case "Y":
                            op_simplectic_vector[var_to_col_index[var_node.id]] = 1
                            op_simplectic_vector[
                                var_to_col_index[var_node.id] + self.n
                            ] = 1
                        case _:
                            continue
                logical_operators.append(op_simplectic_vector)

        H_X = pcm[:, : self.n]
        H_Z = pcm[:, self.n :]
        for lop in logical_operators:
            # Check commutation relations
            lop_X = lop[: self.n]
            lop_Z = lop[self.n :]
            s = (H_X.dot(lop_Z) + H_Z.dot(lop_X)) % 2
            if not all(c == 0 for c in s):
                raise ValueError(
                    f"Logical operators {lop} do not commute with stabilizers."
                )

        # TODO check logical operators aren't product of stabilizers
        # TODO check logical operators X and Z of a given qubit anticommute

        return self

    @classmethod
    def from_pcm(
        cls,
        code_name: str,
        simplectic_pcm: csr_matrix,
        logical_qubits: List[Tuple[List[int], List[int]]],
    ) -> "ErrorCorrectionCode":
        """Construct a code from a given parity check matrix and a set of logical qubits.

        Parameters
        ----------
        code_name : str
            Name of the error correction code.
        simplectic_pcm : csr_matrix
            Parity check matrix in simplectic representation.
        logical_qubits : List[Tuple[List[int], List[int]]]
            List of tuples representing the logical X and Z operators for each logical qubit.

        Returns
        -------
        ErrorCorrectionCode
            Constructed error correction code instance.

        Raises
        ------
        ValueError
            If the parity check matrix has an odd number of columns.
        ValueError
            If the logical operators do not match the number of physical qubits.
        """

        n_row, n_col = simplectic_pcm.shape
        if n_col % 2 != 0:
            raise ValueError(
                "Parity check matrix in simplectic representation must have an even number of columns."
            )

        n = n_col // 2
        k = len(logical_qubits)
        d = min(
            min(sum(1 for p in lop[0] if p != 0) for lop in logical_qubits),
            min(sum(1 for p in lop[1] if p != 0) for lop in logical_qubits),
        )

        tg = TannerGraph.from_pcm(simplectic_pcm)
        logical_qubits_list = []
        for i, (logical_x, logical_z) in enumerate(logical_qubits):
            if len(logical_x) != n or len(logical_z) != n:
                raise ValueError(
                    "Logical operators must have the same length as the number of physical qubits."
                )
            lq = LogicalQubit(
                name=f"{code_name}_LQ_{i}",
                logical_x=logical_x,
                logical_z=logical_z,
            )
            logical_qubits_list.append(lq)

        return cls(
            n=n,
            k=k,
            d=d,
            name=code_name,
            tanner_graph=tg,
            logical_qubits=logical_qubits_list,
        )
