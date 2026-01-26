from functools import cached_property
from typing import List, Tuple
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from uuid import UUID

from .pauli_operator import PauliOperator
from .bit_types import QuantumBit
from .experiment_weight import ExperimentWeight


class Stabiliser(PauliOperator):
    """
    Represents a stabiliser object, extending PauliOperator.

    The length of the Pauli operator string must match the number of data qubits.
    We assume the orders of data qubits and Pauli operators correspond directly.

    The order of the control gates in the syndrome extraction circuit can be specified,
    by default it is the order of data_qubits.
    E.g, if pauli string is "XZZX" and data_qubits are [q0, q1, q2, q3], then the order [0, 1, 2, 3] means the controlled gates are applied in the order :
    q0 -> ancilla, q1 -> ancilla, q2 -> ancilla, q3 -> ancilla.
    If the order is [2, 0, 3, 1], then the controlled gates are applied in the order:
    q2 -> ancilla, q0 -> ancilla, q3 -> ancilla, q1 -> ancilla.

    Attributes:
    - name: Name of the stabiliser
    - data_qubits: List of data qubits involved in this stabiliser
    - ancilla_qubit: Ancilla qubit that stores the measurement result
    - extraction_order: Optional order of data qubits for syndrome extraction, given as
      a list of indices. If empty, the order of data_qubits is used.
    """

    name: str = Field(description="Name of the stabiliser")
    data_qubits: List[QuantumBit] = Field(
        description="List of data qubits involved in this stabiliser"
    )
    ancilla_qubit: QuantumBit = Field(
        description="Ancilla qubit that stores the measurement result"
    )

    extraction_order: List[int] = Field(
        default_factory=list,
        description=(
            "Optional order of data qubits for syndrome extraction; "
            "if empty, the order of data_qubits is used (eg. [0, 1, 2,...] is default)"
        ),
    )

    @field_validator("pauli_string", mode="before")
    @classmethod
    def validate_pauli_string(cls, v: str) -> str:
        """Ensure Pauli string is valid. Ensure that it only contains X, Y, Z. We forbid
        I in order to map qubits to operators in a straightforward way."""
        if v is None:
            return v
        if not all(char in "XYZ" for char in v):
            raise ValueError(
                "Pauli string must contain only 'X', 'Y', or 'Z' characters"
            )
        return v

    @field_validator("data_qubits")
    @classmethod
    def validate_data_qubits(cls, v: List[QuantumBit]) -> List[QuantumBit]:
        """Ensure all qubits in data_qubits are actually data qubits."""
        for qubit in v:
            if qubit.qubit_type != "data":
                raise ValueError(
                    f"Qubit '{qubit.name}' must be a data qubit, not {qubit.qubit_type}"
                )
        return v

    @field_validator("ancilla_qubit")
    @classmethod
    def validate_ancilla_qubit(cls, v: QuantumBit) -> QuantumBit:
        """Ensure the ancilla qubit is actually an ancilla qubit."""
        if v.qubit_type != "ancilla":
            raise ValueError(
                f"Qubit '{v.name}' must be an ancilla qubit, not {v.qubit_type}"
            )
        return v

    @model_validator(mode="after")
    def set_default_extraction_order_and_validate(self) -> "Stabiliser":
        """Set default extraction order if empty and validate pauli length."""
        # Set default extraction order if empty
        if not self.extraction_order:
            self.extraction_order = list(range(len(self.data_qubits)))

        # Ensure Pauli operator length matches number of data qubits
        pauli_str = self.get_string
        if len(pauli_str) != len(self.data_qubits):
            raise ValueError(
                f"Pauli operator length ({len(pauli_str)}) must match "
                f"number of data qubits ({len(self.data_qubits)})"
            )
        return self

    @cached_property
    def compiled_stim_circuit(self) -> Tuple[List[str], ExperimentWeight]:
        """Compile the stabiliser measurement into stim instructions and experiment weight.

        Returns:
            Tuple[List[str], ExperimentWeight]: The stim instructions and experiment weight.
        """
        stim_instructions = []

        # Small optimization for X-only stabilisers where we apply H gates to ancilla only
        # and swap CX control/target.
        if self.is_X_only:
            stim_instructions.append(f"H {self.ancilla_qubit.idx}")
            for index in self.extraction_order:
                stim_instructions.append(
                    f"CX {self.ancilla_qubit.idx} {self.data_qubits[index].idx}"
                )
            stim_instructions.append(f"H {self.ancilla_qubit.idx}")
            stim_instructions.append(f"M {self.ancilla_qubit.idx}")
            return (
                stim_instructions,
                ExperimentWeight(
                    cx_count=len(self.data_qubits), h_count=2, meas_count=1
                ),
            )
        else:
            for index in self.extraction_order:
                if self.pauli_string[index] == "Y":
                    stim_instructions.extend(
                        [
                            f"H {self.data_qubits[index].idx} ",
                            f"CY {self.data_qubits[index].idx} {self.ancilla_qubit.idx} ",
                            f"H {self.data_qubits[index].idx} ",
                        ]
                    )
                elif self.pauli_string[index] == "X":
                    stim_instructions.extend(
                        [
                            f"H {self.data_qubits[index].idx} ",
                            f"CX {self.data_qubits[index].idx} {self.ancilla_qubit.idx} ",
                            f"H {self.data_qubits[index].idx} ",
                        ]
                    )
                elif self.pauli_string[index] == "Z":
                    stim_instructions.append(
                        f"CX {self.data_qubits[index].idx} {self.ancilla_qubit.idx}"
                    )
            stim_instructions.append(f"M {self.ancilla_qubit.idx}")

            experiment_weight = ExperimentWeight(
                cx_count=len(self.data_qubits),
                h_count=len([op for op in self.pauli_string if op == "X"]),
                meas_count=1,
            )

        return stim_instructions, experiment_weight
