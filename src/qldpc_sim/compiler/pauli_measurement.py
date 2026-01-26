from typing import List

from pydantic import BaseModel, Field

from ..qec_objects import LogicalOperator


class PauliMeasurement(BaseModel):
    """Pauli measurement description."""

    target: List[LogicalOperator] = Field(
        default_factory=list,
        description="List of logical operators to measure jointly",
    )
    distance: int = Field(
        default=0,
        description="Code distance for this measurement",
    )
    ancilla_required: int = Field(
        default=0,
        description="Number of ancilla qubits required for this measurement",
    )
