from typing import Literal
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class BitResource(BaseModel):
    """Base class for quantum and classical bits.
    - name: Human-readable name of the bit
    - idx: Index of the qubits (useful to map to stim)
    - id: Unique identifier
    """

    name: str
    idx: str
    id: UUID = Field(default_factory=uuid4)


class ClassicalBit(BitResource):
    """Represents a classical bit."""

    pass


class QuantumBit(BitResource):
    """Represents a quantum bit (qubit)."""

    qubit_type: Literal["data", "ancilla"] = Field(
        description="Whether this qubit is a data or ancilla qubit"
    )
