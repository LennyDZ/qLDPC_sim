from typing import List
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class QuantumMemory(BaseModel):
    size: int = Field(
        default=100, description="The total number of qubits in the quantum memory."
    )

    slots: list[int] = []
    free_slots: list[int] = []
    allocation: dict[UUID, int] = {}

    @model_validator(mode="after")
    def initialize_slots(cls, memory: "QuantumMemory") -> "QuantumMemory":
        memory.slots = list(range(memory.size))
        memory.free_slots = list(range(memory.size))
        return memory

    @property
    def number_of_available_qubits(self) -> int:
        """Returns the number of available qubits in the quantum memory."""
        return len(self.free_slots)

    def allocate_qubit(self, qubit_id: UUID | List[UUID]) -> int:
        """Allocates a qubit in the quantum memory and returns its slot index."""

        if not isinstance(qubit_id, list):
            qubit_id = [qubit_id]

        if len(self.free_slots) < len(qubit_id):
            raise ValueError(
                "Not enough free slots in quantum memory to allocate qubits."
            )

        for qid in qubit_id:
            slot = self.free_slots.pop(0)
            self.allocation[qid] = slot
        return slot

    def free_qubit(self, qubit_id: UUID | List[UUID]) -> None:
        """Frees a qubit in the quantum memory given its slot index."""
        if not isinstance(qubit_id, list):
            qubit_id = [qubit_id]
        for qid in qubit_id:
            slot = self.allocation.pop(qid)
            self.free_slots.append(slot)
        return None
