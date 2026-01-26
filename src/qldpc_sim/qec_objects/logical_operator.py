from typing import List, Set, Tuple
from pydantic import BaseModel, Field

from .bit_types import QuantumBit
from .pauli_operator import PauliOperator


class LogicalOperator(PauliOperator):
    """Class representing a logical operator in a stabiliser code.
    Inherits from PauliOperator and adds frame tracking information.

    - frame_tracking: A tuple representing the frame tracking information
      for the logical operator. (e.g., (X_frame, Z_frame))
    """

    name: str = Field(description="Name of the logical operator")
    frame_tracking: Tuple[int, int] = Field(
        description="Frame tracking information for the logical operator"
    )

    qubits: List[QuantumBit] = Field(
        description="Qubits on which the logical operator acts"
    )
