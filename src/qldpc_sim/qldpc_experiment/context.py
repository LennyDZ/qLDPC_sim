from typing import Dict, List
from uuid import UUID

from pydantic import BaseModel

from qldpc_sim.data_structure.logical_operator import LogicalOperator
from qldpc_sim.qldpc_experiment.quantum_memory import QuantumMemory
from qldpc_sim.qldpc_experiment.record import MeasurementRecord

from ..data_structure.logical_qubit import LogicalQubit
from ..qec_code import ErrorCorrectionCode


class Context(BaseModel):
    """Context model for qLDPC simulation.

    Attributes:
        logical_qubits (List[LogicalQubit]): List of logical qubits in the code.
        codes (List[ErrorCorrectionCode]): List of error correction codes used in the simulation.
        initial_assignement (Dict[UUID, ErrorCorrectionCode]): Initial assignment of logical qubits to error correction codes.
        record (MeasurementRecord): Record of measurements performed during the simulation.
        memory (QuantumMemory): Quantum memory used in the simulation.
    """

    logical_qubits: List[LogicalQubit]
    codes: List[ErrorCorrectionCode]
    initial_assignement: Dict[LogicalOperator, ErrorCorrectionCode]
    record: MeasurementRecord = MeasurementRecord()
    memory: QuantumMemory = QuantumMemory()
