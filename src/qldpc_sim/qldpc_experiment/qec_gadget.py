from abc import abstractmethod
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pyparsing import ABC

from qldpc_sim.data_structure.pauli import PauliChar, PauliEigenState
from qldpc_sim.qec_code.ec_code import ErrorCorrectionCode
from qldpc_sim.qldpc_experiment.context import Context
from ..data_structure import LogicalOperator, TannerGraph

from .compilers import (
    ApplyGates,
    Compiler,
    DestructiveMeasurementCompiler,
    LogicalPauliCompiler,
    QuantumMemory,
    StabilisersMeasurementCompiler,
)


class QECGadget(ABC, BaseModel):
    model_config = ConfigDict(frozen=True)
    context: Context
    tag: str = Field(default="", init=False)

    @abstractmethod
    def build_compiler_instructions(self, memory: QuantumMemory) -> List[Compiler]:
        pass


class CodeGadget(QECGadget):
    code: ErrorCorrectionCode


class LogicGadget(QECGadget):
    logical_target: List[LogicalOperator]


class LogicalPauli(LogicGadget):
    """Logical Pauli application gadget."""

    def build_compiler_instructions(self) -> List[Compiler]:
        compilers = []
        for lop in self.logical_target:
            # Build tanner with only var nodes corresponding to the logical qubit
            t = TannerGraph(
                variable_nodes=set(lop.target_nodes),
                check_nodes=set(),
                edges=set(),
            )
            compiler = LogicalPauliCompiler(data=t, operator=lop.logical_type)
            compilers.append(compiler)
        return compilers


class StabMeasurement(CodeGadget):
    round: int = Field(
        default=1,
        description="Number of round of stabiliser measurement to perform.",
    )

    def build_compiler_instructions(self) -> List[Compiler]:

        compilers = [
            StabilisersMeasurementCompiler(
                data=self.code.tanner_graph,
                round=self.round,
                tag=f"StabMeasurement_{self.tag}",
            )
        ]
        return compilers


class InitializeCode(CodeGadget):
    """Initialize a code in a given state.
    For code with multiple logical qubits, we only support initializing in the |0> state.
    """

    initial_state: PauliEigenState = Field(default=PauliEigenState.Z_plus)

    def build_compiler_instructions(self) -> List[Compiler]:
        if len(self.code.logical_qubits) > 1:
            if self.initial_state != PauliEigenState.Z_plus:
                raise ValueError(
                    "Init only supported for |0> state for codes with multiple logical qubits."
                )
        match self.initial_state:
            case PauliEigenState.Z_plus:
                return [
                    ApplyGates(
                        target_nodes=self.code.tanner_graph.variable_nodes
                        | self.code.tanner_graph.check_nodes,
                        gates=["RZ"],
                        tag=f"InitializeCode_{self.tag}",
                    )
                ]
            case PauliEigenState.Z_minus:
                return [
                    ApplyGates(
                        target_nodes=self.code.tanner_graph.variable_nodes
                        | self.code.tanner_graph.check_nodes,
                        gates=["RZ"],
                        tag=f"InitializeCode_{self.tag}",
                    ),
                    ApplyGates(
                        target_nodes=set(
                            self.code.logical_qubits[0].logical_x.target_nodes
                        ),
                        gates=["X"],
                        tag=f"InitializeCode_{self.tag}",
                    ),
                ]
            case PauliEigenState.X_plus:
                return [
                    ApplyGates(
                        target_nodes=self.code.tanner_graph.variable_nodes
                        | self.code.tanner_graph.check_nodes,
                        gates=["RX"],
                        tag=f"InitializeCode_{self.tag}",
                    )
                ]
            case PauliEigenState.X_minus:
                return [
                    ApplyGates(
                        target_nodes=self.code.tanner_graph.variable_nodes
                        | self.code.tanner_graph.check_nodes,
                        gates=["RX"],
                        tag=f"InitializeCode_{self.tag}",
                    ),
                    ApplyGates(
                        target_nodes=set(
                            self.code.logical_qubits[0].logical_z.target_nodes
                        ),
                        gates=["Z"],
                        tag=f"InitializeCode_{self.tag}",
                    ),
                ]
            case _:
                raise ValueError("Unsupported initialization")


class LM(LogicGadget):
    """Measure (project) a logical operator in the given basis. This doesn't reset nor free the qubits after measurement, so it can be used for mid-circuit measurements."""

    tag: str = Field(default="Logical Measurement", init=False)
    basis: PauliChar = Field(default=PauliChar.Z)

    def build_compiler_instructions(self) -> List[Compiler]:
        compilers = []
        lop = self.logical_target[0]
        # Build tanner with only var nodes corresponding to the logical qubit
        t = TannerGraph(
            variable_nodes=set(lop.target_nodes),
            check_nodes=set(),
            edges=set(),
        )
        compilers.append(
            DestructiveMeasurementCompiler(
                data=t,
                tag=self.tag,
                basis=self.basis,
                free_qubits=False,
                reset_qubits=False,
            )
        )
        return compilers


class Readout(CodeGadget):
    """Readout all the qubits of a code. This doesn't reset the qubits after measurement, so it can be used for mid-circuit measurements."""

    tag: str = Field(default="Readout")
    basis: PauliChar = Field(default=PauliChar.Z)

    def build_compiler_instructions(self) -> List[Compiler]:
        compilers = []
        compilers.append(
            DestructiveMeasurementCompiler(
                data=self.code.tanner_graph,
                tag=self.tag,
                basis=self.basis,
                reset_qubits=False,
            )
        )
        return compilers


class PauliMeasurement(LogicGadget):
    controlled_action: LogicalPauli = Field(default=None, init=False)
