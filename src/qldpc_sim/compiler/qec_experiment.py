from itertools import chain
from typing import List, Set, Tuple, Union
from pydantic import BaseModel, Field

from .pauli_measurement import PauliMeasurement
from .syndrome_extraction_rounds import (
    SyndromeExtractionRounds,
)
from ..codes import StabiliserCode
from ..qec_objects import ExperimentWeight


class QECExperiment(BaseModel):
    """Quantum error correction experiment description. This is a high-level
    description. An experiment consists of 3 steps:
    1. Allocate qubit resources and initialize logical qubits through codes
    2. Perform a series of logical pauli measurements and syndrome extraction rounds
    3. Decode the measurement results to obtain final logical measurement outcomes

    Attributes:
    - name: Name of the experiment
    - qubit_resource: Amount of qubit resource allocated for the experiment
    - distance: Code distance for the experiment
    - rounds: List of syndrome extraction rounds in the experiment
    """

    name: str = Field(description="Name of the experiment")

    qubit_resource: int = Field(
        default=0,
        description="Amount of qubit resource allocated for the experiment",
    )

    distance: int = Field(
        default=0,
        description="Code distance for the experiment",
    )

    actions: List[Union[SyndromeExtractionRounds, PauliMeasurement]] = Field(
        default_factory=list,
        description="List of actions in the experiment",
    )

    def compile_to_stim(self) -> Tuple[str, ExperimentWeight]:
        """Compile the QEC experiment to a Stim circuit."""
        stim_instruction = []
        experiment_weight = ExperimentWeight()

        for action in self.actions:
            action_stim, action_weight = action.compile()
            stim_instruction.extend(action_stim)
            experiment_weight += action_weight

        return "\n".join(stim_instruction), experiment_weight

    class QECExpBuilder:
        """Builder class for QECExperiment to facilitate step-by-step construction."""

        def __init__(self, name: str, qubit_resource: int, distance: int) -> None:
            self.name = name
            self.codes = []
            self.logicals = []
            self.actions = []
            self.distance = distance
            self.qubit_resource = qubit_resource
            self.available_qubits_count = len(qubit_resource)

        def add_code(self, code) -> "QECExperiment.QECExpBuilder":
            self.codes.append(code)
            self.available_qubits_count -= code.num_qubits
            self.logicals.extend(code.logical_operators)
            return self

        def add_logical_pauli_measurement(
            self, action: PauliMeasurement
        ) -> "QECExperiment.QECExpBuilder":
            if any(target not in self.logicals for target in action.target):
                raise ValueError(
                    f"Targeted logical operators in action {action.name} are not part of the experiment"
                )
            if action.ancilla_required() > self.available_qubits_count:
                raise ValueError(
                    f"Not enough ancilla qubits available for action {action.name}"
                )
            self.actions.append(action)
            resulting_logicals = action.resulting_logicals()
            self.logicals.difference_update(resulting_logicals)
            self.logicals.update(resulting_logicals)

            return self

        def add_code_syndrome_extraction_rounds(
            self, code: Union[StabiliserCode, List[StabiliserCode]], rounds: int
        ) -> "QECExperiment.QECExpBuilder":
            if isinstance(code, list):
                for c in code:
                    if c not in self.codes:
                        raise ValueError(
                            f"Code {c.name} not part of the experiment, cannot add syndrome extraction rounds"
                        )
            else:
                if code not in self.codes:
                    raise ValueError(
                        f"Code {code.name} not part of the experiment, cannot add syndrome extraction rounds"
                    )
                code = [code]

            stabilisers_to_measure = list(
                chain.from_iterable(c.stabilisers for c in code)
            )
            self.actions.append(
                SyndromeExtractionRounds(
                    f"extract syndrome for {code.name}", stabilisers_to_measure, rounds
                )
            )
            return self

        def add_global_syndrome_extraction_rounds(
            self, rounds: int
        ) -> "QECExperiment.QECExpBuilder":
            stabilisers_to_measure = list(
                chain.from_iterable(c.stabilisers for c in self.codes)
            )
            self.actions.append(
                SyndromeExtractionRounds(
                    name="extract global syndrome",
                    stabilisers=stabilisers_to_measure,
                    number_of_rounds=rounds,
                )
            )
            return self

        def build(self) -> "QECExperiment":
            return QECExperiment(
                name=self.name,
                qubit_resource=5,
                distance=self.distance,
                actions=self.actions,
            )
