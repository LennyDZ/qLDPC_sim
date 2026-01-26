from typing import List, Optional, Tuple
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from ..qec_objects import Stabiliser, ExperimentWeight


class SyndromeExtractionRounds(BaseModel):
    """Syndrome extraction round description.
    - stabilisers: List of stabilisers to measure in this round, in order.
    """

    name: Optional[str] = Field(
        default="",
        description="Name of the syndrome extraction round",
    )
    id: UUID = Field(
        default_factory=lambda: uuid4(),
        description="Unique identifier for the syndrome extraction round",
    )
    stabilisers: List[Stabiliser] = Field(default_factory=list)
    number_of_rounds: int = Field(
        default=1,
        description="Number of times to repeat this syndrome extraction round",
    )

    def compile(self) -> Tuple[str, ExperimentWeight]:
        """Compile the syndrome extraction round into a stim circuit snippet
        and associated experiment weights.

        Returns:
            Tuple[str, ExperimentWeights]: The stim circuit snippet and experiment weights.
        """

        single_round_weight = ExperimentWeight()
        stim_instructions = []

        for s in self.stabilisers:
            stab_meas, smw = s.compiled_stim_circuit
            single_round_weight += smw
            stim_instructions.extend(stab_meas)

        single_round = stim_instructions

        total_weight = ExperimentWeight()
        final_stim_instructions = []
        for _ in range(self.number_of_rounds):
            final_stim_instructions.extend(single_round)
            total_weight += single_round_weight  # accumulate weights for each round

        return final_stim_instructions, total_weight
