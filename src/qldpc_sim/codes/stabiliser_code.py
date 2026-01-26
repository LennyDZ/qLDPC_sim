from typing import List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ..qec_objects import LogicalOperator, Stabiliser


class StabiliserCode(BaseModel):
    name: str
    id: UUID = Field(default_factory=uuid4)
    stabilisers: List[Stabiliser] = Field(default_factory=list)
    logical_operators: List[LogicalOperator] = Field(default_factory=list)
    num_qubits: int = Field(default=0)
    distance: int = Field(default=0)
