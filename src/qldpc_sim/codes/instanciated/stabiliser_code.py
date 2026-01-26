from dataclasses import Field
from typing import List
from uuid import UUID, uuid4

from pydantic import BaseModel

from ..qec_objects import LogicalOperator, Stabiliser


class StabiliserCode(BaseModel):
    name: str
    id: UUID = Field(default_factory=uuid4)
    stabilisers: List[Stabiliser] = Field(default_factory=list)
    logical_operators: List[LogicalOperator] = Field(default_factory=list)
