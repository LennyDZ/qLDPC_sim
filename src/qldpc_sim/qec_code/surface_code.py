from pydantic import Field, model_validator
from scipy.sparse import lil_matrix

from qldpc_sim.data_structure.tanner_graph import (
    CheckNode,
    TannerEdge,
    TannerGraph,
    VariableNode,
    VariableNode,
)
from .ec_code import ErrorCorrectionCode
from .code_property import CSSMixin, LDPCMixin


def surface_code_pcm(d: int) -> None:
    """Generates the parity-check matrix for a surface code of distance d."""
    # Implementation to generate the PCM for surface code
    pass


class SurfaceCode(CSSMixin, LDPCMixin, ErrorCorrectionCode):
    tanner_graph: TannerGraph = Field(init=False)

    @model_validator(mode="after")
    def validate_code_params(self) -> "SurfaceCode":
        if not self.n:
            self.n = self.d**2 + (self.d - 1) ** 2
        if self.n != (self.d**2 + (self.d - 1) ** 2):
            raise ValueError("For surface codes, n must be equal to d squared.")
        # if self.k != 1:
        #     raise ValueError("For surface codes, k must be equal to 1.")
        return self

    @model_validator(mode="after")
    def build_tanner_graph(self) -> "SurfaceCode":
        # Build the Tanner graph from the parity-check matrix
        return self
