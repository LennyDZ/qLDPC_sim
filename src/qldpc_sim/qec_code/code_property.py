from pydantic import field_validator

from qldpc_sim.data_structure.tanner_graph import TannerGraph


class CSSMixin:
    @field_validator("tanner_graph")
    def validate_css_code(cls, tg: TannerGraph) -> TannerGraph:
        for check, edges in tg.index_by_check:
            if any(e.type not in {"X", "Z"} for e in edges):
                raise ValueError("Edges must be of type 'X' or 'Z' in a CSS code.")
            if not all(e.type == edges[0].type for e in edges):
                raise ValueError(
                    "All edges connected to a check node must be of the same type in a CSS code."
                )
            check.type = edges[0].type
        return tg


class LDPCMixin:
    @field_validator("tanner_graph")
    def validate_ldpc_code(cls, tg: TannerGraph) -> TannerGraph:
        # TODO: Is it possible to check for LDPC property or not?
        return tg
