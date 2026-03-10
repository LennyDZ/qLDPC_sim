from functools import cached_property
from typing import Dict, List, Set, Tuple
from uuid import UUID, uuid4
import numpy as np
from scipy.sparse import csr_matrix

from pydantic.dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .pauli import PauliChar


class TannerNode(BaseModel):
    """Node in a Tanner graph.

    Attributes:
        id (UUID): Unique identifier of the node.
        tag (str): human readable tag (mostly used for debugging).
    """

    model_config = ConfigDict(frozen=True)
    id: UUID = Field(
        description="Unique identifier of the node",
        default_factory=uuid4,
    )
    tag: str = Field(
        description="Human readable tag (mostly used for debugging)",
        default="",
    )
    coordinates: Tuple[int, ...] = Field(
        description="Coordinates of the node in some space (optional, for visualization or geometric codes)",
        default=(),
    )


class VariableNode(TannerNode):
    """Variable node in a Tanner graph."""

    pass


class CheckNode(TannerNode):
    """Check node in a Tanner graph."""

    check_type: PauliChar | None = Field(
        description="Type of the check node (X, Z, Y) or None when unspecified.",
        default=None,
    )

    @property
    def pauli_type(self) -> PauliChar | None:
        return self.check_type


class TannerEdge(BaseModel):
    """
    Edge in a Tanner graph, connecting a variable node to a check node.

    Attributes:
        variable_node (VariableNode): The variable node connected by the edge.
        check_node (CheckNode): The check node connected by the edge.
        pauli_checked (PauliChar): The type of Pauli operator checked by this edge (X, Z, or Y).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    variable_node: VariableNode
    check_node: CheckNode
    pauli_checked: PauliChar


class TannerGraph(BaseModel):
    """Tanner graph representation of a quantum error-correcting code.

    Attributes:
        variable_nodes (Set[VariableNode]): Set of variable nodes in the Tanner graph.
        check_nodes (Set[CheckNode]): Set of check nodes in the Tanner graph.
        edges (Set[TannerEdge]): Set of edges connecting variable nodes to check nodes.

    Properties:
        number_of_nodes: Returns the total number of nodes in the Tanner graph.
        index_by_check: Returns a mapping from check nodes to their corresponding edges.
        index_by_variable: Returns a mapping from variable nodes to their corresponding edges.
    """

    variable_nodes: Set[VariableNode] = Field(
        default_factory=set, description="Set of variable nodes in the Tanner graph."
    )
    check_nodes: Set[CheckNode] = Field(
        default_factory=set, description="Set of check nodes in the Tanner graph."
    )
    edges: Set[TannerEdge] = Field(
        default_factory=set,
        description="Set of edges connecting variable nodes to check nodes.",
    )

    @field_validator("edges")
    def validate_edges(cls, edges, info):
        """Validate that edges connect existing variable and check nodes in the Tanner graph."""
        values = info.data or {}
        var_node_ids = values.get("variable_nodes", [])
        check_node_ids = values.get("check_nodes", [])

        for edge in edges:
            if edge.variable_node not in var_node_ids:
                raise ValueError(
                    f"Edge variable node {edge.variable_node.id} not in variable nodes."
                )
            if edge.check_node not in check_node_ids:
                raise ValueError(
                    f"Edge check node {edge.check_node.id} not in check nodes."
                )
        return edges

    @model_validator(mode="after")
    def validate_check_types(cls, graph: "TannerGraph") -> "TannerGraph":
        """Validate that any declared check type matches the Pauli type of attached edges."""
        for edge in graph.edges:
            declared_pauli = edge.check_node.pauli_type
            if declared_pauli is not None and edge.pauli_checked != declared_pauli:
                raise ValueError(
                    f"Edge with Pauli type {edge.pauli_checked} connected to check node {edge.check_node.id} with declared type {edge.check_node.check_type}."
                )
        return graph

    @property
    def number_of_nodes(self) -> int:
        """Returns the total number of nodes in the Tanner graph."""
        return len(self.variable_nodes) + len(self.check_nodes)

    @cached_property
    def index_by_check(self) -> Dict[CheckNode, Set[TannerEdge]]:
        """Returns a mapping from check nodes to the edges connected to them."""
        index = {}
        for edge in self.edges:
            if edge.check_node not in index:
                index[edge.check_node] = set()
            index[edge.check_node].add(edge)
        return index

    @cached_property
    def index_by_variable(self) -> Dict[VariableNode, Set[TannerEdge]]:
        """Returns a mapping from variable nodes to the edges connected to them."""
        index = {}
        for edge in self.edges:
            if edge.variable_node not in index:
                index[edge.variable_node] = set()
            index[edge.variable_node].add(edge)
        return index

    def get_neighbourhood(self, node: TannerNode) -> Set[TannerNode]:
        """Returns the set of neighboring nodes connected to the given node."""
        neighbors = set()
        if node in self.variable_nodes:
            for edge in self.index_by_variable.get(node, set()):
                neighbors.add(edge.check_node)
        elif node in self.check_nodes:
            for edge in self.index_by_check.get(node, set()):
                neighbors.add(edge.variable_node)
        else:
            raise ValueError("Node is not in the Tanner graph.")
        return neighbors

    def degree(self, node: TannerNode) -> int:
        """Returns the degree of the given node in the Tanner graph."""
        if node in self.variable_nodes:
            return len(self.index_by_variable.get(node, set()))
        elif node in self.check_nodes:
            return len(self.index_by_check.get(node, set()))
        else:
            raise ValueError("Node is not in the Tanner graph.")

    def get_support(self, logical: Set[VariableNode], type: PauliChar) -> "TannerGraph":
        """Returns the Tanner graph corresponding to the support of the given logical operator.
        O(E) on first call, O(1) on subsequent calls."""
        var_nodes = set()
        check_nodes = set()
        edges = set()
        for dq in logical:
            if dq not in self.variable_nodes:
                raise ValueError(
                    f"Logical qubit {dq} not in Tanner graph variable nodes."
                )
            var_nodes.add(dq)
            for edge in self.index_by_variable.get(dq, set()):
                if edge.pauli_checked == type.dual():
                    check_nodes.add(edge.check_node)
                    edges.add(edge)

        return TannerGraph(
            variable_nodes=var_nodes,
            check_nodes=check_nodes,
            edges=edges,
        )

    @cached_property
    def parity_check_matrix(
        self,
    ) -> Tuple[csr_matrix, List[VariableNode], List[CheckNode]]:
        """Returns the parity-check matrix representation of the Tanner graph.
        O(#Check) (+ O(E) if check index not cached).

         Returns:
            Tuple[csr_matrix, List[VariableNode], List[CheckNode]]: The parity-check matrix in CSR format,
            list of variable nodes, and list of check nodes. The list are in the same order as the matrix rows and columns.
        """
        var_node_list = list(self.variable_nodes)
        check_node_list = list(self.check_nodes)
        # Map variable node to index
        var_node_index = {node.id: idx for idx, node in enumerate(var_node_list)}
        n_var = len(var_node_list)

        check_adj = self.index_by_check

        data = []
        row_indices = []
        col_indices = []
        for row_idx, check in enumerate(check_node_list):
            if check not in check_adj:
                continue
            for edge in check_adj[check]:
                row_indices.append(row_idx)
                data.append(1)
                if (
                    edge.pauli_checked == PauliChar.X
                    or edge.pauli_checked == PauliChar.Y
                ):
                    col_indices.append(var_node_index[edge.variable_node.id])
                if (
                    edge.pauli_checked == PauliChar.Z
                    or edge.pauli_checked == PauliChar.Y
                ):
                    col_indices.append(var_node_index[edge.variable_node.id] + n_var)

        return (
            csr_matrix(
                (data, (row_indices, col_indices)),
                shape=(len(check_node_list), 2 * n_var),
            ),
            var_node_list,
            check_node_list,
        )

    @staticmethod
    def from_pcm(Hx: csr_matrix, Hz: csr_matrix, code_name: str = "") -> "TannerGraph":
        """Constructs a Tanner graph from a given CSS parity-check matrix pairs.
        This method preserve the order of the columns while tagging the variable nodes.
        Variable nodes are tagged as "v_{column_index}_{code_name}" and check nodes are tagged as "c_{row_index}_{code_name}" for X checks and "c_{row_index + num_X_checks}_{code_name}" for Z checks.

        Args:
            Hx (csr_matrix): The parity-check matrix for X stabilizers in CSR format.
            Hz (csr_matrix): The parity-check matrix for Z stabilizers in CSR format.
        Returns:
            TannerGraph: The constructed Tanner graph.
        """
        x_check_count, num_vars = 0, 0
        z_check_count, num_vars_z = 0, 0
        if Hx != None:
            x_check_count, num_vars = Hx.shape
        if Hz != None:
            z_check_count, num_vars_z = Hz.shape
            num_vars_z = Hz.shape[1]
            if num_vars != num_vars_z and Hx and Hz:
                raise ValueError("Hx and Hz must have the same number of columns.")
            num_vars = num_vars_z
        variable_nodes = [
            VariableNode(tag=f"v_{i}_{code_name}") for i in range(num_vars)
        ]

        check_nodes = [
            CheckNode(tag=f"c_{j}_{code_name}", check_type=PauliChar.X)
            for j in range(x_check_count)
        ]

        check_nodes.extend(
            [
                CheckNode(
                    tag=f"c_{j +x_check_count}_{code_name}",
                    check_type=PauliChar.Z,
                )
                for j in range(z_check_count)
            ]
        )
        edges = set()

        for i in range(x_check_count):
            start = Hx.indptr[i]
            end = Hx.indptr[i + 1]
            for idx in range(start, end):
                j = Hx.indices[idx]
                edges.add(
                    TannerEdge(
                        variable_node=variable_nodes[j],
                        check_node=check_nodes[i],
                        pauli_checked=PauliChar.X,
                    )
                )

        for i in range(z_check_count):
            start = Hz.indptr[i]
            end = Hz.indptr[i + 1]
            for idx in range(start, end):
                j = Hz.indices[idx]
                edges.add(
                    TannerEdge(
                        variable_node=variable_nodes[j],
                        check_node=check_nodes[x_check_count + i],
                        pauli_checked=PauliChar.Z,
                    )
                )

        return TannerGraph(
            variable_nodes=set(variable_nodes),
            check_nodes=set(check_nodes),
            edges=edges,
        )

    def visualize(self, periodic: bool = False):
        """Plot the TannerGraph, with variable nodes as circles and check nodes as squares. Edges are colored according to the Pauli type they check (e.g. X in red, Z in blue, Y in purple). Node tags are displayed next to the nodes for clarity."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import ConnectionPatch

        edge_color = {
            PauliChar.X: "tab:red",
            PauliChar.Z: "tab:blue",
            PauliChar.Y: "tab:purple",
        }

        all_nodes = list(self.variable_nodes) + list(self.check_nodes)
        if not all_nodes:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_title("Empty Tanner Graph")
            ax.set_axis_off()
            return fig, ax

        coord_lengths = {len(node.coordinates) for node in all_nodes}
        if 0 in coord_lengths and len(coord_lengths) > 1:
            raise ValueError("Either all nodes must have coordinates or none.")
        if len(coord_lengths) > 1:
            raise ValueError("All node coordinates must have the same dimension.")

        dim = coord_lengths.pop()
        if dim not in {0, 2, 3}:
            raise ValueError("Node coordinates must be empty, 2D, or 3D.")

        def _draw_single_axis(ax, pos):
            for edge in self.edges:
                x1, y1 = pos[edge.variable_node]
                x2, y2 = pos[edge.check_node]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_color.get(edge.pauli_checked, "gray"),
                    linewidth=1.6,
                    alpha=0.85,
                    zorder=1,
                )

            var_nodes = sorted(self.variable_nodes, key=lambda n: n.tag)
            check_nodes = sorted(self.check_nodes, key=lambda n: n.tag)

            ax.scatter(
                [pos[n][0] for n in var_nodes],
                [pos[n][1] for n in var_nodes],
                marker="o",
                s=120,
                color="black",
                zorder=3,
                label="Variable",
            )
            ax.scatter(
                [pos[n][0] for n in check_nodes],
                [pos[n][1] for n in check_nodes],
                marker="s",
                s=130,
                color="dimgray",
                zorder=3,
                label="Check",
            )

            for node in var_nodes + check_nodes:
                x, y = pos[node]
                ax.text(x + 0.03, y + 0.03, node.tag, fontsize=8)

            legend_items = [
                Line2D(
                    [0], [0], marker="o", linestyle="", color="black", label="Variable"
                ),
                Line2D(
                    [0], [0], marker="s", linestyle="", color="dimgray", label="Check"
                ),
                Line2D([0], [0], color="tab:red", label="X edge"),
                Line2D([0], [0], color="tab:blue", label="Z edge"),
                Line2D([0], [0], color="tab:purple", label="Y edge"),
            ]
            ax.legend(handles=legend_items, loc="best", fontsize=8)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.12, linewidth=0.6, linestyle=":")

        if dim == 0:
            fig, ax = plt.subplots(figsize=(9, 5))

            var_nodes = sorted(self.variable_nodes, key=lambda n: n.tag)
            check_nodes = sorted(self.check_nodes, key=lambda n: n.tag)
            css_like = all(
                c.check_type in {PauliChar.X, PauliChar.Z} for c in check_nodes
            )

            pos = {}
            for i, node in enumerate(var_nodes):
                pos[node] = (0.0, float(-i))

            if css_like:
                x_checks = [c for c in check_nodes if c.check_type == PauliChar.X]
                z_checks = [c for c in check_nodes if c.check_type == PauliChar.Z]
                for i, node in enumerate(sorted(x_checks, key=lambda n: n.tag)):
                    pos[node] = (-1.0, float(-i))
                for i, node in enumerate(sorted(z_checks, key=lambda n: n.tag)):
                    pos[node] = (1.0, float(-i))
            else:
                for i, node in enumerate(check_nodes):
                    pos[node] = (1.0, float(-i))

            _draw_single_axis(ax, pos)
            ax.set_title("Tanner Graph (Bipartite Layout)")
            return fig, ax

        if dim == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            pos = {
                node: (node.coordinates[0], node.coordinates[1]) for node in all_nodes
            }
            if not periodic:
                _draw_single_axis(ax, pos)
                ax.set_title("Tanner Graph (2D Coordinates)")
                return fig, ax

            # Periodic rendering for toroidal layouts: wrap-across edges are drawn
            # as two boundary-touching segments so periodic connectivity is explicit.
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            span_x = (max_x - min_x) + 1
            span_y = (max_y - min_y) + 1
            pad = 0.45

            for edge in self.edges:
                x1, y1 = pos[edge.variable_node]
                x2, y2 = pos[edge.check_node]
                color = edge_color.get(edge.pauli_checked, "gray")

                dx = x2 - x1
                dy = y2 - y1
                wraps_x = abs(dx) > (span_x / 2)
                wraps_y = abs(dy) > (span_y / 2)

                if not wraps_x and not wraps_y:
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=1.6,
                        alpha=0.85,
                        zorder=1,
                    )
                    continue

                if wraps_x and not wraps_y:
                    if dx > 0:
                        ax.plot(
                            [x1, min_x - pad],
                            [y1, y1],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [max_x + pad, x2],
                            [y2, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    else:
                        ax.plot(
                            [x1, max_x + pad],
                            [y1, y1],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [min_x - pad, x2],
                            [y2, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    continue

                if wraps_y and not wraps_x:
                    if dy > 0:
                        ax.plot(
                            [x1, x1],
                            [y1, min_y - pad],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [x2, x2],
                            [max_y + pad, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    else:
                        ax.plot(
                            [x1, x1],
                            [y1, max_y + pad],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                        ax.plot(
                            [x2, x2],
                            [min_y - pad, y2],
                            color=color,
                            linewidth=1.6,
                            alpha=0.85,
                            zorder=1,
                        )
                    continue

                # Fallback for rare diagonal wrap cases.
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    linewidth=1.2,
                    alpha=0.6,
                    zorder=1,
                    linestyle="--",
                )

            var_nodes = sorted(self.variable_nodes, key=lambda n: n.tag)
            check_nodes = sorted(self.check_nodes, key=lambda n: n.tag)

            ax.scatter(
                [pos[n][0] for n in var_nodes],
                [pos[n][1] for n in var_nodes],
                marker="o",
                s=120,
                color="black",
                zorder=3,
                label="Variable",
            )
            ax.scatter(
                [pos[n][0] for n in check_nodes],
                [pos[n][1] for n in check_nodes],
                marker="s",
                s=130,
                color="dimgray",
                zorder=3,
                label="Check",
            )

            for node in var_nodes + check_nodes:
                x, y = pos[node]
                ax.text(x + 0.03, y + 0.03, node.tag, fontsize=8)

            legend_items = [
                Line2D(
                    [0], [0], marker="o", linestyle="", color="black", label="Variable"
                ),
                Line2D(
                    [0], [0], marker="s", linestyle="", color="dimgray", label="Check"
                ),
                Line2D([0], [0], color="tab:red", label="X edge"),
                Line2D([0], [0], color="tab:blue", label="Z edge"),
                Line2D([0], [0], color="tab:purple", label="Y edge"),
            ]
            ax.legend(handles=legend_items, loc="best", fontsize=8)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.12, linewidth=0.6, linestyle=":")
            ax.set_xlim(min_x - (pad + 0.15), max_x + (pad + 0.15))
            ax.set_ylim(min_y - (pad + 0.15), max_y + (pad + 0.15))
            ax.set_title("Tanner Graph (2D Toroidal Coordinates)")
            return fig, ax

        planes = sorted({node.coordinates[2] for node in all_nodes})
        fig, axes = plt.subplots(
            1,
            len(planes),
            figsize=(6 * len(planes), 5),
            squeeze=False,
        )
        axes_list = list(axes[0])
        plane_to_ax = {plane: axes_list[i] for i, plane in enumerate(planes)}

        pos_2d = {
            node: (node.coordinates[0], node.coordinates[1]) for node in all_nodes
        }
        cross_plane_edges = []
        for edge in self.edges:
            p_var = edge.variable_node.coordinates[2]
            p_chk = edge.check_node.coordinates[2]
            if p_var == p_chk:
                ax = plane_to_ax[p_var]
                x1, y1 = pos_2d[edge.variable_node]
                x2, y2 = pos_2d[edge.check_node]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_color.get(edge.pauli_checked, "gray"),
                    linewidth=1.6,
                    alpha=0.85,
                    zorder=1,
                )
            else:
                cross_plane_edges.append(edge)

        for plane, ax in plane_to_ax.items():
            plane_var = sorted(
                [n for n in self.variable_nodes if n.coordinates[2] == plane],
                key=lambda n: n.tag,
            )
            plane_check = sorted(
                [n for n in self.check_nodes if n.coordinates[2] == plane],
                key=lambda n: n.tag,
            )

            if plane_var:
                ax.scatter(
                    [pos_2d[n][0] for n in plane_var],
                    [pos_2d[n][1] for n in plane_var],
                    marker="o",
                    s=120,
                    color="black",
                    zorder=3,
                )
            if plane_check:
                ax.scatter(
                    [pos_2d[n][0] for n in plane_check],
                    [pos_2d[n][1] for n in plane_check],
                    marker="s",
                    s=130,
                    color="dimgray",
                    zorder=3,
                )

            for node in plane_var + plane_check:
                x, y = pos_2d[node]
                ax.text(x + 0.03, y + 0.03, node.tag, fontsize=8)

            ax.set_title(f"Plane {plane}")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.12, linewidth=0.6, linestyle=":")

        for edge in cross_plane_edges:
            x1, y1 = pos_2d[edge.variable_node]
            x2, y2 = pos_2d[edge.check_node]
            a1 = plane_to_ax[edge.variable_node.coordinates[2]]
            a2 = plane_to_ax[edge.check_node.coordinates[2]]
            connector = ConnectionPatch(
                xyA=(x1, y1),
                xyB=(x2, y2),
                coordsA="data",
                coordsB="data",
                axesA=a1,
                axesB=a2,
                color=edge_color.get(edge.pauli_checked, "gray"),
                linestyle="--",
                linewidth=1.1,
                alpha=0.7,
            )
            fig.add_artist(connector)

        fig.suptitle("Tanner Graph (3D Coordinates by Plane)")
        fig.tight_layout()
        return fig, axes_list

    def __or__(self, other: "TannerGraph") -> "TannerGraph":
        """Returns the union of two Tanner graphs. The resulting Tanner graph contains all variable nodes, check nodes, and edges from both graphs. If there are overlapping nodes or edges (i.e. nodes or edges with the same id), they are merged into a single node or edge in the resulting graph."""

        joined_variable_nodes = self.variable_nodes.union(other.variable_nodes)
        joined_check_nodes = self.check_nodes.union(other.check_nodes)
        joined_edges = self.edges.union(other.edges)
        return TannerGraph(
            variable_nodes=joined_variable_nodes,
            check_nodes=joined_check_nodes,
            edges=joined_edges,
        )

    def __eq__(self, value):
        """
        Returns True if the Tanner graph is equal to another Tanner graph. Two Tanner graphs are equal if they have the same variable nodes, check nodes, and edges.
        Equal Tanner graphs not only have the same structure but also the same node and edge ids.
        """
        if not isinstance(value, TannerGraph):
            return NotImplemented
        if self.variable_nodes != value.variable_nodes:
            return False
        if self.check_nodes != value.check_nodes:
            return False
        if self.edges != value.edges:
            return False
        return True

    def __contains__(self, item: TannerNode | TannerEdge) -> bool:
        """Check if a node or edge is in the Tanner graph."""
        if isinstance(item, VariableNode):
            return item in self.variable_nodes
        elif isinstance(item, CheckNode):
            return item in self.check_nodes
        elif isinstance(item, TannerEdge):
            return item in self.edges
        else:
            raise ValueError("Item must be a VariableNode, CheckNode, or TannerEdge.")

    def is_disjoint(self, other: "TannerGraph") -> bool:
        """Returns True if the Tanner graph is disjoint with another Tanner graph. Meaning they do not share any node (based on node id)."""
        if not isinstance(other, TannerGraph):
            raise ValueError("Can only check disjointness with another TannerGraph.")
        return self.variable_nodes.isdisjoint(
            other.variable_nodes
        ) and self.check_nodes.isdisjoint(other.check_nodes)
