import random
from typing import Dict, List, Tuple
from .tanner_graph import CheckNode, TannerEdge, TannerGraph, TannerNode, VariableNode


class TannerGraphAlgebra:
    """This class provides utility function to work with tanner graphs."""

    def connect(
        graph1: TannerGraph, graph2: TannerGraph, connecting_edges: List[TannerEdge]
    ) -> TannerGraph:
        """Connect two tanner graph into a new Tanner graph, using the connecting edges provided.

        Parameters
        ----------
        graph1 : TannerGraph
        graph2 : TannerGraph
        connecting_edges : List[TannerEdge]

        Returns
        -------
        TannerGraph

        Raises
        ------
        ValueError
            If the two graphs have overlapping nodes, or if the connecting edges do not connect nodes from different graphs.
        ValueError
            If the connecting edges do not connect nodes from different graphs.
        """
        # Implementation of merging two Tanner graphs
        if not graph1.variable_nodes.isdisjoint(
            graph2.variable_nodes
        ) or not graph1.check_nodes.isdisjoint(graph2.check_nodes):
            raise ValueError("Graphs have overlapping nodes; cannot merge.")

        for ce in connecting_edges:
            valid_edge = (
                ce.variable_node in graph1.variable_nodes
                and ce.check_node in graph2.check_nodes
                or ce.variable_node in graph2.variable_nodes
                and ce.check_node in graph1.check_nodes
            )
            if not valid_edge:
                raise ValueError(
                    "Connecting edge does not connect nodes from different graphs."
                )
        merged_variable_nodes = graph1.variable_nodes.union(graph2.variable_nodes)
        merged_check_nodes = graph1.check_nodes.union(graph2.check_nodes)
        merged_edges = graph1.edges.union(graph2.edges).union(set(connecting_edges))
        return TannerGraph(
            variable_nodes=merged_variable_nodes,
            check_nodes=merged_check_nodes,
            edges=merged_edges,
        )

    def dual_graph(
        graph: TannerGraph,
    ) -> Tuple[TannerGraph, Dict[TannerNode, TannerNode]]:
        """Return a dual of the Tanner graph. A dual graph is constructed by swapping the variable and check nodes, and connecting them according to the original edges. The Pauli checked on the edges are also dualized (X<->Z, Y->Y).

        Parameters
        ----------
        graph : TannerGraph
            The Tanner graph to be dualized.

        Returns
        -------
        Tuple[TannerGraph, Dict[TannerNode, TannerNode]]
            The dual Tanner graph, and a mapping from the original nodes to the "corresponding" new nodes in the dual graph.
        """

        def dual_tag(tag: str) -> str:
            t = ""
            if tag.endswith("_T"):
                t += tag[:-2]
            else:
                t += tag + "_T"
            return "_" + t

        check_node = random.choice(tuple(graph.check_nodes))
        check_type = check_node.check_type.dual()

        check_to_variable = {
            check: VariableNode(tag=dual_tag(check.tag)) for check in graph.check_nodes
        }
        variable_to_check = {
            variable: CheckNode(
                tag=dual_tag(variable.tag),
                check_type=check_type,
            )
            for variable in graph.variable_nodes
        }

        dual_variable_nodes = set(check_to_variable.values())
        dual_check_nodes = set(variable_to_check.values())

        dual_edges = {
            TannerEdge(
                variable_node=check_to_variable[edge.check_node],
                check_node=variable_to_check[edge.variable_node],
                pauli_checked=edge.pauli_checked.dual(),
            )
            for edge in graph.edges
        }

        old_to_new_nodes = {**check_to_variable, **variable_to_check}

        return (
            TannerGraph(
                variable_nodes=dual_variable_nodes,
                check_nodes=dual_check_nodes,
                edges=dual_edges,
            ),
            old_to_new_nodes,
        )

    def indexed_dual_graph(
        graph: TannerGraph, index: Dict[int, TannerNode]
    ) -> Tuple[TannerGraph, Dict[int, TannerNode]]:
        """Return the dual Tanner graph and an assignment of integers, corresponding to the structure of the original graph."""
        dual_graph, old_to_new_nodes = TannerGraphAlgebra.dual_graph(graph)
        n_index = {idx: old_to_new_nodes[node] for idx, node in index.items()}
        return dual_graph, n_index

    def index_nodes(graph: TannerGraph) -> Dict[int, TannerNode]:
        """Assigns a unique integer index to each node in the Tanner graph."""
        index = {}
        current_index = 0
        for node in graph.variable_nodes.union(graph.check_nodes):
            index[current_index] = node
            current_index += 1
        return index


import matplotlib.pyplot as plt
import networkx as nx
from collections import deque


def plot_tanner_graph(tanner_graph: TannerGraph, layer_spacing=10, node_spacing=12):
    """
    Plot a Tanner graph with nodes arranged in layers according to the number of leading underscores
    or a tag substring of the form "l<i>".

    Args:
        tanner_graph (TannerGraph): Tanner graph instance.
        layer_spacing (float): Vertical spacing between layers.
        node_spacing (float): Horizontal spacing between nodes in the same layer.
    """
    G = nx.Graph()

    # Add all nodes to the graph
    for n in list(tanner_graph.variable_nodes) + list(tanner_graph.check_nodes):
        G.add_node(
            n.id,
            tag=n.tag,
            type="variable" if n.tag.strip("_").startswith("v") else "check",
        )

    # Add edges
    for e in tanner_graph.edges:
        G.add_edge(e.variable_node.id, e.check_node.id)

    def get_layer_index(tag: str):
        import re

        match = re.search(r"l(\d+)", tag)
        if match:
            return int(match.group(1)) + 1
        if tag.startswith("_"):
            underscore_count = len(tag) - len(tag.lstrip("_"))
            if underscore_count == 0:
                return 0
            return underscore_count
        return 0

    # Determine layers
    layers = {}
    for n in G.nodes:
        node_tag = G.nodes[n]["tag"]
        layer_index = get_layer_index(node_tag)

        layers.setdefault(layer_index, []).append(n)

    # Sort nodes in each layer by their index
    for layer_num, nodes in layers.items():

        def get_index(node_id):
            tag = G.nodes[node_id]["tag"]
            # find the first integer in the tag
            import re

            match = re.search(r"\d+", tag)
            return int(match.group()) if match else 0

        layers[layer_num] = sorted(nodes, key=get_index)

    # Assign positions
    positions = {}
    for layer_num, nodes in layers.items():
        y = -layer_num * layer_spacing  # top layer is y=0, next below

        tag_to_nodes = {}
        for node_id in nodes:
            tag_to_nodes.setdefault(G.nodes[node_id]["tag"], []).append(node_id)

        def get_index_by_tag(tag):
            import re

            match = re.search(r"\d+", tag)
            return int(match.group()) if match else 0

        ordered_tags = sorted(tag_to_nodes.keys(), key=get_index_by_tag)
        for tag_index, tag in enumerate(ordered_tags):
            group = tag_to_nodes[tag]
            base_x = tag_index * node_spacing
            if len(group) == 1:
                positions[group[0]] = (base_x, y)
                continue

            offset_step = node_spacing * 0.7
            y_step = layer_spacing * 0.4
            for idx, node_id in enumerate(group):
                offset = (idx - (len(group) - 1) / 2) * offset_step
                y_offset = (idx - (len(group) - 1) / 2) * y_step
                positions[node_id] = (base_x + offset, y + y_offset)

    # Scale figure size for notebooks so labels have more room
    max_nodes_per_layer = max((len(nodes) for nodes in layers.values()), default=1)
    max_layer_index = max(layers.keys(), default=0)
    fig_width = max(8, max_nodes_per_layer * node_spacing * 0.25)
    fig_height = max(6, (max_layer_index + 1) * layer_spacing * 0.25)
    plt.figure(figsize=(fig_width, fig_height))

    # Draw nodes
    var_nodes = [n.id for n in tanner_graph.variable_nodes]
    check_nodes = [n.id for n in tanner_graph.check_nodes]

    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=var_nodes,
        node_color="skyblue",
        node_size=600,
        label="Variable",
    )
    nx.draw_networkx_nodes(
        G,
        positions,
        nodelist=check_nodes,
        node_color="lightcoral",
        node_shape="s",
        node_size=600,
        label="Check",
    )

    # Draw edges
    nx.draw_networkx_edges(G, positions)

    # Draw labels
    labels = {node_id: G.nodes[node_id]["tag"] for node_id in G.nodes}
    nx.draw_networkx_labels(G, positions, labels=labels, font_size=10)

    plt.axis("off")
    plt.legend(scatterpoints=1)
    plt.title("Layered Tanner Graph")
    plt.show()
