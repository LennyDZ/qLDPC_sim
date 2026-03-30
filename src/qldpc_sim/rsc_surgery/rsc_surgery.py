
from functools import cached_property
from typing import Dict
from qldpc_sim.qec_code.rotated_surface_code import RotatedSurfaceCode
from qldpc_sim.qldpc_experiment.record import EventType, OutcomeSet

from ..data_structure import (
    TannerGraph,
    LogicalOperator,
    CheckNode,
    TannerEdge,
    VariableNode,
    PauliChar,
)
from ..data_structure import TannerGraphAlgebra as tga
from ..qldpc_experiment import (
    PauliMeasurement,
    ApplyGates,
    MeasurementCompiler,
    StabilisersMeasurementCompiler,
)

class SurgeMeasurement(PauliMeasurement):

    distance: int

    @cached_property
    def tanner_supports(self) -> Dict[LogicalOperator, TannerGraph]:
        tanners = {}
        for lop in self.logical_targets:
            code = self.context.initial_assignement[lop]
            new_support = code.tanner_graph.get_support(
                lop.target_nodes, lop.logical_type
            )
            new_key = lop
            for key, other_support in tanners.items():
                # Check if supports are disjoint or not
                # All joint support are merged together, as they will be measured by the same ancilla structure.

                if not new_support.is_disjoint(other_support):
                    raise ValueError(
                        "Error in building measurement: two logical operators have overlapping support"
                    )
                # if new_support == other_support:
                #     raise ValueError(
                #         "Error in building measurement: two logical operators have the same support",
                #         "This means either the same logical operator appear twice in the operator list, or a logical operator is fully supported by a higher weight logical operator. In both cases, the measurement is not well defined (for now at least).",
                #     )
                # else:
                #     other_support = tanners[key]
                #     new_key = key + lop if isinstance(key, tuple) else (key, lop)
                #     new_support = new_support | other_support

            tanners[new_key] = new_support
        return tanners

    def check_feasibility(self):
        if len(self.logical_targets) != 2:
            raise ValueError(
                "Only two logical operators can be measured together in a lattice surgery measurement for now."
            )
        c1 = self.context.initial_assignement[self.logical_targets[0]]
        c2 = self.context.initial_assignement[self.logical_targets[1]]
        if not isinstance(c1, RotatedSurfaceCode) or not isinstance(c2, RotatedSurfaceCode):
                raise ValueError(
                    "Error in building measurement: lattice surgery only works with instance of RotatedSurfaceCode."
                )
        return True

    def cost(self):
        # Implement cost evaluation specific to Measurement
        return 0

    def build_compiler_instructions(self):
        compilers = []
        basis = self.logical_targets[0].logical_type
        # I. Evalute feasability, cost
        if not self.check_feasibility():
            raise ValueError(
                "Measurement instruction is not feasible with the given memory."
            )
        # II. Build ancilla TannerGraphs
        lop_tanners = self.tanner_supports
        tanners = list(lop_tanners.values())

        # Compute classical correction of logicals that commute with the supports used.
        anticommuting_lop_by_node = {}
        corrections = {}
        for lop in self.logical_targets:
            code = self.context.initial_assignement[lop]
            for lq in code.logical_qubits:
                if lq.logical_x == lop or lq.logical_z == lop:
                    shared_target_nodes = set(lq.logical_x.target_nodes) & set(
                        lq.logical_z.target_nodes
                    )
                    if len(shared_target_nodes) > 1:
                        raise ValueError(
                            "Error in building measurement: a logical operator share more than 1 qubit with its anticommuting logical operator, this is not supported for now. Try finding a canonical basis for the code."
                        )
                    anticommuting_lop_by_node[shared_target_nodes.pop()] = (
                        lq.logical_x if lq.logical_z == lop else lq.logical_z
                    )
                    break

        new_nodes = set()
        new_edges = set()
        for i in range(self.distance):
            new_nodes.add(VariableNode(tag=f"rsc_surge_anc_{i}", coordinates=(0, 2*i, 1, 0)))
        for i in range(self.distance + 1):
            new_nodes.add(CheckNode(tag=f"rsc_surge_check_{i}", coordinates=((-1)**i, 2*i-1, 1, 0), check_type=basis))

        map_new_nodes = {n.coordinates[1]: n for n in new_nodes}
        map_new_nodes_reverse = {self.distance*2-2-k: v for k, v in map_new_nodes.items()}
        for i, n in map_new_nodes.items():
            if isinstance(n, VariableNode):
                check_n_above = map_new_nodes[i+1]
                check_n_below = map_new_nodes[i-1]
                new_edges.add(TannerEdge(variable_node=n, check_node=check_n_above, pauli_checked=check_n_above.check_type))
                new_edges.add(TannerEdge(variable_node=n, check_node=check_n_below, pauli_checked=check_n_below.check_type))

        anc_tanner = TannerGraph(
            variable_nodes=[n for n in new_nodes if isinstance(n, VariableNode)],
            check_nodes=[n for n in new_nodes if isinstance(n, CheckNode)],
            edges=new_edges,
        )
        connecting_edges = set()
        for idx, supp in enumerate(tanners):
            check = sorted(list(supp.check_nodes), key=lambda n: n.coordinates[:2])
            var = sorted(list(supp.variable_nodes), key=lambda n: n.coordinates[:2])
            if idx == 1:
                map_new_nodes = map_new_nodes_reverse
            varying_idx = 0 if var[0].coordinates[0] != var[-1].coordinates[0] else 1
            
            for c in check:
                i = c.coordinates[varying_idx]
                if c.coordinates[(varying_idx+1)%2] == 1:
                    if c.coordinates[varying_idx] == 1:
                        variant = True
                    continue
                if c.coordinates[(varying_idx+1)%2] == 2*(self.distance-1)-1:
                    continue
                if i>-1:
                    new_edge = TannerEdge(variable_node=map_new_nodes[i-1], check_node=c, pauli_checked=c.check_type)
                    connecting_edges.add(new_edge)
                if i<len(map_new_nodes)-1:
                    new_edge = TannerEdge(variable_node=map_new_nodes[i+1], check_node=c, pauli_checked=c.check_type)
                    connecting_edges.add(new_edge)
            for v in var:
                i = v.coordinates[varying_idx]
                if i %4== 0:
                    if varying_idx == 0:
                        c = map_new_nodes[i+1]
                    else:
                        c = map_new_nodes[i-1]
                else:
                    if varying_idx == 0:
                        c = map_new_nodes[i-1]
                    else:
                        c = map_new_nodes[i+1]
                new_edge = TannerEdge(variable_node=v, check_node=c, pauli_checked=c.check_type)
                
                connecting_edges.add(new_edge)

        

        c1 = self.context.initial_assignement[self.logical_targets[0]].tanner_graph
        c2 = self.context.initial_assignement[self.logical_targets[1]].tanner_graph
        merged_tanner = tga.connect(c1 | c2, anc_tanner, connecting_edges=connecting_edges)
        tga.visualize(merged_tanner, highlight_nodes=self.logical_targets[0].target_nodes)
        

        init_ancilla = [
            ApplyGates(
                tag=f"init_{self.tag}",
                target_nodes=anc_tanner.variable_nodes,
                gates=["RX"] if basis.dual() == PauliChar.X else ["RZ"],
            ),
        ]
        stab_measurement = StabilisersMeasurementCompiler(
            data=merged_tanner,
            round=self.distance,
            tag=f"ckbb_{self.tag}",
        )
        stab_measurement = StabilisersMeasurementCompiler(
            data=merged_tanner,
            round=self.distance,
            tag=f"merged_stab_{self.tag}",
        )

        readout_ancilla = MeasurementCompiler(
            data=TannerGraph(variable_nodes=anc_tanner.variable_nodes, check_nodes=set(), edges=set()),
            tag=f"readout_bridge_ancilla_{self.tag}",
            reset_qubits=True,
            free_qubits=True,
            basis=basis.dual(),
        )
        compilers.extend(init_ancilla)
        compilers.append(stab_measurement)
        compilers.append(readout_ancilla)

        outcomes = []
        stab_in_parity = set(
            [n for n in anc_tanner.check_nodes if n.check_type == basis]
        )
        parity_outcome_nodes = OutcomeSet(
            tag=f"{self.tag}_parity_outcome",
            type=EventType.OBSERVABLE,
            size=len(stab_in_parity),
            measured_nodes=stab_in_parity,
            target=self.logical_targets,
        )
        
        outcomes.append(parity_outcome_nodes)
        correct_targ = self.logical_targets[0]
        targ_op = None
        code = self.context.initial_assignement[correct_targ]
        for lq in code.logical_qubits:
            if lq.logical_x == correct_targ:
                targ_op = lq.logical_z
            elif lq.logical_z == correct_targ:
                targ_op = lq.logical_x
        correction = {n for n in anc_tanner.variable_nodes}
        

        if targ_op is not None:
            outcomes.append(
                OutcomeSet(
                    tag=f"{self.tag}log_corr_{correct_targ.id}",
                    type=EventType.FRAME_CORRECTION,
                    size=len(correction),
                    measured_nodes=correction,
                    target=targ_op,
                )
            )
        return compilers, outcomes

