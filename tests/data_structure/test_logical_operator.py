"""Tests for logical_operator module."""

import pytest
from qldpc_sim.data_structure.logical_operator import LogicalOperator
from qldpc_sim.data_structure.pauli import PauliChar, PauliString
from qldpc_sim.data_structure.tanner_graph import VariableNode


class TestLogicalOperator:
    """Test suite for LogicalOperator class."""

    def test_initialization(self):
        """Test LogicalOperator initialization."""
        v0 = VariableNode(tag="v0")
        v1 = VariableNode(tag="v1")
        op = LogicalOperator(
            logical_type=PauliChar.X,
            operator=PauliString(string=(PauliChar.X, PauliChar.I)),
            target_nodes=(v0, v1),
        )

        assert op.logical_type == PauliChar.X
        assert op.operator.string == (PauliChar.X, PauliChar.I)
        assert op.target_nodes == (v0, v1)
        assert op.id is not None

    def test_length_mismatch_raises(self):
        """Operator and target_nodes lengths must match."""
        v0 = VariableNode(tag="v0")
        v1 = VariableNode(tag="v1")

        with pytest.raises(ValueError, match="Length of Pauli string"):
            LogicalOperator(
                logical_type=PauliChar.Z,
                operator=PauliString(string=(PauliChar.Z,)),
                target_nodes=(v0, v1),
            )

    def test_default_operator_when_not_provided(self):
        """If omitted, operator defaults to logical_type on each target node."""
        v0 = VariableNode(tag="v0")
        v1 = VariableNode(tag="v1")
        v2 = VariableNode(tag="v2")

        op = LogicalOperator(
            logical_type=PauliChar.Z,
            target_nodes=(v0, v1, v2),
        )

        assert op.operator.string == (PauliChar.Z, PauliChar.Z, PauliChar.Z)

    def test_requires_variable_node_targets(self):
        """Invalid target node payload should be rejected by validation."""
        with pytest.raises(Exception):
            LogicalOperator(
                logical_type=PauliChar.X,
                operator=PauliString(string=(PauliChar.X,)),
                target_nodes=("not-a-node",),
            )


class TestLogicalOperatorProperties:
    """Test suite for LogicalOperator properties."""

    def test_instances_have_distinct_ids(self):
        """Two operators with same payload should still get unique IDs."""
        v0 = VariableNode(tag="v0")
        op1 = LogicalOperator(
            logical_type=PauliChar.X,
            operator=PauliString(string=(PauliChar.X,)),
            target_nodes=(v0,),
        )
        op2 = LogicalOperator(
            logical_type=PauliChar.X,
            operator=PauliString(string=(PauliChar.X,)),
            target_nodes=(v0,),
        )

        assert op1.id != op2.id

    def test_equivalence(self):
        """Models with same explicit values (same id) compare equal."""
        v0 = VariableNode(tag="v0")
        op1 = LogicalOperator(
            logical_type=PauliChar.Z,
            operator=PauliString(string=(PauliChar.Z,)),
            target_nodes=(v0,),
        )
        op2 = LogicalOperator.model_validate(op1.model_dump())

        assert op1 == op2

    def test_model_is_frozen(self):
        """LogicalOperator is immutable after creation."""
        v0 = VariableNode(tag="v0")
        op = LogicalOperator(
            logical_type=PauliChar.X,
            operator=PauliString(string=(PauliChar.X,)),
            target_nodes=(v0,),
        )

        with pytest.raises(Exception):
            op.logical_type = PauliChar.Z
