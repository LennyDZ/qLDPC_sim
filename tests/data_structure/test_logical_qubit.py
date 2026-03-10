"""Tests for logical_qubit module."""

import pytest
from pydantic import ValidationError

from qldpc_sim.data_structure.logical_operator import LogicalOperator
from qldpc_sim.data_structure.logical_qubit import LogicalQubit
from qldpc_sim.data_structure.pauli import PauliChar, PauliString
from qldpc_sim.data_structure.tanner_graph import VariableNode


@pytest.fixture
def logical_operator_factory():
    """Factory fixture to build LogicalOperator via Pydantic validation."""

    def _make(logical_type: PauliChar, tag: str) -> LogicalOperator:
        return LogicalOperator.model_validate(
            {
                "logical_type": logical_type,
                "operator": PauliString(string=(logical_type,)),
                "target_nodes": (VariableNode(tag=tag),),
            }
        )

    return _make


class TestLogicalQubit:
    """Test suite for LogicalQubit class."""

    def test_initialization(self, logical_operator_factory):
        """Test LogicalQubit initialization."""
        lx = logical_operator_factory(PauliChar.X, "vx")
        lz = logical_operator_factory(PauliChar.Z, "vz")

        qubit = LogicalQubit(name="L0", logical_x=lx, logical_z=lz)

        assert qubit.name == "L0"
        assert qubit.logical_x == lx
        assert qubit.logical_z == lz
        assert qubit.id is not None

    def test_name_is_required(self, logical_operator_factory):
        """Name is required by the model."""
        lx = logical_operator_factory(PauliChar.X, "vx")
        lz = logical_operator_factory(PauliChar.Z, "vz")

        with pytest.raises(ValidationError):
            LogicalQubit(logical_x=lx, logical_z=lz)

    def test_logical_operators_are_required(self, logical_operator_factory):
        """Both logical_x and logical_z must be provided."""
        lx = logical_operator_factory(PauliChar.X, "vx")
        lz = logical_operator_factory(PauliChar.Z, "vz")

        with pytest.raises(ValidationError):
            LogicalQubit(name="L0", logical_z=lz)

        with pytest.raises(ValidationError):
            LogicalQubit(name="L0", logical_x=lx)


class TestLogicalQubitOperations:
    """Test suite for LogicalQubit operations."""

    def test_instances_have_distinct_ids(self, logical_operator_factory):
        """Two qubits with same payload should still get unique IDs."""
        lx = logical_operator_factory(PauliChar.X, "vx")
        lz = logical_operator_factory(PauliChar.Z, "vz")

        q1 = LogicalQubit(name="L0", logical_x=lx, logical_z=lz)
        q2 = LogicalQubit(name="L0", logical_x=lx, logical_z=lz)

        assert q1.id != q2.id

    def test_model_is_frozen(self, logical_operator_factory):
        """LogicalQubit is immutable after creation."""
        lx = logical_operator_factory(PauliChar.X, "vx")
        lz = logical_operator_factory(PauliChar.Z, "vz")
        qubit = LogicalQubit(name="L0", logical_x=lx, logical_z=lz)

        with pytest.raises(Exception):
            qubit.name = "L1"
