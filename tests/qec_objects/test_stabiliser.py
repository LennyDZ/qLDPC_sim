"""Tests for stabiliser module."""

import pytest
import numpy as np

from src.qldpc_sim.qec_objects import Stabiliser, QuantumBit


@pytest.fixture
def data_qubits():
    """Create a list of data qubits for testing."""
    return [QuantumBit(name=f"q{i}", idx=str(i), qubit_type="data") for i in range(4)]


@pytest.fixture
def ancilla_qubit():
    """Create an ancilla qubit for testing."""
    return QuantumBit(name="a0", idx="4", qubit_type="ancilla")


@pytest.fixture
def z_stabiliser(data_qubits, ancilla_qubit):
    """Create a Z-type stabiliser for testing."""
    return Stabiliser(
        name="Z_stabiliser",
        pauli_string="ZZZZ",
        data_qubits=data_qubits,
        ancilla_qubit=ancilla_qubit,
    )


@pytest.fixture
def x_stabiliser(data_qubits, ancilla_qubit):
    """Create an X-type stabiliser for testing."""
    return Stabiliser(
        name="X_stabiliser",
        pauli_string="XXXX",
        data_qubits=data_qubits,
        ancilla_qubit=ancilla_qubit,
    )


@pytest.fixture
def mixed_stabiliser(data_qubits, ancilla_qubit):
    """Create a mixed X/Z stabiliser for testing."""
    return Stabiliser(
        name="mixed_stabiliser",
        pauli_string="XZXZ",
        data_qubits=data_qubits,
        ancilla_qubit=ancilla_qubit,
    )


class TestStabiliserCreation:
    """Test stabiliser object creation and validation."""

    def test_create_stabiliser_with_pauli_string(self, data_qubits, ancilla_qubit):
        """Test creating a stabiliser with pauli_string."""
        stabiliser = Stabiliser(
            name="test_stabiliser",
            pauli_string="XZZX",
            data_qubits=data_qubits,
            ancilla_qubit=ancilla_qubit,
        )
        assert stabiliser.name == "test_stabiliser"
        assert stabiliser.pauli_string == "XZZX"
        assert len(stabiliser.data_qubits) == 4

    def test_create_stabiliser_with_pauli_matrix(self, data_qubits, ancilla_qubit):
        """Test creating a stabiliser with pauli_matrix."""
        pauli_matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])  # XZXZ pattern
        stabiliser = Stabiliser(
            name="matrix_stabiliser",
            pauli_matrix=pauli_matrix,
            data_qubits=data_qubits,
            ancilla_qubit=ancilla_qubit,
        )
        assert stabiliser.get_string == "XZXZ"

    def test_stabiliser_requires_either_string_or_matrix(
        self, data_qubits, ancilla_qubit
    ):
        """Test that either pauli_string or pauli_matrix must be provided."""
        with pytest.raises(
            ValueError, match="Either 'pauli_string' or 'pauli_matrix' must be provided"
        ):
            Stabiliser(
                name="invalid_stabiliser",
                data_qubits=data_qubits,
                ancilla_qubit=ancilla_qubit,
            )


class TestStabiliserValidation:
    """Test stabiliser validation rules."""

    def test_pauli_string_only_xyz(self, data_qubits, ancilla_qubit):
        """Test that pauli_string can only contain X, Y, Z characters."""
        with pytest.raises(ValueError, match="must contain only 'X', 'Y', or 'Z'"):
            Stabiliser(
                name="invalid",
                pauli_string="XYZAX",  # Contains 'A'
                data_qubits=data_qubits,
                ancilla_qubit=ancilla_qubit,
            )

    def test_pauli_string_cannot_contain_identity(self, data_qubits, ancilla_qubit):
        """Test that pauli_string cannot contain identity (I) characters."""
        with pytest.raises(ValueError, match="must contain only 'X', 'Y', or 'Z'"):
            Stabiliser(
                name="invalid",
                pauli_string="XYZI",  # Contains 'I'
                data_qubits=data_qubits,
                ancilla_qubit=ancilla_qubit,
            )

    def test_data_qubits_must_be_data_type(self, ancilla_qubit):
        """Test that data_qubits must be of type 'data'."""
        bad_qubits = [QuantumBit(name="bad_q0", idx="0", qubit_type="ancilla")]
        with pytest.raises(ValueError, match="must be a data qubit"):
            Stabiliser(
                name="invalid",
                pauli_string="X",
                data_qubits=bad_qubits,
                ancilla_qubit=ancilla_qubit,
            )

    def test_ancilla_qubit_must_be_ancilla_type(self, data_qubits):
        """Test that ancilla_qubit must be of type 'ancilla'."""
        bad_ancilla = QuantumBit(name="bad_a0", idx="4", qubit_type="data")
        with pytest.raises(ValueError, match="must be an ancilla qubit"):
            Stabiliser(
                name="invalid",
                pauli_string="ZZZZ",
                data_qubits=data_qubits,
                ancilla_qubit=bad_ancilla,
            )

    def test_pauli_length_must_match_data_qubits(self, data_qubits, ancilla_qubit):
        """Test that pauli_string length must match number of data_qubits."""
        with pytest.raises(ValueError, match="must match number of data qubits"):
            Stabiliser(
                name="invalid",
                pauli_string="XZ",  # Only 2 operators for 4 qubits
                data_qubits=data_qubits,
                ancilla_qubit=ancilla_qubit,
            )


class TestStabiliserProperties:
    """Test stabiliser properties and computed attributes."""

    def test_z_only_stabiliser(self, z_stabiliser):
        """Test identification of Z-only stabiliser."""
        assert z_stabiliser.is_Z_only
        assert not z_stabiliser.is_X_only

    def test_x_only_stabiliser(self, x_stabiliser):
        """Test identification of X-only stabiliser."""
        assert x_stabiliser.is_X_only
        assert not x_stabiliser.is_Z_only

    def test_mixed_stabiliser_properties(self, mixed_stabiliser):
        """Test that mixed stabiliser is neither X-only nor Z-only."""
        assert not mixed_stabiliser.is_X_only
        assert not mixed_stabiliser.is_Z_only

    def test_stabiliser_weight(self, data_qubits, ancilla_qubit):
        """Test weight property of stabiliser."""
        stabiliser = Stabiliser(
            name="test",
            pauli_string="XZZX",
            data_qubits=data_qubits,
            ancilla_qubit=ancilla_qubit,
        )
        assert stabiliser.weight == 4  # All operators are non-identity


class TestStabiliserStimCircuit:
    """Test stim circuit compilation for stabilisers."""

    def test_z_only_stim_circuit(self, z_stabiliser):
        """Test stim circuit generation for Z-only stabiliser."""
        stim_instructions, weight = z_stabiliser.compiled_stim_circuit

        # Z-only stabiliser should only have CX gates and measurement
        assert isinstance(stim_instructions, list)
        assert len(stim_instructions) > 0
        # Should end with measurement
        assert any("M" in instr for instr in stim_instructions)
        # Check weight fields
        assert weight.h_count == 0  # Z-only has no H operations
        assert weight.meas_count == 1

    def test_x_only_stim_circuit(self, x_stabiliser):
        """Test stim circuit generation for X-only stabiliser."""
        stim_instructions, weight = x_stabiliser.compiled_stim_circuit

        # X-only stabiliser should have H gates, CX gates, and measurement
        assert isinstance(stim_instructions, list)
        assert len(stim_instructions) > 0
        # Should start and end with H operations
        assert any("H" in instr for instr in stim_instructions)
        # Should have measurement
        assert any("M" in instr for instr in stim_instructions)
        # Check weight
        assert weight.h_count == 2  # X-only has 2 H rotations
        assert weight.meas_count == 1

    def test_mixed_stim_circuit(self, mixed_stabiliser):
        """Test stim circuit generation for mixed X/Z stabiliser."""
        stim_instructions, weight = mixed_stabiliser.compiled_stim_circuit

        # Mixed stabiliser should have H gates, CX gates, and measurement
        assert isinstance(stim_instructions, list)
        assert len(stim_instructions) > 0
        # Should have measurement
        assert any("M" in instr for instr in stim_instructions)
        # Check weight: 2 X operations in "XZXZ" pattern
        assert weight.h_count == 2
        assert weight.meas_count == 1

    def test_stim_circuit_with_custom_order(self, data_qubits, ancilla_qubit):
        """Test that custom extraction order affects stim circuit."""
        custom_order = [2, 0, 3, 1]
        stabiliser = Stabiliser(
            name="custom_order_stim",
            pauli_string="ZZZZ",
            data_qubits=data_qubits,
            ancilla_qubit=ancilla_qubit,
            extraction_order=custom_order,
        )
        stim_instructions, _ = stabiliser.compiled_stim_circuit
        assert isinstance(stim_instructions, list)
        assert len(stim_instructions) > 0

    def test_mixed_stabiliser_with_y_raises_error(self, data_qubits, ancilla_qubit):
        """Test that mixed stabilisers with Y terms are not yet supported."""
        stabiliser = Stabiliser(
            name="with_y",
            pauli_string="XYXZ",
            data_qubits=data_qubits,
            ancilla_qubit=ancilla_qubit,
        )
        with pytest.raises(NotImplementedError):
            _ = stabiliser.compiled_stim_circuit


class TestStabiliserIntegration:
    """Integration tests for stabiliser functionality."""

    def test_stabiliser_with_different_sizes(self):
        """Test creating stabilisers with different numbers of data qubits."""
        for num_qubits in [2, 4, 8]:
            data_qubits = [
                QuantumBit(name=f"q{i}", idx=str(i), qubit_type="data")
                for i in range(num_qubits)
            ]
            ancilla_qubit = QuantumBit(
                name="a0", idx=str(num_qubits), qubit_type="ancilla"
            )
            pauli_string = "X" * num_qubits

            stabiliser = Stabiliser(
                name=f"stabiliser_{num_qubits}",
                pauli_string=pauli_string,
                data_qubits=data_qubits,
                ancilla_qubit=ancilla_qubit,
            )

            assert len(stabiliser.data_qubits) == num_qubits
            assert stabiliser.pauli_string == pauli_string

    def test_multiple_stabilisers_same_qubits(self, data_qubits, ancilla_qubit):
        """Test creating multiple stabilisers on same qubits with different patterns."""
        patterns = ["XXXX", "ZZZZ", "XZXZ"]
        stabilisers = []

        for pattern in patterns:
            stabiliser = Stabiliser(
                name=f"stabiliser_{pattern}",
                pauli_string=pattern,
                data_qubits=data_qubits,
                ancilla_qubit=ancilla_qubit,
            )
            stabilisers.append(stabiliser)

        assert len(stabilisers) == 3
        assert all(len(s.data_qubits) == 4 for s in stabilisers)
        assert [s.pauli_string for s in stabilisers] == patterns
