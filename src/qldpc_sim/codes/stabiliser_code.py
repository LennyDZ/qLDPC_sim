from typing import List, Set, Tuple, Literal
from functools import cached_property
from pydantic import BaseModel, Field, field_validator, computed_field
from uuid import UUID, uuid4
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix, spmatrix

from ..type import PauliOperator, QuantumBit
from ..qec_objects.stabiliser import Stabiliser


class ParityCheckMatrix(BaseModel):
    """
    Represents parity check matrices for quantum error correction.

    For stabilizer codes, we have two matrices:
    - Hx: stabilizer generators measuring Z parity (detect X errors)
    - Hz: stabilizer generators measuring X parity (detect Z errors)

    The full H matrix is [Hx | Hz] (horizontally concatenated).

    Matrices are stored as scipy sparse CSR format for efficient storage and operations.
    """

    name: str = Field(description="Name of the parity check matrix")
    id: UUID = Field(default_factory=uuid4)

    # Sparse matrices in CSR format
    hx_matrix: spmatrix = Field(
        default_factory=lambda: csr_matrix((0, 0), dtype=np.int_),
        description="Hx matrix (X checks) as scipy sparse CSR matrix",
    )

    hz_matrix: spmatrix = Field(
        default_factory=lambda: csr_matrix((0, 0), dtype=np.int_),
        description="Hz matrix (Z checks) as scipy sparse CSR matrix",
    )

    # Just to tell pydantic to allow arbitrary types like spmatrix
    class Config:
        arbitrary_types_allowed = True

    @field_validator("hx_matrix", "hz_matrix", mode="before")
    @classmethod
    def convert_to_sparse(cls, v):
        """Convert input to CSR sparse matrix if needed."""
        if isinstance(v, spmatrix):
            if not isinstance(v, csr_matrix):
                v = csr_matrix(v, dtype=np.int_)
            else:
                v = v.astype(np.int_)
        elif isinstance(v, np.ndarray):
            v = csr_matrix(v, dtype=np.int_)
        elif isinstance(v, list):
            # Convert from row sparse format [List[List[int]]]
            v = _list_of_lists_to_csr(v)
        return v

    @field_validator("hx_matrix", "hz_matrix")
    @classmethod
    def validate_matrix_dims(cls, v: spmatrix, info) -> spmatrix:
        """Validate matrices have same number of columns."""
        if v.shape[0] == 0 and v.shape[1] == 0:
            # Empty matrix, skip validation
            return v

        # Check if other matrix exists and has matching columns
        hx = info.data.get("hx_matrix")
        hz = info.data.get("hz_matrix")

        if hx is not None and hz is not None:
            if hx.shape[1] != hz.shape[1]:
                raise ValueError(
                    f"hx_matrix and hz_matrix must have same number of columns. "
                    f"Got {hx.shape[1]} and {hz.shape[1]}"
                )
        return v

    @computed_field
    @property
    def num_variables(self) -> int:
        """Number of variables (qubits) - computed from matrix columns."""
        if self.hx_matrix.shape[1] > 0:
            return self.hx_matrix.shape[1]
        elif self.hz_matrix.shape[1] > 0:
            return self.hz_matrix.shape[1]
        else:
            return 0

    @property
    def num_x_checks(self) -> int:
        """Number of X check constraints."""
        return self.hx_matrix.shape[0]

    @property
    def num_z_checks(self) -> int:
        """Number of Z check constraints."""
        return self.hz_matrix.shape[0]

    def hx_as_dense_array(self) -> NDArray[np.int_]:
        """Return Hx matrix as a dense numpy array (GF(2))."""
        return self.hx_matrix.toarray().astype(np.int_)

    def hz_as_dense_array(self) -> NDArray[np.int_]:
        """Return Hz matrix as a dense numpy array (GF(2))."""
        return self.hz_matrix.toarray().astype(np.int_)

    @computed_field
    @property
    def full_h_matrix(self) -> spmatrix:
        """Return the full H matrix [Hx | Hz] as a sparse matrix."""
        if self.num_z_checks == 0:
            return self.hx_matrix.copy()
        return sparse.hstack([self.hx_matrix, self.hz_matrix], format="csr")

    def full_h_dense(self) -> NDArray[np.int_]:
        """Return the full H matrix [Hx | Hz] as a dense array."""
        hx_dense = self.hx_as_dense_array()
        hz_dense = self.hz_as_dense_array()
        return np.hstack([hx_dense, hz_dense]) if self.num_z_checks > 0 else hx_dense

    @classmethod
    def from_dense_matrix(
        cls,
        name: str,
        hx_matrix: NDArray[np.int_],
        hz_matrix: NDArray[np.int_] | None = None,
    ) -> "ParityCheckMatrix":
        """
        Create a ParityCheckMatrix from dense numpy arrays.

        Automatically converts to sparse CSR format.
        """
        if hx_matrix.ndim != 2:
            raise ValueError("hx_matrix must be 2-dimensional")

        hx_sparse = csr_matrix(hx_matrix, dtype=np.int_)

        if hz_matrix is not None:
            if hz_matrix.ndim != 2:
                raise ValueError("hz_matrix must be 2-dimensional")
            if hz_matrix.shape[1] != hx_matrix.shape[1]:
                raise ValueError(
                    "hx_matrix and hz_matrix must have same number of columns"
                )
            hz_sparse = csr_matrix(hz_matrix, dtype=np.int_)
        else:
            hz_sparse = csr_matrix((0, hx_matrix.shape[1]), dtype=np.int_)

        return cls(
            name=name,
            hx_matrix=hx_sparse,
            hz_matrix=hz_sparse,
        )

    @classmethod
    def from_sparse_matrix(
        cls,
        name: str,
        hx_matrix: spmatrix,
        hz_matrix: spmatrix | None = None,
    ) -> "ParityCheckMatrix":
        """
        Create a ParityCheckMatrix from scipy sparse matrices.

        Automatically converts to CSR format.
        """
        hx_sparse = csr_matrix(hx_matrix, dtype=np.int_)

        if hz_matrix is not None:
            hz_sparse = csr_matrix(hz_matrix, dtype=np.int_)
            if hz_sparse.shape[1] != hx_sparse.shape[1]:
                raise ValueError(
                    "hx_matrix and hz_matrix must have same number of columns"
                )
        else:
            hz_sparse = csr_matrix((0, hx_sparse.shape[1]), dtype=np.int_)

        return cls(
            name=name,
            hx_matrix=hx_sparse,
            hz_matrix=hz_sparse,
        )

    def to_row_sparse_format(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Convert to row-sparse format (List of Lists of column indices).

        Useful for serialization or interfacing with other libraries.
        """
        hx_lol = [
            self.hx_matrix[i].nonzero()[1].tolist() for i in range(self.num_x_checks)
        ]
        hz_lol = [
            self.hz_matrix[i].nonzero()[1].tolist() for i in range(self.num_z_checks)
        ]
        return hx_lol, hz_lol

    def to_tanner_graph(self, name: str | None = None) -> "TannerGraph":
        """Convert parity check matrices to Tanner graph."""
        graph_name = name or f"{self.name} (Tanner)"

        # Create check nodes: X checks followed by Z checks
        check_nodes = []

        # X check nodes
        for i in range(self.num_x_checks):
            check_nodes.append(
                TannerGraphNode(
                    node_type="check",
                    check_type="X",
                    label=f"X{i}",
                    index=i,
                )
            )

        # Z check nodes
        for i in range(self.num_z_checks):
            check_nodes.append(
                TannerGraphNode(
                    node_type="check",
                    check_type="Z",
                    label=f"Z{i}",
                    index=i,
                )
            )

        # Create variable nodes (one per qubit)
        variable_nodes = [
            TannerGraphNode(node_type="variable", label=f"V{j}", index=j)
            for j in range(self.num_variables)
        ]

        # Create edges based on matrix entries
        edges = []

        # Edges from X checks (using sparse matrix row iteration)
        for i in range(self.num_x_checks):
            row = self.hx_matrix.getrow(i)
            for j in row.nonzero()[1]:
                edges.append((check_nodes[i].node_id, variable_nodes[j].node_id))

        # Edges from Z checks
        for i in range(self.num_z_checks):
            row = self.hz_matrix.getrow(i)
            for j in row.nonzero()[1]:
                check_node_idx = self.num_x_checks + i
                edges.append(
                    (check_nodes[check_node_idx].node_id, variable_nodes[j].node_id)
                )

        return TannerGraph(
            name=graph_name,
            check_nodes=check_nodes,
            variable_nodes=variable_nodes,
            edges=edges,
        )


def _list_of_lists_to_csr(
    lol: List[List[int]], num_cols: int | None = None
) -> csr_matrix:
    """Convert row-sparse format (list of lists) to CSR matrix."""
    if not lol:
        return csr_matrix((0, num_cols or 0), dtype=np.int_)

    if num_cols is None:
        max_col = max((max(row) if row else -1) for row in lol)
        num_cols = max_col + 1 if max_col >= 0 else 0

    row_indices = []
    col_indices = []
    for i, row in enumerate(lol):
        for col in row:
            row_indices.append(i)
            col_indices.append(col)

    data = np.ones(len(row_indices), dtype=np.int_)
    return csr_matrix(
        (data, (row_indices, col_indices)), shape=(len(lol), num_cols), dtype=np.int_
    )


class TannerGraphNode(BaseModel):
    """Represents a node in a Tanner graph."""

    node_id: UUID = Field(default_factory=uuid4)
    node_type: Literal["check", "variable"] = Field(
        description="Whether this is a check node or variable node"
    )
    check_type: Literal["X", "Z"] | None = Field(
        default=None,
        description="For check nodes: whether it's an X check (detect Z) or Z check (detect X). None for variable nodes",
    )
    label: str = Field(description="Label/name of the node")
    index: int = Field(description="Index of the check or variable")


class TannerGraph(BaseModel):
    """
    Represents a Tanner graph: bipartite graph representation of a parity check matrix.

    A Tanner graph has two types of nodes:
    - Check nodes: representing stabiliser constraints (X and Z checks)
    - Variable nodes: representing qubits

    Edges connect check nodes to variable nodes based on the parity check matrix.
    """

    name: str = Field(description="Name of the Tanner graph")
    id: UUID = Field(default_factory=uuid4)

    check_nodes: List[TannerGraphNode] = Field(description="List of check nodes")
    variable_nodes: List[TannerGraphNode] = Field(description="List of variable nodes")

    # Edges represented as pairs of node IDs
    edges: List[Tuple[UUID, UUID]] = Field(
        default_factory=list,
        description="List of edges as (check_node_id, variable_node_id) pairs",
    )

    @field_validator("check_nodes")
    @classmethod
    def validate_check_nodes(cls, v: List[TannerGraphNode]) -> List[TannerGraphNode]:
        """Ensure all check nodes have correct type and check_type."""
        for node in v:
            if node.node_type != "check":
                raise ValueError(
                    f"Check node '{node.label}' has wrong type: {node.node_type}"
                )
            if node.check_type not in ("X", "Z"):
                raise ValueError(
                    f"Check node '{node.label}' has invalid check_type: {node.check_type}"
                )
        return v

    @field_validator("variable_nodes")
    @classmethod
    def validate_variable_nodes(cls, v: List[TannerGraphNode]) -> List[TannerGraphNode]:
        """Ensure all variable nodes have correct type."""
        for node in v:
            if node.node_type != "variable":
                raise ValueError(
                    f"Variable node '{node.label}' has wrong type: {node.node_type}"
                )
        return v

    @field_validator("edges")
    @classmethod
    def validate_edges(
        cls, edges: List[Tuple[UUID, UUID]], info
    ) -> List[Tuple[UUID, UUID]]:
        """Validate that all edges connect valid nodes."""
        check_nodes = info.data.get("check_nodes", [])
        variable_nodes = info.data.get("variable_nodes", [])

        check_node_ids: Set[UUID] = {node.node_id for node in check_nodes}
        variable_node_ids: Set[UUID] = {node.node_id for node in variable_nodes}

        for check_id, var_id in edges:
            if check_id not in check_node_ids:
                raise ValueError(f"Edge references unknown check node: {check_id}")
            if var_id not in variable_node_ids:
                raise ValueError(f"Edge references unknown variable node: {var_id}")

        return edges

    def to_parity_check_matrix(self, name: str | None = None) -> "ParityCheckMatrix":
        """Convert Tanner graph to parity check matrices."""
        matrix_name = name or f"{self.name} (Matrix)"

        num_variables = len(self.variable_nodes)

        # Create mapping from node IDs to indices
        var_id_to_idx = {node.node_id: node.index for node in self.variable_nodes}

        # Separate check nodes by type
        x_check_nodes = [n for n in self.check_nodes if n.check_type == "X"]
        z_check_nodes = [n for n in self.check_nodes if n.check_type == "Z"]

        num_x_checks = len(x_check_nodes)
        num_z_checks = len(z_check_nodes)

        # Initialize matrices (row format for sparse construction)
        hx_rows: List[List[int]] = [[] for _ in range(num_x_checks)]
        hz_rows: List[List[int]] = [[] for _ in range(num_z_checks)]

        # Add edges to matrices
        for check_id, var_id in self.edges:
            var_idx = var_id_to_idx[var_id]

            # Find which check node this is
            check_node = next(n for n in self.check_nodes if n.node_id == check_id)

            if check_node.check_type == "X":
                hx_rows[check_node.index].append(var_idx)
            elif check_node.check_type == "Z":
                hz_rows[check_node.index].append(var_idx)

        # Sort each row for consistency
        for row in hx_rows:
            row.sort()
        for row in hz_rows:
            row.sort()

        # Convert to sparse matrices
        hx_sparse = _list_of_lists_to_csr(hx_rows, num_variables)
        hz_sparse = _list_of_lists_to_csr(hz_rows, num_variables)

        return ParityCheckMatrix(
            name=matrix_name,
            hx_matrix=hx_sparse,
            hz_matrix=hz_sparse,
        )


class StabiliserCode(BaseModel):
    """
    Base class for stabiliser codes.

    A stabiliser code is defined by:
    - A set of data qubits (logical information carriers)
    - A set of ancilla qubits (used for syndrome measurements)
    - A set of stabiliser generators
    """

    name: str = Field(description="Name of the stabiliser code")
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")

    data_qubits: List[QuantumBit] = Field(description="List of data qubits in the code")
    ancilla_qubits: List[QuantumBit] = Field(
        description="List of ancilla qubits in the code"
    )
    stabilisers: List[Stabiliser] = Field(
        default_factory=list, description="List of stabiliser generators for the code"
    )
    parity_check_matrix: "ParityCheckMatrix | None" = Field(
        default=None,
        description="Parity check matrix representation (stored if provided, lazy computed otherwise)",
    )
    tanner_graph: "TannerGraph | None" = Field(
        default=None,
        description="Tanner graph representation (stored if provided, lazy computed otherwise)",
    )

    @field_validator("data_qubits")
    @classmethod
    def validate_data_qubits(cls, v: List[QuantumBit]) -> List[QuantumBit]:
        """Ensure data_qubits is not empty and all qubits are data qubits."""
        if not v:
            raise ValueError(
                "data_qubits cannot be empty - at least one data qubit is required"
            )
        for qubit in v:
            if qubit.qubit_type != "data":
                raise ValueError(
                    f"Expected data qubit, got {qubit.qubit_type} for '{qubit.name}'"
                )
        return v

    @field_validator("ancilla_qubits")
    @classmethod
    def validate_ancilla_qubits(cls, v: List[QuantumBit]) -> List[QuantumBit]:
        """Ensure all qubits are ancilla qubits."""
        for qubit in v:
            if qubit.qubit_type != "ancilla":
                raise ValueError(
                    f"Expected ancilla qubit, got {qubit.qubit_type} for '{qubit.name}'"
                )
        return v

    @field_validator("stabilisers")
    @classmethod
    def validate_stabilisers(cls, v: List[Stabiliser], info) -> List[Stabiliser]:
        """
        Validate that all stabilisers reference qubits that exist in the code.
        """
        data_qubits = info.data.get("data_qubits", [])
        ancilla_qubits = info.data.get("ancilla_qubits", [])

        all_qubit_ids: Set[UUID] = {q.id for q in data_qubits + ancilla_qubits}

        for stab in v:
            # Check that ancilla qubit exists
            if stab.ancilla_qubit.id not in all_qubit_ids:
                raise ValueError(
                    f"Stabiliser '{stab.name}' references unknown ancilla qubit "
                    f"'{stab.ancilla_qubit.name}'"
                )
            # Check that all data qubits exist
            for dq in stab.data_qubits:
                if dq.id not in all_qubit_ids:
                    raise ValueError(
                        f"Stabiliser '{stab.name}' references unknown data qubit "
                        f"'{dq.name}'"
                    )

        return v

    @computed_field
    @property
    def all_qubits(self) -> List[QuantumBit]:
        """Return all qubits (data + ancilla) in the code."""
        return self.data_qubits + self.ancilla_qubits

    @computed_field
    @property
    def n(self) -> int:
        """Return the total number of qubits in the code."""
        return len(self.data_qubits) + len(self.ancilla_qubits)

    @cached_property
    def k(self) -> int:
        """Return the number of logical qubits encoded in the code (cached on first access)."""
        # TODO: Implement k computation
        pass

    @cached_property
    def exact_distance(self) -> int | None:
        """Return the exact distance of the code (cached on first access)."""
        # TODO: Implement exact distance computation
        pass

    @cached_property
    def estimated_distance(self) -> int | None:
        """Return the estimated distance of the code (cached on first access)."""
        # TODO: Implement estimated distance computation
        pass

    @cached_property
    def pcm(self) -> "ParityCheckMatrix":
        """Return parity check matrix, computing from stabilisers if not stored."""
        if self.parity_check_matrix is not None:
            return self.parity_check_matrix
        
        for q in self.ancilla_qubits:
            for dq in self.data_qubits:
                if 
        raise NotImplementedError(
            "PCM computation from stabilisers not yet implemented"
        )

    @cached_property
    def graph(self) -> "TannerGraph":
        """Return Tanner graph, computing from PCM if not stored."""
        if self.tanner_graph is not None:
            return self.tanner_graph
        # Compute from PCM
        return self.pcm.to_tanner_graph(name=f"{self.name} (Tanner)")

    def get_num_stabilisers(self) -> int:
        """Return the number of stabiliser generators."""
        return len(self.stabilisers)

    @classmethod
    def from_stabilisers(
        cls,
        name: str,
        stabilisers: List[Stabiliser],
    ) -> "StabiliserCode":
        """
        Create a StabiliserCode from a list of Stabiliser objects.

        Data and ancilla qubits must be provided and must exactly match the qubits
        referenced in the stabilisers.

        Raises:
            ValueError: If provided qubits don't match stabiliser requirements.
        """
        # Extract unique qubits from stabilisers
        extracted_data_qubits = list(
            {dq for stab in stabilisers for dq in stab.data_qubits}
        )
        extracted_ancilla_qubits = list({stab.ancilla_qubit for stab in stabilisers})

        return cls(
            name=name,
            data_qubits=extracted_data_qubits,
            ancilla_qubits=extracted_ancilla_qubits,
            stabilisers=stabilisers,
        )

    @classmethod
    def from_parity_check_matrices(
        cls,
        name: str,
        hx_matrix: List[List[int]],
        hz_matrix: List[List[int]],
        data_qubits: List[QuantumBit],
        ancilla_qubits: List[QuantumBit],
    ) -> "StabiliserCode":
        """
        Create a StabiliserCode from Hx and Hz parity check matrices.

        Data and ancilla qubits must be provided and must match the matrix dimensions:
        - data_qubits count must equal the number of matrix columns (variables)
        - ancilla_qubits count must equal the total number of checks (rows of Hx + rows of Hz)

        Raises:
            ValueError: If qubit counts don't match matrix dimensions.
        """
        # Create PCM (num_variables will be computed from matrix columns)
        pcm = ParityCheckMatrix(
            name=f"{name} PCM",
            hx_matrix=hx_matrix,
            hz_matrix=hz_matrix,
        )

        # Validate qubit counts
        if len(data_qubits) != pcm.num_variables:
            raise ValueError(
                f"Expected {pcm.num_variables} data qubits for matrix with {pcm.num_variables} columns, "
                f"but got {len(data_qubits)}"
            )

        expected_ancilla = pcm.num_x_checks + pcm.num_z_checks
        if len(ancilla_qubits) != expected_ancilla:
            raise ValueError(
                f"Expected {expected_ancilla} ancilla qubits for matrix with "
                f"{pcm.num_x_checks} X-checks and {pcm.num_z_checks} Z-checks, "
                f"but got {len(ancilla_qubits)}"
            )

        # Build stabilisers by looping through rows of full H matrix
        stabilisers: List[Stabiliser] = []

        # Loop through checks
        for row_idx in range(pcm.num_x_checks):
            row_x = pcm.hx_matrix.getrow(row_idx)
            row_z = pcm.hz_matrix.getrow(row_idx)
            col_indices_x = row_x.nonzero()[1].tolist()
            col_indices_z = row_z.nonzero()[1].tolist()

            # Build Pauli string: X if only in x, Z if only in z, Y if in both, I otherwise
            pauli_string = ""
            qubits_in_check = []
            for qubit_idx in range(pcm.num_variables):
                in_x = qubit_idx in col_indices_x
                in_z = qubit_idx in col_indices_z

                if in_x and in_z:
                    pauli_string += "Y"
                    qubits_in_check.append(data_qubits[qubit_idx])
                elif in_x:
                    pauli_string += "X"
                    qubits_in_check.append(data_qubits[qubit_idx])
                elif in_z:
                    pauli_string += "Z"
                    qubits_in_check.append(data_qubits[qubit_idx])

            pauli_stab = PauliOperator(pauli_string=pauli_string)

            stab = Stabiliser(
                name=f"{name}_check_{row_idx}",
                ancilla_qubit=ancilla_qubits[row_idx],
                data_qubits=qubits_in_check,
                pauli_operator=pauli_stab,
            )
            stabilisers.append(stab)

        return cls(
            name=name,
            data_qubits=data_qubits,
            ancilla_qubits=ancilla_qubits,
            stabilisers=stabilisers,
            parity_check_matrix=pcm,
        )

    @classmethod
    def from_tanner_graph(
        cls,
        name: str,
        tanner_graph: TannerGraph,
        data_qubits: List[QuantumBit],
        ancilla_qubits: List[QuantumBit],
    ) -> "StabiliserCode":
        """
        Create a StabiliserCode from a Tanner graph.

        Data and ancilla qubits must be provided and must match the graph structure:
        - data_qubits count must equal the number of variable nodes
        - ancilla_qubits count must equal the number of check nodes

        Raises:
            ValueError: If provided qubits don't match graph structure.
        """
        num_variables = len(tanner_graph.variable_nodes)
        num_checks = len(tanner_graph.check_nodes)

        # Validate data qubits
        if len(data_qubits) != num_variables:
            raise ValueError(
                f"Expected {num_variables} data qubits for graph with {num_variables} variables, "
                f"but got {len(data_qubits)}"
            )

        # Validate ancilla qubits
        if len(ancilla_qubits) != num_checks:
            raise ValueError(
                f"Expected {num_checks} ancilla qubits for graph with {num_checks} check nodes, "
                f"but got {len(ancilla_qubits)}"
            )

        return cls(
            name=name,
            data_qubits=data_qubits,
            ancilla_qubits=ancilla_qubits,
            stabilisers=[],
            tanner_graph=tanner_graph,
        )
