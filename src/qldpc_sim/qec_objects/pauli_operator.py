from functools import cached_property
from typing import List
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field


class PauliOperator(BaseModel):
    """
    Represents a Pauli operator in two equivalent representations:
    - String format: X, Y, Z, I characters (e.g., "XYZI")
    - Matrix format: 2-row numpy array where:
      * Row 1: X component
      * Row 2: Z component

    One representation must be provided, the other is computed on demand.
    """

    pauli_string: str | None = Field(
        default=None, description="Pauli operators as string (X, Y, Z, I)"
    )
    pauli_matrix: NDArray[np.int_] | None = Field(
        default=None, description="Pauli operators as 2-row numpy array"
    )

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def check_one_present(self):
        """Ensure at least one of pauli_string or pauli_matrix is provided."""
        if self.pauli_string is None and self.pauli_matrix is None:
            raise ValueError("Either 'pauli_string' or 'pauli_matrix' must be provided")
        return self

    @field_validator("pauli_string", mode="before")
    @classmethod
    def validate_pauli_string(cls, v: str | None) -> str | None:
        """Validate pauli_string format if provided."""
        if v is None:
            return v
        if not all(char in "XYZI" for char in v):
            invalid_chars = set(v) - {"X", "Y", "Z", "I"}
            raise ValueError(
                f"pauli_string must only contain X, Y, Z, I. "
                f"Found invalid characters: {invalid_chars}"
            )
        return v

    @field_validator("pauli_matrix", mode="before")
    @classmethod
    def validate_pauli_matrix(
        cls, v: NDArray[np.int_] | List[List[int]] | None
    ) -> NDArray[np.int_] | None:
        """Validate and convert pauli_matrix to numpy array if provided."""
        if v is None:
            return v

        # Convert to numpy array if needed
        arr = np.array(v, dtype=np.int_)

        if arr.ndim != 2:
            raise ValueError(
                f"Matrix format must be 2-dimensional, got shape {arr.shape}"
            )
        if arr.shape[0] != 2:
            raise ValueError(
                f"Matrix format must have exactly 2 rows, got {arr.shape[0]}"
            )
        # Validate all entries are 0 or 1
        if not np.all((arr == 0) | (arr == 1)):
            invalid_positions = np.where((arr != 0) & (arr != 1))
            raise ValueError(
                f"Matrix entries must be 0 or 1. "
                f"Found invalid values at positions: {list(zip(invalid_positions[0], invalid_positions[1]))}"
            )
        return arr

    @computed_field
    @property
    def get_string(self) -> str:
        """Return string representation (compute if needed)."""
        if self.pauli_string is not None:
            return self.pauli_string

        # Convert from matrix format
        x_row = self.pauli_matrix[0]
        z_row = self.pauli_matrix[1]
        result = ""
        for x, z in zip(x_row, z_row):
            if x == 1 and z == 1:
                result += "Y"
            elif x == 1:
                result += "X"
            elif z == 1:
                result += "Z"
            else:
                result += "I"
        return result

    @computed_field
    @property
    def get_matrix(self) -> NDArray[np.int_]:
        """Return matrix representation (compute if needed)."""
        if self.pauli_matrix is not None:
            return self.pauli_matrix

        # Convert from string format
        x_row = []
        z_row = []
        for op in self.pauli_string:
            if op in ("X", "Y"):
                x_row.append(1)
            else:
                x_row.append(0)
            if op in ("Z", "Y"):
                z_row.append(1)
            else:
                z_row.append(0)
        return np.array([x_row, z_row], dtype=np.int_)

    @property
    def weight(self) -> int:
        """Return the number of non-identity operators."""
        if self.pauli_string is not None:
            return sum(1 for op in self.pauli_string if op != "I")
        else:
            # For matrix format: count columns where at least one entry is 1
            return int(
                np.sum((self.pauli_matrix[0] == 1) | (self.pauli_matrix[1] == 1))
            )

    @property
    def is_X_only(self) -> bool:
        """Check if the Pauli operator consists only of X and I."""
        pauli_str = self.get_string
        return all(op in ("X", "I") for op in pauli_str)

    @property
    def is_Z_only(self) -> bool:
        """Check if the Pauli operator consists only of Z and I."""
        pauli_str = self.get_string
        return all(op in ("Z", "I") for op in pauli_str)
