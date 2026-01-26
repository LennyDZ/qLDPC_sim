from pydantic import BaseModel, Field


class ExperimentWeight(BaseModel):
    """
    Represents the weight of an experiment in a QEC simulation.

    Attributes:
    -
    """

    cx_count: int = Field(
        default=0, description="Weight associated with CX gate operations"
    )
    h_count: int = Field(
        default=0, description="Weight associated with H gate operations"
    )
    meas_count: int = Field(
        default=0, description="Weight associated with measurement operations"
    )

    syndrome_extraction_weight: int = Field(
        default=0, description="Weight associated with syndrome extraction rounds"
    )

    def _add_counts(
        self, other: "ExperimentWeight", in_place: bool = False
    ) -> "ExperimentWeight":
        if in_place:
            self.cx_count += other.cx_count
            self.h_count += other.h_count
            self.meas_count += other.meas_count
            self.syndrome_extraction_weight += other.syndrome_extraction_weight
            return self
        else:
            # return a new object
            return ExperimentWeight(
                cx_count=self.cx_count + other.cx_count,
                h_count=self.h_count + other.h_count,
                meas_count=self.meas_count + other.meas_count,
                syndrome_extraction_weight=self.syndrome_extraction_weight
                + other.syndrome_extraction_weight,
            )

    def __add__(self, other: "ExperimentWeight") -> "ExperimentWeight":
        """Support + operator (returns new object)."""
        return self._add_counts(other, in_place=False)

    def __iadd__(self, other: "ExperimentWeight") -> "ExperimentWeight":
        """Support += operator (modifies in-place)."""
        return self._add_counts(other, in_place=True)
