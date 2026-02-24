"""STEDConfig and ArrayComparisonMode for STED algorithm configuration.

STEDConfig is a frozen (immutable) dataclass holding the algorithm
parameters.  ArrayComparisonMode selects how arrays are compared:
ordered (positional), unordered (set-like), or auto-detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto


class ArrayComparisonMode(StrEnum):
    """How to compare JSON arrays during STED computation.

    - ORDERED:   Positional alignment via DP sequence edit distance.
    - UNORDERED: Set-like matching via Hungarian algorithm.
    - AUTO:      Infer from array content (scalars → unordered, objects → ordered).
    """

    ORDERED = auto()
    UNORDERED = auto()
    AUTO = auto()


@dataclass(frozen=True, slots=True)
class STEDConfig:
    """Immutable configuration for the STED algorithm.

    Attributes:
        w_s: Structural weight in [0, 1].
        w_c: Content weight in [0, 1].  Must satisfy w_s + w_c ≈ 1.0.
        lambda_unmatched: Penalty multiplier for unmatched children (≥ 0).
        array_comparison_mode: How arrays are compared.
        type_coercion: When True, numeric strings are coerced to numbers before
            content comparison (e.g. "123" == 123 -> distance 0.0).  Default False.
        null_equals_missing: When True, a JSON null value is treated as equivalent
            to a missing key.  Default False.
    """

    w_s: float = 0.5
    w_c: float = 0.5
    lambda_unmatched: float = 0.1
    array_comparison_mode: ArrayComparisonMode = ArrayComparisonMode.ORDERED
    type_coercion: bool = False
    null_equals_missing: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.w_s <= 1.0:
            msg = f"w_s must be in [0, 1], got {self.w_s}"
            raise ValueError(msg)
        if not 0.0 <= self.w_c <= 1.0:
            msg = f"w_c must be in [0, 1], got {self.w_c}"
            raise ValueError(msg)
        if abs(self.w_s + self.w_c - 1.0) >= 1e-9:
            msg = f"w_s + w_c must sum to 1.0, got {self.w_s + self.w_c}"
            raise ValueError(msg)
        if self.lambda_unmatched < 0.0:
            msg = f"lambda_unmatched must be >= 0.0, got {self.lambda_unmatched}"
            raise ValueError(msg)
