"""ComparisonResult dataclass for semantic comparison output.

This module provides the rich result type returned by compare() calls.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ComparisonResult"]


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Rich result of a compare() call.

    Attributes:
        similarity_score: Normalised similarity in [0.0, 1.0].  1.0 is identical.
        matched_pairs: Sequence of (left_path, right_path) JSON Pointer pairs for
            KEY nodes that were matched across the two documents.
        key_mappings: Mapping from raw left key name to raw right key name for
            each matched KEY pair.
        unmatched_left: JSON Pointer paths of KEY nodes present in the left
            document that had no counterpart in the right document.
        unmatched_right: JSON Pointer paths of KEY nodes present in the right
            document that had no counterpart in the left document.
        computation_time_ms: Wall-clock duration of the comparison in milliseconds.
    """

    similarity_score: float
    matched_pairs: list[tuple[str, str]]
    key_mappings: dict[str, str]
    unmatched_left: list[str]
    unmatched_right: list[str]
    computation_time_ms: float
