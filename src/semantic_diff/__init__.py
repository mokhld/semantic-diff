"""Semantic diff - structural similarity scoring for JSON documents."""

from __future__ import annotations

from semantic_diff.algorithm.config import ArrayComparisonMode, STEDConfig
from semantic_diff.api import (
    compare,
    consistency_score,
    is_equivalent,
    similarity_score,
)
from semantic_diff.comparator import STEDComparator
from semantic_diff.result import ComparisonResult

__version__: str = "0.1.0"
__all__: list[str] = [
    "ArrayComparisonMode",
    "ComparisonResult",
    "STEDComparator",
    "STEDConfig",
    "compare",
    "consistency_score",
    "is_equivalent",
    "similarity_score",
]
