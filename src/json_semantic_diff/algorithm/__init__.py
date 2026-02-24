"""algorithm subpackage — public API for the STED algorithm.

Provides the core semantic similarity algorithm, its configuration, and
array-comparison mode control.  Import from this module (not from
sub-modules directly) to stay on the stable public interface.

Example::

    from json_semantic_diff.algorithm import STEDAlgorithm, STEDConfig, ArrayComparisonMode
    from json_semantic_diff.backends import StaticBackend

    algo = STEDAlgorithm(backend=StaticBackend())
    score = algo.compute({"user_name": "Alice"}, {"userName": "Alice"})
    # score > 0.85  (naming-convention equivalents)
"""

from __future__ import annotations

from json_semantic_diff.algorithm.config import ArrayComparisonMode, STEDConfig
from json_semantic_diff.algorithm.sted import STEDAlgorithm

__all__ = ["ArrayComparisonMode", "STEDAlgorithm", "STEDConfig"]
