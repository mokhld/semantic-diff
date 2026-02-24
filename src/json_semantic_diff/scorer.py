"""ConsistencyScorer: measures generator stability across multiple JSON samples.

The consistency score quantifies how stable a JSON generator is when called
multiple times. It penalizes both low average similarity (generator produces
different outputs) and high variance (generator is erratic).

Formula:
    pairwise = [comparator.compare(docs[i], docs[j]).similarity_score
                for all (i, j) pairs with i < j]
    score = max(0.0, mean(pairwise) - std(pairwise))

This means:
- Identical documents: mean=1.0, std=0.0 -> score=1.0
- Consistently mediocre: mean=0.6, std=0.0 -> score=0.6
- Erratic (high variance): mean=0.6, std=0.4 -> score=max(0, 0.2)=0.2
- Structurally different: mean~0.0, std~0.0 -> score~0.0
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from json_semantic_diff.algorithm.config import STEDConfig
from json_semantic_diff.comparator import STEDComparator

__all__ = ["ConsistencyScorer"]


class ConsistencyScorer:
    """Measures generator stability across multiple JSON samples.

    Creates a single ``STEDComparator`` instance that is reused across all
    pairwise comparisons, enabling embedding cache reuse across the entire
    document set.

    The formula ``max(0, mean(pairwise) - std(pairwise))`` penalizes erratic
    generators even when their average similarity might appear acceptable.

    Example::

        from json_semantic_diff.scorer import ConsistencyScorer

        scorer = ConsistencyScorer()
        docs = [{"user_name": "Alice"}, {"user_name": "Alice"}, {"user_name": "Alice"}]
        print(scorer.compute(docs))  # 1.0

        mixed = [{"a": 1}, {"z": 99}, {"a": 1}]
        print(scorer.compute(mixed))  # < 0.5
    """

    def __init__(
        self,
        config: STEDConfig | None = None,
        max_cache_size: int = 512,
    ) -> None:
        """Initialise the scorer with a single reusable comparator.

        Args:
            config: Algorithm hyper-parameters forwarded to ``STEDComparator``.
                Defaults to ``STEDConfig()`` when None.
            max_cache_size: Maximum embedding cache size forwarded to
                ``STEDComparator``.  Defaults to 512.
        """
        self._comparator = STEDComparator(config=config, max_cache_size=max_cache_size)

    def compute(self, docs: list[Any]) -> float:
        """Compute the consistency score for a list of JSON documents.

        Args:
            docs: List of JSON values (dicts, lists, scalars). Order does not
                matter — all C(N, 2) unique pairs are evaluated.

        Returns:
            A float in [0.0, 1.0].
            - Returns 1.0 for empty lists or single-document lists (trivially
              consistent — no pairs to compare).
            - Returns 1.0 when all pairwise scores are 1.0 (identical generator).
            - Returns lower values when the generator produces diverse or erratic
              outputs.
        """
        n = len(docs)
        if n <= 1:
            return 1.0

        pairwise_scores = [
            self._comparator.compare(docs[i], docs[j]).similarity_score
            for i, j in itertools.combinations(range(n), 2)
        ]

        scores = np.array(pairwise_scores, dtype=float)
        mean = float(np.mean(scores))
        std = float(np.std(scores))  # population std (ddof=0)
        return float(np.clip(mean - std, 0.0, 1.0))
