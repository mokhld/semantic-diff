"""StaticBackend: Levenshtein edit-distance similarity on normalized JSON keys.

Uses KeyNormalizer to convert keys across all naming conventions (camelCase,
PascalCase, snake_case, kebab-case) to a canonical lowercase form before
computing edit-distance similarity.  This provides useful key discrimination
without any ML dependencies beyond numpy.

This backend satisfies the EmbeddingBackend Protocol structurally (via the
``embed`` method stub) without inheriting from it.
"""

from __future__ import annotations

import numpy as np

from json_semantic_diff.tree.normalizer import KeyNormalizer

# Module-level singleton — KeyNormalizer is stateless, safe to share.
_normalizer = KeyNormalizer()


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.

    Uses a space-optimized rolling-row dynamic-programming implementation.
    The shorter string is always placed on the inner loop to minimise
    the allocation size.

    Args:
        a: First string.
        b: Second string.

    Returns:
        The minimum number of single-character edits (insertions, deletions,
        or substitutions) required to transform ``a`` into ``b``.
    """
    # Early exit for identical strings
    if a == b:
        return 0

    # Swap so that `b` is the shorter string (inner loop / row allocation)
    if len(a) < len(b):
        a, b = b, a

    # Empty-string edge case: distance equals the length of the other string
    if len(b) == 0:
        return len(a)

    # Initialize the rolling row for DP (only one row needed at a time)
    prev_row = list(range(len(b) + 1))

    for i, ch_a in enumerate(a):
        curr_row = [i + 1] + [0] * len(b)
        for j, ch_b in enumerate(b):
            insert_cost = curr_row[j] + 1
            delete_cost = prev_row[j + 1] + 1
            replace_cost = prev_row[j] + (0 if ch_a == ch_b else 1)
            curr_row[j + 1] = min(insert_cost, delete_cost, replace_cost)
        prev_row = curr_row

    return prev_row[len(b)]


class StaticBackend:
    """Levenshtein-based similarity backend requiring only numpy.

    Satisfies the ``EmbeddingBackend`` Protocol structurally — no inheritance
    needed.  Normalizes keys via ``KeyNormalizer`` before computing edit-distance
    similarity, so "user_name", "userName", "UserName", and "user-name" all
    score 1.0 against each other.

    Example::

        from json_semantic_diff.backends import StaticBackend

        backend = StaticBackend()
        backend.similarity("user_name", "userName")   # 1.0
        backend.similarity("user_name", "address")    # < 0.5
    """

    def embed(self, strings: list[str]) -> np.ndarray:
        """Return a stub embedding array satisfying the EmbeddingBackend Protocol.

        Normalizes each string via KeyNormalizer and returns an (N, 1) float64
        array of normalized string lengths.  This is a minimal Protocol stub —
        use ``similarity`` for actual key comparison.

        Args:
            strings: List of raw key strings to embed.

        Returns:
            Shape ``(N, 1)`` float64 numpy array.
        """
        lengths = [float(len(_normalizer.normalize(s))) for s in strings]
        return np.array(lengths, dtype=np.float64).reshape(-1, 1)

    def similarity(self, a: str, b: str) -> float:
        """Compute normalized Levenshtein similarity between two JSON keys.

        Both keys are normalized via ``KeyNormalizer`` before comparison, so
        naming-convention differences (camelCase vs snake_case, etc.) are ignored.

        Similarity is defined as::

            1.0 - levenshtein(norm_a, norm_b) / max(len(norm_a), len(norm_b), 1)

        The ``max(..., 1)`` guard prevents ZeroDivisionError when both strings
        are empty (both normalize to "" — distance 0, result 1.0).

        Args:
            a: First raw JSON key string.
            b: Second raw JSON key string.

        Returns:
            Float in [0.0, 1.0] — 1.0 means identical after normalization,
            0.0 means maximally different (every character replaced).
        """
        norm_a = _normalizer.normalize(a)
        norm_b = _normalizer.normalize(b)
        distance = _levenshtein_distance(norm_a, norm_b)
        denom = max(len(norm_a), len(norm_b), 1)
        return 1.0 - distance / denom
