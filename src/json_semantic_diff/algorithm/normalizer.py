"""Per-level similarity normalizer for the STED algorithm.

Implements the STED paper normalization formula that converts a raw edit
distance (d_matched) into a similarity score in [0, 1], penalizing
unmatched children via lambda_ * |n_left - n_right|.

Formula::

    STED(T1, T2) = 1 - min(1, [d_matched + lambda_ * |n_left - n_right|]
                                / max(n_left, n_right, 1))

The ``max(..., 1)`` guard in the denominator prevents ZeroDivisionError
when both child lists are empty (n_left == n_right == 0).
"""

from __future__ import annotations


def normalize_similarity(
    d_matched: float,
    n_left: int,
    n_right: int,
    lambda_: float,
) -> float:
    """Normalize a raw child-matching distance to a [0, 1] similarity score.

    Args:
        d_matched: Sum of costs for the matched child pairs (raw distance from
            Hungarian or DP alignment).
        n_left:    Number of children in the left (A) node.
        n_right:   Number of children in the right (B) node.
        lambda_:   Penalty multiplier for unmatched children (>=0).
                   Use ``STEDConfig.lambda_unmatched``.

    Returns:
        Float in [0.0, 1.0] — 1.0 means identical child sets, 0.0 means
        maximally different.
    """
    unmatched_penalty = lambda_ * abs(n_left - n_right)
    total_cost = d_matched + unmatched_penalty
    denominator = max(n_left, n_right, 1)
    return 1.0 - min(1.0, total_cost / denominator)
