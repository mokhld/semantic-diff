"""HungarianMatcher: optimal bipartite assignment with np.inf guard.

Wraps scipy's ``linear_sum_assignment`` so that infinite-cost cells never
reach the solver (which would raise ``ValueError``).  After assignment,
pairs that landed on originally-infinite positions are filtered out.

Guard value formula: ``finite_max * 2.0 + 1.0``
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]


def hungarian_match(
    cost_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal bipartite assignment with np.inf guard.

    Args:
        cost_matrix: 2-D cost matrix of shape ``(m, n)``.  May contain
            ``np.inf`` to mark forbidden assignments.

    Returns:
        Tuple ``(row_ind, col_ind)`` of 1-D integer arrays giving the
        optimal assignment, with any pair whose *original* cost was
        infinite removed.  Empty arrays are returned when no valid
        assignment exists.
    """
    if cost_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    cost = np.asarray(cost_matrix, dtype=float)

    inf_mask = np.isinf(cost)

    # All-inf: no valid assignment
    if inf_mask.all():
        return np.array([], dtype=int), np.array([], dtype=int)

    # Replace inf with a guard value that dominates all finite costs
    if inf_mask.any():
        finite_max = float(cost[~inf_mask].max())
        guard_value = finite_max * 2.0 + 1.0
        cost = np.where(inf_mask, guard_value, cost)

    row_ind, col_ind = linear_sum_assignment(cost)

    # Filter out pairs whose original cost was infinite
    if inf_mask.any():
        original = np.asarray(cost_matrix, dtype=float)
        keep = np.isfinite(original[row_ind, col_ind])
        row_ind = row_ind[keep]
        col_ind = col_ind[keep]

    return row_ind, col_ind
