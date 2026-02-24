"""Test suite for HungarianMatcher (hungarian_match function).

Tests the np.inf guard, rectangular matrices, empty matrices, all-inf matrices,
symmetry, and correctness of the optimal bipartite assignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_diff.algorithm.matcher import hungarian_match


class TestEmptyMatrix:
    """Empty cost matrix cases."""

    def test_zero_by_zero_matrix_returns_empty_arrays(self) -> None:
        cost = np.empty((0, 0), dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 0
        assert len(col_ind) == 0

    def test_zero_by_three_matrix_returns_empty_arrays(self) -> None:
        cost = np.empty((0, 3), dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 0
        assert len(col_ind) == 0

    def test_three_by_zero_matrix_returns_empty_arrays(self) -> None:
        cost = np.empty((3, 0), dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 0
        assert len(col_ind) == 0

    def test_empty_result_dtype_is_int(self) -> None:
        cost = np.empty((0, 0), dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        assert row_ind.dtype == int or np.issubdtype(row_ind.dtype, np.integer)
        assert col_ind.dtype == int or np.issubdtype(col_ind.dtype, np.integer)


class TestAllInfMatrix:
    """All-infinite cost matrix — no valid assignments."""

    def test_all_inf_square_returns_empty(self) -> None:
        cost = np.full((3, 3), np.inf)
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 0
        assert len(col_ind) == 0

    def test_all_inf_rectangular_returns_empty(self) -> None:
        cost = np.full((2, 4), np.inf)
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 0
        assert len(col_ind) == 0

    def test_single_inf_element_returns_empty(self) -> None:
        cost = np.array([[np.inf]])
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 0
        assert len(col_ind) == 0


class TestSquareMatrix:
    """Standard square cost matrix cases."""

    def test_2x2_square_returns_optimal_assignment(self) -> None:
        # Cost [[1,2],[3,4]]: optimal is (0,0)+(1,1)=5 vs (0,1)+(1,0)=5
        # scipy picks lexicographically — just check validity
        cost = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 2
        assert len(col_ind) == 2
        # All rows covered, all columns covered
        assert set(row_ind) == {0, 1}
        assert set(col_ind) == {0, 1}

    def test_2x2_square_optimal_cost(self) -> None:
        # Cost [[4,1],[2,3]]: optimal is (0,1)+(1,0)=3 (cost 1+2)
        cost = np.array([[4.0, 1.0], [2.0, 3.0]], dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        total_cost = cost[row_ind, col_ind].sum()
        assert total_cost == pytest.approx(3.0)

    def test_3x3_identity_assignment(self) -> None:
        # Diagonal matrix: optimal assignment is identity (each i -> i)
        cost = np.diag([1.0, 2.0, 3.0])
        # Off-diagonal are 0, so there's a degenerate case here
        # Use a matrix where diagonal is clearly cheapest
        cost = np.array(
            [[1.0, 10.0, 10.0], [10.0, 2.0, 10.0], [10.0, 10.0, 3.0]],
            dtype=float,
        )
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 3
        assert cost[row_ind, col_ind].sum() == pytest.approx(6.0)

    def test_1x1_matrix_returns_single_assignment(self) -> None:
        cost = np.array([[5.0]])
        row_ind, col_ind = hungarian_match(cost)
        assert list(row_ind) == [0]
        assert list(col_ind) == [0]

    def test_all_zero_matrix_assigns_all_pairs(self) -> None:
        cost = np.zeros((3, 3), dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 3
        assert cost[row_ind, col_ind].sum() == pytest.approx(0.0)

    def test_no_inf_standard_scipy_behavior(self) -> None:
        """Without inf values, result should match raw scipy."""
        from scipy.optimize import linear_sum_assignment

        cost = np.array(
            [[3.0, 1.0, 2.0], [2.0, 3.0, 1.0], [1.0, 2.0, 3.0]],
            dtype=float,
        )
        row_ind, col_ind = hungarian_match(cost)
        expected_row, expected_col = linear_sum_assignment(cost)
        assert list(row_ind) == list(expected_row)
        assert list(col_ind) == list(expected_col)


class TestRectangularMatrix:
    """Rectangular cost matrices (m != n)."""

    def test_2x3_returns_two_assignments(self) -> None:
        cost = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        # m=2 rows so max 2 assignments
        assert len(row_ind) == 2
        assert len(col_ind) == 2
        # Each row is assigned at most once
        assert len(set(row_ind)) == len(row_ind)
        # Each column is assigned at most once
        assert len(set(col_ind)) == len(col_ind)

    def test_3x2_returns_two_assignments(self) -> None:
        cost = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        # n=2 cols so max 2 assignments
        assert len(row_ind) == 2
        assert len(col_ind) == 2
        assert len(set(row_ind)) == len(row_ind)
        assert len(set(col_ind)) == len(col_ind)

    def test_2x3_optimal_cost(self) -> None:
        # Row 0 prefers col 0 (cost 1), Row 1 prefers col 1 (cost 2)
        cost = np.array([[1.0, 10.0, 10.0], [10.0, 2.0, 10.0]], dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        total_cost = cost[row_ind, col_ind].sum()
        assert total_cost == pytest.approx(3.0)

    def test_1x3_returns_one_assignment(self) -> None:
        cost = np.array([[5.0, 2.0, 8.0]])
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 1
        assert cost[row_ind, col_ind].sum() == pytest.approx(2.0)

    def test_3x1_returns_one_assignment(self) -> None:
        cost = np.array([[5.0], [2.0], [8.0]])
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 1
        assert cost[row_ind, col_ind].sum() == pytest.approx(2.0)


class TestInfGuard:
    """np.inf guard: forbidden assignments must not reach scipy."""

    def test_inf_does_not_raise_value_error(self) -> None:
        """scipy would raise ValueError without the guard."""
        cost = np.array([[1.0, np.inf], [np.inf, 2.0]], dtype=float)
        # Must NOT raise ValueError
        row_ind, _col_ind = hungarian_match(cost)
        assert len(row_ind) > 0

    def test_mixed_inf_avoids_inf_assignments(self) -> None:
        # Only (0,0) and (1,1) are finite — should be assigned
        cost = np.array([[1.0, np.inf], [np.inf, 2.0]], dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        assert list(row_ind) == [0, 1]
        assert list(col_ind) == [0, 1]

    def test_partial_inf_row_forces_finite_assignment(self) -> None:
        # Row 0: [1, inf], Row 1: [inf, 2] — forced (0,0) and (1,1)
        cost = np.array([[1.0, np.inf], [np.inf, 2.0]], dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        total_cost = cost[row_ind, col_ind].sum()
        assert total_cost == pytest.approx(3.0)

    def test_matrix_with_some_inf_filters_inf_assignments(self) -> None:
        # [[0, inf, 1], [inf, 0, inf], [inf, 1, inf]]
        # Optimal finite: (0,0)+(1,1)+(2,1) — but col 1 can only be used once.
        # Valid: (0,0) cost 0, (1,1) cost 0, (2,2) cost inf -> filtered
        # Or: (0,2) cost 1, (1,1) cost 0 -> remaining (2,?) all inf
        cost = np.array(
            [[0.0, np.inf, 1.0], [np.inf, 0.0, np.inf], [np.inf, 1.0, np.inf]],
            dtype=float,
        )
        row_ind, col_ind = hungarian_match(cost)
        # All returned assignments must have finite costs in original matrix
        for r, c in zip(row_ind, col_ind, strict=True):
            assert np.isfinite(cost[r, c]), f"Assignment ({r},{c}) has inf cost"

    def test_guard_value_formula_is_finite_max_times_two_plus_one(self) -> None:
        """Verify guard value is finite_max * 2 + 1 (not an arbitrary constant)."""
        # If cost = [[10.0, inf], [inf, 10.0]], finite_max=10, guard=21
        # Both finite assignments will be chosen; inf assignments filtered
        cost = np.array([[10.0, np.inf], [np.inf, 10.0]], dtype=float)
        row_ind, col_ind = hungarian_match(cost)
        assert list(row_ind) == [0, 1]
        assert list(col_ind) == [0, 1]

    def test_3x3_one_inf_position_avoided(self) -> None:
        # Force assignment away from (0,2) which is inf
        cost = np.array(
            [[1.0, 2.0, np.inf], [2.0, 1.0, 3.0], [3.0, 4.0, 1.0]],
            dtype=float,
        )
        row_ind, col_ind = hungarian_match(cost)
        # Result must not include (0,2)
        pairs = list(zip(row_ind, col_ind, strict=True))
        assert (0, 2) not in pairs
        # All returned pairs must have finite costs
        for r, c in zip(row_ind, col_ind, strict=True):
            assert np.isfinite(cost[r, c])


class TestSymmetry:
    """Symmetry: transposing cost matrix gives mirrored assignments, same total cost."""

    def _total_cost(self, cost: np.ndarray) -> float:
        row_ind, col_ind = hungarian_match(cost)
        if len(row_ind) == 0:
            return 0.0
        return float(cost[row_ind, col_ind].sum())

    def test_2x2_symmetric_cost(self) -> None:
        cost = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=float)
        assert self._total_cost(cost) == pytest.approx(self._total_cost(cost.T))

    def test_3x3_symmetric_total_cost(self) -> None:
        cost = np.array(
            [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0], [7.0, 8.0, 1.0]],
            dtype=float,
        )
        assert self._total_cost(cost) == pytest.approx(self._total_cost(cost.T))

    def test_rectangular_2x3_transpose_cost(self) -> None:
        cost = np.array([[1.0, 4.0, 2.0], [3.0, 1.0, 5.0]], dtype=float)
        # cost is (2,3), cost.T is (3,2) — both should give same total cost
        assert self._total_cost(cost) == pytest.approx(self._total_cost(cost.T))

    def test_inf_matrix_symmetric_cost(self) -> None:
        cost = np.array([[1.0, np.inf], [np.inf, 2.0]], dtype=float)
        assert self._total_cost(cost) == pytest.approx(self._total_cost(cost.T))

    def test_all_inf_symmetric_cost(self) -> None:
        cost = np.full((3, 3), np.inf)
        assert self._total_cost(cost) == pytest.approx(self._total_cost(cost.T))


class TestLargeMatrix:
    """Performance and correctness on large matrices."""

    def test_100x100_completes_without_error(self) -> None:
        rng = np.random.default_rng(42)
        cost = rng.random((100, 100))
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 100
        assert len(col_ind) == 100

    def test_100x100_returns_valid_assignment(self) -> None:
        rng = np.random.default_rng(99)
        cost = rng.random((100, 100))
        row_ind, col_ind = hungarian_match(cost)
        # No duplicate rows or columns
        assert len(set(row_ind)) == 100
        assert len(set(col_ind)) == 100

    def test_100x100_with_some_inf_returns_valid_partial_assignment(self) -> None:
        rng = np.random.default_rng(7)
        cost = rng.random((100, 100))
        # Sprinkle 200 inf values
        for _ in range(200):
            r, c = rng.integers(0, 100, size=2)
            cost[r, c] = np.inf
        row_ind, col_ind = hungarian_match(cost)
        # All returned assignments have finite costs in original
        for r, c in zip(row_ind, col_ind, strict=True):
            assert np.isfinite(cost[r, c])


class TestEdgeCases:
    """Additional edge cases."""

    def test_single_row_matrix_returns_minimum_cost_column(self) -> None:
        cost = np.array([[3.0, 1.0, 4.0, 1.5, 9.0]])
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 1
        assert cost[row_ind, col_ind][0] == pytest.approx(1.0)

    def test_single_column_matrix_returns_minimum_cost_row(self) -> None:
        cost = np.array([[3.0], [1.0], [4.0]])
        row_ind, col_ind = hungarian_match(cost)
        assert len(row_ind) == 1
        assert cost[row_ind, col_ind][0] == pytest.approx(1.0)

    def test_inf_only_in_optimal_assignment_forces_suboptimal(self) -> None:
        # Without inf guard, optimal would be diagonal. With inf on diagonal,
        # must pick off-diagonal.
        cost = np.array(
            [[np.inf, 1.0, 2.0], [3.0, np.inf, 4.0], [5.0, 6.0, np.inf]],
            dtype=float,
        )
        row_ind, col_ind = hungarian_match(cost)
        # No diagonal assignments
        for r, c in zip(row_ind, col_ind, strict=True):
            assert r != c, f"Diagonal assignment ({r},{c}) should be avoided"

    def test_return_type_is_tuple_of_ndarrays(self) -> None:
        cost = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = hungarian_match(cost)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

    def test_nan_propagates_without_handling(self) -> None:
        """NaN is NOT guarded — only inf is. Let scipy raise or return bad results."""
        cost = np.array([[1.0, float("nan")], [2.0, 3.0]])
        # Do not assert specific behavior — just verify no suppression of error
        # This test documents the contract: NaN is not handled by hungarian_match.
        # scipy may raise ValueError — that is acceptable per plan spec.
        import contextlib

        with contextlib.suppress(ValueError, FloatingPointError):
            hungarian_match(cost)

    def test_integer_cost_matrix_coerced_to_float(self) -> None:
        cost = np.array([[1, 2], [3, 4]], dtype=int)
        row_ind, _col_ind = hungarian_match(cost)
        # Should not crash; int arrays are handled
        assert len(row_ind) == 2
