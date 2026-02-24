"""Unit tests for normalize_similarity in normalizer.py.

Tests verify:
- Formula correctness (STED paper formula)
- Boundary conditions (both-empty, large costs, lambda=0)
- Output always in [0.0, 1.0]
- No ZeroDivisionError when n_left == n_right == 0
"""

from __future__ import annotations

import pytest

from semantic_diff.algorithm.normalizer import normalize_similarity


class TestNormalizeSimilarity:
    """Comprehensive tests for the normalize_similarity function."""

    # ------------------------------------------------------------------
    # Identity cases (d_matched == 0)
    # ------------------------------------------------------------------

    def test_identical_children_symmetric(self) -> None:
        """Zero raw distance with equal-size child sets -> 1.0 similarity."""
        result = normalize_similarity(d_matched=0.0, n_left=5, n_right=5, lambda_=0.1)
        assert result == pytest.approx(1.0)

    def test_identical_children_single(self) -> None:
        """Zero raw distance, 1 child each -> 1.0 similarity."""
        result = normalize_similarity(d_matched=0.0, n_left=1, n_right=1, lambda_=0.1)
        assert result == pytest.approx(1.0)

    def test_identical_children_large(self) -> None:
        """Zero raw distance, many children -> 1.0 similarity."""
        result = normalize_similarity(
            d_matched=0.0, n_left=100, n_right=100, lambda_=0.1
        )
        assert result == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # Maximum distance cases (d_matched == max)
    # ------------------------------------------------------------------

    def test_max_distance_symmetric(self) -> None:
        """Raw distance equal to max child count -> 0.0 similarity."""
        # d_matched = n_left = n_right = 5, lambda_ = 0 -> cost/denom = 5/5 = 1.0
        result = normalize_similarity(d_matched=5.0, n_left=5, n_right=5, lambda_=0.0)
        assert result == pytest.approx(0.0)

    def test_max_distance_clips_to_zero(self) -> None:
        """Cost exceeding denominator clips to 0.0 via min(1, ...) guard."""
        # d_matched = 10, denom = 5 -> cost/denom = 2.0 -> min(1,2)=1 -> 1-1=0
        result = normalize_similarity(d_matched=10.0, n_left=5, n_right=5, lambda_=0.0)
        assert result == pytest.approx(0.0)

    def test_large_cost_clipped(self) -> None:
        """Very large d_matched clips to 0.0, never goes negative."""
        result = normalize_similarity(d_matched=1e9, n_left=3, n_right=3, lambda_=0.1)
        assert result == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # Empty child set cases (ZeroDivisionError guard)
    # ------------------------------------------------------------------

    def test_both_empty_no_division_error(self) -> None:
        """n_left == n_right == 0 -> 1.0 (max(0,0,1)=1, cost=0 -> 1.0)."""
        result = normalize_similarity(d_matched=0.0, n_left=0, n_right=0, lambda_=0.1)
        assert result == pytest.approx(1.0)

    def test_both_empty_zero_lambda(self) -> None:
        """Both empty with lambda_=0 -> still 1.0."""
        result = normalize_similarity(d_matched=0.0, n_left=0, n_right=0, lambda_=0.0)
        assert result == pytest.approx(1.0)

    def test_both_empty_large_lambda(self) -> None:
        """Both empty with large lambda_ -> 1.0 (no delta to penalize)."""
        result = normalize_similarity(d_matched=0.0, n_left=0, n_right=0, lambda_=100.0)
        assert result == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # Unmatched children (lambda_ penalty)
    # ------------------------------------------------------------------

    def test_unmatched_children_penalized(self) -> None:
        """More children on one side increases penalty via lambda_."""
        # n_left=3, n_right=5, delta=2, lambda_=0.5
        # d_matched=0, penalty=0.5*2=1.0, total=1.0, denom=5 -> 1-0.2=0.8
        result = normalize_similarity(d_matched=0.0, n_left=3, n_right=5, lambda_=0.5)
        assert result == pytest.approx(0.8)

    def test_unmatched_children_lambda_zero_no_penalty(self) -> None:
        """lambda_=0 means unmatched children add no penalty."""
        # Different n_left, n_right but lambda_=0 and d_matched=0 -> similarity=1.0
        result = normalize_similarity(d_matched=0.0, n_left=3, n_right=10, lambda_=0.0)
        assert result == pytest.approx(1.0)

    def test_unmatched_delta_is_absolute(self) -> None:
        """Delta uses absolute value: n_left-n_right and n_right-n_left give same result."""
        r1 = normalize_similarity(d_matched=0.0, n_left=3, n_right=5, lambda_=0.1)
        r2 = normalize_similarity(d_matched=0.0, n_left=5, n_right=3, lambda_=0.1)
        assert r1 == pytest.approx(r2)

    def test_one_side_empty(self) -> None:
        """One side empty, other has children: large penalty applied."""
        # n_left=0, n_right=5, delta=5, lambda_=0.1, d_matched=0
        # penalty=0.1*5=0.5, total=0.5, denom=5 -> 1-0.1=0.9
        result = normalize_similarity(d_matched=0.0, n_left=0, n_right=5, lambda_=0.1)
        assert result == pytest.approx(0.9)

    # ------------------------------------------------------------------
    # Formula correctness
    # ------------------------------------------------------------------

    def test_formula_manual_calculation(self) -> None:
        """Manually computed expected value matches function output."""
        # d_matched=2.0, n_left=4, n_right=6, lambda_=0.2
        # delta = |4-6| = 2, penalty = 0.2*2 = 0.4
        # total = 2.0 + 0.4 = 2.4, denom = max(4,6,1) = 6
        # ratio = 2.4/6 = 0.4, result = 1 - min(1, 0.4) = 0.6
        result = normalize_similarity(d_matched=2.0, n_left=4, n_right=6, lambda_=0.2)
        assert result == pytest.approx(0.6)

    def test_formula_partial_match(self) -> None:
        """Partial match (some cost, equal sizes) -> between 0 and 1."""
        # d_matched=2, n_left=5, n_right=5, lambda_=0
        # total=2, denom=5 -> 1-0.4=0.6
        result = normalize_similarity(d_matched=2.0, n_left=5, n_right=5, lambda_=0.0)
        assert result == pytest.approx(0.6)

    def test_denominator_uses_max(self) -> None:
        """Denominator is max(n_left, n_right, 1), not n_left+n_right."""
        # n_left=2, n_right=4 -> denom=4 (not 6)
        # d_matched=0, penalty=0.1*2=0.2, total=0.2, denom=4 -> 1-0.05=0.95
        result = normalize_similarity(d_matched=0.0, n_left=2, n_right=4, lambda_=0.1)
        assert result == pytest.approx(0.95)

    # ------------------------------------------------------------------
    # Output bounds: always in [0.0, 1.0]
    # ------------------------------------------------------------------

    def test_result_never_negative(self) -> None:
        """Result is never below 0.0."""
        result = normalize_similarity(
            d_matched=1000.0, n_left=1, n_right=1, lambda_=1.0
        )
        assert result >= 0.0

    def test_result_never_exceeds_one(self) -> None:
        """Result is never above 1.0."""
        result = normalize_similarity(d_matched=0.0, n_left=5, n_right=5, lambda_=0.0)
        assert result <= 1.0

    @pytest.mark.parametrize(
        ("d_matched", "n_left", "n_right", "lambda_"),
        [
            (0.0, 0, 0, 0.0),
            (0.0, 1, 1, 0.1),
            (0.5, 3, 3, 0.1),
            (1.0, 1, 1, 0.0),
            (2.0, 2, 4, 0.5),
            (0.0, 10, 5, 0.2),
            (100.0, 1, 1, 1.0),
            (0.3, 7, 7, 0.05),
        ],
    )
    def test_bounds_parametrized(
        self, d_matched: float, n_left: int, n_right: int, lambda_: float
    ) -> None:
        """Result always in [0.0, 1.0] across varied inputs."""
        result = normalize_similarity(d_matched, n_left, n_right, lambda_)
        assert 0.0 <= result <= 1.0

    # ------------------------------------------------------------------
    # Edge: lambda_=0 behaviour
    # ------------------------------------------------------------------

    def test_lambda_zero_ignores_count_mismatch(self) -> None:
        """With lambda_=0, only d_matched matters (no unmatched penalty)."""
        r1 = normalize_similarity(d_matched=1.0, n_left=5, n_right=5, lambda_=0.0)
        r2 = normalize_similarity(d_matched=1.0, n_left=5, n_right=100, lambda_=0.0)
        # r2 has larger denominator (100) so r2 > r1
        assert r1 < r2

    def test_lambda_zero_equal_sizes_partial(self) -> None:
        """lambda_=0, d_matched=1, n=2 -> 1 - 1/2 = 0.5."""
        result = normalize_similarity(d_matched=1.0, n_left=2, n_right=2, lambda_=0.0)
        assert result == pytest.approx(0.5)
