"""Unit tests for the public API functions: compare, is_equivalent, similarity_score."""

from __future__ import annotations

import pytest

from json_semantic_diff import (
    ArrayComparisonMode,
    ComparisonResult,
    STEDComparator,
    STEDConfig,
    compare,
    is_equivalent,
    similarity_score,
)


class TestCompare:
    """Tests for the compare() function."""

    def test_identical_dicts_return_similarity_1(self) -> None:
        result = compare({"a": 1}, {"a": 1})
        assert isinstance(result, ComparisonResult)
        assert result.similarity_score == pytest.approx(1.0)

    def test_different_dicts_return_similarity_below_1(self) -> None:
        result = compare({"a": 1}, {"b": 2})
        assert isinstance(result, ComparisonResult)
        assert result.similarity_score < 1.0

    def test_result_has_all_fields(self) -> None:
        result = compare({"a": 1}, {"a": 1})
        assert result.matched_pairs is not None
        assert result.key_mappings is not None
        assert result.unmatched_left is not None
        assert result.unmatched_right is not None
        assert result.computation_time_ms > 0

    def test_config_passthrough_type_coercion(self) -> None:
        result = compare(
            {"x": "123"}, {"x": 123}, config=STEDConfig(type_coercion=True)
        )
        assert result.similarity_score == pytest.approx(1.0)

    def test_config_passthrough_null_equals_missing(self) -> None:
        result = compare({"x": None}, {}, config=STEDConfig(null_equals_missing=True))
        assert result.similarity_score == pytest.approx(1.0)

    def test_no_global_state_between_calls(self) -> None:
        r1 = compare({"a": 1}, {"a": 1})
        r2 = compare({"a": 1}, {"a": 1})
        assert r1.similarity_score == r2.similarity_score


class TestIsEquivalent:
    """Tests for the is_equivalent() function."""

    def test_camel_snake_case_equivalent_at_default_threshold(self) -> None:
        assert is_equivalent({"user_name": "John"}, {"userName": "John"}) is True

    def test_different_keys_not_equivalent(self) -> None:
        assert is_equivalent({"a": 1}, {"b": 2}) is False

    def test_identical_is_equivalent(self) -> None:
        assert is_equivalent({"a": 1}, {"a": 1}, threshold=1.0) is True

    def test_returns_bool_type(self) -> None:
        result = is_equivalent({"a": 1}, {"a": 1})
        assert isinstance(result, bool)

    def test_custom_threshold(self) -> None:
        # Identical dicts should always pass any threshold up to 1.0
        assert is_equivalent({"a": 1}, {"a": 1}, threshold=1.0) is True


class TestSimilarityScore:
    """Tests for the similarity_score() function."""

    def test_identical_returns_1(self) -> None:
        score = similarity_score({"a": 1}, {"a": 1})
        assert score == pytest.approx(1.0)

    def test_different_returns_float_in_range(self) -> None:
        score = similarity_score({"a": 1}, {"b": 2})
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_config_passthrough_type_coercion(self) -> None:
        score = similarity_score(
            {"x": "123"}, {"x": 123}, config=STEDConfig(type_coercion=True)
        )
        assert score == pytest.approx(1.0)


class TestTopLevelImports:
    """Verify all public symbols are importable from the top-level package."""

    def test_all_symbols_importable(self) -> None:
        # This test passes if the imports at the top of this file succeed.
        assert compare is not None
        assert is_equivalent is not None
        assert similarity_score is not None
        assert ComparisonResult is not None
        assert STEDConfig is not None
        assert STEDComparator is not None
        assert ArrayComparisonMode is not None
