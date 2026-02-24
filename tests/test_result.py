"""Tests for ComparisonResult frozen dataclass.

Covers:
- Successful construction with all six required fields
- Frozen (immutable) enforcement: setting any field raises FrozenInstanceError
- All fields accessible by name
- Equality: two ComparisonResult with identical values are equal
- Score range is NOT validated by the dataclass (it is a data container)
- __all__ export
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from json_semantic_diff.result import ComparisonResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(**overrides: object) -> ComparisonResult:
    """Return a valid ComparisonResult, optionally overriding specific fields."""
    defaults: dict[str, object] = {
        "similarity_score": 0.95,
        "matched_pairs": [("/a", "/a")],
        "key_mappings": {"a": "a"},
        "unmatched_left": [],
        "unmatched_right": [],
        "computation_time_ms": 1.5,
    }
    defaults.update(overrides)
    return ComparisonResult(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestComparisonResultConstruction:
    def test_basic_construction_succeeds(self) -> None:
        result = ComparisonResult(
            similarity_score=0.95,
            matched_pairs=[("/a", "/a")],
            key_mappings={"a": "a"},
            unmatched_left=[],
            unmatched_right=[],
            computation_time_ms=1.5,
        )
        assert result is not None

    def test_construction_with_empty_matches(self) -> None:
        result = ComparisonResult(
            similarity_score=0.0,
            matched_pairs=[],
            key_mappings={},
            unmatched_left=["/x"],
            unmatched_right=["/y"],
            computation_time_ms=0.1,
        )
        assert result.similarity_score == pytest.approx(0.0)

    def test_construction_with_multiple_pairs(self) -> None:
        result = ComparisonResult(
            similarity_score=1.0,
            matched_pairs=[("/a", "/a"), ("/b", "/b"), ("/c/d", "/c/d")],
            key_mappings={"a": "a", "b": "b", "d": "d"},
            unmatched_left=[],
            unmatched_right=[],
            computation_time_ms=3.7,
        )
        assert len(result.matched_pairs) == 3
        assert len(result.key_mappings) == 3

    def test_score_of_one_is_accepted(self) -> None:
        result = make_result(similarity_score=1.0)
        assert result.similarity_score == pytest.approx(1.0)

    def test_score_of_zero_is_accepted(self) -> None:
        result = make_result(similarity_score=0.0)
        assert result.similarity_score == pytest.approx(0.0)

    def test_score_outside_range_is_not_validated(self) -> None:
        """The dataclass is a data container; callers are responsible for validation."""
        result = make_result(similarity_score=1.5)
        assert result.similarity_score == pytest.approx(1.5)

    def test_negative_score_is_not_validated(self) -> None:
        result = make_result(similarity_score=-0.1)
        assert result.similarity_score == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


class TestComparisonResultFieldAccess:
    def test_similarity_score_accessible(self) -> None:
        result = make_result(similarity_score=0.75)
        assert result.similarity_score == pytest.approx(0.75)

    def test_matched_pairs_accessible(self) -> None:
        pairs = [("/foo", "/foo"), ("/bar", "/baz")]
        result = make_result(matched_pairs=pairs)
        assert result.matched_pairs == pairs

    def test_key_mappings_accessible(self) -> None:
        mappings = {"foo": "foo", "bar": "baz"}
        result = make_result(key_mappings=mappings)
        assert result.key_mappings == mappings

    def test_unmatched_left_accessible(self) -> None:
        result = make_result(unmatched_left=["/x", "/y"])
        assert result.unmatched_left == ["/x", "/y"]

    def test_unmatched_right_accessible(self) -> None:
        result = make_result(unmatched_right=["/z"])
        assert result.unmatched_right == ["/z"]

    def test_computation_time_ms_accessible(self) -> None:
        result = make_result(computation_time_ms=42.0)
        assert result.computation_time_ms == pytest.approx(42.0)

    def test_all_six_fields_in_dataclass_fields(self) -> None:
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(ComparisonResult)}
        assert field_names == {
            "similarity_score",
            "matched_pairs",
            "key_mappings",
            "unmatched_left",
            "unmatched_right",
            "computation_time_ms",
        }


# ---------------------------------------------------------------------------
# Frozen (immutability)
# ---------------------------------------------------------------------------


class TestComparisonResultFrozen:
    def test_similarity_score_is_frozen(self) -> None:
        result = make_result()
        with pytest.raises(FrozenInstanceError):
            result.similarity_score = 0.5  # type: ignore[misc]

    def test_matched_pairs_is_frozen(self) -> None:
        result = make_result()
        with pytest.raises(FrozenInstanceError):
            result.matched_pairs = []  # type: ignore[misc]

    def test_key_mappings_is_frozen(self) -> None:
        result = make_result()
        with pytest.raises(FrozenInstanceError):
            result.key_mappings = {}  # type: ignore[misc]

    def test_unmatched_left_is_frozen(self) -> None:
        result = make_result()
        with pytest.raises(FrozenInstanceError):
            result.unmatched_left = ["/new"]  # type: ignore[misc]

    def test_unmatched_right_is_frozen(self) -> None:
        result = make_result()
        with pytest.raises(FrozenInstanceError):
            result.unmatched_right = ["/new"]  # type: ignore[misc]

    def test_computation_time_ms_is_frozen(self) -> None:
        result = make_result()
        with pytest.raises(FrozenInstanceError):
            result.computation_time_ms = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


class TestComparisonResultEquality:
    def test_equal_instances_are_equal(self) -> None:
        r1 = make_result()
        r2 = make_result()
        assert r1 == r2

    def test_different_score_not_equal(self) -> None:
        r1 = make_result(similarity_score=0.9)
        r2 = make_result(similarity_score=0.8)
        assert r1 != r2

    def test_different_matched_pairs_not_equal(self) -> None:
        r1 = make_result(matched_pairs=[("/a", "/a")])
        r2 = make_result(matched_pairs=[("/b", "/b")])
        assert r1 != r2

    def test_different_unmatched_left_not_equal(self) -> None:
        r1 = make_result(unmatched_left=["/x"])
        r2 = make_result(unmatched_left=["/y"])
        assert r1 != r2

    def test_different_computation_time_not_equal(self) -> None:
        r1 = make_result(computation_time_ms=1.0)
        r2 = make_result(computation_time_ms=2.0)
        assert r1 != r2


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestComparisonResultExports:
    def test_all_exports_comparison_result(self) -> None:
        import json_semantic_diff.result as mod

        assert "ComparisonResult" in mod.__all__

    def test_importable_from_module(self) -> None:
        from json_semantic_diff.result import ComparisonResult as CR

        assert CR is ComparisonResult
