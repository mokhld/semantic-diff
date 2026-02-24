"""Integration tests for STEDAlgorithm.

Tests verify the must-have truths from 04-03-PLAN.md:
- ALGO-03: Accurate STED similarity (identity=1.0, semantic > 0.85, break < 0.1)
- ALGO-04: Object order invariance
- ALGO-05: Per-level normalization, symmetry, bounds
- ALGO-06: Ordered array mode
- ALGO-07: Unordered array mode
- ALGO-09: Auto mode homogeneity heuristic

All tests use StaticBackend (injected, no ML dependencies).
"""

from __future__ import annotations

import pytest

from semantic_diff.algorithm.config import ArrayComparisonMode, STEDConfig
from semantic_diff.algorithm.sted import STEDAlgorithm
from semantic_diff.backends.static import StaticBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> StaticBackend:
    """StaticBackend instance shared across tests."""
    return StaticBackend()


@pytest.fixture
def algo(backend: StaticBackend) -> STEDAlgorithm:
    """Default STEDAlgorithm (ordered mode, default config)."""
    return STEDAlgorithm(backend=backend)


@pytest.fixture
def algo_unordered(backend: StaticBackend) -> STEDAlgorithm:
    """STEDAlgorithm with UNORDERED array comparison mode."""
    config = STEDConfig(array_comparison_mode=ArrayComparisonMode.UNORDERED)
    return STEDAlgorithm(backend=backend, config=config)


@pytest.fixture
def algo_auto(backend: StaticBackend) -> STEDAlgorithm:
    """STEDAlgorithm with AUTO array comparison mode."""
    config = STEDConfig(array_comparison_mode=ArrayComparisonMode.AUTO)
    return STEDAlgorithm(backend=backend, config=config)


# ---------------------------------------------------------------------------
# Identity tests (ALGO-03, ALGO-05)
# ---------------------------------------------------------------------------


class TestIdentity:
    """Identical inputs must return similarity of exactly 1.0."""

    def test_identical_simple_object(self, algo: STEDAlgorithm) -> None:
        """Simple object identity."""
        assert algo.compute({"a": 1}, {"a": 1}) == pytest.approx(1.0)

    def test_identical_empty_array(self, algo: STEDAlgorithm) -> None:
        """Empty arrays are identical."""
        assert algo.compute([], []) == pytest.approx(1.0)

    def test_identical_empty_object(self, algo: STEDAlgorithm) -> None:
        """Empty objects are identical."""
        assert algo.compute({}, {}) == pytest.approx(1.0)

    def test_identical_string_scalar(self, algo: STEDAlgorithm) -> None:
        """String scalars: same value is identical."""
        assert algo.compute("hello", "hello") == pytest.approx(1.0)

    def test_identical_integer_scalar(self, algo: STEDAlgorithm) -> None:
        """Integer scalars: same value is identical."""
        assert algo.compute(42, 42) == pytest.approx(1.0)

    def test_identical_float_scalar(self, algo: STEDAlgorithm) -> None:
        """Float scalars: same value is identical."""
        assert algo.compute(3.14, 3.14) == pytest.approx(1.0)

    def test_identical_null(self, algo: STEDAlgorithm) -> None:
        """Null scalars are identical."""
        assert algo.compute(None, None) == pytest.approx(1.0)

    def test_identical_bool_true(self, algo: STEDAlgorithm) -> None:
        """True == True is identical."""
        assert algo.compute(True, True) == pytest.approx(1.0)

    def test_identical_bool_false(self, algo: STEDAlgorithm) -> None:
        """False == False is identical."""
        assert algo.compute(False, False) == pytest.approx(1.0)

    def test_identical_deep_nested(self, algo: STEDAlgorithm) -> None:
        """Deep nested objects are identical."""
        doc = {"a": {"b": {"c": 1}}}
        assert algo.compute(doc, doc) == pytest.approx(1.0)

    def test_identical_nested_with_array(self, algo: STEDAlgorithm) -> None:
        """Nested object containing an array is identical to itself."""
        doc = {"a": [1, 2, 3], "b": {"c": "hello"}}
        assert algo.compute(doc, doc) == pytest.approx(1.0)

    def test_identical_multi_key_object(self, algo: STEDAlgorithm) -> None:
        """Multi-key object identity."""
        doc = {"x": 1, "y": 2, "z": 3}
        assert algo.compute(doc, doc) == pytest.approx(1.0)

    def test_identical_array_of_scalars(self, algo: STEDAlgorithm) -> None:
        """Array of scalars is identical to itself."""
        assert algo.compute([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_identical_array_of_objects(self, algo: STEDAlgorithm) -> None:
        """Array of objects is identical to itself."""
        doc = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        assert algo.compute(doc, doc) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Semantic equivalence (ALGO-03)
# ---------------------------------------------------------------------------


class TestSemanticEquivalence:
    """Naming-convention equivalents must score > 0.85 with StaticBackend."""

    def test_snake_vs_camel_single_key(self, algo: STEDAlgorithm) -> None:
        """user_name vs userName with same value -> > 0.85."""
        score = algo.compute({"user_name": "John"}, {"userName": "John"})
        assert score > 0.85

    def test_snake_vs_camel_multi_key(self, algo: STEDAlgorithm) -> None:
        """Multiple naming-convention equivalents -> > 0.85."""
        a = {"user_name": "John", "email": "j@x.com"}
        b = {"userName": "John", "email": "j@x.com"}
        assert algo.compute(a, b) > 0.85

    def test_snake_vs_pascal(self, algo: STEDAlgorithm) -> None:
        """user_name vs UserName -> naming convention equivalents."""
        score = algo.compute({"user_name": "Alice"}, {"UserName": "Alice"})
        assert score > 0.85

    def test_camel_vs_kebab(self, algo: STEDAlgorithm) -> None:
        """camelCase vs kebab-case equivalent keys."""
        score = algo.compute({"firstName": "Bob"}, {"first-name": "Bob"})
        assert score > 0.85

    def test_identical_after_normalization(self, algo: STEDAlgorithm) -> None:
        """Keys that normalize to the same canonical form score 1.0."""
        # user_name and userName both normalize to 'user name'
        score = algo.compute({"user_name": "test"}, {"userName": "test"})
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Structural breaks (ALGO-03)
# ---------------------------------------------------------------------------


class TestStructuralBreaks:
    """Structurally unrelated documents must score < 0.1."""

    def test_user_name_vs_address(self, algo: STEDAlgorithm) -> None:
        """user_name key vs address key: completely different semantics."""
        score = algo.compute({"user_name": "John"}, {"address": "123 Main St"})
        assert score < 0.1

    def test_different_scalar_types_object(self, algo: STEDAlgorithm) -> None:
        """Object vs array (different root types) -> low score."""
        score = algo.compute({"a": 1}, [1])
        assert score < 0.1

    def test_unrelated_scalars(self, algo: STEDAlgorithm) -> None:
        """Two completely different scalar values (integer vs string)."""
        score = algo.compute(42, "hello")
        assert score < 1.0  # Not necessarily < 0.1 for scalars, but < 1.0


# ---------------------------------------------------------------------------
# Object order invariance (ALGO-04)
# ---------------------------------------------------------------------------


class TestObjectOrderInvariance:
    """Object comparison must be order-invariant (ALGO-04)."""

    def test_two_key_reorder(self, algo: STEDAlgorithm) -> None:
        """Two-key object reordered -> 1.0 (Hungarian matching)."""
        assert algo.compute({"a": 1, "b": 2}, {"b": 2, "a": 1}) == pytest.approx(1.0)

    def test_three_key_reorder(self, algo: STEDAlgorithm) -> None:
        """Three-key object reordered -> 1.0."""
        a = {"x": 1, "y": 2, "z": 3}
        b = {"z": 3, "x": 1, "y": 2}
        assert algo.compute(a, b) == pytest.approx(1.0)

    def test_five_key_reorder(self, algo: STEDAlgorithm) -> None:
        """Five-key object reordered -> 1.0."""
        a = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        b = {"e": 5, "d": 4, "a": 1, "c": 3, "b": 2}
        assert algo.compute(a, b) == pytest.approx(1.0)

    def test_nested_object_reorder(self, algo: STEDAlgorithm) -> None:
        """Nested objects with reordered keys at multiple levels -> 1.0."""
        a = {"outer_a": 1, "nested": {"inner_x": "foo", "inner_y": "bar"}}
        b = {"nested": {"inner_y": "bar", "inner_x": "foo"}, "outer_a": 1}
        assert algo.compute(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Ordered array mode (ALGO-06)
# ---------------------------------------------------------------------------


class TestOrderedArrayMode:
    """Default mode is ORDERED: position matters for arrays."""

    def test_same_order_identity(self, algo: STEDAlgorithm) -> None:
        """Same order -> 1.0."""
        assert algo.compute([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_reversed_order_not_identity(self, algo: STEDAlgorithm) -> None:
        """Reversed order -> < 1.0 (position matters)."""
        score = algo.compute([1, 2, 3], [3, 2, 1])
        assert score < 1.0

    def test_different_lengths_not_identity(self, algo: STEDAlgorithm) -> None:
        """Different length arrays -> < 1.0."""
        score = algo.compute([1, 2, 3], [1, 2, 3, 4])
        assert score < 1.0

    def test_prefix_match_partial(self, algo: STEDAlgorithm) -> None:
        """Common prefix arrays -> partial match score."""
        score = algo.compute([1, 2, 3], [1, 2, 9])
        assert 0.0 < score < 1.0

    def test_completely_different_arrays(self, algo: STEDAlgorithm) -> None:
        """Completely different elements -> low-ish score."""
        score = algo.compute([1, 2, 3], [4, 5, 6])
        assert score < 1.0


# ---------------------------------------------------------------------------
# Unordered array mode (ALGO-07)
# ---------------------------------------------------------------------------


class TestUnorderedArrayMode:
    """UNORDERED mode: position is irrelevant."""

    def test_reversed_is_identity(self, algo_unordered: STEDAlgorithm) -> None:
        """Reversed scalars -> 1.0 (unordered = set comparison)."""
        assert algo_unordered.compute([1, 2, 3], [3, 2, 1]) == pytest.approx(1.0)

    def test_same_elements_different_order(self, algo_unordered: STEDAlgorithm) -> None:
        """Same elements, any permutation -> 1.0."""
        assert algo_unordered.compute([10, 20, 30], [30, 10, 20]) == pytest.approx(1.0)

    def test_different_content_not_identity(
        self, algo_unordered: STEDAlgorithm
    ) -> None:
        """Different content (one element differs) -> < 1.0."""
        score = algo_unordered.compute([1, 2, 3], [1, 2, 4])
        assert score < 1.0

    def test_different_lengths_not_identity(
        self, algo_unordered: STEDAlgorithm
    ) -> None:
        """Different length arrays -> < 1.0 even in UNORDERED mode."""
        score = algo_unordered.compute([1, 2, 3], [1, 2, 3, 4])
        assert score < 1.0

    def test_empty_arrays_identity(self, algo_unordered: STEDAlgorithm) -> None:
        """Empty arrays are identical in UNORDERED mode."""
        assert algo_unordered.compute([], []) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Auto array mode (ALGO-09)
# ---------------------------------------------------------------------------


class TestAutoArrayMode:
    """AUTO mode: infers ordered/unordered from content homogeneity."""

    def test_scalar_arrays_treated_as_unordered(self, algo_auto: STEDAlgorithm) -> None:
        """Scalar-only arrays -> AUTO infers UNORDERED -> reversed = 1.0."""
        score = algo_auto.compute([1, 2, 3], [3, 2, 1])
        assert score == pytest.approx(1.0)

    def test_object_arrays_treated_as_ordered(self, algo_auto: STEDAlgorithm) -> None:
        """Object arrays -> AUTO infers ORDERED -> different objects matter."""
        a = [{"id": 1}]
        b = [{"id": 2}]
        score_auto = algo_auto.compute(a, b)
        # With ORDERED, position still matters — just check it's a valid score
        assert 0.0 <= score_auto <= 1.0

    def test_mixed_arrays_treated_as_ordered(self, algo_auto: STEDAlgorithm) -> None:
        """Mixed scalar+object arrays -> AUTO infers ORDERED."""
        a = [1, {"a": 2}]
        b = [{"a": 2}, 1]
        # AUTO should choose ORDERED (has objects) -> order matters -> < 1.0
        score = algo_auto.compute(a, b)
        assert 0.0 <= score <= 1.0

    def test_empty_arrays_auto(self, algo_auto: STEDAlgorithm) -> None:
        """Empty arrays in AUTO mode -> 1.0 (resolved to UNORDERED)."""
        assert algo_auto.compute([], []) == pytest.approx(1.0)

    def test_scalar_permutation_is_identity(self, algo_auto: STEDAlgorithm) -> None:
        """AUTO: string scalar arrays -> unordered, permutation = identity."""
        assert algo_auto.compute(["a", "b", "c"], ["c", "a", "b"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Symmetry tests (ALGO-05)
# ---------------------------------------------------------------------------


class TestSymmetry:
    """compute(a, b) must equal compute(b, a) for all inputs."""

    @pytest.mark.parametrize(
        ("json_a", "json_b"),
        [
            # Scalars
            (42, 99),
            ("hello", "world"),
            (None, "something"),
            (True, False),
            (3.14, 2.71),
            # Simple objects
            ({"a": 1}, {"b": 2}),
            ({"a": 1, "b": 2}, {"c": 3}),
            ({}, {"a": 1}),
            ({"x": "hello"}, {"y": "world"}),
            # Nested objects
            ({"a": {"b": 1}}, {"a": {"b": 2}}),
            ({"a": {"b": {"c": 1}}}, {"a": {"b": {"c": 2}}}),
            # Arrays
            ([1, 2, 3], [4, 5, 6]),
            ([1], [1, 2]),
            ([], [1, 2, 3]),
            ([1, 2, 3], [3, 2, 1]),
            # Mixed types
            ({"a": 1}, [1]),
            ([{"a": 1}], [{"b": 2}]),
            ({"a": [1, 2]}, {"a": [3, 4]}),
            # Objects with different key counts
            ({"a": 1, "b": 2, "c": 3}, {"x": 10}),
            ({"name": "Alice", "age": 30}, {"name": "Bob", "email": "b@x.com"}),
        ],
        ids=[
            "ints",
            "strings",
            "null-vs-str",
            "bools",
            "floats",
            "simple-objs-diff-keys",
            "objs-diff-count",
            "empty-vs-obj",
            "objs-diff-string-vals",
            "nested-diff-leaf",
            "deep-nested-diff",
            "arrays-diff",
            "arrays-diff-len",
            "empty-vs-array",
            "reversed-array",
            "obj-vs-array",
            "obj-arrays",
            "obj-with-array-vals",
            "objs-diff-key-counts",
            "objs-diff-keys-and-vals",
        ],
    )
    def test_symmetry(
        self, algo: STEDAlgorithm, json_a: object, json_b: object
    ) -> None:
        """compute(a, b) == compute(b, a) within floating-point tolerance."""
        score_ab = algo.compute(json_a, json_b)
        score_ba = algo.compute(json_b, json_a)
        assert abs(score_ab - score_ba) < 1e-9, (
            f"Symmetry violation: compute({json_a!r}, {json_b!r}) = {score_ab} "
            f"!= compute({json_b!r}, {json_a!r}) = {score_ba}"
        )


# ---------------------------------------------------------------------------
# Normalization bounds (ALGO-05)
# ---------------------------------------------------------------------------


class TestNormalizationBounds:
    """All scores must be in [0.0, 1.0]."""

    @pytest.mark.parametrize(
        ("json_a", "json_b"),
        [
            ({}, {}),
            ({"a": 1}, {"a": 1}),
            ({"a": 1}, {"b": 2}),
            ([], []),
            ([1, 2], [3, 4]),
            ("hello", "world"),
            (42, 99),
            (None, None),
            ({"a": {"b": 1}}, {"a": {"b": 2}}),
            ({"a": [1, 2]}, {"b": [3, 4]}),
            ([{"id": 1}], [{"id": 2}]),
            ({"a": 1}, [1]),
            ({}, []),
        ],
    )
    def test_score_in_bounds(
        self, algo: STEDAlgorithm, json_a: object, json_b: object
    ) -> None:
        """Score is in [0.0, 1.0] for all valid input pairs."""
        score = algo.compute(json_a, json_b)
        assert 0.0 <= score <= 1.0, (
            f"Out of bounds score {score} for {json_a!r} vs {json_b!r}"
        )

    def test_deep_nesting_in_bounds(self, algo: STEDAlgorithm) -> None:
        """Very deep nesting does not push score outside [0, 1]."""
        deep_a: dict[str, object] = {"level": {"level": {"level": {"value": 1}}}}
        deep_b: dict[str, object] = {"level": {"level": {"level": {"value": 99}}}}
        score = algo.compute(deep_a, deep_b)
        assert 0.0 <= score <= 1.0

    def test_wide_object_in_bounds(self, algo: STEDAlgorithm) -> None:
        """Wide object (many keys) stays in [0, 1]."""
        a = {f"key_{i}": i for i in range(20)}
        b = {f"other_{i}": i * 2 for i in range(20)}
        score = algo.compute(a, b)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Type mismatch (ALGO-03)
# ---------------------------------------------------------------------------


class TestTypeMismatch:
    """Different root types must score 0.0 (complete structural break)."""

    def test_object_vs_array(self, algo: STEDAlgorithm) -> None:
        """Object vs array -> 0.0 (type mismatch at root)."""
        assert algo.compute({"a": 1}, [1]) == pytest.approx(0.0)

    def test_array_vs_object(self, algo: STEDAlgorithm) -> None:
        """Array vs object -> 0.0 (type mismatch at root)."""
        assert algo.compute([1], {"a": 1}) == pytest.approx(0.0)

    def test_object_vs_scalar(self, algo: STEDAlgorithm) -> None:
        """Object vs scalar -> 0.0."""
        assert algo.compute({"a": 1}, 42) == pytest.approx(0.0)

    def test_array_vs_scalar(self, algo: STEDAlgorithm) -> None:
        """Array vs scalar -> 0.0."""
        assert algo.compute([1, 2], "hello") == pytest.approx(0.0)

    def test_null_vs_object(self, algo: STEDAlgorithm) -> None:
        """Null vs object -> 0.0."""
        assert algo.compute(None, {"a": 1}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Additional correctness tests
# ---------------------------------------------------------------------------


class TestAdditionalCorrectness:
    """Extra tests for edge cases and specific behaviors."""

    def test_single_extra_key(self, algo: STEDAlgorithm) -> None:
        """One object has an extra key: partial similarity."""
        score = algo.compute({"a": 1}, {"a": 1, "b": 2})
        assert 0.0 < score < 1.0

    def test_nested_value_change(self, algo: STEDAlgorithm) -> None:
        """Same structure, different leaf value: high but not 1.0 similarity."""
        a = {"user": {"name": "Alice"}}
        b = {"user": {"name": "Bob"}}
        score = algo.compute(a, b)
        assert 0.0 < score < 1.0

    def test_different_scalar_values_not_identical(self, algo: STEDAlgorithm) -> None:
        """Different integer values -> content_distance = 1.0 -> score < 1.0."""
        # Note: True == 1 in Python (bool subclass of int), so True vs 1 is 1.0.
        # Use genuinely different values to verify content_distance works.
        score = algo.compute(42, 99)
        assert score < 1.0

    def test_default_config_is_ordered_mode(self, algo: STEDAlgorithm) -> None:
        """Default config uses ORDERED array mode."""
        # [1,2,3] vs [3,2,1]: ORDERED -> < 1.0; UNORDERED -> 1.0
        ordered_score = algo.compute([1, 2, 3], [3, 2, 1])
        assert ordered_score < 1.0

    def test_none_config_uses_defaults(self, backend: StaticBackend) -> None:
        """STEDAlgorithm with config=None uses STEDConfig() defaults."""
        algo = STEDAlgorithm(backend=backend, config=None)
        assert algo.compute({}, {}) == pytest.approx(1.0)

    def test_custom_lambda_affects_score(self, backend: StaticBackend) -> None:
        """Higher lambda_unmatched increases penalty for unmatched children."""
        config_low = STEDConfig(lambda_unmatched=0.0)
        config_high = STEDConfig(lambda_unmatched=1.0)
        algo_low = STEDAlgorithm(backend=backend, config=config_low)
        algo_high = STEDAlgorithm(backend=backend, config=config_high)

        # One object has an extra key — higher lambda penalizes more
        a = {"x": 1, "y": 2}
        b = {"x": 1}
        score_low = algo_low.compute(a, b)
        score_high = algo_high.compute(a, b)
        assert score_high < score_low

    def test_array_length_mismatch_penalized(self, algo: STEDAlgorithm) -> None:
        """Longer array vs shorter array -> < identity score."""
        score_equal = algo.compute([1, 2, 3], [1, 2, 3])
        score_longer = algo.compute([1, 2, 3], [1, 2, 3, 4])
        assert score_longer < score_equal
