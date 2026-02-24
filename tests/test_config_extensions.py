"""Tests for STEDConfig extension fields: type_coercion and null_equals_missing.

Covers:
- Default values: type_coercion=False, null_equals_missing=False
- Successful construction with each new field set to True
- Frozen (immutable) enforcement on new fields
- Backward compatibility: STEDConfig() behaviour identical to before
- Type coercion integration with _content_distance (SCALAR nodes)
  - type_coercion=True: "123" vs 123 -> 0.0
  - type_coercion=False (default): "123" vs 123 -> 1.0
  - Float coercion: "3.14" vs 3.14 -> 0.0
  - Non-numeric: "hello" vs 42 -> 1.0 (coercion fails, falls back)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from json_semantic_diff.algorithm.config import ArrayComparisonMode, STEDConfig
from json_semantic_diff.algorithm.costs import _content_distance
from json_semantic_diff.tree.nodes import NodeType, TreeNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_scalar(value: object, path: str = "/val") -> TreeNode:
    """Create a SCALAR TreeNode with the given value."""
    return TreeNode(
        node_type=NodeType.SCALAR,
        label=str(value),
        path=path,
        value=value,
    )


# ---------------------------------------------------------------------------
# New field defaults
# ---------------------------------------------------------------------------


class TestSTEDConfigNewFieldDefaults:
    def test_type_coercion_default_false(self) -> None:
        config = STEDConfig()
        assert config.type_coercion is False

    def test_null_equals_missing_default_false(self) -> None:
        config = STEDConfig()
        assert config.null_equals_missing is False

    def test_both_defaults_false(self) -> None:
        config = STEDConfig()
        assert not config.type_coercion
        assert not config.null_equals_missing


# ---------------------------------------------------------------------------
# Construction with new fields
# ---------------------------------------------------------------------------


class TestSTEDConfigNewFieldConstruction:
    def test_type_coercion_true_constructs(self) -> None:
        config = STEDConfig(type_coercion=True)
        assert config.type_coercion is True

    def test_null_equals_missing_true_constructs(self) -> None:
        config = STEDConfig(null_equals_missing=True)
        assert config.null_equals_missing is True

    def test_both_new_fields_true_constructs(self) -> None:
        config = STEDConfig(type_coercion=True, null_equals_missing=True)
        assert config.type_coercion is True
        assert config.null_equals_missing is True

    def test_new_fields_combined_with_existing(self) -> None:
        config = STEDConfig(
            w_s=0.7,
            w_c=0.3,
            type_coercion=True,
            null_equals_missing=True,
            array_comparison_mode=ArrayComparisonMode.UNORDERED,
        )
        assert config.w_s == pytest.approx(0.7)
        assert config.w_c == pytest.approx(0.3)
        assert config.type_coercion is True
        assert config.null_equals_missing is True
        assert config.array_comparison_mode == ArrayComparisonMode.UNORDERED


# ---------------------------------------------------------------------------
# Frozen enforcement on new fields
# ---------------------------------------------------------------------------


class TestSTEDConfigNewFieldsFrozen:
    def test_type_coercion_is_frozen(self) -> None:
        config = STEDConfig(type_coercion=True)
        with pytest.raises(FrozenInstanceError):
            config.type_coercion = False  # type: ignore[misc]

    def test_null_equals_missing_is_frozen(self) -> None:
        config = STEDConfig(null_equals_missing=True)
        with pytest.raises(FrozenInstanceError):
            config.null_equals_missing = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestSTEDConfigBackwardCompatibility:
    def test_default_w_s_unchanged(self) -> None:
        config = STEDConfig()
        assert config.w_s == 0.5

    def test_default_w_c_unchanged(self) -> None:
        config = STEDConfig()
        assert config.w_c == 0.5

    def test_default_lambda_unmatched_unchanged(self) -> None:
        config = STEDConfig()
        assert config.lambda_unmatched == pytest.approx(0.1)

    def test_default_array_comparison_mode_unchanged(self) -> None:
        config = STEDConfig()
        assert config.array_comparison_mode == ArrayComparisonMode.ORDERED

    def test_weights_still_sum_to_one(self) -> None:
        config = STEDConfig()
        assert abs(config.w_s + config.w_c - 1.0) < 1e-9

    def test_still_hashable(self) -> None:
        config = STEDConfig()
        assert isinstance(hash(config), int)

    def test_equality_still_works(self) -> None:
        c1 = STEDConfig()
        c2 = STEDConfig()
        assert c1 == c2

    def test_custom_weights_still_validate(self) -> None:
        config = STEDConfig(w_s=0.8, w_c=0.2)
        assert config.w_s == pytest.approx(0.8)
        assert config.w_c == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Type coercion integration with _content_distance
# ---------------------------------------------------------------------------


class TestTypeCoercionInContentDistance:
    def test_string_int_equal_with_type_coercion(self) -> None:
        """SCALAR("123") vs SCALAR(123) with type_coercion=True -> 0.0."""
        node_a = make_scalar("123")
        node_b = make_scalar(123)
        config = STEDConfig(type_coercion=True)
        assert _content_distance(node_a, node_b, config) == pytest.approx(0.0)

    def test_string_int_different_without_type_coercion(self) -> None:
        """SCALAR("123") vs SCALAR(123) with type_coercion=False -> 1.0."""
        node_a = make_scalar("123")
        node_b = make_scalar(123)
        config = STEDConfig()  # type_coercion=False by default
        assert _content_distance(node_a, node_b, config) == pytest.approx(1.0)

    def test_float_string_equal_with_type_coercion(self) -> None:
        """SCALAR("3.14") vs SCALAR(3.14) with type_coercion=True -> 0.0."""
        node_a = make_scalar("3.14")
        node_b = make_scalar(3.14)
        config = STEDConfig(type_coercion=True)
        assert _content_distance(node_a, node_b, config) == pytest.approx(0.0)

    def test_non_numeric_string_vs_int_with_type_coercion(self) -> None:
        """SCALAR("hello") vs SCALAR(42) with type_coercion=True -> 1.0.

        Coercion of "hello" to float raises ValueError, falls back to 1.0.
        """
        node_a = make_scalar("hello")
        node_b = make_scalar(42)
        config = STEDConfig(type_coercion=True)
        assert _content_distance(node_a, node_b, config) == pytest.approx(1.0)

    def test_int_vs_float_coercion_equal(self) -> None:
        """SCALAR(1) vs SCALAR(1.0) -> equal without coercion (Python int/float equality)."""
        node_a = make_scalar(1)
        node_b = make_scalar(1.0)
        config = STEDConfig()  # default
        # Python: 1 == 1.0 is True, so content distance = 0.0 even without coercion
        assert _content_distance(node_a, node_b, config) == pytest.approx(0.0)

    def test_different_numeric_values_with_type_coercion(self) -> None:
        """SCALAR("123") vs SCALAR(456) with type_coercion=True -> 1.0 (different values)."""
        node_a = make_scalar("123")
        node_b = make_scalar(456)
        config = STEDConfig(type_coercion=True)
        assert _content_distance(node_a, node_b, config) == pytest.approx(1.0)

    def test_identical_strings_no_coercion_needed(self) -> None:
        """SCALAR("hello") vs SCALAR("hello") -> 0.0 regardless of type_coercion."""
        node_a = make_scalar("hello")
        node_b = make_scalar("hello")
        for coerce in (True, False):
            config = STEDConfig(type_coercion=coerce)
            assert _content_distance(node_a, node_b, config) == pytest.approx(0.0)

    def test_int_string_reversed_direction(self) -> None:
        """SCALAR(123) vs SCALAR("123") with type_coercion=True -> 0.0 (order symmetric)."""
        node_a = make_scalar(123)
        node_b = make_scalar("123")
        config = STEDConfig(type_coercion=True)
        assert _content_distance(node_a, node_b, config) == pytest.approx(0.0)
