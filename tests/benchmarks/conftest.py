"""Deterministic object generators for performance benchmarks.

All generators produce fixed, reproducible objects. No random values.
Three tiers: 10-key flat, 100-key nested, 500-key deeply nested.
Each tier provides both "similar" and "dissimilar" pair generators.

Object shapes use nesting to keep per-level key counts manageable
for the O(n^3) Hungarian matching at each level.
"""

from __future__ import annotations

from typing import Any

import pytest


def generate_flat_object(num_keys: int, prefix: str = "key") -> dict[str, Any]:
    """Generate a flat dict with deterministic string values."""
    return {f"{prefix}_{i}": f"value_{i}" for i in range(num_keys)}


def _make_similar_flat(
    num_keys: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate flat similar pair (snake_case vs camelCase)."""
    left = {f"field_name_{i}": f"value_{i}" for i in range(num_keys)}
    right = {f"fieldName{i}": f"value_{i}" for i in range(num_keys)}
    return left, right


def _make_similar_nested_100() -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate 100-key nested similar pair.

    Structure: 10 sections x (9 leaf keys each) = 100 total keys.
    Top-level Hungarian: 10x10 = 100 similarity computations.
    Per-section Hungarian: 9x9 = 81 computations, 10 sections = 810.
    Total: ~910 computations — expected ~80-100ms on StaticBackend.
    """
    left: dict[str, Any] = {}
    right: dict[str, Any] = {}
    for i in range(10):
        sub_l = {f"field_name_{i}_{j}": f"value_{i}_{j}" for j in range(9)}
        sub_r = {f"fieldName{i}_{j}": f"value_{i}_{j}" for j in range(9)}
        left[f"section_{i}"] = sub_l
        right[f"section{i}"] = sub_r
    return left, right


def _make_similar_nested_500() -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate 500-key deeply nested similar pair.

    Structure: 5 sections x 5 groups x (8 leaf keys each) + structural keys.
    = 5 + 25 + 200 leaf = ~500 total keys across 3 levels.
    Keeps per-level key counts at 5-8 for manageable Hungarian matching.
    """
    left: dict[str, Any] = {}
    right: dict[str, Any] = {}
    for i in range(5):
        mid_l: dict[str, Any] = {}
        mid_r: dict[str, Any] = {}
        for j in range(5):
            leaf_l: dict[str, Any] = {
                f"field_name_{i}_{j}_{k}": f"value_{i}_{j}_{k}" for k in range(8)
            }
            leaf_r: dict[str, Any] = {
                f"fieldName{i}_{j}_{k}": f"value_{i}_{j}_{k}" for k in range(8)
            }
            # Add a nested sub-object within each group
            inner_l: dict[str, Any] = {
                f"detail_{k}": f"d_{i}_{j}_{k}" for k in range(6)
            }
            inner_r: dict[str, Any] = {f"detail{k}": f"d_{i}_{j}_{k}" for k in range(6)}
            leaf_l[f"details_{i}_{j}"] = inner_l
            leaf_r[f"details{i}_{j}"] = inner_r
            mid_l[f"group_{j}"] = leaf_l
            mid_r[f"group{j}"] = leaf_r
        left[f"section_{i}"] = mid_l
        right[f"section{i}"] = mid_r
    return left, right


def _make_dissimilar_flat(num_keys: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate flat dissimilar pair (different domains)."""
    left = {f"user_field_{i}": f"user_value_{i}" for i in range(num_keys)}
    right = {f"product_attr_{i}": f"product_data_{i}" for i in range(num_keys)}
    return left, right


def _make_dissimilar_nested_100() -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate 100-key nested dissimilar pair (user vs product domains)."""
    left: dict[str, Any] = {}
    right: dict[str, Any] = {}
    for i in range(10):
        left[f"user_section_{i}"] = {
            f"user_field_{i}_{j}": f"user_val_{i}_{j}" for j in range(9)
        }
        right[f"product_section_{i}"] = {
            f"product_attr_{i}_{j}": f"prod_val_{i}_{j}" for j in range(9)
        }
    return left, right


def _make_dissimilar_nested_500() -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate 500-key deeply nested dissimilar pair."""
    left: dict[str, Any] = {}
    right: dict[str, Any] = {}
    for i in range(5):
        mid_l: dict[str, Any] = {}
        mid_r: dict[str, Any] = {}
        for j in range(5):
            mid_l[f"user_group_{j}"] = {
                f"user_field_{i}_{j}_{k}": f"u_val_{i}_{j}_{k}" for k in range(14)
            }
            mid_r[f"product_group_{j}"] = {
                f"product_attr_{i}_{j}_{k}": f"p_val_{i}_{j}_{k}" for k in range(14)
            }
        left[f"user_section_{i}"] = mid_l
        right[f"product_section_{i}"] = mid_r
    return left, right


# --- Fixtures for each size tier ---


@pytest.fixture
def pair_10key_similar() -> tuple[dict[str, Any], dict[str, Any]]:
    """10-key flat similar pair (snake_case vs camelCase)."""
    return _make_similar_flat(10)


@pytest.fixture
def pair_10key_dissimilar() -> tuple[dict[str, Any], dict[str, Any]]:
    """10-key flat dissimilar pair (different domains)."""
    return _make_dissimilar_flat(10)


@pytest.fixture
def pair_100key_similar() -> tuple[dict[str, Any], dict[str, Any]]:
    """100-key nested similar pair (10 sections x 9 leaf keys)."""
    return _make_similar_nested_100()


@pytest.fixture
def pair_100key_dissimilar() -> tuple[dict[str, Any], dict[str, Any]]:
    """100-key nested dissimilar pair (user vs product domains)."""
    return _make_dissimilar_nested_100()


@pytest.fixture
def pair_500key_similar() -> tuple[dict[str, Any], dict[str, Any]]:
    """500-key deeply nested similar pair (5 sections x 5 groups x ~15 leaves)."""
    return _make_similar_nested_500()


@pytest.fixture
def pair_500key_dissimilar() -> tuple[dict[str, Any], dict[str, Any]]:
    """500-key deeply nested dissimilar pair (user vs product domains)."""
    return _make_dissimilar_nested_500()
