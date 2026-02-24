"""Tests for StaticBackend discrimination, symmetry, identity, and edge cases.

Verifies that:
- Related JSON keys (same concept, different naming conventions) score > 0.85.
- Unrelated JSON keys score < 0.5.
- Similarity is symmetric and identity returns 1.0.
- Edge cases (empty strings, single chars) are handled gracefully.
- The internal _levenshtein_distance function is correct on known cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from semantic_diff.backends import StaticBackend
from semantic_diff.backends.static import _levenshtein_distance

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend() -> StaticBackend:
    return StaticBackend()


# ---------------------------------------------------------------------------
# Discrimination tests
# ---------------------------------------------------------------------------


def test_related_keys_score_high(backend: StaticBackend) -> None:
    """user_name and userName both normalize to 'user name' — score should be 1.0."""
    score = backend.similarity("user_name", "userName")
    assert score > 0.85, f"Expected > 0.85, got {score}"


def test_unrelated_keys_score_low(backend: StaticBackend) -> None:
    """user_name and address are semantically unrelated — score should be < 0.5."""
    score = backend.similarity("user_name", "address")
    assert score < 0.5, f"Expected < 0.5, got {score}"


def test_similarity_discriminates(backend: StaticBackend) -> None:
    """Related key pair scores higher than unrelated key pair."""
    related = backend.similarity("user_name", "userName")
    unrelated = backend.similarity("user_name", "address")
    assert related > unrelated, (
        f"Related ({related:.4f}) should exceed unrelated ({unrelated:.4f})"
    )


def test_all_four_conventions_equivalent(backend: StaticBackend) -> None:
    """All naming conventions for 'user name' normalize identically — pairwise > 0.99."""
    variants = ["userName", "user_name", "UserName", "user-name"]
    for i, a in enumerate(variants):
        for b in variants[i + 1 :]:
            score = backend.similarity(a, b)
            assert score > 0.99, (
                f"Expected pairwise similarity > 0.99 for {a!r}/{b!r}, got {score}"
            )


# ---------------------------------------------------------------------------
# Symmetry and identity tests
# ---------------------------------------------------------------------------


def test_similarity_is_symmetric(backend: StaticBackend) -> None:
    """similarity(a, b) == similarity(b, a) for all tested pairs."""
    pairs = [
        ("user_name", "userName"),
        ("address", "city"),
        ("firstName", "last_name"),
        ("id", "identifier"),
        ("", "hello"),
    ]
    for a, b in pairs:
        forward = backend.similarity(a, b)
        backward = backend.similarity(b, a)
        assert abs(forward - backward) < 1e-9, (
            f"Asymmetry detected for {a!r}/{b!r}: {forward} vs {backward}"
        )


def test_identical_strings_score_one(backend: StaticBackend) -> None:
    """similarity(a, a) == 1.0 for any string."""
    for key in ["user_name", "address", "id", "createdAt", ""]:
        score = backend.similarity(key, key)
        assert score == 1.0, f"Expected 1.0 for identical pair {key!r}, got {score}"


def test_empty_strings_score_one(backend: StaticBackend) -> None:
    """similarity("", "") == 1.0 — both normalize to empty string."""
    assert backend.similarity("", "") == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_char_strings(backend: StaticBackend) -> None:
    """similarity("a", "b") returns a float in [0.0, 1.0]."""
    score = backend.similarity("a", "b")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"


def test_completely_different_long_strings(backend: StaticBackend) -> None:
    """Very different long strings should score low."""
    score = backend.similarity("abcdefgh", "zyxwvuts")
    assert score < 0.5, f"Expected low score for unrelated strings, got {score}"


def test_embed_returns_correct_shape(backend: StaticBackend) -> None:
    """embed(["a", "b", "c"]) returns shape (3, 1)."""
    result = backend.embed(["a", "b", "c"])
    assert result.shape == (3, 1), f"Expected (3, 1), got {result.shape}"


def test_embed_returns_float64(backend: StaticBackend) -> None:
    """embed(["a"]) returns dtype float64."""
    result = backend.embed(["a"])
    assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"


# ---------------------------------------------------------------------------
# Levenshtein correctness tests
# ---------------------------------------------------------------------------


def test_levenshtein_known_distances() -> None:
    """Verify _levenshtein_distance against well-known reference cases."""
    assert _levenshtein_distance("kitten", "sitting") == 3
    assert _levenshtein_distance("", "abc") == 3
    assert _levenshtein_distance("abc", "") == 3
    assert _levenshtein_distance("abc", "abc") == 0
    assert _levenshtein_distance("", "") == 0
    assert _levenshtein_distance("a", "b") == 1


def test_levenshtein_is_symmetric() -> None:
    """_levenshtein_distance(a, b) == _levenshtein_distance(b, a) for multiple pairs."""
    pairs = [
        ("kitten", "sitting"),
        ("abc", "xyz"),
        ("hello", ""),
        ("user name", "address"),
    ]
    for a, b in pairs:
        assert _levenshtein_distance(a, b) == _levenshtein_distance(b, a), (
            f"Asymmetric result for {a!r}/{b!r}"
        )
