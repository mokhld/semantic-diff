"""Unit tests for EmbeddingCache.

Tests cover:
- Cache hits (cached strings bypass backend on second embed() call)
- Partial cache hits (only uncached strings flow through to backend)
- LRU eviction (silent eviction at max_size; evicted strings re-embed on next call)
- Instance isolation (separate EmbeddingCache instances do not share state)
- Similarity delegation (delegates to backend.similarity() when available)
- Cosine fallback for similarity (when backend has no similarity() method)
- Input order preservation (result row i corresponds to strings[i])
- Properties (max_size and curr_size return correct values)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from semantic_diff.backends import StaticBackend
from semantic_diff.cache import EmbeddingCache

# ---------------------------------------------------------------------------
# Spy helper
# ---------------------------------------------------------------------------


def _make_spy_backend(backend: StaticBackend) -> tuple[StaticBackend, list[list[str]]]:
    """Monkey-patch *backend.embed* with a spy that records call arguments.

    Returns (backend, call_log) where call_log collects each list[str] passed
    to embed().  The spy delegates to the original implementation so results
    are unchanged.
    """
    call_log: list[list[str]] = []
    original_embed = backend.embed

    def spy_embed(strings: list[str]) -> np.ndarray:
        call_log.append(list(strings))
        return original_embed(strings)

    backend.embed = spy_embed  # type: ignore[method-assign]
    return backend, call_log


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCachedStringsNotReEmbedded:
    """Cached strings must never hit the backend on subsequent embed() calls."""

    def test_no_backend_calls_on_second_embed(self) -> None:
        raw_backend = StaticBackend()
        spy_backend, call_log = _make_spy_backend(raw_backend)

        cache = EmbeddingCache(spy_backend)
        # First call — populates cache
        cache.embed(["hello", "world"])
        call_log.clear()

        # Second call — must serve entirely from cache
        result = cache.embed(["hello", "world"])

        assert len(call_log) == 0, (
            f"Expected 0 backend embed calls on second invocation, got {len(call_log)}"
        )
        assert result.shape == (2, 1)  # StaticBackend returns (N, 1)


class TestPartialCacheHit:
    """Only uncached strings should be forwarded to the backend."""

    def test_only_uncached_strings_sent_to_backend(self) -> None:
        raw_backend = StaticBackend()
        spy_backend, call_log = _make_spy_backend(raw_backend)

        cache = EmbeddingCache(spy_backend)
        # Seed cache with "a" and "b"
        cache.embed(["a", "b"])
        call_log.clear()

        # "b" is cached; "c" is not
        cache.embed(["b", "c"])

        assert len(call_log) == 1, f"Expected 1 backend call, got {len(call_log)}"
        assert call_log[0] == ["c"], (
            f"Expected only 'c' sent to backend, got {call_log[0]}"
        )


class TestLRUEviction:
    """LRU eviction should occur silently; evicted strings are re-embedded."""

    def test_eviction_does_not_raise(self) -> None:
        raw_backend = StaticBackend()
        cache = EmbeddingCache(raw_backend, max_size=3)

        cache.embed(["a", "b", "c"])
        assert cache.curr_size == 3

        # Access "a" so it becomes recently used; "b" becomes LRU
        cache.embed(["a"])
        # Embed "d" — should evict "b" (LRU)
        cache.embed(["d"])

        # No error; cache still has 3 entries
        assert cache.curr_size == 3

    def test_evicted_string_re_embedded_by_backend(self) -> None:
        raw_backend = StaticBackend()
        spy_backend, call_log = _make_spy_backend(raw_backend)

        cache = EmbeddingCache(spy_backend, max_size=3)
        cache.embed(["a", "b", "c"])
        # Access "a" so "b" is LRU
        cache.embed(["a"])
        # Evict "b" by inserting "d"
        cache.embed(["d"])
        call_log.clear()

        # "b" was evicted — must go back to backend
        cache.embed(["b"])

        assert len(call_log) == 1, (
            f"Expected 1 backend call for evicted 'b', got {len(call_log)}"
        )
        assert call_log[0] == ["b"]


class TestInstanceIsolation:
    """Two EmbeddingCache instances must not share cache state."""

    def test_separate_instances_have_isolated_caches(self) -> None:
        backend1 = StaticBackend()
        backend2 = StaticBackend()

        cache1 = EmbeddingCache(backend1)
        cache2 = EmbeddingCache(backend2)

        # Populate cache1 with "x"
        cache1.embed(["x"])

        # cache2 must remain empty
        assert cache2.curr_size == 0, (
            f"Expected cache2.curr_size == 0, got {cache2.curr_size}"
        )


class TestSimilarityDelegation:
    """similarity() must delegate to backend.similarity() when available."""

    def test_similarity_delegates_to_backend(self) -> None:
        backend = StaticBackend()
        cache = EmbeddingCache(backend)

        # EmbeddingCache.similarity must match the backend directly
        expected = backend.similarity("user_name", "userName")
        actual = cache.similarity("user_name", "userName")

        assert actual == pytest.approx(expected, abs=1e-9)

    def test_similarity_with_identical_keys(self) -> None:
        backend = StaticBackend()
        cache = EmbeddingCache(backend)

        # Identical normalized keys should yield 1.0
        result = cache.similarity("user_name", "user_name")
        assert result == pytest.approx(1.0, abs=1e-9)


class TestSimilarityCosineFallback:
    """similarity() must compute cosine when backend has no similarity() method."""

    def test_cosine_fallback_identical_strings(self) -> None:
        """Cosine of identical vectors must be close to 1.0."""

        class _NoSimilarityBackend:
            """Minimal backend: embed() only, no similarity()."""

            def embed(self, strings: list[str]) -> np.ndarray:
                # Fixed unit vector in 3D — identical for all strings
                vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                return np.stack([vec for _ in strings])

        backend: Any = _NoSimilarityBackend()
        cache = EmbeddingCache(backend)

        result = cache.similarity("hello", "hello")

        assert not hasattr(backend, "similarity"), (
            "Backend must lack similarity() for this test"
        )
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_cosine_fallback_orthogonal_vectors(self) -> None:
        """Cosine of orthogonal vectors must be close to 0.0."""
        _vecs: dict[str, np.ndarray] = {
            "a": np.array([1.0, 0.0], dtype=np.float64),
            "b": np.array([0.0, 1.0], dtype=np.float64),
        }

        class _OrthogonalBackend:
            """Returns orthogonal unit vectors for "a" and "b"."""

            def embed(self, strings: list[str]) -> np.ndarray:
                return np.stack([_vecs[s] for s in strings])

        backend: Any = _OrthogonalBackend()
        cache = EmbeddingCache(backend)

        result = cache.similarity("a", "b")
        assert result == pytest.approx(0.0, abs=1e-6)


class TestEmbedPreservesInputOrder:
    """Row i of the returned array must correspond to strings[i]."""

    def test_input_order_preserved(self) -> None:
        backend = StaticBackend()
        cache = EmbeddingCache(backend)

        # Get reference embeddings for each string individually
        vec_c = backend.embed(["c"])[0]
        vec_a = backend.embed(["a"])[0]
        vec_b = backend.embed(["b"])[0]

        # Embed in a non-alphabetical order
        result = cache.embed(["c", "a", "b"])

        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result[0], vec_c, err_msg="Row 0 must be 'c'")
        np.testing.assert_array_equal(result[1], vec_a, err_msg="Row 1 must be 'a'")
        np.testing.assert_array_equal(result[2], vec_b, err_msg="Row 2 must be 'b'")

    def test_input_order_preserved_after_partial_cache_hit(self) -> None:
        """Order must be maintained even when some strings are cached and some are not."""
        backend = StaticBackend()
        cache = EmbeddingCache(backend)

        # Pre-cache "a" and "c"
        cache.embed(["a", "c"])
        # Now embed ["c", "b", "a"] — "b" is uncached, "a"/"c" are cached
        result = cache.embed(["c", "b", "a"])

        ref_c = backend.embed(["c"])[0]
        ref_b = backend.embed(["b"])[0]
        ref_a = backend.embed(["a"])[0]

        np.testing.assert_array_equal(result[0], ref_c, err_msg="Row 0 must be 'c'")
        np.testing.assert_array_equal(result[1], ref_b, err_msg="Row 1 must be 'b'")
        np.testing.assert_array_equal(result[2], ref_a, err_msg="Row 2 must be 'a'")


class TestProperties:
    """max_size and curr_size properties must report correct values."""

    def test_max_size_reflects_configured_value(self) -> None:
        backend = StaticBackend()
        cache = EmbeddingCache(backend, max_size=128)

        assert cache.max_size == 128

    def test_curr_size_starts_at_zero(self) -> None:
        backend = StaticBackend()
        cache = EmbeddingCache(backend)

        assert cache.curr_size == 0

    def test_curr_size_increases_after_embedding(self) -> None:
        backend = StaticBackend()
        cache = EmbeddingCache(backend, max_size=512)

        assert cache.curr_size == 0
        cache.embed(["alpha", "beta"])
        assert cache.curr_size == 2
        cache.embed(["gamma"])
        assert cache.curr_size == 3

    def test_curr_size_does_not_exceed_max_size(self) -> None:
        backend = StaticBackend()
        cache = EmbeddingCache(backend, max_size=2)

        cache.embed(["x", "y", "z"])  # z evicts x (LRU)
        assert cache.curr_size == 2
        assert cache.curr_size <= cache.max_size
