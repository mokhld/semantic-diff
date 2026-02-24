"""EmbeddingCache: LRU-backed caching proxy for any EmbeddingBackend.

Wraps any EmbeddingBackend-conformant object and transparently caches
embedding results in memory. Cached strings bypass the backend on
subsequent ``embed()`` calls. LRU eviction occurs silently when
``max_size`` is exceeded — no error is raised.

Each ``EmbeddingCache`` instance maintains its own ``LRUCache`` — there is
no class-level shared state, so two separate instances never interfere
with each other.

Example::

    from json_semantic_diff.cache import EmbeddingCache
    from json_semantic_diff.backends import StaticBackend

    backend = StaticBackend()
    cache = EmbeddingCache(backend, max_size=512)

    # First call hits the backend
    vecs = cache.embed(["user_name", "address"])

    # Second call is fully served from memory — backend never called
    vecs_again = cache.embed(["user_name", "address"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from cachetools import LRUCache

if TYPE_CHECKING:
    from json_semantic_diff.protocols import EmbeddingBackend


class EmbeddingCache:
    """LRU-backed caching proxy around any EmbeddingBackend.

    Satisfies the ``EmbeddingBackend`` Protocol structurally (no inheritance
    required). Each instance maintains its own ``LRUCache`` — no cross-instance
    sharing. LRU eviction is silent: the least-recently-used entry is dropped
    when ``max_size`` is exceeded.

    Args:
        backend: Any object satisfying the ``EmbeddingBackend`` Protocol
            (has an ``embed(strings: list[str]) -> np.ndarray`` method).
        max_size: Maximum number of string embeddings to hold in memory.
            Defaults to 512. When exceeded, the least-recently-used entry
            is silently evicted.
    """

    def __init__(self, backend: EmbeddingBackend, max_size: int = 512) -> None:
        # Store as Any at runtime — structural duck-typing, no Protocol coupling.
        self._backend: Any = backend
        self._cache: LRUCache[str, np.ndarray] = LRUCache(maxsize=max_size)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_size(self) -> int:
        """The maximum number of entries this cache can hold."""
        return int(self._cache.maxsize)

    @property
    def curr_size(self) -> int:
        """The current number of entries stored in the cache."""
        return int(self._cache.currsize)

    # ------------------------------------------------------------------
    # EmbeddingBackend Protocol surface
    # ------------------------------------------------------------------

    def embed(self, strings: list[str]) -> np.ndarray:
        """Return embeddings for ``strings``; only uncached strings hit the backend.

        Maintains input order: row ``i`` of the returned array corresponds to
        ``strings[i]``. Individual row vectors (shape ``(D,)``) are stored in
        the cache; the full ``(N, D)`` matrix is reconstructed via
        ``np.stack`` on return.

        Args:
            strings: List of strings to embed. May be empty.

        Returns:
            Shape ``(N, D)`` float64 numpy array where ``N = len(strings)``.
        """
        results: dict[str, np.ndarray] = {}
        uncached = [s for s in strings if s not in self._cache]

        if uncached:
            embeddings: np.ndarray = self._backend.embed(uncached)
            for s, vec in zip(uncached, embeddings, strict=True):
                # Store individual row vectors (shape (D,)) — not the full matrix.
                # This prevents shape inconsistency between cached and uncached paths.
                self._cache[s] = vec
                results[s] = vec

        # Populate results for strings already in cache (not in uncached batch).
        for s in strings:
            if s not in results:
                results[s] = self._cache[s]

        # Stack in input order to produce consistent (N, D) shape.
        return np.stack([results[s] for s in strings])

    def similarity(self, a: str, b: str) -> float:
        """Return similarity score between two strings.

        If the wrapped backend exposes a ``similarity()`` method (e.g.
        ``StaticBackend`` with Levenshtein), delegates directly. This avoids
        the degenerate cosine issue with backends whose ``embed()`` returns
        non-discriminative representations (e.g. ``StaticBackend``'s ``(N,1)``
        stub arrays).

        For backends without ``similarity()`` (e.g. future ML backends in
        Phases 8/9), embeds both strings and returns their cosine similarity.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Float in [0.0, 1.0] representing semantic similarity.
        """
        if hasattr(self._backend, "similarity"):
            return float(self._backend.similarity(a, b))

        # Fallback: embed and compute cosine similarity.
        vecs = self.embed([a, b])
        dot = float(np.dot(vecs[0], vecs[1]))
        norm_a = float(np.linalg.norm(vecs[0]))
        norm_b = float(np.linalg.norm(vecs[1]))
        return dot / (norm_a * norm_b + 1e-9)
