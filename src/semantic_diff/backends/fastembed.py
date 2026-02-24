"""FastEmbedBackend: ONNX embedding backend via the fastembed library.

Wraps ``fastembed.TextEmbedding`` with a lazy import so that the base install
(no fastembed installed) never triggers an ``ImportError`` at module level.
The ``fastembed`` package is only required when ``FastEmbedBackend`` is
*instantiated*.

Install the optional dependency with::

    pip install semantic-diff[fastembed]

Example::

    from semantic_diff.backends.fastembed import FastEmbedBackend

    backend = FastEmbedBackend()
    vecs = backend.embed(["user_name", "address"])
    print(vecs.shape)   # (2, 384)
    print(vecs.dtype)   # float32

**Model selection note:**
``BAAI/bge-small-en-v1.5`` was evaluated against a key-name discrimination
benchmark and produced a gap of ~0.16 (below the 0.25 threshold).
``sentence-transformers/all-MiniLM-L6-v2`` was substituted as the default and
produces a gap of ~0.29 with all related naming-convention pairs scoring 1.0
and all unrelated pairs scoring below 0.72 through the full STED stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FastEmbedBackend:
    """ONNX embedding backend wrapping ``fastembed.TextEmbedding``.

    Performs a lazy import of ``fastembed`` inside ``__init__``, so importing
    this module on a base install (no fastembed) does not raise
    ``ImportError``.  The error is deferred until the class is *instantiated*.

    Args:
        model_name: HuggingFace model identifier supported by fastembed.
            Defaults to ``"sentence-transformers/all-MiniLM-L6-v2"`` (384-dim,
            ONNX-optimized, Apache-2.0).  This was selected over
            ``"BAAI/bge-small-en-v1.5"`` after a discrimination benchmark
            found bge-small-en-v1.5 failed the gap threshold.
        intra_op_num_threads: Number of ONNX Runtime intra-op threads.
            Pass ``1`` to prevent thread explosion in multi-worker environments.
            ``None`` (default) lets fastembed/ONNX choose automatically.

    Raises:
        ImportError: If ``fastembed`` is not installed.  The message includes
            the install command.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        intra_op_num_threads: int | None = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for FastEmbedBackend. "
                "Install it with: pip install semantic-diff[fastembed]"
            ) from exc

        # Use Any annotation to avoid NameError when fastembed is absent at
        # type-check time (fastembed has no bundled stub).
        self._model: Any = TextEmbedding(
            model_name=model_name,
            threads=intra_op_num_threads,
        )
        self._model_name = model_name

    def embed(self, strings: list[str]) -> np.ndarray:
        """Return embeddings for ``strings`` as a float32 ndarray.

        FastEmbed's ``TextEmbedding.embed()`` returns a generator of
        ``(D,)`` float32 arrays — this method materialises the generator and
        stacks the rows into an ``(N, D)`` matrix.

        Args:
            strings: Input strings to embed.  May be empty.

        Returns:
            Shape ``(N, D)`` numpy array with ``dtype=float32`` where
            ``N = len(strings)`` and ``D`` is the model's embedding dimension
            (e.g. 384 for ``all-MiniLM-L6-v2`` or ``bge-small-en-v1.5``).
        """
        if not strings:
            return np.empty((0, 384), dtype=np.float32)

        # embed() returns a generator of (D,) float32 arrays — must materialise.
        vectors = list(self._model.embed(strings))
        return np.stack(vectors).astype(np.float32)


# ---------------------------------------------------------------------------
# Module-level pre-warm singleton
# ---------------------------------------------------------------------------
# Instantiating FastEmbedBackend triggers ONNX model loading, which is the
# dominant cold-start cost (~1-2 s). Doing this at import time amortises the
# cost to a one-time hit rather than the first call inside a running program.
# The try/except ensures base installs (no fastembed) are completely unaffected.

try:
    _DEFAULT_BACKEND = FastEmbedBackend()
except ImportError:
    _DEFAULT_BACKEND = None  # type: ignore[assignment]
