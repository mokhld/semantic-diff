"""OpenAIBackend: Cloud embedding backend via the OpenAI embeddings API.

Wraps ``openai.OpenAI`` with a lazy import so that the base install
(no openai/tenacity installed) never triggers an ``ImportError`` at module
level.  The ``openai`` and ``tenacity`` packages are only required when
``OpenAIBackend`` is *instantiated*.

The API key is read exclusively from the ``OPENAI_API_KEY`` environment
variable.  It is never accepted as a constructor parameter and never appears
in ``repr()``, ``str()``, or log output.

Rate-limited requests (HTTP 429) are retried automatically with jittered
exponential backoff via ``tenacity``.  The underlying ``openai.OpenAI`` client
is created with ``max_retries=0`` to prevent double-retry.

Install the optional dependency with::

    pip install semantic-diff[openai]

Example::

    from semantic_diff.backends.openai import OpenAIBackend

    backend = OpenAIBackend()
    vecs = backend.embed(["user_name", "address"])
    print(vecs.shape)   # (2, 1536)
    print(vecs.dtype)   # float32
"""

from __future__ import annotations

from typing import Any

import numpy as np


class OpenAIBackend:
    """OpenAI embedding backend using ``text-embedding-3-small``.

    Reads the API key exclusively from the ``OPENAI_API_KEY`` environment
    variable.  The key never appears in ``repr()``, ``str()``, or log output.

    Performs a lazy import of ``openai`` and ``tenacity`` inside ``__init__``,
    so importing this module on a base install does not raise
    ``ImportError``.  The error is deferred until the class is *instantiated*.

    Rate-limited API calls are retried with jittered exponential backoff via
    tenacity (up to 6 attempts).  The underlying ``openai.OpenAI`` client is
    created with ``max_retries=0`` to prevent double-retry.

    Args:
        model_name: OpenAI embedding model identifier.  Defaults to
            ``"text-embedding-3-small"`` (1536-dim, best quality/cost trade-off
            per OpenAI recommendation).

    Raises:
        ImportError: If ``openai`` or ``tenacity`` is not installed.  The
            message includes the install command.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
    ) -> None:
        try:
            from openai import OpenAI, RateLimitError
            from tenacity import (
                retry,
                retry_if_exception_type,
                stop_after_attempt,
                wait_random_exponential,
            )
        except ImportError as exc:
            raise ImportError(
                "openai and tenacity are required for OpenAIBackend. "
                "Install with: pip install semantic-diff[openai]"
            ) from exc

        self._model_name = model_name
        # max_retries=0: tenacity is the sole retry controller.
        # The SDK default (max_retries=2) would create double-retry:
        # up to 6 (tenacity) * 3 (SDK) = 18 HTTP calls per embed().
        # Use Any annotation: openai is a lazy import, not available at
        # class definition time for type resolution.
        self._client: Any = OpenAI(max_retries=0)

        # Build the retry decorator after RateLimitError is in scope.
        # @retry cannot reference RateLimitError at class definition time
        # because openai is not imported until __init__ runs.
        _retry = retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(6),
        )
        # Wrap the raw API call with the retry decorator.
        self._call_api = _retry(self._raw_call)

    def __repr__(self) -> str:
        """Return a safe repr that never exposes the API key."""
        return f"OpenAIBackend(model={self._model_name!r})"

    def embed(self, strings: list[str]) -> np.ndarray:
        """Return embeddings for ``strings`` as a float32 (N, 1536) ndarray.

        Args:
            strings: Input strings to embed.  May be empty.

        Returns:
            Shape ``(N, 1536)`` numpy array with ``dtype=float32`` where
            ``N = len(strings)``.  Returns an empty ``(0, 1536)`` array
            without making any API calls when ``strings`` is empty.
        """
        if not strings:
            return np.empty((0, 1536), dtype=np.float32)
        return self._call_api(strings)  # type: ignore[no-any-return]

    def _raw_call(self, strings: list[str]) -> np.ndarray:
        """Make the raw embeddings API call — retried by tenacity via ``_call_api``.

        Args:
            strings: Non-empty list of strings to embed.

        Returns:
            Shape ``(N, 1536)`` numpy array with ``dtype=float32``.
        """
        response = self._client.embeddings.create(
            model=self._model_name,
            input=strings,
        )
        # Sort by index defensively — API guarantees input-order but
        # sorting prevents subtle bugs if that assumption ever changes.
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return np.array(
            [item.embedding for item in sorted_data],
            dtype=np.float32,
        )
