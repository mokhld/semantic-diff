"""EmbeddingBackend Protocol for json-semantic-diff backend extension point.

Defines the structural interface all embedding backends must satisfy.
Users can plug in custom backends without inheriting from any base class —
any class with a conformant ``embed`` method passes ``isinstance`` checks.

Example::

    import numpy as np
    from json_semantic_diff.protocols import EmbeddingBackend

    class MyBackend:
        def embed(self, strings: list[str]) -> np.ndarray:
            # Return shape (N, D) float64 embedding matrix
            return np.zeros((len(strings), 768))

    assert isinstance(MyBackend(), EmbeddingBackend)  # True — structural conformance
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Structural protocol for embedding backends.

    Any class implementing ``embed(self, strings: list[str]) -> np.ndarray``
    satisfies this protocol at runtime — no inheritance required.

    The ``embed`` method must:
    - Accept a list of strings as input.
    - Return a 2-D numpy array of shape ``(len(strings), D)`` for some embedding
      dimension ``D >= 1``.
    - Return dtype ``float64`` (or compatible floating-point dtype).
    """

    def embed(self, strings: list[str]) -> np.ndarray: ...
