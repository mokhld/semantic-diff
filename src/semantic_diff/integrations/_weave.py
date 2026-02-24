"""W&B Weave scorer adapter for semantic-diff.

Provides a factory function ``WeaveScorer`` that wraps an
``STEDComparator`` in the W&B Weave scorer interface (``weave.Scorer``
subclass pattern).

The ``weave.Scorer`` base class and ``@weave.op`` decorator are imported
lazily inside the factory function — base-install users who do not have
``weave`` installed will only see an ImportError when this factory is called,
not at module import time.

Install the optional SDK dependency with::

    pip install semantic-diff[weave]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from semantic_diff.comparator import STEDComparator

__all__ = ["WeaveScorer"]


def WeaveScorer(
    comparator: STEDComparator,
) -> Any:
    """Create a W&B Weave-compatible scorer instance.

    The returned object is a ``weave.Scorer`` subclass instance whose
    ``score()`` method is decorated with ``@weave.op`` for Weave tracking.

    The class is defined inside the factory to keep all ``weave`` imports
    lazy — only triggered when ``WeaveScorer()`` is called, not at module
    load time.

    Args:
        comparator: An ``STEDComparator`` instance to use for scoring.

    Returns:
        An instance of a ``weave.Scorer`` subclass with a ``score()`` method
        that returns ``{"semantic_similarity": float}``.

    Raises:
        ImportError: If ``weave`` is not installed.
    """
    try:
        import weave
        from weave import Scorer
    except ImportError as exc:
        raise ImportError(
            "weave is required: pip install semantic-diff[weave]"
        ) from exc

    class _WeaveSTEDScorer(Scorer):  # type: ignore[misc]
        @weave.op  # type: ignore[untyped-decorator]
        def score(self, output: Any, target: Any = None) -> dict[str, float]:
            if target is None:
                return {"semantic_similarity": 0.0}
            score: float = comparator.compare(output, target).similarity_score
            return {"semantic_similarity": score}

    return _WeaveSTEDScorer()
