"""LangSmith evaluator adapter for semantic-diff.

Provides a factory function ``LangSmithEvaluator`` that wraps an
``STEDComparator`` in the LangSmith evaluator interface (function-based
pattern).  The returned callable is compatible with ``langsmith.evaluate()``.

Install the optional SDK dependency with::

    pip install semantic-diff[langsmith]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from semantic_diff.comparator import STEDComparator

__all__ = ["LangSmithEvaluator"]


def LangSmithEvaluator(
    comparator: STEDComparator,
    output_key: str = "output",
) -> Any:
    """Create a LangSmith-compatible evaluator function.

    The returned callable follows the function-based LangSmith evaluator
    pattern (NOT the RunEvaluator ABC pattern) to avoid class-definition-time
    imports.  All LangSmith SDK imports are lazy — base-install users who do
    not have langsmith installed will only see an ImportError when this factory
    is called, not at module import time.

    Args:
        comparator: An ``STEDComparator`` instance to use for scoring.
        output_key: Key to extract from ``run.outputs`` and ``example.outputs``.
            Defaults to ``"output"``.

    Returns:
        A callable ``_evaluator(run, example=None)`` that returns a LangSmith
        ``EvaluationResult`` with ``key="semantic_similarity"`` and a score in
        ``[0.0, 1.0]``.

    Raises:
        ImportError: If ``langsmith`` is not installed.
    """
    try:
        import langsmith  # noqa: F401
        from langsmith.evaluation.evaluator import EvaluationResult
    except ImportError as exc:
        raise ImportError(
            "langsmith is required: pip install semantic-diff[langsmith]"
        ) from exc

    def _evaluator(run: Any, example: Any = None) -> Any:
        actual: Any = (run.outputs or {}).get(output_key, "")
        expected: Any = (example.outputs or {}).get(output_key, "") if example else ""
        score: float = comparator.compare(actual, expected).similarity_score
        return EvaluationResult(key="semantic_similarity", score=score)

    return _evaluator
