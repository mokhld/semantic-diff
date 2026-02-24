"""Braintrust scorer adapter for json-semantic-diff.

Provides a factory function ``BraintrustScorer`` that wraps an
``STEDComparator`` in the Braintrust scorer interface (plain function pattern).

No Braintrust SDK import is required — the scorer is a plain Python function
with the signature that Braintrust expects.  Install Braintrust separately
if you also need ``braintrust.Eval()`` runner support::

    pip install json-semantic-diff[braintrust]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from json_semantic_diff.comparator import STEDComparator

__all__ = ["BraintrustScorer"]


def BraintrustScorer(
    comparator: STEDComparator,
) -> Any:
    """Create a Braintrust-compatible scorer function.

    The returned callable follows Braintrust's scorer interface.  It requires
    no Braintrust SDK imports — it is a plain function factory.

    Args:
        comparator: An ``STEDComparator`` instance to use for scoring.

    Returns:
        A callable ``_scorer(input, output, expected=None, metadata=None)``
        that returns a ``float`` in ``[0.0, 1.0]`` or ``None`` when no
        ``expected`` is provided.  The function name is set to
        ``"semantic_similarity"`` for Braintrust display purposes.
    """

    def _scorer(
        input: Any,
        output: Any,
        expected: Any = None,
        metadata: Any = None,
    ) -> float | None:
        if expected is None:
            return None
        return comparator.compare(output, expected).similarity_score

    _scorer.__name__ = "semantic_similarity"
    return _scorer
