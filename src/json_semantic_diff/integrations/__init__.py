"""Integrations subpackage for json-semantic-diff.

Contains integration adapters for external frameworks:
- pytest plugin (auto-discovered via pytest11 entry point)
- LangSmith evaluator adapter (LangSmithEvaluator)
- Braintrust scorer adapter (BraintrustScorer)
- W&B Weave scorer adapter (WeaveScorer)

Adapters with optional SDK dependencies (LangSmith, Weave) are imported
conditionally — a missing SDK does not prevent the package from loading.
BraintrustScorer has no SDK dependency and is always available.
"""

from __future__ import annotations

__all__: list[str] = []

# BraintrustScorer has no SDK dependency — always importable.
from json_semantic_diff.integrations._braintrust import BraintrustScorer

__all__.append("BraintrustScorer")

# LangSmith requires the langsmith SDK (pip install json-semantic-diff[langsmith]).
try:
    from json_semantic_diff.integrations._langsmith import LangSmithEvaluator

    __all__.append("LangSmithEvaluator")
except ImportError:
    pass

# Weave requires the weave SDK (pip install json-semantic-diff[weave]).
try:
    from json_semantic_diff.integrations._weave import WeaveScorer

    __all__.append("WeaveScorer")
except ImportError:
    pass

__all__ = sorted(__all__)
