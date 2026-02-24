"""Backends subpackage for json-semantic-diff.

The base install provides only ``StaticBackend`` — a zero-ML-dependency
Levenshtein-based backend.  Optional backends (FastEmbed, OpenAI) are
available via extras:

    pip install json-semantic-diff[fastembed]   # FastEmbed ONNX backend
    pip install json-semantic-diff[openai]      # OpenAI embeddings backend

All backends satisfy the ``EmbeddingBackend`` Protocol structurally.
"""

from json_semantic_diff.backends.static import StaticBackend

# __all__ lists the names that are always available at import time.
# FastEmbedBackend and OpenAIBackend are conditionally imported below and
# added to __all__ only when their optional dependencies are installed.
# The list is kept alphabetically sorted (RUF022 compliance).
__all__ = ["StaticBackend"]

try:
    from json_semantic_diff.backends.fastembed import FastEmbedBackend

    __all__ = sorted([*__all__, "FastEmbedBackend"])
except ImportError:
    pass

try:
    from json_semantic_diff.backends.openai import OpenAIBackend

    __all__ = sorted([*__all__, "OpenAIBackend"])
except ImportError:
    pass
