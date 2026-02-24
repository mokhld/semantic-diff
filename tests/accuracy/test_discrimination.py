"""Discrimination benchmark tests for FastEmbedBackend model validation.

These tests validate that all-MiniLM-L6-v2 (the default FastEmbedBackend model
after Phase 8 benchmark) can distinguish semantically related key names
(>0.90) from unrelated ones (<0.75) through the full STEDComparator stack.

**Threshold rationale (Phase 8 empirical finding):**
The STED algorithm with a single-key object and equal values (e.g.
``{"key_a": "x"}`` vs ``{"key_b": "x"}``) has a mathematically bounded
score range of [0.5, 1.0] -- the minimum possible score is 0.5 regardless
of key similarity, because content distance is 0 and the normalization
formula is ``1 - w_s * (1 - key_sim) = 1 - 0.5 * (1 - key_sim)``.  This
means the "< 0.1" threshold from the ROADMAP SC2 is physically unreachable
for single-key equal-value comparisons.  The correct thresholds are:

- Related keys (naming-convention equivalents): score > 0.90
  Actual: all-MiniLM-L6-v2 scores 1.0 for all camelCase/snake_case pairs.
- Unrelated keys (semantically different): score < 0.75
  Actual: all-MiniLM-L6-v2 scores 0.58-0.71 for the unrelated pairs.
- Discrimination gap: >= 0.25
  Actual: ~0.29 single-pair gap, ~0.36 multi-pair average.

These thresholds are validated empirically against all-MiniLM-L6-v2 and
documented in PROJECT.md Key Decisions.

All tests skip cleanly when fastembed is not installed.
"""

import pytest

pytest.importorskip(
    "fastembed", reason="fastembed extra not installed — skip accuracy tests"
)

from json_semantic_diff.backends.fastembed import FastEmbedBackend
from json_semantic_diff.comparator import STEDComparator


@pytest.fixture(scope="module")
def fastembed_comparator() -> STEDComparator:
    """Single FastEmbedBackend instance shared across all discrimination tests.

    Using module scope avoids repeated ONNX model initialization (~1-2s each).
    The EmbeddingCache inside STEDComparator is also reused, so subsequent
    tests benefit from cached embeddings.
    """
    return STEDComparator(backend=FastEmbedBackend())


class TestRelatedKeysAboveThreshold:
    """SC2 (revised): Semantically equivalent key names must score > 0.90.

    All naming-convention pairs (snake_case vs camelCase) score 1.0 with
    all-MiniLM-L6-v2 through the full STED stack.  The threshold is 0.90
    (vs the original ROADMAP 0.85) to reflect the empirically observed scores.
    """

    @pytest.mark.parametrize(
        ("left_key", "right_key"),
        [
            ("user_name", "userName"),
            ("first_name", "firstName"),
            ("email_address", "emailAddress"),
            ("phone_number", "phoneNumber"),
            ("created_at", "createdAt"),
        ],
    )
    def test_related_keys_score_above_090(
        self,
        fastembed_comparator: STEDComparator,
        left_key: str,
        right_key: str,
    ) -> None:
        result = fastembed_comparator.compare({left_key: "x"}, {right_key: "x"})
        assert result.similarity_score > 0.90, (
            f"Related keys '{left_key}' vs '{right_key}' "
            f"scored {result.similarity_score:.4f} (expected > 0.90)"
        )


class TestUnrelatedKeysBelowThreshold:
    """SC2 (revised): Structurally different key names must score < 0.75.

    The STED algorithm's minimum possible score for single-key equal-value
    objects is 0.5 (mathematical lower bound).  The empirical upper bound
    for semantically unrelated keys via all-MiniLM-L6-v2 is ~0.72, so 0.75
    is a conservative but achievable threshold.
    """

    @pytest.mark.parametrize(
        ("left_key", "right_key"),
        [
            ("user_name", "address"),
            ("email", "age"),
            ("created_at", "price"),
            ("first_name", "total_amount"),
            ("phone_number", "description"),
        ],
    )
    def test_unrelated_keys_score_below_075(
        self,
        fastembed_comparator: STEDComparator,
        left_key: str,
        right_key: str,
    ) -> None:
        result = fastembed_comparator.compare({left_key: "x"}, {right_key: "x"})
        assert result.similarity_score < 0.75, (
            f"Unrelated keys '{left_key}' vs '{right_key}' "
            f"scored {result.similarity_score:.4f} (expected < 0.75)"
        )


class TestDiscriminationGap:
    """SC3 (revised): Gap between related and unrelated must be >= 0.25.

    The original ROADMAP SC3 required gap >= 0.5, which assumed the STED
    algorithm would produce near-zero scores for unrelated keys.  The correct
    lower bound after empirical validation is 0.25: all-MiniLM-L6-v2 produces
    a single-pair gap of ~0.29 and a multi-pair average gap of ~0.36.
    """

    def test_discrimination_gap_gate(
        self, fastembed_comparator: STEDComparator
    ) -> None:
        """If this test fails, the default model must be re-evaluated."""
        related = fastembed_comparator.compare(
            {"user_name": "x"}, {"userName": "x"}
        ).similarity_score
        unrelated = fastembed_comparator.compare(
            {"user_name": "x"}, {"address": "x"}
        ).similarity_score
        gap = related - unrelated
        assert gap >= 0.25, (
            f"Discrimination gap {gap:.4f} < 0.25 threshold -- "
            "all-MiniLM-L6-v2 fails short-phrase key discrimination. "
            "Re-evaluate default model selection."
        )

    def test_gap_across_multiple_pairs(
        self, fastembed_comparator: STEDComparator
    ) -> None:
        """Average gap across multiple pairs must also be >= 0.25."""
        related_pairs = [
            ("user_name", "userName"),
            ("first_name", "firstName"),
            ("email_address", "emailAddress"),
        ]
        unrelated_pairs = [
            ("user_name", "address"),
            ("email", "age"),
            ("created_at", "price"),
        ]
        related_scores = [
            fastembed_comparator.compare({lk: "x"}, {rk: "x"}).similarity_score
            for lk, rk in related_pairs
        ]
        unrelated_scores = [
            fastembed_comparator.compare({lk: "x"}, {rk: "x"}).similarity_score
            for lk, rk in unrelated_pairs
        ]
        avg_related = sum(related_scores) / len(related_scores)
        avg_unrelated = sum(unrelated_scores) / len(unrelated_scores)
        avg_gap = avg_related - avg_unrelated
        assert avg_gap >= 0.25, (
            f"Average discrimination gap {avg_gap:.4f} < 0.25 -- "
            f"related avg: {avg_related:.4f}, unrelated avg: {avg_unrelated:.4f}"
        )
