"""Accuracy benchmark suite for STED algorithm validation.

Validates PACK-02: the library's discrimination claims are backed by statistical
evidence measured against a human-judged ground truth dataset.

The dataset contains ~45 hand-curated JSON pairs spanning five categories:
- identical: exact matches (expected score 1.0)
- key_renaming: camelCase/snake_case differences (expected score 0.70-1.0)
- structural_differences: extra/missing keys, value changes (expected score 0.05-0.90)
- unrelated: completely different structures (expected score 0.0-0.10)
- edge_cases: null, empty, deep nesting (expected score 0.0-1.0)

Statistical methodology:
- Pearson correlation between expected and actual scores must exceed 0.85
- Precision at the 0.85 equivalence threshold must exceed 0.90
- Category ordering must hold: identical > key_renaming > structural > unrelated

All tests use the default StaticBackend (Levenshtein on normalized keys) for
deterministic, CI-friendly execution with zero ML dependencies.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from scipy.stats import pearsonr

from json_semantic_diff import compare

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def similarity_pairs() -> list[dict]:  # type: ignore[type-arg]
    """Load the human-judged similarity pairs dataset."""
    data = json.loads((FIXTURES_DIR / "similarity_pairs.json").read_text())
    return data["pairs"]  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def scored_pairs(similarity_pairs: list[dict]) -> list[dict]:  # type: ignore[type-arg]
    """Compute actual scores for all pairs using StaticBackend."""
    results = []
    for pair in similarity_pairs:
        actual = compare(pair["left"], pair["right"]).similarity_score
        results.append(
            {
                **pair,
                "actual_score": actual,
            }
        )
    return results


class TestPearsonCorrelation:
    """PACK-02 SC1: Pearson correlation >0.85 against human-judged dataset."""

    def test_pearson_correlation_above_085(
        self,
        scored_pairs: list[dict],  # type: ignore[type-arg]
    ) -> None:
        """Pearson correlation between expected and actual scores exceeds 0.85."""
        expected = [p["expected_score"] for p in scored_pairs]
        actual = [p["actual_score"] for p in scored_pairs]
        correlation: float
        p_value: float
        correlation, p_value = pearsonr(expected, actual)  # type: ignore[assignment]
        assert correlation > 0.85, (
            f"Pearson correlation {correlation:.4f} < 0.85 threshold"
        )
        assert p_value < 0.05, f"p-value {p_value:.4f} not statistically significant"


class TestPrecisionAtThreshold:
    """PACK-02 SC1: Precision >0.90 at the 0.85 equivalence threshold."""

    def test_precision_above_090(
        self,
        scored_pairs: list[dict],  # type: ignore[type-arg]
    ) -> None:
        """Of pairs predicted equivalent (score >= 0.85), >90% truly are."""
        threshold = 0.85
        true_positives = 0
        predicted_positives = 0
        for p in scored_pairs:
            predicted_equivalent = p["actual_score"] >= threshold
            truly_equivalent = p["expected_score"] >= threshold
            if predicted_equivalent:
                predicted_positives += 1
                if truly_equivalent:
                    true_positives += 1
        precision = (
            true_positives / predicted_positives if predicted_positives > 0 else 1.0
        )
        assert precision > 0.90, (
            f"Precision {precision:.4f} < 0.90 at threshold {threshold} "
            f"({true_positives}/{predicted_positives} true positives)"
        )


class TestCategoryBehavior:
    """Verify expected behavior patterns across categories."""

    def test_identical_pairs_score_one(
        self,
        scored_pairs: list[dict],  # type: ignore[type-arg]
    ) -> None:
        """All identical pairs must score exactly 1.0."""
        identical = [p for p in scored_pairs if p["category"] == "identical"]
        for p in identical:
            assert p["actual_score"] == 1.0, (
                f"Identical pair '{p['id']}' scored {p['actual_score']:.4f} "
                f"(expected 1.0)"
            )

    def test_unrelated_pairs_score_low(
        self,
        scored_pairs: list[dict],  # type: ignore[type-arg]
    ) -> None:
        """All unrelated pairs must score below 0.30."""
        unrelated = [p for p in scored_pairs if p["category"] == "unrelated"]
        for p in unrelated:
            assert p["actual_score"] < 0.30, (
                f"Unrelated pair '{p['id']}' scored {p['actual_score']:.4f} "
                f"(expected < 0.30)"
            )

    def test_category_ordering(
        self,
        scored_pairs: list[dict],  # type: ignore[type-arg]
    ) -> None:
        """Mean scores must follow: identical > key_renaming > structural > unrelated."""
        categories = {}
        for p in scored_pairs:
            cat = p["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(p["actual_score"])

        means = {cat: sum(scores) / len(scores) for cat, scores in categories.items()}

        assert means["identical"] > means["key_renaming"], (
            f"identical ({means['identical']:.4f}) should > "
            f"key_renaming ({means['key_renaming']:.4f})"
        )
        assert means["key_renaming"] > means["structural_differences"], (
            f"key_renaming ({means['key_renaming']:.4f}) should > "
            f"structural_differences ({means['structural_differences']:.4f})"
        )
        assert means["structural_differences"] > means["unrelated"], (
            f"structural_differences ({means['structural_differences']:.4f}) should > "
            f"unrelated ({means['unrelated']:.4f})"
        )


class TestSymmetry:
    """Verify similarity is symmetric: compare(A, B) == compare(B, A)."""

    def test_symmetry_across_pairs(
        self,
        similarity_pairs: list[dict],  # type: ignore[type-arg]
    ) -> None:
        """Spot-check symmetry across 10 deterministic pairs."""
        # Use every 4th pair for a deterministic selection of ~10 pairs
        selected = similarity_pairs[::4][:10]
        for p in selected:
            score_ab = compare(p["left"], p["right"]).similarity_score
            score_ba = compare(p["right"], p["left"]).similarity_score
            assert score_ab == score_ba, (
                f"Asymmetry in pair '{p['id']}': "
                f"A->B={score_ab:.4f}, B->A={score_ba:.4f}"
            )
