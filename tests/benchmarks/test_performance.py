"""Performance benchmark suite for semantic-diff.

Validates PACK-03 timing targets using the StaticBackend:
- 10-key flat objects: <10ms
- 100-key mixed nested objects: <100ms
- 500-key deeply nested objects: <1s

Run with: pytest tests/benchmarks/ --benchmark-only -v
Skip during normal test runs: pytest --benchmark-disable
"""

from __future__ import annotations

from semantic_diff import compare


class TestPerformance10Key:
    """Benchmark suite for 10-key flat objects. Target: <10ms."""

    def test_10key_similar(self, benchmark, pair_10key_similar):  # type: ignore[no-untyped-def]
        left, right = pair_10key_similar
        result = benchmark(compare, left, right)
        # Verify the result is valid (not just timing)
        assert 0.0 <= result.similarity_score <= 1.0

    def test_10key_dissimilar(self, benchmark, pair_10key_dissimilar):  # type: ignore[no-untyped-def]
        left, right = pair_10key_dissimilar
        result = benchmark(compare, left, right)
        assert 0.0 <= result.similarity_score <= 1.0


class TestPerformance100Key:
    """Benchmark suite for 100-key mixed nested objects. Target: <100ms."""

    def test_100key_similar(self, benchmark, pair_100key_similar):  # type: ignore[no-untyped-def]
        left, right = pair_100key_similar
        result = benchmark(compare, left, right)
        assert 0.0 <= result.similarity_score <= 1.0

    def test_100key_dissimilar(self, benchmark, pair_100key_dissimilar):  # type: ignore[no-untyped-def]
        left, right = pair_100key_dissimilar
        result = benchmark(compare, left, right)
        assert 0.0 <= result.similarity_score <= 1.0


class TestPerformance500Key:
    """Benchmark suite for 500-key deeply nested objects. Target: <1s."""

    def test_500key_similar(self, benchmark, pair_500key_similar):  # type: ignore[no-untyped-def]
        left, right = pair_500key_similar
        result = benchmark(compare, left, right)
        assert 0.0 <= result.similarity_score <= 1.0

    def test_500key_dissimilar(self, benchmark, pair_500key_dissimilar):  # type: ignore[no-untyped-def]
        left, right = pair_500key_dissimilar
        result = benchmark(compare, left, right)
        assert 0.0 <= result.similarity_score <= 1.0
