"""pytest plugin for json-semantic-diff.

Auto-discovered by pytest via the pytest11 entry point declared in pyproject.toml.
When the package is installed (even in editable mode), pytest discovers this plugin
automatically -- no conftest.py changes are needed.

Source: https://docs.pytest.org/en/stable/how-to/writing_plugins.html
"""

from __future__ import annotations

from typing import Any

import pytest

from json_semantic_diff import STEDConfig, compare


@pytest.fixture(scope="session")
def assert_json_equivalent() -> Any:
    """Fixture that returns a callable JSON equivalence asserter.

    The fixture is session-scoped because the returned callable is stateless
    (delegates to compare() which creates a fresh STEDComparator per call).

    Usage in tests::

        def test_rename(assert_json_equivalent):
            assert_json_equivalent({"user_name": "John"}, {"userName": "John"})

        def test_structural_break(assert_json_equivalent):
            with pytest.raises(AssertionError, match=r"similarity="):
                assert_json_equivalent({"name": "x"}, {"address": "y"})

    Args:
        No arguments -- the fixture is injected by pytest.

    Returns:
        A callable ``_assert(actual, expected, threshold=0.85, config=None) -> None``
        that raises ``AssertionError`` when the similarity score is below threshold.
    """

    def _assert(
        actual: Any,
        expected: Any,
        threshold: float = 0.85,
        config: STEDConfig | None = None,
    ) -> None:
        """Assert that two JSON documents are semantically equivalent.

        Args:
            actual:    The actual JSON value produced by the code under test.
            expected:  The expected/reference JSON value.
            threshold: Minimum similarity score to consider equivalent.
                       Defaults to 0.85 (passes benign naming differences,
                       rejects structural breaks).
            config:    Optional STEDConfig for custom algorithm parameters.

        Raises:
            AssertionError: When similarity_score < threshold, with a message
                including the score, threshold, actual/expected values,
                key_mappings, unmatched_left, and unmatched_right.
        """
        result = compare(actual, expected, config=config)
        if result.similarity_score < threshold:
            raise AssertionError(
                f"JSON documents not equivalent: "
                f"similarity={result.similarity_score:.4f} < threshold={threshold}\n"
                f"  actual:   {actual}\n"
                f"  expected: {expected}\n"
                f"  key_mappings: {result.key_mappings}\n"
                f"  unmatched_left:  {result.unmatched_left}\n"
                f"  unmatched_right: {result.unmatched_right}"
            )

    return _assert
