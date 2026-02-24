"""Integration tests for the json-semantic-diff pytest plugin.

These tests verify that the assert_json_equivalent fixture is auto-discovered
via the pytest11 entry point and behaves correctly.

NOTE: These tests require json-semantic-diff to be installed (even in editable mode
via ``pip install -e .``). The pytest11 entry point is only registered at
install time -- running from a raw source checkout without installing will not
discover the fixture.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from json_semantic_diff import STEDConfig


def test_fixture_passes_equivalent_docs(assert_json_equivalent: Any) -> None:
    """Equivalent documents (camelCase vs snake_case) should pass at default threshold."""
    # user_name and userName are semantically equivalent -- STED maps them with score > 0.85
    assert_json_equivalent({"user_name": "John"}, {"userName": "John"})


def test_fixture_fails_structural_break(assert_json_equivalent: Any) -> None:
    """Documents with different structure should raise AssertionError."""
    with pytest.raises(AssertionError, match=r"similarity="):
        assert_json_equivalent({"user_name": "John"}, {"address": "123 Main St"})


def test_fixture_custom_threshold(assert_json_equivalent: Any) -> None:
    """Custom threshold parameter should be respected."""
    # threshold=0.0 means any score passes
    assert_json_equivalent({"a": 1}, {"b": 1}, threshold=0.0)

    # threshold=1.0 requires perfect identity -- non-identical keys should fail
    with pytest.raises(AssertionError, match=r"similarity="):
        assert_json_equivalent({"a": 1}, {"b": 1}, threshold=1.0)


def test_fixture_custom_config(assert_json_equivalent: Any) -> None:
    """Custom STEDConfig parameter should be forwarded to compare()."""
    # With type_coercion=True, "123" and 123 should be treated as equal
    assert_json_equivalent(
        {"x": "123"},
        {"x": 123},
        config=STEDConfig(type_coercion=True),
    )


def test_fixture_error_message_contents(assert_json_equivalent: Any) -> None:
    """AssertionError message should contain all required diagnostic fields."""
    with pytest.raises(AssertionError) as exc_info:
        assert_json_equivalent({"user_name": "John"}, {"address": "123 Main St"})

    error_message = str(exc_info.value)
    assert "similarity=" in error_message
    assert "threshold=" in error_message
    assert "unmatched_left" in error_message
    assert "unmatched_right" in error_message


def test_fixture_returns_callable(assert_json_equivalent: Any) -> None:
    """The fixture should return a callable, not None or a direct assertion result."""
    assert callable(assert_json_equivalent), (
        "assert_json_equivalent fixture must return a callable, not a direct value"
    )
    assert assert_json_equivalent is not None


def test_plugin_discovery() -> None:
    """Verify assert_json_equivalent appears in pytest --fixtures output."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--fixtures", "-q"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert "assert_json_equivalent" in result.stdout, (
        f"assert_json_equivalent not found in pytest --fixtures output.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
