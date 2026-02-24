"""Placeholder test verifying package import."""


def test_import() -> None:
    """Verify top-level package is importable."""
    import json_semantic_diff

    assert json_semantic_diff.__version__ is not None
    assert json_semantic_diff.__version__ == "0.1.0"
