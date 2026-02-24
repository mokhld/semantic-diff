"""Placeholder test verifying package import."""


def test_import() -> None:
    """Verify top-level package is importable."""
    import semantic_diff

    assert semantic_diff.__version__ is not None
    assert semantic_diff.__version__ == "0.1.0"
