"""Simple test to check if pytest works."""

def test_basic():
    """Basic test that should always pass."""
    assert 1 + 1 == 2

def test_imports():
    """Test basic Python imports."""
    import sys
    import os
    assert sys.version_info.major >= 3
    assert os.path.exists('.')
