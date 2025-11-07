"""Pytest configuration for morag-image package tests."""

import os
import pytest
from pathlib import Path

# Define test resources directory
@pytest.fixture
def resources_dir():
    """Path to test resources directory."""
    return Path(__file__).parent / "resources"

# Skip tests requiring API key if not available
def pytest_configure(config):
    """Configure pytest."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring API key"
    )
    config.addinivalue_line(
        "markers", "requires_tesseract: mark test as requiring Tesseract OCR"
    )
    config.addinivalue_line(
        "markers", "requires_easyocr: mark test as requiring EasyOCR"
    )

# Skip tests requiring API key if not available
def pytest_runtest_setup(item):
    """Set up tests."""
    # Skip tests requiring API key
    if "requires_api_key" in item.keywords and not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("Test requires GOOGLE_API_KEY environment variable")

    # Skip tests requiring Tesseract
    if "requires_tesseract" in item.keywords:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except (ImportError, Exception):
            pytest.skip("Test requires Tesseract OCR to be installed")

    # Skip tests requiring EasyOCR
    if "requires_easyocr" in item.keywords:
        try:
            import easyocr
        except ImportError:
            pytest.skip("Test requires EasyOCR to be installed")
