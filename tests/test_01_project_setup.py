import pytest
import sys
from pathlib import Path
import importlib.util

def test_project_structure():
    """Test that all required directories and files exist."""
    required_dirs = [
        "packages/morag",
        "packages/morag-core",
        "packages/morag-document",
        "packages/morag-services",
        "packages/morag-embedding",
        "tests",
        "cli",
        "docs"
    ]

    for dir_path in required_dirs:
        assert Path(dir_path).exists(), f"Required directory {dir_path} does not exist"

def test_required_files():
    """Test that all required files exist."""
    required_files = [
        "packages/morag/src/morag/__init__.py",
        "packages/morag-core/src/morag_core/__init__.py",
        "packages/morag-core/src/morag_core/config.py",
        "packages/morag-core/src/morag_core/exceptions.py",
        "pyproject.toml",
        ".env.example",
        ".gitignore",
        "README.md"
    ]

    for file_path in required_files:
        assert Path(file_path).exists(), f"Required file {file_path} does not exist"

def test_config_import():
    """Test that configuration can be imported."""
    try:
        # Add packages to path for import
        packages_path = Path("packages/morag-core/src").resolve()
        if str(packages_path) not in sys.path:
            sys.path.insert(0, str(packages_path))

        from morag_core.config import settings
        assert settings is not None
        assert hasattr(settings, 'api_host')
        assert hasattr(settings, 'gemini_api_key')
    except ImportError as e:
        pytest.fail(f"Failed to import config: {e}")

def test_environment_variables():
    """Test that environment variables are properly handled."""
    # Add packages to path for import
    packages_path = Path("packages/morag-core/src").resolve()
    if str(packages_path) not in sys.path:
        sys.path.insert(0, str(packages_path))

    from morag_core.config import Settings

    # Test with minimal required env vars
    import os
    original_key = os.environ.get('MORAG_GEMINI_API_KEY')
    original_gemini_key = os.environ.get('GEMINI_API_KEY')

    try:
        # Clear both possible environment variables
        os.environ.pop('MORAG_GEMINI_API_KEY', None)
        os.environ.pop('GEMINI_API_KEY', None)

        # Set test value
        os.environ['MORAG_GEMINI_API_KEY'] = 'test_key_123'

        # Create new settings instance with explicit environment override
        test_settings = Settings(_env_file=None)  # Don't load from .env file
        assert test_settings.gemini_api_key == 'test_key_123'
    finally:
        # Restore original values
        if original_key:
            os.environ['MORAG_GEMINI_API_KEY'] = original_key
        else:
            os.environ.pop('MORAG_GEMINI_API_KEY', None)

        if original_gemini_key:
            os.environ['GEMINI_API_KEY'] = original_gemini_key
        else:
            os.environ.pop('GEMINI_API_KEY', None)

def test_dependencies_installed():
    """Test that all required dependencies are installed."""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'celery',
        'redis',
        'qdrant_client',
        'structlog'
    ]

    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            assert spec is not None, f"Package {package} is not installed"
        except ImportError:
            pytest.fail(f"Required package {package} is not available")

def test_package_installation():
    """Test that the package can be installed in development mode."""
    import subprocess
    import sys

    # This test assumes we're in the project root
    result = subprocess.run([
        sys.executable, '-c',
        'import sys; sys.path.insert(0, "packages/morag/src"); import morag; print("Package installed successfully")'
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Package installation test failed: {result.stderr}"
    assert "Package installed successfully" in result.stdout
