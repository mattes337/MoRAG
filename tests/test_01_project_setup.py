import pytest
import sys
from pathlib import Path
import importlib.util

def test_project_structure():
    """Test that all required directories and files exist."""
    required_dirs = [
        "src/morag",
        "src/morag/api",
        "src/morag/core",
        "src/morag/processors",
        "src/morag/services",
        "src/morag/utils",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        "docker",
        "scripts",
        "docs"
    ]

    for dir_path in required_dirs:
        assert Path(dir_path).exists(), f"Required directory {dir_path} does not exist"

def test_required_files():
    """Test that all required files exist."""
    required_files = [
        "src/morag/__init__.py",
        "src/morag/core/__init__.py",
        "src/morag/core/config.py",
        "src/morag/core/exceptions.py",
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
        # Add src to path for import
        src_path = Path("src").resolve()
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from morag_core.config import settings
        assert settings is not None
        assert hasattr(settings, 'api_host')
        assert hasattr(settings, 'gemini_api_key')
    except ImportError as e:
        pytest.fail(f"Failed to import config: {e}")

def test_environment_variables():
    """Test that environment variables are properly handled."""
    # Add src to path for import
    src_path = Path("src").resolve()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from morag_core.config import Settings
    
    # Test with minimal required env vars
    import os
    original_key = os.environ.get('GEMINI_API_KEY')
    
    try:
        os.environ['GEMINI_API_KEY'] = 'test_key_123'
        test_settings = Settings()
        assert test_settings.gemini_api_key == 'test_key_123'
    finally:
        if original_key:
            os.environ['GEMINI_API_KEY'] = original_key
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
        'import sys; sys.path.insert(0, "src"); import morag; print("Package installed successfully")'
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Package installation test failed: {result.stderr}"
    assert "Package installed successfully" in result.stdout
