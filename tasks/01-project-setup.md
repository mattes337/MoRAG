# Task 01: Project Setup and Configuration

## Overview
Set up the initial project structure, environment configuration, and dependency management for the MoRAG ingestion pipeline.

## Prerequisites
- Python 3.9+
- Git
- Docker (optional, for containerization)

## Dependencies
None (this is the foundation task)

## Implementation Steps

### 1. Project Structure
Create the following directory structure:
```
morag/
├── src/
│   ├── morag/
│   │   ├── __init__.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes/
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   └── exceptions.py
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── document.py
│   │   │   ├── audio.py
│   │   │   ├── video.py
│   │   │   ├── image.py
│   │   │   └── web.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── embedding.py
│   │   │   ├── storage.py
│   │   │   └── chunking.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docker/
├── scripts/
├── docs/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── Dockerfile
```

### 2. Environment Configuration
Create `.env.example`:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-pro
GEMINI_EMBEDDING_MODEL=text-embedding-004

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=morag_documents
QDRANT_API_KEY=

# Redis Configuration (for Celery)
REDIS_URL=redis://localhost:6379/0

# Task Queue
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# File Storage
UPLOAD_DIR=./uploads
TEMP_DIR=./temp
MAX_FILE_SIZE=100MB

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
API_KEY_HEADER=X-API-Key
ALLOWED_ORIGINS=["http://localhost:3000"]

# Processing Limits
MAX_CHUNK_SIZE=1000
MAX_CONCURRENT_TASKS=10
WEBHOOK_TIMEOUT=30
```

### 3. Dependency Management
Create `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "morag"
version = "0.1.0"
description = "Multimodal RAG Ingestion Pipeline"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "celery>=5.3.0",
    "redis>=5.0.0",
    "qdrant-client>=1.7.0",
    "google-generativeai>=0.3.0",
    "unstructured[all-docs]>=0.11.0",
    "spacy>=3.7.0",
    "openai-whisper>=20231117",
    "yt-dlp>=2023.12.30",
    "beautifulsoup4>=4.12.0",
    "markdownify>=0.11.6",
    "python-multipart>=0.0.6",
    "aiofiles>=23.2.1",
    "httpx>=0.25.0",
    "python-dotenv>=1.0.0",
    "structlog>=23.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.7.0",
    "pre-commit>=3.5.0",
]

docling = [
    "docling>=1.0.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/morag"
Repository = "https://github.com/yourusername/morag"
Documentation = "https://github.com/yourusername/morag/docs"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 88
target-version = ['py39']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src/morag --cov-report=html --cov-report=term-missing"
```

### 4. Core Configuration Module
Create `src/morag/core/config.py`:
```python
from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Gemini API
    gemini_api_key: str
    gemini_model: str = "gemini-pro"
    gemini_embedding_model: str = "text-embedding-004"
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "morag_documents"
    qdrant_api_key: Optional[str] = None
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Task Queue
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # File Storage
    upload_dir: str = "./uploads"
    temp_dir: str = "./temp"
    max_file_size: str = "100MB"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Security
    api_key_header: str = "X-API-Key"
    allowed_origins: List[str] = ["http://localhost:3000"]
    
    # Processing Limits
    max_chunk_size: int = 1000
    max_concurrent_tasks: int = 10
    webhook_timeout: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### 5. Git Configuration
Create `.gitignore`:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env
.env.local
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Uploads and temp files
uploads/
temp/
*.tmp

# Testing
.coverage
htmlcov/
.pytest_cache/
.tox/

# Docker
.dockerignore

# Data
*.db
*.sqlite
data/
```

## Mandatory Testing Requirements

### 1. Setup Validation Tests
Create `tests/test_01_project_setup.py`:
```python
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
        from morag.core.config import settings
        assert settings is not None
        assert hasattr(settings, 'api_host')
        assert hasattr(settings, 'gemini_api_key')
    except ImportError as e:
        pytest.fail(f"Failed to import config: {e}")

def test_environment_variables():
    """Test that environment variables are properly handled."""
    from morag.core.config import Settings

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
        'google.generativeai',
        'unstructured',
        'spacy',
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
        'import morag; print("Package installed successfully")'
    ], capture_output=True, text=True)

    assert result.returncode == 0, f"Package installation test failed: {result.stderr}"
    assert "Package installed successfully" in result.stdout
```

### 2. Configuration Tests
Create `tests/test_01_config_validation.py`:
```python
import pytest
import os
from unittest.mock import patch
from morag.core.config import Settings

class TestConfigValidation:
    """Test configuration validation and defaults."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}, clear=True):
            settings = Settings()

            assert settings.api_host == "0.0.0.0"
            assert settings.api_port == 8000
            assert settings.gemini_model == "gemini-pro"
            assert settings.gemini_embedding_model == "text-embedding-004"
            assert settings.qdrant_host == "localhost"
            assert settings.qdrant_port == 6333

    def test_required_env_vars(self):
        """Test that required environment variables are validated."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception):  # Should fail without GEMINI_API_KEY
                Settings()

    def test_env_override(self):
        """Test that environment variables override defaults."""
        test_env = {
            'GEMINI_API_KEY': 'test_key',
            'API_PORT': '9000',
            'QDRANT_HOST': 'custom-host',
            'LOG_LEVEL': 'DEBUG'
        }

        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings()

            assert settings.api_port == 9000
            assert settings.qdrant_host == "custom-host"
            assert settings.log_level == "DEBUG"
```

### 3. Test Execution Instructions
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run setup validation tests
pytest tests/test_01_project_setup.py -v

# Run configuration tests
pytest tests/test_01_config_validation.py -v

# Run all Task 01 tests with coverage
pytest tests/test_01_*.py -v --cov=src/morag/core --cov-report=html
```

## Success Criteria (MANDATORY - ALL MUST PASS)
- [ ] All project structure tests pass
- [ ] All required files exist and are accessible
- [ ] Configuration imports successfully
- [ ] Environment variable handling works correctly
- [ ] All dependencies are properly installed
- [ ] Package can be imported without errors
- [ ] Configuration validation works as expected
- [ ] Test coverage is >90% for core configuration module

## Advancement Blocker
**⚠️ CRITICAL: Cannot proceed to Task 02 until ALL tests pass and coverage requirements are met.**

## Next Steps
- Task 02: API Framework Setup
- Task 14: Gemini Integration (can be done in parallel)
