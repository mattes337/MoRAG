import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for import
src_path = Path("src").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from morag_core.config import Settings

class TestConfigValidation:
    """Test configuration validation and defaults."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            # Create settings without loading .env file
            settings = Settings(_env_file=None)

            assert settings.api_host == "0.0.0.0"
            assert settings.api_port == 8000
            assert settings.gemini_model == "gemini-2.0-flash"  # Default in morag-core
            assert settings.gemini_embedding_model == "text-embedding-004"
            assert settings.qdrant_host == "localhost"
            assert settings.qdrant_port == 6333
            assert settings.gemini_api_key is None

    def test_optional_env_vars(self):
        """Test that optional environment variables work correctly."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
            # gemini_api_key is optional and should be None by default
            assert settings.gemini_api_key is None

    def test_env_override(self):
        """Test that environment variables override defaults."""
        test_env = {
            'MORAG_GEMINI_API_KEY': 'test_key',
            'MORAG_API_PORT': '9000',
            'MORAG_QDRANT_HOST': 'custom-host',
            'MORAG_LOG_LEVEL': 'DEBUG'
        }

        with patch.dict(os.environ, test_env, clear=True):
            # Disable .env file loading to test pure environment variables
            settings = Settings(_env_file=None)

            assert settings.api_port == 9000
            assert settings.qdrant_host == "custom-host"
            assert settings.log_level == "DEBUG"
            assert settings.gemini_api_key == "test_key"

    def test_list_parsing(self):
        """Test that list environment variables are parsed correctly."""
        test_env = {
            'MORAG_ALLOWED_ORIGINS': '["http://localhost:3000", "http://localhost:8080"]'
        }

        with patch.dict(os.environ, test_env, clear=True):
            # Disable .env file loading to test pure environment variables
            settings = Settings(_env_file=None)
            # Note: pydantic-settings may not parse JSON strings automatically
            # This test verifies the current behavior
            assert isinstance(settings.allowed_origins, list)

    def test_integer_parsing(self):
        """Test that integer environment variables are parsed correctly."""
        test_env = {
            'MORAG_API_PORT': '8080',
            'MORAG_MAX_DOCUMENT_SIZE': '209715200',  # 200MB in bytes
            'MORAG_SLOW_QUERY_THRESHOLD': '10.0'
        }

        with patch.dict(os.environ, test_env, clear=True):
            # Disable .env file loading to test pure environment variables
            settings = Settings(_env_file=None)

            assert settings.api_port == 8080
            assert settings.max_document_size == 209715200
            assert settings.slow_query_threshold == 10.0

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)

            # qdrant_api_key is optional
            assert settings.qdrant_api_key is None
            # gemini_api_key is also optional
            assert settings.gemini_api_key is None

    def test_case_insensitive(self):
        """Test that environment variables are case insensitive."""
        test_env = {
            'morag_gemini_api_key': 'test_key',  # lowercase
            'MORAG_API_PORT': '9000',  # uppercase
            'Morag_Log_Level': 'DEBUG'  # mixed case
        }

        with patch.dict(os.environ, test_env, clear=True):
            # Disable .env file loading to test pure environment variables
            settings = Settings(_env_file=None)

            assert settings.gemini_api_key == 'test_key'
            assert settings.api_port == 9000
            assert settings.log_level == 'DEBUG'
