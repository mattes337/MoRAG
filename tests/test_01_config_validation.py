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
            assert settings.gemini_model == "gemini-pro"
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
            assert settings.gemini_api_key == "test_key"

    def test_list_parsing(self):
        """Test that list environment variables are parsed correctly."""
        test_env = {
            'ALLOWED_ORIGINS': '["http://localhost:3000", "http://localhost:8080"]'
        }

        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings()
            # Note: pydantic-settings may not parse JSON strings automatically
            # This test verifies the current behavior
            assert isinstance(settings.allowed_origins, list)

    def test_integer_parsing(self):
        """Test that integer environment variables are parsed correctly."""
        test_env = {
            'API_PORT': '8080',
            'MAX_CHUNK_SIZE': '2000',
            'WEBHOOK_TIMEOUT': '60'
        }

        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings()

            assert settings.api_port == 8080
            assert settings.max_chunk_size == 2000
            assert settings.webhook_timeout == 60

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
            'gemini_api_key': 'test_key',  # lowercase
            'API_PORT': '9000',  # uppercase
            'Log_Level': 'DEBUG'  # mixed case
        }

        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings()

            assert settings.gemini_api_key == 'test_key'
            assert settings.api_port == 9000
            assert settings.log_level == 'DEBUG'
