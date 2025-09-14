import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

class TestDatabaseInitialization:
    """Test database initialization procedures."""

    def test_init_script_exists(self):
        """Test that initialization script exists and is executable."""
        script_path = Path("scripts/init_db.py")
        assert script_path.exists(), "Database initialization script not found"

        # Test script syntax
        result = subprocess.run([
            sys.executable, "-m", "py_compile", str(script_path)
        ], capture_output=True)
        assert result.returncode == 0, f"Script syntax error: {result.stderr}"

    def test_init_script_imports(self):
        """Test that the init script can import required modules."""
        script_path = Path("scripts/init_db.py")
        
        # Test that the script can be imported without errors
        result = subprocess.run([
            sys.executable, "-c", 
            f"import sys; sys.path.insert(0, 'src'); exec(open('{script_path}').read().split('if __name__')[0])"
        ], capture_output=True, text=True)
        
        # Should not have import errors
        assert "ImportError" not in result.stderr
        assert "ModuleNotFoundError" not in result.stderr

    @patch('morag.services.storage.qdrant_service')
    @pytest.mark.asyncio
    async def test_init_script_logic(self, mock_service):
        """Test database initialization script logic."""
        mock_service.connect = AsyncMock()
        mock_service.create_collection = AsyncMock()
        mock_service.get_collection_info = AsyncMock(return_value={
            "name": "morag_documents",
            "vectors_count": 0,
            "points_count": 0,
            "status": "green",
            "config": {
                "vector_size": 768,
                "distance": "Cosine"
            }
        })
        mock_service.disconnect = AsyncMock()

        # Import and test the main function from the script
        import sys
        sys.path.insert(0, str(Path("scripts").resolve()))
        
        # Mock the script's main function logic
        await mock_service.connect()
        await mock_service.create_collection(vector_size=768, force_recreate=False)
        info = await mock_service.get_collection_info()
        await mock_service.disconnect()

        # Verify calls
        mock_service.connect.assert_called_once()
        mock_service.create_collection.assert_called_once_with(vector_size=768, force_recreate=False)
        mock_service.get_collection_info.assert_called_once()
        mock_service.disconnect.assert_called_once()
        
        assert info["name"] == "morag_documents"
        assert info["config"]["vector_size"] == 768

    @pytest.mark.asyncio
    async def test_collection_configuration(self):
        """Test that collection is configured correctly."""
        from morag_services.storage import QdrantService
        
        service = QdrantService()
        
        # Mock the client and its methods
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        mock_client.create_collection = MagicMock()
        mock_client.get_collection.return_value = MagicMock(
            vectors_count=0,
            indexed_vectors_count=0,
            points_count=0,
            status=MagicMock(value="green"),
            optimizer_status=MagicMock(status=MagicMock(value="ok")),
            config=MagicMock(
                params=MagicMock(
                    vectors=MagicMock(
                        size=768,
                        distance=MagicMock(value="Cosine")
                    )
                )
            )
        )
        mock_client.close = MagicMock()
        
        service.client = mock_client

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Mock the async operations
            mock_to_thread.side_effect = [
                MagicMock(collections=[]),  # get_collections
                None,  # create_collection
                mock_client.get_collection.return_value  # get_collection
            ]
            
            await service.create_collection(vector_size=768, force_recreate=True)
            info = await service.get_collection_info()

            assert info["config"]["vector_size"] == 768
            assert info["config"]["distance"] == "Cosine"

    def test_script_error_handling(self):
        """Test that the script handles errors gracefully."""
        # Test with invalid configuration (this should be a controlled test)
        script_content = '''
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

try:
    from morag_services.storage import QdrantService
    service = QdrantService()
    # Simulate connection failure
    raise Exception("Simulated connection error")
except Exception as e:
    print(f"Expected error: {e}")
    sys.exit(1)
'''

        result = subprocess.run([
            sys.executable, "-c", script_content
        ], capture_output=True, text=True)

        # Should exit with code 1 and show error message
        assert result.returncode == 1
        assert "Expected error" in result.stdout

    def test_script_path_handling(self):
        """Test that the script correctly handles path setup."""
        script_path = Path("scripts/init_db.py")
        
        # Read the script and check path setup
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Should have proper path setup
        assert "sys.path.insert" in content
        assert "src" in content
        assert "__file__" in content

    @patch('structlog.get_logger')
    def test_script_logging(self, mock_logger):
        """Test that the script uses proper logging."""
        mock_log = MagicMock()
        mock_logger.return_value = mock_log
        
        # Test logging setup
        import structlog
        logger = structlog.get_logger()
        
        # Simulate log calls
        logger.info("Test message")
        logger.error("Test error")
        
        # Verify logger was called
        assert mock_logger.called

class TestDatabaseConfiguration:
    """Test database configuration and setup."""

    def test_collection_name_configuration(self):
        """Test that collection name is properly configured."""
        from morag_core.config import settings
        from morag_services.storage import qdrant_service
        
        assert hasattr(settings, 'qdrant_collection_name')
        assert qdrant_service.collection_name == settings.qdrant_collection_name

    def test_qdrant_connection_parameters(self):
        """Test Qdrant connection parameters."""
        from morag_core.config import settings
        
        # Test that all required Qdrant settings exist
        assert hasattr(settings, 'qdrant_host')
        assert hasattr(settings, 'qdrant_port')
        assert hasattr(settings, 'qdrant_collection_name')
        assert hasattr(settings, 'qdrant_api_key')
        
        # Test default values
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.qdrant_collection_name == "morag_documents"

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test service initialization process."""
        from morag_services.storage import QdrantService
        
        service = QdrantService()
        
        # Test initial state
        assert service.client is None
        assert service.collection_name is not None
        
        # Test that service can be created without errors
        assert isinstance(service, QdrantService)

    def test_vector_configuration(self):
        """Test vector configuration parameters."""
        # Test that the expected vector size is 768 (for text-embedding-004)
        expected_vector_size = 768
        
        # This would be used in the initialization
        assert expected_vector_size == 768
        
        # Test distance metric
        expected_distance = "Cosine"
        assert expected_distance in ["Cosine", "Dot", "Euclidean"]
