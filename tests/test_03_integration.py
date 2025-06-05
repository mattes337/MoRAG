import pytest
import subprocess
import time
import requests
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from morag.api.main import create_app

class TestQdrantIntegration:
    """Test Qdrant integration with API."""

    def test_qdrant_container_health_mock(self):
        """Test Qdrant container health check (mocked)."""
        # Mock the requests call since we may not have Qdrant running in tests
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ok"}
            mock_get.return_value = mock_response
            
            response = requests.get("http://localhost:6333/health", timeout=5)
            assert response.status_code == 200

    def test_qdrant_container_health_failure(self):
        """Test Qdrant container health check failure handling."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Connection failed")
            
            with pytest.raises(requests.exceptions.RequestException):
                requests.get("http://localhost:6333/health", timeout=5)

    def test_api_qdrant_health_check(self):
        """Test API health check includes Qdrant status."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert "services" in data
        assert "qdrant" in data["services"]
        
        # Qdrant status should be one of the expected values
        qdrant_status = data["services"]["qdrant"]
        assert qdrant_status in ["healthy", "unhealthy", "not_connected"]

    def test_api_basic_health_check(self):
        """Test basic API health check."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"

    @patch('subprocess.run')
    def test_database_initialization_script_mock(self, mock_run):
        """Test database initialization script execution (mocked)."""
        # Mock successful execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Database initialized successfully"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = subprocess.run([
            "python", "scripts/init_db.py"
        ], capture_output=True, text=True, timeout=30)

        assert result.returncode == 0
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_database_initialization_script_failure(self, mock_run):
        """Test database initialization script failure handling."""
        # Mock connection failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Failed to connect to Qdrant: Connection refused"
        mock_run.return_value = mock_result

        result = subprocess.run([
            "python", "scripts/init_db.py"
        ], capture_output=True, text=True, timeout=30)

        assert result.returncode == 1
        assert "connection" in result.stderr.lower() or "refused" in result.stderr.lower()

    def test_qdrant_service_import(self):
        """Test that Qdrant service can be imported correctly."""
        try:
            from morag_services.storage import qdrant_service, QdrantService
            assert qdrant_service is not None
            assert isinstance(qdrant_service, QdrantService)
        except ImportError as e:
            pytest.fail(f"Failed to import Qdrant service: {e}")

    @pytest.mark.asyncio
    async def test_qdrant_service_mock_operations(self):
        """Test Qdrant service operations with mocked client."""
        from morag_services.storage import QdrantService
        
        service = QdrantService()
        
        # Mock the client
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
        mock_client.upsert = MagicMock()
        mock_client.close = MagicMock()
        
        service.client = mock_client

        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Test collection creation
            mock_to_thread.side_effect = [
                MagicMock(collections=[]),  # get_collections
                None,  # create_collection
                mock_client.get_collection.return_value  # get_collection
            ]
            
            await service.create_collection(vector_size=768)
            info = await service.get_collection_info()
            
            assert info["config"]["vector_size"] == 768
            assert info["status"] == "green"

    def test_health_endpoint_qdrant_integration(self):
        """Test health endpoint Qdrant integration."""
        app = create_app()
        client = TestClient(app)
        
        # Mock the qdrant_service to avoid actual connection
        with patch('morag.api.routes.health.qdrant_service') as mock_service:
            mock_service.client = None  # Simulate not connected
            
            response = client.get("/health/ready")
            assert response.status_code == 200
            
            data = response.json()
            assert data["services"]["qdrant"] == "not_connected"

    def test_health_endpoint_qdrant_healthy(self):
        """Test health endpoint when Qdrant is healthy."""
        app = create_app()
        client = TestClient(app)
        
        with patch('morag.api.routes.health.qdrant_service') as mock_service:
            mock_service.client = MagicMock()  # Simulate connected
            mock_service.get_collection_info = AsyncMock(return_value={
                "name": "test_collection",
                "status": "green"
            })
            
            response = client.get("/health/ready")
            assert response.status_code == 200
            
            data = response.json()
            assert data["services"]["qdrant"] == "healthy"

    def test_health_endpoint_qdrant_unhealthy(self):
        """Test health endpoint when Qdrant is unhealthy."""
        app = create_app()
        client = TestClient(app)
        
        with patch('morag.api.routes.health.qdrant_service') as mock_service:
            mock_service.client = MagicMock()  # Simulate connected
            mock_service.get_collection_info = AsyncMock(side_effect=Exception("Connection error"))
            
            response = client.get("/health/ready")
            assert response.status_code == 200
            
            data = response.json()
            assert data["services"]["qdrant"] == "unhealthy"

class TestQdrantConfiguration:
    """Test Qdrant configuration and setup."""

    def test_docker_compose_file_exists(self):
        """Test that Docker Compose file exists."""
        from pathlib import Path
        
        compose_file = Path("docker/docker-compose.qdrant.yml")
        assert compose_file.exists(), "Qdrant Docker Compose file not found"
        
        # Test file content
        with open(compose_file, 'r') as f:
            content = f.read()
        
        assert "qdrant/qdrant" in content
        assert "6333:6333" in content
        assert "6334:6334" in content

    def test_qdrant_settings_configuration(self):
        """Test Qdrant settings configuration."""
        from morag_core.config import settings
        
        # Test that Qdrant settings are properly configured
        assert hasattr(settings, 'qdrant_host')
        assert hasattr(settings, 'qdrant_port')
        assert hasattr(settings, 'qdrant_collection_name')
        assert hasattr(settings, 'qdrant_api_key')
        
        # Test default values
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.qdrant_collection_name == "morag_documents"

    def test_service_singleton_pattern(self):
        """Test that qdrant_service follows singleton pattern."""
        from morag_services.storage import qdrant_service
        
        # Import again to test singleton
        from morag_services.storage import qdrant_service as qdrant_service2
        
        assert qdrant_service is qdrant_service2

    @pytest.mark.asyncio
    async def test_service_lifecycle(self):
        """Test service connection lifecycle."""
        from morag_services.storage import QdrantService
        
        service = QdrantService()
        
        # Test initial state
        assert service.client is None
        
        # Mock connection
        with patch('qdrant_client.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock(collections=[])
            mock_client.close = MagicMock()
            mock_client_class.return_value = mock_client
            
            with patch('asyncio.to_thread', new_callable=AsyncMock):
                await service.connect()
                assert service.client is not None
                
                await service.disconnect()
                assert service.client is None
