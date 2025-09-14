"""Tests for database server array functionality."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from morag.database_factory import (
    DatabaseConnectionFactory, parse_database_servers,
    get_neo4j_storages, get_qdrant_storages
)
from morag_graph import DatabaseServerConfig, DatabaseType


class TestDatabaseConnectionFactory:
    """Test database connection factory functionality."""
    
    def test_parse_database_servers_valid(self):
        """Test parsing valid database server configurations."""
        server_data = [
            {
                "type": "neo4j",
                "hostname": "localhost",
                "port": 7687,
                "username": "neo4j",
                "password": "password",
                "database_name": "test_db"
            },
            {
                "type": "qdrant",
                "hostname": "localhost",
                "port": 6333,
                "password": "api_key",
                "database_name": "test_collection"
            }
        ]
        
        configs = parse_database_servers(server_data)
        
        assert len(configs) == 2
        assert configs[0].type == DatabaseType.NEO4J
        assert configs[0].hostname == "localhost"
        assert configs[0].port == 7687
        assert configs[1].type == DatabaseType.QDRANT
        assert configs[1].hostname == "localhost"
        assert configs[1].port == 6333
    
    def test_parse_database_servers_invalid(self):
        """Test parsing invalid database server configurations."""
        server_data = [
            {
                "type": "invalid_type",
                "hostname": "localhost"
            },
            {
                "type": "neo4j",
                "hostname": "localhost",
                "port": 7687
            }
        ]
        
        configs = parse_database_servers(server_data)
        
        # Should skip invalid config and return only valid one
        assert len(configs) == 1
        assert configs[0].type == DatabaseType.NEO4J
    
    def test_parse_database_servers_empty(self):
        """Test parsing empty database server configurations."""
        configs = parse_database_servers(None)
        assert configs == []
        
        configs = parse_database_servers([])
        assert configs == []
    
    @patch('morag.database_factory.Neo4jStorage')
    async def test_create_neo4j_storage(self, mock_neo4j_storage):
        """Test creating Neo4j storage from configuration."""
        config = DatabaseServerConfig(
            type=DatabaseType.NEO4J,
            hostname="neo4j://localhost:7687",
            username="neo4j",
            password="password",
            database_name="test_db"
        )

        # Mock the connect method
        mock_storage_instance = mock_neo4j_storage.return_value
        mock_storage_instance.connect = AsyncMock()

        factory = DatabaseConnectionFactory()
        storage = await factory.create_neo4j_storage(config)

        mock_neo4j_storage.assert_called_once()
        mock_storage_instance.connect.assert_called_once()
        assert storage is not None
    
    @patch('morag.database_factory.QdrantStorage')
    def test_create_qdrant_storage(self, mock_qdrant_storage):
        """Test creating Qdrant storage from configuration."""
        config = DatabaseServerConfig(
            type=DatabaseType.QDRANT,
            hostname="localhost",
            port=6333,
            password="api_key",
            database_name="test_collection"
        )
        
        factory = DatabaseConnectionFactory()
        storage = factory.create_qdrant_storage(config)
        
        mock_qdrant_storage.assert_called_once()
        assert storage is not None
    
    def test_create_storage_invalid_type(self):
        """Test creating storage with invalid type."""
        config = DatabaseServerConfig(
            type=DatabaseType.NEO4J,
            hostname="localhost"
        )
        
        factory = DatabaseConnectionFactory()
        
        # Test wrong type for Qdrant creation
        with pytest.raises(ValueError):
            factory.create_qdrant_storage(config)


class TestDatabaseServerIntegration:
    """Test database server integration with API endpoints."""
    
    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app for testing."""
        from morag.server import create_app
        app = create_app()
        return TestClient(app)
    
    def test_search_with_database_servers(self, mock_app):
        """Test search endpoint with custom database servers."""
        request_data = {
            "query": "test query",
            "limit": 5,
            "database_servers": [
                {
                    "type": "qdrant",
                    "hostname": "localhost",
                    "port": 6333,
                    "database_name": "test_collection"
                }
            ]
        }
        
        # This will likely fail due to missing dependencies, but tests the structure
        response = mock_app.post("/search", json=request_data)
        # We expect either success or a specific error, not a validation error
        assert response.status_code in [200, 500, 503]
    
    def test_enhanced_query_with_database_servers(self, mock_app):
        """Test enhanced query endpoint with custom database servers."""
        request_data = {
            "query": "test query",
            "max_results": 5,
            "database_servers": [
                {
                    "type": "neo4j",
                    "hostname": "neo4j://localhost:7687",
                    "username": "neo4j",
                    "password": "password",
                    "database_name": "test_db"
                }
            ]
        }
        
        response = mock_app.post("/api/v2/query", json=request_data)
        # We expect either success or a specific error, not a validation error
        assert response.status_code in [200, 500, 503]
    
    def test_graph_analytics_with_database_servers(self, mock_app):
        """Test graph analytics endpoint with custom database servers."""
        request_data = {
            "metric_type": "overview",
            "database_servers": [
                {
                    "type": "neo4j",
                    "hostname": "neo4j://localhost:7687",
                    "username": "neo4j",
                    "password": "password",
                    "database_name": "test_db"
                }
            ]
        }
        
        response = mock_app.post("/api/v2/graph/analytics", json=request_data)
        # We expect either success or a specific error, not a validation error
        assert response.status_code in [200, 500, 503]


class TestDatabaseServerConfig:
    """Test database server configuration models."""
    
    def test_database_server_config_creation(self):
        """Test creating database server configuration."""
        config = DatabaseServerConfig(
            type=DatabaseType.NEO4J,
            hostname="localhost",
            port=7687,
            username="neo4j",
            password="password",
            database_name="test_db"
        )
        
        assert config.type == DatabaseType.NEO4J
        assert config.hostname == "localhost"
        assert config.port == 7687
        assert config.username == "neo4j"
        assert config.password == "password"
        assert config.database_name == "test_db"
    
    def test_database_server_config_connection_key(self):
        """Test database server configuration connection key generation."""
        config = DatabaseServerConfig(
            type=DatabaseType.NEO4J,
            hostname="localhost",
            port=7687,
            username="neo4j",
            database_name="test_db"
        )
        
        key = config.get_connection_key()
        expected = "neo4j:localhost:7687:neo4j:test_db"
        assert key == expected
    
    def test_database_server_config_is_default(self):
        """Test checking if configuration is default."""
        # Default config
        config = DatabaseServerConfig(type=DatabaseType.NEO4J)
        assert config.is_default_config()
        
        # Non-default config
        config = DatabaseServerConfig(
            type=DatabaseType.NEO4J,
            hostname="localhost"
        )
        assert not config.is_default_config()


if __name__ == "__main__":
    pytest.main([__file__])
