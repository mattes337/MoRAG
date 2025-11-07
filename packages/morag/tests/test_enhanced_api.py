"""Tests for enhanced API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json

from morag.models.enhanced_query import (
    EnhancedQueryRequest, QueryType, ExpansionStrategy, FusionStrategy
)
from morag.server import create_app


@pytest.fixture
def test_client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_hybrid_system():
    """Mock hybrid retrieval system."""
    mock = AsyncMock()
    mock.retrieve.return_value = [
        {
            'id': 'result_1',
            'content': 'Test content 1',
            'score': 0.9,
            'document_id': 'doc_1',
            'source_type': 'vector',
            'metadata': {'test': 'data'}
        },
        {
            'id': 'result_2',
            'content': 'Test content 2',
            'score': 0.8,
            'document_id': 'doc_2',
            'source_type': 'graph',
            'metadata': {'test': 'data2'}
        }
    ]
    return mock


@pytest.fixture
def mock_graph_engine():
    """Mock graph engine."""
    mock = MagicMock()

    # Mock entity
    mock_entity = MagicMock()
    mock_entity.id = "entity_1"
    mock_entity.name = "Test Entity"
    mock_entity.type = "CONCEPT"
    mock_entity.dict.return_value = {
        'id': 'entity_1',
        'name': 'Test Entity',
        'type': 'CONCEPT',
        'properties': {}
    }

    mock.get_entity.return_value = mock_entity
    mock.find_entities_by_name.return_value = [mock_entity]
    mock.get_entity_relations.return_value = []
    mock.find_shortest_paths.return_value = []
    mock.explore_from_entity.return_value = []

    # Mock statistics
    mock_stats = MagicMock()
    mock_stats.entity_count = 100
    mock_stats.relation_count = 200
    mock_stats.avg_degree = 2.0
    mock_stats.connected_components = 5
    mock_stats.density = 0.02
    mock.get_graph_statistics.return_value = mock_stats

    return mock


class TestEnhancedQueryEndpoints:
    """Test enhanced query endpoints."""

    @patch('morag.dependencies.get_hybrid_retrieval_coordinator')
    def test_enhanced_query_basic(self, mock_get_hybrid, test_client, mock_hybrid_system):
        """Test basic enhanced query."""
        mock_get_hybrid.return_value = mock_hybrid_system

        request_data = {
            "query": "What is machine learning?",
            "query_type": "simple",
            "max_results": 5
        }

        response = test_client.post("/api/v2/query", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "query_id" in data
        assert "results" in data
        assert "graph_context" in data
        assert data["query"] == request_data["query"]
        assert len(data["results"]) <= 5

    @patch('morag.dependencies.get_hybrid_retrieval_coordinator')
    def test_enhanced_query_with_graph_context(self, mock_get_hybrid, test_client, mock_hybrid_system):
        """Test enhanced query with graph context."""
        mock_get_hybrid.return_value = mock_hybrid_system

        request_data = {
            "query": "Tell me about neural networks",
            "query_type": "entity_focused",
            "expansion_strategy": "breadth_first",
            "expansion_depth": 2,
            "include_graph_context": True
        }

        response = test_client.post("/api/v2/query", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "graph_context" in data
        assert "entities" in data["graph_context"]
        assert "expansion_path" in data["graph_context"]

    @patch('morag.dependencies.get_graph_engine')
    def test_entity_query(self, mock_get_graph, test_client, mock_graph_engine):
        """Test entity query endpoint."""
        mock_get_graph.return_value = mock_graph_engine

        request_data = {
            "entity_name": "machine learning",
            "include_relations": True,
            "relation_depth": 2
        }

        response = test_client.post("/api/v2/entity/query", json=request_data)
        # Should return 200, 404 (entity not found), or 503 (graph engine not available)
        assert response.status_code in [200, 404, 503]

        if response.status_code == 200:
            data = response.json()
            assert "entity" in data
            assert "relations" in data
            assert "relation_count" in data

    @patch('morag.dependencies.get_graph_engine')
    def test_graph_traversal(self, mock_get_graph, test_client, mock_graph_engine):
        """Test graph traversal endpoint."""
        mock_get_graph.return_value = mock_graph_engine

        request_data = {
            "start_entity": "entity_1",
            "end_entity": "entity_2",
            "traversal_type": "shortest_path",
            "max_depth": 3
        }

        response = test_client.post("/api/v2/graph/traverse", json=request_data)
        # Should return 200, 500 (graph error), or 503 (graph engine not available)
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "paths" in data
            assert "processing_time_ms" in data
            assert "total_paths_found" in data

    @patch('morag.dependencies.get_graph_engine')
    def test_graph_analytics(self, mock_get_graph, test_client, mock_graph_engine):
        """Test graph analytics endpoint."""
        mock_get_graph.return_value = mock_graph_engine

        response = test_client.get("/api/v2/graph/analytics?metric_type=overview")
        assert response.status_code == 200

        data = response.json()
        assert "entity_count" in data
        assert "relation_count" in data
        assert "avg_degree" in data

    def test_query_validation_error(self, test_client):
        """Test query validation error."""
        request_data = {
            "query": "",  # Empty query should fail validation
            "max_results": 5
        }

        response = test_client.post("/api/v2/query", json=request_data)
        assert response.status_code == 400
        assert "Invalid query" in response.json()["detail"]


class TestBackwardCompatibility:
    """Test backward compatibility with legacy API."""

    @patch('morag.dependencies.get_hybrid_retrieval_coordinator')
    def test_legacy_query_compatibility(self, mock_get_hybrid, test_client, mock_hybrid_system):
        """Test legacy query compatibility."""
        mock_get_hybrid.return_value = mock_hybrid_system

        request_data = {
            "query": "What is AI?",
            "max_results": 10
        }

        response = test_client.post("/api/v1/query", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_results" in data
        assert "processing_time_ms" in data

        # Check legacy format
        for result in data["results"]:
            assert "id" in result
            assert "content" in result
            assert "score" in result
            assert "metadata" in result

    def test_migration_guide(self, test_client):
        """Test migration guide endpoint."""
        response = test_client.get("/api/v1/migration-guide")
        assert response.status_code == 200

        data = response.json()
        assert "migration_guide" in data
        assert "migration_steps" in data["migration_guide"]
        assert "examples" in data["migration_guide"]

    def test_legacy_health_check(self, test_client):
        """Test legacy health check."""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["deprecated"] is True
        assert "migration_info" in data

    def test_legacy_status(self, test_client):
        """Test legacy status endpoint."""
        response = test_client.get("/api/v1/status")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "deprecated"
        assert "replacement" in data
        assert "features" in data


class TestQueryValidation:
    """Test query validation functionality."""

    def test_query_too_short(self, test_client):
        """Test query too short validation."""
        request_data = {
            "query": "AI",  # Very short query
            "max_results": 5
        }

        response = test_client.post("/api/v2/query", json=request_data)
        assert response.status_code == 400

    def test_query_too_long(self, test_client):
        """Test query too long validation."""
        request_data = {
            "query": "A" * 1001,  # Very long query
            "max_results": 5
        }

        response = test_client.post("/api/v2/query", json=request_data)
        assert response.status_code == 400

    def test_invalid_parameters(self, test_client):
        """Test invalid parameter validation."""
        request_data = {
            "query": "What is machine learning?",
            "max_results": 0,  # Invalid max_results
        }

        response = test_client.post("/api/v2/query", json=request_data)
        assert response.status_code == 422  # Pydantic validation error
