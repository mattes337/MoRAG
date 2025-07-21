"""Comprehensive tests for all updated REST endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import uuid
from datetime import datetime

# Test models directly without importing the full server
try:
    from morag.models.legacy import LegacyQueryRequest, LegacyQueryResponse, LegacyResult
    from morag.models.enhanced_query import EnhancedQueryRequest, QueryType, ExpansionStrategy
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Models not available: {e}")
    MODELS_AVAILABLE = False


@pytest.fixture
def test_client():
    """Create test client."""
    if not MODELS_AVAILABLE:
        pytest.skip("Models not available")

    # Try to create app, skip if dependencies missing
    try:
        from morag.server import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        return TestClient(app)
    except ImportError as e:
        pytest.skip(f"Server dependencies not available: {e}")


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
        }
    ]
    return mock


@pytest.fixture
def mock_graph_engine():
    """Mock graph engine."""
    mock = AsyncMock()
    mock.get_entity.return_value = MagicMock(
        id='entity_1',
        name='Test Entity',
        type='CONCEPT',
        properties={'description': 'Test entity'}
    )
    mock.get_entity_relations.return_value = []
    mock.find_shortest_paths.return_value = []
    mock.get_graph_statistics.return_value = MagicMock(
        entity_count=100,
        relation_count=200,
        avg_degree=2.0,
        connected_components=1,
        density=0.02
    )
    return mock


class TestModels:
    """Test the updated models directly."""

    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_legacy_query_request(self):
        """Test legacy query request model."""
        request = LegacyQueryRequest(
            query="What is AI?",
            max_results=10,
            min_score=0.1
        )
        assert request.query == "What is AI?"
        assert request.max_results == 10
        assert request.min_score == 0.1

    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_legacy_query_request_with_limit(self):
        """Test legacy query request with limit field."""
        request = LegacyQueryRequest(
            query="What is AI?",
            limit=15
        )
        assert request.max_results == 15  # Should map limit to max_results

    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_legacy_query_response(self):
        """Test legacy query response model."""
        results = [
            LegacyResult(
                id="result_1",
                content="Test content",
                score=0.9,
                metadata={"test": "data"}
            )
        ]
        response = LegacyQueryResponse(
            query="What is AI?",
            results=results,
            total_results=1,
            processing_time_ms=100.0
        )
        assert response.query == "What is AI?"
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.total == 1  # Should set legacy field

    @pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
    def test_enhanced_query_request(self):
        """Test enhanced query request model."""
        request = EnhancedQueryRequest(
            query="What is machine learning?",
            query_type=QueryType.ENTITY_FOCUSED,
            max_results=5,
            expansion_strategy=ExpansionStrategy.ADAPTIVE
        )
        assert request.query == "What is machine learning?"
        assert request.query_type == QueryType.ENTITY_FOCUSED
        assert request.max_results == 5
        assert request.expansion_strategy == ExpansionStrategy.ADAPTIVE


class TestLegacyEndpoints:
    """Test legacy API endpoints for backward compatibility."""
    
    @patch('morag.dependencies.get_hybrid_retrieval_coordinator')
    def test_legacy_query(self, mock_get_hybrid, test_client, mock_hybrid_system):
        """Test legacy query endpoint."""
        mock_get_hybrid.return_value = mock_hybrid_system
        
        request_data = {
            "query": "What is AI?",
            "max_results": 10,
            "min_score": 0.1
        }
        
        response = test_client.post("/api/v1/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_results" in data
        assert "processing_time_ms" in data
        
        # Check deprecation headers
        assert "X-API-Latest-Version" in response.headers
        assert "Warning" in response.headers
    
    def test_legacy_health(self, test_client):
        """Test legacy health endpoint."""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["deprecated"] is True
        assert "migration_info" in data
        
        # Check deprecation headers
        assert "X-API-Latest-Version" in response.headers
    
    def test_migration_guide(self, test_client):
        """Test migration guide endpoint."""
        response = test_client.get("/api/v1/migration-guide")
        assert response.status_code == 200
        
        data = response.json()
        assert "migration_guide" in data
        assert "timeline" in data
        assert "support" in data


class TestEnhancedQueryEndpoints:
    """Test enhanced query API endpoints."""
    
    @patch('morag.dependencies.get_hybrid_retrieval_coordinator')
    def test_enhanced_query(self, mock_get_hybrid, test_client, mock_hybrid_system):
        """Test enhanced query endpoint."""
        mock_get_hybrid.return_value = mock_hybrid_system
        
        request_data = {
            "query": "What is machine learning?",
            "query_type": "simple",
            "max_results": 5,
            "expansion_strategy": "adaptive",
            "include_graph_context": True
        }
        
        response = test_client.post("/api/v2/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "query_id" in data
        assert "results" in data
        assert "graph_context" in data
        assert "processing_time_ms" in data
    
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
        assert response.status_code == 200
        
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
        assert response.status_code == 200
        
        data = response.json()
        assert "start_entity" in data
        assert "paths" in data
        assert "processing_time_ms" in data
    
    @patch('morag.dependencies.get_graph_engine')
    def test_graph_analytics(self, mock_get_graph, test_client, mock_graph_engine):
        """Test graph analytics endpoint."""
        mock_get_graph.return_value = mock_graph_engine
        
        request_data = {
            "metric_type": "overview"
        }
        
        response = test_client.post("/api/v2/graph/analytics", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "entity_count" in data
        assert "relation_count" in data


class TestIntelligentRetrievalEndpoint:
    """Test intelligent retrieval endpoint."""
    
    @patch('morag.endpoints.intelligent_retrieval.get_intelligent_retrieval_service')
    def test_intelligent_query(self, mock_get_service, test_client):
        """Test intelligent query endpoint."""
        mock_service = AsyncMock()
        mock_response = MagicMock()
        mock_response.query_id = str(uuid.uuid4())
        mock_response.key_facts = [{"fact": "test fact", "score": 0.9}]
        mock_service.retrieve_intelligently.return_value = mock_response
        mock_get_service.return_value = mock_service
        
        request_data = {
            "query": "Tell me about neural networks",
            "max_iterations": 3
        }
        
        response = test_client.post("/api/v2/intelligent-query", json=request_data)
        assert response.status_code == 200
    
    def test_intelligent_query_health(self, test_client):
        """Test intelligent query health endpoint."""
        response = test_client.get("/api/v2/intelligent-query/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data


class TestReasoningEndpoint:
    """Test multi-hop reasoning endpoint."""
    
    @patch('morag.dependencies.get_reasoning_path_finder')
    @patch('morag.dependencies.get_iterative_retriever')
    @patch('morag.dependencies.REASONING_AVAILABLE', True)
    def test_reasoning_query(self, mock_get_retriever, mock_get_finder, test_client):
        """Test reasoning query endpoint."""
        # Mock path finder
        mock_finder = AsyncMock()
        mock_finder.find_reasoning_paths.return_value = []
        mock_get_finder.return_value = mock_finder
        
        # Mock iterative retriever
        mock_retriever = AsyncMock()
        mock_context = MagicMock()
        mock_context.entities = {}
        mock_context.relations = {}
        mock_context.documents = {}
        mock_context.paths = []
        mock_context.metadata = {}
        mock_retriever.refine_context.return_value = mock_context
        mock_get_retriever.return_value = mock_retriever
        
        request_data = {
            "query": "How does AI relate to machine learning?",
            "start_entities": ["artificial intelligence"],
            "strategy": "forward_chaining",
            "max_depth": 3
        }
        
        response = test_client.post("/api/v2/reasoning/query", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "query" in data
        assert "strategy" in data
        assert "reasoning_time_ms" in data
    
    def test_reasoning_strategies(self, test_client):
        """Test reasoning strategies endpoint."""
        response = test_client.get("/api/v2/reasoning/strategies")
        assert response.status_code == 200
        
        data = response.json()
        assert "strategies" in data
        assert "forward_chaining" in data["strategies"]


class TestRemoteJobsEndpoint:
    """Test remote jobs endpoint."""
    
    @patch('morag.endpoints.remote_jobs.get_remote_job_service')
    def test_create_remote_job(self, mock_get_service, test_client):
        """Test create remote job endpoint."""
        mock_service = MagicMock()
        mock_job = MagicMock()
        mock_job.id = str(uuid.uuid4())
        mock_job.status = "pending"
        mock_job.created_at = datetime.now()
        mock_service.create_job.return_value = mock_job
        mock_get_service.return_value = mock_service
        
        request_data = {
            "source_file_path": "/path/to/file.mp4",
            "content_type": "video",
            "task_options": {"quality": "high"}
        }
        
        response = test_client.post("/api/v1/remote-jobs/", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "created_at" in data


class TestErrorHandling:
    """Test error handling across all endpoints."""
    
    def test_invalid_request_data(self, test_client):
        """Test handling of invalid request data."""
        # Test with invalid JSON
        response = test_client.post("/api/v2/query", json={})
        assert response.status_code == 422  # Validation error
    
    def test_nonexistent_endpoint(self, test_client):
        """Test handling of nonexistent endpoints."""
        response = test_client.get("/api/v2/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, test_client):
        """Test handling of wrong HTTP methods."""
        response = test_client.get("/api/v2/query")  # Should be POST
        assert response.status_code == 405
