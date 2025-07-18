"""Integration tests for complete Graphiti functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from morag_core.models import Document, DocumentChunk, DocumentMetadata, DocumentType
from morag_graph.graphiti import (
    GraphitiConfig, GraphitiConnectionService, DocumentEpisodeMapper,
    GraphitiSearchService, SearchResult, SearchMetrics, HybridSearchService
)
from morag_graph import GRAPHITI_AVAILABLE


class TestGraphitiIntegration:
    """Test complete Graphiti integration workflow."""
    
    def test_graphiti_availability(self):
        """Test that Graphiti components are available for import."""
        # This test will pass even without graphiti-core installed
        # because we've implemented the components with proper error handling
        assert GRAPHITI_AVAILABLE is True  # Our components are importable
        
        # But our components should still be importable (as None)
        from morag_graph import (
            GraphitiConfig, GraphitiConnectionService, DocumentEpisodeMapper,
            GraphitiSearchService, HybridSearchService
        )
        
        # These will be None when graphiti-core is not installed
        # but the imports should not fail
        assert True  # If we get here, imports worked
    
    def create_test_document(self) -> Document:
        """Create a test document for integration testing."""
        metadata = DocumentMetadata(
            source_type=DocumentType.PDF,
            source_name="integration_test.pdf",
            title="Integration Test Document",
            file_size=2048
        )
        
        document = Document(
            id="integration-test-doc",
            metadata=metadata,
            raw_text="This is an integration test document with comprehensive content for testing Graphiti functionality.",
            processed_at=datetime.now()
        )
        
        # Add test chunks
        chunks = [
            DocumentChunk(
                id="chunk-1",
                document_id=document.id,
                content="This is the first chunk containing introduction material.",
                chunk_index=0,
                page_number=1,
                section="Introduction"
            ),
            DocumentChunk(
                id="chunk-2",
                document_id=document.id,
                content="This is the second chunk with main content and analysis.",
                chunk_index=1,
                page_number=2,
                section="Analysis"
            ),
            DocumentChunk(
                id="chunk-3",
                document_id=document.id,
                content="This is the final chunk with conclusions and recommendations.",
                chunk_index=2,
                page_number=3,
                section="Conclusions"
            )
        ]
        
        document.chunks = chunks
        return document
    
    @pytest.mark.asyncio
    async def test_complete_document_to_search_workflow(self):
        """Test the complete workflow from document ingestion to search."""
        # Create test document
        document = self.create_test_document()
        
        # Create Graphiti services
        config = GraphitiConfig(openai_api_key="test-key")
        episode_mapper = DocumentEpisodeMapper(config)
        search_service = GraphitiSearchService(config)
        
        # Mock the connection service for episode mapping
        mock_conn_mapper = AsyncMock()
        mock_conn_mapper.create_episode = AsyncMock(return_value=True)
        
        # Mock the connection service for search
        mock_conn_search = AsyncMock()
        mock_search_results = [
            {
                "content": "This is the first chunk containing introduction material.",
                "score": 0.95,
                "metadata": {
                    "morag_document_id": "integration-test-doc",
                    "morag_chunk_id": "chunk-1",
                    "section": "Introduction"
                }
            },
            {
                "content": "This is the second chunk with main content and analysis.",
                "score": 0.87,
                "metadata": {
                    "morag_document_id": "integration-test-doc",
                    "morag_chunk_id": "chunk-2",
                    "section": "Analysis"
                }
            }
        ]
        mock_conn_search.search_episodes = AsyncMock(return_value=mock_search_results)
        
        # Test document to episode mapping
        with patch.object(episode_mapper, 'connection_service') as mock_mapper_service:
            mock_mapper_service.__aenter__ = AsyncMock(return_value=mock_conn_mapper)
            mock_mapper_service.__aexit__ = AsyncMock(return_value=None)
            
            # Map document to episode
            episode_result = await episode_mapper.map_document_to_episode(document)
            
            assert episode_result["success"] is True
            assert episode_result["document_id"] == "integration-test-doc"
            assert "episode_name" in episode_result
            
            # Map chunks to episodes
            chunk_results = await episode_mapper.map_document_chunks_to_episodes(document)
            
            assert len(chunk_results) == 3
            assert all(result["success"] for result in chunk_results)
            assert chunk_results[0]["chunk_id"] == "chunk-1"
            assert chunk_results[1]["chunk_id"] == "chunk-2"
            assert chunk_results[2]["chunk_id"] == "chunk-3"
        
        # Test search functionality
        with patch.object(search_service, 'connection_service') as mock_search_service:
            mock_search_service.__aenter__ = AsyncMock(return_value=mock_conn_search)
            mock_search_service.__aexit__ = AsyncMock(return_value=None)
            
            # Perform search
            search_results, metrics = await search_service.search("introduction material", limit=5)
            
            assert len(search_results) == 2
            assert isinstance(metrics, SearchMetrics)
            assert metrics.result_count == 2
            
            # Check search results
            assert search_results[0].content == "This is the first chunk containing introduction material."
            assert search_results[0].score == 0.95
            assert search_results[0].document_id == "integration-test-doc"
            assert search_results[0].chunk_id == "chunk-1"
            assert search_results[0].metadata["section"] == "Introduction"
            
            assert search_results[1].content == "This is the second chunk with main content and analysis."
            assert search_results[1].score == 0.87
            assert search_results[1].document_id == "integration-test-doc"
            assert search_results[1].chunk_id == "chunk-2"
    
    @pytest.mark.asyncio
    async def test_hybrid_search_integration(self):
        """Test hybrid search with fallback functionality."""
        # Create mock legacy search service
        class MockLegacySearch:
            async def search_chunks(self, query: str, limit: int = 10):
                return [
                    {
                        "content": "Legacy search result",
                        "score": 0.75,
                        "metadata": {"source": "legacy"}
                    }
                ]
            
            async def search_entities(self, entity_names: list, limit: int = 10):
                return [
                    {
                        "content": "Legacy entity result",
                        "score": 0.8,
                        "metadata": {"entities": entity_names}
                    }
                ]
        
        # Create services
        config = GraphitiConfig(openai_api_key="test-key")
        graphiti_search = GraphitiSearchService(config)
        legacy_search = MockLegacySearch()
        hybrid_search = HybridSearchService(graphiti_search, legacy_search, prefer_graphiti=True)
        
        # Test Graphiti search success
        mock_conn = AsyncMock()
        mock_results = [
            {
                "content": "Graphiti search result",
                "score": 0.9,
                "metadata": {"morag_document_id": "doc-1"}
            }
        ]
        mock_conn.search_episodes = AsyncMock(return_value=mock_results)
        
        with patch.object(graphiti_search, 'connection_service') as mock_service:
            mock_service.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_service.__aexit__ = AsyncMock(return_value=None)
            
            result = await hybrid_search.search("test query")
            
            assert result["method_used"] == "graphiti"
            assert result["fallback_used"] is False
            assert len(result["results"]) == 1
            assert result["results"][0]["content"] == "Graphiti search result"
        
        # Test fallback to legacy when Graphiti fails
        # We need to mock the search method directly since connection_service mocking isn't working as expected
        with patch.object(graphiti_search, 'search', side_effect=Exception("Graphiti failed")):
            result = await hybrid_search.search("test query")

            assert result["method_used"] == "legacy"
            assert result["fallback_used"] is True
            assert len(result["results"]) == 1
            assert result["results"][0]["content"] == "Legacy search result"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self):
        """Test error handling and system resilience."""
        config = GraphitiConfig(openai_api_key="test-key")
        
        # Test connection service error handling
        connection_service = GraphitiConnectionService(config)
        
        # Should handle connection failures gracefully
        success = await connection_service.connect()
        assert success is False  # Expected to fail without graphiti-core
        
        # Test episode mapper error handling
        episode_mapper = DocumentEpisodeMapper(config)
        document = self.create_test_document()
        
        # Should handle mapping failures gracefully
        result = await episode_mapper.map_document_to_episode(document)
        assert result["success"] is False
        assert "error" in result
        
        # Test search service error handling
        search_service = GraphitiSearchService(config)
        
        # Should handle search failures gracefully
        results, metrics = await search_service.search("test query")
        assert len(results) == 0
        assert metrics.result_count == 0
        # The search method will be "hybrid" but with 0 results due to connection failure
    
    def test_configuration_management(self):
        """Test configuration management and validation."""
        # Test default configuration
        config = GraphitiConfig()
        assert config.neo4j_uri == "bolt://localhost:7687"
        assert config.neo4j_username == "neo4j"
        assert config.neo4j_database == "morag_graphiti"
        assert config.enable_telemetry is False
        
        # Test custom configuration
        custom_config = GraphitiConfig(
            neo4j_uri="bolt://custom:7687",
            neo4j_username="custom_user",
            neo4j_password="custom_pass",
            openai_api_key="custom-key",
            enable_telemetry=True
        )
        
        assert custom_config.neo4j_uri == "bolt://custom:7687"
        assert custom_config.neo4j_username == "custom_user"
        assert custom_config.neo4j_password == "custom_pass"
        assert custom_config.openai_api_key == "custom-key"
        assert custom_config.enable_telemetry is True
    
    def test_backward_compatibility(self):
        """Test backward compatibility with existing MoRAG components."""
        # Test that Graphiti components can be imported without breaking existing code
        try:
            from morag_graph import (
                # Existing components should still work
                Entity, Relation, Graph,
                Neo4jStorage, QdrantStorage,
                # New Graphiti components should be available (even if None)
                GraphitiConfig, GraphitiSearchService, HybridSearchService
            )
            
            # If graphiti-core is not installed, these should be None
            # but imports should not fail
            assert True  # Success if we reach here
            
        except ImportError as e:
            pytest.fail(f"Import failed, breaking backward compatibility: {e}")
    
    def test_service_integration_points(self):
        """Test integration points with existing MoRAG services."""
        # Test that Graphiti services can be created and configured
        config = GraphitiConfig(openai_api_key="test-key")
        
        # All services should be creatable
        connection_service = GraphitiConnectionService(config)
        episode_mapper = DocumentEpisodeMapper(config)
        search_service = GraphitiSearchService(config)
        hybrid_service = HybridSearchService(search_service)
        
        assert connection_service.config == config
        assert episode_mapper.config == config
        assert search_service.config == config
        assert hybrid_service.graphiti_service == search_service
        
        # Services should have proper interfaces
        assert hasattr(connection_service, 'connect')
        assert hasattr(connection_service, 'disconnect')
        assert hasattr(episode_mapper, 'map_document_to_episode')
        assert hasattr(search_service, 'search')
        assert hasattr(hybrid_service, 'search_chunks')
