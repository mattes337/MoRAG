"""Tests for recursive fact retrieval system."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

# Import the components we're testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "morag-reasoning" / "src"))

from morag_reasoning import (
    RecursiveFactRetrievalService,
    RecursiveFactRetrievalRequest,
    GraphTraversalAgent,
    FactCriticAgent,
    RawFact,
    ScoredFact,
    FinalFact,
    LLMClient,
    LLMConfig
)


class TestRecursiveFactModels:
    """Test the data models."""
    
    def test_raw_fact_creation(self):
        """Test RawFact model creation."""
        fact = RawFact(
            fact_text="Test fact",
            source_node_id="node_123",
            extracted_from_depth=1
        )
        assert fact.fact_text == "Test fact"
        assert fact.source_node_id == "node_123"
        assert fact.extracted_from_depth == 1
        assert fact.source_property is None
        assert fact.source_qdrant_chunk_id is None
    
    def test_scored_fact_creation(self):
        """Test ScoredFact model creation."""
        fact = ScoredFact(
            fact_text="Test fact",
            source_node_id="node_123",
            extracted_from_depth=1,
            score=0.8,
            source_description="Test source"
        )
        assert fact.score == 0.8
        assert fact.source_description == "Test source"
    
    def test_final_fact_creation(self):
        """Test FinalFact model creation."""
        fact = FinalFact(
            fact_text="Test fact",
            source_node_id="node_123",
            extracted_from_depth=1,
            score=0.8,
            source_description="Test source",
            final_decayed_score=0.6
        )
        assert fact.final_decayed_score == 0.6
    
    def test_request_validation(self):
        """Test request model validation."""
        # Valid request
        request = RecursiveFactRetrievalRequest(user_query="Test query")
        assert request.user_query == "Test query"
        assert request.max_depth == 3  # default
        assert request.decay_rate == 0.2  # default
        
        # Test validation ranges
        with pytest.raises(ValueError):
            RecursiveFactRetrievalRequest(user_query="Test", max_depth=0)  # Below minimum
        
        with pytest.raises(ValueError):
            RecursiveFactRetrievalRequest(user_query="Test", decay_rate=1.5)  # Above maximum


class TestGraphTraversalAgent:
    """Test the GraphTraversalAgent."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = MagicMock()
        mock_client.get_model.return_value = "google-gla:gemini-1.5-flash"
        return mock_client
    
    @pytest.fixture
    def mock_neo4j_storage(self):
        """Create a mock Neo4j storage."""
        mock_storage = AsyncMock()
        return mock_storage
    
    @pytest.fixture
    def mock_qdrant_storage(self):
        """Create a mock Qdrant storage."""
        mock_storage = AsyncMock()
        return mock_storage
    
    @pytest.fixture
    def graph_traversal_agent(self, mock_llm_client, mock_neo4j_storage, mock_qdrant_storage):
        """Create a GraphTraversalAgent instance."""
        with patch('morag_reasoning.graph_traversal_agent.Agent') as mock_agent:
            mock_agent.return_value = MagicMock()
            return GraphTraversalAgent(
                llm_client=mock_llm_client,
                neo4j_storage=mock_neo4j_storage,
                qdrant_storage=mock_qdrant_storage
            )
    
    @pytest.mark.asyncio
    async def test_get_node_context(self, graph_traversal_agent, mock_neo4j_storage, mock_qdrant_storage):
        """Test getting node context."""
        # Mock entity data
        mock_entity = MagicMock()
        mock_entity.id = "test_node"
        mock_entity.name = "Test Entity"
        mock_entity.type = "Person"
        mock_entity.confidence = 0.9
        mock_entity.properties = {"age": 30}
        
        mock_neo4j_storage.get_entity.return_value = mock_entity
        mock_neo4j_storage.get_neighbors.return_value = []
        mock_qdrant_storage.get_chunks_by_entity_id.return_value = []
        
        context = await graph_traversal_agent._get_node_context("test_node")
        
        assert context.node_id == "test_node"
        assert context.node_properties["name"] == "Test Entity"
        assert context.node_properties["type"] == "Person"
        assert len(context.qdrant_content) == 0
        assert len(context.neighbors_and_relations) == 0
    
    @pytest.mark.asyncio
    async def test_get_node_context_with_error(self, graph_traversal_agent, mock_neo4j_storage, mock_qdrant_storage):
        """Test getting node context when errors occur."""
        # Mock errors
        mock_neo4j_storage.get_entity.side_effect = Exception("Neo4j error")
        mock_qdrant_storage.get_chunks_by_entity_id.side_effect = Exception("Qdrant error")
        
        context = await graph_traversal_agent._get_node_context("test_node")
        
        # Should return minimal context without crashing
        assert context.node_id == "test_node"
        assert context.node_properties == {"id": "test_node"}
        assert len(context.qdrant_content) == 0
        assert len(context.neighbors_and_relations) == 0


class TestFactCriticAgent:
    """Test the FactCriticAgent."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = MagicMock()
        mock_client.get_model.return_value = "google-gla:gemini-1.5-flash"
        return mock_client

    @pytest.fixture
    def fact_critic_agent(self, mock_llm_client):
        """Create a FactCriticAgent instance."""
        with patch('morag_reasoning.fact_critic_agent.Agent') as mock_agent:
            mock_agent.return_value = MagicMock()
            return FactCriticAgent(llm_client=mock_llm_client)
    
    def test_apply_relevance_decay(self, fact_critic_agent):
        """Test relevance decay application."""
        scored_facts = [
            ScoredFact(
                fact_text="Fact at depth 0",
                source_node_id="node1",
                extracted_from_depth=0,
                score=0.8,
                source_description="Source 1"
            ),
            ScoredFact(
                fact_text="Fact at depth 2",
                source_node_id="node2",
                extracted_from_depth=2,
                score=0.8,
                source_description="Source 2"
            )
        ]
        
        final_facts = fact_critic_agent.apply_relevance_decay(scored_facts, decay_rate=0.2)
        
        assert len(final_facts) == 2
        
        # Fact at depth 0: score = 0.8 * (1 - 0.2 * 0) = 0.8
        assert final_facts[0].final_decayed_score == 0.8
        
        # Fact at depth 2: score = 0.8 * (1 - 0.2 * 2) = 0.8 * 0.6 = 0.48
        assert final_facts[1].final_decayed_score == 0.48
        
        # Should be sorted by final score (highest first)
        assert final_facts[0].final_decayed_score >= final_facts[1].final_decayed_score


class TestRecursiveFactRetrievalService:
    """Test the main service."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_client = MagicMock()
        mock_client.get_model.return_value = "google-gla:gemini-1.5-flash"
        return mock_client
    
    @pytest.fixture
    def mock_neo4j_storage(self):
        """Create a mock Neo4j storage."""
        mock_storage = AsyncMock()
        return mock_storage
    
    @pytest.fixture
    def mock_qdrant_storage(self):
        """Create a mock Qdrant storage."""
        mock_storage = AsyncMock()
        return mock_storage
    
    @pytest.fixture
    def service(self, mock_llm_client, mock_neo4j_storage, mock_qdrant_storage):
        """Create a RecursiveFactRetrievalService instance."""
        return RecursiveFactRetrievalService(
            llm_client=mock_llm_client,
            neo4j_storage=mock_neo4j_storage,
            qdrant_storage=mock_qdrant_storage
        )
    
    @pytest.mark.asyncio
    async def test_extract_initial_entities_empty(self, service):
        """Test extracting initial entities with empty query."""
        entities = await service._extract_initial_entities("")
        assert entities == []
    
    @pytest.mark.asyncio
    async def test_map_entities_to_nodes_empty(self, service):
        """Test mapping entities to nodes with empty list."""
        node_ids = await service._map_entities_to_nodes([])
        assert node_ids == []
    
    @pytest.mark.asyncio
    async def test_get_node_name(self, service, mock_neo4j_storage):
        """Test getting node name."""
        # Mock entity with name
        mock_entity = MagicMock()
        mock_entity.name = "Test Entity"
        mock_neo4j_storage.get_entity.return_value = mock_entity
        
        name = await service._get_node_name("test_node")
        assert name == "Test Entity"
        
        # Test with no entity found
        mock_neo4j_storage.get_entity.return_value = None
        name = await service._get_node_name("test_node")
        assert name == "test_node"
        
        # Test with error
        mock_neo4j_storage.get_entity.side_effect = Exception("Error")
        name = await service._get_node_name("test_node")
        assert name == "test_node"
    
    def test_parse_next_nodes(self, service):
        """Test parsing next nodes string."""
        # Test tuple format
        next_nodes = service._parse_next_nodes("(node1, rel1), (node2, rel2)")
        assert "node1" in next_nodes
        assert "node2" in next_nodes
        
        # Test comma-separated format
        next_nodes = service._parse_next_nodes("node1, node2, node3")
        assert next_nodes == ["node1", "node2", "node3"]
        
        # Test empty/invalid format
        next_nodes = service._parse_next_nodes("NONE")
        assert next_nodes == []
        
        next_nodes = service._parse_next_nodes("invalid format")
        assert next_nodes == []


@pytest.mark.asyncio
async def test_integration_basic_flow():
    """Test basic integration flow with mocked components."""
    # This test verifies that all components can be instantiated and basic flow works
    
    # Create mock clients
    mock_llm_client = MagicMock()
    mock_llm_client.get_model.return_value = "google-gla:gemini-1.5-flash"
    
    mock_neo4j_storage = AsyncMock()
    mock_qdrant_storage = AsyncMock()
    
    # Create service
    service = RecursiveFactRetrievalService(
        llm_client=mock_llm_client,
        neo4j_storage=mock_neo4j_storage,
        qdrant_storage=mock_qdrant_storage
    )
    
    # Verify service components are initialized
    assert service.llm_client is not None
    assert service.neo4j_storage is not None
    assert service.qdrant_storage is not None
    assert service.entity_service is not None
    assert service.graph_traversal_agent is not None
    assert service.fact_critic_agent is not None


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
