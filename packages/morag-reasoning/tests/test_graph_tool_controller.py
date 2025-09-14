"""Tests for Graph Tool Controller.

Tests the Gemini function calling schema implementation for graph operations.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from morag_reasoning.graph_tool_controller import (
    GraphToolController,
    ToolCall,
    ToolResult,
    ToolCallError,
    ActionTrace
)


class TestGraphToolController:
    """Test cases for GraphToolController."""
    
    @pytest.fixture
    def controller(self):
        """Create a test controller instance."""
        return GraphToolController(
            max_hops=2,
            score_threshold=0.8,
            max_entities_per_call=5,
            max_neighbors_per_entity=3,
            max_chunks_per_entity=2
        )
    
    @pytest.fixture
    def mock_services(self):
        """Mock external services."""
        mock_graph_store = Mock()
        mock_fact_extractor = Mock()
        mock_embedding_service = Mock()
        
        return {
            'graph_store': mock_graph_store,
            'fact_extractor': mock_fact_extractor,
            'embedding_service': mock_embedding_service
        }
    
    def test_function_specs(self, controller):
        """Test that function specs are correctly defined."""
        specs = controller.get_function_specs()
        
        assert len(specs) == 5
        
        # Check all required functions are present
        function_names = {spec['name'] for spec in specs}
        expected_names = {
            'extract_entities', 'match_entity', 'expand_neighbors',
            'fetch_chunk', 'extract_facts'
        }
        assert function_names == expected_names
        
        # Check extract_entities spec
        extract_spec = next(s for s in specs if s['name'] == 'extract_entities')
        assert 'text' in extract_spec['parameters']['properties']
        assert extract_spec['parameters']['required'] == ['text']
        
        # Check expand_neighbors spec has depth limits
        expand_spec = next(s for s in specs if s['name'] == 'expand_neighbors')
        depth_prop = expand_spec['parameters']['properties']['depth']
        assert depth_prop['minimum'] == 1
        assert depth_prop['maximum'] == 3
    
    @pytest.mark.asyncio
    async def test_tool_policy_enforced(self, controller):
        """Test that unauthorized tools are rejected."""
        unauthorized_call = ToolCall(name="delete_node", args={})
        
        with pytest.raises(ToolCallError) as exc_info:
            await controller.handle_tool_call(unauthorized_call)
        
        assert "not allowed" in str(exc_info.value)
        assert "delete_node" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_extract_entities_success(self, controller, mock_services):
        """Test successful entity extraction."""
        # Mock fact extractor
        mock_fact = Mock()
        mock_fact.subject = "Python"
        mock_fact.object = "programming language"
        mock_fact.fact_id = "fact_1"
        
        mock_services['fact_extractor'].extract_facts = AsyncMock(
            return_value=[mock_fact]
        )
        
        controller.fact_extractor = mock_services['fact_extractor']
        
        call = ToolCall(name="extract_entities", args={"text": "Python is a programming language"})
        result = await controller.handle_tool_call(call)
        
        assert result.error is None
        assert "entities" in result.result
        assert "Python" in result.result["entities"]
        assert "programming language" in result.result["entities"]
        assert result.result["count"] == 2
    
    @pytest.mark.asyncio
    async def test_extract_entities_missing_text(self, controller):
        """Test entity extraction with missing text parameter."""
        call = ToolCall(name="extract_entities", args={})
        result = await controller.handle_tool_call(call)
        
        assert result.error is not None
        assert "Text parameter is required" in result.error
    
    @pytest.mark.asyncio
    async def test_match_entity_success(self, controller, mock_services):
        """Test successful entity matching."""
        # Mock entity
        mock_entity = Mock()
        mock_entity.id = "entity_123"
        mock_entity.name = "Python Programming"
        mock_entity.type = "Technology"
        
        mock_services['graph_store'].search_entities = AsyncMock(
            return_value=[mock_entity]
        )
        
        controller.graph_store = mock_services['graph_store']
        
        call = ToolCall(name="match_entity", args={"name": "Python"})
        result = await controller.handle_tool_call(call)
        
        assert result.error is None
        assert result.result["entity_id"] == "entity_123"
        assert result.result["canonical_name"] == "Python Programming"
        assert result.result["confidence"] == 1.0
    
    @pytest.mark.asyncio
    async def test_match_entity_not_found(self, controller, mock_services):
        """Test entity matching when entity is not found."""
        mock_services['graph_store'].search_entities = AsyncMock(return_value=[])
        controller.graph_store = mock_services['graph_store']
        
        call = ToolCall(name="match_entity", args={"name": "NonexistentEntity"})
        result = await controller.handle_tool_call(call)
        
        assert result.error is None
        assert result.result["entity_id"] is None
        assert result.result["confidence"] == 0.0
    
    @pytest.mark.asyncio
    async def test_expand_neighbors_success(self, controller, mock_services):
        """Test successful neighbor expansion."""
        # Mock neighbors
        mock_neighbor1 = Mock()
        mock_neighbor1.id = "neighbor_1"
        mock_neighbor1.name = "Related Entity 1"
        mock_neighbor1.type = "Concept"
        mock_neighbor1.properties = {"category": "tech"}
        
        mock_neighbor2 = Mock()
        mock_neighbor2.id = "neighbor_2"
        mock_neighbor2.name = "Related Entity 2"
        mock_neighbor2.type = "Tool"
        mock_neighbor2.properties = {"category": "software"}
        
        mock_services['graph_store'].get_neighbors = AsyncMock(
            return_value=[mock_neighbor1, mock_neighbor2]
        )
        
        controller.graph_store = mock_services['graph_store']
        
        call = ToolCall(name="expand_neighbors", args={"entity_id": "entity_123", "depth": 2})
        result = await controller.handle_tool_call(call)
        
        assert result.error is None
        assert result.result["count"] == 2
        assert result.result["depth_used"] == 2
        assert len(result.result["neighbors"]) == 2
        
        neighbor_ids = [n["id"] for n in result.result["neighbors"]]
        assert "neighbor_1" in neighbor_ids
        assert "neighbor_2" in neighbor_ids
    
    @pytest.mark.asyncio
    async def test_expand_neighbors_depth_limit(self, controller, mock_services):
        """Test that depth limits are enforced."""
        mock_services['graph_store'].get_neighbors = AsyncMock(return_value=[])
        controller.graph_store = mock_services['graph_store']
        
        # Request depth > max_hops (2)
        call = ToolCall(name="expand_neighbors", args={"entity_id": "entity_123", "depth": 5})
        result = await controller.handle_tool_call(call)
        
        assert result.error is None
        assert result.result["depth_used"] == 2  # Should be limited to max_hops
        
        # Verify the call was made with limited depth
        mock_services['graph_store'].get_neighbors.assert_called_with(
            "entity_123", max_depth=2
        )
    
    @pytest.mark.asyncio
    async def test_fetch_chunk_success(self, controller, mock_services):
        """Test successful chunk fetching."""
        # Mock entity
        mock_entity = Mock()
        mock_entity.id = "entity_123"
        mock_entity.name = "Python"
        
        mock_services['graph_store'].get_entity = AsyncMock(return_value=mock_entity)
        controller.graph_store = mock_services['graph_store']
        
        call = ToolCall(name="fetch_chunk", args={"entity_id": "entity_123"})
        result = await controller.handle_tool_call(call)
        
        assert result.error is None
        assert result.result["entity_name"] == "Python"
        assert "chunks" in result.result
        assert result.result["count"] == 0  # No chunks in mock implementation
    
    @pytest.mark.asyncio
    async def test_fetch_chunk_entity_not_found(self, controller, mock_services):
        """Test chunk fetching when entity is not found."""
        mock_services['graph_store'].get_entity = AsyncMock(return_value=None)
        controller.graph_store = mock_services['graph_store']
        
        call = ToolCall(name="fetch_chunk", args={"entity_id": "nonexistent"})
        result = await controller.handle_tool_call(call)
        
        assert result.error is None
        assert "Entity not found" in result.result["error"]
        assert result.result["count"] == 0
    
    @pytest.mark.asyncio
    async def test_extract_facts_success(self, controller, mock_services):
        """Test successful fact extraction."""
        # Mock facts
        mock_fact1 = Mock()
        mock_fact1.fact_id = "fact_1"
        mock_fact1.subject = "Python"
        mock_fact1.predicate = "is_a"
        mock_fact1.object = "programming language"
        mock_fact1.confidence = 0.9
        mock_fact1.metadata = {"domain": "technology"}
        
        mock_fact2 = Mock()
        mock_fact2.fact_id = "fact_2"
        mock_fact2.subject = "Python"
        mock_fact2.predicate = "used_for"
        mock_fact2.object = "web development"
        mock_fact2.confidence = 0.7  # Below threshold (0.8)
        mock_fact2.metadata = {"domain": "technology"}
        
        mock_services['fact_extractor'].extract_facts = AsyncMock(
            return_value=[mock_fact1, mock_fact2]
        )
        
        controller.fact_extractor = mock_services['fact_extractor']
        
        call = ToolCall(name="extract_facts", args={"text": "Python is a programming language used for web development"})
        result = await controller.handle_tool_call(call)
        
        assert result.error is None
        assert result.result["total_extracted"] == 2
        assert result.result["filtered_count"] == 1  # Only fact1 passes threshold
        assert result.result["score_threshold"] == 0.8
        
        # Check that facts have structured citations
        facts = result.result["facts"]
        assert len(facts) == 1
        fact = facts[0]
        assert fact["source"].startswith("[document:extracted_text:")
        assert "fact_id=fact_1" in fact["source"]
    
    @pytest.mark.asyncio
    async def test_citation_format_validation(self, controller):
        """Test that citations follow the structured format."""
        # Test various citation formats that should be generated
        test_cases = [
            "[document:extracted_text:0:fact_id=fact_1]",
            "[document:unknown:0:entity_id=entity_123]",
            "[pdf:research.pdf:1:page=15:chapter=2.2]",
            "[audio:interview.mp3:3:timecode=00:15:30]"
        ]
        
        import re
        citation_pattern = r"^\[\w+:[^:]+:\d+(?::[^\]]+)*\]$"
        
        for citation in test_cases:
            assert re.match(citation_pattern, citation), f"Citation '{citation}' doesn't match pattern"
    
    def test_action_traces(self, controller):
        """Test action trace recording."""
        # Initially empty
        assert len(controller.get_action_traces()) == 0
        
        # Add a mock trace
        trace = ActionTrace(
            tool_name="extract_entities",
            args={"text": "test"},
            result={"entities": ["test"]},
            execution_time=0.1,
            timestamp="123456789"
        )
        controller.action_traces.append(trace)
        
        traces = controller.get_action_traces()
        assert len(traces) == 1
        assert traces[0].tool_name == "extract_entities"
        
        # Test clearing traces
        controller.clear_action_traces()
        assert len(controller.get_action_traces()) == 0
    
    def test_controller_stats(self, controller):
        """Test controller statistics."""
        # Add some mock traces
        traces = [
            ActionTrace("extract_entities", {}, {}, 0.1, "1"),
            ActionTrace("match_entity", {}, {}, 0.2, "2"),
            ActionTrace("extract_entities", {}, {}, 0.15, "3"),
            ActionTrace("expand_neighbors", {}, None, 0.05, "4", error="Test error")
        ]
        
        controller.action_traces.extend(traces)
        
        stats = controller.get_stats()
        
        assert stats["total_calls"] == 4
        assert stats["tool_counts"]["extract_entities"] == 2
        assert stats["tool_counts"]["match_entity"] == 1
        assert stats["tool_counts"]["expand_neighbors"] == 1
        assert stats["total_execution_time"] == 0.5
        assert stats["error_count"] == 1
        assert stats["success_rate"] == 0.75
    
    @pytest.mark.asyncio
    async def test_max_entities_limit(self, controller, mock_services):
        """Test that entity extraction respects max entities limit."""
        # Create more facts than the limit (5)
        mock_facts = []
        for i in range(10):
            mock_fact = Mock()
            mock_fact.subject = f"Entity_{i}"
            mock_fact.object = f"Object_{i}"
            mock_fact.fact_id = f"fact_{i}"
            mock_facts.append(mock_fact)
        
        mock_services['fact_extractor'].extract_facts = AsyncMock(return_value=mock_facts)
        controller.fact_extractor = mock_services['fact_extractor']
        
        call = ToolCall(name="extract_entities", args={"text": "Long text with many entities"})
        result = await controller.handle_tool_call(call)
        
        assert result.error is None
        # Should be limited by max_entities_per_call (5) * 2 (subject + object) = 10 max
        # But due to set deduplication, actual count may be less
        assert result.result["count"] <= 10
    
    @pytest.mark.asyncio
    async def test_max_neighbors_limit(self, controller, mock_services):
        """Test that neighbor expansion respects max neighbors limit."""
        # Create more neighbors than the limit (3)
        mock_neighbors = []
        for i in range(10):
            mock_neighbor = Mock()
            mock_neighbor.id = f"neighbor_{i}"
            mock_neighbor.name = f"Neighbor {i}"
            mock_neighbor.type = "Entity"
            mock_neighbor.properties = {}
            mock_neighbors.append(mock_neighbor)
        
        mock_services['graph_store'].get_neighbors = AsyncMock(return_value=mock_neighbors)
        controller.graph_store = mock_services['graph_store']
        
        call = ToolCall(name="expand_neighbors", args={"entity_id": "entity_123"})
        result = await controller.handle_tool_call(call)
        
        assert result.error is None
        assert result.result["count"] == 3  # Limited to max_neighbors_per_entity
        assert len(result.result["neighbors"]) == 3


class TestToolCallIntegration:
    """Integration tests for tool calling workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test a complete workflow using multiple tools."""
        controller = GraphToolController()
        
        # Mock all services
        with patch.multiple(
            controller,
            fact_extractor=AsyncMock(),
            graph_store=AsyncMock(),
            embedding_service=AsyncMock()
        ):
            # Step 1: Extract entities
            mock_fact = Mock()
            mock_fact.subject = "Python"
            mock_fact.object = "programming language"
            mock_fact.fact_id = "fact_1"
            
            controller.fact_extractor.extract_facts.return_value = [mock_fact]
            
            extract_call = ToolCall(name="extract_entities", args={"text": "Python is great"})
            extract_result = await controller.handle_tool_call(extract_call)
            
            assert extract_result.error is None
            assert "Python" in extract_result.result["entities"]
            
            # Step 2: Match entity
            mock_entity = Mock()
            mock_entity.id = "python_123"
            mock_entity.name = "Python Programming Language"
            
            controller.graph_store.search_entities.return_value = [mock_entity]
            
            match_call = ToolCall(name="match_entity", args={"name": "Python"})
            match_result = await controller.handle_tool_call(match_call)
            
            assert match_result.error is None
            assert match_result.result["entity_id"] == "python_123"
            
            # Step 3: Expand neighbors
            mock_neighbor = Mock()
            mock_neighbor.id = "django_456"
            mock_neighbor.name = "Django Framework"
            mock_neighbor.type = "Framework"
            mock_neighbor.properties = {}
            
            controller.graph_store.get_neighbors.return_value = [mock_neighbor]
            
            expand_call = ToolCall(name="expand_neighbors", args={"entity_id": "python_123"})
            expand_result = await controller.handle_tool_call(expand_call)
            
            assert expand_result.error is None
            assert expand_result.result["count"] == 1
            assert expand_result.result["neighbors"][0]["name"] == "Django Framework"
            
            # Verify action traces
            traces = controller.get_action_traces()
            assert len(traces) == 3
            assert traces[0].tool_name == "extract_entities"
            assert traces[1].tool_name == "match_entity"
            assert traces[2].tool_name == "expand_neighbors"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery in tool calls."""
        controller = GraphToolController()
        
        # Test with failing service
        with patch.object(controller, 'fact_extractor', None):
            # This should trigger service initialization
            with patch('morag_reasoning.graph_tool_controller.GraphFactExtractor') as mock_extractor_class:
                mock_extractor_class.side_effect = Exception("Service unavailable")
                
                call = ToolCall(name="extract_entities", args={"text": "test"})
                result = await controller.handle_tool_call(call)
                
                assert result.error is not None
                assert "Service unavailable" in result.error
                
                # Verify error is recorded in traces
                traces = controller.get_action_traces()
                assert len(traces) == 1
                assert traces[0].error is not None