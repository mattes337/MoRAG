"""Tests for path selection components."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from morag_reasoning.path_selection import PathSelectionAgent, ReasoningPathFinder, PathRelevanceScore
from morag_graph.operations import GraphPath


class TestPathSelectionAgent:
    """Test path selection agent."""
    
    def test_init(self, mock_llm_client):
        """Test agent initialization."""
        agent = PathSelectionAgent(mock_llm_client, max_paths=5)
        assert agent.llm_client == mock_llm_client
        assert agent.max_paths == 5
        assert len(agent.strategies) == 3
        assert "forward_chaining" in agent.strategies
        assert "backward_chaining" in agent.strategies
        assert "bidirectional" in agent.strategies
    
    def test_strategies_configuration(self, path_selection_agent):
        """Test reasoning strategies configuration."""
        strategies = path_selection_agent.strategies
        
        # Test forward chaining strategy
        forward = strategies["forward_chaining"]
        assert forward.name == "forward_chaining"
        assert forward.max_depth == 4
        assert not forward.bidirectional
        assert forward.use_weights
        
        # Test backward chaining strategy
        backward = strategies["backward_chaining"]
        assert backward.name == "backward_chaining"
        assert backward.max_depth == 3
        assert not backward.bidirectional
        assert backward.use_weights
        
        # Test bidirectional strategy
        bidirectional = strategies["bidirectional"]
        assert bidirectional.name == "bidirectional"
        assert bidirectional.max_depth == 5
        assert bidirectional.bidirectional
        assert bidirectional.use_weights
    
    @pytest.mark.asyncio
    async def test_select_paths_empty_input(self, path_selection_agent):
        """Test path selection with empty input."""
        result = await path_selection_agent.select_paths("test query", [])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_select_paths_success(self, path_selection_agent, sample_graph_paths):
        """Test successful path selection."""
        query = "What is the relationship between Apple and Steve Jobs?"
        result = await path_selection_agent.select_paths(query, sample_graph_paths)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(path, PathRelevanceScore) for path in result)
        
        # Check that paths are sorted by relevance score
        scores = [path.relevance_score for path in result]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_select_paths_with_strategy(self, path_selection_agent, sample_graph_paths):
        """Test path selection with specific strategy."""
        query = "Find connection between entities"
        result = await path_selection_agent.select_paths(
            query, sample_graph_paths, strategy="bidirectional"
        )
        
        assert isinstance(result, list)
        assert len(result) <= path_selection_agent.max_paths
    
    def test_describe_path_single_entity(self, path_selection_agent, sample_entities):
        """Test path description for single entity."""
        path = GraphPath(entities=[sample_entities[0]], relations=[])
        description = path_selection_agent._describe_path(path)
        assert "Apple Inc." in description
        assert "Single entity" in description
    
    def test_describe_path_multiple_entities(self, path_selection_agent, sample_graph_paths):
        """Test path description for multiple entities."""
        path = sample_graph_paths[0]  # Apple -> Steve Jobs
        description = path_selection_agent._describe_path(path)
        assert "Apple Inc." in description
        assert "Steve Jobs" in description
        assert "FOUNDED_BY" in description
        assert "-->" in description
    
    def test_describe_path_empty(self, path_selection_agent):
        """Test path description for empty path."""
        path = GraphPath(entities=[], relations=[])
        description = path_selection_agent._describe_path(path)
        assert "Unknown" in description
    
    def test_fallback_path_selection(self, path_selection_agent, sample_graph_paths):
        """Test fallback path selection mechanism."""
        result = path_selection_agent._fallback_path_selection(sample_graph_paths)
        
        assert isinstance(result, list)
        assert len(result) <= path_selection_agent.max_paths
        assert all(isinstance(path, PathRelevanceScore) for path in result)
        assert all(path.confidence == 5.0 for path in result)  # Medium confidence for fallback
        assert all("Fallback scoring" in path.reasoning for path in result)
    
    @pytest.mark.asyncio
    async def test_score_paths_forward_chaining(self, path_selection_agent, sample_graph_paths):
        """Test path scoring with forward chaining strategy."""
        # Create initial path scores
        path_scores = [
            PathRelevanceScore(
                path=path,
                relevance_score=5.0,
                confidence=7.0,
                reasoning="Test path"
            ) for path in sample_graph_paths
        ]
        
        result = await path_selection_agent._score_paths(
            "test query", path_scores, "forward_chaining"
        )
        
        assert len(result) == len(path_scores)
        assert all(isinstance(score, PathRelevanceScore) for score in result)
    
    @pytest.mark.asyncio
    async def test_score_paths_length_penalty(self, path_selection_agent, sample_graph_paths):
        """Test path scoring with length penalty."""
        # Create a long path that should be penalized
        long_path = sample_graph_paths[-1]  # 3-entity path
        path_score = PathRelevanceScore(
            path=long_path,
            relevance_score=8.0,
            confidence=9.0,
            reasoning="Long path"
        )
        
        result = await path_selection_agent._score_paths(
            "test query", [path_score], "backward_chaining"  # max_depth=3
        )
        
        # Score should remain the same since path length equals max_depth
        assert result[0].relevance_score == 8.0
    
    def test_create_path_selection_prompt(self, path_selection_agent, sample_graph_paths):
        """Test creation of path selection prompt."""
        query = "Test query"
        prompt = path_selection_agent._create_path_selection_prompt(query, sample_graph_paths)
        
        assert query in prompt
        assert "Available Paths:" in prompt
        assert "JSON" in prompt
        assert "relevance_score" in prompt
        assert "confidence" in prompt
        assert "reasoning" in prompt


class TestReasoningPathFinder:
    """Test reasoning path finder."""
    
    def test_init(self, mock_graph_engine, path_selection_agent):
        """Test path finder initialization."""
        finder = ReasoningPathFinder(mock_graph_engine, path_selection_agent)
        assert finder.graph_engine == mock_graph_engine
        assert finder.path_selector == path_selection_agent
    
    @pytest.mark.asyncio
    async def test_find_reasoning_paths_no_paths(self, reasoning_path_finder):
        """Test path finding when no paths are discovered."""
        # Mock graph engine to return no paths
        reasoning_path_finder.graph_engine.traverse = AsyncMock(return_value={"paths": []})
        
        result = await reasoning_path_finder.find_reasoning_paths(
            "test query", ["entity1"], strategy="forward_chaining"
        )
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_find_reasoning_paths_forward_chaining(self, reasoning_path_finder, sample_graph_paths):
        """Test path finding with forward chaining strategy."""
        # Mock graph engine to return paths
        reasoning_path_finder.graph_engine.traverse = AsyncMock(
            return_value={"paths": sample_graph_paths}
        )

        # Mock the path selector's select_paths method
        reasoning_path_finder.path_selector.select_paths = AsyncMock(
            return_value=[
                PathRelevanceScore(
                    path=sample_graph_paths[0],
                    relevance_score=8.5,
                    confidence=9.0,
                    reasoning="Test path"
                )
            ]
        )

        result = await reasoning_path_finder.find_reasoning_paths(
            "test query", ["Apple Inc."], strategy="forward_chaining"
        )

        assert isinstance(result, list)
        # Should call path selector
        reasoning_path_finder.path_selector.select_paths.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_reasoning_paths_bidirectional(self, reasoning_path_finder, sample_graph_paths):
        """Test path finding with bidirectional strategy."""
        # Mock bidirectional path finding
        reasoning_path_finder.graph_engine.find_bidirectional_paths = AsyncMock(
            return_value=sample_graph_paths
        )
        
        result = await reasoning_path_finder.find_reasoning_paths(
            "test query", 
            ["Apple Inc."], 
            target_entities=["Steve Jobs"],
            strategy="bidirectional"
        )
        
        assert isinstance(result, list)
        reasoning_path_finder.graph_engine.find_bidirectional_paths.assert_called()
    
    @pytest.mark.asyncio
    async def test_find_reasoning_paths_backward_chaining(self, reasoning_path_finder, sample_graph_paths):
        """Test path finding with backward chaining strategy."""
        # Mock backward traversal
        reasoning_path_finder.graph_engine.traverse_backward = AsyncMock(
            return_value=sample_graph_paths
        )
        
        result = await reasoning_path_finder.find_reasoning_paths(
            "test query",
            ["Apple Inc."],
            target_entities=["Steve Jobs"],
            strategy="backward_chaining"
        )
        
        assert isinstance(result, list)
        reasoning_path_finder.graph_engine.traverse_backward.assert_called()
    
    @pytest.mark.asyncio
    async def test_find_reasoning_paths_fallback_methods(self, reasoning_path_finder, sample_graph_paths):
        """Test path finding with fallback methods when advanced methods not available."""
        # Remove advanced methods to test fallbacks
        delattr(reasoning_path_finder.graph_engine, 'find_bidirectional_paths')
        delattr(reasoning_path_finder.graph_engine, 'traverse_backward')
        delattr(reasoning_path_finder.graph_engine, 'traverse')
        
        # Mock fallback methods
        reasoning_path_finder.graph_engine.find_neighbors = AsyncMock(
            return_value=[sample_graph_paths[0].entities[1]]  # Return Steve Jobs as neighbor
        )
        reasoning_path_finder.graph_engine.find_shortest_path = AsyncMock(
            return_value=sample_graph_paths[0]
        )
        
        result = await reasoning_path_finder.find_reasoning_paths(
            "test query", ["Apple Inc."], strategy="forward_chaining"
        )
        
        assert isinstance(result, list)
        reasoning_path_finder.graph_engine.find_neighbors.assert_called()
    
    def test_deduplicate_paths(self, reasoning_path_finder, sample_graph_paths):
        """Test path deduplication."""
        # Create duplicate paths
        duplicate_paths = sample_graph_paths + sample_graph_paths[:2]
        
        result = reasoning_path_finder._deduplicate_paths(duplicate_paths)
        
        # Should remove duplicates
        assert len(result) == len(sample_graph_paths)
        
        # Check that entity sequences are unique
        sequences = set()
        for path in result:
            sequence = tuple(entity.name for entity in path.entities)
            assert sequence not in sequences
            sequences.add(sequence)
    
    @pytest.mark.asyncio
    async def test_discover_paths_max_paths_limit(self, reasoning_path_finder, sample_graph_paths):
        """Test that path discovery respects max_paths limit."""
        # Mock to return many paths
        many_paths = sample_graph_paths * 20  # 80 paths
        reasoning_path_finder.graph_engine.traverse = AsyncMock(
            return_value={"paths": many_paths}
        )
        
        result = await reasoning_path_finder._discover_paths(
            ["Apple Inc."], None, "forward_chaining", max_paths=10
        )
        
        assert len(result) <= 10
