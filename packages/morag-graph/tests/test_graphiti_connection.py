"""Tests for Graphiti connection service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from morag_graph.graphiti.connection import GraphitiConnectionService, create_connection_service
from morag_graph.graphiti.config import GraphitiConfig


class TestGraphitiConnectionService:
    """Test Graphiti connection service."""
    
    def test_init(self):
        """Test service initialization."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = GraphitiConnectionService(config)
        
        assert service.config == config
        assert service._graphiti is None
        assert service._connected is False
        assert service.is_connected is False
    
    def test_init_without_config(self):
        """Test service initialization without config."""
        service = GraphitiConnectionService()
        
        assert service.config is None
        assert service._graphiti is None
        assert service._connected is False
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = GraphitiConnectionService(config)
        
        # Mock the create_graphiti_instance function
        mock_graphiti = AsyncMock()
        mock_graphiti.add_episode = AsyncMock()
        
        with patch('morag_graph.graphiti.connection.create_graphiti_instance', return_value=mock_graphiti):
            result = await service.connect()
            
            assert result is True
            assert service.is_connected is True
            assert service._graphiti == mock_graphiti
            mock_graphiti.add_episode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = GraphitiConnectionService(config)
        
        # Mock the create_graphiti_instance function to raise an exception
        with patch('morag_graph.graphiti.connection.create_graphiti_instance', side_effect=Exception("Connection failed")):
            result = await service.connect()
            
            assert result is False
            assert service.is_connected is False
            assert service._graphiti is None
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection."""
        service = GraphitiConnectionService()
        mock_graphiti = AsyncMock()
        mock_graphiti.close = AsyncMock()
        
        service._graphiti = mock_graphiti
        service._connected = True
        
        await service.disconnect()
        
        assert service._graphiti is None
        assert service._connected is False
        mock_graphiti.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect_without_close_method(self):
        """Test disconnection when graphiti instance doesn't have close method."""
        service = GraphitiConnectionService()
        mock_graphiti = MagicMock()  # No close method
        
        service._graphiti = mock_graphiti
        service._connected = True
        
        await service.disconnect()
        
        assert service._graphiti is None
        assert service._connected is False
    
    @pytest.mark.asyncio
    async def test_validate_connection_success(self):
        """Test successful connection validation."""
        service = GraphitiConnectionService()
        mock_graphiti = AsyncMock()
        mock_graphiti.add_episode = AsyncMock()
        
        service._graphiti = mock_graphiti
        
        result = await service.validate_connection()
        
        assert result is True
        mock_graphiti.add_episode.assert_called_once()
        
        # Check that the call was made with expected parameters
        call_args = mock_graphiti.add_episode.call_args
        assert call_args[1]['content'] == "This is a connection test episode."
        assert call_args[1]['source_description'] == "MoRAG connection test"
        assert "connection_test_" in call_args[1]['name']
    
    @pytest.mark.asyncio
    async def test_validate_connection_failure(self):
        """Test connection validation failure."""
        service = GraphitiConnectionService()
        mock_graphiti = AsyncMock()
        mock_graphiti.add_episode = AsyncMock(side_effect=Exception("Validation failed"))
        
        service._graphiti = mock_graphiti
        
        result = await service.validate_connection()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_connection_no_graphiti(self):
        """Test connection validation without graphiti instance."""
        service = GraphitiConnectionService()
        
        result = await service.validate_connection()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_episode_success(self):
        """Test successful episode creation."""
        service = GraphitiConnectionService()
        mock_graphiti = AsyncMock()
        mock_graphiti.add_episode = AsyncMock()
        
        service._graphiti = mock_graphiti
        service._connected = True
        
        result = await service.create_episode(
            name="test_episode",
            content="Test content",
            source_description="Test source",
            metadata={"key": "value"}
        )
        
        assert result is True
        mock_graphiti.add_episode.assert_called_once_with(
            name="test_episode",
            content="Test content",
            source_description="Test source",
            metadata={"key": "value"}
        )
    
    @pytest.mark.asyncio
    async def test_create_episode_not_connected(self):
        """Test episode creation when not connected."""
        service = GraphitiConnectionService()
        
        result = await service.create_episode("test", "content")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_episode_failure(self):
        """Test episode creation failure."""
        service = GraphitiConnectionService()
        mock_graphiti = AsyncMock()
        mock_graphiti.add_episode = AsyncMock(side_effect=Exception("Creation failed"))
        
        service._graphiti = mock_graphiti
        service._connected = True
        
        result = await service.create_episode("test", "content")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_search_episodes_success(self):
        """Test successful episode search."""
        service = GraphitiConnectionService()
        mock_graphiti = AsyncMock()
        mock_results = [{"name": "episode1", "content": "content1"}]
        mock_graphiti.search = AsyncMock(return_value=mock_results)
        
        service._graphiti = mock_graphiti
        service._connected = True
        
        results = await service.search_episodes("test query", limit=5)
        
        assert results == mock_results
        mock_graphiti.search.assert_called_once_with(query="test query", limit=5)
    
    @pytest.mark.asyncio
    async def test_search_episodes_not_connected(self):
        """Test episode search when not connected."""
        service = GraphitiConnectionService()
        
        results = await service.search_episodes("test query")
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_episodes_failure(self):
        """Test episode search failure."""
        service = GraphitiConnectionService()
        mock_graphiti = AsyncMock()
        mock_graphiti.search = AsyncMock(side_effect=Exception("Search failed"))
        
        service._graphiti = mock_graphiti
        service._connected = True
        
        results = await service.search_episodes("test query")
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = GraphitiConnectionService(config)
        
        mock_graphiti = AsyncMock()
        mock_graphiti.add_episode = AsyncMock()
        mock_graphiti.close = AsyncMock()
        
        with patch('morag_graph.graphiti.connection.create_graphiti_instance', return_value=mock_graphiti):
            async with service as ctx_service:
                assert ctx_service == service
                assert service.is_connected is True
            
            assert service.is_connected is False
            mock_graphiti.close.assert_called_once()


class TestCreateConnectionService:
    """Test connection service creation function."""
    
    def test_create_service_with_config(self):
        """Test creating service with config."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = create_connection_service(config)
        
        assert isinstance(service, GraphitiConnectionService)
        assert service.config == config
    
    def test_create_service_without_config(self):
        """Test creating service without config."""
        service = create_connection_service()
        
        assert isinstance(service, GraphitiConnectionService)
        assert service.config is None
