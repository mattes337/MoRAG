"""Tests for hybrid episode mapping functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from morag_core.models import Document, DocumentChunk, DocumentMetadata
from morag_graph.graphiti.episode_mapper import (
    DocumentEpisodeMapper,
    EpisodeStrategy,
    ContextLevel,
    create_hybrid_episode_mapper,
    create_contextual_chunk_mapper
)
from morag_graph.graphiti.config import GraphitiConfig


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    metadata = DocumentMetadata(
        title="Test Document",
        source_name="test.pdf",
        source_type="document",
        mime_type="application/pdf"
    )
    
    document = Document(metadata=metadata)
    
    # Add sample chunks
    chunks_data = [
        {"content": "This is the first chunk about introduction.", "section": "Introduction"},
        {"content": "This is the second chunk about methodology.", "section": "Methodology"},
        {"content": "This is the third chunk about results.", "section": "Results"}
    ]
    
    for i, chunk_data in enumerate(chunks_data):
        chunk = DocumentChunk(
            document_id=document.id,
            content=chunk_data["content"],
            chunk_index=i,
            section=chunk_data["section"]
        )
        document.chunks.append(chunk)
    
    document.raw_text = "\n\n".join([chunk.content for chunk in document.chunks])
    return document


@pytest.fixture
def mock_config():
    """Create a mock Graphiti config."""
    return GraphitiConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        openai_api_key="test-key"
    )


class TestDocumentEpisodeMapper:
    """Test the enhanced DocumentEpisodeMapper."""
    
    def test_mapper_initialization(self, mock_config):
        """Test mapper initialization with different strategies."""
        # Test default initialization
        mapper = DocumentEpisodeMapper(mock_config)
        assert mapper.strategy == EpisodeStrategy.HYBRID
        assert mapper.context_level == ContextLevel.RICH
        assert mapper.enable_ai_summarization == True
        
        # Test custom initialization
        mapper = DocumentEpisodeMapper(
            config=mock_config,
            strategy=EpisodeStrategy.CHUNK_ONLY,
            context_level=ContextLevel.MINIMAL,
            enable_ai_summarization=False
        )
        assert mapper.strategy == EpisodeStrategy.CHUNK_ONLY
        assert mapper.context_level == ContextLevel.MINIMAL
        assert mapper.enable_ai_summarization == False
    
    def test_convenience_functions(self, mock_config):
        """Test convenience functions for creating mappers."""
        # Test hybrid mapper
        hybrid_mapper = create_hybrid_episode_mapper(mock_config, ContextLevel.RICH)
        assert hybrid_mapper.strategy == EpisodeStrategy.HYBRID
        assert hybrid_mapper.context_level == ContextLevel.RICH
        
        # Test contextual chunk mapper
        chunk_mapper = create_contextual_chunk_mapper(mock_config, ContextLevel.COMPREHENSIVE)
        assert chunk_mapper.strategy == EpisodeStrategy.CONTEXTUAL_CHUNKS
        assert chunk_mapper.context_level == ContextLevel.COMPREHENSIVE
    
    def test_basic_document_summary_generation(self, mock_config, sample_document):
        """Test basic document summary generation."""
        mapper = DocumentEpisodeMapper(
            config=mock_config,
            enable_ai_summarization=False  # Disable AI to test basic functionality
        )
        
        summary = mapper._generate_basic_document_summary(sample_document)
        
        assert "Test Document" in summary
        assert "document" in summary.lower()
        assert "3 sections" in summary or "Contains 3" in summary
        assert len(summary) > 0
    
    def test_basic_chunk_context_generation(self, mock_config, sample_document):
        """Test basic chunk context generation."""
        mapper = DocumentEpisodeMapper(
            config=mock_config,
            enable_ai_summarization=False
        )
        
        chunk = sample_document.chunks[1]  # Middle chunk
        document_summary = "Test document about research methodology"
        
        context = mapper._generate_basic_chunk_context(
            chunk, sample_document, document_summary, 1
        )
        
        assert "Section 2 of 3" in context
        assert "middle" in context.lower()
        assert "methodology" in context.lower()
        assert len(context) > 0
    
    def test_surrounding_chunks_context(self, mock_config, sample_document):
        """Test surrounding chunks context extraction."""
        mapper = DocumentEpisodeMapper(mock_config)
        
        # Test middle chunk
        context = mapper._get_surrounding_chunks_context(
            sample_document.chunks, 1, window_size=1
        )
        
        assert "Previous section" in context
        assert "Next section" in context
        assert "introduction" in context.lower()
        assert "results" in context.lower()
    
    def test_enhanced_chunk_content_creation(self, mock_config, sample_document):
        """Test enhanced chunk content creation."""
        mapper = DocumentEpisodeMapper(
            config=mock_config,
            context_level=ContextLevel.COMPREHENSIVE
        )
        
        chunk = sample_document.chunks[0]
        contextual_summary = "This section introduces the research topic and objectives."
        
        enhanced_content = mapper._create_enhanced_chunk_content(
            chunk, contextual_summary, sample_document, 0
        )
        
        assert "=== CONTEXTUAL SUMMARY ===" in enhanced_content
        assert contextual_summary in enhanced_content
        assert "=== DOCUMENT CONTEXT ===" in enhanced_content
        assert "=== ORIGINAL CONTENT ===" in enhanced_content
        assert chunk.content in enhanced_content
        assert "Test Document" in enhanced_content
    
    def test_contextual_chunk_metadata_generation(self, mock_config, sample_document):
        """Test contextual chunk metadata generation."""
        mapper = DocumentEpisodeMapper(mock_config)
        
        chunk = sample_document.chunks[0]
        contextual_summary = "Introduction section summary"
        document_summary = "Document about research"
        
        metadata = mapper._generate_contextual_chunk_metadata(
            sample_document, chunk, 0, contextual_summary, document_summary
        )
        
        # Check core fields
        assert metadata["morag_document_id"] == sample_document.id
        assert metadata["morag_chunk_id"] == chunk.id
        assert metadata["chunk_index"] == 0
        assert metadata["chunk_length"] == len(chunk.content)
        assert metadata["total_chunks"] == 3
        
        # Check context fields
        assert metadata["contextual_summary"] == contextual_summary
        assert metadata["context_level"] == ContextLevel.RICH.value
        assert metadata["strategy"] == EpisodeStrategy.HYBRID.value
        assert metadata["document_summary"] == document_summary
        
        # Check document fields
        assert metadata["document_title"] == "Test Document"
        assert metadata["section"] == "Introduction"
    
    @patch('morag_graph.graphiti.episode_mapper.GraphitiConnectionService')
    async def test_hybrid_mapping_structure(self, mock_connection_service, mock_config, sample_document):
        """Test the structure of hybrid mapping results."""
        # Mock the connection service
        mock_conn = AsyncMock()
        mock_conn.create_episode = AsyncMock(return_value=True)
        mock_connection_service.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_connection_service.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mapper = DocumentEpisodeMapper(
            config=mock_config,
            strategy=EpisodeStrategy.HYBRID,
            enable_ai_summarization=False  # Disable AI for testing
        )
        
        # Mock the connection service on the mapper
        mapper.connection_service = mock_connection_service.return_value
        
        result = await mapper.map_document_hybrid(
            document=sample_document,
            episode_name_prefix="test_doc"
        )
        
        # Check result structure
        assert result["strategy"] == "hybrid"
        assert "document_episode" in result
        assert "chunk_episodes" in result
        assert "success" in result
        assert "total_episodes" in result
        
        # Verify episodes were created
        assert mock_conn.create_episode.call_count >= 4  # 1 document + 3 chunks


class TestEpisodeStrategies:
    """Test different episode strategies."""
    
    def test_episode_strategy_enum(self):
        """Test episode strategy enumeration."""
        assert EpisodeStrategy.DOCUMENT_ONLY.value == "document_only"
        assert EpisodeStrategy.CHUNK_ONLY.value == "chunk_only"
        assert EpisodeStrategy.HYBRID.value == "hybrid"
        assert EpisodeStrategy.CONTEXTUAL_CHUNKS.value == "contextual_chunks"
    
    def test_context_level_enum(self):
        """Test context level enumeration."""
        assert ContextLevel.MINIMAL.value == "minimal"
        assert ContextLevel.STANDARD.value == "standard"
        assert ContextLevel.RICH.value == "rich"
        assert ContextLevel.COMPREHENSIVE.value == "comprehensive"


class TestIntegration:
    """Integration tests for hybrid episode mapping."""
    
    @patch('morag_graph.graphiti.episode_mapper.GraphitiConnectionService')
    async def test_contextual_chunks_mapping(self, mock_connection_service, mock_config, sample_document):
        """Test contextual chunks mapping without document episode."""
        # Mock the connection service
        mock_conn = AsyncMock()
        mock_conn.create_episode = AsyncMock(return_value=True)
        mock_connection_service.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_connection_service.return_value.__aexit__ = AsyncMock(return_value=None)
        
        mapper = create_contextual_chunk_mapper(
            config=mock_config,
            context_level=ContextLevel.COMPREHENSIVE
        )
        
        # Mock the connection service on the mapper
        mapper.connection_service = mock_connection_service.return_value
        
        results = await mapper.map_document_chunks_to_contextual_episodes(
            document=sample_document,
            chunk_episode_prefix="contextual_test"
        )
        
        # Check results
        assert len(results) == 3  # One for each chunk
        
        for i, result in enumerate(results):
            assert result["success"] == True
            assert result["chunk_index"] == i
            assert result["document_id"] == sample_document.id
            assert "contextual_summary" in result
            assert "enhanced_content_length" in result
            assert result["context_level"] == ContextLevel.COMPREHENSIVE.value
        
        # Verify episodes were created
        assert mock_conn.create_episode.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__])
