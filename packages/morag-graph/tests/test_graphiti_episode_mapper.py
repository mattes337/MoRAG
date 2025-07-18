"""Tests for Graphiti episode mapper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from morag_core.models import Document, DocumentChunk, DocumentMetadata, DocumentType
from morag_graph.graphiti.episode_mapper import DocumentEpisodeMapper, create_episode_mapper
from morag_graph.graphiti.config import GraphitiConfig


class TestDocumentEpisodeMapper:
    """Test document episode mapper."""
    
    def test_init(self):
        """Test mapper initialization."""
        config = GraphitiConfig(openai_api_key="test-key")
        mapper = DocumentEpisodeMapper(config)
        
        assert mapper.config == config
        assert mapper.connection_service is not None
        assert mapper.connection_service.config == config
    
    def test_init_without_config(self):
        """Test mapper initialization without config."""
        mapper = DocumentEpisodeMapper()
        
        assert mapper.config is None
        assert mapper.connection_service is not None
        assert mapper.connection_service.config is None
    
    def create_test_document(self) -> Document:
        """Create a test document for testing."""
        metadata = DocumentMetadata(
            source_type=DocumentType.PDF,
            source_name="test.pdf",
            title="Test Document",
            file_size=1024
        )
        
        document = Document(
            id="test-doc-123",
            metadata=metadata,
            raw_text="This is a test document with some content.",
            processed_at=datetime.now()
        )
        
        # Add some chunks
        chunk1 = DocumentChunk(
            id="chunk-1",
            document_id=document.id,
            content="This is the first chunk of content.",
            chunk_index=0,
            page_number=1,
            section="Introduction"
        )
        
        chunk2 = DocumentChunk(
            id="chunk-2", 
            document_id=document.id,
            content="This is the second chunk of content.",
            chunk_index=1,
            page_number=1,
            section="Main Content"
        )
        
        document.chunks = [chunk1, chunk2]
        return document
    
    def test_generate_episode_name(self):
        """Test episode name generation."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()
        
        episode_name = mapper._generate_episode_name(document)
        
        assert "Test Document" in episode_name
        assert len(episode_name) > len("Test Document")  # Should include timestamp
    
    def test_generate_episode_name_no_title(self):
        """Test episode name generation without title."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()
        document.metadata.title = None

        episode_name = mapper._generate_episode_name(document)

        assert "test.pdf" in episode_name or "document_test-doc-123" in episode_name
    
    def test_generate_episode_content(self):
        """Test episode content generation."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()
        
        content = mapper._generate_episode_content(document)
        
        assert "Document Title: Test Document" in content
        assert "Source File: test.pdf" in content
        assert "Content Summary:" in content
        assert "Document contains 2 chunks:" in content
        assert "Chunk 1:" in content
        assert "Chunk 2:" in content
    
    def test_generate_source_description(self):
        """Test source description generation."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()
        
        description = mapper._generate_source_description(document)
        
        assert "MoRAG document" in description
        assert "type: DocumentType.PDF" in description
        assert "file: test.pdf" in description
    
    def test_generate_episode_metadata(self):
        """Test episode metadata generation."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()
        
        metadata = mapper._generate_episode_metadata(document)
        
        assert metadata["morag_document_id"] == "test-doc-123"
        assert metadata["morag_source"] == "document_episode_mapper"
        assert metadata["chunk_count"] == 2
        assert metadata["document_title"] == "Test Document"
        assert metadata["document_filename"] == "test.pdf"
        assert metadata["document_source_type"] == DocumentType.PDF
        assert metadata["document_file_size"] == 1024
        assert "created_at" in metadata
        assert "document_processed_at" in metadata
    
    def test_generate_chunk_episode_metadata(self):
        """Test chunk episode metadata generation."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()
        chunk = document.chunks[0]
        
        metadata = mapper._generate_chunk_episode_metadata(document, chunk, 0)
        
        assert metadata["morag_document_id"] == "test-doc-123"
        assert metadata["morag_chunk_id"] == "chunk-1"
        assert metadata["morag_source"] == "chunk_episode_mapper"
        assert metadata["chunk_index"] == 0
        assert metadata["original_chunk_index"] == 0
        assert metadata["page_number"] == 1
        assert metadata["section"] == "Introduction"
        assert metadata["document_title"] == "Test Document"
        assert metadata["document_filename"] == "test.pdf"
        assert "created_at" in metadata
        assert "chunk_length" in metadata
    
    @pytest.mark.asyncio
    async def test_map_document_to_episode_success(self):
        """Test successful document to episode mapping."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()

        # Mock the connection service completely
        mock_conn = AsyncMock()
        mock_conn.create_episode = AsyncMock(return_value=True)

        with patch.object(mapper, 'connection_service') as mock_service:
            mock_service.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_service.__aexit__ = AsyncMock(return_value=None)
            result = await mapper.map_document_to_episode(document)
        
        assert result["success"] is True
        assert result["document_id"] == "test-doc-123"
        assert "episode_name" in result
        assert "content_length" in result
        assert "metadata" in result
        
        # Verify episode creation was called
        mock_conn.create_episode.assert_called_once()
        call_args = mock_conn.create_episode.call_args
        assert "Test Document" in call_args[1]["name"]
        assert "Document Title: Test Document" in call_args[1]["content"]
        assert "MoRAG document" in call_args[1]["source_description"]
        assert call_args[1]["metadata"]["morag_document_id"] == "test-doc-123"
    
    @pytest.mark.asyncio
    async def test_map_document_to_episode_failure(self):
        """Test failed document to episode mapping."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()

        # Mock the connection service to fail
        mock_conn = AsyncMock()
        mock_conn.create_episode = AsyncMock(return_value=False)

        with patch.object(mapper, 'connection_service') as mock_service:
            mock_service.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_service.__aexit__ = AsyncMock(return_value=None)
            result = await mapper.map_document_to_episode(document)
        
        assert result["success"] is False
        assert result["document_id"] == "test-doc-123"
        assert "error" in result
        assert "Failed to create episode in Graphiti" in result["error"]
    
    @pytest.mark.asyncio
    async def test_map_document_to_episode_exception(self):
        """Test document to episode mapping with exception."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()

        # Mock the connection service to raise exception
        with patch.object(mapper, 'connection_service') as mock_service:
            mock_service.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))
            result = await mapper.map_document_to_episode(document)

        assert result["success"] is False
        assert result["document_id"] == "test-doc-123"
        assert "error" in result
        assert "Connection failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_map_document_chunks_to_episodes_success(self):
        """Test successful document chunks to episodes mapping."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()

        # Mock the connection service
        mock_conn = AsyncMock()
        mock_conn.create_episode = AsyncMock(return_value=True)

        with patch.object(mapper, 'connection_service') as mock_service:
            mock_service.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_service.__aexit__ = AsyncMock(return_value=None)
            results = await mapper.map_document_chunks_to_episodes(document)
        
        assert len(results) == 2
        
        # Check first chunk result
        assert results[0]["success"] is True
        assert results[0]["chunk_id"] == "chunk-1"
        assert results[0]["chunk_index"] == 0
        assert results[0]["document_id"] == "test-doc-123"
        assert "episode_name" in results[0]
        
        # Check second chunk result
        assert results[1]["success"] is True
        assert results[1]["chunk_id"] == "chunk-2"
        assert results[1]["chunk_index"] == 1
        assert results[1]["document_id"] == "test-doc-123"
        
        # Verify episode creation was called twice
        assert mock_conn.create_episode.call_count == 2
    
    @pytest.mark.asyncio
    async def test_map_document_chunks_to_episodes_no_chunks(self):
        """Test mapping document with no chunks."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()
        document.chunks = []
        
        results = await mapper.map_document_chunks_to_episodes(document)
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_map_document_chunks_to_episodes_with_prefix(self):
        """Test mapping document chunks with custom prefix."""
        mapper = DocumentEpisodeMapper()
        document = self.create_test_document()

        # Mock the connection service
        mock_conn = AsyncMock()
        mock_conn.create_episode = AsyncMock(return_value=True)

        with patch.object(mapper, 'connection_service') as mock_service:
            mock_service.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_service.__aexit__ = AsyncMock(return_value=None)
            results = await mapper.map_document_chunks_to_episodes(
                document,
                chunk_episode_prefix="custom_prefix"
            )
        
        assert len(results) == 2
        
        # Check that custom prefix was used
        call_args_list = mock_conn.create_episode.call_args_list
        assert "custom_prefix_chunk_1" in call_args_list[0][1]["name"]
        assert "custom_prefix_chunk_2" in call_args_list[1][1]["name"]


class TestCreateEpisodeMapper:
    """Test episode mapper creation function."""
    
    def test_create_mapper_with_config(self):
        """Test creating mapper with config."""
        config = GraphitiConfig(openai_api_key="test-key")
        mapper = create_episode_mapper(config)
        
        assert isinstance(mapper, DocumentEpisodeMapper)
        assert mapper.config == config
    
    def test_create_mapper_without_config(self):
        """Test creating mapper without config."""
        mapper = create_episode_mapper()
        
        assert isinstance(mapper, DocumentEpisodeMapper)
        assert mapper.config is None
