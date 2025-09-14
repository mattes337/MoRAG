"""Tests for semantic chunking system."""

import pytest
import asyncio
import sys
import os

# Add the packages directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-core', 'src'))

from morag_core.chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    SemanticChunker,
    ChunkerFactory,
    create_chunker,
)
from morag_core.ai import SemanticChunkingAgent


class TestChunkingConfig:
    """Test chunking configuration."""
    
    def test_config_creation(self):
        """Test creating chunking configuration."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            max_chunk_size=3000,
            min_chunk_size=400,
            overlap_size=150
        )
        
        assert config.strategy == ChunkingStrategy.SEMANTIC
        assert config.max_chunk_size == 3000
        assert config.min_chunk_size == 400
        assert config.overlap_size == 150
    
    def test_config_defaults(self):
        """Test configuration defaults."""
        config = ChunkingConfig()
        
        assert config.strategy == ChunkingStrategy.SEMANTIC
        assert config.max_chunk_size == 4000
        assert config.min_chunk_size == 500
        assert config.overlap_size == 200
        assert config.min_confidence == 0.6
        assert config.use_ai_analysis is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ChunkingConfig(max_chunk_size=2000, min_chunk_size=500)
        config.validate_config()  # Should not raise
        
        # Invalid config - min >= max
        with pytest.raises(ValueError, match="min_chunk_size must be less than max_chunk_size"):
            config = ChunkingConfig(max_chunk_size=1000, min_chunk_size=1000)
            config.validate_config()
        
        # Invalid config - overlap >= min
        with pytest.raises(ValueError, match="overlap_size must be less than min_chunk_size"):
            config = ChunkingConfig(min_chunk_size=500, overlap_size=500)
            config.validate_config()
    
    def test_content_type_configs(self):
        """Test content-type specific configurations."""
        # Document config
        doc_config = ChunkingConfig.for_documents(max_chunk_size=5000)
        assert doc_config.max_chunk_size == 5000
        assert doc_config.content_type == "document"
        assert doc_config.preserve_code_blocks is True
        assert doc_config.preserve_tables is True
        
        # Audio config
        audio_config = ChunkingConfig.for_audio_transcripts(max_chunk_size=2500)
        assert audio_config.max_chunk_size == 2500
        assert audio_config.content_type == "audio"
        assert audio_config.strategy == ChunkingStrategy.TOPIC_BASED
        assert audio_config.respect_paragraph_boundaries is False
        
        # Video config
        video_config = ChunkingConfig.for_video_transcripts()
        assert video_config.content_type == "video"
        assert video_config.strategy == ChunkingStrategy.TOPIC_BASED
        
        # Web config
        web_config = ChunkingConfig.for_web_content()
        assert web_config.content_type == "web"
        assert web_config.strategy == ChunkingStrategy.HYBRID
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=3000,
            min_chunk_size=300
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["strategy"] == "hybrid"
        assert config_dict["max_chunk_size"] == 3000
        assert config_dict["min_chunk_size"] == 300
        
        # Test from_dict
        restored_config = ChunkingConfig.from_dict(config_dict)
        assert restored_config.strategy == ChunkingStrategy.HYBRID
        assert restored_config.max_chunk_size == 3000
        assert restored_config.min_chunk_size == 300


class TestSemanticChunkingAgent:
    """Test the semantic chunking agent."""
    
    def test_agent_creation(self):
        """Test creating semantic chunking agent."""
        agent = SemanticChunkingAgent(min_confidence=0.7)
        
        assert agent.min_confidence == 0.7
        assert "semantic chunking agent" in agent.get_system_prompt().lower()
    
    def test_agent_system_prompt(self):
        """Test the system prompt contains required elements."""
        agent = SemanticChunkingAgent()
        prompt = agent.get_system_prompt()
        
        # Check for key elements
        assert "topic boundaries" in prompt.lower()
        assert "confidence" in prompt.lower()
        assert "position" in prompt.lower()
        assert "chunk" in prompt.lower()
    
    def test_size_based_chunking(self):
        """Test size-based chunking fallback."""
        agent = SemanticChunkingAgent()
        
        # Test small text (no chunking needed)
        small_text = "This is a small text that doesn't need chunking."
        chunks = agent._size_based_chunking(small_text, 1000)
        assert len(chunks) == 1
        assert chunks[0] == small_text
        
        # Test large text (chunking needed)
        large_text = "This is a sentence. " * 100  # Create a large text
        chunks = agent._size_based_chunking(large_text, 200)
        assert len(chunks) > 1
        
        # Check that chunks don't exceed max size (with some tolerance for word boundaries)
        for chunk in chunks:
            assert len(chunk) <= 250  # Allow some tolerance for word boundaries


class TestSemanticChunker:
    """Test the semantic chunker."""
    
    def test_chunker_creation(self):
        """Test creating semantic chunker."""
        config = ChunkingConfig(max_chunk_size=2000, strategy=ChunkingStrategy.SIZE_BASED)
        chunker = SemanticChunker(config)
        
        assert chunker.config.max_chunk_size == 2000
        assert chunker.config.strategy == ChunkingStrategy.SIZE_BASED
    
    def test_chunker_with_defaults(self):
        """Test creating chunker with default configuration."""
        chunker = SemanticChunker()
        
        assert chunker.config is not None
        assert isinstance(chunker.config, ChunkingConfig)
        assert chunker.config.strategy == ChunkingStrategy.SEMANTIC
    
    @pytest.mark.asyncio
    async def test_size_based_chunking(self):
        """Test size-based chunking."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SIZE_BASED,
            max_chunk_size=200,
            min_chunk_size=50,
            overlap_size=20,
            use_ai_analysis=False
        )
        chunker = SemanticChunker(config)
        
        text = "This is a test sentence. " * 20  # Create text that needs chunking
        chunks = await chunker.chunk_text(text)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 250  # Allow some tolerance for word boundaries
    
    @pytest.mark.asyncio
    async def test_sentence_based_chunking(self):
        """Test sentence-based chunking."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE_BASED,
            max_chunk_size=200,
            min_chunk_size=50,
            overlap_size=20,
            use_ai_analysis=False
        )
        chunker = SemanticChunker(config)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = await chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        # Each chunk should end with sentence punctuation (except possibly the last one)
        for chunk in chunks[:-1]:
            assert chunk.rstrip().endswith(('.', '!', '?'))
    
    @pytest.mark.asyncio
    async def test_paragraph_based_chunking(self):
        """Test paragraph-based chunking."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH_BASED,
            max_chunk_size=300,
            min_chunk_size=50,
            overlap_size=20,
            use_ai_analysis=False
        )
        chunker = SemanticChunker(config)
        
        text = """First paragraph with some content.
This is still the first paragraph.

Second paragraph with different content.
This is still the second paragraph.

Third paragraph with more content."""
        
        chunks = await chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        # Should respect paragraph boundaries
        for chunk in chunks:
            assert len(chunk.strip()) > 0
    
    @pytest.mark.asyncio
    async def test_overlap_application(self):
        """Test overlap between chunks."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SIZE_BASED,
            max_chunk_size=200,
            min_chunk_size=50,
            overlap_size=20,
            use_ai_analysis=False
        )
        chunker = SemanticChunker(config)
        
        text = "This is a test sentence. " * 20
        chunks = await chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that chunks have overlap
            for i in range(1, len(chunks)):
                # The second chunk should start with some content from the first chunk
                assert len(chunks[i]) > len(chunks[i].lstrip())  # Has leading content
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        chunker = SemanticChunker()
        
        # Empty text
        chunks = await chunker.chunk_text("")
        assert chunks == []
        
        # Whitespace-only text
        chunks = await chunker.chunk_text("   \n\t   ")
        assert chunks == []
    
    @pytest.mark.asyncio
    async def test_config_override_with_kwargs(self):
        """Test overriding configuration with kwargs."""
        config = ChunkingConfig(max_chunk_size=2000, strategy=ChunkingStrategy.SIZE_BASED)
        chunker = SemanticChunker(config)
        
        text = "This is a test. " * 50
        
        # Override max_chunk_size via kwargs
        chunks = await chunker.chunk_text(text, max_chunk_size=100, strategy=ChunkingStrategy.SIZE_BASED)

        # Should use the overridden max_chunk_size
        # Note: The chunking algorithm may create larger chunks due to word boundary preservation
        for chunk in chunks:
            assert len(chunk) <= 200  # Allow tolerance for word boundaries and sentence preservation


class TestChunkerFactory:
    """Test the chunker factory."""
    
    def test_factory_create_chunker(self):
        """Test creating chunker with factory."""
        chunker = ChunkerFactory.create_chunker(content_type="document")
        
        assert isinstance(chunker, SemanticChunker)
        assert chunker.config.content_type == "document"
        assert chunker.config.preserve_code_blocks is True
    
    def test_factory_content_type_chunkers(self):
        """Test creating content-type specific chunkers."""
        # Document chunker
        doc_chunker = ChunkerFactory.create_document_chunker(max_chunk_size=5000)
        assert doc_chunker.config.max_chunk_size == 5000
        assert doc_chunker.config.content_type == "document"
        
        # Audio chunker
        audio_chunker = ChunkerFactory.create_audio_chunker(max_chunk_size=2500)
        assert audio_chunker.config.max_chunk_size == 2500
        assert audio_chunker.config.content_type == "audio"
        
        # Video chunker
        video_chunker = ChunkerFactory.create_video_chunker()
        assert video_chunker.config.content_type == "video"
        
        # Web chunker
        web_chunker = ChunkerFactory.create_web_chunker()
        assert web_chunker.config.content_type == "web"
    
    def test_convenience_function(self):
        """Test convenience function for creating chunkers."""
        # Create with content type
        chunker = create_chunker(content_type="audio", max_chunk_size=2000)
        assert chunker.config.content_type == "audio"
        assert chunker.config.max_chunk_size == 2000
        
        # Create with strategy
        chunker = create_chunker(strategy=ChunkingStrategy.SIZE_BASED, max_chunk_size=1500)
        assert chunker.config.strategy == ChunkingStrategy.SIZE_BASED
        assert chunker.config.max_chunk_size == 1500


class TestIntegration:
    """Integration tests for the complete chunking system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_chunking(self):
        """Test end-to-end chunking workflow."""
        # Create a chunker for documents
        chunker = create_chunker(
            content_type="document",
            strategy=ChunkingStrategy.SIZE_BASED,
            max_chunk_size=500,
            min_chunk_size=100,
            overlap_size=50
        )
        
        # Sample document text
        text = """
        This is the first paragraph of a document. It contains some important information
        about the topic we're discussing. The content is meaningful and should be preserved
        in the chunking process.
        
        This is the second paragraph. It continues the discussion from the first paragraph
        and adds more details. The chunking system should handle this appropriately and
        create meaningful chunks that preserve context.
        
        The third paragraph concludes our document. It summarizes the key points and
        provides a conclusion to the discussion. This should also be handled well by
        the chunking system.
        """
        
        chunks = await chunker.chunk_text(text)
        
        # Verify chunking results
        assert len(chunks) > 0
        assert all(len(chunk.strip()) > 0 for chunk in chunks)
        assert all(len(chunk) <= 600 for chunk in chunks)  # Allow some tolerance
        
        # Verify that all original content is preserved
        combined_text = " ".join(chunks)
        original_words = set(text.split())
        combined_words = set(combined_text.split())
        
        # Most words should be preserved (allowing for some overlap)
        preserved_ratio = len(original_words.intersection(combined_words)) / len(original_words)
        assert preserved_ratio > 0.8  # At least 80% of words preserved


if __name__ == "__main__":
    pytest.main([__file__])
