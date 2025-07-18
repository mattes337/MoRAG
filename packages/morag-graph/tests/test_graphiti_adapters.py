"""Tests for Graphiti adapter layer."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from morag_core.models import Document, DocumentChunk, DocumentMetadata, DocumentType
from morag_graph.models import Entity, Relation
from morag_graph.graphiti.adapters import (
    BaseAdapter, BatchAdapter, AdapterRegistry, ConversionResult, ConversionDirection,
    AdapterError, ConversionError, ValidationError, adapter_registry,
    DocumentAdapter, DocumentChunkAdapter, EntityAdapter, RelationAdapter,
    ADAPTERS_AVAILABLE
)


class TestConversionResult:
    """Test ConversionResult dataclass."""
    
    def test_init_success(self):
        """Test successful result initialization."""
        result = ConversionResult(success=True, data={"test": "data"})
        
        assert result.success is True
        assert result.data == {"test": "data"}
        assert result.error is None
        assert result.warnings == []
        assert result.metadata == {}
    
    def test_init_failure(self):
        """Test failure result initialization."""
        result = ConversionResult(
            success=False,
            error="Test error",
            warnings=["Warning 1", "Warning 2"]
        )
        
        assert result.success is False
        assert result.error == "Test error"
        assert result.warnings == ["Warning 1", "Warning 2"]
        assert result.metadata == {}


class TestAdapterRegistry:
    """Test AdapterRegistry functionality."""
    
    def test_init(self):
        """Test registry initialization."""
        registry = AdapterRegistry()
        
        assert len(registry._adapters) == 0
        assert len(registry._batch_adapters) == 0
    
    def test_register_adapter(self):
        """Test adapter registration."""
        registry = AdapterRegistry()
        mock_adapter = MagicMock(spec=BaseAdapter)
        mock_adapter.get_stats.return_value = {"total_conversions": 0}
        
        registry.register_adapter("test_adapter", mock_adapter)
        
        assert "test_adapter" in registry._adapters
        assert registry.get_adapter("test_adapter") == mock_adapter
    
    def test_register_batch_adapter(self):
        """Test batch adapter registration."""
        registry = AdapterRegistry()
        mock_single_adapter = MagicMock(spec=BaseAdapter)
        mock_single_adapter.get_stats.return_value = {"total_conversions": 0}
        mock_batch_adapter = BatchAdapter(mock_single_adapter)
        
        registry.register_batch_adapter("test_batch", mock_batch_adapter)
        
        assert "test_batch" in registry._batch_adapters
        assert registry.get_batch_adapter("test_batch") == mock_batch_adapter
    
    def test_list_adapters(self):
        """Test listing all adapters."""
        registry = AdapterRegistry()
        
        # Register single adapter
        mock_adapter = MagicMock(spec=BaseAdapter)
        mock_adapter.get_stats.return_value = {"total_conversions": 5}
        registry.register_adapter("single", mock_adapter)
        
        # Register batch adapter
        mock_single_adapter = MagicMock(spec=BaseAdapter)
        mock_single_adapter.get_stats.return_value = {"total_conversions": 10}
        mock_batch_adapter = BatchAdapter(mock_single_adapter)
        registry.register_batch_adapter("batch", mock_batch_adapter)
        
        result = registry.list_adapters()
        
        assert "single_adapters" in result
        assert "batch_adapters" in result
        assert result["single_adapters"]["single"]["total_conversions"] == 5
        assert result["batch_adapters"]["batch"]["total_conversions"] == 10


class TestDocumentAdapter:
    """Test DocumentAdapter functionality."""
    
    def create_test_document(self):
        """Create a test document."""
        metadata = DocumentMetadata(
            title="Test Document",
            author="Test Author",
            source_path="/test/document.pdf",
            mime_type="application/pdf",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            updated_at=datetime(2024, 1, 2, 12, 0, 0)
        )
        
        chunks = [
            DocumentChunk(
                id="chunk_1",
                content="This is the first chunk of content.",
                document_id="test-doc-123",
                chunk_index=0,
                start_char=0,
                end_char=35
            ),
            DocumentChunk(
                id="chunk_2", 
                content="This is the second chunk of content.",
                document_id="test-doc-123",
                chunk_index=1,
                start_char=36,
                end_char=72
            )
        ]
        
        return Document(
            id="test-doc-123",
            content="This is the first chunk of content. This is the second chunk of content.",
            metadata=metadata,
            chunks=chunks,
            document_type=DocumentType.PDF
        )
    
    def test_init(self):
        """Test adapter initialization."""
        adapter = DocumentAdapter()
        
        assert adapter.strict_validation is True
        assert adapter.include_chunks is True
        assert adapter.conversion_stats["total_conversions"] == 0
    
    def test_init_with_options(self):
        """Test adapter initialization with options."""
        adapter = DocumentAdapter(strict_validation=False, include_chunks=False)
        
        assert adapter.strict_validation is False
        assert adapter.include_chunks is False
    
    def test_to_graphiti_success(self):
        """Test successful document to Graphiti conversion."""
        adapter = DocumentAdapter()
        document = self.create_test_document()
        
        result = adapter.to_graphiti(document)
        
        assert result.success is True
        assert result.data is not None
        assert "name" in result.data
        assert "content" in result.data
        assert "source_description" in result.data
        assert "metadata" in result.data
        assert result.data["metadata"]["morag_document_id"] == "test-doc-123"
    
    def test_to_graphiti_without_chunks(self):
        """Test document conversion without chunks."""
        adapter = DocumentAdapter(include_chunks=False)
        document = self.create_test_document()
        
        result = adapter.to_graphiti(document)
        
        assert result.success is True
        assert "chunk_episodes" not in result.data
    
    def test_to_graphiti_validation_error(self):
        """Test conversion with validation error."""
        adapter = DocumentAdapter(strict_validation=True)
        
        result = adapter.to_graphiti(None)
        
        assert result.success is False
        assert "Validation failed" in result.error
    
    def test_validate_input_morag_to_graphiti(self):
        """Test input validation for MoRAG to Graphiti conversion."""
        adapter = DocumentAdapter()
        document = self.create_test_document()
        
        # Valid document
        errors = adapter.validate_input(document, ConversionDirection.MORAG_TO_GRAPHITI)
        assert len(errors) == 0
        
        # Invalid document (None)
        errors = adapter.validate_input(None, ConversionDirection.MORAG_TO_GRAPHITI)
        assert len(errors) > 0
        assert "Input data cannot be None" in errors
        
        # Invalid document (wrong type)
        errors = adapter.validate_input("not a document", ConversionDirection.MORAG_TO_GRAPHITI)
        assert len(errors) > 0
        assert "Input must be a Document instance" in errors
    
    def test_validate_input_graphiti_to_morag(self):
        """Test input validation for Graphiti to MoRAG conversion."""
        adapter = DocumentAdapter()
        
        # Valid episode data
        episode_data = {"content": "test content", "name": "test episode"}
        errors = adapter.validate_input(episode_data, ConversionDirection.GRAPHITI_TO_MORAG)
        assert len(errors) == 0
        
        # Invalid episode data (missing content)
        episode_data = {"name": "test episode"}
        errors = adapter.validate_input(episode_data, ConversionDirection.GRAPHITI_TO_MORAG)
        assert len(errors) > 0
        assert "Episode data must contain 'content' field" in errors
    
    def test_generate_episode_name(self):
        """Test episode name generation."""
        adapter = DocumentAdapter()
        document = self.create_test_document()
        
        episode_name = adapter._generate_episode_name(document)
        
        assert "Test Document" in episode_name
        assert len(episode_name) > len("Test Document")  # Should include timestamp
    
    def test_generate_episode_content(self):
        """Test episode content generation."""
        adapter = DocumentAdapter()
        document = self.create_test_document()
        
        content = adapter._generate_episode_content(document)
        
        assert "Document Title: Test Document" in content
        assert "Source File: document.pdf" in content
        assert "Author: Test Author" in content
        assert "Content Summary:" in content
        assert "Document contains 2 chunks:" in content
    
    def test_get_stats(self):
        """Test conversion statistics."""
        adapter = DocumentAdapter()
        document = self.create_test_document()
        
        # Initial stats
        stats = adapter.get_stats()
        assert stats["total_conversions"] == 0
        assert stats["success_rate"] == 0.0
        
        # After successful conversion
        adapter.to_graphiti(document)
        stats = adapter.get_stats()
        assert stats["total_conversions"] == 1
        assert stats["successful_conversions"] == 1
        assert stats["success_rate"] == 1.0


class TestDocumentChunkAdapter:
    """Test DocumentChunkAdapter functionality."""
    
    def create_test_chunk(self):
        """Create a test document chunk."""
        return DocumentChunk(
            id="test-chunk-456",
            content="This is a test chunk content.",
            document_id="test-doc-123",
            chunk_index=0,
            start_char=0,
            end_char=29
        )
    
    def test_to_graphiti_success(self):
        """Test successful chunk to Graphiti conversion."""
        adapter = DocumentChunkAdapter()
        chunk = self.create_test_chunk()
        
        result = adapter.to_graphiti(chunk)
        
        assert result.success is True
        assert result.data is not None
        assert "name" in result.data
        assert "content" in result.data
        assert result.data["content"] == "This is a test chunk content."
        assert result.data["metadata"]["morag_chunk_id"] == "test-chunk-456"


class TestBatchAdapter:
    """Test BatchAdapter functionality."""
    
    def test_init(self):
        """Test batch adapter initialization."""
        mock_adapter = MagicMock(spec=BaseAdapter)
        batch_adapter = BatchAdapter(mock_adapter, batch_size=50)
        
        assert batch_adapter.single_adapter == mock_adapter
        assert batch_adapter.batch_size == 50
    
    def test_batch_to_graphiti(self):
        """Test batch conversion to Graphiti."""
        mock_adapter = MagicMock(spec=BaseAdapter)
        mock_adapter.to_graphiti.return_value = ConversionResult(success=True, data={"test": "data"})
        
        batch_adapter = BatchAdapter(mock_adapter, batch_size=2)
        test_items = ["item1", "item2", "item3"]
        
        results = batch_adapter.batch_to_graphiti(test_items)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert mock_adapter.to_graphiti.call_count == 3
    
    def test_batch_to_graphiti_with_error(self):
        """Test batch conversion with error handling."""
        mock_adapter = MagicMock(spec=BaseAdapter)
        mock_adapter.to_graphiti.side_effect = [
            ConversionResult(success=True, data={"test": "data"}),
            Exception("Test error"),
            ConversionResult(success=True, data={"test": "data"})
        ]
        
        batch_adapter = BatchAdapter(mock_adapter, batch_size=2)
        test_items = ["item1", "item2", "item3"]
        
        results = batch_adapter.batch_to_graphiti(test_items)
        
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert "Batch conversion error" in results[1].error
        assert results[2].success is True


class TestGlobalAdapterRegistry:
    """Test global adapter registry."""
    
    def test_global_registry_exists(self):
        """Test that global registry exists."""
        assert adapter_registry is not None
        assert isinstance(adapter_registry, AdapterRegistry)
    
    def test_adapters_available_flag(self):
        """Test ADAPTERS_AVAILABLE flag."""
        # This should be True since we're importing successfully
        assert ADAPTERS_AVAILABLE is True
