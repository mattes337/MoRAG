"""Tests for metadata service functionality."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, patch

from morag_services import metadata_service
from morag_core.models import (
    FileMetadata, ContentMetadata, ProcessingMetadata, 
    UserMetadata, SystemMetadata, ComprehensiveMetadata
)

class TestMetadataService:
    """Test metadata service functionality."""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary test file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for metadata extraction")
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.mark.asyncio
    async def test_extract_file_metadata(self, temp_file):
        """Test file metadata extraction."""
        metadata = await metadata_service.extract_file_metadata(
            temp_file, 
            original_filename="test_document.txt"
        )
        
        assert isinstance(metadata, FileMetadata)
        assert metadata.original_filename == "test_document.txt"
        assert metadata.file_size > 0
        assert metadata.mime_type == "text/plain"
        assert metadata.file_extension == "txt"
        assert metadata.file_hash is not None
        assert len(metadata.file_hash) == 64  # SHA-256 hash length
        assert metadata.file_path == str(temp_file)
    
    @pytest.mark.asyncio
    async def test_extract_content_metadata(self, temp_file):
        """Test content metadata extraction."""
        extracted_data = {
            'title': 'Test Document',
            'author': 'Test Author',
            'word_count': 100,
            'language': 'en',
            'creation_date': '2024-01-01T00:00:00Z'
        }
        
        metadata = await metadata_service.extract_content_metadata(
            temp_file, 
            'document', 
            extracted_data
        )
        
        assert isinstance(metadata, ContentMetadata)
        assert metadata.title == 'Test Document'
        assert metadata.author == 'Test Author'
        assert metadata.word_count == 100
        assert metadata.language == 'en'
        assert metadata.creation_date is not None
    
    def test_create_processing_metadata(self):
        """Test processing metadata creation."""
        processing_steps = ['parsing', 'chunking', 'embedding']
        metadata = metadata_service.create_processing_metadata(
            processing_steps,
            chunk_count=5,
            embedding_model='text-embedding-004'
        )
        
        assert isinstance(metadata, ProcessingMetadata)
        assert metadata.processing_steps == processing_steps
        assert metadata.chunk_count == 5
        assert metadata.embedding_model == 'text-embedding-004'
        assert metadata.processor_version == '1.0.0'
    
    def test_create_user_metadata(self):
        """Test user metadata creation."""
        user_data = {
            'tags': ['important', 'research'],
            'categories': ['documents'],
            'notes': 'Test notes',
            'priority': 3
        }
        
        metadata = metadata_service.create_user_metadata(user_data)
        
        assert isinstance(metadata, UserMetadata)
        assert metadata.tags == ['important', 'research']
        assert metadata.categories == ['documents']
        assert metadata.notes == 'Test notes'
        assert metadata.priority == 3
    
    def test_create_system_metadata(self):
        """Test system metadata creation."""
        metadata = metadata_service.create_system_metadata(
            'test-ingestion-id',
            'test-collection',
            'test-task-id'
        )
        
        assert isinstance(metadata, SystemMetadata)
        assert metadata.ingestion_id == 'test-ingestion-id'
        assert metadata.collection_name == 'test-collection'
        assert metadata.task_id == 'test-task-id'
        assert metadata.storage_backend == 'qdrant'
        assert metadata.version == 1
    
    @pytest.mark.asyncio
    async def test_create_comprehensive_metadata(self, temp_file):
        """Test comprehensive metadata creation."""
        user_data = {'tags': ['test'], 'priority': 1}
        extracted_data = {'title': 'Test', 'word_count': 50}
        processing_steps = ['parsing', 'chunking']
        
        metadata = await metadata_service.create_comprehensive_metadata(
            file_path=temp_file,
            ingestion_id='test-id',
            collection_name='test-collection',
            content_type='document',
            user_data=user_data,
            extracted_data=extracted_data,
            processing_steps=processing_steps,
            task_id='test-task',
            original_filename='test.txt'
        )
        
        assert isinstance(metadata, ComprehensiveMetadata)
        assert metadata.file is not None
        assert metadata.content is not None
        assert metadata.processing is not None
        assert metadata.user is not None
        assert metadata.system is not None
        
        # Test to_dict method
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert 'file' in metadata_dict
        assert 'system' in metadata_dict
        
        # Test searchable text
        searchable = metadata.get_searchable_text()
        assert 'test.txt' in searchable
        assert 'Test' in searchable
        assert 'test' in searchable
    
    def test_parse_date_formats(self):
        """Test date parsing from various formats."""
        # Test datetime object
        dt = datetime.now()
        parsed = metadata_service._parse_date(dt)
        assert parsed == dt
        
        # Test ISO string
        iso_string = "2024-01-01T12:00:00Z"
        parsed = metadata_service._parse_date(iso_string)
        assert parsed is not None
        assert parsed.year == 2024
        
        # Test invalid string
        parsed = metadata_service._parse_date("invalid-date")
        assert parsed is None
        
        # Test None
        parsed = metadata_service._parse_date(None)
        assert parsed is None

class TestMetadataModels:
    """Test metadata model functionality."""
    
    def test_comprehensive_metadata_searchable_text(self):
        """Test searchable text extraction from comprehensive metadata."""
        file_meta = FileMetadata(
            original_filename="test.pdf",
            file_size=1000,
            mime_type="application/pdf",
            file_extension="pdf"
        )
        
        content_meta = ContentMetadata(
            title="Research Paper",
            author="John Doe"
        )
        
        user_meta = UserMetadata(
            tags=["research", "important"],
            categories=["papers"],
            notes="Important research document"
        )
        
        system_meta = SystemMetadata(
            ingestion_id="test-id",
            collection_name="test-collection"
        )
        
        comprehensive = ComprehensiveMetadata(
            file=file_meta,
            content=content_meta,
            user=user_meta,
            system=system_meta
        )
        
        searchable = comprehensive.get_searchable_text()
        assert "test.pdf" in searchable
        assert "Research Paper" in searchable
        assert "John Doe" in searchable
        assert "research" in searchable
        assert "important" in searchable
        assert "papers" in searchable
        assert "Important research document" in searchable

if __name__ == "__main__":
    pytest.main([__file__])
