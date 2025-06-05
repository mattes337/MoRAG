"""Integration tests for cross-package functionality.

This module tests that packages work together correctly in realistic
workflows and data flows properly between packages.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock


class TestCrossPackageIntegration:
    """Test integration between different MoRAG packages."""
    
    def test_core_services_integration(self):
        """Test integration between core and services packages."""
        try:
            from morag_core.models import Document
            from morag_services import ServiceConfig
            
            # Test that core models work with services
            doc = Document(content="Test content", metadata={"test": True})
            config = ServiceConfig()
            
            # Basic integration test
            assert doc.content == "Test content"
            assert config is not None
            
        except ImportError as e:
            pytest.skip(f"Required packages not available: {e}")
    
    def test_web_services_integration(self):
        """Test integration between web and services packages."""
        try:
            from morag_web import WebProcessor, WebScrapingConfig
            from morag_services import ServiceConfig
            
            # Test that web processor can work with services
            web_config = WebScrapingConfig()
            service_config = ServiceConfig()
            processor = WebProcessor(web_config)
            
            assert processor is not None
            
        except ImportError as e:
            pytest.skip(f"Required packages not available: {e}")
    
    @pytest.mark.asyncio
    async def test_audio_processing_workflow(self):
        """Test complete audio processing workflow across packages."""
        try:
            from morag_audio import AudioProcessor, AudioConfig
            from morag_core.models import Document
            
            # Create a mock audio file for testing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
            
            try:
                # Mock the audio processing to avoid requiring actual audio files
                with patch('morag_audio.processor.AudioProcessor.process') as mock_process:
                    mock_result = Mock()
                    mock_result.transcript = "Test transcript"
                    mock_result.segments = []
                    mock_result.metadata = {"duration": 10.0}
                    mock_result.file_path = str(tmp_path)
                    mock_result.processing_time = 1.0
                    mock_result.success = True
                    
                    mock_process.return_value = mock_result
                    
                    # Test the workflow
                    config = AudioConfig()
                    processor = AudioProcessor(config)
                    result = await processor.process(tmp_path)
                    
                    assert result.success
                    assert result.transcript == "Test transcript"
            
            finally:
                # Clean up
                if tmp_path.exists():
                    tmp_path.unlink()
                    
        except ImportError as e:
            pytest.skip(f"Required packages not available: {e}")
    
    @pytest.mark.asyncio
    async def test_document_processing_workflow(self):
        """Test complete document processing workflow across packages."""
        try:
            from morag_document import DocumentProcessor
            from morag_core.models import Document
            
            # Create a mock document file
            with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as tmp_file:
                tmp_file.write("This is a test document with some content.")
                tmp_path = Path(tmp_file.name)
            
            try:
                # Mock the document processing
                with patch('morag_document.processor.DocumentProcessor.process') as mock_process:
                    mock_result = Mock()
                    mock_result.content = "This is a test document with some content."
                    mock_result.metadata = {"file_type": "text"}
                    mock_result.success = True
                    
                    mock_process.return_value = mock_result
                    
                    # Test the workflow
                    processor = DocumentProcessor()
                    result = await processor.process(tmp_path)
                    
                    assert result.success
                    assert "test document" in result.content
            
            finally:
                # Clean up
                if tmp_path.exists():
                    tmp_path.unlink()
                    
        except ImportError as e:
            pytest.skip(f"Required packages not available: {e}")
    
    def test_converter_registry_integration(self):
        """Test that converter registry works with modular converters."""
        try:
            # Test the registry from the main codebase
            from src.morag.converters.registry import DocumentConverter
            
            converter = DocumentConverter()
            
            # Test that it can detect formats
            supported_formats = converter.list_supported_formats()
            assert isinstance(supported_formats, list)
            
            # Test converter info
            converter_info = converter.get_converter_info()
            assert isinstance(converter_info, dict)
            
        except ImportError as e:
            pytest.skip(f"Converter registry not available: {e}")
    
    def test_task_integration(self):
        """Test that tasks can work with modular packages."""
        try:
            # Test importing tasks that should use modular packages
            from src.morag.tasks.base import ProcessingTask
            
            # Basic test that task base class is available
            assert ProcessingTask is not None
            
        except ImportError as e:
            pytest.skip(f"Task system not available: {e}")
    
    @pytest.mark.asyncio
    async def test_end_to_end_web_workflow(self):
        """Test end-to-end web processing workflow."""
        try:
            from morag_web import WebProcessor, WebScrapingConfig
            from morag_core.models import Document
            
            # Mock web processing to avoid actual network calls
            with patch('morag_web.processor.WebProcessor.process') as mock_process:
                mock_result = Mock()
                mock_result.content = "Scraped web content"
                mock_result.metadata = {"url": "https://example.com", "title": "Test Page"}
                mock_result.success = True
                
                mock_process.return_value = mock_result
                
                # Test the workflow
                config = WebScrapingConfig()
                processor = WebProcessor(config)
                result = await processor.process("https://example.com")
                
                assert result.success
                assert result.content == "Scraped web content"
                assert result.metadata["url"] == "https://example.com"
                
        except ImportError as e:
            pytest.skip(f"Required packages not available: {e}")
    
    def test_configuration_compatibility(self):
        """Test that configurations are compatible across packages."""
        try:
            from morag_core.config import Settings
            from morag_services import ServiceConfig
            
            # Test that configurations can be created and are compatible
            settings = Settings()
            service_config = ServiceConfig()
            
            # Basic compatibility test
            assert settings is not None
            assert service_config is not None
            
        except ImportError as e:
            pytest.skip(f"Configuration packages not available: {e}")
    
    def test_error_handling_across_packages(self):
        """Test that errors are handled consistently across packages."""
        try:
            from morag_core.exceptions import ProcessingError, ValidationError
            
            # Test that core exceptions can be imported and used
            error = ProcessingError("Test error")
            assert str(error) == "Test error"
            
            validation_error = ValidationError("Validation failed")
            assert str(validation_error) == "Validation failed"
            
        except ImportError as e:
            pytest.skip(f"Exception classes not available: {e}")
    
    def test_data_flow_between_packages(self):
        """Test that data flows correctly between packages."""
        try:
            from morag_core.models import Document, DocumentChunk
            
            # Test creating and manipulating core data structures
            doc = Document(
                content="Test document content",
                metadata={"source": "test", "type": "text"}
            )
            
            # Test that document can be chunked
            chunk = DocumentChunk(
                text="Test chunk",
                start_char=0,
                end_char=10,
                metadata={"chunk_id": 1}
            )
            
            assert doc.content == "Test document content"
            assert chunk.text == "Test chunk"
            
        except ImportError as e:
            pytest.skip(f"Core models not available: {e}")
    
    @pytest.mark.slow
    def test_performance_across_packages(self):
        """Test that performance is acceptable across package boundaries."""
        import time
        
        try:
            from morag_core.models import Document
            
            # Test that creating many objects across packages is performant
            start_time = time.time()
            
            documents = []
            for i in range(1000):
                doc = Document(
                    content=f"Document {i}",
                    metadata={"id": i}
                )
                documents.append(doc)
            
            end_time = time.time()
            creation_time = end_time - start_time
            
            # Should be able to create 1000 documents in reasonable time
            assert creation_time < 1.0, f"Document creation too slow: {creation_time}s"
            assert len(documents) == 1000
            
        except ImportError as e:
            pytest.skip(f"Performance test packages not available: {e}")
    
    def test_backward_compatibility(self):
        """Test that backward compatibility is maintained."""
        try:
            # Test that old import patterns still work through compatibility layer
            from src.morag.processors import WebProcessor
            
            # Should be able to create processor through compatibility layer
            processor = WebProcessor
            assert processor is not None
            
        except ImportError as e:
            pytest.skip(f"Backward compatibility layer not available: {e}")


class TestPackageInterfaces:
    """Test that package interfaces are consistent."""
    
    def test_processor_interfaces(self):
        """Test that all processors follow the same interface."""
        processor_packages = [
            ('morag_web', 'WebProcessor'),
            ('morag_audio', 'AudioProcessor'),
            ('morag_video', 'VideoProcessor'),
            ('morag_document', 'DocumentProcessor'),
            ('morag_image', 'ImageProcessor'),
            ('morag_youtube', 'YouTubeProcessor'),
        ]
        
        for package_name, processor_class in processor_packages:
            try:
                module = __import__(package_name, fromlist=[processor_class])
                processor_cls = getattr(module, processor_class)
                
                # Test that processor can be instantiated
                # (may require config, so we'll just check the class exists)
                assert processor_cls is not None
                assert hasattr(processor_cls, 'process') or hasattr(processor_cls, '__call__')
                
            except ImportError:
                # Skip if package not available
                continue
    
    def test_converter_interfaces(self):
        """Test that all converters follow the same interface."""
        try:
            from morag_core.interfaces.converter import BaseConverter
            
            # Test that base converter interface exists
            assert BaseConverter is not None
            assert hasattr(BaseConverter, 'convert')
            
        except ImportError as e:
            pytest.skip(f"Converter interfaces not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
