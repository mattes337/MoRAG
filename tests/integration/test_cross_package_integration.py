"""Integration tests for cross-package functionality.

This module tests that packages work together correctly in realistic
workflows and data flows properly between packages.

Testing Approach:
- Create real test files with known properties
- Mock only external services (APIs, heavy dependencies)
- Test actual processor logic and behavior verification
- Verify real metadata extraction, error handling, and file processing
- Avoid mocking the classes/methods being tested (anti-pattern)

This ensures tests actually verify behavior instead of being tautological.
"""

import pytest
import asyncio
import tempfile
import wave
import struct
import math
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock


@pytest.fixture
def mock_whisper_service():
    """Mock only the external Whisper service, not the processor logic.

    This demonstrates the correct approach: mock external dependencies,
    test actual processor behavior.
    """
    from morag_audio.models import AudioSegment

    def transcribe_func(file_path):
        """Mock transcription that returns realistic test data."""
        segments = [
            AudioSegment(start=0.0, end=2.0, text="Test audio content", confidence=0.95),
            AudioSegment(start=2.0, end=4.0, text="for integration testing", confidence=0.92)
        ]
        transcript = "Test audio content for integration testing"
        return segments, transcript

    return transcribe_func


def create_test_audio_file(file_path: Path, duration: float = 5.0, sample_rate: int = 16000, frequency: int = 440) -> Path:
    """Create a test audio file with known properties.

    Args:
        file_path: Path where to create the file
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        frequency: Tone frequency in Hz

    Returns:
        Path to the created file
    """
    frames = int(duration * sample_rate)

    with wave.open(str(file_path), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Generate a simple sine wave
        for i in range(frames):
            value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav_file.writeframes(struct.pack('<h', value))

    return file_path


def create_test_document_file(file_path: Path, content: str = "This is a test document with some content.") -> Path:
    """Create a test document file with known content.

    Args:
        file_path: Path where to create the file
        content: Content to write

    Returns:
        Path to the created file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return file_path


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
        """Test complete audio processing workflow with real files and mock external services."""
        try:
            from morag_audio import AudioProcessor, AudioConfig
            from morag_audio.models import AudioSegment

            # Create a real test audio file with known properties
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            # Create actual audio file
            create_test_audio_file(tmp_path, duration=3.0, sample_rate=16000, frequency=440)

            try:
                # Mock only the external Whisper transcription service, not the processor
                mock_segments = [
                    AudioSegment(start=0.0, end=1.5, text="Test audio", confidence=0.95),
                    AudioSegment(start=1.5, end=3.0, text="segment processing", confidence=0.92)
                ]

                with patch('morag_audio.processor.AudioProcessor._transcribe_with_faster_whisper') as mock_transcribe:
                    mock_transcribe.return_value = (mock_segments, "Test audio segment processing")

                    # Configure for local processing (not REST API)
                    config = AudioConfig(
                        use_rest_api=False,
                        enable_diarization=False,  # Disable to avoid additional mocking
                        enable_topic_segmentation=False
                    )
                    processor = AudioProcessor(config)

                    # Test the actual processor behavior
                    result = await processor.process(tmp_path)

                    # Verify processor's actual behavior, not mock behavior
                    assert result.success, f"Processing failed: {result.error_message}"
                    assert result.transcript == "Test audio segment processing"
                    assert len(result.segments) == 2
                    assert result.file_path == str(tmp_path)
                    assert result.processing_time > 0

                    # Verify metadata extraction actually worked
                    assert "file_name" in result.metadata
                    assert "file_size" in result.metadata
                    assert "mime_type" in result.metadata
                    assert result.metadata["file_name"] == tmp_path.name
                    assert result.metadata["file_size"] == tmp_path.stat().st_size
                    assert result.metadata["mime_type"] == "audio/wav"

                    # Verify transcription was called (external service was mocked)
                    mock_transcribe.assert_called_once()

            finally:
                # Clean up
                if tmp_path.exists():
                    tmp_path.unlink()

        except ImportError as e:
            pytest.skip(f"Required packages not available: {e}")

    @pytest.mark.asyncio
    async def test_document_processing_workflow(self):
        """Test complete document processing workflow with real files."""
        try:
            from morag_document import DocumentProcessor

            # Create a real test document file with known content
            test_content = "This is a test document with multiple sentences. It contains various information for processing."
            with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as tmp_file:
                tmp_file.write(test_content)
                tmp_path = Path(tmp_file.name)

            try:
                # Test the actual processor behavior (no mocking of the processor itself)
                processor = DocumentProcessor()
                result = await processor.process(tmp_path)

                # Verify processor's actual behavior
                assert result.success, f"Processing failed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
                assert test_content in result.content, "Original content should be preserved"

                # Verify document processing actually worked
                assert hasattr(result, 'metadata'), "Result should have metadata"
                if hasattr(result, 'metadata') and result.metadata:
                    # Check that file metadata was extracted
                    assert result.metadata.get("source_name") == tmp_path.name or \
                           result.metadata.get("file_name") == tmp_path.name, \
                           "File name should be captured in metadata"

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
        """Test end-to-end web processing workflow with mock network layer."""
        try:
            from morag_web import WebProcessor, WebScrapingConfig

            test_url = "https://example.com"
            test_html = """
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <h1>Test Content</h1>
                    <p>This is scraped web content for testing.</p>
                </body>
            </html>
            """

            # Mock only the network layer (playwright), not the processor itself
            with patch('morag_web.processor.WebProcessor._scrape_with_playwright') as mock_scrape:
                mock_scrape.return_value = {
                    'content': 'Test Content\nThis is scraped web content for testing.',
                    'metadata': {'url': test_url, 'title': 'Test Page', 'status_code': 200}
                }

                # Test the actual processor behavior
                config = WebScrapingConfig()
                processor = WebProcessor(config)
                result = await processor.process(test_url)

                # Verify processor's actual behavior
                assert result.success, f"Processing failed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
                assert "Test Content" in result.content
                assert "scraped web content" in result.content

                # Verify metadata processing worked
                if hasattr(result, 'metadata') and result.metadata:
                    assert result.metadata.get("url") == test_url
                    assert "title" in result.metadata or "Test Page" in str(result.metadata)

                # Verify external service was called
                mock_scrape.assert_called_once_with(test_url)

        except ImportError as e:
            pytest.skip(f"Required packages not available: {e}")

    @pytest.mark.asyncio
    async def test_metadata_extraction_behavior(self):
        """Test that metadata extraction actually works across different file types."""
        test_files = []

        try:
            # Create different test files
            audio_file = Path(tempfile.mktemp(suffix='.wav'))
            text_file = Path(tempfile.mktemp(suffix='.txt'))

            create_test_audio_file(audio_file, duration=2.0)
            create_test_document_file(text_file, "Test content for metadata extraction")
            test_files.extend([audio_file, text_file])

            # Test audio metadata extraction
            from morag_audio import AudioProcessor, AudioConfig

            # Mock only the transcription, test metadata extraction
            with patch('morag_audio.processor.AudioProcessor._transcribe_with_faster_whisper') as mock_transcribe:
                mock_transcribe.return_value = ([], "test")

                config = AudioConfig(enable_diarization=False, enable_topic_segmentation=False)
                processor = AudioProcessor(config)
                result = await processor.process(audio_file)

                # Verify actual metadata extraction behavior
                assert result.metadata["file_name"] == audio_file.name
                assert result.metadata["file_size"] > 0  # Should have calculated real file size
                assert result.metadata["mime_type"] == "audio/wav"
                assert "checksum" in result.metadata  # Should calculate actual checksum

                # Different files should have different checksums
                audio_checksum = result.metadata["checksum"]
                assert len(audio_checksum) == 64  # SHA256 hex string length

            # Test document metadata extraction
            from morag_document import DocumentProcessor
            doc_processor = DocumentProcessor()
            doc_result = await doc_processor.process(text_file)

            if hasattr(doc_result, 'metadata') and doc_result.metadata:
                # Should extract different metadata for different file types
                doc_metadata = doc_result.metadata
                assert doc_metadata.get("source_name") == text_file.name or \
                       doc_metadata.get("file_name") == text_file.name

        except ImportError as e:
            pytest.skip(f"Required packages not available: {e}")
        finally:
            # Clean up all test files
            for file_path in test_files:
                if file_path.exists():
                    file_path.unlink()

    @pytest.mark.asyncio
    async def test_error_handling_behavior(self):
        """Test that error handling actually works, not just mock returns."""
        try:
            from morag_audio import AudioProcessor, AudioConfig

            # Test with a non-existent file
            non_existent_file = Path("/tmp/definitely_not_a_real_file.wav")

            config = AudioConfig(enable_diarization=False, enable_topic_segmentation=False)
            processor = AudioProcessor(config)

            # Should handle file not found gracefully
            result = await processor.process(non_existent_file)

            # Verify actual error handling behavior
            assert not result.success
            assert result.error_message is not None
            assert "not found" in result.error_message.lower() or "no such file" in result.error_message.lower()
            assert result.transcript == ""
            assert len(result.segments) == 0

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
