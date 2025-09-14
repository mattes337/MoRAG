"""Tests for the AudioService class."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from morag_audio.service import AudioService, AudioServiceError
from morag_audio.processor import AudioProcessor, AudioConfig, AudioProcessingResult, AudioSegment
from morag_audio.converters import AudioConverter, AudioConversionResult


@pytest.fixture
def audio_config():
    """Return a basic audio config for testing."""
    return AudioConfig(
        model_size="tiny",
        enable_diarization=False,
        enable_topic_segmentation=False,
        device="cpu"
    )


@pytest.fixture
def mock_processor():
    """Return a mock audio processor."""
    mock = AsyncMock(spec=AudioProcessor)
    
    # Set up mock process method
    segments = [
        AudioSegment(start=0.0, end=5.0, text="This is test segment 1."),
        AudioSegment(start=5.0, end=10.0, text="This is test segment 2."),
        AudioSegment(start=10.0, end=15.0, text="This is test segment 3.")
    ]
    
    transcript = "This is test segment 1. This is test segment 2. This is test segment 3."
    
    metadata = {
        "duration": 15.0,
        "channels": 2,
        "sample_rate": 44100,
        "word_count": 15,
        "segment_count": 3
    }
    
    mock.process.return_value = AudioProcessingResult(
        transcript=transcript,
        segments=segments,
        metadata=metadata,
        file_path="test.mp3",
        processing_time=1.5,
        success=True
    )
    
    return mock


@pytest.fixture
def mock_converter():
    """Return a mock audio converter."""
    mock = AsyncMock(spec=AudioConverter)
    
    # Set up mock convert_to_markdown method
    markdown_content = """# Audio Transcription: test.mp3

## Metadata
- **Duration**: 00:00:15
- **Word Count**: 15
- **Segment Count**: 3

## Full Transcript
This is test segment 1. This is test segment 2. This is test segment 3.

## Detailed Transcript
[00:00:00] This is test segment 1.
[00:00:05] This is test segment 2.
[00:00:10] This is test segment 3.
"""
    
    mock.convert_to_markdown.return_value = AudioConversionResult(
        content=markdown_content,
        metadata={},
        processing_time=0.5,
        success=True
    )
    
    return mock


@pytest.mark.asyncio
async def test_service_initialization(audio_config):
    """Test that the service initializes correctly."""
    with patch("morag_audio.service.AudioProcessor", return_value=MagicMock()), \
         patch("morag_audio.service.AudioConverter", return_value=MagicMock()):
        
        service = AudioService(config=audio_config)
        assert service.config == audio_config
        assert service.processor is not None
        assert service.converter is not None
        assert service.output_dir is None


@pytest.mark.asyncio
async def test_service_with_output_dir(audio_config):
    """Test service initialization with output directory."""
    with patch("morag_audio.service.AudioProcessor", return_value=MagicMock()), \
         patch("morag_audio.service.AudioConverter", return_value=MagicMock()), \
         patch("morag_audio.service.ensure_directory_exists") as mock_ensure_dir:
        
        output_dir = "/tmp/audio_output"
        service = AudioService(config=audio_config, output_dir=output_dir)
        
        assert service.output_dir == Path(output_dir)
        mock_ensure_dir.assert_called_once_with(Path(output_dir))


@pytest.mark.asyncio
async def test_process_file_not_found(audio_config):
    """Test that service raises an error for non-existent files."""
    with patch("morag_audio.service.AudioProcessor", return_value=MagicMock()), \
         patch("morag_audio.service.AudioConverter", return_value=MagicMock()), \
         patch("pathlib.Path.exists", return_value=False):
        
        service = AudioService(config=audio_config)
        
        with pytest.raises(AudioServiceError, match="File not found"):
            await service.process_file("nonexistent_file.mp3")


@pytest.mark.asyncio
async def test_process_file_success(mock_processor, mock_converter):
    """Test successful file processing without saving output."""
    with patch("morag_audio.service.AudioProcessor", return_value=mock_processor), \
         patch("morag_audio.service.AudioConverter", return_value=mock_converter), \
         patch("pathlib.Path.exists", return_value=True):
        
        service = AudioService()
        service.processor = mock_processor
        service.converter = mock_converter
        
        result = await service.process_file("test.mp3", save_output=False)
        
        assert result["success"] is True
        assert "processing_time" in result
        assert "metadata" in result
        assert "output_files" in result
        assert len(result["output_files"]) == 0  # No files saved
        assert "content" in result  # Content included when not saving


@pytest.mark.asyncio
async def test_process_file_with_output(mock_processor, mock_converter, tmp_path):
    """Test file processing with output saving."""
    with patch("morag_audio.service.AudioProcessor", return_value=mock_processor), \
         patch("morag_audio.service.AudioConverter", return_value=mock_converter), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open", MagicMock()), \
         patch("json.dump", MagicMock()):
        
        service = AudioService(output_dir=tmp_path)
        service.processor = mock_processor
        service.converter = mock_converter
        
        # Mock _save_output_files to return file paths
        mock_output_files = {
            "transcript": str(tmp_path / "test_transcript.txt"),
            "markdown": str(tmp_path / "test.md"),
            "segments": str(tmp_path / "test_segments.json"),
            "metadata": str(tmp_path / "test_metadata.json")
        }
        
        with patch.object(service, "_save_output_files", new_callable=AsyncMock) as mock_save:
            mock_save.return_value = mock_output_files
            
            result = await service.process_file("test.mp3", save_output=True)
            
            assert result["success"] is True
            assert "output_files" in result
            assert result["output_files"] == mock_output_files
            assert "content" not in result  # Content not included when saving


@pytest.mark.asyncio
async def test_process_file_with_embedding_service(mock_processor, mock_converter):
    """Test file processing with embedding service."""
    mock_embedding_service = AsyncMock()
    mock_embedding_service.embed_text.return_value = [0.1, 0.2, 0.3]  # Mock embedding vector
    mock_embedding_service.embed_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    
    with patch("morag_audio.service.AudioProcessor", return_value=mock_processor), \
         patch("morag_audio.service.AudioConverter", return_value=mock_converter), \
         patch("pathlib.Path.exists", return_value=True):
        
        service = AudioService(embedding_service=mock_embedding_service)
        service.processor = mock_processor
        service.converter = mock_converter
        
        result = await service.process_file("test.mp3", save_output=False)
        
        assert result["success"] is True
        assert "transcript_embedding" in result["metadata"]
        mock_embedding_service.embed_text.assert_called_once()
        mock_embedding_service.embed_batch.assert_called_once()


@pytest.mark.asyncio
async def test_process_file_error_handling(audio_config):
    """Test error handling during file processing."""
    mock_processor = AsyncMock(spec=AudioProcessor)
    mock_processor.process.side_effect = Exception("Processing error")
    
    with patch("morag_audio.service.AudioProcessor", return_value=mock_processor), \
         patch("morag_audio.service.AudioConverter", return_value=MagicMock()), \
         patch("pathlib.Path.exists", return_value=True):
        
        service = AudioService(config=audio_config)
        
        result = await service.process_file("test.mp3")
        
        assert result["success"] is False
        assert "error" in result
        assert "Processing error" in result["error"]