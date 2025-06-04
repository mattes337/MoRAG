"""Tests for the AudioProcessor class."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from morag_audio.processor import AudioProcessor, AudioConfig, AudioProcessingError, AudioSegment


@pytest.fixture
def audio_config():
    """Return a basic audio config for testing."""
    return AudioConfig(
        model_size="tiny",
        enable_diarization=False,
        enable_topic_segmentation=False,
        device="cpu",
        vad_filter=True
    )


@pytest.fixture
def mock_transcriber():
    """Return a mock transcriber for testing."""
    mock = MagicMock()
    
    # Mock transcribe method to return sample segments
    segments = []
    for i in range(3):
        segment = MagicMock()
        segment.start = i * 5.0
        segment.end = (i + 1) * 5.0
        segment.text = f"This is test segment {i+1}."
        segment.avg_logprob = 0.9
        segments.append(segment)
    
    mock.transcribe.return_value = (segments, {"language": "en"})
    return mock


@pytest.mark.asyncio
async def test_processor_initialization(audio_config):
    """Test that the processor initializes correctly."""
    with patch("morag_audio.processor.WhisperModel", return_value=MagicMock()):
        processor = AudioProcessor(audio_config)
        assert processor.config == audio_config
        assert processor.transcriber is not None


@pytest.mark.asyncio
async def test_processor_file_not_found(audio_config):
    """Test that processor raises an error for non-existent files."""
    with patch("morag_audio.processor.WhisperModel", return_value=MagicMock()):
        processor = AudioProcessor(audio_config)
        
        with pytest.raises(AudioProcessingError, match="File not found"):
            await processor.process("nonexistent_file.mp3")


@pytest.mark.asyncio
async def test_extract_metadata():
    """Test metadata extraction from audio file."""
    with patch("morag_audio.processor.WhisperModel", return_value=MagicMock()):
        processor = AudioProcessor()
        
        # Mock the audio file
        mock_file = MagicMock()
        mock_file.name = "test.mp3"
        mock_file.stat.return_value.st_size = 1024
        mock_file.suffix = ".mp3"
        mock_file.exists.return_value = True
        
        # Mock pydub AudioSegment
        mock_audio = MagicMock()
        mock_audio.channels = 2
        mock_audio.frame_rate = 44100
        mock_audio.sample_width = 2
        mock_audio.__len__.return_value = 60000  # 60 seconds in ms
        
        with patch("pathlib.Path.exists", return_value=True), \
             patch("morag_audio.processor.PydubSegment.from_file", return_value=mock_audio), \
             patch("morag_audio.processor.mutagen.File", return_value={"title": "Test Title"}):
            
            metadata = await processor._extract_metadata(mock_file)
            
            assert metadata["filename"] == "test.mp3"
            assert metadata["file_size"] == 1024
            assert metadata["file_extension"] == "mp3"
            assert metadata["duration"] == 60.0
            assert metadata["channels"] == 2
            assert metadata["sample_rate"] == 44100
            assert metadata["bit_depth"] == 16


@pytest.mark.asyncio
async def test_transcribe_audio(mock_transcriber):
    """Test audio transcription."""
    with patch("morag_audio.processor.WhisperModel", return_value=mock_transcriber):
        processor = AudioProcessor()
        processor.transcriber = mock_transcriber
        
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        
        segments, transcript = await processor._transcribe_audio(mock_file)
        
        assert len(segments) == 3
        assert all(isinstance(segment, AudioSegment) for segment in segments)
        assert "This is test segment 1." in transcript
        assert "This is test segment 2." in transcript
        assert "This is test segment 3." in transcript


@pytest.mark.asyncio
async def test_process_success():
    """Test successful audio processing."""
    with patch("morag_audio.processor.WhisperModel", return_value=MagicMock()), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(AudioProcessor, "_extract_metadata", new_callable=AsyncMock) as mock_extract, \
         patch.object(AudioProcessor, "_transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
        
        # Set up mocks
        mock_extract.return_value = {"duration": 60.0, "channels": 2}
        
        segments = [
            AudioSegment(start=0.0, end=5.0, text="This is test segment 1."),
            AudioSegment(start=5.0, end=10.0, text="This is test segment 2."),
            AudioSegment(start=10.0, end=15.0, text="This is test segment 3.")
        ]
        transcript = "This is test segment 1. This is test segment 2. This is test segment 3."
        mock_transcribe.return_value = (segments, transcript)
        
        # Create processor and process file
        processor = AudioProcessor()
        result = await processor.process("test.mp3")
        
        # Check result
        assert result.success is True
        assert result.transcript == transcript
        assert len(result.segments) == 3
        assert result.metadata["word_count"] == 15  # 5 words per segment
        assert result.metadata["segment_count"] == 3
        assert result.metadata["duration"] == 60.0


@pytest.mark.asyncio
async def test_process_with_diarization():
    """Test audio processing with speaker diarization."""
    config = AudioConfig(enable_diarization=True)
    
    with patch("morag_audio.processor.WhisperModel", return_value=MagicMock()), \
         patch("morag_audio.processor.Pipeline", return_value=MagicMock()), \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(AudioProcessor, "_extract_metadata", new_callable=AsyncMock) as mock_extract, \
         patch.object(AudioProcessor, "_transcribe_audio", new_callable=AsyncMock) as mock_transcribe, \
         patch.object(AudioProcessor, "_apply_diarization", new_callable=AsyncMock) as mock_diarize:
        
        # Set up mocks
        mock_extract.return_value = {"duration": 60.0}
        
        segments = [
            AudioSegment(start=0.0, end=5.0, text="This is test segment 1."),
            AudioSegment(start=5.0, end=10.0, text="This is test segment 2."),
            AudioSegment(start=10.0, end=15.0, text="This is test segment 3.")
        ]
        transcript = "This is test segment 1. This is test segment 2. This is test segment 3."
        mock_transcribe.return_value = (segments, transcript)
        
        # Mock diarization to add speaker info
        diarized_segments = [
            AudioSegment(start=0.0, end=5.0, text="This is test segment 1.", speaker="Speaker A"),
            AudioSegment(start=5.0, end=10.0, text="This is test segment 2.", speaker="Speaker B"),
            AudioSegment(start=10.0, end=15.0, text="This is test segment 3.", speaker="Speaker A")
        ]
        mock_diarize.return_value = diarized_segments
        
        # Create processor and process file
        processor = AudioProcessor(config)
        processor.diarization_pipeline = MagicMock()  # Mock the pipeline
        result = await processor.process("test.mp3")
        
        # Check result
        assert result.success is True
        assert len(result.segments) == 3
        assert result.segments[0].speaker == "Speaker A"
        assert result.segments[1].speaker == "Speaker B"
        assert result.segments[2].speaker == "Speaker A"
        assert result.metadata["num_speakers"] == 2
        assert set(result.metadata["speakers"]) == {"Speaker A", "Speaker B"}