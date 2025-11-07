"""Tests for the AudioProcessor class."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from morag_audio.processor import AudioProcessor, AudioConfig, AudioProcessingError, AudioSegment
from morag_audio.processor import AudioProcessingResult
from morag_audio import AudioTranscriptSegment


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
def rest_audio_config():
    """Return a REST API audio config for testing."""
    return AudioConfig(
        model_size="whisper-1",
        use_rest_api=True,
        openai_api_key="test-api-key",
        api_base_url="https://api.openai.com/v1",
        timeout=30
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
async def test_processor_initialization_rest_api(rest_audio_config):
    """Test that the processor initializes correctly with REST API."""
    with patch("morag_audio.processor.RestTranscriptionService") as mock_rest_service:
        processor = AudioProcessor(rest_audio_config)
        assert processor.config == rest_audio_config
        assert processor.rest_transcription_service is not None
        mock_rest_service.assert_called_once_with(rest_audio_config)


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
async def test_transcribe_audio_rest_api(rest_audio_config):
    """Test audio transcription with REST API."""
    with patch("morag_audio.processor.RestTranscriptionService") as mock_rest_service_class:
        # Mock the service instance
        mock_service = AsyncMock()
        mock_rest_service_class.return_value = mock_service

        # Mock transcription response
        mock_segments = [
            AudioTranscriptSegment("Hello world", 0.0, 2.0, 0.95, "en"),
            AudioTranscriptSegment("from REST API", 2.0, 4.0, 0.90, "en")
        ]
        mock_service.transcribe_audio.return_value = (
            "Hello world from REST API",
            mock_segments,
            "en"
        )
        mock_service.convert_to_markdown.return_value = "# Transcript\n\nHello world from REST API"

        processor = AudioProcessor(rest_audio_config)
        mock_file = MagicMock()
        mock_file.exists.return_value = True

        result = await processor._transcribe_audio(mock_file, rest_audio_config)

        assert isinstance(result, AudioProcessingResult)
        assert result.text == "Hello world from REST API"
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.markdown_transcript == "# Transcript\n\nHello world from REST API"
        mock_service.transcribe_audio.assert_called_once_with(mock_file)
        mock_service.convert_to_markdown.assert_called_once()


@pytest.mark.asyncio
async def test_transcribe_audio_rest_api_error(rest_audio_config):
    """Test audio transcription with REST API error."""
    with patch("morag_audio.processor.RestTranscriptionService") as mock_rest_service_class:
        # Mock the service instance to raise an error
        mock_service = AsyncMock()
        mock_rest_service_class.return_value = mock_service
        mock_service.transcribe_audio.side_effect = Exception("REST API failed")

        processor = AudioProcessor(rest_audio_config)
        mock_file = MagicMock()
        mock_file.exists.return_value = True

        with pytest.raises(Exception, match="REST API failed"):
            await processor._transcribe_audio(mock_file, rest_audio_config)


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
async def test_process_success_rest_api(rest_audio_config):
    """Test successful audio processing with REST API."""
    with patch("morag_audio.processor.RestTranscriptionService") as mock_rest_service_class, \
         patch("pathlib.Path.exists", return_value=True), \
         patch.object(AudioProcessor, "_extract_metadata", new_callable=AsyncMock) as mock_extract:

        # Mock the service instance
        mock_service = AsyncMock()
        mock_rest_service_class.return_value = mock_service

        # Set up mocks
        mock_extract.return_value = {"duration": 60.0, "channels": 2}

        # Mock transcription result
        mock_result = AudioProcessingResult(
            text="Hello world from REST API",
            language="en",
            confidence=0.95,
            duration=60.0,
            segments=[
                AudioTranscriptSegment("Hello world", 0.0, 2.0, 0.95, "en"),
                AudioTranscriptSegment("from REST API", 2.0, 4.0, 0.90, "en")
            ],
            metadata={"duration": 60.0, "channels": 2},
            processing_time=2.0,
            model_used="whisper-1",
            markdown_transcript="# Transcript\n\nHello world from REST API"
        )

        # Mock _transcribe_audio to return the result directly
        with patch.object(AudioProcessor, "_transcribe_audio", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = mock_result

            # Create processor and process file
            processor = AudioProcessor(rest_audio_config)
            result = await processor.process("test.mp3")

            # Check result
            assert result.success is True
            assert result.transcript == "Hello world from REST API"
            assert result.language == "en"
            assert len(result.segments) == 2
            assert result.markdown_transcript == "# Transcript\n\nHello world from REST API"


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
