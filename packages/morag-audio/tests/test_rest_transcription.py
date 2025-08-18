"""Tests for the RestTranscriptionService class."""

import pytest
import aiohttp
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from morag_audio.processor import AudioConfig
from morag_audio.rest_transcription import RestTranscriptionService, RestTranscriptionError
from morag_audio.processor import AudioTranscriptSegment


@pytest.fixture
def rest_config():
    """Return a REST API config for testing."""
    return AudioConfig(
        model_size="whisper-1",
        use_rest_api=True,
        openai_api_key="test-api-key",
        api_base_url="https://api.openai.com/v1",
        timeout=30
    )


@pytest.fixture
def mock_audio_file():
    """Return a mock audio file for testing."""
    mock_file = MagicMock()
    mock_file.name = "test.wav"
    mock_file.read_bytes.return_value = b"fake audio data"
    return mock_file


@pytest.fixture
def mock_openai_response():
    """Return a mock OpenAI API response."""
    return {
        "text": "Hello world, this is a test transcription.",
        "language": "en",
        "duration": 5.0,
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world,",
                "tokens": [15496, 1002, 11],
                "temperature": 0.0,
                "avg_logprob": -0.3,
                "compression_ratio": 1.2,
                "no_speech_prob": 0.1
            },
            {
                "id": 1,
                "seek": 250,
                "start": 2.5,
                "end": 5.0,
                "text": " this is a test transcription.",
                "tokens": [341, 318, 257, 1332, 28535, 13],
                "temperature": 0.0,
                "avg_logprob": -0.25,
                "compression_ratio": 1.3,
                "no_speech_prob": 0.05
            }
        ]
    }


class TestRestTranscriptionService:
    """Test cases for RestTranscriptionService."""

    def test_initialization(self, rest_config):
        """Test service initialization."""
        service = RestTranscriptionService(rest_config)
        assert service.config == rest_config
        assert service.api_key == "test-api-key"
        assert service.base_url == "https://api.openai.com/v1"
        assert service.timeout == 30

    def test_initialization_missing_api_key(self):
        """Test service initialization with missing API key."""
        config = AudioConfig(use_rest_api=True)
        with pytest.raises(RestTranscriptionError, match="OpenAI API key is required"):
            RestTranscriptionService(config)

    @pytest.mark.asyncio
    async def test_transcribe_audio_success(self, rest_config, mock_audio_file, mock_openai_response):
        """Test successful audio transcription."""
        service = RestTranscriptionService(rest_config)
        
        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_openai_response
        
        # Mock aiohttp session
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            text, segments, language = await service.transcribe_audio(mock_audio_file)
            
            # Verify results
            assert text == "Hello world, this is a test transcription."
            assert language == "en"
            assert len(segments) == 2
            
            # Check first segment
            assert isinstance(segments[0], AudioTranscriptSegment)
            assert segments[0].text == "Hello world,"
            assert segments[0].start_time == 0.0
            assert segments[0].end_time == 2.5
            assert segments[0].confidence == 0.7  # Converted from avg_logprob
            assert segments[0].language == "en"
            
            # Check second segment
            assert segments[1].text == " this is a test transcription."
            assert segments[1].start_time == 2.5
            assert segments[1].end_time == 5.0

    @pytest.mark.asyncio
    async def test_transcribe_audio_http_error(self, rest_config, mock_audio_file):
        """Test audio transcription with HTTP error."""
        service = RestTranscriptionService(rest_config)
        
        # Mock aiohttp response with error
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text.return_value = "Bad Request"
        
        # Mock aiohttp session
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(RestTranscriptionError, match="HTTP 400: Bad Request"):
                await service.transcribe_audio(mock_audio_file)

    @pytest.mark.asyncio
    async def test_transcribe_audio_network_error(self, rest_config, mock_audio_file):
        """Test audio transcription with network error."""
        service = RestTranscriptionService(rest_config)
        
        # Mock aiohttp session to raise exception
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ClientError("Network error")
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(RestTranscriptionError, match="Network error during transcription"):
                await service.transcribe_audio(mock_audio_file)

    @pytest.mark.asyncio
    async def test_transcribe_audio_timeout(self, rest_config, mock_audio_file):
        """Test audio transcription with timeout."""
        service = RestTranscriptionService(rest_config)
        
        # Mock aiohttp session to raise timeout
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ServerTimeoutError("Timeout")
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(RestTranscriptionError, match="Request timeout during transcription"):
                await service.transcribe_audio(mock_audio_file)

    def test_convert_to_markdown(self, rest_config):
        """Test conversion of transcript to markdown."""
        service = RestTranscriptionService(rest_config)
        
        segments = [
            AudioTranscriptSegment("Hello world,", 0.0, 2.5, 0.95, "en"),
            AudioTranscriptSegment(" this is a test.", 2.5, 5.0, 0.90, "en")
        ]
        
        markdown = service.convert_to_markdown("Hello world, this is a test.", segments)
        
        expected = """# Audio Transcript

**Language:** en  
**Duration:** 5.0 seconds

## Transcript

Hello world, this is a test.

## Segments

| Time | Speaker | Text |
|------|---------|------|
| 00:00 - 00:02 |  | Hello world, |
| 00:02 - 00:05 |  |  this is a test. |
"""
        
        assert markdown.strip() == expected.strip()

    def test_convert_to_markdown_with_speakers(self, rest_config):
        """Test conversion of transcript to markdown with speaker information."""
        service = RestTranscriptionService(rest_config)
        
        segments = [
            AudioTranscriptSegment("Hello world,", 0.0, 2.5, 0.95, "en", speaker="Speaker A"),
            AudioTranscriptSegment(" this is a test.", 2.5, 5.0, 0.90, "en", speaker="Speaker B")
        ]
        
        markdown = service.convert_to_markdown("Hello world, this is a test.", segments)
        
        expected = """# Audio Transcript

**Language:** en  
**Duration:** 5.0 seconds

## Transcript

Hello world, this is a test.

## Segments

| Time | Speaker | Text |
|------|---------|------|
| 00:00 - 00:02 | Speaker A | Hello world, |
| 00:02 - 00:05 | Speaker B |  this is a test. |
"""
        
        assert markdown.strip() == expected.strip()

    def test_logprob_to_confidence_conversion(self, rest_config):
        """Test conversion of log probability to confidence score."""
        service = RestTranscriptionService(rest_config)
        
        # Test various log probability values
        assert service._logprob_to_confidence(0.0) == 1.0  # Perfect confidence
        assert service._logprob_to_confidence(-0.5) == 0.6  # Good confidence
        assert service._logprob_to_confidence(-1.0) == 0.37  # Moderate confidence
        assert service._logprob_to_confidence(-2.0) == 0.14  # Low confidence
        assert service._logprob_to_confidence(-5.0) == 0.01  # Very low confidence

    def test_format_time(self, rest_config):
        """Test time formatting for markdown output."""
        service = RestTranscriptionService(rest_config)
        
        assert service._format_time(0.0) == "00:00"
        assert service._format_time(65.5) == "01:05"
        assert service._format_time(3661.2) == "61:01"
        assert service._format_time(125.7) == "02:05"