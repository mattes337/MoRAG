"""Tests for YouTube transcript functionality."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from morag_youtube.transcript import (
    YouTubeTranscriptService,
    YouTubeTranscript,
    TranscriptSegment
)
from morag_core.exceptions import ProcessingError
from youtube_transcript_api import YouTubeTranscriptApi


class TestYouTubeTranscriptService:
    """Test cases for YouTubeTranscriptService."""
    
    @pytest.fixture
    def transcript_service(self):
        """Create transcript service instance."""
        return YouTubeTranscriptService()
    
    def test_extract_video_id_standard_url(self, transcript_service):
        """Test extracting video ID from standard YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = transcript_service.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_video_id_short_url(self, transcript_service):
        """Test extracting video ID from short YouTube URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        video_id = transcript_service.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_video_id_embed_url(self, transcript_service):
        """Test extracting video ID from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        video_id = transcript_service.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_video_id_direct_id(self, transcript_service):
        """Test extracting video ID when input is already an ID."""
        video_id = transcript_service.extract_video_id("dQw4w9WgXcQ")
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_video_id_invalid_url(self, transcript_service):
        """Test extracting video ID from invalid URL."""
        with pytest.raises(ProcessingError):
            transcript_service.extract_video_id("https://example.com/invalid")
    
    @pytest.mark.asyncio
    @patch.object(YouTubeTranscriptApi, 'list')
    async def test_get_available_transcripts(self, mock_list, transcript_service):
        """Test getting available transcripts."""
        # Mock transcript list
        mock_transcript = Mock()
        mock_transcript.language_code = 'en'
        mock_transcript.language = 'English (auto-generated)'
        mock_transcript.is_generated = True
        mock_transcript.is_translatable = True

        mock_transcript_list = Mock()
        mock_transcript_list.__iter__ = Mock(return_value=iter([mock_transcript]))
        mock_list.return_value = mock_transcript_list
        
        # Test the method
        video_id = "dQw4w9WgXcQ"
        available = await transcript_service.get_available_transcripts(video_id)
        
        assert 'en' in available
        assert available['en']['language'] == 'English (auto-generated)'
        assert available['en']['is_generated'] is True
        assert available['en']['is_translatable'] is True
    
    @pytest.mark.asyncio
    async def test_determine_target_language_no_preference(self, transcript_service):
        """Test language determination with no preference."""
        available = {
            'en': {'is_generated': True},
            'de': {'is_generated': False}  # Manual transcript
        }
        
        # Should prefer manual transcript
        target = await transcript_service._determine_target_language(available, None)
        assert target == 'de'
    
    @pytest.mark.asyncio
    async def test_determine_target_language_with_preference(self, transcript_service):
        """Test language determination with preference."""
        available = {
            'en': {'is_generated': True},
            'de': {'is_generated': False}
        }
        
        # Should use preferred language if available
        target = await transcript_service._determine_target_language(available, 'en')
        assert target == 'en'
    
    @pytest.mark.asyncio
    async def test_determine_target_language_fallback(self, transcript_service):
        """Test language determination with unavailable preference."""
        available = {
            'en': {'is_generated': True},
            'de': {'is_generated': False}
        }
        
        # Should fallback to available language
        target = await transcript_service._determine_target_language(available, 'fr')
        assert target in ['en', 'de']  # Should pick one of the available
    
    def test_seconds_to_srt_time(self, transcript_service):
        """Test SRT time formatting."""
        # Test with 65.5 seconds (1 minute, 5.5 seconds)
        srt_time = transcript_service._seconds_to_srt_time(65.5)
        assert srt_time == "00:01:05,500"
    
    def test_seconds_to_vtt_time(self, transcript_service):
        """Test WebVTT time formatting."""
        # Test with 65.5 seconds (1 minute, 5.5 seconds)
        vtt_time = transcript_service._seconds_to_vtt_time(65.5)
        assert vtt_time == "00:01:05.500"
    
    @pytest.mark.asyncio
    async def test_save_transcript_to_file_text(self, transcript_service):
        """Test saving transcript as text file."""
        # Create sample transcript
        segments = [
            TranscriptSegment("Hello world", 0.0, 2.0),
            TranscriptSegment("This is a test", 2.0, 3.0)
        ]
        transcript = YouTubeTranscript(
            video_id="test123",
            language="en",
            segments=segments,
            full_text="Hello world This is a test",
            is_auto_generated=True
        )
        
        # Test saving to temp file
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "transcript.txt"
            result_path = await transcript_service.save_transcript_to_file(
                transcript, output_path, "text"
            )
            
            assert result_path.exists()
            content = result_path.read_text(encoding='utf-8')
            assert content == "Hello world This is a test"
    
    @pytest.mark.asyncio
    async def test_save_transcript_to_file_srt(self, transcript_service):
        """Test saving transcript as SRT file."""
        # Create sample transcript
        segments = [
            TranscriptSegment("Hello world", 0.0, 2.0),
            TranscriptSegment("This is a test", 2.0, 3.0)
        ]
        transcript = YouTubeTranscript(
            video_id="test123",
            language="en",
            segments=segments,
            full_text="Hello world This is a test",
            is_auto_generated=True
        )
        
        # Test saving to temp file
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "transcript.srt"
            result_path = await transcript_service.save_transcript_to_file(
                transcript, output_path, "srt"
            )
            
            assert result_path.exists()
            content = result_path.read_text(encoding='utf-8')
            assert "1" in content
            assert "00:00:00,000 --> 00:00:02,000" in content
            assert "Hello world" in content


class TestTranscriptSegment:
    """Test cases for TranscriptSegment."""
    
    def test_segment_creation(self):
        """Test creating a transcript segment."""
        segment = TranscriptSegment("Hello world", 0.0, 2.5)
        assert segment.text == "Hello world"
        assert segment.start == 0.0
        assert segment.duration == 2.5
        assert segment.end == 2.5
    
    def test_segment_end_property(self):
        """Test the end property calculation."""
        segment = TranscriptSegment("Test", 10.0, 5.0)
        assert segment.end == 15.0


class TestYouTubeTranscript:
    """Test cases for YouTubeTranscript."""
    
    def test_transcript_creation(self):
        """Test creating a YouTube transcript."""
        segments = [
            TranscriptSegment("Hello", 0.0, 1.0),
            TranscriptSegment("World", 1.0, 2.0)
        ]
        
        transcript = YouTubeTranscript(
            video_id="test123",
            language="en",
            segments=segments,
            full_text="Hello World",
            is_auto_generated=False
        )
        
        assert transcript.video_id == "test123"
        assert transcript.language == "en"
        assert len(transcript.segments) == 2
        assert transcript.full_text == "Hello World"
        assert transcript.is_auto_generated is False
    
    def test_transcript_duration_property(self):
        """Test the duration property calculation."""
        segments = [
            TranscriptSegment("Hello", 0.0, 1.0),  # ends at 1.0
            TranscriptSegment("World", 1.0, 2.5)   # ends at 3.5
        ]
        
        transcript = YouTubeTranscript(
            video_id="test123",
            language="en",
            segments=segments,
            full_text="Hello World",
            is_auto_generated=False
        )
        
        assert transcript.duration == 3.5
    
    def test_transcript_empty_segments(self):
        """Test transcript with no segments."""
        transcript = YouTubeTranscript(
            video_id="test123",
            language="en",
            segments=[],
            full_text="",
            is_auto_generated=False
        )
        
        assert transcript.duration == 0.0
