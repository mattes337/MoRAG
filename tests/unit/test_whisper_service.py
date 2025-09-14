"""Unit tests for Whisper service."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from morag_audio import AudioService, AudioConfig, AudioProcessingResult
from morag_core.exceptions import ExternalServiceError

class TestAudioService:
    """Test cases for AudioService."""

    @pytest.fixture
    def audio_service(self):
        """Create audio service instance."""
        return AudioService()
    
    @pytest.fixture
    def audio_config(self):
        """Create audio config."""
        return AudioConfig(
            model_size="tiny",
            device="cpu",
            compute_type="int8"
        )
    
    @pytest.fixture
    def mock_audio_file(self):
        """Create mock audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"mock audio data")
            return Path(f.name)
    
    def test_whisper_service_initialization(self, whisper_service):
        """Test whisper service initialization."""
        assert whisper_service._models == {}
        assert whisper_service._default_config is not None
    
    @patch('morag.services.whisper_service.WhisperModel')
    def test_get_model_success(self, mock_whisper_model, whisper_service):
        """Test successful model loading."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        result = whisper_service._get_model("tiny", "cpu", "int8")
        
        assert result == mock_model
        mock_whisper_model.assert_called_once_with("tiny", device="cpu", compute_type="int8")
        
        # Test caching - should not create new model
        result2 = whisper_service._get_model("tiny", "cpu", "int8")
        assert result2 == mock_model
        assert mock_whisper_model.call_count == 1  # Still only called once
    
    @patch('morag.services.whisper_service.WhisperModel')
    def test_get_model_failure(self, mock_whisper_model, whisper_service):
        """Test model loading failure."""
        mock_whisper_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(ExternalServiceError, match="Failed to load Whisper model"):
            whisper_service._get_model("tiny", "cpu", "int8")
    
    @pytest.mark.asyncio
    @patch('morag.services.whisper_service.WhisperService._get_model')
    async def test_transcribe_audio_success(self, mock_get_model, whisper_service, mock_audio_file, audio_config):
        """Test successful audio transcription."""
        # Mock model and transcription results
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        mock_segment = Mock()
        mock_segment.text = "Hello world"
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.avg_logprob = -0.5
        
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.duration = 2.0
        mock_info.language_probability = 0.95
        mock_info.duration_after_vad = 1.8
        mock_info.all_language_probs = {"en": 0.95, "es": 0.05}
        
        # Mock the _transcribe_sync method
        with patch.object(whisper_service, '_transcribe_sync', return_value=([mock_segment], mock_info)):
            result = await whisper_service.transcribe_audio(mock_audio_file, audio_config)
        
        assert isinstance(result, AudioProcessingResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 2.0
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"
        assert result.segments[0].start_time == 0.0
        assert result.segments[0].end_time == 2.0
        assert 0.0 <= result.segments[0].confidence <= 1.0  # Confidence should be normalized
    
    @pytest.mark.asyncio
    @patch('morag.services.whisper_service.WhisperService._get_model')
    async def test_transcribe_audio_failure(self, mock_get_model, whisper_service, mock_audio_file, audio_config):
        """Test audio transcription failure."""
        mock_get_model.side_effect = Exception("Transcription failed")
        
        with pytest.raises(ExternalServiceError, match="Audio transcription failed"):
            await whisper_service.transcribe_audio(mock_audio_file, audio_config)
    
    @pytest.mark.asyncio
    @patch('morag.services.whisper_service.WhisperService.transcribe_audio')
    async def test_transcribe_with_chunking(self, mock_transcribe_audio, whisper_service, mock_audio_file, audio_config):
        """Test chunked transcription (currently delegates to regular transcription)."""
        mock_result = AudioProcessingResult(
            text="Hello world",
            language="en",
            confidence=0.9,
            duration=2.0,
            segments=[],
            metadata={},
            processing_time=1.0,
            model_used="tiny"
        )
        mock_transcribe_audio.return_value = mock_result
        
        result = await whisper_service.transcribe_with_chunking(mock_audio_file, audio_config)
        
        assert result == mock_result
        mock_transcribe_audio.assert_called_once_with(mock_audio_file, audio_config)
    
    @pytest.mark.asyncio
    @patch('morag.services.whisper_service.WhisperService._get_model')
    async def test_detect_language_success(self, mock_get_model, whisper_service, mock_audio_file):
        """Test successful language detection."""
        # Mock model and transcription results
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.all_language_probs = {"en": 0.95, "es": 0.05}
        
        mock_model.transcribe.return_value = ([], mock_info)
        
        result = await whisper_service.detect_language(mock_audio_file, "base")
        
        assert result["language"] == "en"
        assert result["language_probability"] == 0.95
        assert result["all_language_probs"] == {"en": 0.95, "es": 0.05}
        
        # Verify model was called with correct parameters
        mock_model.transcribe.assert_called_once_with(
            str(mock_audio_file),
            language=None,  # Auto-detect
            beam_size=1,
            best_of=1,
            temperature=0.0
        )
    
    @pytest.mark.asyncio
    @patch('morag.services.whisper_service.WhisperService._get_model')
    async def test_detect_language_failure(self, mock_get_model, whisper_service, mock_audio_file):
        """Test language detection failure."""
        mock_get_model.side_effect = Exception("Language detection failed")
        
        with pytest.raises(ExternalServiceError, match="Language detection failed"):
            await whisper_service.detect_language(mock_audio_file, "base")
    
    def test_transcribe_sync(self, whisper_service, audio_config):
        """Test synchronous transcription method."""
        mock_model = Mock()
        
        whisper_service._transcribe_sync(mock_model, "test.wav", audio_config)
        
        mock_model.transcribe.assert_called_once_with(
            "test.wav",
            language=audio_config.language,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=None
        )
    
    def test_get_available_models(self, whisper_service):
        """Test getting available models."""
        models = whisper_service.get_available_models()
        
        assert isinstance(models, list)
        assert "tiny" in models
        assert "base" in models
        assert "small" in models
        assert "medium" in models
        assert "large" in models
        assert "large-v2" in models
        assert "large-v3" in models
    
    def test_get_supported_languages(self, whisper_service):
        """Test getting supported languages."""
        languages = whisper_service.get_supported_languages()
        
        assert isinstance(languages, list)
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        assert "de" in languages
        assert "zh" in languages
        assert len(languages) > 50  # Should have many languages
    
    def test_cleanup_models(self, whisper_service):
        """Test model cleanup."""
        # Add some mock models
        whisper_service._models["test_model"] = Mock()
        whisper_service._models["another_model"] = Mock()
        
        assert len(whisper_service._models) == 2
        
        whisper_service.cleanup_models()
        
        assert len(whisper_service._models) == 0
    
    @pytest.mark.asyncio
    @patch('morag.services.whisper_service.WhisperService._get_model')
    async def test_transcribe_audio_empty_segments(self, mock_get_model, whisper_service, mock_audio_file, audio_config):
        """Test transcription with empty segments."""
        # Mock model with no segments
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.duration = 2.0
        mock_info.language_probability = 0.95
        mock_info.duration_after_vad = 1.8
        mock_info.all_language_probs = {"en": 0.95}
        
        # Mock the _transcribe_sync method with empty segments
        with patch.object(whisper_service, '_transcribe_sync', return_value=([], mock_info)):
            result = await whisper_service.transcribe_audio(mock_audio_file, audio_config)
        
        assert result.text == ""
        assert result.confidence == 0.0  # No segments means 0 confidence
        assert len(result.segments) == 0
    
    @pytest.mark.asyncio
    @patch('morag.services.whisper_service.WhisperService._get_model')
    async def test_transcribe_audio_multiple_segments(self, mock_get_model, whisper_service, mock_audio_file, audio_config):
        """Test transcription with multiple segments."""
        # Mock model with multiple segments
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        mock_segment1 = Mock()
        mock_segment1.text = "Hello"
        mock_segment1.start = 0.0
        mock_segment1.end = 1.0
        mock_segment1.avg_logprob = -0.3
        
        mock_segment2 = Mock()
        mock_segment2.text = "world"
        mock_segment2.start = 1.0
        mock_segment2.end = 2.0
        mock_segment2.avg_logprob = -0.7
        
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.duration = 2.0
        mock_info.language_probability = 0.95
        mock_info.duration_after_vad = 1.8
        mock_info.all_language_probs = {"en": 0.95}
        
        # Mock the _transcribe_sync method
        with patch.object(whisper_service, '_transcribe_sync', return_value=([mock_segment1, mock_segment2], mock_info)):
            result = await whisper_service.transcribe_audio(mock_audio_file, audio_config)
        
        assert result.text == "Hello world"
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello"
        assert result.segments[1].text == "world"
        # Confidence should be average of both segments (normalized from log probs)
        # -0.3 -> 0.35, -0.7 -> 0.15, average = 0.25
        assert abs(result.confidence - 0.25) < 0.05
