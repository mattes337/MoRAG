"""Unit tests for audio processor."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from morag_audio import AudioProcessor, AudioConfig, AudioProcessingResult, AudioTranscriptSegment
from morag_core.exceptions import ProcessingError, ExternalServiceError

class TestAudioProcessor:
    """Test cases for AudioProcessor."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create audio processor instance."""
        return AudioProcessor()
    
    @pytest.fixture
    def audio_config(self):
        """Create audio config."""
        return AudioConfig(
            model_size="tiny",  # Use smallest model for tests
            device="cpu",
            compute_type="int8"
        )
    
    @pytest.fixture
    def mock_audio_file(self):
        """Create mock audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create a simple WAV file header (minimal)
            f.write(b'RIFF')
            f.write((44).to_bytes(4, 'little'))  # File size
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write((16).to_bytes(4, 'little'))  # Format chunk size
            f.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
            f.write((1).to_bytes(2, 'little'))   # Number of channels
            f.write((44100).to_bytes(4, 'little'))  # Sample rate
            f.write((88200).to_bytes(4, 'little'))  # Byte rate
            f.write((2).to_bytes(2, 'little'))   # Block align
            f.write((16).to_bytes(2, 'little'))  # Bits per sample
            f.write(b'data')
            f.write((8).to_bytes(4, 'little'))   # Data chunk size
            f.write(b'\x00' * 8)  # Audio data (silence)
            
            return Path(f.name)
    
    def test_audio_processor_initialization(self, audio_processor):
        """Test audio processor initialization."""
        assert audio_processor.config is not None
        assert audio_processor.config.model_size == "base"
        assert not audio_processor._model_loaded
    
    def test_audio_config_defaults(self):
        """Test audio config default values."""
        config = AudioConfig()
        assert config.model_size == "base"
        assert config.language is None
        assert config.chunk_duration == 300
        assert config.overlap_duration == 30
        assert config.quality_threshold == 0.7
        assert "mp3" in config.supported_formats
        assert "wav" in config.supported_formats
    
    def test_validate_audio_file_success(self, audio_processor, mock_audio_file):
        """Test successful audio file validation."""
        config = AudioConfig()
        # Should not raise any exception
        audio_processor._validate_audio_file(mock_audio_file, config)
    
    def test_validate_audio_file_not_found(self, audio_processor):
        """Test validation with non-existent file."""
        config = AudioConfig()
        non_existent_file = Path("non_existent.wav")
        
        with pytest.raises(ProcessingError, match="Audio file not found"):
            audio_processor._validate_audio_file(non_existent_file, config)
    
    def test_validate_audio_file_unsupported_format(self, audio_processor):
        """Test validation with unsupported format."""
        config = AudioConfig(supported_formats=["wav", "mp3"])
        
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"dummy content")
            file_path = Path(f.name)
        
        with pytest.raises(ProcessingError, match="Unsupported audio format"):
            audio_processor._validate_audio_file(file_path, config)
        
        file_path.unlink()  # Clean up
    
    def test_validate_audio_file_too_large(self, audio_processor):
        """Test validation with file too large."""
        config = AudioConfig(max_file_size=10)  # 10 bytes max
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"x" * 20)  # 20 bytes
            file_path = Path(f.name)
        
        with pytest.raises(ProcessingError, match="Audio file too large"):
            audio_processor._validate_audio_file(file_path, config)
        
        file_path.unlink()  # Clean up
    
    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, audio_processor, mock_audio_file):
        """Test successful metadata extraction."""
        metadata = await audio_processor._extract_metadata(mock_audio_file)
        
        assert "file_name" in metadata
        assert "file_size" in metadata
        assert "file_format" in metadata
        assert metadata["file_format"] == "wav"
        assert metadata["file_name"] == mock_audio_file.name
    
    @pytest.mark.asyncio
    async def test_extract_metadata_fallback(self, audio_processor):
        """Test metadata extraction fallback."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"invalid audio data")
            file_path = Path(f.name)
        
        metadata = await audio_processor._extract_metadata(file_path)
        
        # Should still return basic file info
        assert "file_name" in metadata
        assert "duration" in metadata
        assert metadata["duration"] == 0.0
        
        file_path.unlink()  # Clean up
    
    @pytest.mark.asyncio
    async def test_convert_to_wav_already_wav(self, audio_processor, mock_audio_file):
        """Test conversion when file is already WAV."""
        result_path = await audio_processor._convert_to_wav(mock_audio_file)
        assert result_path == mock_audio_file
    
    @pytest.mark.asyncio
    @patch('morag.processors.audio.PydubAudioSegment')
    async def test_convert_to_wav_conversion(self, mock_audio_segment, audio_processor):
        """Test audio format conversion."""
        # Create mock MP3 file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"mock mp3 data")
            mp3_path = Path(f.name)

        # Mock PydubAudioSegment
        mock_audio = Mock()
        mock_audio_segment.from_file.return_value = mock_audio

        try:
            result_path = await audio_processor._convert_to_wav(mp3_path)

            # Should have called PydubAudioSegment.from_file
            mock_audio_segment.from_file.assert_called_once_with(str(mp3_path))

            # Should have called export
            mock_audio.export.assert_called_once()
            
            # Result should be different from input
            assert result_path != mp3_path
            assert result_path.suffix == ".wav"
            
        finally:
            mp3_path.unlink()  # Clean up
            if result_path.exists():
                result_path.unlink()
    
    @pytest.mark.asyncio
    @patch('morag.processors.audio.PydubAudioSegment')
    async def test_convert_to_wav_failure(self, mock_audio_segment, audio_processor):
        """Test audio conversion failure."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"mock mp3 data")
            mp3_path = Path(f.name)

        # Mock PydubAudioSegment to raise exception
        mock_audio_segment.from_file.side_effect = Exception("Conversion failed")
        
        try:
            with pytest.raises(ProcessingError, match="Audio conversion failed"):
                await audio_processor._convert_to_wav(mp3_path)
        finally:
            mp3_path.unlink()  # Clean up
    
    @pytest.mark.asyncio
    @patch('morag.processors.audio.WhisperModel')
    async def test_transcribe_audio_success(self, mock_whisper_model, audio_processor, mock_audio_file, audio_config):
        """Test successful audio transcription."""
        # Mock Whisper model and results
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        # Mock transcription results
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
        
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        result = await audio_processor._transcribe_audio(mock_audio_file, audio_config)
        
        assert isinstance(result, AudioProcessingResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"
        assert result.segments[0].start_time == 0.0
        assert result.segments[0].end_time == 2.0
    
    @pytest.mark.asyncio
    @patch('morag.processors.audio.WhisperModel')
    async def test_transcribe_audio_failure(self, mock_whisper_model, audio_processor, mock_audio_file, audio_config):
        """Test audio transcription failure."""
        # Mock Whisper model to raise exception
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        
        with pytest.raises(ExternalServiceError, match="Audio transcription failed"):
            await audio_processor._transcribe_audio(mock_audio_file, audio_config)
    
    @pytest.mark.asyncio
    @patch('morag.processors.audio.AudioProcessor._transcribe_audio')
    @patch('morag.processors.audio.AudioProcessor._convert_to_wav')
    @patch('morag.processors.audio.AudioProcessor._extract_metadata')
    @patch('morag.processors.audio.AudioProcessor._validate_audio_file')
    async def test_process_audio_file_success(
        self,
        mock_validate,
        mock_extract_metadata,
        mock_convert,
        mock_transcribe,
        audio_processor,
        mock_audio_file,
        audio_config
    ):
        """Test successful audio file processing."""
        # Mock all the methods
        mock_extract_metadata.return_value = {"duration": 2.0, "file_name": "test.wav"}
        mock_convert.return_value = mock_audio_file
        
        mock_transcribe_result = AudioProcessingResult(
            text="Hello world",
            language="en",
            confidence=0.9,
            duration=2.0,
            segments=[AudioTranscriptSegment("Hello world", 0.0, 2.0, 0.9, language="en")],
            metadata={},
            processing_time=1.0,
            model_used="tiny"
        )
        mock_transcribe.return_value = mock_transcribe_result
        
        result = await audio_processor.process_audio_file(mock_audio_file, audio_config)
        
        assert isinstance(result, AudioProcessingResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.9
        assert result.duration == 2.0
        
        # Verify all methods were called
        mock_validate.assert_called_once()
        mock_extract_metadata.assert_called_once()
        mock_convert.assert_called_once()
        mock_transcribe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_audio_file_validation_failure(self, audio_processor):
        """Test audio file processing with validation failure."""
        non_existent_file = Path("non_existent.wav")
        
        with pytest.raises(ProcessingError):
            await audio_processor.process_audio_file(non_existent_file)
    
    def test_audio_segment_creation(self):
        """Test AudioTranscriptSegment creation."""
        from morag_audio import AudioTranscriptSegment

        segment = AudioTranscriptSegment(
            text="Hello world",
            start_time=0.0,
            end_time=2.0,
            confidence=0.9,
            speaker_id="speaker_1",
            language="en"
        )

        assert segment.text == "Hello world"
        assert segment.start_time == 0.0
        assert segment.end_time == 2.0
        assert segment.confidence == 0.9
        assert segment.speaker_id == "speaker_1"
        assert segment.language == "en"
    
    def test_audio_processing_result_creation(self):
        """Test AudioProcessingResult creation."""
        segments = [AudioTranscriptSegment("Hello", 0.0, 1.0, 0.9)]
        metadata = {"file_name": "test.wav"}
        
        result = AudioProcessingResult(
            text="Hello world",
            language="en",
            confidence=0.9,
            duration=2.0,
            segments=segments,
            metadata=metadata,
            processing_time=1.5,
            model_used="base"
        )
        
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.9
        assert result.duration == 2.0
        assert len(result.segments) == 1
        assert result.metadata == metadata
        assert result.processing_time == 1.5
        assert result.model_used == "base"
