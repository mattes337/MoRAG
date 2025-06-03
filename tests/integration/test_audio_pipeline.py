"""Integration tests for audio processing pipeline."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from morag.processors.audio import AudioProcessor, AudioConfig
from morag.services.whisper_service import WhisperService
from morag.tasks.audio_tasks import process_audio_file

class TestAudioPipeline:
    """Integration test cases for audio processing pipeline."""
    
    @pytest.fixture
    def audio_config(self):
        """Create audio config for testing."""
        return AudioConfig(
            model_size="tiny",  # Use smallest model for faster tests
            device="cpu",
            compute_type="int8",
            max_file_size=10 * 1024 * 1024  # 10MB for testing
        )
    
    @pytest.fixture
    def mock_wav_file(self):
        """Create a proper mock WAV file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write a minimal but valid WAV header
            f.write(b'RIFF')
            f.write((36).to_bytes(4, 'little'))  # File size - 8
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
            f.write((0).to_bytes(4, 'little'))   # Data chunk size
            
            return Path(f.name)
    
    @pytest.mark.asyncio
    @patch('morag.processors.audio.WhisperModel')
    async def test_audio_processor_integration(self, mock_whisper_model, audio_config, mock_wav_file):
        """Test audio processor integration with mocked Whisper."""
        # Mock Whisper model
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        # Mock transcription results
        mock_segment = Mock()
        mock_segment.text = "This is a test transcription"
        mock_segment.start = 0.0
        mock_segment.end = 3.0
        mock_segment.avg_logprob = -0.5
        
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.duration = 3.0
        mock_info.language_probability = 0.95
        mock_info.duration_after_vad = 2.8
        mock_info.all_language_probs = {"en": 0.95, "es": 0.05}
        
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        # Create processor and process file
        processor = AudioProcessor(audio_config)
        result = await processor.process_audio_file(mock_wav_file, audio_config)
        
        # Verify results
        assert result.text == "This is a test transcription"
        assert result.language == "en"
        assert result.duration == 3.0
        assert len(result.segments) == 1
        assert result.segments[0].text == "This is a test transcription"
        assert result.model_used == "tiny"
        
        # Verify Whisper was called
        mock_whisper_model.assert_called_once_with("tiny", device="cpu", compute_type="int8")
        mock_model.transcribe.assert_called_once()
        
        # Clean up
        mock_wav_file.unlink()
    
    @pytest.mark.asyncio
    @patch('morag.services.whisper_service.WhisperModel')
    async def test_whisper_service_integration(self, mock_whisper_model, audio_config, mock_wav_file):
        """Test Whisper service integration."""
        # Mock Whisper model
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        # Mock transcription results
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
        mock_info.language_probability = 0.98
        mock_info.duration_after_vad = 1.9
        mock_info.all_language_probs = {"en": 0.98, "fr": 0.02}
        
        mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)
        
        # Create service and transcribe
        service = WhisperService()
        result = await service.transcribe_audio(mock_wav_file, audio_config)
        
        # Verify results
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 2.0
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello"
        assert result.segments[1].text == "world"
        
        # Verify confidence calculation
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.segments[0].confidence <= 1.0
        assert 0.0 <= result.segments[1].confidence <= 1.0
        
        # Clean up
        mock_wav_file.unlink()
    
    @pytest.mark.asyncio
    @patch('morag.processors.audio.WhisperModel')
    @patch('morag.services.chunking.chunking_service')
    @patch('morag.services.summarization.enhanced_summarization_service')
    @patch('morag.services.embedding.gemini_service')
    @patch('morag.services.storage.qdrant_service')
    async def test_full_audio_task_pipeline(
        self,
        mock_qdrant,
        mock_gemini,
        mock_summarization,
        mock_chunking,
        mock_whisper_model,
        mock_wav_file
    ):
        """Test full audio processing task pipeline."""
        # Mock Whisper model
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        # Mock transcription results
        mock_segment = Mock()
        mock_segment.text = "This is a longer test transcription that will be chunked and processed through the full pipeline"
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.avg_logprob = -0.4
        
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.duration = 5.0
        mock_info.language_probability = 0.97
        mock_info.duration_after_vad = 4.8
        mock_info.all_language_probs = {"en": 0.97, "de": 0.03}
        
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        
        # Mock chunking service
        from morag.services.chunking import TextChunk
        mock_chunks = [
            TextChunk(
                text="This is a longer test transcription",
                metadata={"chunk_index": 0}
            ),
            TextChunk(
                text="that will be chunked and processed through the full pipeline",
                metadata={"chunk_index": 1}
            )
        ]
        mock_chunking.chunk_text.return_value = mock_chunks
        
        # Mock enhanced summarization
        mock_summary_result = Mock()
        mock_summary_result.summary = "Test summary"
        mock_summary_result.strategy.value = "abstractive"
        mock_summary_result.quality.overall = 0.85
        mock_summary_result.processing_time = 0.3
        mock_summary_result.refinement_iterations = 0
        mock_summarization.generate_summary.return_value = mock_summary_result
        
        # Mock embedding generation
        mock_embedding_result = Mock()
        mock_embedding_result.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding_result.model = "text-embedding-004"
        mock_embedding_result.token_count = 15
        mock_gemini.generate_embedding.return_value = mock_embedding_result
        
        # Mock vector storage
        mock_qdrant.store_chunk.return_value = "point_456"
        
        # Mock task instance
        mock_task = Mock()
        mock_task.update_status = Mock()
        
        # Execute full pipeline
        result = await process_audio_file(
            mock_task,
            file_path=str(mock_wav_file),
            task_id="integration_test_123",
            config={"model_size": "tiny", "device": "cpu"},
            use_enhanced_summary=True
        )
        
        # Verify final result
        assert result["task_id"] == "integration_test_123"
        assert result["status"] == "completed"
        assert result["audio_result"]["language"] == "en"
        assert result["audio_result"]["confidence"] > 0.0
        assert result["chunks_processed"] == 2
        assert result["total_chunks"] == 2
        
        # Verify all services were called
        mock_whisper_model.assert_called_once()
        mock_chunking.chunk_text.assert_called_once()
        assert mock_summarization.generate_summary.call_count == 2  # One per chunk
        assert mock_gemini.generate_embedding.call_count == 2  # One per chunk
        assert mock_qdrant.store_chunk.call_count == 2  # One per chunk
        
        # Verify processed chunks
        processed_chunks = result["processed_chunks"]
        assert len(processed_chunks) == 2
        for chunk in processed_chunks:
            assert "chunk_id" in chunk
            assert "point_id" in chunk
            assert chunk["point_id"] == "point_456"
            assert "metadata" in chunk
        
        # Clean up
        mock_wav_file.unlink()
    
    @pytest.mark.asyncio
    async def test_audio_config_validation(self):
        """Test audio configuration validation."""
        # Test default config
        config = AudioConfig()
        assert config.model_size == "base"
        assert config.device == "cpu"
        assert config.compute_type == "int8"
        
        # Test custom config
        custom_config = AudioConfig(
            model_size="small",
            device="cuda",
            compute_type="float16",
            language="es",
            chunk_duration=600,
            quality_threshold=0.8
        )
        assert custom_config.model_size == "small"
        assert custom_config.device == "cuda"
        assert custom_config.compute_type == "float16"
        assert custom_config.language == "es"
        assert custom_config.chunk_duration == 600
        assert custom_config.quality_threshold == 0.8
    
    @pytest.mark.asyncio
    @patch('morag.services.whisper_service.WhisperModel')
    async def test_language_detection_integration(self, mock_whisper_model, mock_wav_file):
        """Test language detection integration."""
        # Mock Whisper model
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model
        
        # Mock language detection results
        mock_info = Mock()
        mock_info.language = "es"
        mock_info.language_probability = 0.92
        mock_info.all_language_probs = {"es": 0.92, "en": 0.06, "fr": 0.02}
        
        mock_model.transcribe.return_value = ([], mock_info)
        
        # Create service and detect language
        service = WhisperService()
        result = await service.detect_language(mock_wav_file, "base")
        
        # Verify results
        assert result["language"] == "es"
        assert result["language_probability"] == 0.92
        assert result["all_language_probs"]["es"] == 0.92
        assert result["all_language_probs"]["en"] == 0.06
        
        # Verify model was called with correct parameters
        mock_model.transcribe.assert_called_once_with(
            str(mock_wav_file),
            language=None,  # Auto-detect
            beam_size=1,
            best_of=1,
            temperature=0.0
        )
        
        # Clean up
        mock_wav_file.unlink()
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_wav_file):
        """Test error handling in audio processing pipeline."""
        # Test with invalid audio config
        invalid_config = AudioConfig(max_file_size=1)  # 1 byte max
        processor = AudioProcessor(invalid_config)
        
        # Should raise ProcessingError for file too large
        with pytest.raises(Exception):  # Could be ProcessingError or other
            await processor.process_audio_file(mock_wav_file, invalid_config)
        
        # Clean up
        mock_wav_file.unlink()
