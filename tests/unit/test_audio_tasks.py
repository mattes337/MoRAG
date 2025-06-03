"""Unit tests for audio tasks."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import the actual task functions, not the Celery tasks
from morag.tasks import audio_tasks
from morag.processors.audio import AudioConfig, AudioProcessingResult, AudioTranscriptSegment
from morag.services.chunking import ChunkInfo

class TestAudioTasks:
    """Test cases for audio processing tasks."""
    
    @pytest.fixture
    def mock_audio_file(self):
        """Create mock audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"mock audio data")
            return str(Path(f.name))
    
    @pytest.fixture
    def mock_audio_result(self):
        """Create mock audio processing result."""
        return AudioProcessingResult(
            text="Hello world, this is a test audio transcription.",
            language="en",
            confidence=0.9,
            duration=5.0,
            segments=[
                AudioTranscriptSegment("Hello world,", 0.0, 2.0, 0.95, language="en"),
                AudioTranscriptSegment("this is a test", 2.0, 4.0, 0.88, language="en"),
                AudioTranscriptSegment("audio transcription.", 4.0, 5.0, 0.87, language="en")
            ],
            metadata={"file_name": "test.wav", "sample_rate": 44100},
            processing_time=2.5,
            model_used="base"
        )
    
    @pytest.fixture
    def mock_text_chunks(self):
        """Create mock text chunks."""
        return [
            ChunkInfo(
                text="Hello world, this is a test",
                start_char=0,
                end_char=27,
                sentence_count=1,
                word_count=6,
                entities=[],
                topics=[],
                chunk_type="semantic"
            ),
            ChunkInfo(
                text="audio transcription.",
                start_char=28,
                end_char=48,
                sentence_count=1,
                word_count=2,
                entities=[],
                topics=[],
                chunk_type="semantic"
            )
        ]
    
    @pytest.mark.asyncio
    @patch('morag.tasks.audio_tasks.audio_processor')
    @patch('morag.tasks.audio_tasks.chunking_service')
    @patch('morag.tasks.audio_tasks.enhanced_summarization_service')
    @patch('morag.tasks.audio_tasks.gemini_service')
    @patch('morag.tasks.audio_tasks.qdrant_service')
    async def test_process_audio_file_success(
        self,
        mock_qdrant,
        mock_gemini,
        mock_summarization,
        mock_chunking,
        mock_audio_processor,
        mock_audio_file,
        mock_audio_result,
        mock_text_chunks
    ):
        """Test successful audio file processing."""
        # Mock task instance
        mock_task = Mock()
        mock_task.update_status = AsyncMock()
        
        # Mock audio processor
        mock_audio_processor.process_audio_file.return_value = mock_audio_result
        
        # Mock chunking service
        mock_chunking.chunk_text.return_value = mock_text_chunks
        
        # Mock enhanced summarization
        mock_summary_result = Mock()
        mock_summary_result.summary = "Test summary"
        mock_summary_result.strategy.value = "abstractive"
        mock_summary_result.quality.overall = 0.8
        mock_summary_result.processing_time = 0.5
        mock_summary_result.refinement_iterations = 1
        mock_summarization.generate_summary.return_value = mock_summary_result
        
        # Mock embedding generation
        mock_embedding_result = Mock()
        mock_embedding_result.embedding = [0.1, 0.2, 0.3]
        mock_embedding_result.model = "text-embedding-004"
        mock_embedding_result.token_count = 10
        mock_gemini.generate_embedding.return_value = mock_embedding_result
        
        # Mock vector storage
        mock_qdrant.store_chunk.return_value = "point_123"
        
        # Execute task
        result = await audio_tasks.process_audio_file(
            mock_task,
            mock_audio_file,
            "test_task_123",
            {"model_size": "base"},
            True
        )
        
        # Verify result
        assert result["task_id"] == "test_task_123"
        assert result["status"] == "completed"
        assert result["audio_result"]["text"] == mock_audio_result.text
        assert result["audio_result"]["language"] == "en"
        assert result["audio_result"]["confidence"] == 0.9
        assert result["chunks_processed"] == 2
        assert result["total_chunks"] == 2
        
        # Verify method calls
        mock_audio_processor.process_audio_file.assert_called_once()
        mock_chunking.chunk_text.assert_called_once()
        assert mock_summarization.generate_summary.call_count == 2  # One per chunk
        assert mock_gemini.generate_embedding.call_count == 2  # One per chunk
        assert mock_qdrant.store_chunk.call_count == 2  # One per chunk
        
        # Verify status updates
        assert mock_task.update_status.call_count >= 3  # At least processing stages + success
    
    @pytest.mark.asyncio
    @patch('morag.tasks.audio_tasks.audio_processor')
    async def test_process_audio_file_audio_processing_failure(
        self,
        mock_audio_processor,
        mock_audio_file
    ):
        """Test audio file processing with audio processing failure."""
        # Mock task instance
        mock_task = Mock()
        mock_task.update_status = AsyncMock()
        
        # Mock audio processor to raise exception
        mock_audio_processor.process_audio_file.side_effect = Exception("Audio processing failed")
        
        # Execute task and expect exception
        with pytest.raises(Exception, match="Audio processing failed"):
            await audio_tasks.process_audio_file(
                mock_task,
                mock_audio_file,
                "test_task_123"
            )
        
        # Verify failure status was set
        mock_task.update_status.assert_called_with("FAILURE", {"error": "Audio processing failed: Audio processing failed"})
    
    @pytest.mark.asyncio
    @patch('morag.tasks.audio_tasks.audio_processor')
    @patch('morag.tasks.audio_tasks.chunking_service')
    @patch('morag.tasks.audio_tasks.gemini_service')
    @patch('morag.tasks.audio_tasks.qdrant_service')
    async def test_process_audio_file_basic_summary(
        self,
        mock_qdrant,
        mock_gemini,
        mock_chunking,
        mock_audio_processor,
        mock_audio_file,
        mock_audio_result,
        mock_text_chunks
    ):
        """Test audio file processing with basic summarization."""
        # Mock task instance
        mock_task = Mock()
        mock_task.update_status = AsyncMock()
        
        # Mock audio processor
        mock_audio_processor.process_audio_file.return_value = mock_audio_result
        
        # Mock chunking service
        mock_chunking.chunk_text.return_value = mock_text_chunks
        
        # Mock basic summarization
        mock_summary_result = Mock()
        mock_summary_result.summary = "Basic summary"
        mock_gemini.generate_summary.return_value = mock_summary_result
        
        # Mock embedding generation
        mock_embedding_result = Mock()
        mock_embedding_result.embedding = [0.1, 0.2, 0.3]
        mock_embedding_result.model = "text-embedding-004"
        mock_embedding_result.token_count = 10
        mock_gemini.generate_embedding.return_value = mock_embedding_result
        
        # Mock vector storage
        mock_qdrant.store_chunk.return_value = "point_123"
        
        # Execute task with basic summary
        result = await audio_tasks.process_audio_file(
            mock_task,
            mock_audio_file,
            "test_task_123",
            None,
            False
        )
        
        # Verify result
        assert result["status"] == "completed"
        assert result["metadata"]["use_enhanced_summary"] is False
        
        # Verify basic summarization was used
        assert mock_gemini.generate_summary.call_count == 2  # One per chunk
    
    @pytest.mark.asyncio
    @patch('morag.tasks.audio_tasks.whisper_service')
    async def test_detect_audio_language_success(
        self,
        mock_whisper_service,
        mock_audio_file
    ):
        """Test successful audio language detection."""
        # Mock task instance
        mock_task = Mock()
        mock_task.update_status = AsyncMock()
        
        # Mock whisper service
        mock_language_result = {
            "language": "en",
            "language_probability": 0.95,
            "all_language_probs": {"en": 0.95, "es": 0.05}
        }
        mock_whisper_service.detect_language.return_value = mock_language_result
        
        # Execute task
        result = await audio_tasks.detect_audio_language(
            mock_task,
            mock_audio_file,
            "test_task_123",
            "base"
        )
        
        # Verify result
        assert result["task_id"] == "test_task_123"
        assert result["status"] == "completed"
        assert result["language"] == "en"
        assert result["language_probability"] == 0.95
        assert result["all_language_probs"] == {"en": 0.95, "es": 0.05}
        
        # Verify method calls
        mock_whisper_service.detect_language.assert_called_once_with(
            audio_path=mock_audio_file,
            model_size="base"
        )
        
        # Verify status updates
        mock_task.update_status.assert_called_with("SUCCESS", result)
    
    @pytest.mark.asyncio
    @patch('morag.tasks.audio_tasks.whisper_service')
    async def test_detect_audio_language_failure(
        self,
        mock_whisper_service,
        mock_audio_file
    ):
        """Test audio language detection failure."""
        # Mock task instance
        mock_task = Mock()
        mock_task.update_status = AsyncMock()
        
        # Mock whisper service to raise exception
        mock_whisper_service.detect_language.side_effect = Exception("Language detection failed")
        
        # Execute task and expect exception
        with pytest.raises(Exception, match="Language detection failed"):
            await audio_tasks.detect_audio_language(
                mock_task,
                mock_audio_file,
                "test_task_123"
            )
        
        # Verify failure status was set
        mock_task.update_status.assert_called_with("FAILURE", {"error": "Audio language detection failed: Language detection failed"})
    
    @pytest.mark.asyncio
    @patch('morag.tasks.audio_tasks.audio_processor')
    async def test_transcribe_audio_segments_success(
        self,
        mock_audio_processor,
        mock_audio_file,
        mock_audio_result
    ):
        """Test successful audio segment transcription."""
        # Mock task instance
        mock_task = Mock()
        mock_task.update_status = AsyncMock()
        
        # Mock audio processor
        mock_audio_processor.process_audio_file.return_value = mock_audio_result
        
        # Define segments to transcribe
        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 2.0, "end": 4.0}
        ]
        
        # Execute task
        result = await audio_tasks.transcribe_audio_segments(
            mock_task,
            mock_audio_file,
            "test_task_123",
            segments
        )
        
        # Verify result
        assert result["task_id"] == "test_task_123"
        assert result["status"] == "completed"
        assert result["segments_processed"] == 2
        assert len(result["segment_results"]) == 2
        
        # Verify segment results
        segment_result_1 = result["segment_results"][0]
        assert segment_result_1["segment_index"] == 0
        assert segment_result_1["start_time"] == 0.0
        assert segment_result_1["end_time"] == 2.0
        assert segment_result_1["text"] == "Hello world,"  # Should match first segment
        assert segment_result_1["language"] == "en"
        
        # Verify method calls
        assert mock_audio_processor.process_audio_file.call_count == 2  # One per segment
        
        # Verify status updates
        mock_task.update_status.assert_called_with("SUCCESS", result)
    
    @pytest.mark.asyncio
    @patch('morag.tasks.audio_tasks.audio_processor')
    async def test_transcribe_audio_segments_partial_failure(
        self,
        mock_audio_processor,
        mock_audio_file,
        mock_audio_result
    ):
        """Test audio segment transcription with partial failure."""
        # Mock task instance
        mock_task = Mock()
        mock_task.update_status = AsyncMock()
        
        # Mock audio processor - first call succeeds, second fails
        mock_audio_processor.process_audio_file.side_effect = [
            mock_audio_result,
            Exception("Segment processing failed")
        ]
        
        # Define segments to transcribe
        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 2.0, "end": 4.0}
        ]
        
        # Execute task
        result = await audio_tasks.transcribe_audio_segments(
            mock_task,
            mock_audio_file,
            "test_task_123",
            segments
        )
        
        # Verify result
        assert result["status"] == "completed"
        assert result["segments_processed"] == 2
        
        # First segment should succeed
        assert result["segment_results"][0]["text"] == "Hello world,"
        assert "error" not in result["segment_results"][0]
        
        # Second segment should have error
        assert result["segment_results"][1]["text"] == ""
        assert "error" in result["segment_results"][1]
        assert "Segment processing failed" in result["segment_results"][1]["error"]
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_segments_empty_segments(self, mock_audio_file):
        """Test audio segment transcription with empty segments list."""
        # Mock task instance
        mock_task = Mock()
        mock_task.update_status = AsyncMock()
        
        # Execute task with empty segments
        result = await audio_tasks.transcribe_audio_segments(
            mock_task,
            mock_audio_file,
            "test_task_123",
            []
        )
        
        # Verify result
        assert result["status"] == "completed"
        assert result["segments_processed"] == 0
        assert len(result["segment_results"]) == 0
