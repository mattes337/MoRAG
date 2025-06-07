"""Unit tests for video tasks."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from morag_video.tasks import (
    process_video_file,
    extract_video_audio,
    generate_video_thumbnails
)
from morag_video import VideoConfig, VideoMetadata, VideoProcessingResult

class TestVideoTasks:
    """Test cases for video processing tasks."""
    
    @pytest.fixture
    def mock_video_metadata(self):
        """Create mock video metadata."""
        return VideoMetadata(
            duration=120.0,
            width=1920,
            height=1080,
            fps=30.0,
            codec="h264",
            bitrate=5000000,
            file_size=50000000,
            format="mp4",
            has_audio=True,
            audio_codec="aac",
            creation_time="2024-01-01T00:00:00Z"
        )
    
    @pytest.fixture
    def mock_video_result(self, mock_video_metadata):
        """Create mock video processing result."""
        return VideoProcessingResult(
            audio_path=Path("/tmp/audio.wav"),
            thumbnails=[Path("/tmp/thumb1.jpg"), Path("/tmp/thumb2.jpg")],
            keyframes=[Path("/tmp/key1.jpg")],
            metadata=mock_video_metadata,
            processing_time=5.0,
            temp_files=[Path("/tmp/audio.wav"), Path("/tmp/thumb1.jpg")]
        )
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.video_processor')
    @patch('morag.tasks.video_tasks.process_audio_file')
    async def test_process_video_file_success(self, mock_audio_task, mock_processor, mock_video_result):
        """Test successful video file processing."""
        # Mock video processor
        mock_processor.process_video.return_value = mock_video_result
        mock_processor.cleanup_temp_files = Mock()
        
        # Mock audio processing task
        mock_audio_task.return_value = {
            "text": "Transcribed audio content",
            "chunks": ["chunk1", "chunk2"],
            "embeddings_stored": 2
        }
        
        # Create mock task instance
        task_instance = Mock()
        task_instance.update_status = AsyncMock()
        
        # Execute task
        result = await process_video_file(
            task_instance,
            "/path/to/video.mp4",
            "test_task_id",
            config={"extract_audio": True, "generate_thumbnails": True},
            process_audio=True
        )
        
        # Verify results
        assert "video_metadata" in result
        assert result["video_metadata"]["duration"] == 120.0
        assert result["video_metadata"]["has_audio"] is True
        assert len(result["thumbnails"]) == 2
        assert len(result["keyframes"]) == 1
        assert result["audio_processing_result"] is not None
        
        # Verify processor was called
        mock_processor.process_video.assert_called_once()
        mock_processor.cleanup_temp_files.assert_called_once()
        
        # Verify audio task was called
        mock_audio_task.assert_called_once()
        
        # Verify status updates
        task_instance.update_status.assert_called()
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.video_processor')
    async def test_process_video_file_no_audio_processing(self, mock_processor, mock_video_result):
        """Test video processing without audio processing."""
        mock_processor.process_video.return_value = mock_video_result
        mock_processor.cleanup_temp_files = Mock()
        
        task_instance = Mock()
        task_instance.update_status = AsyncMock()
        
        result = await process_video_file(
            task_instance,
            "/path/to/video.mp4",
            "test_task_id",
            process_audio=False
        )
        
        assert result["audio_processing_result"] is None
        mock_processor.cleanup_temp_files.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.video_processor')
    async def test_process_video_file_no_audio_track(self, mock_processor):
        """Test video processing with no audio track."""
        # Create video result without audio
        video_metadata = VideoMetadata(
            duration=120.0,
            width=1920,
            height=1080,
            fps=30.0,
            codec="h264",
            bitrate=5000000,
            file_size=50000000,
            format="mp4",
            has_audio=False,
            audio_codec=None,
            creation_time=None
        )
        
        video_result = VideoProcessingResult(
            audio_path=None,
            thumbnails=[Path("/tmp/thumb1.jpg")],
            keyframes=[],
            metadata=video_metadata,
            processing_time=3.0,
            temp_files=[]
        )
        
        mock_processor.process_video.return_value = video_result
        mock_processor.cleanup_temp_files = Mock()
        
        task_instance = Mock()
        task_instance.update_status = AsyncMock()
        
        result = await process_video_file(
            task_instance,
            "/path/to/video.mp4",
            "test_task_id",
            process_audio=True
        )
        
        assert result["audio_processing_result"] is None
        assert result["video_metadata"]["has_audio"] is False
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.video_processor')
    async def test_process_video_file_failure(self, mock_processor):
        """Test video processing failure."""
        mock_processor.process_video.side_effect = Exception("Processing failed")
        
        task_instance = Mock()
        task_instance.update_status = AsyncMock()
        
        with pytest.raises(Exception, match="Processing failed"):
            await process_video_file(
                task_instance,
                "/path/to/video.mp4",
                "test_task_id"
            )
        
        # Verify failure status was set
        task_instance.update_status.assert_called_with(
            "FAILURE", 
            {"error": "Video processing failed: Processing failed"}
        )
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.ffmpeg_service')
    async def test_extract_video_audio_success(self, mock_service):
        """Test successful video audio extraction."""
        mock_audio_path = Path("/tmp/extracted_audio.wav")
        mock_service.extract_audio.return_value = mock_audio_path
        
        # Mock file stats
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1000000
            
            task_instance = Mock()
            task_instance.update_status = AsyncMock()
            
            result = await extract_video_audio(
                task_instance,
                "/path/to/video.mp4",
                "test_task_id",
                "wav"
            )
            
            assert result["audio_path"] == str(mock_audio_path)
            assert result["audio_format"] == "wav"
            assert result["file_size"] == 1000000
            
            mock_service.extract_audio.assert_called_once_with(
                Path("/path/to/video.mp4"),
                output_format="wav"
            )
            
            task_instance.update_status.assert_called()
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.ffmpeg_service')
    async def test_extract_video_audio_failure(self, mock_service):
        """Test video audio extraction failure."""
        mock_service.extract_audio.side_effect = Exception("Extraction failed")
        
        task_instance = Mock()
        task_instance.update_status = AsyncMock()
        
        with pytest.raises(Exception, match="Extraction failed"):
            await extract_video_audio(
                task_instance,
                "/path/to/video.mp4",
                "test_task_id"
            )
        
        task_instance.update_status.assert_called_with(
            "FAILURE",
            {"error": "Video audio extraction failed: Extraction failed"}
        )
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.ffmpeg_service')
    async def test_generate_video_thumbnails_success(self, mock_service):
        """Test successful video thumbnail generation."""
        mock_thumbnails = [
            Path("/tmp/thumb1.jpg"),
            Path("/tmp/thumb2.jpg"),
            Path("/tmp/thumb3.jpg")
        ]
        mock_service.generate_thumbnails.return_value = mock_thumbnails
        
        task_instance = Mock()
        task_instance.update_status = AsyncMock()
        
        result = await generate_video_thumbnails(
            task_instance,
            "/path/to/video.mp4",
            "test_task_id",
            count=3,
            size=[640, 480]
        )
        
        assert len(result["thumbnails"]) == 3
        assert result["count"] == 3
        assert result["size"] == [640, 480]
        
        mock_service.generate_thumbnails.assert_called_once_with(
            Path("/path/to/video.mp4"),
            count=3,
            size=(640, 480)
        )
        
        task_instance.update_status.assert_called()
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.ffmpeg_service')
    async def test_generate_video_thumbnails_default_size(self, mock_service):
        """Test thumbnail generation with default size."""
        mock_thumbnails = [Path("/tmp/thumb1.jpg")]
        mock_service.generate_thumbnails.return_value = mock_thumbnails
        
        task_instance = Mock()
        task_instance.update_status = AsyncMock()
        
        result = await generate_video_thumbnails(
            task_instance,
            "/path/to/video.mp4",
            "test_task_id",
            count=1
        )
        
        assert result["size"] == [320, 240]  # Default size
        
        mock_service.generate_thumbnails.assert_called_once_with(
            Path("/path/to/video.mp4"),
            count=1,
            size=(320, 240)
        )
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.ffmpeg_service')
    async def test_generate_video_thumbnails_failure(self, mock_service):
        """Test video thumbnail generation failure."""
        mock_service.generate_thumbnails.side_effect = Exception("Thumbnail generation failed")
        
        task_instance = Mock()
        task_instance.update_status = AsyncMock()
        
        with pytest.raises(Exception, match="Thumbnail generation failed"):
            await generate_video_thumbnails(
                task_instance,
                "/path/to/video.mp4",
                "test_task_id"
            )
        
        task_instance.update_status.assert_called_with(
            "FAILURE",
            {"error": "Video thumbnail generation failed: Thumbnail generation failed"}
        )
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.video_processor')
    async def test_process_video_file_config_parsing(self, mock_processor, mock_video_result):
        """Test video configuration parsing."""
        mock_processor.process_video.return_value = mock_video_result
        mock_processor.cleanup_temp_files = Mock()
        
        task_instance = Mock()
        task_instance.update_status = AsyncMock()
        
        config = {
            "extract_audio": False,
            "generate_thumbnails": True,
            "thumbnail_count": 10,
            "extract_keyframes": True,
            "max_keyframes": 15
        }
        
        await process_video_file(
            task_instance,
            "/path/to/video.mp4",
            "test_task_id",
            config=config,
            process_audio=False
        )
        
        # Verify processor was called with correct config
        call_args = mock_processor.process_video.call_args
        video_config = call_args[0][1]  # Second argument is the config
        
        assert video_config.extract_audio is False
        assert video_config.generate_thumbnails is True
        assert video_config.thumbnail_count == 10
        assert video_config.extract_keyframes is True
        assert video_config.max_keyframes == 15
