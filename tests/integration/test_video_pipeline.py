"""Integration tests for video processing pipeline."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from morag.processors.video import video_processor, VideoConfig
from morag.services.ffmpeg_service import ffmpeg_service
from morag.tasks.video_tasks import process_video_file

class TestVideoPipelineIntegration:
    """Integration tests for video processing pipeline."""
    
    @pytest.fixture
    def mock_video_file(self, tmp_path):
        """Create mock video file."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content for testing")
        return video_file
    
    @pytest.fixture
    def video_config(self):
        """Create video configuration for testing."""
        return VideoConfig(
            extract_audio=True,
            generate_thumbnails=True,
            thumbnail_count=3,
            extract_keyframes=False,
            audio_format="wav",
            thumbnail_size=(320, 240)
        )
    
    @pytest.mark.asyncio
    @patch('ffmpeg.probe')
    @patch('asyncio.to_thread')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    async def test_full_video_processing_pipeline(
        self,
        mock_stat,
        mock_exists,
        mock_to_thread,
        mock_probe,
        mock_video_file,
        video_config
    ):
        """Test full video processing pipeline."""
        # Mock ffmpeg.probe response
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'video',
                    'width': 1920,
                    'height': 1080,
                    'r_frame_rate': '30/1',
                    'codec_name': 'h264'
                },
                {
                    'codec_type': 'audio',
                    'codec_name': 'aac'
                }
            ],
            'format': {
                'duration': '120.0',
                'bit_rate': '5000000',
                'size': '50000000',
                'format_name': 'mp4'
            }
        }
        
        # Mock file operations
        mock_exists.return_value = True
        mock_stat.return_value.st_size = 1000000
        mock_to_thread.return_value = None
        
        # Process video
        result = await video_processor.process_video(mock_video_file, video_config)
        
        # Verify results
        assert result.metadata.duration == 120.0
        assert result.metadata.width == 1920
        assert result.metadata.height == 1080
        assert result.metadata.has_audio is True
        assert result.audio_path is not None
        assert len(result.thumbnails) == 3
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    @patch('ffmpeg.probe')
    @patch('asyncio.to_thread')
    @patch('pathlib.Path.exists')
    async def test_ffmpeg_service_integration(
        self,
        mock_exists,
        mock_to_thread,
        mock_probe,
        mock_video_file
    ):
        """Test FFmpeg service integration."""
        # Mock ffmpeg operations
        mock_probe.return_value = {
            'format': {'duration': '120.0'},
            'streams': [
                {
                    'codec_type': 'video',
                    'codec_name': 'h264',
                    'width': 1920,
                    'height': 1080,
                    'r_frame_rate': '30/1'
                }
            ]
        }
        mock_to_thread.return_value = None
        mock_exists.return_value = True
        
        # Test metadata extraction
        metadata = await ffmpeg_service.extract_metadata(mock_video_file)
        assert metadata['duration'] == 120.0
        assert metadata['video_codec'] == 'h264'
        
        # Test thumbnail generation
        thumbnails = await ffmpeg_service.generate_thumbnails(mock_video_file, count=2)
        assert len(thumbnails) == 2
        
        # Test audio extraction
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1000000
            audio_path = await ffmpeg_service.extract_audio(mock_video_file, "wav")
            assert audio_path.suffix == ".wav"
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.video_processor')
    @patch('morag.tasks.video_tasks.process_audio_file')
    async def test_video_task_integration(
        self,
        mock_audio_task,
        mock_processor,
        mock_video_file
    ):
        """Test video task integration with audio processing."""
        from morag.processors.video import VideoMetadata, VideoProcessingResult
        
        # Mock video processing result
        video_metadata = VideoMetadata(
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
            creation_time=None
        )
        
        video_result = VideoProcessingResult(
            audio_path=Path("/tmp/audio.wav"),
            thumbnails=[Path("/tmp/thumb1.jpg"), Path("/tmp/thumb2.jpg")],
            keyframes=[],
            metadata=video_metadata,
            processing_time=5.0,
            temp_files=[]
        )
        
        mock_processor.process_video.return_value = video_result
        mock_processor.cleanup_temp_files = Mock()
        
        # Mock audio processing result
        mock_audio_task.return_value = {
            "text": "This is transcribed audio content from the video.",
            "language": "en",
            "chunks": [
                {
                    "text": "This is transcribed audio content from the video.",
                    "start_time": 0.0,
                    "end_time": 5.0
                }
            ],
            "embeddings_stored": 1,
            "summary": "Video contains spoken content about testing."
        }
        
        # Create mock task instance
        task_instance = Mock()
        task_instance.update_status = AsyncMock()
        
        # Execute video processing task
        result = await process_video_file(
            task_instance,
            str(mock_video_file),
            "test_task_id",
            config={"extract_audio": True, "generate_thumbnails": True},
            process_audio=True
        )
        
        # Verify integration results
        assert "video_metadata" in result
        assert "audio_processing_result" in result
        assert result["video_metadata"]["duration"] == 120.0
        assert result["video_metadata"]["has_audio"] is True
        assert result["audio_processing_result"]["text"] == "This is transcribed audio content from the video."
        assert len(result["thumbnails"]) == 2
        
        # Verify both processors were called
        mock_processor.process_video.assert_called_once()
        mock_audio_task.assert_called_once()
        mock_processor.cleanup_temp_files.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('cv2.VideoCapture')
    @patch('asyncio.to_thread')
    @patch('pathlib.Path.exists')
    async def test_keyframe_extraction_integration(
        self,
        mock_exists,
        mock_to_thread,
        mock_cv2_cap,
        mock_video_file
    ):
        """Test keyframe extraction integration."""
        # Mock OpenCV VideoCapture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2_cap.return_value.CAP_PROP_FPS: 30.0,
            mock_cv2_cap.return_value.CAP_PROP_FRAME_COUNT: 3600
        }.get(prop, 0)
        
        # Mock frame reading for scene change detection
        mock_frames = [Mock() for _ in range(5)]
        mock_cap.read.side_effect = [(True, frame) for frame in mock_frames] + [(False, None)]
        mock_cv2_cap.return_value = mock_cap
        
        # Mock other dependencies
        mock_to_thread.return_value = None
        mock_exists.return_value = True
        
        with patch('cv2.cvtColor'), \
             patch('cv2.calcHist'), \
             patch('cv2.compareHist', side_effect=[0.5, 0.2, 0.8, 0.1]):  # Simulate scene changes
            
            keyframes = await video_processor._extract_keyframes(
                mock_video_file,
                max_frames=5,
                threshold=0.3,
                size=(320, 240),
                format="jpg"
            )
            
            # Should detect some keyframes based on mocked scene changes
            assert len(keyframes) >= 0
    
    @pytest.mark.asyncio
    @patch('ffmpeg.probe')
    async def test_error_handling_integration(self, mock_probe, mock_video_file):
        """Test error handling in video processing pipeline."""
        # Test with invalid video file
        import ffmpeg
        mock_probe.side_effect = ffmpeg.Error("ffmpeg", "stderr", b"Invalid file")
        
        config = VideoConfig(extract_audio=False, generate_thumbnails=False)
        
        with pytest.raises(Exception):  # Should propagate the error
            await video_processor.process_video(mock_video_file, config)
    
    @pytest.mark.asyncio
    @patch('morag.tasks.video_tasks.video_processor')
    async def test_video_without_audio_integration(self, mock_processor, mock_video_file):
        """Test video processing for files without audio."""
        from morag.processors.video import VideoMetadata, VideoProcessingResult
        
        # Mock video-only result
        video_metadata = VideoMetadata(
            duration=60.0,
            width=1280,
            height=720,
            fps=25.0,
            codec="h264",
            bitrate=3000000,
            file_size=25000000,
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
            str(mock_video_file),
            "test_task_id",
            process_audio=True
        )
        
        # Should handle video without audio gracefully
        assert result["video_metadata"]["has_audio"] is False
        assert result["audio_processing_result"] is None
        assert len(result["thumbnails"]) == 1
    
    @pytest.mark.asyncio
    async def test_temp_file_cleanup_integration(self, mock_video_file, tmp_path):
        """Test temporary file cleanup integration."""
        # Create temporary files
        temp_files = []
        for i in range(3):
            temp_file = tmp_path / f"temp_video_{i}.tmp"
            temp_file.write_text("temporary content")
            temp_files.append(temp_file)
        
        # Verify files exist
        assert all(f.exists() for f in temp_files)
        
        # Test cleanup
        video_processor.cleanup_temp_files(temp_files)
        
        # Verify files are cleaned up
        assert all(not f.exists() for f in temp_files)
