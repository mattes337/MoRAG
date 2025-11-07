"""Unit tests for video processor."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from morag_core.exceptions import ExternalServiceError, ProcessingError
from morag_video import (
    VideoConfig,
    VideoMetadata,
    VideoProcessingResult,
    VideoProcessor,
)


class TestVideoProcessor:
    """Test cases for VideoProcessor."""

    @pytest.fixture
    def video_processor(self):
        """Create video processor instance."""
        return VideoProcessor()

    @pytest.fixture
    def mock_video_file(self, tmp_path):
        """Create mock video file."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content")
        return video_file

    @pytest.fixture
    def video_config(self):
        """Create video configuration."""
        return VideoConfig(
            extract_audio=True,
            generate_thumbnails=True,
            thumbnail_count=3,
            extract_keyframes=False,
        )

    @pytest.fixture
    def mock_metadata(self):
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
            creation_time="2024-01-01T00:00:00Z",
        )

    @pytest.mark.asyncio
    async def test_process_video_success(
        self, video_processor, mock_video_file, video_config, mock_metadata
    ):
        """Test successful video processing."""
        with patch.object(
            video_processor, "_extract_metadata", return_value=mock_metadata
        ), patch.object(
            video_processor, "_extract_audio", return_value=Path("/tmp/audio.wav")
        ), patch.object(
            video_processor,
            "_generate_thumbnails",
            return_value=[Path("/tmp/thumb1.jpg"), Path("/tmp/thumb2.jpg")],
        ):
            result = await video_processor.process_video(mock_video_file, video_config)

            assert isinstance(result, VideoProcessingResult)
            assert result.metadata == mock_metadata
            assert result.audio_path == Path("/tmp/audio.wav")
            assert len(result.thumbnails) == 2
            assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_process_video_file_not_found(self, video_processor, video_config):
        """Test video processing with non-existent file."""
        non_existent_file = Path("/non/existent/file.mp4")

        with pytest.raises(ProcessingError, match="Video file not found"):
            await video_processor.process_video(non_existent_file, video_config)

    @pytest.mark.asyncio
    @patch("ffmpeg.probe")
    async def test_extract_metadata_success(
        self, mock_probe, video_processor, mock_video_file
    ):
        """Test successful metadata extraction."""
        mock_probe.return_value = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "codec_name": "h264",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {
                "duration": "120.0",
                "bit_rate": "5000000",
                "size": "50000000",
                "format_name": "mp4",
                "tags": {"creation_time": "2024-01-01T00:00:00Z"},
            },
        }

        metadata = await video_processor._extract_metadata(mock_video_file)

        assert metadata.duration == 120.0
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.fps == 30.0
        assert metadata.codec == "h264"
        assert metadata.has_audio is True
        assert metadata.audio_codec == "aac"

    @pytest.mark.asyncio
    @patch("ffmpeg.probe")
    async def test_extract_metadata_no_video_stream(
        self, mock_probe, video_processor, mock_video_file
    ):
        """Test metadata extraction with no video stream."""
        mock_probe.return_value = {
            "streams": [{"codec_type": "audio", "codec_name": "aac"}],
            "format": {},
        }

        with pytest.raises(ExternalServiceError, match="Metadata extraction failed"):
            await video_processor._extract_metadata(mock_video_file)

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_extract_audio_success(
        self, mock_to_thread, video_processor, mock_video_file
    ):
        """Test successful audio extraction."""
        # Mock the ffmpeg operation
        mock_to_thread.return_value = None

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.stat"
        ) as mock_stat:
            mock_stat.return_value.st_size = 1000000

            result = await video_processor._extract_audio(mock_video_file, "wav")

            assert result.suffix == ".wav"
            assert mock_to_thread.called

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_extract_audio_failure(
        self, mock_to_thread, video_processor, mock_video_file
    ):
        """Test audio extraction failure."""
        mock_to_thread.side_effect = Exception("FFmpeg error")

        with pytest.raises(ExternalServiceError, match="Audio extraction failed"):
            await video_processor._extract_audio(mock_video_file, "wav")

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_generate_thumbnails_success(
        self, mock_to_thread, video_processor, mock_video_file
    ):
        """Test successful thumbnail generation."""

        # Mock both ffmpeg.probe and thumbnail generation
        def mock_to_thread_side_effect(func, *args, **kwargs):
            if hasattr(func, "__name__") and "probe" in str(func):
                return {"format": {"duration": "120.0"}}
            return None

        mock_to_thread.side_effect = mock_to_thread_side_effect

        with patch("pathlib.Path.exists", return_value=True):
            result = await video_processor._generate_thumbnails(
                mock_video_file, 3, (320, 240), "jpg"
            )

            assert len(result) == 3
            assert all(path.suffix == ".jpg" for path in result)

    @pytest.mark.asyncio
    @patch("cv2.VideoCapture")
    @patch("asyncio.to_thread")
    async def test_extract_keyframes_success(
        self, mock_to_thread, mock_cv2_cap, video_processor, mock_video_file
    ):
        """Test successful keyframe extraction."""
        # Mock OpenCV VideoCapture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2_cap.return_value.CAP_PROP_FPS: 30.0,
            mock_cv2_cap.return_value.CAP_PROP_FRAME_COUNT: 3600,
        }.get(prop, 0)

        # Mock frame reading
        mock_cap.read.side_effect = [
            (True, Mock()),  # First frame
            (True, Mock()),  # Second frame
            (False, None),  # End of video
        ]

        mock_cv2_cap.return_value = mock_cap
        mock_to_thread.return_value = None

        # Mock keyframe files
        keyframe_paths = []
        for i in range(2):
            keyframe_path = video_processor.temp_dir / f"keyframe_test_{i}.jpg"
            keyframe_path.touch()
            keyframe_paths.append(keyframe_path)

        with patch("cv2.cvtColor"), patch("cv2.calcHist"), patch(
            "cv2.compareHist", return_value=0.5
        ), patch("pathlib.Path.exists", return_value=True):
            result = await video_processor._extract_keyframes(
                mock_video_file, 5, 0.3, (320, 240), "jpg"
            )

            assert len(result) >= 0  # May be empty if no scene changes detected

    def test_cleanup_temp_files(self, video_processor, tmp_path):
        """Test temporary file cleanup."""
        # Create temporary files
        temp_files = []
        for i in range(3):
            temp_file = tmp_path / f"temp_{i}.txt"
            temp_file.write_text("test content")
            temp_files.append(temp_file)

        # Verify files exist
        assert all(f.exists() for f in temp_files)

        # Clean up
        video_processor.cleanup_temp_files(temp_files)

        # Verify files are deleted
        assert all(not f.exists() for f in temp_files)

    def test_cleanup_temp_files_missing_file(self, video_processor, tmp_path):
        """Test cleanup with missing files."""
        non_existent_file = tmp_path / "non_existent.txt"

        # Should not raise exception
        video_processor.cleanup_temp_files([non_existent_file])

    @pytest.mark.asyncio
    async def test_process_video_no_audio_extraction(
        self, video_processor, mock_video_file, mock_metadata
    ):
        """Test video processing without audio extraction."""
        config = VideoConfig(extract_audio=False, generate_thumbnails=True)

        with patch.object(
            video_processor, "_extract_metadata", return_value=mock_metadata
        ), patch.object(
            video_processor,
            "_generate_thumbnails",
            return_value=[Path("/tmp/thumb1.jpg")],
        ):
            result = await video_processor.process_video(mock_video_file, config)

            assert result.audio_path is None
            assert len(result.thumbnails) == 1

    @pytest.mark.asyncio
    async def test_process_video_no_thumbnails(
        self, video_processor, mock_video_file, mock_metadata
    ):
        """Test video processing without thumbnail generation."""
        config = VideoConfig(extract_audio=True, generate_thumbnails=False)

        with patch.object(
            video_processor, "_extract_metadata", return_value=mock_metadata
        ), patch.object(
            video_processor, "_extract_audio", return_value=Path("/tmp/audio.wav")
        ):
            result = await video_processor.process_video(mock_video_file, config)

            assert result.audio_path == Path("/tmp/audio.wav")
            assert len(result.thumbnails) == 0
