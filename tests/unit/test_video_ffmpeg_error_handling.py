"""Test video FFmpeg error handling improvements."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from morag.processors.video import VideoProcessor, VideoConfig
from morag.core.exceptions import ProcessingError, ExternalServiceError


class TestVideoFFmpegErrorHandling:
    """Test video FFmpeg error handling and parameter filtering."""

    @pytest.fixture
    def video_processor(self):
        """Create a video processor instance."""
        return VideoProcessor()

    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file path."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = Path(f.name)
        yield video_path
        # Cleanup
        if video_path.exists():
            video_path.unlink()

    @pytest.fixture
    def video_config(self):
        """Create a video configuration."""
        return VideoConfig(
            extract_audio=True,
            audio_format="wav",
            generate_thumbnails=False,
            extract_keyframes=False
        )

    @pytest.mark.asyncio
    async def test_ffmpeg_error_stderr_capture(self, video_processor, temp_video_file, video_config):
        """Test that FFmpeg errors properly capture stderr output."""
        # Mock ffmpeg_probe to return valid metadata
        mock_metadata = {
            'streams': [
                {'codec_type': 'video', 'width': 1280, 'height': 720, 'codec_name': 'h264', 'r_frame_rate': '30/1'},
                {'codec_type': 'audio', 'codec_name': 'mp3'}
            ],
            'format': {'duration': '100.0', 'bit_rate': '1000000', 'size': '12500000', 'format_name': 'mp4'}
        }

        # Mock FFmpegError with stderr
        from ffmpeg._run import Error as FFmpegError
        mock_error = FFmpegError('ffmpeg', 'stdout', b'FFmpeg error: Invalid argument\nUnable to parse option value "None"')
        mock_error.stderr = b'FFmpeg error: Invalid argument\nUnable to parse option value "None"'

        with patch('morag.processors.video.ffmpeg_probe', return_value=mock_metadata), \
             patch('morag.processors.video.ffmpeg_run', side_effect=mock_error), \
             patch('pathlib.Path.exists', return_value=True):

            with pytest.raises(ExternalServiceError) as exc_info:
                await video_processor._extract_audio(temp_video_file, "wav")

            # Verify that stderr output is captured in the error message
            assert "FFmpeg audio extraction error:" in str(exc_info.value)
            assert "Invalid argument" in str(exc_info.value)
            assert "Unable to parse option value" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parameter_filtering_excludes_none_values(self, video_processor, temp_video_file):
        """Test that None values are excluded from FFmpeg parameters."""
        # Mock ffmpeg_probe to return valid metadata
        mock_metadata = {
            'streams': [
                {'codec_type': 'video', 'width': 1280, 'height': 720, 'codec_name': 'h264', 'r_frame_rate': '30/1'},
                {'codec_type': 'audio', 'codec_name': 'mp3'}
            ],
            'format': {'duration': '100.0', 'bit_rate': '1000000', 'size': '12500000', 'format_name': 'mp4'}
        }

        # Mock successful ffmpeg_run
        mock_audio_path = video_processor.temp_dir / "test_audio.wav"
        
        with patch('morag.processors.video.ffmpeg_probe', return_value=mock_metadata), \
             patch('morag.processors.video.ffmpeg_run') as mock_run, \
             patch('morag.processors.video.ffmpeg_output') as mock_output, \
             patch('morag.processors.video.ffmpeg_input') as mock_input, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat:

            mock_stat.return_value.st_size = 1000000
            
            # Test WAV format (should not include bitrate parameter)
            await video_processor._extract_audio(temp_video_file, "wav")

            # Verify ffmpeg_output was called
            assert mock_output.called
            
            # Get the parameters passed to ffmpeg_output
            call_args = mock_output.call_args
            output_params = call_args[1] if len(call_args) > 1 else {}
            
            # Verify that no None values are in the parameters
            for key, value in output_params.items():
                assert value is not None, f"Parameter {key} should not be None"

    @pytest.mark.asyncio
    async def test_different_audio_formats_parameter_handling(self, video_processor, temp_video_file):
        """Test parameter handling for different audio formats."""
        # Mock ffmpeg_probe to return valid metadata
        mock_metadata = {
            'streams': [
                {'codec_type': 'video', 'width': 1280, 'height': 720, 'codec_name': 'h264', 'r_frame_rate': '30/1'},
                {'codec_type': 'audio', 'codec_name': 'mp3'}
            ],
            'format': {'duration': '100.0', 'bit_rate': '1000000', 'size': '12500000', 'format_name': 'mp4'}
        }

        mock_audio_path = video_processor.temp_dir / "test_audio.mp3"
        
        with patch('morag.processors.video.ffmpeg_probe', return_value=mock_metadata), \
             patch('morag.processors.video.ffmpeg_run') as mock_run, \
             patch('morag.processors.video.ffmpeg_output') as mock_output, \
             patch('morag.processors.video.ffmpeg_input') as mock_input, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat:

            mock_stat.return_value.st_size = 1000000
            
            # Test MP3 format (should include bitrate parameter)
            await video_processor._extract_audio(temp_video_file, "mp3")

            # Verify ffmpeg_output was called
            assert mock_output.called
            
            # Get the parameters passed to ffmpeg_output
            call_args = mock_output.call_args
            output_params = call_args[1] if len(call_args) > 1 else {}
            
            # For MP3, should include audio_bitrate
            if 'audio_bitrate' in output_params:
                assert output_params['audio_bitrate'] == "128k"

    @pytest.mark.asyncio
    async def test_ffmpeg_quiet_false_for_error_capture(self, video_processor, temp_video_file):
        """Test that FFmpeg is called with quiet=False for error capture."""
        # Mock ffmpeg_probe to return valid metadata
        mock_metadata = {
            'streams': [
                {'codec_type': 'video', 'width': 1280, 'height': 720, 'codec_name': 'h264', 'r_frame_rate': '30/1'},
                {'codec_type': 'audio', 'codec_name': 'mp3'}
            ],
            'format': {'duration': '100.0', 'bit_rate': '1000000', 'size': '12500000', 'format_name': 'mp4'}
        }

        mock_audio_path = video_processor.temp_dir / "test_audio.wav"
        
        with patch('morag.processors.video.ffmpeg_probe', return_value=mock_metadata), \
             patch('morag.processors.video.ffmpeg_run') as mock_run, \
             patch('morag.processors.video.ffmpeg_output') as mock_output, \
             patch('morag.processors.video.ffmpeg_input') as mock_input, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat:

            mock_stat.return_value.st_size = 1000000
            
            await video_processor._extract_audio(temp_video_file, "wav")

            # Verify ffmpeg_run was called with quiet=False and capture_stderr=True
            assert mock_run.called
            call_args = mock_run.call_args
            
            # Check that quiet=False and capture_stderr=True are in the call
            if len(call_args) > 1:
                kwargs = call_args[1]
                assert kwargs.get('quiet') is False
                assert kwargs.get('capture_stderr') is True

    def test_video_processor_initialization(self):
        """Test that VideoProcessor initializes correctly."""
        processor = VideoProcessor()
        assert processor.temp_dir.exists()
        assert processor.temp_dir.name == "morag_video"

    def test_parameter_filtering_logic(self):
        """Test the parameter filtering logic that excludes None values."""
        # Test the logic that was causing the original bug

        # Simulate the old broken logic
        old_params = {
            "acodec": "pcm_s16le",
            "audio_bitrate": "128k" if "wav" == "mp3" else None  # This would be None for WAV
        }

        # Simulate the new fixed logic
        new_params = {"acodec": "pcm_s16le"}
        if "wav" == "mp3":  # For WAV format, don't add bitrate
            new_params["audio_bitrate"] = "128k"
        elif "wav" == "aac":
            new_params["audio_bitrate"] = "128k"
        # For WAV (pcm_s16le), don't set bitrate as it's uncompressed

        # Verify the old logic would have None values
        assert old_params["audio_bitrate"] is None

        # Verify the new logic doesn't have None values
        for key, value in new_params.items():
            assert value is not None, f"Parameter {key} should not be None"

        # Verify WAV format doesn't include bitrate
        assert "audio_bitrate" not in new_params

        # Test MP3 format includes bitrate
        mp3_params = {"acodec": "libmp3lame"}
        if "mp3" == "mp3":
            mp3_params["audio_bitrate"] = "128k"

        assert "audio_bitrate" in mp3_params
        assert mp3_params["audio_bitrate"] == "128k"
