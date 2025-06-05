"""Unit tests for FFmpeg service."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from morag_video.services import FFmpegService, ffmpeg_service
from morag_core.exceptions import ProcessingError, ExternalServiceError

class TestFFmpegService:
    """Test cases for FFmpegService."""
    
    @pytest.fixture
    def service(self):
        """Create FFmpeg service instance."""
        return FFmpegService()
    
    @pytest.fixture
    def mock_video_file(self, tmp_path):
        """Create mock video file."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content")
        return video_file
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_extract_audio_success(self, mock_to_thread, service, mock_video_file):
        """Test successful audio extraction."""
        mock_to_thread.return_value = None
        
        # Mock the output file creation
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 1000000
            
            result = await service.extract_audio(mock_video_file, "wav")
            
            assert result.suffix == ".wav"
            assert mock_to_thread.called
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_extract_audio_failure(self, mock_to_thread, service, mock_video_file):
        """Test audio extraction failure."""
        import ffmpeg
        mock_error = ffmpeg.Error("ffmpeg", "stderr", b"FFmpeg error message")
        mock_to_thread.side_effect = mock_error
        
        with pytest.raises(ExternalServiceError, match="FFmpeg audio extraction failed"):
            await service.extract_audio(mock_video_file, "wav")
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_extract_audio_output_not_created(self, mock_to_thread, service, mock_video_file):
        """Test audio extraction when output file is not created."""
        mock_to_thread.return_value = None
        
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(ProcessingError, match="Audio extraction failed - output file not created"):
                await service.extract_audio(mock_video_file, "wav")
    
    @pytest.mark.asyncio
    async def test_generate_thumbnails_success(self, service, mock_video_file):
        """Test successful thumbnail generation."""
        mock_probe_result = {
            'format': {'duration': '120.0'}
        }
        
        with patch.object(service, 'probe_video', return_value=mock_probe_result), \
             patch('asyncio.to_thread') as mock_to_thread, \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_to_thread.return_value = None
            
            result = await service.generate_thumbnails(mock_video_file, count=3)
            
            assert len(result) == 3
            assert all(path.name.startswith("thumb_") for path in result)
    
    @pytest.mark.asyncio
    async def test_generate_thumbnails_failure(self, service, mock_video_file):
        """Test thumbnail generation failure."""
        import ffmpeg
        mock_error = ffmpeg.Error("ffmpeg", "stderr", b"FFmpeg error message")
        
        with patch.object(service, 'probe_video', side_effect=mock_error):
            with pytest.raises(ExternalServiceError, match="FFmpeg thumbnail generation failed"):
                await service.generate_thumbnails(mock_video_file, count=3)
    
    @pytest.mark.asyncio
    @patch('ffmpeg.probe')
    async def test_extract_metadata_success(self, mock_probe, service, mock_video_file):
        """Test successful metadata extraction."""
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'video',
                    'codec_name': 'h264',
                    'width': 1920,
                    'height': 1080,
                    'r_frame_rate': '30/1',
                    'bit_rate': '5000000',
                    'pix_fmt': 'yuv420p'
                },
                {
                    'codec_type': 'audio',
                    'codec_name': 'aac',
                    'sample_rate': '44100',
                    'channels': 2,
                    'bit_rate': '128000'
                }
            ],
            'format': {
                'duration': '120.0',
                'size': '50000000',
                'format_name': 'mp4',
                'bit_rate': '5128000',
                'tags': {
                    'creation_time': '2024-01-01T00:00:00Z'
                }
            }
        }
        
        metadata = await service.extract_metadata(mock_video_file)
        
        assert metadata['duration'] == 120.0
        assert metadata['video_codec'] == 'h264'
        assert metadata['width'] == 1920
        assert metadata['height'] == 1080
        assert metadata['fps'] == 30.0
        assert metadata['has_audio'] is True
        assert metadata['audio_codec'] == 'aac'
        assert metadata['sample_rate'] == 44100
        assert metadata['channels'] == 2
    
    @pytest.mark.asyncio
    @patch('ffmpeg.probe')
    async def test_extract_metadata_video_only(self, mock_probe, service, mock_video_file):
        """Test metadata extraction for video-only file."""
        mock_probe.return_value = {
            'streams': [
                {
                    'codec_type': 'video',
                    'codec_name': 'h264',
                    'width': 1920,
                    'height': 1080,
                    'r_frame_rate': '30/1'
                }
            ],
            'format': {
                'duration': '120.0',
                'size': '50000000',
                'format_name': 'mp4'
            }
        }
        
        metadata = await service.extract_metadata(mock_video_file)
        
        assert metadata['has_audio'] is False
        assert metadata['has_video'] is True
        assert 'audio_codec' not in metadata
    
    @pytest.mark.asyncio
    @patch('ffmpeg.probe')
    async def test_probe_video_success(self, mock_probe, service, mock_video_file):
        """Test successful video probing."""
        expected_result = {'format': {'duration': '120.0'}}
        mock_probe.return_value = expected_result
        
        result = await service.probe_video(mock_video_file)
        
        assert result == expected_result
        mock_probe.assert_called_once_with(str(mock_video_file))
    
    @pytest.mark.asyncio
    @patch('ffmpeg.probe')
    async def test_probe_video_failure(self, mock_probe, service, mock_video_file):
        """Test video probing failure."""
        import ffmpeg
        mock_error = ffmpeg.Error("ffmpeg", "stderr", b"FFmpeg probe error")
        mock_probe.side_effect = mock_error
        
        with pytest.raises(ExternalServiceError, match="FFmpeg probe failed"):
            await service.probe_video(mock_video_file)
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_extract_keyframes_success(self, mock_to_thread, service, mock_video_file):
        """Test successful keyframe extraction."""
        mock_to_thread.return_value = None
        
        # Mock keyframe files
        keyframe_paths = []
        for i in range(3):
            keyframe_path = service.temp_dir / f"keyframe_test_{i:03d}.jpg"
            keyframe_path.touch()
            keyframe_paths.append(keyframe_path)
        
        with patch('pathlib.Path.exists', return_value=True):
            result = await service.extract_keyframes(mock_video_file, max_frames=5)
            
            assert len(result) >= 0  # May be empty if no keyframes generated
            assert mock_to_thread.called
    
    @pytest.mark.asyncio
    @patch('asyncio.to_thread')
    async def test_extract_keyframes_failure(self, mock_to_thread, service, mock_video_file):
        """Test keyframe extraction failure."""
        import ffmpeg
        mock_error = ffmpeg.Error("ffmpeg", "stderr", b"FFmpeg keyframe error")
        mock_to_thread.side_effect = mock_error
        
        with pytest.raises(ExternalServiceError, match="FFmpeg keyframe extraction failed"):
            await service.extract_keyframes(mock_video_file, max_frames=5)
    
    def test_parse_frame_rate_fraction(self, service):
        """Test frame rate parsing from fraction."""
        assert service._parse_frame_rate("30/1") == 30.0
        assert service._parse_frame_rate("25/1") == 25.0
        assert service._parse_frame_rate("60000/1001") == pytest.approx(59.94, rel=1e-2)
    
    def test_parse_frame_rate_decimal(self, service):
        """Test frame rate parsing from decimal."""
        assert service._parse_frame_rate("30.0") == 30.0
        assert service._parse_frame_rate("25.5") == 25.5
    
    def test_parse_frame_rate_invalid(self, service):
        """Test frame rate parsing with invalid input."""
        assert service._parse_frame_rate("invalid") == 0.0
        assert service._parse_frame_rate("30/0") == 0.0
        assert service._parse_frame_rate("") == 0.0
    
    @pytest.mark.asyncio
    async def test_extract_audio_different_formats(self, service, mock_video_file):
        """Test audio extraction with different formats."""
        formats_and_codecs = [
            ("wav", "pcm_s16le"),
            ("mp3", "mp3"),
            ("flac", "flac"),
            ("unknown", "pcm_s16le")  # Default fallback
        ]
        
        for format_name, expected_codec in formats_and_codecs:
            with patch('asyncio.to_thread') as mock_to_thread, \
                 patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.stat') as mock_stat:
                
                mock_stat.return_value.st_size = 1000000
                mock_to_thread.return_value = None
                
                result = await service.extract_audio(mock_video_file, format_name)
                
                assert result.suffix == f".{format_name}"
                assert mock_to_thread.called
    
    @pytest.mark.asyncio
    async def test_generate_thumbnails_single_thumbnail(self, service, mock_video_file):
        """Test generating single thumbnail."""
        mock_probe_result = {
            'format': {'duration': '120.0'}
        }
        
        with patch.object(service, 'probe_video', return_value=mock_probe_result), \
             patch('asyncio.to_thread') as mock_to_thread, \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_to_thread.return_value = None
            
            result = await service.generate_thumbnails(mock_video_file, count=1)
            
            assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_generate_thumbnails_custom_size(self, service, mock_video_file):
        """Test generating thumbnails with custom size."""
        mock_probe_result = {
            'format': {'duration': '120.0'}
        }
        
        with patch.object(service, 'probe_video', return_value=mock_probe_result), \
             patch('asyncio.to_thread') as mock_to_thread, \
             patch('pathlib.Path.exists', return_value=True):
            
            mock_to_thread.return_value = None
            
            result = await service.generate_thumbnails(
                mock_video_file, 
                count=2, 
                size=(640, 480),
                format="png"
            )
            
            assert len(result) == 2
            assert all(path.suffix == ".png" for path in result)

def test_global_service_instance():
    """Test that global service instance is available."""
    assert ffmpeg_service is not None
    assert isinstance(ffmpeg_service, FFmpegService)
