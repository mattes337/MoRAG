"""Unit tests for YouTube processor."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile

from morag_youtube import YouTubeProcessor, YouTubeConfig, YouTubeMetadata, YouTubeDownloadResult
from morag_core.exceptions import ProcessingError, ExternalServiceError

class TestYouTubeProcessor:
    """Test cases for YouTubeProcessor."""
    
    @pytest.fixture
    def youtube_processor(self):
        """Create YouTube processor instance."""
        return YouTubeProcessor()
    
    @pytest.fixture
    def youtube_config(self):
        """Create YouTube configuration."""
        return YouTubeConfig(
            quality="best",
            format_preference="mp4",
            extract_audio=False,
            download_subtitles=True,
            max_filesize="100M"
        )
    
    @pytest.fixture
    def mock_metadata(self):
        """Create mock YouTube metadata."""
        return YouTubeMetadata(
            id="dQw4w9WgXcQ",
            title="Test Video",
            description="Test description",
            uploader="Test Uploader",
            upload_date="20230101",
            duration=180.0,
            view_count=1000000,
            like_count=50000,
            comment_count=1000,
            tags=["test", "video"],
            categories=["Music"],
            thumbnail_url="https://example.com/thumb.jpg",
            webpage_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
            channel_id="UCtest",
            channel_url="https://youtube.com/channel/UCtest",
            playlist_id=None,
            playlist_title=None,
            playlist_index=None
        )
    
    @pytest.fixture
    def mock_yt_dlp_info(self):
        """Create mock yt-dlp info dictionary."""
        return {
            'id': 'dQw4w9WgXcQ',
            'title': 'Test Video',
            'description': 'Test description',
            'uploader': 'Test Uploader',
            'upload_date': '20230101',
            'duration': 180,
            'view_count': 1000000,
            'like_count': 50000,
            'comment_count': 1000,
            'tags': ['test', 'video'],
            'categories': ['Music'],
            'thumbnail': 'https://example.com/thumb.jpg',
            'webpage_url': 'https://youtube.com/watch?v=dQw4w9WgXcQ',
            'channel_id': 'UCtest',
            'channel_url': 'https://youtube.com/channel/UCtest'
        }
    
    @pytest.mark.asyncio
    async def test_process_url_metadata_only(self, youtube_processor, youtube_config, mock_metadata):
        """Test processing URL for metadata only."""
        youtube_config.extract_metadata_only = True
        
        with patch.object(youtube_processor, '_extract_metadata_only', return_value=mock_metadata):
            result = await youtube_processor.process_url("https://youtube.com/watch?v=test", youtube_config)
            
            assert isinstance(result, YouTubeDownloadResult)
            assert result.metadata == mock_metadata
            assert result.video_path is None
            assert result.audio_path is None
            assert len(result.subtitle_paths) == 0
            assert len(result.thumbnail_paths) == 0
            assert result.download_time > 0
    
    @pytest.mark.asyncio
    async def test_process_url_with_video_download(self, youtube_processor, youtube_config, mock_metadata, tmp_path):
        """Test processing URL with video download."""
        youtube_config.extract_audio = False
        mock_video_path = tmp_path / "test_video.mp4"
        mock_video_path.write_text("mock video content")
        
        with patch.object(youtube_processor, '_extract_metadata_only', return_value=mock_metadata), \
             patch.object(youtube_processor, '_download_video', return_value=mock_video_path), \
             patch.object(youtube_processor, '_download_subtitles', return_value=[]), \
             patch.object(youtube_processor, '_download_thumbnails', return_value=[]):
            
            result = await youtube_processor.process_url("https://youtube.com/watch?v=test", youtube_config)
            
            assert result.video_path == mock_video_path
            assert result.audio_path is None
            assert result.file_size > 0
    
    @pytest.mark.asyncio
    async def test_process_url_with_audio_download(self, youtube_processor, youtube_config, mock_metadata, tmp_path):
        """Test processing URL with audio download."""
        youtube_config.extract_audio = True
        mock_audio_path = tmp_path / "test_audio.mp3"
        mock_audio_path.write_text("mock audio content")
        
        with patch.object(youtube_processor, '_extract_metadata_only', return_value=mock_metadata), \
             patch.object(youtube_processor, '_download_audio', return_value=mock_audio_path), \
             patch.object(youtube_processor, '_download_subtitles', return_value=[]), \
             patch.object(youtube_processor, '_download_thumbnails', return_value=[]):
            
            result = await youtube_processor.process_url("https://youtube.com/watch?v=test", youtube_config)
            
            assert result.audio_path == mock_audio_path
            assert result.video_path is None
            assert result.file_size > 0
    
    @pytest.mark.asyncio
    @patch('yt_dlp.YoutubeDL')
    async def test_extract_metadata_only_success(self, mock_yt_dlp, youtube_processor, mock_yt_dlp_info):
        """Test successful metadata extraction."""
        mock_ydl = Mock()
        mock_ydl.extract_info.return_value = mock_yt_dlp_info
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        with patch('asyncio.to_thread', return_value=mock_yt_dlp_info):
            metadata = await youtube_processor._extract_metadata_only("https://youtube.com/watch?v=test")
            
            assert metadata.id == "dQw4w9WgXcQ"
            assert metadata.title == "Test Video"
            assert metadata.uploader == "Test Uploader"
            assert metadata.duration == 180.0
            assert metadata.view_count == 1000000
    
    @pytest.mark.asyncio
    @patch('yt_dlp.YoutubeDL')
    async def test_extract_metadata_only_failure(self, mock_yt_dlp, youtube_processor):
        """Test metadata extraction failure."""
        mock_yt_dlp.side_effect = Exception("Network error")
        
        with pytest.raises(ExternalServiceError, match="YouTube metadata extraction failed"):
            await youtube_processor._extract_metadata_only("https://youtube.com/watch?v=test")
    
    @pytest.mark.asyncio
    @patch('yt_dlp.YoutubeDL')
    async def test_download_video_success(self, mock_yt_dlp, youtube_processor, youtube_config, tmp_path):
        """Test successful video download."""
        mock_ydl = Mock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        # Create a mock downloaded file
        mock_video_file = tmp_path / "video_123_test.mp4"
        mock_video_file.write_text("mock video")
        
        with patch('asyncio.to_thread'), \
             patch('pathlib.Path.glob', return_value=[mock_video_file]):
            
            result = await youtube_processor._download_video("https://youtube.com/watch?v=test", youtube_config)
            
            assert result == mock_video_file
    
    @pytest.mark.asyncio
    @patch('yt_dlp.YoutubeDL')
    async def test_download_video_failure(self, mock_yt_dlp, youtube_processor, youtube_config):
        """Test video download failure."""
        mock_yt_dlp.side_effect = Exception("Download failed")
        
        result = await youtube_processor._download_video("https://youtube.com/watch?v=test", youtube_config)
        
        assert result is None
    
    @pytest.mark.asyncio
    @patch('yt_dlp.YoutubeDL')
    async def test_download_audio_success(self, mock_yt_dlp, youtube_processor, youtube_config, tmp_path):
        """Test successful audio download."""
        mock_ydl = Mock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        # Create a mock downloaded file
        mock_audio_file = tmp_path / "audio_123_test.mp3"
        mock_audio_file.write_text("mock audio")
        
        with patch('asyncio.to_thread'), \
             patch('pathlib.Path.glob', return_value=[mock_audio_file]):
            
            result = await youtube_processor._download_audio("https://youtube.com/watch?v=test", youtube_config)
            
            assert result == mock_audio_file
    
    @pytest.mark.asyncio
    @patch('yt_dlp.YoutubeDL')
    async def test_download_subtitles_success(self, mock_yt_dlp, youtube_processor, youtube_config, tmp_path):
        """Test successful subtitle download."""
        mock_ydl = Mock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        # Create mock subtitle files
        mock_sub_files = [
            tmp_path / "subs_123_test.en.vtt",
            tmp_path / "subs_123_test.es.srt"
        ]
        for f in mock_sub_files:
            f.write_text("mock subtitles")
        
        with patch('asyncio.to_thread'), \
             patch('pathlib.Path.glob', side_effect=[mock_sub_files[:1], mock_sub_files[1:]]):
            
            result = await youtube_processor._download_subtitles("https://youtube.com/watch?v=test", youtube_config)
            
            assert len(result) == 2
            assert all(f in result for f in mock_sub_files)
    
    @pytest.mark.asyncio
    @patch('yt_dlp.YoutubeDL')
    async def test_download_thumbnails_success(self, mock_yt_dlp, youtube_processor, youtube_config, tmp_path):
        """Test successful thumbnail download."""
        mock_ydl = Mock()
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        # Create mock thumbnail files
        mock_thumb_files = [
            tmp_path / "thumb_123_test.jpg",
            tmp_path / "thumb_123_test.webp"
        ]
        for f in mock_thumb_files:
            f.write_text("mock thumbnail")
        
        with patch('asyncio.to_thread'), \
             patch('pathlib.Path.glob', side_effect=[mock_thumb_files[:1], [], mock_thumb_files[1:]]):
            
            result = await youtube_processor._download_thumbnails("https://youtube.com/watch?v=test", youtube_config)
            
            assert len(result) == 2
            assert all(f in result for f in mock_thumb_files)
    
    def test_get_video_format_mp4(self, youtube_processor):
        """Test video format selection for MP4."""
        config = YouTubeConfig(format_preference="mp4")
        format_string = youtube_processor._get_video_format(config)
        
        assert "mp4" in format_string
        assert "m4a" in format_string
    
    def test_get_video_format_webm(self, youtube_processor):
        """Test video format selection for WebM."""
        config = YouTubeConfig(format_preference="webm")
        format_string = youtube_processor._get_video_format(config)
        
        assert "webm" in format_string
    
    def test_get_video_format_best(self, youtube_processor):
        """Test video format selection for best quality."""
        config = YouTubeConfig(format_preference="best")
        format_string = youtube_processor._get_video_format(config)
        
        assert format_string == "bv*+ba/b"
    
    @pytest.mark.asyncio
    @patch('yt_dlp.YoutubeDL')
    async def test_process_playlist_success(self, mock_yt_dlp, youtube_processor, youtube_config, mock_metadata):
        """Test successful playlist processing."""
        mock_ydl = Mock()
        mock_playlist_info = {
            'entries': [
                {'id': 'video1', 'url': 'https://youtube.com/watch?v=video1'},
                {'id': 'video2', 'url': 'https://youtube.com/watch?v=video2'}
            ]
        }
        mock_ydl.extract_info.return_value = mock_playlist_info
        mock_yt_dlp.return_value.__enter__.return_value = mock_ydl
        
        with patch('asyncio.to_thread', return_value=mock_playlist_info), \
             patch.object(youtube_processor, 'process_url', return_value=YouTubeDownloadResult(
                 video_path=None, audio_path=None, subtitle_paths=[], thumbnail_paths=[],
                 metadata=mock_metadata, download_time=1.0, file_size=1000, temp_files=[]
             )):
            
            results = await youtube_processor.process_playlist("https://youtube.com/playlist?list=test", youtube_config)
            
            assert len(results) == 2
            assert all(isinstance(r, YouTubeDownloadResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_process_playlist_failure(self, youtube_processor, youtube_config):
        """Test playlist processing failure."""
        with patch('yt_dlp.YoutubeDL', side_effect=Exception("Playlist error")):
            with pytest.raises(ProcessingError, match="Playlist processing failed"):
                await youtube_processor.process_playlist("https://youtube.com/playlist?list=test", youtube_config)
    
    def test_cleanup_temp_files(self, youtube_processor, tmp_path):
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
        youtube_processor.cleanup_temp_files(temp_files)
        
        # Verify files are deleted
        assert all(not f.exists() for f in temp_files)
    
    def test_cleanup_temp_files_missing_file(self, youtube_processor, tmp_path):
        """Test cleanup with missing files."""
        non_existent_file = tmp_path / "non_existent.txt"
        
        # Should not raise exception
        youtube_processor.cleanup_temp_files([non_existent_file])
    
    @pytest.mark.asyncio
    async def test_process_url_exception_handling(self, youtube_processor, youtube_config):
        """Test exception handling in process_url."""
        with patch.object(youtube_processor, '_extract_metadata_only', side_effect=Exception("Test error")):
            with pytest.raises(ProcessingError, match="YouTube processing failed"):
                await youtube_processor.process_url("https://youtube.com/watch?v=test", youtube_config)
