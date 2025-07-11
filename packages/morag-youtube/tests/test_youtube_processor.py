"""Tests for the YouTube processor module."""

import pytest
import os
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from morag_youtube.processor import YouTubeProcessor, YouTubeConfig, YouTubeMetadata

# Sample metadata for mocking
SAMPLE_METADATA = {
    'id': 'dQw4w9WgXcQ',
    'title': 'Rick Astley - Never Gonna Give You Up (Official Music Video)',
    'description': 'Sample description',
    'uploader': 'Rick Astley',
    'upload_date': '20091025',
    'duration': 213.0,
    'view_count': 1000000,
    'like_count': 50000,
    'comment_count': 10000,
    'tags': ['Rick Astley', 'Never Gonna Give You Up'],
    'categories': ['Music'],
    'thumbnail': 'https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
    'webpage_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'channel_id': 'UCuAXFkgsw1L7xaCfnd5JJOw',
    'channel_url': 'https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw',
    'playlist_id': None,
    'playlist_title': None,
    'playlist_index': None,
}

@pytest.fixture
def youtube_processor():
    """Create a YouTubeProcessor instance for testing."""
    return YouTubeProcessor()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

@pytest.mark.asyncio
async def test_youtube_config_defaults():
    """Test that YouTubeConfig has the expected default values."""
    config = YouTubeConfig()
    
    assert config.quality == "best"
    assert config.format_preference == "mp4"
    assert config.extract_audio is True
    assert config.download_subtitles is True
    assert config.subtitle_languages == ["en"]
    assert config.max_filesize == "500M"
    assert config.download_thumbnails is True
    assert config.extract_metadata_only is False

@pytest.mark.asyncio
async def test_supports_format(youtube_processor):
    """Test that the processor correctly identifies supported formats."""
    assert youtube_processor.supports_format("youtube") is True
    assert youtube_processor.supports_format("YOUTUBE") is True
    assert youtube_processor.supports_format("yt") is True
    assert youtube_processor.supports_format("video") is False
    assert youtube_processor.supports_format("mp4") is False

@pytest.mark.asyncio
async def test_process_file_not_supported(youtube_processor):
    """Test that process() method raises an error as expected."""
    with pytest.raises(Exception):
        await youtube_processor.process(Path("test.txt"))

@pytest.mark.asyncio
@patch('yt_dlp.YoutubeDL')
async def test_extract_metadata_only(mock_ytdl, youtube_processor):
    """Test extracting metadata without downloading."""
    # Configure the mock
    mock_instance = MagicMock()
    mock_ytdl.return_value.__enter__.return_value = mock_instance
    mock_instance.extract_info.return_value = SAMPLE_METADATA
    
    # Call the method
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    metadata = await youtube_processor._extract_metadata_only(url)
    
    # Verify the result
    assert metadata.id == SAMPLE_METADATA['id']
    assert metadata.title == SAMPLE_METADATA['title']
    assert metadata.uploader == SAMPLE_METADATA['uploader']
    assert metadata.duration == SAMPLE_METADATA['duration']
    assert metadata.view_count == SAMPLE_METADATA['view_count']
    assert metadata.like_count == SAMPLE_METADATA['like_count']
    assert metadata.comment_count == SAMPLE_METADATA['comment_count']
    
    # Verify the mock was called correctly
    mock_ytdl.assert_called_once()
    mock_instance.extract_info.assert_called_once_with(url, download=False)

@pytest.mark.asyncio
@patch('morag_youtube.processor.YouTubeProcessor._extract_metadata_only')
@patch('morag_youtube.processor.YouTubeProcessor._download_video')
async def test_process_url(mock_download, mock_metadata, youtube_processor):
    """Test processing a URL with both metadata extraction and download."""
    # Configure the mocks
    metadata = YouTubeMetadata(**SAMPLE_METADATA)
    mock_metadata.return_value = metadata
    
    mock_download.return_value = {
        'video_path': Path('/tmp/video.mp4'),
        'audio_path': Path('/tmp/audio.mp3'),
        'subtitle_paths': [Path('/tmp/subs.vtt')],
        'thumbnail_paths': [Path('/tmp/thumb.jpg')],
        'file_size': 10000,
        'temp_files': [Path('/tmp/video.mp4'), Path('/tmp/audio.mp3')]
    }
    
    # Call the method
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = await youtube_processor.process_url(url)
    
    # Verify the result
    assert result.success is True
    assert result.metadata == metadata
    assert result.video_path == Path('/tmp/video.mp4')
    assert result.audio_path == Path('/tmp/audio.mp3')
    assert len(result.subtitle_paths) == 1
    assert result.subtitle_paths[0] == Path('/tmp/subs.vtt')
    assert len(result.thumbnail_paths) == 1
    assert result.thumbnail_paths[0] == Path('/tmp/thumb.jpg')
    assert result.file_size == 10000
    
    # Verify the mocks were called correctly
    mock_metadata.assert_called_once_with(url)
    mock_download.assert_called_once()

@pytest.mark.asyncio
@patch('morag_youtube.processor.YouTubeProcessor._extract_metadata_only')
async def test_process_url_metadata_only(mock_metadata, youtube_processor):
    """Test processing a URL with metadata extraction only."""
    # Configure the mock
    metadata = YouTubeMetadata(**SAMPLE_METADATA)
    mock_metadata.return_value = metadata
    
    # Call the method with metadata-only config
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    config = YouTubeConfig(extract_metadata_only=True)
    result = await youtube_processor.process_url(url, config)
    
    # Verify the result
    assert result.success is True
    assert result.metadata == metadata
    assert result.video_path is None
    assert result.audio_path is None
    assert len(result.subtitle_paths) == 0
    assert len(result.thumbnail_paths) == 0
    assert result.file_size == 0
    
    # Verify the mock was called correctly
    mock_metadata.assert_called_once_with(url)

@pytest.mark.asyncio
@patch('yt_dlp.YoutubeDL')
@patch('asyncio.to_thread')
async def test_download_video(mock_to_thread, mock_ytdl, youtube_processor, temp_dir):
    """Test downloading a video."""
    # Create test files to simulate download
    video_file = temp_dir / "video-id.mp4"
    audio_file = temp_dir / "video-id.mp3"
    subtitle_file = temp_dir / "video-id.en.vtt"
    thumbnail_file = temp_dir / "video-id.jpg"
    
    video_file.touch()
    audio_file.touch()
    subtitle_file.touch()
    thumbnail_file.touch()
    
    # Configure the mocks
    mock_instance = MagicMock()
    mock_ytdl.return_value.__enter__.return_value = mock_instance
    mock_to_thread.return_value = SAMPLE_METADATA
    
    # Patch the temp_dir to use our test directory
    with patch.object(youtube_processor, 'temp_dir', temp_dir):
        # Call the method
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        config = YouTubeConfig()
        result = await youtube_processor._download_video(url, config)
        
        # Verify the result contains the expected files
        assert 'video_path' in result
        assert 'audio_path' in result
        assert 'subtitle_paths' in result
        assert 'thumbnail_paths' in result
        assert 'file_size' in result
        assert 'temp_files' in result
        
        # Verify the mock was called correctly
        mock_ytdl.assert_called_once()
        mock_to_thread.assert_called_once()

@pytest.mark.asyncio
@patch('morag_youtube.processor.YouTubeProcessor.process_url')
async def test_process_playlist(mock_process_url, youtube_processor):
    """Test processing a playlist."""
    # Configure the mock for extract_info to return playlist info
    playlist_info = {
        'entries': [
            {'url': 'https://www.youtube.com/watch?v=video1'},
            {'url': 'https://www.youtube.com/watch?v=video2'},
        ]
    }
    
    # Mock YoutubeDL for playlist extraction
    with patch('yt_dlp.YoutubeDL') as mock_ytdl:
        mock_instance = MagicMock()
        mock_ytdl.return_value.__enter__.return_value = mock_instance
        mock_instance.extract_info.return_value = playlist_info
        
        # Mock process_url to return dummy results
        mock_result1 = MagicMock(success=True)
        mock_result2 = MagicMock(success=True)
        mock_process_url.side_effect = [mock_result1, mock_result2]
        
        # Call the method
        url = "https://www.youtube.com/playlist?list=PLsomething"
        config = YouTubeConfig()
        results = await youtube_processor.process_playlist(url, config)
        
        # Verify the results
        assert len(results) == 2
        assert results[0] == mock_result1
        assert results[1] == mock_result2
        
        # Verify process_url was called for each video
        assert mock_process_url.call_count == 2
        mock_process_url.assert_any_call('https://www.youtube.com/watch?v=video1', config)
        mock_process_url.assert_any_call('https://www.youtube.com/watch?v=video2', config)