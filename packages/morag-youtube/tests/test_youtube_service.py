"""Tests for the YouTube service module."""

import pytest
import os
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock

from morag_youtube.service import YouTubeService
from morag_youtube.processor import YouTubeProcessor, YouTubeConfig, YouTubeMetadata, YouTubeDownloadResult

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
    'thumbnail_url': 'https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
    'webpage_url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'channel_id': 'UCuAXFkgsw1L7xaCfnd5JJOw',
    'channel_url': 'https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw',
    'playlist_id': None,
    'playlist_title': None,
    'playlist_index': None,
}

@pytest.fixture
def youtube_service():
    """Create a YouTubeService instance for testing."""
    return YouTubeService(service_url="http://localhost:8000", service_timeout=60)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

@pytest.fixture
def sample_download_result():
    """Create a sample YouTubeDownloadResult for testing."""
    metadata = YouTubeMetadata(**SAMPLE_METADATA)
    return YouTubeDownloadResult(
        video_path=Path("/tmp/video.mp4"),
        audio_path=Path("/tmp/audio.mp3"),
        subtitle_paths=[Path("/tmp/subs.vtt")],
        thumbnail_paths=[Path("/tmp/thumb.jpg")],
        metadata=metadata,
        processing_time=1.5,
        file_size=10000,
        temp_files=[],
        success=True
    )

@pytest.mark.asyncio
async def test_service_initialization(youtube_service):
    """Test that the service initializes correctly."""
    assert isinstance(youtube_service.processor, YouTubeProcessor)
    assert youtube_service.max_concurrent_downloads == 2
    assert youtube_service.semaphore._value == 2

@pytest.mark.asyncio
@patch.object(YouTubeProcessor, 'process_url')
async def test_process_video(mock_process_url, youtube_service, sample_download_result):
    """Test processing a single video."""
    # Configure the mock
    mock_process_url.return_value = sample_download_result
    
    # Call the method
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = await youtube_service.process_video(url)
    
    # Verify the result
    assert result == sample_download_result
    
    # Verify the mock was called correctly
    mock_process_url.assert_called_once_with(url, None)

@pytest.mark.asyncio
@patch.object(YouTubeProcessor, 'process_url')
async def test_process_videos(mock_process_url, youtube_service, sample_download_result):
    """Test processing multiple videos concurrently."""
    # Configure the mock
    mock_process_url.return_value = sample_download_result
    
    # Call the method
    urls = [
        "https://www.youtube.com/watch?v=video1",
        "https://www.youtube.com/watch?v=video2",
    ]
    results = await youtube_service.process_videos(urls)
    
    # Verify the results
    assert len(results) == 2
    assert results[0] == sample_download_result
    assert results[1] == sample_download_result
    
    # Verify the mock was called correctly
    assert mock_process_url.call_count == 2
    mock_process_url.assert_any_call(urls[0], None)
    mock_process_url.assert_any_call(urls[1], None)

@pytest.mark.asyncio
@patch.object(YouTubeProcessor, 'process_playlist')
async def test_process_playlist(mock_process_playlist, youtube_service, sample_download_result):
    """Test processing a playlist."""
    # Configure the mock
    mock_process_playlist.return_value = [sample_download_result, sample_download_result]
    
    # Call the method
    url = "https://www.youtube.com/playlist?list=PLsomething"
    results = await youtube_service.process_playlist(url)
    
    # Verify the results
    assert len(results) == 2
    assert results[0] == sample_download_result
    assert results[1] == sample_download_result
    
    # Verify the mock was called correctly
    mock_process_playlist.assert_called_once_with(url, None)

@pytest.mark.asyncio
@patch.object(YouTubeProcessor, 'process_url')
async def test_extract_metadata(mock_process_url, youtube_service, sample_download_result):
    """Test extracting metadata without downloading."""
    # Configure the mock
    mock_process_url.return_value = sample_download_result
    
    # Call the method
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    metadata = await youtube_service.extract_metadata(url)
    
    # Verify the result
    assert metadata['id'] == SAMPLE_METADATA['id']
    assert metadata['title'] == SAMPLE_METADATA['title']
    assert metadata['uploader'] == SAMPLE_METADATA['uploader']
    assert metadata['duration'] == SAMPLE_METADATA['duration']
    
    # Verify the mock was called correctly
    mock_process_url.assert_called_once()
    # Verify that extract_metadata_only was set to True in the config
    config = mock_process_url.call_args[0][1]
    assert config.extract_metadata_only is True

@pytest.mark.asyncio
@patch.object(YouTubeProcessor, 'process_url')
async def test_download_video_with_options(mock_process_url, youtube_service, sample_download_result, temp_dir):
    """Test downloading a video with custom options."""
    # Configure the mock
    mock_process_url.return_value = sample_download_result
    
    # Create test files to simulate download
    video_file = temp_dir / "video.mp4"
    audio_file = temp_dir / "audio.mp3"
    subtitle_file = temp_dir / "subs.vtt"
    thumbnail_file = temp_dir / "thumb.jpg"
    
    video_file.touch()
    audio_file.touch()
    subtitle_file.touch()
    thumbnail_file.touch()
    
    # Create a modified result with our test files
    test_result = YouTubeDownloadResult(
        video_path=video_file,
        audio_path=audio_file,
        subtitle_paths=[subtitle_file],
        thumbnail_paths=[thumbnail_file],
        metadata=sample_download_result.metadata,
        processing_time=1.5,
        file_size=10000,
        temp_files=[],
        success=True
    )
    mock_process_url.return_value = test_result
    
    # Call the method
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    
    result = await youtube_service.download_video(
        url,
        output_dir=output_dir,
        quality="high",
        extract_audio=True,
        download_subtitles=True
    )
    
    # Verify the mock was called correctly
    mock_process_url.assert_called_once()
    config = mock_process_url.call_args[0][1]
    assert config.quality == "high"
    assert config.extract_audio is True
    assert config.download_subtitles is True

@pytest.mark.asyncio
@patch.object(YouTubeProcessor, 'process_url')
async def test_download_audio(mock_process_url, youtube_service, sample_download_result, temp_dir):
    """Test downloading only audio from a video."""
    # Create test audio file
    audio_file = temp_dir / "audio.mp3"
    audio_file.touch()
    
    # Create a modified result with our test file
    test_result = YouTubeDownloadResult(
        video_path=None,
        audio_path=audio_file,
        subtitle_paths=[],
        thumbnail_paths=[],
        metadata=sample_download_result.metadata,
        processing_time=1.5,
        file_size=5000,
        temp_files=[],
        success=True
    )
    mock_process_url.return_value = test_result
    
    # Call the method
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    
    audio_path = await youtube_service.download_audio(url, output_dir=output_dir)
    
    # Verify the mock was called correctly
    mock_process_url.assert_called_once()
    config = mock_process_url.call_args[0][1]
    assert config.extract_audio is True
    assert config.download_subtitles is False
    assert config.download_thumbnails is False

@pytest.mark.asyncio
@patch.object(YouTubeProcessor, 'process_url')
async def test_download_subtitles(mock_process_url, youtube_service, sample_download_result, temp_dir):
    """Test downloading only subtitles from a video."""
    # Create test subtitle files
    subtitle_file1 = temp_dir / "subs.en.vtt"
    subtitle_file2 = temp_dir / "subs.fr.vtt"
    subtitle_file1.touch()
    subtitle_file2.touch()
    
    # Create a modified result with our test files
    test_result = YouTubeDownloadResult(
        video_path=None,
        audio_path=None,
        subtitle_paths=[subtitle_file1, subtitle_file2],
        thumbnail_paths=[],
        metadata=sample_download_result.metadata,
        processing_time=1.0,
        file_size=1000,
        temp_files=[],
        success=True
    )
    mock_process_url.return_value = test_result
    
    # Call the method
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    languages = ["en", "fr"]
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    
    subtitle_paths = await youtube_service.download_subtitles(url, languages=languages, output_dir=output_dir)
    
    # Verify the mock was called correctly
    mock_process_url.assert_called_once()
    config = mock_process_url.call_args[0][1]
    assert config.extract_audio is False
    assert config.download_subtitles is True
    assert config.subtitle_languages == languages
    assert config.download_thumbnails is False

@pytest.mark.asyncio
@patch.object(YouTubeProcessor, 'process_url')
async def test_download_thumbnail(mock_process_url, youtube_service, sample_download_result, temp_dir):
    """Test downloading only thumbnail from a video."""
    # Create test thumbnail file
    thumbnail_file = temp_dir / "thumb.jpg"
    thumbnail_file.touch()
    
    # Create a modified result with our test file
    test_result = YouTubeDownloadResult(
        video_path=None,
        audio_path=None,
        subtitle_paths=[],
        thumbnail_paths=[thumbnail_file],
        metadata=sample_download_result.metadata,
        processing_time=0.5,
        file_size=500,
        temp_files=[],
        success=True
    )
    mock_process_url.return_value = test_result
    
    # Call the method
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    output_dir = temp_dir / "output"
    output_dir.mkdir()
    
    thumbnail_path = await youtube_service.download_thumbnail(url, output_dir=output_dir)
    
    # Verify the mock was called correctly
    mock_process_url.assert_called_once()
    config = mock_process_url.call_args[0][1]
    assert config.extract_audio is False
    assert config.download_subtitles is False
    assert config.download_thumbnails is True

@pytest.mark.asyncio
async def test_cleanup_single_result(youtube_service, sample_download_result):
    """Test cleaning up a single result."""
    # Mock the processor's cleanup method
    with patch.object(youtube_service.processor, 'cleanup') as mock_cleanup:
        # Call the method
        youtube_service.cleanup(sample_download_result)
        
        # Verify the mock was called correctly
        mock_cleanup.assert_called_once_with(sample_download_result)

@pytest.mark.asyncio
async def test_cleanup_multiple_results(youtube_service, sample_download_result):
    """Test cleaning up multiple results."""
    # Mock the processor's cleanup method
    with patch.object(youtube_service.processor, 'cleanup') as mock_cleanup:
        # Call the method with a list of results
        results = [sample_download_result, sample_download_result]
        youtube_service.cleanup(results)
        
        # Verify the mock was called correctly
        assert mock_cleanup.call_count == 2
        mock_cleanup.assert_any_call(sample_download_result)