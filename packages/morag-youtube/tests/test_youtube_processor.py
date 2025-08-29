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
    'thumbnail_url': 'https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
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
    assert config.max_filesize is None
    assert config.download_thumbnails is True
    assert config.extract_metadata_only is False
    assert config.extract_transcript is True
    assert config.transcript_language is None
    assert config.transcript_format == "text"
    assert config.prefer_audio_transcription is True
    assert config.cookies_file is None
    assert config.transcript_only is False

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
@patch('morag_youtube.external_service.YouTubeExternalService.transcribe_video')
async def test_process_url_transcribe_only(mock_transcribe, youtube_processor):
    """Test processing URL for transcription only."""
    # Configure the mock
    mock_transcribe.return_value = {
        "success": True,
        "metadata": SAMPLE_METADATA,
        "transcript": {
            "entries": [
                {"text": "Never gonna give you up", "start": 0.0, "duration": 3.0}
            ]
        },
        "transcript_languages": [{"language": "en"}]
    }

    # Call the method
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    config = YouTubeConfig(download_video=False)
    result = await youtube_processor.process_url(url, config)

    # Verify the result
    assert result.success
    assert result.metadata.id == SAMPLE_METADATA['id']
    assert result.metadata.title == SAMPLE_METADATA['title']
    assert result.transcript is not None

    # Verify the mock was called correctly
    mock_transcribe.assert_called_once()
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


def test_convert_apify_result_with_nested_transcript():
    """Test that _convert_apify_result correctly handles nested transcript structure from Apify."""
    processor = YouTubeProcessor()

    # Sample Apify service result with nested transcript structure (like in apify_output.json)
    apify_result = {
        "url": "https://www.youtube.com/watch?v=siBSKuWmV8s",
        "videoId": "siBSKuWmV8s",
        "metadata": {
            "videoId": "siBSKuWmV8s",
            "title": "Building Conversational AI With GPT 5",
            "description": "Test description",
            "viewCount": 102,
            "likeCount": 4,
            "publishDate": "Aug 25, 2025",
            "channelName": "Mosleh",
            "channelId": "UCb-qWVAUsVZ_HnMY3xn43Nw",
            "category": "Science & Technology",
            "keywords": [],
            "thumbnails": [
                {
                    "url": "https://i.ytimg.com/vi/siBSKuWmV8s/maxresdefault.jpg",
                    "width": 1920,
                    "height": 1080
                }
            ]
        },
        "transcript": {
            "transcript": [
                {
                    "index": 0,
                    "text": "In this video, we're going to look into",
                    "start": 0.08,
                    "duration": 1.28,
                    "end": 1.36
                },
                {
                    "index": 1,
                    "text": "how to make a conversational AI agent",
                    "start": 1.36,
                    "duration": 2.079,
                    "end": 3.439
                }
            ]
        }
    }

    # Convert the result
    result = processor._convert_apify_result(apify_result, 0.0)

    # Verify transcript was extracted correctly
    assert result.transcript is not None
    assert result.transcript["text"] == "In this video, we're going to look into how to make a conversational AI agent"
    assert len(result.transcript["segments"]) == 2
    assert result.transcript["segments"][0]["text"] == "In this video, we're going to look into"
    assert result.transcript["segments"][1]["text"] == "how to make a conversational AI agent"

    # Verify metadata was extracted correctly
    assert result.metadata is not None
    assert result.metadata.id == "siBSKuWmV8s"
    assert result.metadata.title == "Building Conversational AI With GPT 5"
    assert result.metadata.uploader == "Mosleh"
    assert result.metadata.view_count == 102
    assert result.metadata.like_count == 4
    assert result.metadata.channel_id == "UCb-qWVAUsVZ_HnMY3xn43Nw"
    assert result.metadata.thumbnail_url == "https://i.ytimg.com/vi/siBSKuWmV8s/maxresdefault.jpg"
    assert result.metadata.categories == ["Science & Technology"]

    # Verify legacy transcript_text field is set
    assert result.transcript_text == "In this video, we're going to look into how to make a conversational AI agent"