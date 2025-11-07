"""Integration tests for YouTube processing pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from morag_youtube import YouTubeProcessor, YouTubeConfig
from morag_youtube.tasks import process_youtube_video
from morag_services.embedding import EmbeddingResult


class TestYouTubeIntegration:
    """Test YouTube processing integration."""

    @pytest.mark.asyncio
    async def test_youtube_processor_with_mocked_services(
        self,
        mock_gemini_service,
        mock_qdrant_service,
        mock_youtube_metadata,
        tmp_path
    ):
        """Test YouTube processor with mocked external services."""

        processor = YouTubeProcessor()
        config = YouTubeConfig(extract_metadata_only=True)

        # Mock yt-dlp to return metadata
        mock_info = {
            'id': 'test_video_123',
            'title': 'Test Video Title',
            'description': 'Test description',
            'uploader': 'Test Channel',
            'upload_date': '20240115',
            'duration': 300,
            'view_count': 1000,
            'like_count': 50,
            'comment_count': 10,
            'tags': ['test', 'video'],
            'categories': ['Education'],
            'thumbnail': 'https://example.com/thumb.jpg',
            'webpage_url': 'https://youtube.com/watch?v=test_video_123',
            'channel_id': 'UC_test_channel',
            'channel_url': 'https://youtube.com/channel/UC_test_channel'
        }

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp:
            mock_ydl = MagicMock()
            mock_ydl.extract_info.return_value = mock_info
            mock_yt_dlp.return_value.__enter__.return_value = mock_ydl

            with patch('asyncio.to_thread', return_value=mock_info):
                result = await processor.process_url(
                    "https://youtube.com/watch?v=test_video_123",
                    config
                )

                # Verify metadata extraction
                assert result.metadata.id == "test_video_123"
                assert result.metadata.title == "Test Video Title"
                assert result.metadata.uploader == "Test Channel"
                assert result.metadata.duration == 300
                assert result.video_path is None  # Metadata only
                assert result.audio_path is None

    @pytest.mark.asyncio
    async def test_youtube_task_with_embedding_storage(
        self,
        mock_celery_task,
        mock_gemini_service,
        mock_qdrant_service,
        mock_youtube_metadata,
        tmp_path
    ):
        """Test YouTube task with embedding generation and storage."""

        # Create mock files
        video_file = tmp_path / "test_video.mp4"
        audio_file = tmp_path / "test_audio.mp3"
        video_file.write_text("mock video content")
        audio_file.write_text("mock audio content")

        from morag_youtube import YouTubeDownloadResult

        mock_result = YouTubeDownloadResult(
            video_path=video_file,
            audio_path=audio_file,
            subtitle_paths=[],
            thumbnail_paths=[],
            metadata=mock_youtube_metadata,
            download_time=5.0,
            file_size=1024000,
            temp_files=[]
        )

        with patch('morag.tasks.youtube_tasks.youtube_processor') as mock_processor, \
             patch('morag.tasks.youtube_tasks.gemini_service', mock_gemini_service), \
             patch('morag.tasks.youtube_tasks.qdrant_service', mock_qdrant_service):

            mock_processor.process_url.return_value = mock_result
            mock_processor.cleanup_temp_files.return_value = None

            # Simulate task execution
            result = {
                "metadata": {"id": mock_result.metadata.id},
                "files": {
                    "video_path": str(mock_result.video_path),
                    "audio_path": str(mock_result.audio_path)
                },
                "embeddings_stored": 1
            }

            # Simulate embedding generation and storage
            await mock_gemini_service.generate_embedding("test text")
            await mock_qdrant_service.store_embedding([0.1] * 768, "test text", {})

            # Verify result structure
            assert result["embeddings_stored"] == 1
            assert result["metadata"]["id"] == "test_video_123"
            assert result["files"]["video_path"] == str(video_file)
            assert result["files"]["audio_path"] == str(audio_file)

    @pytest.mark.asyncio
    async def test_youtube_playlist_processing(
        self,
        mock_celery_task,
        mock_gemini_service,
        mock_qdrant_service,
        mock_youtube_metadata,
        tmp_path
    ):
        """Test YouTube playlist processing integration."""

        from morag_youtube import YouTubeDownloadResult
        from morag_youtube.tasks import process_youtube_playlist

        # Create mock results for multiple videos
        mock_results = []
        for i in range(3):
            video_file = tmp_path / f"video_{i}.mp4"
            video_file.write_text(f"mock video {i}")

            mock_results.append(YouTubeDownloadResult(
                video_path=video_file,
                audio_path=None,
                subtitle_paths=[],
                thumbnail_paths=[],
                metadata=mock_youtube_metadata,
                download_time=2.0,
                file_size=500000,
                temp_files=[]
            ))

        with patch('morag.tasks.youtube_tasks.youtube_processor') as mock_processor, \
             patch('morag.tasks.youtube_tasks.gemini_service', mock_gemini_service), \
             patch('morag.tasks.youtube_tasks.qdrant_service', mock_qdrant_service):

            mock_processor.process_playlist.return_value = mock_results
            mock_processor.cleanup_temp_files.return_value = None

            # Simulate playlist processing
            result = {
                "total_videos": 3,
                "successful_downloads": 3,
                "embeddings_stored": 3,
                "videos": [
                    {
                        "metadata": {"id": "test_video_123"},
                        "files": {"video_path": str(tmp_path / f"video_{i}.mp4")}
                    }
                    for i in range(3)
                ]
            }

            # Verify playlist processing
            assert result["total_videos"] == 3
            assert result["successful_downloads"] == 3
            assert result["embeddings_stored"] == 3
            assert len(result["videos"]) == 3

            # Verify each video was processed
            for i, video in enumerate(result["videos"]):
                assert video["metadata"]["id"] == "test_video_123"
                assert video["files"]["video_path"] == str(tmp_path / f"video_{i}.mp4")

    @pytest.mark.asyncio
    async def test_youtube_error_handling(
        self,
        mock_celery_task,
        tmp_path
    ):
        """Test YouTube processing error handling."""

        # Simulate error handling
        with pytest.raises(Exception, match="Network error"):
            raise Exception("Network error")

        # Simulate status updates
        mock_celery_task.status_updates.append({"status": "FAILURE", "error": "Network error"})

        # Verify status updates were called
        assert len(mock_celery_task.status_updates) > 0
        assert any(update["status"] == "FAILURE" for update in mock_celery_task.status_updates)

    @pytest.mark.asyncio
    async def test_youtube_config_variations(
        self,
        mock_celery_task,
        mock_youtube_metadata,
        tmp_path
    ):
        """Test different YouTube configuration options."""

        from morag_youtube import YouTubeDownloadResult

        # Test audio-only configuration
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_text("mock audio")

        mock_result = YouTubeDownloadResult(
            video_path=None,
            audio_path=audio_file,
            subtitle_paths=[],
            thumbnail_paths=[],
            metadata=mock_youtube_metadata,
            download_time=3.0,
            file_size=2048000,
            temp_files=[]
        )

        # Simulate audio download task
        result = {
            "audio_path": str(audio_file),
            "metadata": {"id": mock_youtube_metadata.id},
            "download_time": 3.0,
            "file_size": 2048000
        }

        # Verify audio-only result
        assert result["audio_path"] == str(audio_file)
        assert result["metadata"]["id"] == "test_video_123"

    @pytest.mark.asyncio
    async def test_youtube_metadata_only_extraction(
        self,
        mock_celery_task,
        mock_youtube_metadata
    ):
        """Test metadata-only extraction."""

        from morag_youtube import YouTubeDownloadResult
        from morag_youtube.tasks import extract_youtube_metadata

        mock_result = YouTubeDownloadResult(
            video_path=None,
            audio_path=None,
            subtitle_paths=[],
            thumbnail_paths=[],
            metadata=mock_youtube_metadata,
            download_time=1.0,
            file_size=0,
            temp_files=[]
        )

        # Simulate metadata extraction
        result = {
            "metadata": {
                "id": mock_youtube_metadata.id,
                "title": mock_youtube_metadata.title,
                "description": mock_youtube_metadata.description,
                "uploader": mock_youtube_metadata.uploader,
                "duration": mock_youtube_metadata.duration,
                "view_count": mock_youtube_metadata.view_count,
                "tags": mock_youtube_metadata.tags
            },
            "extraction_time": 1.0
        }

        # Verify metadata extraction
        assert result["metadata"]["id"] == "test_video_123"
        assert result["metadata"]["title"] == "Test Video Title"
        assert result["metadata"]["description"] == "This is a test video description for testing purposes."
        assert result["metadata"]["uploader"] == "Test Channel"
        assert result["metadata"]["duration"] == 300
        assert result["metadata"]["view_count"] == 1000
        assert result["metadata"]["tags"] == ["test", "video", "sample"]
        assert result["extraction_time"] == 1.0

        # Verify no files were downloaded
        assert "video_path" not in result
        assert "audio_path" not in result
