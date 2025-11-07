"""MoRAG YouTube - YouTube video processing capabilities for the MoRAG system."""

from morag_youtube.apify_service import ApifyYouTubeService, ApifyYouTubeServiceError
from morag_youtube.processor import (
    YouTubeConfig,
    YouTubeDownloadResult,
    YouTubeMetadata,
    YouTubeProcessor,
)
from morag_youtube.service import YouTubeService
from morag_youtube.tasks import (
    ExtractYouTubeMetadataTask,
    ProcessYouTubePlaylistTask,
    ProcessYouTubeVideoTask,
    create_celery_tasks,
    extract_youtube_metadata,
    process_youtube_playlist,
    process_youtube_video,
)
from morag_youtube.transcript import (
    TranscriptSegment,
    YouTubeTranscript,
    YouTubeTranscriptService,
)

__all__ = [
    "YouTubeProcessor",
    "YouTubeConfig",
    "YouTubeMetadata",
    "YouTubeDownloadResult",
    "YouTubeService",
    "ApifyYouTubeService",
    "ApifyYouTubeServiceError",
    "YouTubeTranscriptService",
    "YouTubeTranscript",
    "TranscriptSegment",
    "ProcessYouTubeVideoTask",
    "ProcessYouTubePlaylistTask",
    "ExtractYouTubeMetadataTask",
    "process_youtube_video",
    "process_youtube_playlist",
    "extract_youtube_metadata",
    "create_celery_tasks",
]

__version__ = "0.1.0"
