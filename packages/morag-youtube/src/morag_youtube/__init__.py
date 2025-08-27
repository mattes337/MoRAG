"""MoRAG YouTube - YouTube video processing capabilities for the MoRAG system."""

from morag_youtube.processor import (
    YouTubeProcessor,
    YouTubeConfig,
    YouTubeMetadata,
    YouTubeDownloadResult
)
from morag_youtube.service import YouTubeService
from morag_youtube.transcript import (
    YouTubeTranscriptService,
    YouTubeTranscript,
    TranscriptSegment
)
from morag_youtube.tasks import (
    ProcessYouTubeVideoTask,
    ProcessYouTubePlaylistTask,
    ExtractYouTubeMetadataTask,
    process_youtube_video,
    process_youtube_playlist,
    extract_youtube_metadata,
    create_celery_tasks
)

__all__ = [
    "YouTubeProcessor",
    "YouTubeConfig",
    "YouTubeMetadata",
    "YouTubeDownloadResult",
    "YouTubeService",
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