"""YouTube processing using external transcription service."""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
import structlog

from morag_core.exceptions import ProcessingError
from morag_core.interfaces.processor import BaseProcessor, ProcessingConfig, ProcessingResult

logger = structlog.get_logger(__name__)

@dataclass
class YouTubeConfig(ProcessingConfig):
    """Configuration for YouTube processing using external service."""
    # External service options
    service_url: Optional[str] = None  # URL of external YouTube service
    service_timeout: int = 300  # Timeout in seconds (5 minutes default)

    # Video download options (opt-in only)
    download_video: bool = False  # Whether to download video file
    output_dir: Optional[Path] = None  # Directory to save downloaded files

    # Legacy options (kept for backward compatibility but not used)
    quality: str = "best"
    format_preference: str = "mp4"
    extract_audio: bool = True
    download_subtitles: bool = True
    subtitle_languages: List[str] = field(default_factory=lambda: ["en"])
    max_filesize: Optional[str] = None
    download_thumbnails: bool = True
    extract_metadata_only: bool = False
    extract_transcript: bool = True
    transcript_language: Optional[str] = None
    transcript_format: str = "text"
    prefer_audio_transcription: bool = True
    cookies_file: Optional[str] = None
    transcript_only: bool = False

@dataclass
class YouTubeMetadata:
    """YouTube video metadata."""
    id: str
    title: str
    description: str
    uploader: str
    upload_date: str
    duration: float
    view_count: int
    like_count: Optional[int]
    comment_count: Optional[int]
    tags: List[str]
    categories: List[str]
    thumbnail_url: str
    webpage_url: str
    channel_id: str
    channel_url: str
    playlist_id: Optional[str]
    playlist_title: Optional[str]
    playlist_index: Optional[int]

@dataclass
class YouTubeDownloadResult(ProcessingResult):
    """Result of YouTube processing operation using external service."""
    # External service response data
    metadata: Optional[YouTubeMetadata] = None
    transcript: Optional[Dict[str, Any]] = None
    transcript_languages: List[Dict[str, Any]] = field(default_factory=list)
    formats: List[Dict[str, Any]] = field(default_factory=list)
    total_formats: int = 0

    # Optional video download (when download_video=True)
    video_path: Optional[Path] = None
    output_file: Optional[str] = None
    video_download: Optional[Dict[str, Any]] = None

    # Legacy fields (kept for backward compatibility)
    audio_path: Optional[Path] = None
    subtitle_paths: List[Path] = field(default_factory=list)
    thumbnail_paths: List[Path] = field(default_factory=list)
    file_size: int = 0
    temp_files: List[Path] = field(default_factory=list)
    transcript_path: Optional[Path] = None
    transcript_text: Optional[str] = None
    transcript_language: Optional[str] = None

class YouTubeProcessor(BaseProcessor):
    """YouTube processing service using external transcription API."""

    def __init__(self):
        """Initialize YouTube processor."""
        from .external_service import YouTubeExternalService
        self.external_service = YouTubeExternalService()
        logger.info("YouTube processor initialized with external service")

    async def process_url(
        self,
        url: str,
        config: Optional[YouTubeConfig] = None
    ) -> YouTubeDownloadResult:
        """Process YouTube URL using external transcription service."""
        start_time = time.time()
        config = config or YouTubeConfig()

        try:
            logger.info("Starting YouTube processing with external service", url=url)

            # Configure external service
            if config.service_url:
                self.external_service.service_url = config.service_url
                self.external_service.endpoint = f"{config.service_url}/v1/youtube/transcribe"
            if config.service_timeout:
                self.external_service.timeout = config.service_timeout

            # Call external service
            service_result = await self.external_service.transcribe_video(
                url=url,
                download_video=config.download_video,
                output_dir=config.output_dir
            )

            # Convert external service response to our result format
            result = self._convert_service_result(service_result, start_time)

            logger.info("YouTube processing completed successfully",
                       url=url,
                       has_transcript=bool(result.transcript),
                       video_downloaded=bool(result.video_path))

            return result

        except Exception as e:
            logger.exception("Error processing YouTube URL", url=url, error=str(e))
            return YouTubeDownloadResult(
                metadata=None,
                transcript=None,
                transcript_languages=[],
                formats=[],
                total_formats=0,
                video_path=None,
                output_file=None,
                video_download=None,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def _convert_service_result(self, service_result: Dict[str, Any], start_time: float) -> YouTubeDownloadResult:
        """Convert external service result to YouTubeDownloadResult format.

        Args:
            service_result: Result from external service
            start_time: Processing start time

        Returns:
            YouTubeDownloadResult object
        """
        # Convert metadata if available
        metadata = None
        if service_result.get("metadata"):
            metadata_dict = service_result["metadata"]
            metadata = YouTubeMetadata(
                id=metadata_dict.get("id", ""),
                title=metadata_dict.get("title", ""),
                description=metadata_dict.get("description", ""),
                uploader=metadata_dict.get("uploader", ""),
                upload_date=metadata_dict.get("upload_date", ""),
                duration=metadata_dict.get("duration", 0),
                view_count=metadata_dict.get("view_count", 0),
                like_count=metadata_dict.get("like_count"),
                comment_count=metadata_dict.get("comment_count"),
                tags=metadata_dict.get("tags", []),
                categories=metadata_dict.get("categories", []),
                thumbnail_url=metadata_dict.get("thumbnail", ""),
                webpage_url=f"https://www.youtube.com/watch?v={metadata_dict.get('id', '')}",
                channel_id=metadata_dict.get("uploader_id", ""),
                channel_url=metadata_dict.get("channel_url", ""),
                playlist_id=None,
                playlist_title=None,
                playlist_index=None
            )

        # Set video path if downloaded
        video_path = None
        if service_result.get("output_file"):
            video_path = Path(service_result["output_file"])

        # Extract transcript text for backward compatibility
        transcript_text = None
        transcript_language = None
        if service_result.get("transcript") and "entries" in service_result["transcript"]:
            entries = service_result["transcript"]["entries"]
            transcript_text = " ".join(entry.get("text", "") for entry in entries)
            transcript_language = service_result["transcript"].get("language")

        return YouTubeDownloadResult(
            metadata=metadata,
            transcript=service_result.get("transcript"),
            transcript_languages=service_result.get("transcript_languages", []),
            formats=service_result.get("formats", []),
            total_formats=service_result.get("total_formats", 0),
            video_path=video_path,
            output_file=service_result.get("output_file"),
            video_download=service_result.get("video_download"),
            # Legacy fields for backward compatibility
            transcript_text=transcript_text,
            transcript_language=transcript_language,
            processing_time=time.time() - start_time,
            success=service_result.get("success", False)
        )

    async def process_playlist(
        self,
        playlist_url: str,
        config: Optional[YouTubeConfig] = None
    ) -> List[YouTubeDownloadResult]:
        """Process YouTube playlist using external service.

        Args:
            playlist_url: YouTube playlist URL
            config: Processing configuration

        Returns:
            List of YouTubeDownloadResult objects for each video
        """
        config = config or YouTubeConfig()

        # For now, we'll extract individual video URLs and process them
        # This is a simplified implementation - the external service
        # could potentially handle playlists directly in the future
        logger.warning("Playlist processing not yet implemented with external service")
        return []

    def cleanup(self, result: YouTubeDownloadResult) -> None:
        """Clean up temporary files."""
        # For external service, cleanup is minimal since files are managed externally
        logger.debug("Cleanup completed for external service result")

    async def process(self, file_path: Path, config: Optional[ProcessingConfig] = None) -> ProcessingResult:
        """Process content from file.

        This method is required by the BaseProcessor interface but is not applicable
        for YouTube processing. Use process_url instead.

        Raises:
            ProcessingError: Always raises this error as this method is not supported
        """
        raise ProcessingError("YouTubeProcessor does not support file processing. Use process_url instead.")

    def supports_format(self, format_type: str) -> bool:
        """Check if processor supports the given format.

        Args:
            format_type: Format type to check

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() in ['youtube', 'yt']