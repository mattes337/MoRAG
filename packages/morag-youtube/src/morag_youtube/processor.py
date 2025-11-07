"""YouTube processing using Apify transcription service."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from morag_core.exceptions import ProcessingError
from morag_core.interfaces.processor import (
    BaseProcessor,
    ProcessingConfig,
    ProcessingResult,
)

logger = structlog.get_logger(__name__)


@dataclass
class YouTubeConfig(ProcessingConfig):
    """Configuration for YouTube processing using Apify service."""

    # Apify service options
    apify_timeout: int = 600  # Timeout in seconds (10 minutes default)
    use_proxy: bool = True  # Whether to use Apify proxy

    # Processing options
    extract_metadata: bool = True  # Whether to extract video metadata
    extract_transcript: bool = True  # Whether to extract video transcript

    # Pre-transcribed video support
    pre_transcribed: bool = False  # Whether video is already transcribed
    provided_metadata: Optional[Dict[str, Any]] = None  # Pre-provided metadata
    provided_transcript: Optional[str] = None  # Pre-provided transcript text
    provided_transcript_segments: Optional[
        List[Dict[str, Any]]
    ] = None  # Pre-provided transcript segments


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
    """YouTube processing service using Apify transcription API."""

    def __init__(self):
        """Initialize YouTube processor."""
        from .apify_service import ApifyYouTubeServiceError

        self.apify_service = None  # Initialize on demand
        self.ApifyYouTubeServiceError = ApifyYouTubeServiceError
        logger.info("YouTube processor initialized with Apify service")

    def _get_apify_service(self):
        """Get or create Apify service instance.

        Returns:
            ApifyYouTubeService instance

        Raises:
            ProcessingError: If Apify service cannot be initialized
        """
        if self.apify_service is None:
            try:
                from .apify_service import ApifyYouTubeService

                self.apify_service = ApifyYouTubeService()
            except self.ApifyYouTubeServiceError as e:
                raise ProcessingError(
                    f"Failed to initialize Apify service: {e}. Please check your APIFY_API_TOKEN configuration."
                )
        return self.apify_service

    async def process_url(
        self, url: str, config: Optional[YouTubeConfig] = None
    ) -> YouTubeDownloadResult:
        """Process YouTube URL using Apify transcription service."""
        start_time = time.time()
        config = config or YouTubeConfig()

        try:
            logger.info(
                "Starting YouTube processing with Apify service",
                url=url,
                pre_transcribed=config.pre_transcribed,
            )

            # Handle pre-transcribed videos
            if config.pre_transcribed:
                logger.info("Processing pre-transcribed video", url=url)
                return self._process_pre_transcribed_video(url, config, start_time)

            # Get Apify service (will fail if not configured)
            apify_service = self._get_apify_service()

            # Check service health
            is_healthy = await apify_service.health_check()
            if not is_healthy:
                raise ProcessingError(
                    "Apify service is not available. Please check your APIFY_API_TOKEN configuration."
                )

            # Call Apify service
            service_result = await apify_service.transcribe_video(
                video_url=url,
                extract_metadata=config.extract_metadata,
                extract_transcript=config.extract_transcript,
                use_proxy=config.use_proxy,
            )

            # Convert Apify service response to our result format
            result = self._convert_apify_result(service_result, start_time)

            logger.info(
                "YouTube processing completed successfully",
                url=url,
                has_transcript=bool(result.transcript),
                has_metadata=bool(result.metadata),
            )

            return result

        except self.ApifyYouTubeServiceError as e:
            error_msg = f"Apify service error: {str(e)}"
            logger.error("Apify service failed", url=url, error=str(e))
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
                error_message=error_msg,
            )
        except ProcessingError as e:
            logger.error("Processing error", url=url, error=str(e))
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
                error_message=str(e),
            )
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
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
                error_message=error_msg,
            )

    def _process_pre_transcribed_video(
        self, url: str, config: YouTubeConfig, start_time: float
    ) -> YouTubeDownloadResult:
        """Process a pre-transcribed video using provided metadata and transcript.

        Args:
            url: YouTube video URL
            config: Configuration with pre-provided data
            start_time: Processing start time

        Returns:
            YouTubeDownloadResult with provided data
        """
        logger.info("Processing pre-transcribed video", url=url)

        # Extract video ID from URL
        video_id = self._extract_video_id(url)

        # Create metadata from provided data
        metadata = None
        if config.provided_metadata:
            metadata = YouTubeMetadata(
                id=video_id,
                title=config.provided_metadata.get("title", "Unknown Title"),
                description=config.provided_metadata.get("description", ""),
                uploader=config.provided_metadata.get("channel", "Unknown Channel"),
                upload_date=config.provided_metadata.get("uploadDate", ""),
                duration=float(config.provided_metadata.get("duration", 0)),
                view_count=int(config.provided_metadata.get("viewCount", 0)),
                like_count=config.provided_metadata.get("likeCount"),
                comment_count=config.provided_metadata.get("commentCount"),
                tags=config.provided_metadata.get("tags", []),
                categories=config.provided_metadata.get("categories", []),
                thumbnail_url=config.provided_metadata.get("thumbnail", ""),
                webpage_url=f"https://www.youtube.com/watch?v={video_id}",
                channel_id=config.provided_metadata.get("channelId", ""),
                channel_url=config.provided_metadata.get("channelUrl", ""),
                playlist_id=None,
                playlist_title=None,
                playlist_index=None,
            )

        # Use provided transcript
        transcript = config.provided_transcript or ""
        transcript_segments = config.provided_transcript_segments or []

        # Store transcript segments in the transcript field as a dict
        transcript_data = {"text": transcript, "segments": transcript_segments}

        return YouTubeDownloadResult(
            metadata=metadata,
            transcript=transcript_data,
            transcript_languages=["en"],  # Default to English
            formats=[],
            total_formats=0,
            video_path=None,
            output_file=None,
            video_download=None,
            transcript_text=transcript,  # Legacy field
            processing_time=time.time() - start_time,
            success=True,
            error_message=None,
        )

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL.

        Args:
            url: YouTube video URL

        Returns:
            Video ID string
        """
        import re
        from urllib.parse import parse_qs, urlparse

        # Handle different YouTube URL formats
        if "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        elif "youtube.com/watch" in url:
            parsed = urlparse(url)
            return parse_qs(parsed.query).get("v", ["unknown"])[0]
        elif "youtube.com/embed/" in url:
            return url.split("youtube.com/embed/")[1].split("?")[0]
        else:
            # Try to extract 11-character video ID pattern
            match = re.search(r"[a-zA-Z0-9_-]{11}", url)
            return match.group(0) if match else "unknown"

    def _convert_apify_result(
        self, service_result: Dict[str, Any], start_time: float
    ) -> YouTubeDownloadResult:
        """Convert Apify service result to YouTubeDownloadResult format.

        Args:
            service_result: Result from Apify service
            start_time: Processing start time

        Returns:
            YouTubeDownloadResult object
        """
        # Check if the result indicates an error
        error_message = service_result.get("error")
        if error_message:
            logger.error("Apify result contains error", error=error_message)
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
                error_message=error_message,
            )

        # Check if we have any meaningful content (metadata or transcript)
        has_metadata = bool(service_result.get("metadata"))
        has_transcript = bool(service_result.get("transcript"))

        if not has_metadata and not has_transcript:
            error_msg = "No metadata or transcript found in Apify result"
            logger.error(
                "Apify result is empty", result_keys=list(service_result.keys())
            )
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
                error_message=error_msg,
            )

        # Convert metadata if available
        metadata = None
        if service_result.get("metadata"):
            metadata_dict = service_result["metadata"]

            # Extract thumbnail URL from thumbnails array
            thumbnail_url = ""
            if metadata_dict.get("thumbnails") and len(metadata_dict["thumbnails"]) > 0:
                thumbnail_url = metadata_dict["thumbnails"][0].get("url", "")

            # Parse duration from various possible formats
            duration = 0
            if metadata_dict.get("duration"):
                duration = self._parse_duration(str(metadata_dict["duration"]))
            elif metadata_dict.get("lengthSeconds"):
                duration = float(metadata_dict["lengthSeconds"])

            metadata = YouTubeMetadata(
                id=metadata_dict.get("videoId", ""),
                title=metadata_dict.get("title", ""),
                description=metadata_dict.get("description", ""),
                uploader=metadata_dict.get("channelName", ""),
                upload_date=metadata_dict.get("publishDate", ""),
                duration=duration,
                view_count=int(metadata_dict.get("viewCount", 0)),
                like_count=metadata_dict.get("likeCount"),
                comment_count=metadata_dict.get("commentCount"),
                tags=metadata_dict.get("keywords", []),
                categories=[metadata_dict.get("category", "")]
                if metadata_dict.get("category")
                else [],
                thumbnail_url=thumbnail_url,
                webpage_url=metadata_dict.get("url", ""),
                channel_id=metadata_dict.get("channelId", ""),
                channel_url=metadata_dict.get("channelUrl", ""),
                playlist_id=None,
                playlist_title=None,
                playlist_index=None,
            )

        # Process transcript
        transcript_segments = []
        transcript_text = ""
        transcript_languages = ["en"]  # Default to English

        if service_result.get("transcript"):
            transcript_data = service_result["transcript"]
            # Handle nested transcript structure from Apify
            if isinstance(transcript_data, dict) and "transcript" in transcript_data:
                transcript_segments_data = transcript_data["transcript"]
                if isinstance(transcript_segments_data, list):
                    transcript_segments = transcript_segments_data
                    # Format with timecodes like audio/video files
                    transcript_text = self._format_transcript_with_timecodes(
                        transcript_segments_data
                    )
            elif isinstance(transcript_data, list):
                # Handle segment format
                transcript_segments = transcript_data
                # Format with timecodes like audio/video files
                transcript_text = self._format_transcript_with_timecodes(
                    transcript_data
                )
            elif isinstance(transcript_data, str):
                # Handle plain text format
                transcript_text = transcript_data

        # Store transcript data in the expected format
        transcript_result = {"text": transcript_text, "segments": transcript_segments}

        return YouTubeDownloadResult(
            metadata=metadata,
            transcript=transcript_result,
            transcript_languages=transcript_languages,
            formats=[],  # Apify doesn't provide format info
            total_formats=0,
            video_path=None,  # Apify doesn't download videos by default
            output_file=None,
            video_download=None,
            # Legacy fields for backward compatibility
            transcript_text=transcript_text,
            transcript_language="en",
            processing_time=time.time() - start_time,
            success=True,
            error_message=None,
        )

    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string to float seconds.

        Args:
            duration_str: Duration string (e.g., "3:45", "1:23:45", "180")

        Returns:
            Duration in seconds as float
        """
        if not duration_str:
            return 0.0

        try:
            # If it's already a number, return it
            return float(duration_str)
        except ValueError:
            pass

        try:
            # Parse time format (e.g., "3:45" or "1:23:45")
            parts = duration_str.split(":")
            if len(parts) == 2:  # MM:SS
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:  # HH:MM:SS
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            else:
                return 0.0
        except (ValueError, IndexError):
            return 0.0

    def _format_transcript_with_timecodes(self, segments: List[Dict[str, Any]]) -> str:
        """Format transcript segments with timecodes in the same format as audio/video files.

        Args:
            segments: List of transcript segments with timing information

        Returns:
            Formatted transcript text with timecodes
        """
        if not segments:
            return ""

        formatted_lines = []
        for segment in segments:
            if isinstance(segment, dict):
                text = segment.get("text", "").strip()
                start_time = segment.get("start", segment.get("startTime", None))
                if text and start_time is not None:
                    # Format timestamp as [X.Xs] like in audio/video processing
                    formatted_lines.append(f"[{float(start_time):.1f}s] {text}")
                elif text:
                    # If no timing info, just add the text
                    formatted_lines.append(text)

        return "\n".join(formatted_lines) if formatted_lines else ""

    async def process_playlist(
        self, playlist_url: str, config: Optional[YouTubeConfig] = None
    ) -> List[YouTubeDownloadResult]:
        """Process YouTube playlist using Apify service.

        Args:
            playlist_url: YouTube playlist URL
            config: Processing configuration

        Returns:
            List of YouTubeDownloadResult objects for each video
        """
        config = config or YouTubeConfig()

        # For now, playlist processing is not implemented
        # This would require extracting individual video URLs and processing them
        # or using a specialized Apify actor for playlist processing
        logger.warning("Playlist processing not yet implemented with Apify service")
        raise ProcessingError(
            "Playlist processing is not currently supported. Please process individual video URLs."
        )

    def cleanup(self, result: YouTubeDownloadResult) -> None:
        """Clean up temporary files."""
        # For Apify service, cleanup is minimal since files are managed by Apify
        logger.debug("Cleanup completed for Apify service result")

    async def process(
        self, file_path: Path, config: Optional[ProcessingConfig] = None
    ) -> ProcessingResult:
        """Process content from file.

        This method is required by the BaseProcessor interface but is not applicable
        for YouTube processing. Use process_url instead.

        Raises:
            ProcessingError: Always raises this error as this method is not supported
        """
        raise ProcessingError(
            "YouTubeProcessor does not support file processing. Use process_url instead."
        )

    def supports_format(self, format_type: str) -> bool:
        """Check if processor supports the given format.

        Args:
            format_type: Format type to check

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() in ["youtube", "yt"]
