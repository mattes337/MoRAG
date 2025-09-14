"""YouTube service for high-level YouTube processing operations using external API."""

import asyncio
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import structlog

from morag_core.interfaces.service import BaseService
from morag_core.exceptions import ProcessingError

from .processor import YouTubeProcessor, YouTubeConfig, YouTubeDownloadResult
from .apify_service import ApifyYouTubeService, ApifyYouTubeServiceError

logger = structlog.get_logger(__name__)

class YouTubeService(BaseService):
    """Service for processing YouTube videos using Apify transcription API.

    This service provides high-level methods for transcribing videos and
    extracting metadata using Apify's cloud infrastructure.
    """

    def __init__(self):
        """Initialize YouTube service."""
        self.processor = YouTubeProcessor()
        self.apify_service = None  # Initialize on demand

    async def initialize(self) -> bool:
        """Initialize the service.

        Returns:
            True if initialization was successful
        """
        return True

    async def shutdown(self) -> None:
        """Shutdown the service and release resources."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Dictionary with health status information
        """
        try:
            # Check external service health
            external_health = await self.external_service.health_check()

            return {
                "status": "healthy" if external_health["status"] == "healthy" else "unhealthy",
                "external_service": external_health,
                "processor_available": True
            }

        except Exception as e:
            logger.warning("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "external_service": {"status": "unhealthy", "error": str(e)},
                "processor_available": False
            }
    
    async def process_video(self, url: str, config: Optional[YouTubeConfig] = None) -> YouTubeDownloadResult:
        """Process a single YouTube video using external service.

        Args:
            url: YouTube video URL
            config: Processing configuration

        Returns:
            YouTubeDownloadResult with video information and paths
        """
        return await self.processor.process_url(url, config)
    
    async def process_videos(self, urls: List[str], config: Optional[YouTubeConfig] = None) -> List[Union[YouTubeDownloadResult, BaseException]]:
        """Process multiple YouTube videos using external service.

        Args:
            urls: List of YouTube video URLs
            config: Processing configuration

        Returns:
            List of YouTubeDownloadResult objects
        """
        tasks = [self.process_video(url, config) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_playlist(self, playlist_url: str, config: Optional[YouTubeConfig] = None) -> List[YouTubeDownloadResult]:
        """Process a YouTube playlist.
        
        Args:
            playlist_url: YouTube playlist URL
            config: Processing configuration
            
        Returns:
            List of YouTubeDownloadResult objects for each video in the playlist
        """
        return await self.processor.process_playlist(playlist_url, config)
    
    async def extract_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from a YouTube video without downloading.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary containing video metadata
        """
        config = YouTubeConfig(extract_metadata_only=True)
        result = await self.processor.process_url(url, config)
        
        if not result.success or not result.metadata:
            raise ProcessingError(f"Failed to extract metadata: {result.error_message}")
        
        # Convert metadata to dictionary
        return {
            'id': result.metadata.id,
            'title': result.metadata.title,
            'description': result.metadata.description,
            'uploader': result.metadata.uploader,
            'upload_date': result.metadata.upload_date,
            'duration': result.metadata.duration,
            'view_count': result.metadata.view_count,
            'like_count': result.metadata.like_count,
            'comment_count': result.metadata.comment_count,
            'tags': result.metadata.tags,
            'categories': result.metadata.categories,
            'thumbnail_url': result.metadata.thumbnail_url,
            'webpage_url': result.metadata.webpage_url,
            'channel_id': result.metadata.channel_id,
            'channel_url': result.metadata.channel_url,
            'playlist_id': result.metadata.playlist_id,
            'playlist_title': result.metadata.playlist_title,
            'playlist_index': result.metadata.playlist_index,
        }
    
    async def download_video(self, url: str, output_dir: Optional[Path] = None, 
                           quality: str = "best", extract_audio: bool = True,
                           download_subtitles: bool = True) -> YouTubeDownloadResult:
        """Download a YouTube video with specified options.
        
        Args:
            url: YouTube video URL
            output_dir: Directory to save downloaded files (uses temp dir if None)
            quality: Video quality ("best", "worst", or specific format)
            extract_audio: Whether to extract audio as separate file
            download_subtitles: Whether to download subtitles
            
        Returns:
            YouTubeDownloadResult with video information and paths
        """
        config = YouTubeConfig(
            quality=quality,
            extract_audio=extract_audio,
            download_subtitles=download_subtitles
        )
        
        result = await self.process_video(url, config)
        
        # If output_dir is specified, move files there
        if output_dir and result.success:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Move video file
            if result.video_path and result.video_path.exists():
                new_video_path = output_dir / result.video_path.name
                result.video_path.rename(new_video_path)
                result.video_path = new_video_path
            
            # Move audio file
            if result.audio_path and result.audio_path.exists():
                new_audio_path = output_dir / result.audio_path.name
                result.audio_path.rename(new_audio_path)
                result.audio_path = new_audio_path
            
            # Move subtitle files
            new_subtitle_paths = []
            for sub_path in result.subtitle_paths:
                if sub_path.exists():
                    new_sub_path = output_dir / sub_path.name
                    sub_path.rename(new_sub_path)
                    new_subtitle_paths.append(new_sub_path)
            result.subtitle_paths = new_subtitle_paths
            
            # Move thumbnail files
            new_thumbnail_paths = []
            for thumb_path in result.thumbnail_paths:
                if thumb_path.exists():
                    new_thumb_path = output_dir / thumb_path.name
                    thumb_path.rename(new_thumb_path)
                    new_thumbnail_paths.append(new_thumb_path)
            result.thumbnail_paths = new_thumbnail_paths
        
        return result
    
    async def download_audio(self, url: str, output_dir: Optional[Path] = None) -> Path:
        """Download only the audio from a YouTube video.
        
        Args:
            url: YouTube video URL
            output_dir: Directory to save downloaded files (uses temp dir if None)
            
        Returns:
            Path to the downloaded audio file
        """
        config = YouTubeConfig(
            extract_audio=True,
            download_subtitles=False,
            download_thumbnails=False
        )
        
        result = await self.process_video(url, config)
        
        if not result.success or not result.audio_path:
            raise ProcessingError(f"Failed to download audio: {result.error_message}")
        
        # If output_dir is specified, move file there
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            new_audio_path = output_dir / result.audio_path.name
            result.audio_path.rename(new_audio_path)
            return new_audio_path
        
        return result.audio_path
    
    async def download_subtitles(self, url: str, languages: Optional[List[str]] = None,
                               output_dir: Optional[Path] = None) -> List[Path]:
        """Download subtitles for a YouTube video.
        
        Args:
            url: YouTube video URL
            languages: List of language codes (default: ["en"])
            output_dir: Directory to save downloaded files (uses temp dir if None)
            
        Returns:
            List of paths to the downloaded subtitle files
        """
        languages = languages or ["en"]
        
        config = YouTubeConfig(
            extract_audio=False,
            download_subtitles=True,
            subtitle_languages=languages,
            download_thumbnails=False,
            extract_metadata_only=False
        )
        
        result = await self.process_video(url, config)
        
        if not result.success:
            raise ProcessingError(f"Failed to download subtitles: {result.error_message}")
        
        if not result.subtitle_paths:
            logger.warning("No subtitles found for the video", url=url, languages=languages)
            return []
        
        # If output_dir is specified, move files there
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            new_subtitle_paths = []
            for sub_path in result.subtitle_paths:
                if sub_path.exists():
                    new_sub_path = output_dir / sub_path.name
                    sub_path.rename(new_sub_path)
                    new_subtitle_paths.append(new_sub_path)
            return new_subtitle_paths
        
        return result.subtitle_paths
    
    async def download_thumbnail(self, url: str, output_dir: Optional[Path] = None) -> Path:
        """Download thumbnail for a YouTube video.
        
        Args:
            url: YouTube video URL
            output_dir: Directory to save downloaded files (uses temp dir if None)
            
        Returns:
            Path to the downloaded thumbnail file
        """
        config = YouTubeConfig(
            extract_audio=False,
            download_subtitles=False,
            download_thumbnails=True,
            extract_metadata_only=False
        )
        
        result = await self.process_video(url, config)
        
        if not result.success or not result.thumbnail_paths:
            raise ProcessingError(f"Failed to download thumbnail: {result.error_message}")
        
        # Get the first thumbnail
        thumbnail_path = result.thumbnail_paths[0]
        
        # If output_dir is specified, move file there
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            new_thumbnail_path = output_dir / thumbnail_path.name
            thumbnail_path.rename(new_thumbnail_path)
            return new_thumbnail_path
        
        return thumbnail_path



    async def transcribe_video(
        self,
        url: str,
        config: Optional[YouTubeConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        transcript: Optional[str] = None,
        transcript_segments: Optional[List[Dict[str, Any]]] = None
    ) -> YouTubeDownloadResult:
        """Transcribe a YouTube video using Apify service or process pre-transcribed content.

        Args:
            url: YouTube video URL
            config: Optional configuration for transcription
            metadata: Pre-provided metadata (for pre-transcribed videos)
            transcript: Pre-provided transcript text (for pre-transcribed videos)
            transcript_segments: Pre-provided transcript segments (for pre-transcribed videos)

        Returns:
            YouTubeDownloadResult containing transcription and metadata

        Raises:
            ProcessingError: If transcription fails or Apify is not configured
        """
        try:
            logger.info("Starting video transcription", url=url, pre_transcribed=bool(transcript))

            # Create config with pre-transcribed data if provided
            if metadata or transcript or transcript_segments:
                config = config or YouTubeConfig()
                config.pre_transcribed = True
                config.provided_metadata = metadata
                config.provided_transcript = transcript
                config.provided_transcript_segments = transcript_segments

            # Use the processor's process_url method
            result = await self.processor.process_url(url, config)

            if not result.success:
                raise ProcessingError(f"YouTube transcription failed: {result.error_message}")

            logger.info("Video transcription completed", url=url, success=result.success)
            return result

        except ProcessingError:
            raise
        except Exception as e:
            error_msg = f"Video transcription failed for {url}: {str(e)}"
            logger.error("Video transcription failed", url=url, error=str(e))
            raise ProcessingError(error_msg)


    async def transcribe_videos(
        self,
        urls: List[str],
        config: Optional[YouTubeConfig] = None
    ) -> List[YouTubeDownloadResult]:
        """Transcribe multiple YouTube videos.

        Args:
            urls: List of YouTube video URLs
            config: Optional configuration for transcription

        Returns:
            List of YouTubeDownloadResult objects

        Raises:
            ProcessingError: If transcription fails
        """
        results = []
        config = config or YouTubeConfig()

        logger.info("Starting batch transcription", video_count=len(urls))

        for url in urls:
            try:
                result = await self.transcribe_video(url, config)
                results.append(result)
            except Exception as e:
                logger.error("Failed to transcribe video in batch", url=url, error=str(e))
                # Create a failed result
                failed_result = YouTubeDownloadResult(
                    metadata=None,
                    transcript=None,
                    transcript_segments=[],
                    transcript_languages=[],
                    formats=[],
                    total_formats=0,
                    video_path=None,
                    output_file=None,
                    video_download=None,
                    processing_time=0.0,
                    success=False,
                    error_message=str(e)
                )
                results.append(failed_result)

        logger.info("Batch transcription completed",
                   video_count=len(urls),
                   successful_count=sum(1 for r in results if r.success))

        return results

    def cleanup(self, result: Union[YouTubeDownloadResult, List[YouTubeDownloadResult]]) -> None:
        """Clean up temporary files.
        
        Args:
            result: YouTubeDownloadResult or list of results to clean up
        """
        if isinstance(result, list):
            for single_result in result:
                if isinstance(single_result, YouTubeDownloadResult):
                    self.processor.cleanup(single_result)
        elif isinstance(result, YouTubeDownloadResult):
            self.processor.cleanup(result)