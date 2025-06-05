"""YouTube processing with yt-dlp for video download and metadata extraction."""

import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import asyncio
import structlog
import time
import json

import yt_dlp

from morag_core.exceptions import ProcessingError, ExternalServiceError
from morag_core.interfaces.processor import BaseProcessor, ProcessingConfig, ProcessingResult

logger = structlog.get_logger(__name__)

@dataclass
class YouTubeConfig(ProcessingConfig):
    """Configuration for YouTube processing."""
    quality: str = "best"
    format_preference: str = "mp4"
    extract_audio: bool = True
    download_subtitles: bool = True
    subtitle_languages: List[str] = field(default_factory=lambda: ["en"])
    max_filesize: Optional[str] = "500M"
    download_thumbnails: bool = True
    extract_metadata_only: bool = False

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
    """Result of YouTube download operation."""
    video_path: Optional[Path]
    audio_path: Optional[Path]
    subtitle_paths: List[Path]
    thumbnail_paths: List[Path]
    metadata: YouTubeMetadata
    file_size: int
    temp_files: List[Path] = field(default_factory=list)

class YouTubeProcessor(BaseProcessor):
    """YouTube processing service using yt-dlp."""
    
    def __init__(self):
        """Initialize YouTube processor."""
        self.temp_dir = Path(tempfile.gettempdir()) / "morag_youtube"
        self.temp_dir.mkdir(exist_ok=True)
        
    async def process_url(
        self,
        url: str,
        config: Optional[YouTubeConfig] = None
    ) -> YouTubeDownloadResult:
        """Process YouTube URL with download and metadata extraction."""
        start_time = time.time()
        config = config or YouTubeConfig()
        
        try:
            logger.info("Starting YouTube processing", url=url)
            
            # Extract metadata first
            metadata = await self._extract_metadata_only(url)
            
            # Initialize result
            result = YouTubeDownloadResult(
                video_path=None,
                audio_path=None,
                subtitle_paths=[],
                thumbnail_paths=[],
                metadata=metadata,
                processing_time=0.0,
                file_size=0,
                temp_files=[],
                success=True
            )
            
            # If only metadata extraction is requested, return early
            if config.extract_metadata_only:
                result.processing_time = time.time() - start_time
                return result
            
            # Download video
            download_result = await self._download_video(url, config)
            
            # Update result with download info
            result.video_path = download_result.get('video_path')
            result.audio_path = download_result.get('audio_path')
            result.subtitle_paths = download_result.get('subtitle_paths', [])
            result.thumbnail_paths = download_result.get('thumbnail_paths', [])
            result.file_size = download_result.get('file_size', 0)
            result.temp_files = download_result.get('temp_files', [])
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.exception("Error processing YouTube URL", url=url, error=str(e))
            return YouTubeDownloadResult(
                video_path=None,
                audio_path=None,
                subtitle_paths=[],
                thumbnail_paths=[],
                metadata=None,  # type: ignore
                processing_time=time.time() - start_time,
                file_size=0,
                temp_files=[],
                success=False,
                error_message=str(e)
            )
    
    async def _extract_metadata_only(self, url: str) -> YouTubeMetadata:
        """Extract metadata without downloading."""
        try:
            logger.debug("Extracting YouTube metadata", url=url)
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.to_thread(
                    ydl.extract_info,
                    url,
                    download=False
                )
            
            if not info:
                raise ProcessingError(f"Failed to extract metadata from {url}")
            
            # Extract relevant metadata
            metadata = YouTubeMetadata(
                id=info.get('id', ''),
                title=info.get('title', ''),
                description=info.get('description', ''),
                uploader=info.get('uploader', ''),
                upload_date=info.get('upload_date', ''),
                duration=float(info.get('duration', 0)),
                view_count=int(info.get('view_count', 0)),
                like_count=int(info.get('like_count', 0)) if info.get('like_count') else None,
                comment_count=int(info.get('comment_count', 0)) if info.get('comment_count') else None,
                tags=info.get('tags', []),
                categories=info.get('categories', []),
                thumbnail_url=info.get('thumbnail', ''),
                webpage_url=info.get('webpage_url', ''),
                channel_id=info.get('channel_id', ''),
                channel_url=info.get('channel_url', ''),
                playlist_id=info.get('playlist_id'),
                playlist_title=info.get('playlist_title'),
                playlist_index=info.get('playlist_index')
            )
            
            return metadata
            
        except Exception as e:
            logger.exception("Error extracting YouTube metadata", url=url, error=str(e))
            raise ProcessingError(f"Failed to extract YouTube metadata: {str(e)}")
    
    async def _download_video(self, url: str, config: YouTubeConfig) -> Dict[str, Any]:
        """Download video with yt-dlp."""
        try:
            logger.debug("Downloading YouTube video", url=url)
            
            # Create output directory
            output_dir = self.temp_dir / f"youtube_{int(time.time())}"
            output_dir.mkdir(exist_ok=True)
            
            # Prepare format selection
            format_selection = config.quality
            if config.extract_audio:
                format_selection = f"{format_selection}+bestaudio/best"
            
            # Prepare subtitle options
            subtitle_options = {}
            if config.download_subtitles:
                subtitle_options = {
                    'writesubtitles': True,
                    'subtitleslangs': config.subtitle_languages,
                    'writeautomaticsub': True,
                }
            
            # Prepare yt-dlp options
            ydl_opts = {
                'format': format_selection,
                'outtmpl': str(output_dir / '%(title)s-%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'writethumbnail': config.download_thumbnails,
                'postprocessors': [],
                **subtitle_options
            }
            
            # Add filesize limit if specified
            if config.max_filesize:
                ydl_opts['max_filesize'] = config.max_filesize
            
            # Add audio extraction if requested
            if config.extract_audio:
                ydl_opts['postprocessors'].append({
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                })
            
            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.to_thread(
                    ydl.extract_info,
                    url,
                    download=True
                )
            
            if not info:
                raise ProcessingError(f"Failed to download video from {url}")
            
            # Find downloaded files
            video_path = None
            audio_path = None
            subtitle_paths = []
            thumbnail_paths = []
            temp_files = []
            
            for file_path in output_dir.glob("*"):
                temp_files.append(file_path)
                
                if file_path.suffix.lower() in [".mp4", ".webm", ".mkv"]:
                    video_path = file_path
                elif file_path.suffix.lower() in [".mp3", ".m4a", ".wav"]:
                    audio_path = file_path
                elif file_path.suffix.lower() in [".vtt", ".srt", ".ass"]:
                    subtitle_paths.append(file_path)
                elif file_path.suffix.lower() in [".jpg", ".png", ".webp"]:
                    thumbnail_paths.append(file_path)
            
            # Calculate total file size
            file_size = sum(file.stat().st_size for file in temp_files if file.is_file())
            
            return {
                'video_path': video_path,
                'audio_path': audio_path,
                'subtitle_paths': subtitle_paths,
                'thumbnail_paths': thumbnail_paths,
                'file_size': file_size,
                'temp_files': temp_files
            }
            
        except Exception as e:
            logger.exception("Error downloading YouTube video", url=url, error=str(e))
            raise ProcessingError(f"Failed to download YouTube video: {str(e)}")
    
    async def process_playlist(self, url: str, config: Optional[YouTubeConfig] = None) -> List[YouTubeDownloadResult]:
        """Process YouTube playlist."""
        config = config or YouTubeConfig()
        results = []
        
        try:
            logger.info("Processing YouTube playlist", url=url)
            
            # Extract playlist info
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = await asyncio.to_thread(
                    ydl.extract_info,
                    url,
                    download=False
                )
            
            if not playlist_info or 'entries' not in playlist_info:
                raise ProcessingError(f"Failed to extract playlist info from {url}")
            
            # Process each video in the playlist
            for entry in playlist_info['entries']:
                video_url = entry.get('url')
                if not video_url:
                    continue
                
                try:
                    result = await self.process_url(video_url, config)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        "Error processing playlist video",
                        url=video_url,
                        error=str(e)
                    )
                    # Continue with next video
            
            return results
            
        except Exception as e:
            logger.exception("Error processing YouTube playlist", url=url, error=str(e))
            raise ProcessingError(f"Failed to process YouTube playlist: {str(e)}")
    
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
    
    def cleanup(self, result: YouTubeDownloadResult) -> None:
        """Clean up temporary files."""
        if not result.temp_files:
            return
        
        logger.debug(f"Cleaning up {len(result.temp_files)} temporary files")
        
        for file_path in result.temp_files:
            try:
                if file_path.exists():
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        for child in file_path.glob("*"):
                            if child.is_file():
                                child.unlink()
                        file_path.rmdir()
            except Exception as e:
                logger.warning(f"Failed to clean up file {file_path}: {str(e)}")