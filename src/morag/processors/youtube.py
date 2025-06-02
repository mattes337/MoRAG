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

from morag.core.config import settings
from morag.core.exceptions import ProcessingError, ExternalServiceError

logger = structlog.get_logger()

@dataclass
class YouTubeConfig:
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
class YouTubeDownloadResult:
    """Result of YouTube download operation."""
    video_path: Optional[Path]
    audio_path: Optional[Path]
    subtitle_paths: List[Path]
    thumbnail_paths: List[Path]
    metadata: YouTubeMetadata
    download_time: float
    file_size: int
    temp_files: List[Path] = field(default_factory=list)

class YouTubeProcessor:
    """YouTube processing service using yt-dlp."""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "morag_youtube"
        self.temp_dir.mkdir(exist_ok=True)
        
    async def process_url(
        self,
        url: str,
        config: YouTubeConfig
    ) -> YouTubeDownloadResult:
        """Process YouTube URL with download and metadata extraction."""
        start_time = time.time()
        
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
                download_time=0.0,
                file_size=0,
                temp_files=[]
            )
            
            # If only metadata extraction is requested, return early
            if config.extract_metadata_only:
                result.download_time = time.time() - start_time
                return result
            
            # Download video/audio
            if config.extract_audio:
                audio_path = await self._download_audio(url, config)
                if audio_path:
                    result.audio_path = audio_path
                    result.temp_files.append(audio_path)
                    result.file_size += audio_path.stat().st_size
            else:
                video_path = await self._download_video(url, config)
                if video_path:
                    result.video_path = video_path
                    result.temp_files.append(video_path)
                    result.file_size += video_path.stat().st_size
            
            # Download subtitles if requested
            if config.download_subtitles:
                subtitle_paths = await self._download_subtitles(url, config)
                result.subtitle_paths = subtitle_paths
                result.temp_files.extend(subtitle_paths)
            
            # Download thumbnails if requested
            if config.download_thumbnails:
                thumbnail_paths = await self._download_thumbnails(url, config)
                result.thumbnail_paths = thumbnail_paths
                result.temp_files.extend(thumbnail_paths)
            
            download_time = time.time() - start_time
            result.download_time = download_time
            
            logger.info("YouTube processing completed",
                       url=url,
                       download_time=download_time,
                       video_downloaded=result.video_path is not None,
                       audio_downloaded=result.audio_path is not None,
                       subtitles_count=len(result.subtitle_paths),
                       thumbnails_count=len(result.thumbnail_paths))
            
            return result
            
        except Exception as e:
            logger.error("YouTube processing failed", 
                        url=url,
                        error=str(e))
            raise ProcessingError(f"YouTube processing failed: {str(e)}")
    
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
            
            # Parse metadata
            metadata = YouTubeMetadata(
                id=info.get('id', ''),
                title=info.get('title', ''),
                description=info.get('description', ''),
                uploader=info.get('uploader', ''),
                upload_date=info.get('upload_date', ''),
                duration=float(info.get('duration', 0)),
                view_count=int(info.get('view_count', 0)),
                like_count=info.get('like_count'),
                comment_count=info.get('comment_count'),
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
            
            logger.debug("YouTube metadata extracted",
                        title=metadata.title,
                        duration=metadata.duration,
                        uploader=metadata.uploader)
            
            return metadata
            
        except Exception as e:
            logger.error("YouTube metadata extraction failed",
                        url=url,
                        error=str(e))
            raise ExternalServiceError(f"YouTube metadata extraction failed: {str(e)}", "yt-dlp")
    
    async def _download_video(self, url: str, config: YouTubeConfig) -> Optional[Path]:
        """Download video file."""
        try:
            output_path = self.temp_dir / f"video_{int(time.time())}_%(title)s.%(ext)s"
            
            ydl_opts = {
                'format': self._get_video_format(config),
                'outtmpl': str(output_path),
                'quiet': True,
                'no_warnings': True,
            }
            
            # Add filesize limit if specified
            if config.max_filesize:
                ydl_opts['format'] += f'[filesize<{config.max_filesize}]'
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [url])
            
            # Find the downloaded file
            downloaded_files = list(self.temp_dir.glob(f"video_{int(time.time())}_*"))
            if downloaded_files:
                return downloaded_files[0]
            
            return None
            
        except Exception as e:
            logger.error("Video download failed",
                        url=url,
                        error=str(e))
            return None
    
    async def _download_audio(self, url: str, config: YouTubeConfig) -> Optional[Path]:
        """Download audio file."""
        try:
            output_path = self.temp_dir / f"audio_{int(time.time())}_%(title)s.%(ext)s"
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(output_path),
                'quiet': True,
                'no_warnings': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }]
            }
            
            # Add filesize limit if specified
            if config.max_filesize:
                ydl_opts['format'] += f'[filesize<{config.max_filesize}]'
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [url])
            
            # Find the downloaded file
            downloaded_files = list(self.temp_dir.glob(f"audio_{int(time.time())}_*"))
            if downloaded_files:
                return downloaded_files[0]
            
            return None
            
        except Exception as e:
            logger.error("Audio download failed",
                        url=url,
                        error=str(e))
            return None
    
    async def _download_subtitles(self, url: str, config: YouTubeConfig) -> List[Path]:
        """Download subtitle files."""
        try:
            output_path = self.temp_dir / f"subs_{int(time.time())}_%(title)s.%(ext)s"
            
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': config.subtitle_languages,
                'skip_download': True,
                'outtmpl': str(output_path),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [url])
            
            # Find downloaded subtitle files
            subtitle_files = list(self.temp_dir.glob(f"subs_{int(time.time())}_*.vtt")) + \
                            list(self.temp_dir.glob(f"subs_{int(time.time())}_*.srt"))
            
            return subtitle_files
            
        except Exception as e:
            logger.error("Subtitle download failed",
                        url=url,
                        error=str(e))
            return []
    
    async def _download_thumbnails(self, url: str, config: YouTubeConfig) -> List[Path]:
        """Download thumbnail files."""
        try:
            output_path = self.temp_dir / f"thumb_{int(time.time())}_%(title)s.%(ext)s"
            
            ydl_opts = {
                'writethumbnail': True,
                'skip_download': True,
                'outtmpl': str(output_path),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [url])
            
            # Find downloaded thumbnail files
            thumbnail_files = list(self.temp_dir.glob(f"thumb_{int(time.time())}_*.jpg")) + \
                             list(self.temp_dir.glob(f"thumb_{int(time.time())}_*.png")) + \
                             list(self.temp_dir.glob(f"thumb_{int(time.time())}_*.webp"))
            
            return thumbnail_files
            
        except Exception as e:
            logger.error("Thumbnail download failed",
                        url=url,
                        error=str(e))
            return []
    
    def _get_video_format(self, config: YouTubeConfig) -> str:
        """Get video format string based on configuration."""
        if config.format_preference == "mp4":
            return "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"
        elif config.format_preference == "webm":
            return "bv*[ext=webm]+ba[ext=webm]/b[ext=webm]/bv*+ba/b"
        else:
            return "bv*+ba/b"  # Best video + best audio
    
    async def process_playlist(self, url: str, config: YouTubeConfig) -> List[YouTubeDownloadResult]:
        """Process YouTube playlist."""
        try:
            logger.info("Processing YouTube playlist", url=url)
            
            # Extract playlist info
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = await asyncio.to_thread(
                    ydl.extract_info,
                    url,
                    download=False
                )
            
            results = []
            entries = playlist_info.get('entries', [])
            
            for entry in entries:
                if entry:
                    video_url = entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}"
                    try:
                        result = await self.process_url(video_url, config)
                        results.append(result)
                    except Exception as e:
                        logger.warning("Failed to process playlist entry",
                                     video_url=video_url,
                                     error=str(e))
                        continue
            
            logger.info("Playlist processing completed",
                       url=url,
                       total_videos=len(entries),
                       successful_downloads=len(results))
            
            return results
            
        except Exception as e:
            logger.error("Playlist processing failed",
                        url=url,
                        error=str(e))
            raise ProcessingError(f"Playlist processing failed: {str(e)}")
    
    def cleanup_temp_files(self, temp_files: List[Path]):
        """Clean up temporary files."""
        for file_path in temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug("Temporary file cleaned up", file_path=str(file_path))
            except Exception as e:
                logger.warning("Failed to clean up temporary file",
                             file_path=str(file_path),
                             error=str(e))

# Global instance
youtube_processor = YouTubeProcessor()
