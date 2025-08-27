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
from .transcript import YouTubeTranscriptService, YouTubeTranscript

logger = structlog.get_logger(__name__)

@dataclass
class YouTubeConfig(ProcessingConfig):
    """Configuration for YouTube processing."""
    quality: str = "best"
    format_preference: str = "mp4"
    extract_audio: bool = True
    download_subtitles: bool = True
    subtitle_languages: List[str] = field(default_factory=lambda: ["en"])
    max_filesize: Optional[str] = None  # Disable max filesize to avoid yt-dlp comparison bug
    download_thumbnails: bool = True
    extract_metadata_only: bool = False
    # Transcript-specific options
    extract_transcript: bool = True
    transcript_language: Optional[str] = None  # None = use original language
    transcript_format: str = "text"  # "text", "srt", "vtt"
    # Transcription strategy options
    prefer_audio_transcription: bool = True  # Prefer audio transcription when cookies available
    cookies_file: Optional[str] = None  # Path to cookies file (overrides env var)
    transcript_only: bool = False  # If True, skip video download and use only transcript API
    # Metadata override options
    skip_metadata_extraction: bool = False  # If True, skip YouTube metadata extraction
    provided_metadata: Optional[Dict[str, Any]] = None  # User-provided metadata to use instead

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
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    subtitle_paths: List[Path] = field(default_factory=list)
    thumbnail_paths: List[Path] = field(default_factory=list)
    metadata: Optional[YouTubeMetadata] = None
    file_size: int = 0
    temp_files: List[Path] = field(default_factory=list)
    # Transcript data
    transcript_path: Optional[Path] = None
    transcript_text: Optional[str] = None
    transcript_language: Optional[str] = None

class YouTubeProcessor(BaseProcessor):
    """YouTube processing service using yt-dlp."""
    
    def __init__(self):
        """Initialize YouTube processor."""
        self.temp_dir = Path(tempfile.gettempdir()) / "morag_youtube"
        self.temp_dir.mkdir(exist_ok=True)
        self.transcript_service = YouTubeTranscriptService()
        self.audio_processor = None
        self._initialize_audio_processor()

    def _initialize_audio_processor(self):
        """Initialize audio processor for transcription fallback."""
        try:
            from morag_audio import AudioProcessor, AudioConfig

            # Create audio config optimized for YouTube transcription
            audio_config = AudioConfig(
                model_size="medium",  # Good balance of speed and accuracy
                language=None,  # Auto-detect language
                enable_diarization=False,  # Not needed for YouTube transcription
                enable_topic_segmentation=False,  # Not needed for basic transcription
                device="auto",
                word_timestamps=True,
                use_rest_api=False  # Use local whisper by default
            )

            self.audio_processor = AudioProcessor(audio_config)
            logger.info("Audio processor initialized for YouTube transcription")

        except ImportError:
            logger.warning("Audio processor not available, will use direct transcript extraction only")
            self.audio_processor = None
        except Exception as e:
            logger.warning("Failed to initialize audio processor", error=str(e))
            self.audio_processor = None

    async def process_url(
        self,
        url: str,
        config: Optional[YouTubeConfig] = None,
        provided_file: Optional[Path] = None
    ) -> YouTubeDownloadResult:
        """Process YouTube URL with download and metadata extraction.

        Args:
            url: YouTube video URL
            config: Processing configuration
            provided_file: Optional pre-downloaded file (video or transcript)
                          If provided, skips download/transcription and uses this file
        """
        start_time = time.time()
        config = config or YouTubeConfig()

        try:
            logger.info("Starting YouTube processing", url=url, provided_file=str(provided_file) if provided_file else None)

            # Extract metadata only if not in transcript-only mode or if explicitly requested
            metadata = None
            if config.skip_metadata_extraction and config.provided_metadata:
                # Use provided metadata instead of extracting from YouTube
                logger.info("Using provided metadata instead of extracting from YouTube", url=url)
                metadata = self._create_metadata_from_dict(config.provided_metadata, url)
            elif not config.transcript_only or config.extract_metadata_only:
                # Check if we have cookies available before attempting metadata extraction
                cookies_available = self._has_cookies_available(config)

                if config.transcript_only and not cookies_available:
                    # In transcript-only mode without cookies, skip metadata extraction to avoid yt-dlp bot detection
                    logger.info("Skipping metadata extraction in transcript-only mode without cookies to avoid bot detection",
                               url=url)
                    metadata = None
                else:
                    try:
                        metadata = await self._extract_metadata_only(url, config)
                    except Exception as e:
                        if config.transcript_only:
                            # In transcript-only mode, metadata extraction failure is not critical
                            logger.warning("Metadata extraction failed in transcript-only mode, continuing without metadata",
                                         url=url, error=str(e))
                            metadata = None
                        else:
                            # In normal mode, metadata extraction failure is critical
                            raise

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

            # Handle provided file case
            if provided_file and provided_file.exists():
                logger.info("Using provided file instead of downloading", provided_file=str(provided_file))
                download_result = await self._handle_provided_file(provided_file, config)
            # Skip video download if transcript_only mode is enabled
            elif config.transcript_only:
                logger.info("Transcript-only mode: skipping video download")
                download_result = {
                    'video_path': None,
                    'audio_path': None,
                    'subtitle_paths': [],
                    'thumbnail_paths': [],
                    'file_size': 0,
                    'temp_files': []
                }
            else:
                # Download video
                download_result = await self._download_video(url, config)

            # Update result with download info
            result.video_path = download_result.get('video_path')
            result.audio_path = download_result.get('audio_path')
            result.subtitle_paths = download_result.get('subtitle_paths', [])
            result.thumbnail_paths = download_result.get('thumbnail_paths', [])
            result.file_size = download_result.get('file_size', 0)
            result.temp_files = download_result.get('temp_files', [])

            # Extract transcript if requested and not already provided
            if config.extract_transcript and not (provided_file and self._is_transcript_file(provided_file)):
                transcript_result = await self._extract_transcript_with_fallback(
                    url, config, result.audio_path
                )
                if transcript_result:
                    result.transcript_path = transcript_result.get('transcript_path')
                    result.transcript_text = transcript_result.get('transcript_text')
                    result.transcript_language = transcript_result.get('transcript_language')
                    if transcript_result.get('transcript_path'):
                        result.temp_files.append(transcript_result['transcript_path'])
                else:
                    # If transcript extraction failed and transcript_only mode is enabled, fail the operation
                    if config.transcript_only:
                        error_msg = "Transcript extraction failed in transcript-only mode"
                        logger.error(error_msg, url=url)
                        return YouTubeDownloadResult(
                            video_path=None,
                            audio_path=None,
                            subtitle_paths=[],
                            thumbnail_paths=[],
                            metadata=metadata,
                            processing_time=time.time() - start_time,
                            file_size=0,
                            temp_files=[],
                            success=False,
                            error_message=error_msg
                        )
                    else:
                        # In normal mode, transcript failure is not critical
                        logger.warning("Transcript extraction failed, continuing without transcript", url=url)
            elif provided_file and self._is_transcript_file(provided_file):
                # Use provided transcript file
                logger.info("Using provided transcript file", provided_file=str(provided_file))
                result.transcript_path = provided_file
                result.transcript_text = provided_file.read_text(encoding='utf-8')
                result.transcript_language = config.transcript_language or 'en'

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
    
    async def _extract_metadata_only(self, url: str, config: YouTubeConfig) -> YouTubeMetadata:
        """Extract metadata without downloading."""
        try:
            logger.debug("Extracting YouTube metadata", url=url)

            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                # Add user agent to avoid bot detection
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                # Add headers to appear more like a regular browser
                'http_headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                # Use cookies if available and file exists (config takes precedence over env)
                'cookiefile': self._get_valid_cookies_file(config),
                # Add retry options
                'retries': 3,
                'fragment_retries': 3,
                # Use IPv4 to avoid some blocking
                'force_ipv4': True,
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
                # Add user agent to avoid bot detection
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                # Add headers to appear more like a regular browser
                'http_headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                # Use cookies if available and file exists (config takes precedence over env)
                'cookiefile': self._get_valid_cookies_file(config),
                # Add retry options
                'retries': 3,
                'fragment_retries': 3,
                # Use IPv4 to avoid some blocking
                'force_ipv4': True,
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
                # Add user agent to avoid bot detection
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                # Add headers to appear more like a regular browser
                'http_headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                # Use cookies if available
                'cookiefile': os.getenv('YOUTUBE_COOKIES_FILE'),
                # Add retry options
                'retries': 3,
                'fragment_retries': 3,
                # Use IPv4 to avoid some blocking
                'force_ipv4': True,
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

    async def _extract_transcript_with_fallback(
        self,
        url: str,
        config: YouTubeConfig,
        audio_path: Optional[Path]
    ) -> Optional[Dict[str, Any]]:
        """Extract transcript using intelligent fallback strategy.

        Strategy:
        1. If no cookies available, use direct transcript API only (avoid yt-dlp)
        2. If cookies are available and audio transcription is preferred, try audio transcription first
        3. Fallback to direct transcript API if audio transcription fails

        Args:
            url: YouTube video URL
            config: YouTube configuration
            audio_path: Path to downloaded audio file (if available)

        Returns:
            Dictionary containing transcript data or None if all methods fail
        """
        # Determine if we have cookies available
        cookies_available = self._has_cookies_available(config)

        # If no cookies available, use direct transcript API only to avoid yt-dlp bot detection
        if not cookies_available:
            logger.info("No cookies available, using direct transcript API only to avoid bot detection")
            try:
                return await self._extract_transcript(url, config)
            except Exception as e:
                logger.error("Direct transcript extraction failed without cookies",
                           url=url, error=str(e))
                return None

        # If transcript_only is True, skip video download and go directly to transcript API
        if config.transcript_only:
            logger.info("Transcript-only mode enabled, using direct transcript API")
            try:
                return await self._extract_transcript(url, config)
            except Exception as e:
                logger.error("Direct transcript extraction failed in transcript-only mode",
                           url=url, error=str(e))
                return None

        # Strategy 1: Try audio transcription if preferred and cookies available
        if (config.prefer_audio_transcription and
            cookies_available and
            audio_path and
            audio_path.exists() and
            self.audio_processor):

            logger.info("Attempting audio transcription with downloaded file (cookies available)",
                       audio_path=str(audio_path))

            try:
                return await self._transcribe_audio_file(audio_path, config)
            except Exception as e:
                logger.warning("Audio transcription failed, falling back to direct transcript API",
                              error=str(e))

        # Strategy 2: Try direct transcript API
        logger.info("Using direct transcript API extraction")
        try:
            return await self._extract_transcript(url, config)
        except Exception as e:
            logger.warning("Direct transcript extraction failed, trying audio transcription fallback",
                          url=url, error=str(e))

        # Strategy 3: Fallback to audio transcription if we have an audio file and cookies
        if (cookies_available and
            audio_path and
            audio_path.exists() and
            self.audio_processor):

            logger.info("Attempting audio transcription as fallback method",
                       audio_path=str(audio_path))

            try:
                return await self._transcribe_audio_file(audio_path, config)
            except Exception as e:
                logger.error("Audio transcription fallback also failed",
                           audio_path=str(audio_path), error=str(e))

        logger.error("All transcript extraction methods failed")
        return None

    def _has_cookies_available(self, config: YouTubeConfig) -> bool:
        """Check if cookies are available for YouTube access.

        Args:
            config: YouTube configuration

        Returns:
            True if cookies are available, False otherwise
        """
        # Check config-provided cookies file first
        if config.cookies_file and Path(config.cookies_file).exists():
            logger.debug("Cookies available from config", path=config.cookies_file)
            return True

        # Check environment variable
        env_cookies = os.getenv('YOUTUBE_COOKIES_FILE')
        if env_cookies and Path(env_cookies).exists():
            logger.debug("Cookies available from environment", path=env_cookies)
            return True

        logger.debug("No cookies available")
        return False

    def _get_valid_cookies_file(self, config: YouTubeConfig) -> Optional[str]:
        """Get valid cookies file path if it exists.

        Args:
            config: YouTube configuration

        Returns:
            Valid cookies file path or None if no valid file found
        """
        # Check config-provided cookies file first
        if config.cookies_file and Path(config.cookies_file).exists():
            return config.cookies_file

        # Check environment variable
        env_cookies = os.getenv('YOUTUBE_COOKIES_FILE')
        if env_cookies and Path(env_cookies).exists():
            return env_cookies

        # Return None if no valid cookies file found
        return None

    async def _transcribe_audio_file(
        self,
        audio_path: Path,
        config: YouTubeConfig
    ) -> Dict[str, Any]:
        """Transcribe audio file using audio processor.

        Args:
            audio_path: Path to audio file
            config: YouTube configuration

        Returns:
            Dictionary containing transcript data

        Raises:
            ProcessingError: If transcription fails
        """
        if not self.audio_processor:
            raise ProcessingError("Audio processor not available")

        try:
            logger.debug("Starting audio transcription", audio_path=str(audio_path))

            # Process audio file
            audio_result = await self.audio_processor.process(audio_path)

            if not audio_result.success:
                raise ProcessingError(f"Audio transcription failed: {audio_result.error_message}")

            # Validate that we have actual transcript content
            if not audio_result.transcript or not audio_result.transcript.strip():
                raise ProcessingError("Audio transcription produced empty transcript")

            if not audio_result.segments or len(audio_result.segments) == 0:
                raise ProcessingError("Audio transcription produced no segments")

            # Create output directory for transcript
            output_dir = self.temp_dir / f"transcript_audio_{int(time.time())}"
            output_dir.mkdir(exist_ok=True)

            # Save transcript to file
            # Extract video ID from audio filename or use timestamp
            try:
                # Try to extract video ID from filename (e.g., "title-VIDEO_ID.mp3")
                filename = audio_path.stem
                if '-' in filename:
                    potential_id = filename.split('-')[-1]
                    if len(potential_id) == 11:  # YouTube video IDs are 11 characters
                        video_id = potential_id
                    else:
                        video_id = f"audio_{int(time.time())}"
                else:
                    video_id = f"audio_{int(time.time())}"
            except:
                video_id = f"audio_{int(time.time())}"

            transcript_filename = f"{video_id}_transcript_audio.{config.transcript_format}"
            transcript_path = output_dir / transcript_filename

            # Save based on format
            if config.transcript_format == "text":
                content = audio_result.transcript
            elif config.transcript_format == "srt":
                content = self._convert_audio_segments_to_srt(audio_result.segments)
            elif config.transcript_format == "vtt":
                content = self._convert_audio_segments_to_vtt(audio_result.segments)
            else:
                content = audio_result.transcript

            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Detect language from audio result metadata
            detected_language = getattr(audio_result, 'language', None) or 'unknown'

            logger.info("Successfully transcribed audio file",
                       audio_path=str(audio_path),
                       language=detected_language,
                       segments_count=len(audio_result.segments))

            # Format transcript with timestamps for markdown
            formatted_transcript = self._format_audio_segments_with_timestamps(audio_result.segments)

            return {
                'transcript_path': transcript_path,
                'transcript_text': formatted_transcript,
                'transcript_language': detected_language,
                'is_audio_transcription': True,
                'segments_count': len(audio_result.segments),
                'duration': audio_result.duration if hasattr(audio_result, 'duration') else 0.0,
                'method': 'audio_transcription'
            }

        except Exception as e:
            logger.error("Audio transcription failed", audio_path=str(audio_path), error=str(e))
            raise ProcessingError(f"Audio transcription failed: {str(e)}")

    def _convert_audio_segments_to_srt(self, segments) -> str:
        """Convert audio segments to SRT format."""
        srt_content = []

        for i, segment in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(segment.start)
            end_time = self._seconds_to_srt_time(segment.end)

            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(segment.text.strip())
            srt_content.append("")  # Empty line between segments

        return "\n".join(srt_content)

    def _convert_audio_segments_to_vtt(self, segments) -> str:
        """Convert audio segments to WebVTT format."""
        vtt_content = ["WEBVTT", ""]

        for segment in segments:
            start_time = self._seconds_to_vtt_time(segment.start)
            end_time = self._seconds_to_vtt_time(segment.end)

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(segment.text.strip())
            vtt_content.append("")  # Empty line between segments

        return "\n".join(vtt_content)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to WebVTT time format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    def _format_audio_segments_with_timestamps(self, segments) -> str:
        """Format audio segments with timestamps for markdown.

        Args:
            segments: List of audio segments with start, end, and text

        Returns:
            Formatted transcript text with timestamps
        """
        formatted_lines = []

        for segment in segments:
            # Format timestamp as [MM:SS]
            start_time = segment.start
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"

            # Clean up the text
            text = segment.text.strip()
            if text:
                formatted_lines.append(f"{timestamp} {text}")

        return '\n'.join(formatted_lines)

    def _format_transcript_with_timestamps(self, segments) -> str:
        """Format transcript segments with timestamps for markdown.

        Args:
            segments: List of transcript segments with start, duration, and text

        Returns:
            Formatted transcript text with timestamps
        """
        formatted_lines = []

        for segment in segments:
            # Format timestamp as [MM:SS]
            start_time = segment.start
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"

            # Clean up the text
            text = segment.text.strip()
            if text:
                formatted_lines.append(f"{timestamp} {text}")

        return '\n'.join(formatted_lines)

    async def _extract_transcript(self, url: str, config: YouTubeConfig) -> Dict[str, Any]:
        """Extract transcript using youtube-transcript-api.

        Args:
            url: YouTube video URL
            config: YouTube configuration

        Returns:
            Dictionary containing transcript data
        """
        try:
            logger.debug("Extracting YouTube transcript", url=url)

            # Extract transcript using the transcript service
            transcript = await self.transcript_service.extract_transcript(
                url,
                language=config.transcript_language
            )

            # Create output directory for transcript
            output_dir = self.temp_dir / f"transcript_{int(time.time())}"
            output_dir.mkdir(exist_ok=True)

            # Save transcript to file
            transcript_filename = f"{transcript.video_id}_transcript.{config.transcript_format}"
            transcript_path = output_dir / transcript_filename

            await self.transcript_service.save_transcript_to_file(
                transcript,
                transcript_path,
                format_type=config.transcript_format
            )

            logger.info("Successfully extracted transcript",
                       video_id=transcript.video_id,
                       language=transcript.language,
                       segments_count=len(transcript.segments),
                       is_auto_generated=transcript.is_auto_generated)

            # Format transcript with timestamps for markdown
            formatted_transcript = self._format_transcript_with_timestamps(transcript.segments)

            return {
                'transcript_path': transcript_path,
                'transcript_text': formatted_transcript,
                'transcript_language': transcript.language,
                'is_auto_generated': transcript.is_auto_generated,
                'segments_count': len(transcript.segments),
                'duration': transcript.duration,
                'method': 'direct_transcript_api'
            }

        except Exception as e:
            logger.error("Failed to extract transcript", url=url, error=str(e))
            raise ProcessingError(f"Transcript extraction failed: {str(e)}")

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

    async def _handle_provided_file(self, provided_file: Path, config: YouTubeConfig) -> Dict[str, Any]:
        """Handle a provided file (video or transcript) instead of downloading.

        Args:
            provided_file: Path to the provided file
            config: YouTube configuration

        Returns:
            Dictionary containing file paths and metadata
        """
        logger.info("Processing provided file", file_path=str(provided_file), file_size=provided_file.stat().st_size)

        if self._is_transcript_file(provided_file):
            # Provided file is a transcript (markdown)
            return {
                'video_path': None,
                'audio_path': None,
                'subtitle_paths': [],
                'thumbnail_paths': [],
                'file_size': provided_file.stat().st_size,
                'temp_files': []
            }
        else:
            # Provided file is a video/audio file
            file_extension = provided_file.suffix.lower()

            if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
                # Video file
                video_path = provided_file
                audio_path = None

                # Extract audio if requested
                if config.extract_audio and self.audio_processor:
                    try:
                        # Create audio output path
                        audio_output = provided_file.parent / f"{provided_file.stem}.mp3"

                        # Extract audio using audio processor
                        await self.audio_processor.extract_audio(provided_file, audio_output)
                        audio_path = audio_output

                        logger.info("Extracted audio from provided video", audio_path=str(audio_path))
                    except Exception as e:
                        logger.warning("Failed to extract audio from provided video", error=str(e))

                return {
                    'video_path': video_path,
                    'audio_path': audio_path,
                    'subtitle_paths': [],
                    'thumbnail_paths': [],
                    'file_size': provided_file.stat().st_size,
                    'temp_files': [audio_path] if audio_path else []
                }
            elif file_extension in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']:
                # Audio file
                return {
                    'video_path': None,
                    'audio_path': provided_file,
                    'subtitle_paths': [],
                    'thumbnail_paths': [],
                    'file_size': provided_file.stat().st_size,
                    'temp_files': []
                }
            else:
                # Unknown file type, treat as video
                logger.warning("Unknown file type, treating as video", file_extension=file_extension)
                return {
                    'video_path': provided_file,
                    'audio_path': None,
                    'subtitle_paths': [],
                    'thumbnail_paths': [],
                    'file_size': provided_file.stat().st_size,
                    'temp_files': []
                }

    def _is_transcript_file(self, file_path: Path) -> bool:
        """Check if the provided file is a transcript file (markdown).

        Args:
            file_path: Path to check

        Returns:
            True if file appears to be a transcript
        """
        if not file_path.exists():
            return False

        # Check file extension
        if file_path.suffix.lower() in ['.md', '.txt']:
            return True

        # For other extensions, check content
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')[:1000]  # Read first 1KB

            # Look for transcript-like patterns
            transcript_indicators = [
                '# Youtube Analysis:',
                '## Youtube Information',
                '## Transcript',
                'youtube.com',
                'youtu.be',
                'Video ID:',
                'Duration:',
                'Uploader:'
            ]

            content_lower = content.lower()
            return any(indicator.lower() in content_lower for indicator in transcript_indicators)

        except Exception:
            return False

    def _create_metadata_from_dict(self, metadata_dict: Dict[str, Any], url: str) -> YouTubeMetadata:
        """Create YouTubeMetadata from a dictionary of provided metadata.

        Args:
            metadata_dict: Dictionary containing metadata fields
            url: YouTube URL for fallback values

        Returns:
            YouTubeMetadata object
        """
        # Extract video ID from URL for fallback
        video_id = "unknown"
        try:
            if self.youtube_service:
                video_id = self.youtube_service.transcript_service.extract_video_id(url)
            else:
                # Fallback: extract video ID using regex
                import re
                patterns = [
                    r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
                    r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
                    r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
                ]
                for pattern in patterns:
                    match = re.search(pattern, url)
                    if match:
                        video_id = match.group(1)
                        break
        except Exception:
            pass

        return YouTubeMetadata(
            id=metadata_dict.get('id', video_id),
            title=metadata_dict.get('title', 'Unknown Title'),
            description=metadata_dict.get('description', ''),
            uploader=metadata_dict.get('uploader', 'Unknown'),
            upload_date=metadata_dict.get('upload_date', ''),
            duration=float(metadata_dict.get('duration', 0)),
            view_count=int(metadata_dict.get('view_count', 0)),
            like_count=metadata_dict.get('like_count'),
            comment_count=metadata_dict.get('comment_count'),
            tags=metadata_dict.get('tags', []),
            categories=metadata_dict.get('categories', []),
            thumbnail_url=metadata_dict.get('thumbnail_url', ''),
            webpage_url=metadata_dict.get('webpage_url', url),
            channel_id=metadata_dict.get('channel_id', ''),
            channel_url=metadata_dict.get('channel_url', ''),
            playlist_id=metadata_dict.get('playlist_id'),
            playlist_title=metadata_dict.get('playlist_title'),
            playlist_index=metadata_dict.get('playlist_index')
        )