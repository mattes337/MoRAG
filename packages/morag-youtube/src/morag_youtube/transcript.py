"""YouTube transcript extraction using youtube-transcript-api."""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import structlog

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a single transcript segment with timing information."""
    text: str
    start: float
    duration: float
    
    @property
    def end(self) -> float:
        """End time of the segment."""
        return self.start + self.duration


@dataclass
class YouTubeTranscript:
    """Complete transcript data for a YouTube video."""
    video_id: str
    language: str
    segments: List[TranscriptSegment]
    full_text: str
    is_auto_generated: bool
    
    @property
    def duration(self) -> float:
        """Total duration of the transcript."""
        if not self.segments:
            return 0.0
        return max(segment.end for segment in self.segments)


class YouTubeTranscriptService:
    """Service for extracting transcripts from YouTube videos using youtube-transcript-api."""
    
    def __init__(self):
        """Initialize the transcript service."""
        self.text_formatter = TextFormatter()

    def _get_transcript_with_headers(self, video_id: str, target_language: str) -> List[Dict[str, Any]]:
        """Get transcript using the correct API with enhanced headers.

        Args:
            video_id: YouTube video ID
            target_language: Target language code

        Returns:
            List of transcript segments
        """
        # Use the correct API with enhanced headers
        import requests

        # Create a session with realistic browser headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        })

        try:
            # Use the correct new API
            ytt_api = YouTubeTranscriptApi(http_client=session)

            # Fetch transcript using the new API
            fetched_transcript = ytt_api.fetch(video_id, languages=[target_language])

            # Convert to raw data format
            return fetched_transcript.to_raw_data()

        except (TranscriptsDisabled, VideoUnavailable) as e:
            raise ProcessingError(f"Transcript unavailable for video {video_id}: {str(e)}")
        except (NoTranscriptFound, CouldNotRetrieveTranscript) as e:
            raise ProcessingError(f"Could not retrieve transcript for video {video_id} in language {target_language}: {str(e)}")
        except Exception as e:
            # For other errors (like XML parsing), provide detailed info
            raise ProcessingError(f"Failed to extract transcript for video {video_id}: {str(e)}")
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video ID string
            
        Raises:
            ProcessingError: If video ID cannot be extracted
        """
        # Common YouTube URL patterns
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # If no pattern matches, assume the URL is just the video ID
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url
            
        raise ProcessingError(f"Could not extract video ID from URL: {url}")
    
    async def get_available_transcripts(self, video_id: str) -> Dict[str, Dict[str, Any]]:
        """Get list of available transcripts for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary mapping language codes to transcript metadata
        """
        try:
            # Use the correct API
            ytt_api = YouTubeTranscriptApi()
            transcript_list = await asyncio.to_thread(
                ytt_api.list, video_id
            )
            
            available = {}
            
            # Add manually created transcripts
            for transcript in transcript_list:
                available[transcript.language_code] = {
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                }
            
            logger.info("Found available transcripts", 
                       video_id=video_id, 
                       languages=list(available.keys()))
            
            return available
            
        except Exception as e:
            logger.warning("Failed to get transcript list", 
                          video_id=video_id, 
                          error=str(e))
            return {}
    
    async def extract_transcript(
        self, 
        url: str, 
        language: Optional[str] = None
    ) -> YouTubeTranscript:
        """Extract transcript from YouTube video.
        
        Args:
            url: YouTube video URL or video ID
            language: Preferred language code (if None, uses original language)
            
        Returns:
            YouTubeTranscript object with transcript data
            
        Raises:
            ProcessingError: If transcript extraction fails
        """
        video_id = self.extract_video_id(url)
        
        try:
            # Get available transcripts
            available_transcripts = await self.get_available_transcripts(video_id)
            
            if not available_transcripts:
                raise ProcessingError(f"No transcripts available for video {video_id}")
            
            # Determine which language to use
            target_language = await self._determine_target_language(
                available_transcripts, language
            )
            
            logger.info("Extracting transcript",
                       video_id=video_id,
                       target_language=target_language)

            logger.debug("Calling YouTubeTranscriptApi.get_transcript",
                        video_id=video_id,
                        target_language=target_language)

            # Extract the transcript with enhanced headers to bypass bot detection
            transcript_data = await asyncio.to_thread(
                self._get_transcript_with_headers,
                video_id,
                target_language
            )

            logger.debug("Raw transcript data received",
                        video_id=video_id,
                        data_type=type(transcript_data).__name__,
                        data_length=len(transcript_data) if transcript_data else 0,
                        first_few_items=transcript_data[:3] if transcript_data else None)
            
            # Convert to our format
            segments = [
                TranscriptSegment(
                    text=item['text'].strip(),
                    start=item['start'],
                    duration=item['duration']
                )
                for item in transcript_data
                if item['text'].strip()  # Skip empty segments
            ]
            
            # Generate full text
            full_text = ' '.join(segment.text for segment in segments)
            
            # Check if it's auto-generated
            is_auto_generated = available_transcripts.get(target_language, {}).get('is_generated', False)
            
            transcript = YouTubeTranscript(
                video_id=video_id,
                language=target_language,
                segments=segments,
                full_text=full_text,
                is_auto_generated=is_auto_generated
            )
            
            logger.info("Successfully extracted transcript",
                       video_id=video_id,
                       language=target_language,
                       segments_count=len(segments),
                       duration=transcript.duration,
                       is_auto_generated=is_auto_generated)
            
            return transcript
            
        except Exception as e:
            logger.error("Transcript extraction failed",
                        video_id=video_id,
                        error=str(e),
                        error_type=type(e).__name__)

            error_msg = f"Failed to extract transcript for video {video_id}: {str(e)}"
            raise ProcessingError(error_msg)
    
    async def _determine_target_language(
        self, 
        available_transcripts: Dict[str, Dict[str, Any]], 
        preferred_language: Optional[str]
    ) -> str:
        """Determine which language transcript to use.
        
        Args:
            available_transcripts: Available transcript metadata
            preferred_language: User-specified preferred language
            
        Returns:
            Language code to use for transcript extraction
        """
        if not available_transcripts:
            raise ProcessingError("No transcripts available")
        
        # If no language specified, find the original language
        if not preferred_language:
            # Look for manually created (non-generated) transcripts first
            manual_transcripts = {
                lang: info for lang, info in available_transcripts.items()
                if not info.get('is_generated', True)
            }
            
            if manual_transcripts:
                # Use the first manual transcript (likely original language)
                original_language = next(iter(manual_transcripts.keys()))
                logger.info("Using original language transcript", 
                           language=original_language)
                return original_language
            
            # If only auto-generated transcripts, use the first available
            original_language = next(iter(available_transcripts.keys()))
            logger.info("Using first available auto-generated transcript", 
                       language=original_language)
            return original_language
        
        # If language specified, check if it's available
        if preferred_language in available_transcripts:
            logger.info("Using requested language", language=preferred_language)
            return preferred_language
        
        # Try common language variations
        language_variations = [
            preferred_language.lower(),
            preferred_language.upper(),
            preferred_language[:2],  # Just language code without region
        ]
        
        for variation in language_variations:
            if variation in available_transcripts:
                logger.info("Using language variation", 
                           requested=preferred_language, 
                           found=variation)
                return variation
        
        # Fallback to original language
        original_language = next(iter(available_transcripts.keys()))
        logger.warning("Requested language not available, using original", 
                      requested=preferred_language, 
                      fallback=original_language)
        return original_language
    
    async def save_transcript_to_file(
        self, 
        transcript: YouTubeTranscript, 
        output_path: Path,
        format_type: str = "text"
    ) -> Path:
        """Save transcript to file.
        
        Args:
            transcript: YouTubeTranscript object
            output_path: Output file path
            format_type: Format type ("text", "srt", "vtt")
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "text":
            content = transcript.full_text
        elif format_type == "srt":
            content = self._format_as_srt(transcript.segments)
        elif format_type == "vtt":
            content = self._format_as_vtt(transcript.segments)
        else:
            raise ProcessingError(f"Unsupported format type: {format_type}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("Transcript saved to file", 
                   path=str(output_path), 
                   format=format_type)
        
        return output_path
    
    def _format_as_srt(self, segments: List[TranscriptSegment]) -> str:
        """Format transcript segments as SRT subtitle format."""
        srt_content = []
        
        for i, segment in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(segment.start)
            end_time = self._seconds_to_srt_time(segment.end)
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(segment.text)
            srt_content.append("")  # Empty line between segments
        
        return "\n".join(srt_content)
    
    def _format_as_vtt(self, segments: List[TranscriptSegment]) -> str:
        """Format transcript segments as WebVTT format."""
        vtt_content = ["WEBVTT", ""]
        
        for segment in segments:
            start_time = self._seconds_to_vtt_time(segment.start)
            end_time = self._seconds_to_vtt_time(segment.end)
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(segment.text)
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
