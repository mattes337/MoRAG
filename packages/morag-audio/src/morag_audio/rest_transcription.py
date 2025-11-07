"""REST-based transcription service using OpenAI Whisper API."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiohttp
import aiofiles

from .models import AudioSegment, AudioConfig

logger = logging.getLogger(__name__)


class RestTranscriptionError(Exception):
    """Exception raised for REST transcription errors."""
    pass


class RestTranscriptionService:
    """Service for transcribing audio using OpenAI Whisper REST API."""

    def __init__(self, config: AudioConfig):
        """Initialize the REST transcription service.

        Args:
            config: Audio configuration containing API settings
        """
        self.config = config
        if not config.openai_api_key:
            raise RestTranscriptionError("OpenAI API key is required for REST transcription")

    async def transcribe_audio(self, audio_path: Path) -> tuple[str, List[AudioSegment], str]:
        """Transcribe audio file using OpenAI Whisper API.

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of (transcript_text, segments, detected_language)

        Raises:
            RestTranscriptionError: If transcription fails
        """

        try:
            # Prepare the API request
            url = f"{self.config.api_base_url}/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {self.config.openai_api_key}"
            }

            # Prepare form data
            data = aiohttp.FormData()

            # Read and add the audio file
            async with aiofiles.open(audio_path, 'rb') as audio_file:
                audio_content = await audio_file.read()
                data.add_field('file', audio_content,
                             filename=audio_path.name,
                             content_type='audio/mpeg')

            # Add other parameters
            data.add_field('model', 'whisper-1')
            data.add_field('response_format', 'verbose_json')
            data.add_field('timestamp_granularities[]', 'segment')

            if self.config.language:
                data.add_field('language', self.config.language)

            # Make the API request
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, data=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RestTranscriptionError(
                            f"OpenAI API request failed with status {response.status}: {error_text}"
                        )

                    result = await response.json()

            # Parse the response
            transcript_text = result.get('text', '')
            detected_language = result.get('language', 'unknown')
            segments = self._parse_segments(result.get('segments', []))

            logger.info(f"Successfully transcribed audio with {len(segments)} segments")
            return transcript_text, segments, detected_language

        except aiohttp.ClientError as e:
            raise RestTranscriptionError(f"HTTP client error: {str(e)}")
        except Exception as e:
            raise RestTranscriptionError(f"Transcription failed: {str(e)}")

    def _parse_segments(self, api_segments: List[Dict[str, Any]]) -> List[AudioSegment]:
        """Parse OpenAI API segments into AudioSegment objects.

        Args:
            api_segments: List of segment dictionaries from OpenAI API

        Returns:
            List of AudioSegment objects
        """

        segments = []

        for segment in api_segments:
            audio_segment = AudioSegment(
                start=float(segment.get('start', 0.0)),
                end=float(segment.get('end', 0.0)),
                text=segment.get('text', '').strip(),
                speaker=None,  # OpenAI API doesn't provide speaker info
                confidence=None,  # OpenAI API doesn't provide confidence scores
                topic_id=None,
                topic_label=None
            )
            segments.append(audio_segment)

        return segments

    def convert_to_markdown(self, transcript: str, segments: List[AudioSegment]) -> str:
        """Convert transcript and segments to markdown format.

        Args:
            transcript: Full transcript text
            segments: List of audio segments with timecodes

        Returns:
            Markdown formatted transcript
        """
        markdown_lines = []
        markdown_lines.append("# Audio Transcript\n")

        # Add full transcript
        markdown_lines.append("## Full Transcript")
        markdown_lines.append(transcript)
        markdown_lines.append("")

        # Add timestamped segments
        markdown_lines.append("## Timestamped Segments\n")

        for i, segment in enumerate(segments, 1):
            start_time = self._format_timestamp(segment.start)
            end_time = self._format_timestamp(segment.end)

            markdown_lines.append(f"### Segment {i}")
            markdown_lines.append(f"**Time:** {start_time} - {end_time}")
            markdown_lines.append(f"**Text:** {segment.text}")
            markdown_lines.append("")

        return "\n".join(markdown_lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def _logprob_to_confidence(self, logprob: float) -> float:
        """Convert log probability to confidence score.

        Args:
            logprob: Log probability value

        Returns:
            Confidence score between 0 and 1
        """
        import math
        if logprob == 0.0:
            return 1.0
        # Convert log probability to confidence using exponential
        return max(0.0, min(1.0, math.exp(logprob)))
