"""Audio document converter using markitdown framework."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import structlog
from morag_core.interfaces.converter import (
    ConversionError,
    ConversionOptions,
    ConversionResult,
    QualityScore,
    UnsupportedFormatError,
)
from morag_core.models.document import Document, DocumentType
from morag_document.services.markitdown_service import MarkitdownService

if TYPE_CHECKING:
    from morag_audio.processor import AudioProcessingResult, AudioSegment

logger = structlog.get_logger(__name__)


@dataclass
class AudioConversionOptions:
    """Options for audio conversion to markdown."""

    include_timestamps: bool = True
    include_speakers: bool = True
    include_topics: bool = True
    timestamp_format: str = (
        "[%H:%M:%S]"  # Format for timestamps (legacy, now using dynamic format)
    )
    group_by_speaker: bool = False  # Group consecutive segments by the same speaker (disabled for per-line timestamps)
    group_by_topic: bool = True  # Group segments by topic
    include_metadata: bool = True  # Include metadata section in markdown
    metadata_fields: List[str] = field(
        default_factory=lambda: [
            "duration",
            "num_speakers",
            "speakers",
            "num_topics",
            "word_count",
        ]
    )


@dataclass
class AudioConversionResult:
    """Result of audio conversion to markdown."""

    content: str
    metadata: Dict[str, Any]
    processing_time: float
    success: bool = True
    error_message: Optional[str] = None


class AudioConverter:
    """Audio document converter using markitdown framework."""

    def __init__(self):
        """Initialize Audio converter."""
        self.name = "MoRAG Audio Converter"
        self.supported_formats: Set[str] = {
            "audio",
            "mp3",
            "wav",
            "m4a",
            "flac",
            "aac",
            "ogg",
            "wma",
            "mp4",
            "avi",
            "mov",
            "mkv",
        }
        self.markitdown_service = MarkitdownService()

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() in {
            "audio",
            "mp3",
            "wav",
            "m4a",
            "flac",
            "aac",
            "ogg",
            "wma",
            "mp4",
            "avi",
            "mov",
            "mkv",
        }

    async def convert(
        self, file_path: Union[str, Path], options: Optional[ConversionOptions] = None
    ) -> ConversionResult:
        """Convert audio file to text using markitdown.

        Args:
            file_path: Path to audio file
            options: Conversion options

        Returns:
            Conversion result with document

        Raises:
            ConversionError: If conversion fails
            UnsupportedFormatError: If audio format is not supported
        """
        file_path = Path(file_path)
        options = options or ConversionOptions()

        # Validate input
        if not file_path.exists():
            raise ConversionError(f"Audio file not found: {file_path}")

        # Detect format if not specified
        format_type = options.format_type or file_path.suffix.lower().lstrip(".")

        # Check if format is supported
        if not await self.supports_format(format_type):
            raise UnsupportedFormatError(
                f"Format '{format_type}' is not supported by audio converter"
            )

        try:
            # Use markitdown for audio transcription
            logger.info(
                "Converting audio file with markitdown", file_path=str(file_path)
            )

            result = await self.markitdown_service.convert_file(file_path)

            # Create document
            document = Document(
                id=options.document_id,
                title=options.title or file_path.stem,
                raw_text=result.text_content,
                document_type=DocumentType.AUDIO,
                file_path=str(file_path),
                metadata={
                    "file_size": file_path.stat().st_size,
                    "format": format_type,
                    "conversion_method": "markitdown",
                    **result.metadata,
                },
            )

            # Calculate quality score
            quality_score = QualityScore(
                overall_score=0.9,  # High score for markitdown transcription
                text_extraction_score=0.9,
                structure_preservation_score=0.8,
                metadata_extraction_score=0.9,
                issues_detected=[],
            )

            return ConversionResult(
                document=document, quality_score=quality_score, warnings=[]
            )

        except Exception as e:
            logger.error(
                "Audio conversion failed", error=str(e), file_path=str(file_path)
            )
            raise ConversionError(f"Failed to convert audio file: {e}") from e

    async def convert_to_json(
        self,
        result: "AudioProcessingResult",
        options: Optional[AudioConversionOptions] = None,
    ) -> Dict[str, Any]:
        """Convert audio processing result to structured JSON.

        Args:
            result: Audio processing result
            options: Conversion options

        Returns:
            Dictionary with structured JSON data
        """
        options = options or AudioConversionOptions()

        try:
            if not result.success:
                return {
                    "title": "",
                    "filename": result.file_path,
                    "metadata": result.metadata,
                    "topics": [],
                    "error": result.error_message,
                }

            # Group segments by topic if topic segmentation is enabled
            topics = []
            if options.include_topics and any(
                hasattr(segment, "topic") and segment.topic is not None
                for segment in result.segments
            ):
                # Group by topic
                topic_groups = {}
                for segment in result.segments:
                    topic_id = (
                        getattr(segment, "topic", 0) if hasattr(segment, "topic") else 0
                    )
                    if topic_id not in topic_groups:
                        topic_groups[topic_id] = []
                    topic_groups[topic_id].append(segment)

                # Create topic entries
                for topic_id, segments in sorted(topic_groups.items()):
                    topic_data = {
                        "timestamp": int(segments[0].start) if segments else 0,
                        "sentences": [],
                    }

                    for segment in segments:
                        sentence = {
                            "timestamp": int(segment.start),
                            "speaker": getattr(segment, "speaker", 1)
                            if hasattr(segment, "speaker")
                            else 1,
                            "text": segment.text,
                        }
                        topic_data["sentences"].append(sentence)

                    topics.append(topic_data)
            else:
                # Single topic with all segments
                topic_data = {
                    "timestamp": int(result.segments[0].start)
                    if result.segments
                    else 0,
                    "sentences": [],
                }

                for segment in result.segments:
                    sentence = {
                        "timestamp": int(segment.start),
                        "speaker": getattr(segment, "speaker", 1)
                        if hasattr(segment, "speaker")
                        else 1,
                        "text": segment.text,
                    }
                    topic_data["sentences"].append(sentence)

                topics.append(topic_data)

            # Extract title from filename
            filename = Path(result.file_path).name
            title = Path(result.file_path).stem

            return {
                "title": title,
                "filename": filename,
                "metadata": result.metadata,
                "topics": topics,
            }

        except Exception as e:
            logger.error("Failed to convert to JSON", error=str(e))
            return {
                "title": "",
                "filename": result.file_path,
                "metadata": result.metadata,
                "topics": [],
                "error": str(e),
            }

    async def convert_to_markdown(
        self,
        result: "AudioProcessingResult",
        options: Optional[AudioConversionOptions] = None,
    ) -> AudioConversionResult:
        """Convert audio processing result to structured markdown.

        Args:
            result: Audio processing result
            options: Conversion options

        Returns:
            AudioConversionResult with markdown content
        """
        start_time = time.time()
        options = options or AudioConversionOptions()

        try:
            if not result.success:
                raise ConversionError(
                    f"Cannot convert unsuccessful processing result: {result.error_message}"
                )

            # Check if we have either segments or transcript
            if not result.segments and not result.transcript:
                raise ConversionError(
                    "Cannot convert result: no segments or transcript available"
                )

            markdown_content = []

            # Add title
            file_name = Path(result.file_path).name
            markdown_content.append(f"# Audio Transcription: {file_name}\n")

            # Add metadata section if requested
            if options.include_metadata:
                markdown_content.append("## Metadata\n")
                for field in options.metadata_fields:
                    if field in result.metadata:
                        value = result.metadata[field]
                        if field == "duration" and isinstance(value, (int, float)):
                            # Format duration as HH:MM:SS
                            hours, remainder = divmod(int(value), 3600)
                            minutes, seconds = divmod(remainder, 60)
                            value = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        markdown_content.append(
                            f"- **{field.replace('_', ' ').title()}**: {value}"
                        )
                markdown_content.append("\n")

            # Choose between Full Transcript and Detailed Transcript (mutually exclusive)
            # Prefer detailed transcript when segments are available, otherwise use full transcript
            if result.segments and len(result.segments) > 0:
                # Add detailed transcript section with speakers and timestamps
                markdown_content.append("## Detailed Transcript")
                markdown_content.append("")

                # Group segments by topic if both topic grouping and topic inclusion are enabled, and topic information is available
                if (
                    options.group_by_topic
                    and options.include_topics
                    and any(
                        hasattr(segment, "topic_id") and segment.topic_id is not None
                        for segment in result.segments
                    )
                ):
                    topic_groups = {}
                    for segment in result.segments:
                        topic_id = getattr(segment, "topic_id", None)
                        if topic_id is not None:
                            if topic_id not in topic_groups:
                                topic_groups[topic_id] = []
                            topic_groups[topic_id].append(segment)
                        else:
                            # Handle segments without topic_id
                            if -1 not in topic_groups:
                                topic_groups[-1] = []
                            topic_groups[-1].append(segment)

                    # Get topic information from result metadata if available
                    topic_info_map = {}
                    if (
                        hasattr(result, "topic_segmentation")
                        and result.topic_segmentation
                    ):
                        for topic_segment in result.topic_segmentation.segments:
                            # Extract topic ID from topic_segment.topic_id (e.g., "TOPIC_00" -> 0)
                            if (
                                hasattr(topic_segment, "topic_id")
                                and topic_segment.topic_id
                            ):
                                topic_id_str = topic_segment.topic_id
                                if topic_id_str.startswith("TOPIC_"):
                                    try:
                                        topic_id = int(topic_id_str.split("_")[1])
                                        topic_info_map[topic_id] = topic_segment
                                    except (IndexError, ValueError):
                                        pass

                    # Process each topic group
                    for topic_id, segments in sorted(topic_groups.items()):
                        if topic_id != -1:
                            # Get topic information
                            topic_segment = topic_info_map.get(topic_id)
                            topic_title = None
                            topic_start_seconds = None

                            if topic_segment:
                                topic_title = getattr(topic_segment, "title", None)
                                topic_start_seconds = int(
                                    getattr(topic_segment, "start_time", 0)
                                )

                            # Fallback to calculating from segments if no topic info
                            if not topic_title or topic_start_seconds is None:
                                if segments:
                                    start_time = min(seg.start for seg in segments)
                                    topic_start_seconds = int(start_time)
                                    topic_title = topic_title or f"Topic {topic_id + 1}"
                                else:
                                    topic_start_seconds = 0
                                    topic_title = topic_title or f"Topic {topic_id + 1}"

                            # Format topic header: # Topic Name [timestamp_in_seconds]
                            markdown_content.append(
                                f"# {topic_title} [{topic_start_seconds}]"
                            )
                        else:
                            markdown_content.append("# Ungrouped Content [0]")

                        markdown_content.append("")

                        # Process segments within this topic
                        formatted_segments = self._format_segments(segments, options)
                        markdown_content.extend(formatted_segments)
                        markdown_content.append("")
                else:
                    # Process all segments without topic grouping
                    formatted_segments = self._format_segments(result.segments, options)
                    markdown_content.extend(formatted_segments)
            else:
                # Fallback to full transcript when no segments are available
                markdown_content.append("## Full Transcript")
                markdown_content.append("")
                if result.transcript and result.transcript.strip():
                    markdown_content.append(result.transcript.strip())
                markdown_content.append("")

            # Join all content and remove excessive empty lines
            content = "\n".join(markdown_content)
            # Clean up multiple consecutive empty lines
            import re

            content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

            processing_time = time.time() - start_time

            return AudioConversionResult(
                content=content,
                metadata=result.metadata,
                processing_time=processing_time,
                success=True,
            )

        except Exception as e:
            logger.error("Audio conversion to markdown failed", error=str(e))

            processing_time = time.time() - start_time
            return AudioConversionResult(
                content="",
                metadata={"error": str(e)},
                processing_time=processing_time,
                success=False,
                error_message=str(e),
            )

    def _format_segments(
        self, segments: List["AudioSegment"], options: AudioConversionOptions
    ) -> List[str]:
        """Format segments into markdown lines."""
        markdown_lines = []

        if options.group_by_speaker and any(segment.speaker for segment in segments):
            # Group consecutive segments by speaker, but still use per-line timestamps
            current_speaker = None
            current_segments = []

            for segment in segments:
                speaker = segment.speaker or "Unknown"

                if speaker != current_speaker:
                    # Output the previous speaker's segments
                    if current_speaker and current_segments:
                        for seg in current_segments:
                            # Skip empty segments
                            if not seg.text or not seg.text.strip():
                                continue

                            # Build line in [timecode][speaker] format
                            line_content = ""

                            # Add timestamp if requested
                            if options.include_timestamps:
                                hours, remainder = divmod(int(seg.start), 3600)
                                minutes, seconds = divmod(remainder, 60)

                                # Use MM:SS format for content under 1 hour, HH:MM:SS for longer content
                                if hours > 0:
                                    timestamp = (
                                        f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                                    )
                                else:
                                    timestamp = f"[{minutes:02d}:{seconds:02d}]"
                                line_content += timestamp

                            # Add speaker if available and requested
                            if options.include_speakers:
                                line_content += f"[{current_speaker}]"

                            # Add the text
                            text_content = seg.text.strip()
                            if text_content:
                                line_content += f" {text_content}"

                            # Add to markdown lines if not empty
                            if line_content.strip():
                                markdown_lines.append(line_content)

                    # Start new speaker
                    current_speaker = speaker
                    current_segments = [segment] if segment.text.strip() else []
                else:
                    # Continue with the same speaker
                    if segment.text.strip():
                        current_segments.append(segment)

            # Output the last speaker's segments
            if current_speaker and current_segments:
                for seg in current_segments:
                    # Skip empty segments
                    if not seg.text or not seg.text.strip():
                        continue

                    # Build line in [timecode][speaker] format
                    line_content = ""

                    # Add timestamp if requested
                    if options.include_timestamps:
                        hours, remainder = divmod(int(seg.start), 3600)
                        minutes, seconds = divmod(remainder, 60)

                        # Use MM:SS format for content under 1 hour, HH:MM:SS for longer content
                        if hours > 0:
                            timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                        else:
                            timestamp = f"[{minutes:02d}:{seconds:02d}]"
                        line_content += timestamp

                    # Add speaker if available and requested
                    if options.include_speakers:
                        line_content += f"[{current_speaker}]"

                    # Add the text
                    text_content = seg.text.strip()
                    if text_content:
                        line_content += f" {text_content}"

                    # Add to markdown lines if not empty
                    if line_content.strip():
                        markdown_lines.append(line_content)
        else:
            # Process each segment individually
            for segment in segments:
                # Skip empty segments
                if not segment.text or not segment.text.strip():
                    continue

                line_parts = []

                # Build the line in the new [timecode][speaker] format
                line_content = ""

                # Add timestamp if requested
                if options.include_timestamps:
                    hours, remainder = divmod(int(segment.start), 3600)
                    minutes, seconds = divmod(remainder, 60)

                    # Use MM:SS format for content under 1 hour, HH:MM:SS for longer content
                    if hours > 0:
                        timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                    else:
                        timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    line_content += timestamp

                # Add speaker if available and requested (directly concatenated, no space)
                if options.include_speakers and segment.speaker:
                    line_content += f"[{segment.speaker}]"

                # Add the text (with a space before text content)
                text_content = segment.text.strip()
                if text_content:
                    line_content += f" {text_content}"

                # Add to markdown lines if not empty
                if line_content.strip():
                    markdown_lines.append(line_content)

        # Filter out any empty lines that might have been added
        return [line for line in markdown_lines if line.strip()]
