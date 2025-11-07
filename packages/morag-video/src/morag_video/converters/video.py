"""Video converter module for MoRAG using markitdown framework."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import structlog
from morag_core.converters import ConversionQualityValidator
from morag_core.interfaces.converter import (
    ConversionError,
    ConversionOptions,
    ConversionResult,
    QualityScore,
    UnsupportedFormatError,
)
from morag_core.models.document import Document, DocumentType
from morag_document.services.markitdown_service import MarkitdownService
from morag_video.processor import VideoProcessingResult

logger = structlog.get_logger(__name__)


@dataclass
class VideoConversionOptions:
    """Options for video conversion."""

    include_metadata: bool = True
    include_thumbnails: bool = True
    include_keyframes: bool = True
    include_transcript: bool = True
    include_timestamps: bool = True
    include_speakers: bool = True
    include_topics: bool = True
    group_by_speaker: bool = False
    group_by_topic: bool = False
    max_thumbnails: int = 5
    max_keyframes: int = 10
    include_ocr: bool = True


class VideoConverter:
    """Video document converter using markitdown framework."""

    def __init__(self):
        """Initialize Video converter."""
        self.supported_formats: Set[str] = {
            "video",
            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm",
            "flv",
            "wmv",
        }
        self.markitdown_service = MarkitdownService()
        self.quality_validator = ConversionQualityValidator()
        logger.info("Video converter initialized")

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() in self.supported_formats

    async def convert(
        self, file_path: Union[str, Path], options: Optional[ConversionOptions] = None
    ) -> ConversionResult:
        """Convert video file to text using markitdown.

        Args:
            file_path: Path to video file
            options: Conversion options

        Returns:
            Conversion result with document

        Raises:
            ConversionError: If conversion fails
            UnsupportedFormatError: If video format is not supported
        """
        file_path = Path(file_path)
        options = options or ConversionOptions()

        # Validate input
        if not file_path.exists():
            raise ConversionError(f"Video file not found: {file_path}")

        # Detect format if not specified
        format_type = options.format_type or file_path.suffix.lower().lstrip(".")

        # Check if format is supported
        if not await self.supports_format(format_type):
            raise UnsupportedFormatError(
                f"Format '{format_type}' is not supported by video converter"
            )

        try:
            # Use markitdown for video transcription
            logger.info(
                "Converting video file with markitdown", file_path=str(file_path)
            )

            result = await self.markitdown_service.convert_file(file_path)

            # Create document
            document = Document(
                id=options.document_id,
                title=options.title or file_path.stem,
                raw_text=result.text_content,
                document_type=DocumentType.VIDEO,
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
                "Video conversion failed", error=str(e), file_path=str(file_path)
            )
            raise ConversionError(f"Failed to convert video file: {e}") from e

        except Exception as e:
            logger.error(
                "Video conversion failed", file_path=str(file_path), error=str(e)
            )
            raise

    async def convert_to_markdown(
        self,
        result: VideoProcessingResult,
        options: Optional[VideoConversionOptions] = None,
    ) -> str:
        """Convert video processing result to markdown.

        Args:
            result: Video processing result
            options: Conversion options

        Returns:
            Markdown string
        """
        options = options or VideoConversionOptions()

        # Create markdown content
        markdown_parts = []

        # Add title in format: Video Analysis: filename.ext
        if hasattr(result, "file_path") and result.file_path:
            filename = Path(result.file_path).name
            markdown_parts.append(f"# Video Analysis: {filename}\n")
        else:
            markdown_parts.append("# Video Analysis\n")

        # Add metadata section (standardized bullet point format)
        if options.include_metadata:
            markdown_parts.append("## Video Information\n")
            markdown_parts.append(
                f"- **Duration**: {self._format_duration(result.metadata.duration)}"
            )
            markdown_parts.append(
                f"- **Resolution**: {result.metadata.width}x{result.metadata.height}"
            )
            markdown_parts.append(f"- **FPS**: {result.metadata.fps}")
            markdown_parts.append(f"- **Format**: {result.metadata.format}")
            markdown_parts.append(f"- **Video Codec**: {result.metadata.codec}")

            if result.metadata.has_audio:
                markdown_parts.append(
                    f"- **Audio**: Yes ({result.metadata.audio_codec})"
                )
            else:
                markdown_parts.append("- **Audio**: No")

            if result.metadata.creation_time:
                markdown_parts.append(
                    f"- **Creation Time**: {result.metadata.creation_time}"
                )

            markdown_parts.append("\n")

        # Add OCR results if available
        if options.include_ocr and result.ocr_results:
            markdown_parts.append("## Text Detected in Video\n")

            for image_path, text in result.ocr_results.items():
                if text.strip():
                    image_name = Path(image_path).name
                    markdown_parts.append(f"### Text from {image_name}\n")
                    markdown_parts.append(f"{text.strip()}\n\n")

        # Add transcript section if available
        if options.include_transcript and result.audio_processing_result:
            markdown_parts.append("## Transcript\n")

            # Check if we have segments with proper attributes
            has_segments = (
                hasattr(result.audio_processing_result, "segments")
                and result.audio_processing_result.segments
                and len(result.audio_processing_result.segments) > 0
            )

            if has_segments:
                segments = result.audio_processing_result.segments

                # Check if segments have speaker information
                has_speakers = any(
                    hasattr(seg, "speaker") and seg.speaker and seg.speaker.strip()
                    for seg in segments
                )

                # Check if segments have topic information
                has_topics = any(
                    hasattr(seg, "topic_id") and seg.topic_id is not None
                    for seg in segments
                )

                logger.info(
                    "Processing segments for markdown",
                    segments_count=len(segments),
                    has_speakers=has_speakers,
                    has_topics=has_topics,
                    group_by_speaker=options.group_by_speaker,
                    group_by_topic=options.group_by_topic,
                )

                # Handle different transcript formats based on options and available data
                if options.group_by_topic and options.include_topics and has_topics:
                    # Group by topic using segments
                    markdown_parts.append(
                        self._format_segments_by_topic(segments, options)
                    )
                elif (
                    options.group_by_speaker
                    and options.include_speakers
                    and has_speakers
                ):
                    # Group by speaker using segments
                    markdown_parts.append(
                        self._format_segments_by_speaker(segments, options)
                    )
                else:
                    # Use regular segments with timestamps
                    logger.info("Using timestamped segments format")
                    markdown_parts.append(self._format_segments(segments, options))
            elif result.audio_processing_result.transcript:
                # Fallback to full transcript
                logger.warning("No segments available, using full transcript")
                markdown_parts.append(result.audio_processing_result.transcript)
            else:
                logger.warning("No transcript or segments available")
                markdown_parts.append("*No transcript available*")

            markdown_parts.append("\n")

        # Add thumbnails section if available
        if options.include_thumbnails and result.thumbnails:
            markdown_parts.append("## Thumbnails\n")

            # Limit the number of thumbnails based on options
            thumbnails_to_show = result.thumbnails[: options.max_thumbnails]

            for i, thumbnail_path in enumerate(thumbnails_to_show):
                timestamp = self._get_timestamp_from_filename(thumbnail_path)
                markdown_parts.append(f"### Thumbnail {i+1} {timestamp}\n")
                markdown_parts.append(f"![Thumbnail {i+1}]({thumbnail_path})\n\n")

            if len(result.thumbnails) > options.max_thumbnails:
                markdown_parts.append(
                    f"*{len(result.thumbnails) - options.max_thumbnails} more thumbnails not shown*\n\n"
                )

        # Add keyframes section if available
        if options.include_keyframes and result.keyframes:
            markdown_parts.append("## Key Frames\n")

            # Limit the number of keyframes based on options
            keyframes_to_show = result.keyframes[: options.max_keyframes]

            for i, keyframe_path in enumerate(keyframes_to_show):
                timestamp = self._get_timestamp_from_filename(keyframe_path)
                markdown_parts.append(f"### Key Frame {i+1} {timestamp}\n")
                markdown_parts.append(f"![Key Frame {i+1}]({keyframe_path})\n\n")

                # Add OCR text for this keyframe if available
                if (
                    options.include_ocr
                    and result.ocr_results
                    and str(keyframe_path) in result.ocr_results
                ):
                    ocr_text = result.ocr_results[str(keyframe_path)]
                    if ocr_text.strip():
                        markdown_parts.append(
                            f"**Text detected:**\n\n{ocr_text.strip()}\n\n"
                        )

            if len(result.keyframes) > options.max_keyframes:
                markdown_parts.append(
                    f"*{len(result.keyframes) - options.max_keyframes} more key frames not shown*\n\n"
                )

        return "\n".join(markdown_parts)

    def _calculate_quality_score(
        self, result: VideoProcessingResult, options: VideoConversionOptions
    ) -> float:
        """Calculate a quality score for the conversion result."""
        score = 0.0
        total_weight = 0.0

        # Base score for having metadata
        if result.metadata:
            score += 0.2
            total_weight += 0.2

        # Score for thumbnails
        if options.include_thumbnails and result.thumbnails:
            thumbnail_weight = 0.15
            thumbnail_score = min(
                1.0, len(result.thumbnails) / 5
            )  # Max score at 5+ thumbnails
            score += thumbnail_weight * thumbnail_score
            total_weight += thumbnail_weight

        # Score for keyframes
        if options.include_keyframes and result.keyframes:
            keyframe_weight = 0.15
            keyframe_score = min(
                1.0, len(result.keyframes) / 10
            )  # Max score at 10+ keyframes
            score += keyframe_weight * keyframe_score
            total_weight += keyframe_weight

        # Score for transcript
        if (
            options.include_transcript
            and result.audio_processing_result
            and result.audio_processing_result.transcript
        ):
            transcript_weight = 0.3

            # Base transcript score
            transcript_score = 0.5

            # Bonus for segments
            if result.audio_processing_result.segments:
                transcript_score += 0.2

            # Bonus for speaker diarization
            if (
                options.group_by_speaker
                and result.audio_processing_result.speaker_segments
            ):
                transcript_score += 0.15

            # Bonus for topic segmentation
            if options.group_by_topic and result.audio_processing_result.topic_segments:
                transcript_score += 0.15

            score += transcript_weight * min(1.0, transcript_score)
            total_weight += transcript_weight

        # Score for OCR
        if options.include_ocr and result.ocr_results:
            ocr_weight = 0.2

            # Calculate how many keyframes have OCR text
            frames_with_text = sum(
                1 for text in result.ocr_results.values() if text.strip()
            )
            total_frames = len(result.ocr_results)

            if total_frames > 0:
                ocr_score = frames_with_text / total_frames
                score += ocr_weight * ocr_score
                total_weight += ocr_weight

        # Normalize score
        if total_weight > 0:
            return score / total_weight
        return 0.0

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to HH:MM:SS format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _get_timestamp_from_filename(self, file_path: Union[str, Path]) -> str:
        """Extract timestamp from filename if available."""
        filename = Path(file_path).stem

        # Try to find timestamp in format like "keyframe_00h05m30s" or "thumbnail_330s"
        import re

        # Check for hours, minutes, seconds format
        match = re.search(r"(\d+)h(\d+)m(\d+)s", filename)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            if hours > 0:
                return f"({hours:02d}:{minutes:02d}:{seconds:02d})"
            return f"({minutes:02d}:{seconds:02d})"

        # Check for seconds format
        match = re.search(r"_(\d+)s", filename)
        if match:
            total_seconds = int(match.group(1))
            minutes, seconds = divmod(total_seconds, 60)
            hours, minutes = divmod(minutes, 60)

            if hours > 0:
                return f"({hours:02d}:{minutes:02d}:{seconds:02d})"
            return f"({minutes:02d}:{seconds:02d})"

        return ""

    def _format_segments(
        self, segments: List[Any], options: Optional[VideoConversionOptions] = None
    ) -> str:
        """Format transcript segments with new [timecode][speaker] format."""
        options = options or VideoConversionOptions()
        result = []

        for segment in segments:
            # Skip empty segments
            if (
                not hasattr(segment, "text")
                or not segment.text
                or not segment.text.strip()
            ):
                continue

            # Build line in [timecode][speaker] format
            line_content = ""

            # Add timestamp if requested (single start time, not range)
            if options.include_timestamps:
                start_time = getattr(segment, "start", 0)
                hours, remainder = divmod(int(start_time), 3600)
                minutes, seconds = divmod(remainder, 60)

                # Use MM:SS format for content under 1 hour, HH:MM:SS for longer content
                if hours > 0:
                    timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                else:
                    timestamp = f"[{minutes:02d}:{seconds:02d}]"
                line_content += timestamp

            # Add speaker if available and requested (directly concatenated, no space)
            if options.include_speakers:
                speaker = getattr(segment, "speaker", None)
                if speaker:
                    line_content += f"[{speaker}]"

            # Add the text (with a space before text content)
            text_content = segment.text.strip()
            if text_content:
                line_content += f" {text_content}"

            # Add to result if not empty
            if line_content.strip():
                result.append(line_content)

        return "\n".join(result)

    def _format_segments_by_speaker(
        self, segments: List[Any], options: Optional[VideoConversionOptions] = None
    ) -> str:
        """Format transcript segments grouped by speaker."""
        options = options or VideoConversionOptions()

        # Group segments by speaker
        speaker_groups = {}
        for segment in segments:
            speaker = getattr(segment, "speaker", "Unknown Speaker")
            if speaker not in speaker_groups:
                speaker_groups[speaker] = []
            speaker_groups[speaker].append(segment)

        result = []
        for speaker, speaker_segments in speaker_groups.items():
            # Calculate speaker start timestamp (single timestamp in seconds)
            if speaker_segments:
                start_time = min(getattr(seg, "start", 0) for seg in speaker_segments)
                speaker_start_seconds = int(start_time)
                result.append(f"# {speaker} [{speaker_start_seconds}]")
            else:
                result.append(f"# {speaker} [0]")

            result.append("")  # Empty line after header

            for segment in speaker_segments:
                # Skip empty segments
                if (
                    not hasattr(segment, "text")
                    or not segment.text
                    or not segment.text.strip()
                ):
                    continue

                # Build line in [timecode][speaker] format
                line_content = ""

                # Add timestamp if requested (single start time, not range)
                if options.include_timestamps:
                    start_time = getattr(segment, "start", 0)
                    hours, remainder = divmod(int(start_time), 3600)
                    minutes, seconds = divmod(remainder, 60)

                    # Use MM:SS format for content under 1 hour, HH:MM:SS for longer content
                    if hours > 0:
                        timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                    else:
                        timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    line_content += timestamp

                # Add speaker if requested (directly concatenated, no space)
                if options.include_speakers:
                    line_content += f"[{speaker}]"

                # Add the text (with a space before text content)
                text_content = segment.text.strip()
                if text_content:
                    line_content += f" {text_content}"

                # Add to result if not empty
                if line_content.strip():
                    result.append(line_content)

            result.append("")  # Single empty line between speakers

        # Filter out any empty lines and join
        filtered_result = [line for line in result if line is not None]
        return "\n".join(filtered_result)

    def _format_segments_by_topic(
        self, segments: List[Any], options: Optional[VideoConversionOptions] = None
    ) -> str:
        """Format transcript segments grouped by topic."""
        options = options or VideoConversionOptions()

        # Group segments by topic
        topic_groups = {}
        for segment in segments:
            topic_id = getattr(segment, "topic_id", None)
            topic_key = (
                f"Topic {topic_id + 1}" if topic_id is not None else "Ungrouped Content"
            )
            if topic_key not in topic_groups:
                topic_groups[topic_key] = []
            topic_groups[topic_key].append(segment)

        result = []
        for topic, topic_segments in topic_groups.items():
            # Calculate topic start timestamp (single timestamp in seconds)
            if topic_segments:
                start_time = min(getattr(seg, "start", 0) for seg in topic_segments)
                topic_start_seconds = int(start_time)
                result.append(f"# {topic} [{topic_start_seconds}]")
            else:
                result.append(f"# {topic} [0]")

            result.append("")  # Empty line after header

            for segment in topic_segments:
                # Skip empty segments
                if (
                    not hasattr(segment, "text")
                    or not segment.text
                    or not segment.text.strip()
                ):
                    continue

                # Build line in [timecode][speaker] format
                line_content = ""

                # Add timestamp if requested (single start time, not range)
                if options.include_timestamps:
                    start_time = getattr(segment, "start", 0)
                    hours, remainder = divmod(int(start_time), 3600)
                    minutes, seconds = divmod(remainder, 60)

                    # Use MM:SS format for content under 1 hour, HH:MM:SS for longer content
                    if hours > 0:
                        timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                    else:
                        timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    line_content += timestamp

                # Add speaker if available and requested (directly concatenated, no space)
                if options.include_speakers:
                    speaker = getattr(segment, "speaker", None)
                    if speaker:
                        line_content += f"[{speaker}]"

                # Add the text (with a space before text content)
                text_content = segment.text.strip()
                if text_content:
                    line_content += f" {text_content}"

                # Add to result if not empty
                if line_content.strip():
                    result.append(line_content)

            result.append("")  # Single empty line between topics

        # Filter out any empty lines and join
        filtered_result = [line for line in result if line is not None]
        return "\n".join(filtered_result)

    def _format_speaker_segments(self, speaker_segments: Dict[str, List[Any]]) -> str:
        """Format transcript segments grouped by speaker."""
        result = []

        for speaker, segments in speaker_segments.items():
            result.append(f"### Speaker {speaker}\n")

            for segment in segments:
                timestamp = f"[{self._format_duration(segment.start)} - {self._format_duration(segment.end)}]"
                result.append(f"{timestamp} {segment.text}\n")

            result.append("\n")

        return "\n".join(result)

    def _format_topic_segments(self, topic_segments: List[Any]) -> str:
        """Format transcript segments grouped by topic."""
        result = []

        for topic_segment in topic_segments:
            topic = topic_segment.topic if hasattr(topic_segment, "topic") else "Topic"
            result.append(f"### {topic}\n")

            if hasattr(topic_segment, "segments") and topic_segment.segments:
                for segment in topic_segment.segments:
                    timestamp = f"[{self._format_duration(segment.start)} - {self._format_duration(segment.end)}]"

                    if hasattr(segment, "speaker") and segment.speaker:
                        result.append(
                            f"{timestamp} **Speaker {segment.speaker}:** {segment.text}\n"
                        )
                    else:
                        result.append(f"{timestamp} {segment.text}\n")
            elif hasattr(topic_segment, "text") and topic_segment.text:
                timestamp = f"[{self._format_duration(topic_segment.start)} - {self._format_duration(topic_segment.end)}]"
                result.append(f"{timestamp} {topic_segment.text}\n")

            result.append("\n")

        return "\n".join(result)
