"""Audio to Markdown converter with speaker diarization and topic segmentation."""

import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass, field
import structlog

from morag_core.exceptions import ProcessingError as ConversionError
from morag_audio.processor import AudioProcessingResult, AudioSegment

logger = structlog.get_logger(__name__)


@dataclass
class AudioConversionOptions:
    """Options for audio conversion to markdown."""
    include_timestamps: bool = True
    include_speakers: bool = True
    include_topics: bool = True
    timestamp_format: str = "[%H:%M:%S]"  # Format for timestamps
    group_by_speaker: bool = True  # Group consecutive segments by the same speaker
    group_by_topic: bool = True  # Group segments by topic
    include_metadata: bool = True  # Include metadata section in markdown
    metadata_fields: List[str] = field(default_factory=lambda: [
        "duration", "num_speakers", "speakers", "num_topics", "word_count"
    ])


@dataclass
class AudioConversionResult:
    """Result of audio conversion to markdown."""
    content: str
    metadata: Dict[str, Any]
    processing_time: float
    success: bool = True
    error_message: Optional[str] = None


class AudioConverter:
    """Converts audio processing results to structured markdown."""

    def __init__(self):
        """Initialize the audio converter."""
        self.name = "MoRAG Audio Converter"
        self.supported_formats = ['audio', 'mp3', 'wav', 'm4a', 'flac', 'mp4', 'avi', 'mov', 'mkv']
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        return format_type.lower() in self.supported_formats
    
    async def convert_to_json(self,
                             result: AudioProcessingResult,
                             options: Optional[AudioConversionOptions] = None) -> Dict[str, Any]:
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
                    "error": result.error_message
                }

            # Group segments by topic if topic segmentation is enabled
            topics = []
            if options.include_topics and any(hasattr(segment, 'topic') and segment.topic is not None for segment in result.segments):
                # Group by topic
                topic_groups = {}
                for segment in result.segments:
                    topic_id = getattr(segment, 'topic', 0) if hasattr(segment, 'topic') else 0
                    if topic_id not in topic_groups:
                        topic_groups[topic_id] = []
                    topic_groups[topic_id].append(segment)

                # Create topic entries
                for topic_id, segments in sorted(topic_groups.items()):
                    topic_data = {
                        "timestamp": int(segments[0].start) if segments else 0,
                        "sentences": []
                    }

                    for segment in segments:
                        sentence = {
                            "timestamp": int(segment.start),
                            "speaker": getattr(segment, 'speaker', 1) if hasattr(segment, 'speaker') else 1,
                            "text": segment.text
                        }
                        topic_data["sentences"].append(sentence)

                    topics.append(topic_data)
            else:
                # Single topic with all segments
                topic_data = {
                    "timestamp": int(result.segments[0].start) if result.segments else 0,
                    "sentences": []
                }

                for segment in result.segments:
                    sentence = {
                        "timestamp": int(segment.start),
                        "speaker": getattr(segment, 'speaker', 1) if hasattr(segment, 'speaker') else 1,
                        "text": segment.text
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
                "topics": topics
            }

        except Exception as e:
            logger.error("Failed to convert to JSON", error=str(e))
            return {
                "title": "",
                "filename": result.file_path,
                "metadata": result.metadata,
                "topics": [],
                "error": str(e)
            }

    async def convert_to_markdown(self,
                                result: AudioProcessingResult,
                                options: Optional[AudioConversionOptions] = None) -> AudioConversionResult:
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
            if not result.success or not result.segments:
                raise ConversionError(f"Cannot convert unsuccessful processing result: {result.error_message}")
            
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
                        markdown_content.append(f"- **{field.replace('_', ' ').title()}**: {value}")
                markdown_content.append("\n")
            
            # Add transcript section
            markdown_content.append("## Full Transcript\n")
            markdown_content.append(result.transcript)
            markdown_content.append("\n")
            
            # Add detailed transcript section with speakers and timestamps
            markdown_content.append("## Detailed Transcript\n")
            
            # Group segments by topic if requested
            if options.group_by_topic and any(segment.topic_id is not None for segment in result.segments):
                topic_groups = {}
                for segment in result.segments:
                    topic_id = segment.topic_id if segment.topic_id is not None else -1
                    if topic_id not in topic_groups:
                        topic_groups[topic_id] = []
                    topic_groups[topic_id].append(segment)
                
                # Process each topic group
                for topic_id, segments in sorted(topic_groups.items()):
                    if topic_id != -1:
                        markdown_content.append(f"### Topic {topic_id + 1}\n")
                    else:
                        markdown_content.append("### Ungrouped Content\n")
                    
                    # Process segments within this topic
                    markdown_content.extend(self._format_segments(segments, options))
                    markdown_content.append("\n")
            else:
                # Process all segments without topic grouping
                markdown_content.extend(self._format_segments(result.segments, options))
            
            # Join all content
            content = "\n".join(markdown_content)
            
            processing_time = time.time() - start_time
            
            return AudioConversionResult(
                content=content,
                metadata=result.metadata,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error("Audio conversion to markdown failed", error=str(e))
            
            processing_time = time.time() - start_time
            return AudioConversionResult(
                content="",
                metadata={"error": str(e)},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _format_segments(self, segments: List[AudioSegment], options: AudioConversionOptions) -> List[str]:
        """Format segments into markdown lines."""
        markdown_lines = []
        
        if options.group_by_speaker and any(segment.speaker for segment in segments):
            # Group consecutive segments by speaker
            current_speaker = None
            current_speaker_text = []
            current_start = None
            current_end = None
            
            for segment in segments:
                speaker = segment.speaker or "Unknown"
                
                # Format timestamp if requested
                timestamp = ""
                if options.include_timestamps:
                    # Convert seconds to HH:MM:SS format
                    hours, remainder = divmod(int(segment.start), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    timestamp = options.timestamp_format.replace("%H", f"{hours:02d}")\
                                                .replace("%M", f"{minutes:02d}")\
                                                .replace("%S", f"{seconds:02d}")
                
                if speaker != current_speaker:
                    # Output the previous speaker's text
                    if current_speaker and current_speaker_text:
                        speaker_timestamp = ""
                        if options.include_timestamps and current_start is not None and current_end is not None:
                            # Format start-end range
                            start_hours, start_remainder = divmod(int(current_start), 3600)
                            start_minutes, start_seconds = divmod(start_remainder, 60)
                            end_hours, end_remainder = divmod(int(current_end), 3600)
                            end_minutes, end_seconds = divmod(end_remainder, 60)
                            
                            start_time = f"{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d}"
                            end_time = f"{end_hours:02d}:{end_minutes:02d}:{end_seconds:02d}"
                            speaker_timestamp = f" ({start_time} - {end_time})"
                        
                        markdown_lines.append(f"**{current_speaker}{speaker_timestamp}**: {' '.join(current_speaker_text)}\n")
                    
                    # Start new speaker
                    current_speaker = speaker
                    current_speaker_text = [segment.text]
                    current_start = segment.start
                    current_end = segment.end
                else:
                    # Continue with the same speaker
                    current_speaker_text.append(segment.text)
                    current_end = segment.end
            
            # Output the last speaker's text
            if current_speaker and current_speaker_text:
                speaker_timestamp = ""
                if options.include_timestamps and current_start is not None and current_end is not None:
                    # Format start-end range
                    start_hours, start_remainder = divmod(int(current_start), 3600)
                    start_minutes, start_seconds = divmod(start_remainder, 60)
                    end_hours, end_remainder = divmod(int(current_end), 3600)
                    end_minutes, end_seconds = divmod(end_remainder, 60)
                    
                    start_time = f"{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d}"
                    end_time = f"{end_hours:02d}:{end_minutes:02d}:{end_seconds:02d}"
                    speaker_timestamp = f" ({start_time} - {end_time})"
                
                markdown_lines.append(f"**{current_speaker}{speaker_timestamp}**: {' '.join(current_speaker_text)}\n")
        else:
            # Process each segment individually
            for segment in segments:
                line_parts = []
                
                # Add timestamp if requested
                if options.include_timestamps:
                    hours, remainder = divmod(int(segment.start), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    timestamp = options.timestamp_format.replace("%H", f"{hours:02d}")\
                                                .replace("%M", f"{minutes:02d}")\
                                                .replace("%S", f"{seconds:02d}")
                    line_parts.append(timestamp)
                
                # Add speaker if available and requested
                if options.include_speakers and segment.speaker:
                    line_parts.append(f"**{segment.speaker}**:")
                
                # Add the text
                line_parts.append(segment.text)
                
                # Join parts and add to markdown lines
                markdown_lines.append(" ".join(line_parts))
            
            # Add a blank line at the end
            if markdown_lines:
                markdown_lines.append("")
        
        return markdown_lines