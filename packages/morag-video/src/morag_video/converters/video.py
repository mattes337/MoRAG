"""Video converter module for MoRAG."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import structlog

from morag_core.converters import ConversionResult, ConversionQualityValidator
from morag_video.processor import VideoProcessingResult

logger = structlog.get_logger(__name__)


@dataclass
class VideoConversionOptions:
    """Options for video conversion."""
    include_metadata: bool = True
    include_thumbnails: bool = True
    include_keyframes: bool = True
    include_transcript: bool = True
    group_by_speaker: bool = False
    group_by_topic: bool = False
    max_thumbnails: int = 5
    max_keyframes: int = 10
    include_ocr: bool = True


class VideoConverter:
    """Converts video processing results to structured formats."""

    SUPPORTED_FORMATS = ["mp4", "avi", "mov", "mkv", "webm"]

    def __init__(self):
        """Initialize the video converter."""
        self.quality_validator = ConversionQualityValidator()
        logger.info("Video converter initialized")

    async def convert(self, 
                     file_path: Union[str, Path], 
                     options: Optional[VideoConversionOptions] = None) -> ConversionResult:
        """Convert a video file to structured markdown.
        
        Args:
            file_path: Path to the video file
            options: Conversion options
            
        Returns:
            ConversionResult with structured markdown
        """
        from morag_video.processor import VideoProcessor, VideoConfig
        
        file_path = Path(file_path)
        options = options or VideoConversionOptions()
        
        # Configure video processor based on conversion options
        config = VideoConfig(
            extract_audio=options.include_transcript,
            generate_thumbnails=options.include_thumbnails,
            extract_keyframes=options.include_keyframes,
            enable_enhanced_audio=options.group_by_speaker or options.group_by_topic,
            enable_speaker_diarization=options.group_by_speaker,
            enable_topic_segmentation=options.group_by_topic,
            enable_ocr=options.include_ocr
        )
        
        processor = VideoProcessor(config)
        
        try:
            # Process the video
            logger.info("Processing video for conversion", file_path=str(file_path))
            result = await processor.process_video(file_path)
            
            # Create structured markdown
            markdown = await self.convert_to_markdown(result, options)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(result, options)
            
            # Clean up temporary files
            processor.cleanup_temp_files(result.temp_files)
            
            return ConversionResult(
                content=markdown,
                quality_score=quality_score,
                metadata={
                    "duration": result.metadata.duration,
                    "resolution": f"{result.metadata.width}x{result.metadata.height}",
                    "fps": result.metadata.fps,
                    "format": result.metadata.format,
                    "has_audio": result.metadata.has_audio,
                    "thumbnails_count": len(result.thumbnails),
                    "keyframes_count": len(result.keyframes),
                    "has_transcript": result.audio_processing_result is not None and 
                                     result.audio_processing_result.transcript is not None,
                    "has_ocr": bool(result.ocr_results)
                }
            )
            
        except Exception as e:
            logger.error("Video conversion failed", file_path=str(file_path), error=str(e))
            raise

    async def convert_to_markdown(self, 
                                result: VideoProcessingResult, 
                                options: Optional[VideoConversionOptions] = None) -> str:
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
        
        # Add title
        if hasattr(result, "file_path") and result.file_path:
            title = Path(result.file_path).stem
            markdown_parts.append(f"# {title}\n")
        else:
            markdown_parts.append("# Video Analysis\n")
        
        # Add metadata section
        if options.include_metadata:
            markdown_parts.append("## Video Information\n")
            markdown_parts.append("| Property | Value |\n|----------|-------|")
            markdown_parts.append(f"| Duration | {self._format_duration(result.metadata.duration)} |")
            markdown_parts.append(f"| Resolution | {result.metadata.width}x{result.metadata.height} |")
            markdown_parts.append(f"| FPS | {result.metadata.fps} |")
            markdown_parts.append(f"| Format | {result.metadata.format} |")
            markdown_parts.append(f"| Video Codec | {result.metadata.codec} |")
            
            if result.metadata.has_audio:
                markdown_parts.append(f"| Audio | Yes ({result.metadata.audio_codec}) |")
            else:
                markdown_parts.append("| Audio | No |")
                
            if result.metadata.creation_time:
                markdown_parts.append(f"| Creation Time | {result.metadata.creation_time} |")
                
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
        if options.include_transcript and result.audio_processing_result and result.audio_processing_result.transcript:
            markdown_parts.append("## Transcript\n")

            # Handle different transcript formats based on options
            if options.group_by_speaker and result.audio_processing_result.segments:
                # Group by speaker using segments
                markdown_parts.append(self._format_segments_by_speaker(result.audio_processing_result.segments))
            elif options.group_by_topic and result.audio_processing_result.segments:
                # Group by topic using segments
                markdown_parts.append(self._format_segments_by_topic(result.audio_processing_result.segments))
            elif result.audio_processing_result.segments:
                # Use regular segments
                markdown_parts.append(self._format_segments(result.audio_processing_result.segments))
            else:
                # Use full transcript
                markdown_parts.append(result.audio_processing_result.transcript)

            markdown_parts.append("\n")
        
        # Add thumbnails section if available
        if options.include_thumbnails and result.thumbnails:
            markdown_parts.append("## Thumbnails\n")
            
            # Limit the number of thumbnails based on options
            thumbnails_to_show = result.thumbnails[:options.max_thumbnails]
            
            for i, thumbnail_path in enumerate(thumbnails_to_show):
                timestamp = self._get_timestamp_from_filename(thumbnail_path)
                markdown_parts.append(f"### Thumbnail {i+1} {timestamp}\n")
                markdown_parts.append(f"![Thumbnail {i+1}]({thumbnail_path})\n\n")
                
            if len(result.thumbnails) > options.max_thumbnails:
                markdown_parts.append(f"*{len(result.thumbnails) - options.max_thumbnails} more thumbnails not shown*\n\n")
        
        # Add keyframes section if available
        if options.include_keyframes and result.keyframes:
            markdown_parts.append("## Key Frames\n")
            
            # Limit the number of keyframes based on options
            keyframes_to_show = result.keyframes[:options.max_keyframes]
            
            for i, keyframe_path in enumerate(keyframes_to_show):
                timestamp = self._get_timestamp_from_filename(keyframe_path)
                markdown_parts.append(f"### Key Frame {i+1} {timestamp}\n")
                markdown_parts.append(f"![Key Frame {i+1}]({keyframe_path})\n\n")
                
                # Add OCR text for this keyframe if available
                if options.include_ocr and result.ocr_results and str(keyframe_path) in result.ocr_results:
                    ocr_text = result.ocr_results[str(keyframe_path)]
                    if ocr_text.strip():
                        markdown_parts.append(f"**Text detected:**\n\n{ocr_text.strip()}\n\n")
                
            if len(result.keyframes) > options.max_keyframes:
                markdown_parts.append(f"*{len(result.keyframes) - options.max_keyframes} more key frames not shown*\n\n")
        
        return "\n".join(markdown_parts)

    def _calculate_quality_score(self, 
                               result: VideoProcessingResult, 
                               options: VideoConversionOptions) -> float:
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
            thumbnail_score = min(1.0, len(result.thumbnails) / 5)  # Max score at 5+ thumbnails
            score += thumbnail_weight * thumbnail_score
            total_weight += thumbnail_weight
        
        # Score for keyframes
        if options.include_keyframes and result.keyframes:
            keyframe_weight = 0.15
            keyframe_score = min(1.0, len(result.keyframes) / 10)  # Max score at 10+ keyframes
            score += keyframe_weight * keyframe_score
            total_weight += keyframe_weight
        
        # Score for transcript
        if options.include_transcript and result.audio_processing_result and result.audio_processing_result.transcript:
            transcript_weight = 0.3
            
            # Base transcript score
            transcript_score = 0.5
            
            # Bonus for segments
            if result.audio_processing_result.segments:
                transcript_score += 0.2
            
            # Bonus for speaker diarization
            if options.group_by_speaker and result.audio_processing_result.speaker_segments:
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
            frames_with_text = sum(1 for text in result.ocr_results.values() if text.strip())
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
        match = re.search(r'(\d+)h(\d+)m(\d+)s', filename)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            if hours > 0:
                return f"({hours:02d}:{minutes:02d}:{seconds:02d})"
            return f"({minutes:02d}:{seconds:02d})"
        
        # Check for seconds format
        match = re.search(r'_(\d+)s', filename)
        if match:
            total_seconds = int(match.group(1))
            minutes, seconds = divmod(total_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            
            if hours > 0:
                return f"({hours:02d}:{minutes:02d}:{seconds:02d})"
            return f"({minutes:02d}:{seconds:02d})"
        
        return ""

    def _format_segments(self, segments: List[Any]) -> str:
        """Format transcript segments."""
        result = []

        for segment in segments:
            timestamp = f"[{self._format_duration(segment.start)} - {self._format_duration(segment.end)}]"
            result.append(f"{timestamp} {segment.text}\n")

        return "\n".join(result)

    def _format_segments_by_speaker(self, segments: List[Any]) -> str:
        """Format transcript segments grouped by speaker."""
        # Group segments by speaker
        speaker_groups = {}
        for segment in segments:
            speaker = getattr(segment, 'speaker', 'Unknown Speaker')
            if speaker not in speaker_groups:
                speaker_groups[speaker] = []
            speaker_groups[speaker].append(segment)

        result = []
        for speaker, speaker_segments in speaker_groups.items():
            result.append(f"### {speaker}\n")

            for segment in speaker_segments:
                timestamp = f"[{self._format_duration(segment.start)} - {self._format_duration(segment.end)}]"
                result.append(f"{timestamp} {segment.text}\n")

            result.append("\n")

        return "\n".join(result)

    def _format_segments_by_topic(self, segments: List[Any]) -> str:
        """Format transcript segments grouped by topic."""
        # Group segments by topic
        topic_groups = {}
        for segment in segments:
            topic_id = getattr(segment, 'topic_id', None)
            topic_key = f"Topic {topic_id + 1}" if topic_id is not None else "Ungrouped Content"
            if topic_key not in topic_groups:
                topic_groups[topic_key] = []
            topic_groups[topic_key].append(segment)

        result = []
        for topic, topic_segments in topic_groups.items():
            result.append(f"### {topic}\n")

            for segment in topic_segments:
                timestamp = f"[{self._format_duration(segment.start)} - {self._format_duration(segment.end)}]"
                result.append(f"{timestamp} {segment.text}\n")

            result.append("\n")

        return "\n".join(result)

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
                        result.append(f"{timestamp} **Speaker {segment.speaker}:** {segment.text}\n")
                    else:
                        result.append(f"{timestamp} {segment.text}\n")
            elif hasattr(topic_segment, "text") and topic_segment.text:
                timestamp = f"[{self._format_duration(topic_segment.start)} - {self._format_duration(topic_segment.end)}]"
                result.append(f"{timestamp} {topic_segment.text}\n")
            
            result.append("\n")
        
        return "\n".join(result)