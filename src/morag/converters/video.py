"""Video to Markdown converter using existing MoRAG video processor."""

import time
from pathlib import Path
from typing import Union
import structlog

from .base import BaseConverter, ConversionOptions, ConversionResult, QualityScore
from .quality import ConversionQualityValidator
from ..processors.video import video_processor

logger = structlog.get_logger(__name__)


class VideoConverter(BaseConverter):
    """Video to Markdown converter using MoRAG's video processor."""
    
    def __init__(self):
        super().__init__("MoRAG Video Converter")
        self.supported_formats = ['video', 'mp4', 'avi', 'mov', 'mkv']
        self.quality_validator = ConversionQualityValidator()
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        return format_type.lower() in self.supported_formats
    
    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert video to structured markdown with transcription and keyframes.
        
        Args:
            file_path: Path to video file
            options: Conversion options
            
        Returns:
            ConversionResult with markdown content
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        await self.validate_input(file_path)
        
        logger.info(
            "Starting video conversion",
            file_path=str(file_path),
            extract_keyframes=options.format_options.get('extract_keyframes', True),
            include_audio=options.format_options.get('include_audio', True)
        )
        
        try:
            # Use existing MoRAG video processor
            video_result = await video_processor.process_video(str(file_path))
            
            # Convert to structured markdown
            markdown_content = await self._create_structured_markdown(video_result, options)
            
            # Calculate quality score
            quality_score = self.quality_validator.validate_conversion(str(file_path), ConversionResult(
                content=markdown_content,
                metadata=video_result.metadata
            ))
            
            processing_time = time.time() - start_time
            
            result = ConversionResult(
                content=markdown_content,
                metadata=self._enhance_metadata(video_result.metadata, file_path),
                quality_score=quality_score,
                processing_time=processing_time,
                success=True,
                original_format='video',
                converter_used=self.name
            )
            
            logger.info(
                "Video conversion completed",
                processing_time=processing_time,
                quality_score=quality_score.overall_score,
                word_count=result.word_count,
                duration=video_result.metadata.get('duration', 0)
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Video conversion failed: {str(e)}"
            
            logger.error(
                "Video conversion failed",
                error=str(e),
                error_type=type(e).__name__,
                processing_time=processing_time
            )
            
            return ConversionResult(
                content="",
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=error_msg,
                original_format='video',
                converter_used=self.name
            )
    
    async def _create_structured_markdown(self, video_result, options: ConversionOptions) -> str:
        """Create structured markdown from video processing result.
        
        Args:
            video_result: Video processing result from MoRAG processor
            options: Conversion options
            
        Returns:
            Structured markdown content
        """
        sections = []
        
        # Document header
        filename = video_result.metadata.get('filename', 'Video File')
        title = filename.rsplit('.', 1)[0]  # Remove extension
        sections.append(f"# Video Analysis: {title}")
        sections.append("")
        
        # Metadata section
        if options.include_metadata:
            sections.append("## Video Information")
            sections.append("")
            
            duration = video_result.metadata.get('duration', 0)
            duration_str = f"{duration:.1f} seconds" if duration < 60 else f"{duration/60:.1f} minutes"
            
            metadata_items = [
                ("**Source**", video_result.metadata.get('filename', 'Unknown')),
                ("**Duration**", duration_str),
                ("**Resolution**", video_result.metadata.get('resolution', 'Unknown')),
                ("**Frame Rate**", f"{video_result.metadata.get('fps', 'Unknown')} fps"),
                ("**Format**", video_result.metadata.get('format', 'Unknown')),
                ("**Processing Method**", video_result.metadata.get('processor_used', 'MoRAG Video Processor'))
            ]
            
            for label, value in metadata_items:
                if value and value != 'Unknown':
                    sections.append(f"{label}: {value}")
            
            sections.append("")
        
        # Summary section (if available)
        if hasattr(video_result, 'summary') and video_result.summary:
            sections.append("## Summary")
            sections.append("")
            sections.append(video_result.summary)
            sections.append("")
        
        # Audio transcript section - format as topic with timestamps
        if hasattr(video_result, 'audio_transcript') and video_result.audio_transcript:
            # Calculate total duration for timestamp
            duration = video_result.metadata.get('duration', 0)
            duration_str = self._format_timestamp(duration)

            sections.append(f"# Audio Content [00:00 - {duration_str}]")
            sections.append("")

            # Format transcript with speaker labels
            transcript_lines = video_result.audio_transcript.split('\n')
            for line in transcript_lines:
                line = line.strip()
                if line:
                    # Add speaker label if not already present
                    if not line.startswith('Speaker_') and not line.startswith('SPEAKER_'):
                        sections.append(f"Speaker_00: {line}")
                    else:
                        sections.append(line)
            sections.append("")
        elif hasattr(video_result, 'transcript') and video_result.transcript:
            # Calculate total duration for timestamp
            duration = video_result.metadata.get('duration', 0)
            duration_str = self._format_timestamp(duration)

            sections.append(f"# Audio Content [00:00 - {duration_str}]")
            sections.append("")

            # Format transcript with speaker labels
            transcript_lines = video_result.transcript.split('\n')
            for line in transcript_lines:
                line = line.strip()
                if line:
                    # Add speaker label if not already present
                    if not line.startswith('Speaker_') and not line.startswith('SPEAKER_'):
                        sections.append(f"Speaker_00: {line}")
                    else:
                        sections.append(line)
            sections.append("")
        
        # Keyframes section
        if hasattr(video_result, 'keyframes') and video_result.keyframes and options.extract_images:
            sections.append("## Visual Timeline")
            sections.append("")
            
            for i, keyframe in enumerate(video_result.keyframes, 1):
                timestamp = keyframe.get('timestamp', 0)
                timestamp_str = self._format_timestamp(timestamp)
                
                sections.append(f"### Keyframe {i} ({timestamp_str})")
                
                if keyframe.get('description'):
                    sections.append(f"**Scene**: {keyframe['description']}")
                
                if keyframe.get('objects'):
                    objects_str = ", ".join(keyframe['objects'])
                    sections.append(f"**Visual Elements**: {objects_str}")
                
                if keyframe.get('text_content'):
                    sections.append(f"**Text Visible**: {keyframe['text_content']}")
                
                sections.append("")
        
        # Scene analysis (if available)
        if hasattr(video_result, 'scenes') and video_result.scenes:
            sections.append("## Scene Analysis")
            sections.append("")
            
            for i, scene in enumerate(video_result.scenes, 1):
                start_time = self._format_timestamp(scene.get('start_time', 0))
                end_time = self._format_timestamp(scene.get('end_time', 0))
                
                sections.append(f"### Scene {i} ({start_time} - {end_time})")
                
                if scene.get('description'):
                    sections.append(f"**Description**: {scene['description']}")
                
                if scene.get('activity'):
                    sections.append(f"**Activity**: {scene['activity']}")
                
                sections.append("")

        return "\n".join(sections)
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _enhance_metadata(self, original_metadata: dict, file_path: Path) -> dict:
        """Enhance metadata with additional information.
        
        Args:
            original_metadata: Original metadata from processor
            file_path: Path to original file
            
        Returns:
            Enhanced metadata dictionary
        """
        enhanced = original_metadata.copy()
        
        # Add file information
        enhanced.update({
            'original_filename': file_path.name,
            'file_size': file_path.stat().st_size,
            'conversion_format': 'video_to_markdown',
            'converter_version': '1.0.0',
            'file_extension': file_path.suffix.lower()
        })
        
        return enhanced
