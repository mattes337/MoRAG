"""Video to Markdown converter using existing MoRAG video processor."""

import time
from pathlib import Path
from typing import Union
import structlog

from .base import BaseConverter, ConversionOptions, ConversionResult, QualityScore
from .quality import ConversionQualityValidator
from ..processors.video import video_processor, VideoConfig

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
            # Create video configuration from conversion options
            video_config = VideoConfig(
                extract_audio=options.format_options.get('include_audio', True),
                generate_thumbnails=options.format_options.get('generate_thumbnails', True),
                thumbnail_count=options.format_options.get('thumbnail_count', 5),
                extract_keyframes=options.format_options.get('extract_keyframes', True),
                max_keyframes=options.format_options.get('max_keyframes', 10),
                audio_format=options.format_options.get('audio_format', 'wav'),
                thumbnail_size=tuple(options.format_options.get('thumbnail_size', [320, 240])),
                thumbnail_format=options.format_options.get('thumbnail_format', 'jpg'),
                keyframe_threshold=options.format_options.get('keyframe_threshold', 0.3)
            )

            # Use existing MoRAG video processor
            video_result = await video_processor.process_video(file_path, video_config)

            # Convert to structured markdown
            markdown_content = await self._create_structured_markdown(video_result, options, file_path)
            
            # Calculate quality score
            # Convert VideoMetadata to dictionary for quality validation
            metadata_dict = self._enhance_metadata(video_result.metadata, file_path)
            quality_score = self.quality_validator.validate_conversion(str(file_path), ConversionResult(
                content=markdown_content,
                metadata=metadata_dict
            ))
            
            processing_time = time.time() - start_time
            
            result = ConversionResult(
                content=markdown_content,
                metadata=metadata_dict,  # Use the already converted metadata
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
                duration=video_result.metadata.duration
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
    
    async def _create_structured_markdown(self, video_result, options: ConversionOptions, file_path: Path) -> str:
        """Create structured markdown from video processing result.

        Args:
            video_result: Video processing result from MoRAG processor
            options: Conversion options
            file_path: Path to the original video file

        Returns:
            Structured markdown content
        """
        sections = []

        # Document header - use file path since metadata doesn't have filename
        title = file_path.stem
        sections.append(f"# Video Analysis: {title}")
        sections.append("")

        # Metadata section
        if options.include_metadata:
            sections.append("## Video Information")
            sections.append("")

            duration = video_result.metadata.duration
            duration_str = f"{duration:.1f} seconds" if duration < 60 else f"{duration/60:.1f} minutes"

            # Create resolution string
            resolution = f"{video_result.metadata.width}x{video_result.metadata.height}"

            metadata_items = [
                ("**Duration**", duration_str),
                ("**Resolution**", resolution),
                ("**Frame Rate**", f"{video_result.metadata.fps:.1f} fps"),
                ("**Format**", video_result.metadata.format),
                ("**Codec**", video_result.metadata.codec),
                ("**File Size**", f"{video_result.metadata.file_size / (1024*1024):.1f} MB"),
                ("**Has Audio**", "Yes" if video_result.metadata.has_audio else "No"),
                ("**Processing Method**", "MoRAG Video Processor")
            ]

            if video_result.metadata.has_audio and video_result.metadata.audio_codec:
                metadata_items.append(("**Audio Codec**", video_result.metadata.audio_codec))

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
        # Note: Audio transcription integration needs to be implemented
        # The video processor extracts audio but doesn't transcribe it yet
        if video_result.audio_path and video_result.metadata.has_audio:
            # Calculate total duration for timestamp
            duration = video_result.metadata.duration
            duration_str = self._format_timestamp(duration)

            sections.append(f"# Audio Content [00:00 - {duration_str}]")
            sections.append("")
            sections.append("*Audio track extracted but transcription not yet integrated.*")
            sections.append(f"*Audio file: {video_result.audio_path.name}*")
            sections.append("")

            # TODO: Integrate audio transcription
            # This would require calling the audio processor on the extracted audio
            # and formatting the result according to user preferences
        elif not video_result.metadata.has_audio:
            sections.append("# Audio Content")
            sections.append("")
            sections.append("*No audio track detected in this video.*")
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
    
    def _enhance_metadata(self, original_metadata, file_path: Path) -> dict:
        """Enhance metadata with additional information.

        Args:
            original_metadata: Original VideoMetadata from processor
            file_path: Path to original file

        Returns:
            Enhanced metadata dictionary
        """
        # Convert VideoMetadata dataclass to dictionary
        enhanced = {
            'duration': original_metadata.duration,
            'width': original_metadata.width,
            'height': original_metadata.height,
            'fps': original_metadata.fps,
            'codec': original_metadata.codec,
            'bitrate': original_metadata.bitrate,
            'file_size': original_metadata.file_size,
            'format': original_metadata.format,
            'has_audio': original_metadata.has_audio,
            'audio_codec': original_metadata.audio_codec,
            'creation_time': original_metadata.creation_time
        }

        # Add file information
        enhanced.update({
            'original_filename': file_path.name,
            'original_file_size': file_path.stat().st_size,
            'conversion_format': 'video_to_markdown',
            'converter_version': '1.0.0',
            'file_extension': file_path.suffix.lower()
        })

        return enhanced
