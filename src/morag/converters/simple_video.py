"""Simple video converter that extracts basic metadata and audio transcription."""

import time
from pathlib import Path
from typing import Union
import structlog

from .base import BaseConverter, ConversionOptions, ConversionResult
from ..processors.video import video_processor
from ..core.exceptions import ProcessingError
from .quality import ConversionQualityValidator

logger = structlog.get_logger(__name__)


class SimpleVideoConverter(BaseConverter):
    """Simple video converter that focuses on basic transcription without advanced features."""

    def __init__(self):
        super().__init__("Simple Video Converter")
        self.quality_validator = ConversionQualityValidator()

    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        supported_formats = ['video', 'mp4', 'avi', 'mov', 'mkv']
        return format_type.lower() in supported_formats
    
    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert video to basic markdown with transcription.
        
        Args:
            file_path: Path to video file
            options: Conversion options
            
        Returns:
            ConversionResult with basic markdown content
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        await self.validate_input(file_path)
        
        logger.info(
            "Starting simple video conversion",
            file_path=str(file_path),
            converter=self.name
        )
        
        try:
            # Use basic video processing without advanced features
            from ..processors.video import VideoConfig
            
            # Create simplified config
            video_config = VideoConfig(
                extract_audio=True,
                enable_enhanced_audio=False,  # Disable enhanced features
                extract_keyframes=False,      # Disable keyframes
                generate_thumbnails=False,    # Disable thumbnails
                thumbnail_count=0,
                max_keyframes=0,
                audio_format='mp3'
            )
            
            # Process video with basic settings
            video_result = await video_processor.process_video(file_path, video_config)
            
            # Create simple markdown content
            markdown_content = await self._create_simple_markdown(video_result, file_path)
            
            # Create basic metadata
            metadata = {
                'title': file_path.stem,
                'format': 'video',
                'converter': self.name,
                'processing_time': time.time() - start_time,
                'file_size': file_path.stat().st_size,
                'audio_extracted': video_result.audio_path is not None
            }
            
            # Add video metadata if available
            if video_result.metadata:
                metadata.update({
                    'duration': getattr(video_result.metadata, 'duration', None),
                    'width': getattr(video_result.metadata, 'width', None),
                    'height': getattr(video_result.metadata, 'height', None),
                    'fps': getattr(video_result.metadata, 'fps', None)
                })
            
            # Calculate word count
            word_count = len(markdown_content.split()) if markdown_content else 0
            
            # Create result
            result = ConversionResult(
                content=markdown_content,
                metadata=metadata,
                success=True,
                word_count=word_count,
                processing_time=time.time() - start_time,
                converter_used=self.name
            )
            
            # Calculate quality score
            result.quality_score = self.quality_validator.validate_conversion(str(file_path), result)
            
            logger.info(
                "Simple video conversion completed",
                file_path=str(file_path),
                word_count=word_count,
                processing_time=result.processing_time,
                quality_score=result.quality_score.overall_score if result.quality_score else None
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Simple video conversion failed",
                file_path=str(file_path),
                error=str(e),
                error_type=type(e).__name__
            )
            
            return ConversionResult(
                content="",
                metadata={'title': file_path.stem, 'format': 'video', 'converter': self.name},
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                converter_used=self.name
            )
    
    async def _create_simple_markdown(self, video_result, file_path: Path) -> str:
        """Create simple markdown content from video processing result.
        
        Args:
            video_result: Video processing result
            file_path: Original file path
            
        Returns:
            Simple markdown content
        """
        content_parts = []
        
        # Title
        content_parts.append(f"# {file_path.stem}")
        content_parts.append("")
        
        # Basic metadata
        if video_result.metadata:
            content_parts.append("## Video Information")
            if hasattr(video_result.metadata, 'duration') and video_result.metadata.duration:
                duration_min = int(video_result.metadata.duration // 60)
                duration_sec = int(video_result.metadata.duration % 60)
                content_parts.append(f"- **Duration**: {duration_min}:{duration_sec:02d}")
            
            if hasattr(video_result.metadata, 'width') and video_result.metadata.width:
                content_parts.append(f"- **Resolution**: {video_result.metadata.width}x{video_result.metadata.height}")
            
            if hasattr(video_result.metadata, 'fps') and video_result.metadata.fps:
                content_parts.append(f"- **Frame Rate**: {video_result.metadata.fps:.1f} fps")
            
            content_parts.append("")
        
        # Audio transcription if available
        if hasattr(video_result, 'audio_processing_result') and video_result.audio_processing_result:
            audio_result = video_result.audio_processing_result
            
            if hasattr(audio_result, 'transcript') and audio_result.transcript:
                content_parts.append("## Transcript")
                content_parts.append("")
                
                # Simple transcript format
                if hasattr(audio_result.transcript, 'segments') and audio_result.transcript.segments:
                    for segment in audio_result.transcript.segments:
                        if hasattr(segment, 'text') and segment.text.strip():
                            # Format: [MM:SS] Text
                            if hasattr(segment, 'start_time'):
                                start_min = int(segment.start_time // 60)
                                start_sec = int(segment.start_time % 60)
                                content_parts.append(f"[{start_min}:{start_sec:02d}] {segment.text.strip()}")
                            else:
                                content_parts.append(f"{segment.text.strip()}")
                else:
                    # Fallback to simple text if segments not available
                    transcript_text = getattr(audio_result.transcript, 'text', '')
                    if transcript_text:
                        content_parts.append(transcript_text)
                
                content_parts.append("")
        
        # Processing information
        content_parts.append("## Processing Details")
        content_parts.append(f"- **Converter**: {self.name}")
        content_parts.append(f"- **Audio Extracted**: {'Yes' if video_result.audio_path else 'No'}")
        
        return "\n".join(content_parts)
