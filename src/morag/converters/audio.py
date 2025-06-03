"""Audio to Markdown converter using existing MoRAG audio processor."""

import time
from pathlib import Path
from typing import Union
import structlog

from .base import BaseConverter, ConversionOptions, ConversionResult, QualityScore
from .quality import ConversionQualityValidator
from ..processors.audio import audio_processor

logger = structlog.get_logger(__name__)


class AudioConverter(BaseConverter):
    """Audio to Markdown converter using MoRAG's audio processor."""
    
    def __init__(self):
        super().__init__("MoRAG Audio Converter")
        self.supported_formats = ['audio', 'mp3', 'wav', 'm4a', 'flac']
        self.quality_validator = ConversionQualityValidator()
    
    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        return format_type.lower() in self.supported_formats
    
    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert audio to structured markdown with transcription.
        
        Args:
            file_path: Path to audio file
            options: Conversion options
            
        Returns:
            ConversionResult with markdown content
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        await self.validate_input(file_path)
        
        logger.info(
            "Starting audio conversion",
            file_path=str(file_path),
            enable_diarization=options.format_options.get('enable_diarization', False),
            include_timestamps=options.format_options.get('include_timestamps', True)
        )
        
        try:
            # Use existing MoRAG audio processor
            audio_result = await audio_processor.process_audio(str(file_path))
            
            # Convert to structured markdown
            markdown_content = await self._create_structured_markdown(audio_result, options)
            
            # Calculate quality score
            quality_score = self.quality_validator.validate_conversion(str(file_path), ConversionResult(
                content=markdown_content,
                metadata=audio_result.metadata
            ))
            
            processing_time = time.time() - start_time
            
            result = ConversionResult(
                content=markdown_content,
                metadata=self._enhance_metadata(audio_result.metadata, file_path),
                quality_score=quality_score,
                processing_time=processing_time,
                success=True,
                original_format='audio',
                converter_used=self.name
            )
            
            logger.info(
                "Audio conversion completed",
                processing_time=processing_time,
                quality_score=quality_score.overall_score,
                word_count=result.word_count,
                duration=audio_result.metadata.get('duration', 0)
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Audio conversion failed: {str(e)}"
            
            logger.error(
                "Audio conversion failed",
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
                original_format='audio',
                converter_used=self.name
            )
    
    async def _create_structured_markdown(self, audio_result, options: ConversionOptions) -> str:
        """Create structured markdown from audio processing result.
        
        Args:
            audio_result: Audio processing result from MoRAG processor
            options: Conversion options
            
        Returns:
            Structured markdown content
        """
        sections = []
        
        # Document header
        filename = audio_result.metadata.get('filename', 'Audio File')
        title = filename.replace('.mp3', '').replace('.wav', '').replace('.m4a', '').replace('.flac', '')
        sections.append(f"# Audio Transcription: {title}")
        sections.append("")
        
        # Metadata section
        if options.include_metadata:
            sections.append("## Audio Information")
            sections.append("")
            
            duration = audio_result.metadata.get('duration', 0)
            duration_str = f"{duration:.1f} seconds" if duration < 60 else f"{duration/60:.1f} minutes"
            
            metadata_items = [
                ("**Source**", audio_result.metadata.get('filename', 'Unknown')),
                ("**Duration**", duration_str),
                ("**Language**", audio_result.metadata.get('language', 'Auto-detected')),
                ("**Transcription Model**", audio_result.metadata.get('model_used', 'Whisper')),
                ("**Processing Method**", audio_result.metadata.get('processor_used', 'MoRAG Audio Processor'))
            ]
            
            for label, value in metadata_items:
                if value and value != 'Unknown':
                    sections.append(f"{label}: {value}")
            
            sections.append("")
        
        # Summary section (if available)
        if audio_result.summary:
            sections.append("## Summary")
            sections.append("")
            sections.append(audio_result.summary)
            sections.append("")
        
        # Transcript section
        sections.append("## Transcript")
        sections.append("")
        
        if hasattr(audio_result, 'segments') and audio_result.segments:
            # Detailed transcript with timestamps
            for segment in audio_result.segments:
                if options.format_options.get('include_timestamps', True):
                    start_time = self._format_timestamp(segment.get('start', 0))
                    end_time = self._format_timestamp(segment.get('end', 0))
                    sections.append(f"**[{start_time} - {end_time}]**")
                
                text = segment.get('text', '').strip()
                if text:
                    sections.append(text)
                    sections.append("")
        else:
            # Simple transcript
            if hasattr(audio_result, 'transcript') and audio_result.transcript:
                sections.append(audio_result.transcript)
            elif hasattr(audio_result, 'text') and audio_result.text:
                sections.append(audio_result.text)
            else:
                sections.append("*No transcript available*")
            sections.append("")
        
        # Processing details
        sections.append("## Processing Details")
        sections.append("")
        sections.append(f"**Transcription Engine**: {audio_result.metadata.get('model_used', 'Whisper')}")
        
        if 'confidence' in audio_result.metadata:
            confidence = audio_result.metadata['confidence']
            sections.append(f"**Average Confidence**: {confidence:.2f}")
        
        if 'word_count' in audio_result.metadata:
            sections.append(f"**Word Count**: {audio_result.metadata['word_count']}")
        
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
            'conversion_format': 'audio_to_markdown',
            'converter_version': '1.0.0',
            'file_extension': file_path.suffix.lower()
        })
        
        return enhanced
