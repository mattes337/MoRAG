"""Document converter registry and main conversion orchestrator."""

import asyncio
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Union
import structlog

from .base import BaseConverter, ConversionOptions, ConversionResult, ConversionError, UnsupportedFormatError

logger = structlog.get_logger(__name__)


class DocumentConverter:
    """Main document converter that orchestrates format-specific converters."""
    
    def __init__(self):
        self.converters: Dict[str, BaseConverter] = {}
        self.fallback_converters: Dict[str, List[BaseConverter]] = {}
        self._register_default_converters()
    
    def register_converter(self, format_type: str, converter: BaseConverter, is_primary: bool = True):
        """Register a converter for a specific format.
        
        Args:
            format_type: Format this converter handles (e.g., 'pdf', 'docx')
            converter: Converter instance
            is_primary: Whether this is the primary converter for the format
        """
        if is_primary:
            self.converters[format_type] = converter
            logger.info(f"Registered primary converter for {format_type}: {converter.name}")
        else:
            if format_type not in self.fallback_converters:
                self.fallback_converters[format_type] = []
            self.fallback_converters[format_type].append(converter)
            logger.info(f"Registered fallback converter for {format_type}: {converter.name}")
    
    def _register_default_converters(self):
        """Register default converters for common formats."""
        # Import and register existing MoRAG converters
        try:
            from .pdf import PDFConverter
            self.register_converter('pdf', PDFConverter())
            logger.info("Registered PDF converter")
        except ImportError as e:
            logger.warning("PDF converter not available", error=str(e))
        except Exception as e:
            logger.error("Failed to register PDF converter", error=str(e))

        try:
            from .audio import AudioConverter
            self.register_converter('audio', AudioConverter())
            logger.info("Registered Audio converter")
        except ImportError as e:
            logger.warning("Audio converter not available", error=str(e))
        except Exception as e:
            logger.error("Failed to register Audio converter", error=str(e))

        try:
            from .video import VideoConverter
            self.register_converter('video', VideoConverter())
            logger.info("Registered Video converter")
        except ImportError as e:
            logger.warning("Video converter not available", error=str(e))
        except Exception as e:
            logger.error("Failed to register Video converter", error=str(e))

        try:
            from .office import OfficeConverter
            office_converter = OfficeConverter()
            for format_type in ['word', 'excel', 'powerpoint']:
                self.register_converter(format_type, office_converter)
            logger.info("Registered Office converter")
        except ImportError as e:
            logger.warning("Office converter not available", error=str(e))
        except Exception as e:
            logger.error("Failed to register Office converter", error=str(e))

        try:
            from .web import WebConverter
            self.register_converter('web', WebConverter())
            logger.info("Registered Web converter")
        except ImportError as e:
            logger.warning("Web converter not available", error=str(e))
        except Exception as e:
            logger.error("Failed to register Web converter", error=str(e))
    
    def detect_format(self, file_path: Union[str, Path]) -> str:
        """Detect document format using multiple methods.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detected format type
        """
        path = Path(file_path)
        
        # Method 1: File extension
        extension = path.suffix.lower().lstrip('.')
        format_mapping = {
            'pdf': 'pdf',
            'doc': 'word', 'docx': 'word',
            'xls': 'excel', 'xlsx': 'excel',
            'ppt': 'powerpoint', 'pptx': 'powerpoint',
            'mp3': 'audio', 'wav': 'audio', 'm4a': 'audio', 'flac': 'audio',
            'mp4': 'video', 'avi': 'video', 'mov': 'video', 'mkv': 'video',
            'html': 'web', 'htm': 'web',
            'txt': 'text', 'md': 'markdown', 'markdown': 'markdown'
        }
        
        if extension in format_mapping:
            return format_mapping[extension]
        
        # Method 2: MIME type detection
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            mime_mapping = {
                'application/pdf': 'pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'word',
                'application/msword': 'word',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'excel',
                'application/vnd.ms-excel': 'excel',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'powerpoint',
                'application/vnd.ms-powerpoint': 'powerpoint',
                'audio/mpeg': 'audio',
                'audio/wav': 'audio',
                'video/mp4': 'video',
                'video/avi': 'video',
                'text/html': 'web',
                'text/plain': 'text',
                'text/markdown': 'markdown'
            }
            
            if mime_type in mime_mapping:
                return mime_mapping[mime_type]
        
        # Default to extension without dot
        return extension
    
    def get_converter(self, format_type: str) -> Optional[BaseConverter]:
        """Get the primary converter for a format.
        
        Args:
            format_type: Format to get converter for
            
        Returns:
            Converter instance or None if not found
        """
        return self.converters.get(format_type)
    
    def get_fallback_converters(self, format_type: str) -> List[BaseConverter]:
        """Get fallback converters for a format.
        
        Args:
            format_type: Format to get fallback converters for
            
        Returns:
            List of fallback converter instances
        """
        return self.fallback_converters.get(format_type, [])
    
    async def convert_to_markdown(
        self, 
        file_path: Union[str, Path], 
        options: Optional[ConversionOptions] = None
    ) -> ConversionResult:
        """Convert document to markdown using appropriate converter.
        
        Args:
            file_path: Path to document to convert
            options: Conversion options
            
        Returns:
            ConversionResult with converted content
            
        Raises:
            UnsupportedFormatError: If no converter available for format
            ConversionError: If conversion fails
        """
        if options is None:
            options = ConversionOptions()
        
        path = Path(file_path)
        format_type = self.detect_format(path)
        
        logger.info(
            "Starting document conversion",
            file_path=str(path),
            format_type=format_type,
            chunking_strategy=options.chunking_strategy.value
        )
        
        # Get primary converter
        converter = self.get_converter(format_type)
        if not converter:
            raise UnsupportedFormatError(f"No converter available for format: {format_type}")
        
        # Try primary converter
        try:
            result = await converter.convert(path, options)
            result.original_format = format_type
            result.converter_used = converter.name
            
            # Check quality and try fallback if needed
            if options.enable_fallback and result.quality_score:
                if result.quality_score.overall_score < options.min_quality_threshold:
                    logger.warning(
                        "Primary converter quality below threshold, trying fallback",
                        quality_score=result.quality_score.overall_score,
                        threshold=options.min_quality_threshold
                    )
                    
                    fallback_result = await self._try_fallback_converters(path, format_type, options)
                    if fallback_result and fallback_result.quality_score:
                        if fallback_result.quality_score.overall_score > result.quality_score.overall_score:
                            logger.info("Fallback converter produced better result")
                            fallback_result.fallback_used = True
                            return fallback_result
            
            logger.info(
                "Document conversion completed",
                converter=converter.name,
                quality_score=result.quality_score.overall_score if result.quality_score else None,
                word_count=result.word_count,
                processing_time=result.processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Primary converter failed",
                converter=converter.name,
                error=str(e),
                error_type=type(e).__name__
            )
            
            if options.enable_fallback:
                fallback_result = await self._try_fallback_converters(path, format_type, options)
                if fallback_result:
                    fallback_result.fallback_used = True
                    return fallback_result
            
            raise ConversionError(f"Conversion failed: {str(e)}")
    
    async def _try_fallback_converters(
        self, 
        file_path: Path, 
        format_type: str, 
        options: ConversionOptions
    ) -> Optional[ConversionResult]:
        """Try fallback converters for a format.
        
        Args:
            file_path: Path to document
            format_type: Document format
            options: Conversion options
            
        Returns:
            ConversionResult if successful, None otherwise
        """
        fallback_converters = self.get_fallback_converters(format_type)
        
        for converter in fallback_converters:
            try:
                logger.info(f"Trying fallback converter: {converter.name}")
                result = await converter.convert(file_path, options)
                result.original_format = format_type
                result.converter_used = converter.name
                
                logger.info(
                    "Fallback converter succeeded",
                    converter=converter.name,
                    quality_score=result.quality_score.overall_score if result.quality_score else None
                )
                
                return result
                
            except Exception as e:
                logger.warning(
                    "Fallback converter failed",
                    converter=converter.name,
                    error=str(e)
                )
                continue
        
        logger.error("All fallback converters failed")
        return None
    
    def list_supported_formats(self) -> List[str]:
        """Get list of all supported formats.
        
        Returns:
            List of supported format types
        """
        formats = set(self.converters.keys())
        formats.update(self.fallback_converters.keys())
        return sorted(list(formats))
    
    def get_converter_info(self) -> Dict[str, Dict[str, any]]:
        """Get information about registered converters.
        
        Returns:
            Dictionary with converter information
        """
        info = {}
        
        for format_type, converter in self.converters.items():
            info[format_type] = {
                'primary_converter': converter.name,
                'fallback_converters': [c.name for c in self.get_fallback_converters(format_type)]
            }
        
        return info


# Global instance
document_converter = DocumentConverter()
