"""Converter factory for markdown conversion stage."""

import re
from typing import Dict, Any, Union
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

# Import services - these are optional for fallback processing
try:
    from morag_services import MoRAGServices, ContentType
    SERVICES_AVAILABLE = True
except ImportError as e:
    SERVICES_AVAILABLE = False
    # Create placeholder types for when services are not available
    class MoRAGServices:  # type: ignore
        pass
    class ContentType:  # type: ignore
        pass

# Import YouTube processor for fallback when services are not available
try:
    from morag_youtube.service import YouTubeService
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    class YouTubeService:  # type: ignore
        pass


class ConverterFactory:
    """Factory for creating and managing content converters."""
    
    def __init__(self):
        """Initialize converter factory."""
        self.services = None
        self._initialized = False

    async def initialize(self):
        """Initialize converter services."""
        if self._initialized:
            return
            
        try:
            if SERVICES_AVAILABLE:
                self.services = MoRAGServices()
                await self.services.initialize()
                logger.info("Converter services initialized successfully")
            else:
                logger.warning("MoRAG services not available, using fallback converters")
            
            self._initialized = True
            
        except Exception as e:
            logger.error("Failed to initialize converter services", error=str(e))
            # Continue without services - use fallback methods

    def should_use_markitdown(self, file_path: Union[Path, 'URLPath'], content_type) -> bool:
        """Determine if markitdown should be used for conversion."""
        try:
            # Convert content_type to string if it's an enum
            if hasattr(content_type, 'value'):
                content_type_str = content_type.value.lower()
            else:
                content_type_str = str(content_type).lower()
            
            # Get file extension
            if hasattr(file_path, 'suffix'):
                file_ext = file_path.suffix.lower()
            else:
                file_ext = Path(str(file_path)).suffix.lower()
            
            # Markitdown is good for documents and structured data
            document_types = ['document', 'pdf', 'doc', 'docx', 'txt', 'md', 'html']
            structured_types = ['csv', 'xlsx', 'xls', 'json', 'xml']
            image_types = ['image', 'jpg', 'jpeg', 'png', 'gif', 'bmp']
            
            # Use markitdown for documents and structured data
            if any(dt in content_type_str for dt in document_types):
                return True
                
            if any(st in content_type_str for st in structured_types):
                return True
                
            # Use markitdown for images (OCR)
            if any(it in content_type_str for it in image_types):
                return True
            
            # Check by file extension
            markitdown_extensions = {
                '.pdf', '.doc', '.docx', '.txt', '.md', '.html', '.htm',
                '.csv', '.xlsx', '.xls', '.json', '.xml',
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'
            }
            
            if file_ext in markitdown_extensions:
                return True
            
            # Don't use markitdown for media files
            media_types = ['audio', 'video', 'youtube']
            if any(mt in content_type_str for mt in media_types):
                return False
            
            # Don't use markitdown for web content (use scrapers instead)
            if 'web' in content_type_str:
                return False
            
            # Default to markitdown for unknown types
            return True
            
        except Exception as e:
            logger.warning("Error determining markitdown usage", file=str(file_path), error=str(e))
            return True  # Default to markitdown

    def validate_conversion_quality(self, content: str, file_path: Union[Path, 'URLPath']) -> bool:
        """Validate the quality of converted content."""
        try:
            if not content or len(content.strip()) < 10:
                logger.warning("Conversion produced very short content", file=str(file_path))
                return False
            
            # Get file extension for format-specific validation
            if hasattr(file_path, 'suffix'):
                file_ext = file_path.suffix.lower()
            else:
                file_ext = Path(str(file_path)).suffix.lower()
            
            # Validate based on file type
            if file_ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx']:
                return self._validate_document_conversion(content)
            elif file_ext in ['.html', '.htm']:
                return self._validate_html_conversion(content)
            elif file_ext in ['.csv', '.xlsx', '.xls']:
                return self._validate_data_conversion(content)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                return self._validate_image_conversion(content)
            elif file_ext in ['.json', '.xml']:
                return self._validate_structured_data_conversion(content)
            elif file_ext in ['.txt', '.md', '.rst']:
                return self._validate_text_conversion(content)
            elif file_ext in ['.mp3', '.wav', '.mp4', '.avi', '.mov', '.wmv', '.flv']:
                return self._validate_media_conversion(content)
            else:
                return self._validate_general_conversion(content)
                
        except Exception as e:
            logger.warning("Error validating conversion quality", file=str(file_path), error=str(e))
            return True  # Assume valid if validation fails

    def _validate_document_conversion(self, content: str) -> bool:
        """Validate document file conversion (PDF, DOC, PPT, etc.)."""
        # Check for markdown structure
        markdown_patterns = [
            r'^#+ ',  # Headers
            r'^\* ',  # Bullet points
            r'^\d+\. ',  # Numbered lists
            r'\*\*.*?\*\*',  # Bold text
            r'`.*?`',  # Code
            r'^\|.*\|',  # Tables
        ]

        pattern_matches = sum(1 for pattern in markdown_patterns
                            if re.search(pattern, content, re.MULTILINE))

        # Check for signs of poor conversion
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        if not non_empty_lines:
            return False

        # Check for very long lines (sign of poor text extraction)
        long_lines = sum(1 for line in non_empty_lines if len(line) > 200)
        long_line_ratio = long_lines / len(non_empty_lines) if non_empty_lines else 0

        # Check line break density
        line_break_ratio = content.count('\n') / max(1, len(content))

        # Document should have some structure and reasonable formatting
        has_structure = pattern_matches >= 1 or '\n\n' in content
        has_reasonable_formatting = long_line_ratio < 0.7 and line_break_ratio > 0.005

        return has_structure and has_reasonable_formatting

    def _validate_html_conversion(self, content: str) -> bool:
        """Validate HTML conversion."""
        # HTML should convert to structured markdown
        has_headers = bool(re.search(r'^#+ ', content, re.MULTILINE))
        has_paragraphs = '\n\n' in content
        has_lists = bool(re.search(r'^\* |^\d+\. ', content, re.MULTILINE))

        # Should not contain excessive raw HTML tags
        html_tag_ratio = len(re.findall(r'<[^>]+>', content)) / max(1, len(content.split()))

        return (has_headers or has_paragraphs or has_lists) and html_tag_ratio < 0.1

    def _validate_data_conversion(self, content: str) -> bool:
        """Validate data file conversion (CSV, Excel)."""
        # Data files should convert to tables or structured content
        has_tables = bool(re.search(r'^\|.*\|', content, re.MULTILINE))
        has_structured_data = bool(re.search(r'^\w+:\s+\w+', content, re.MULTILINE))
        has_headers = bool(re.search(r'^#+ ', content, re.MULTILINE))

        # Should not be just a wall of text
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        avg_line_length = sum(len(line) for line in non_empty_lines) / max(1, len(non_empty_lines))

        return (has_tables or has_structured_data or has_headers) and avg_line_length < 300

    def _validate_image_conversion(self, content: str) -> bool:
        """Validate image conversion (OCR results)."""
        # Images should produce meaningful text content
        if len(content.strip()) < 3:
            return False

        # Should not be just error messages
        error_indicators = ['error', 'failed', 'unable', 'cannot', 'not supported', 'no text found']
        content_lower = content.lower()

        return not any(indicator in content_lower for indicator in error_indicators)

    def _validate_structured_data_conversion(self, content: str) -> bool:
        """Validate structured data conversion (JSON, XML)."""
        # Should convert to readable markdown format
        has_headers = bool(re.search(r'^#+ ', content, re.MULTILINE))
        has_structure = bool(re.search(r'^\w+:\s+', content, re.MULTILINE))
        has_code_blocks = '```' in content

        return has_headers or has_structure or has_code_blocks

    def _validate_text_conversion(self, content: str) -> bool:
        """Validate text file conversion."""
        # Text files should preserve content reasonably
        return len(content.strip()) > 0

    def _validate_media_conversion(self, content: str) -> bool:
        """Validate media file conversion (audio/video transcription)."""
        # Media files should produce transcribed text
        if len(content.strip()) < 10:
            return False

        # Should not be just error messages
        error_indicators = ['error', 'failed', 'unable', 'cannot', 'not supported', 'no audio', 'no video']
        content_lower = content.lower()

        return not any(indicator in content_lower for indicator in error_indicators)

    def _validate_general_conversion(self, content: str) -> bool:
        """General validation for unknown file types."""
        # Basic check - should have some content and not be obviously broken
        if len(content.strip()) < 5:
            return False

        # Should not be just error messages
        error_indicators = ['error', 'failed', 'unable', 'cannot', 'not supported']
        content_lower = content.lower()

        return not any(indicator in content_lower for indicator in error_indicators)

    def get_converter_type(self, file_path: Union[Path, 'URLPath'], content_type) -> str:
        """Determine which converter type to use."""
        try:
            # Convert content_type to string if it's an enum
            if hasattr(content_type, 'value'):
                content_type_str = content_type.value.lower()
            else:
                content_type_str = str(content_type).lower()
            
            # Determine converter type based on content
            if 'youtube' in content_type_str:
                return 'youtube'
            elif 'video' in content_type_str:
                return 'video'
            elif 'audio' in content_type_str:
                return 'audio'
            elif 'web' in content_type_str:
                return 'web'
            elif 'image' in content_type_str:
                return 'markitdown'
            elif any(doc_type in content_type_str for doc_type in ['document', 'pdf', 'doc', 'txt']):
                return 'markitdown'
            else:
                # Default to markitdown for most content
                return 'markitdown'
                
        except Exception as e:
            logger.warning("Error determining converter type", file=str(file_path), error=str(e))
            return 'markitdown'

    async def get_services(self) -> 'MoRAGServices':
        """Get initialized MoRAG services instance."""
        if not self._initialized:
            await self.initialize()
        return self.services

    def is_services_available(self) -> bool:
        """Check if MoRAG services are available."""
        return SERVICES_AVAILABLE and self.services is not None


__all__ = ["ConverterFactory"]