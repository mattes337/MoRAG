"""Markitdown service wrapper for document conversion."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Union
import structlog

from morag_core.interfaces.converter import ConversionError, UnsupportedFormatError
from morag_core.config import get_settings

logger = structlog.get_logger(__name__)


class MarkitdownService:
    """Service wrapper for Microsoft's markitdown library."""
    
    def __init__(self):
        """Initialize markitdown service."""
        self._markitdown = None
        self._initialized = False
        self.settings = get_settings()
        
    async def _initialize(self) -> None:
        """Initialize markitdown client lazily."""
        if self._initialized:
            return
            
        try:
            # Import markitdown dynamically to handle optional dependency
            from markitdown import MarkItDown
            
            # Initialize markitdown with configuration
            self._markitdown = MarkItDown()
            
            # Configure markitdown options based on settings
            await self._configure_markitdown()
            
            self._initialized = True
            logger.info("Markitdown service initialized successfully")
            
        except ImportError as e:
            raise ConversionError(
                "Markitdown is not installed. Please install with: pip install markitdown"
            ) from e
        except Exception as e:
            raise ConversionError(f"Failed to initialize markitdown service: {e}") from e
    
    async def _configure_markitdown(self) -> None:
        """Configure markitdown based on settings."""
        if not self._markitdown:
            return
            
        # Configure Azure Document Intelligence if enabled
        if hasattr(self.settings, 'markitdown_use_azure_doc_intel') and self.settings.markitdown_use_azure_doc_intel:
            if hasattr(self.settings, 'markitdown_azure_endpoint') and self.settings.markitdown_azure_endpoint:
                logger.info("Configuring markitdown with Azure Document Intelligence")
                # Note: Azure DI configuration will be implemented in Phase 3

        # Configure LLM-based image description if enabled
        if hasattr(self.settings, 'markitdown_use_llm_image_description') and self.settings.markitdown_use_llm_image_description:
            logger.info("Configuring markitdown with LLM image description")
            # Note: LLM image description will be implemented in Phase 3
    
    async def convert_file(
        self, 
        file_path: Union[str, Path], 
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convert file to markdown using markitdown.
        
        Args:
            file_path: Path to file to convert
            options: Optional conversion options
            
        Returns:
            Markdown content as string
            
        Raises:
            ConversionError: If conversion fails
            UnsupportedFormatError: If format is not supported
        """
        await self._initialize()
        
        file_path = Path(file_path)
        options = options or {}
        
        # Validate file exists
        if not file_path.exists():
            raise ConversionError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ConversionError(f"Not a file: {file_path}")
        
        try:
            logger.info("Converting file with markitdown", file_path=str(file_path))
            
            # Run markitdown conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._convert_sync, 
                str(file_path), 
                options
            )
            
            if not result or not result.text_content:
                raise ConversionError(f"Markitdown returned empty content for file: {file_path}")

            # Additional validation for all supported file types
            if not self._validate_conversion_quality(result.text_content, file_path):
                raise ConversionError(f"Conversion failed - output does not appear to be proper markdown: {file_path}")

            logger.info(
                "File converted successfully",
                file_path=str(file_path),
                content_length=len(result.text_content)
            )

            return result.text_content
            
        except Exception as e:
            if isinstance(e, (ConversionError, UnsupportedFormatError)):
                raise
            logger.error("Markitdown conversion failed", file_path=str(file_path), error=str(e))
            raise ConversionError(f"Failed to convert file with markitdown: {e}") from e
    
    def _convert_sync(self, file_path: str, options: Dict[str, Any]):
        """Synchronous conversion method for thread pool execution."""
        try:
            # For markdown files, try to read directly with proper encoding first
            if file_path.lower().endswith(('.md', '.markdown')):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Create a simple result object that mimics markitdown's output
                    class SimpleResult:
                        def __init__(self, text_content):
                            self.text_content = text_content
                    return SimpleResult(content)
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with different encodings
                    for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            class SimpleResult:
                                def __init__(self, text_content):
                                    self.text_content = text_content
                            return SimpleResult(content)
                        except UnicodeDecodeError:
                            continue
                    # If all encodings fail, fall through to markitdown

            return self._markitdown.convert(file_path)
        except Exception as e:
            # Check if it's an unsupported format error
            if "not supported" in str(e).lower() or "unsupported" in str(e).lower():
                raise UnsupportedFormatError(f"Format not supported by markitdown: {e}")
            raise ConversionError(f"Markitdown conversion failed: {e}")
    
    async def get_supported_formats(self) -> list[str]:
        """Get list of formats supported by markitdown.
        
        Returns:
            List of supported file extensions
        """
        await self._initialize()
        
        # Markitdown supports these formats (as of version 0.0.1a2)
        return [
            'pdf', 'docx', 'pptx', 'xlsx', 'xls', 'doc', 'ppt',
            'html', 'htm', 'xml', 'csv', 'json', 'txt', 'md',
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp',
            'mp3', 'wav', 'mp4', 'avi', 'mov', 'wmv', 'flv',
            'zip', 'epub', 'ipynb'
        ]
    
    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported by markitdown.
        
        Args:
            format_type: File format/extension to check
            
        Returns:
            True if format is supported
        """
        supported_formats = await self.get_supported_formats()
        return format_type.lower().lstrip('.') in supported_formats
    
    async def get_conversion_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about what markitdown would extract from a file.
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            Dictionary with conversion information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConversionError(f"File not found: {file_path}")
        
        format_type = file_path.suffix.lower().lstrip('.')
        is_supported = await self.supports_format(format_type)
        
        return {
            'file_path': str(file_path),
            'format': format_type,
            'supported': is_supported,
            'file_size': file_path.stat().st_size,
            'service': 'markitdown'
        }

    def _validate_conversion_quality(self, content: str, file_path: Path) -> bool:
        """Validate that conversion produced proper markdown rather than raw text.

        Args:
            content: Converted content
            file_path: Original file path

        Returns:
            True if conversion appears successful, False otherwise
        """
        if not content or len(content.strip()) < 10:
            return False

        file_ext = file_path.suffix.lower()

        # For certain file types, we expect specific content patterns
        if file_ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx']:
            return self._validate_document_conversion(content, file_ext)
        elif file_ext in ['.html', '.htm']:
            return self._validate_html_conversion(content)
        elif file_ext in ['.csv', '.xlsx', '.xls']:
            return self._validate_data_conversion(content)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
            return self._validate_image_conversion(content)
        elif file_ext in ['.json', '.xml']:
            return self._validate_structured_data_conversion(content)
        elif file_ext in ['.txt', '.md']:
            return self._validate_text_conversion(content)
        else:
            # For other formats, use general validation
            return self._validate_general_conversion(content)

    def _validate_document_conversion(self, content: str, file_ext: str) -> bool:
        """Validate document file conversion (PDF, DOC, PPT, etc.)."""
        import re

        # Check for basic markdown structure
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

        # Check for signs of raw text extraction
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        if not non_empty_lines:
            return False

        # If most lines are very long, it's likely raw text
        long_lines = sum(1 for line in non_empty_lines if len(line) > 200)
        long_line_ratio = long_lines / len(non_empty_lines) if non_empty_lines else 0

        # Check for very few line breaks relative to content length
        line_break_ratio = content.count('\n') / max(1, len(content))

        # Document should have some structure and reasonable formatting
        has_structure = pattern_matches >= 1 or long_line_ratio < 0.7
        has_reasonable_breaks = line_break_ratio > 0.005

        return has_structure and has_reasonable_breaks

    def _validate_html_conversion(self, content: str) -> bool:
        """Validate HTML conversion."""
        # HTML should convert to structured markdown
        import re

        # Should have some markdown structure or at least proper paragraphs
        has_headers = bool(re.search(r'^#+ ', content, re.MULTILINE))
        has_paragraphs = '\n\n' in content
        has_lists = bool(re.search(r'^\* |^\d+\. ', content, re.MULTILINE))

        # Should not contain raw HTML tags (unless intentionally preserved)
        html_tag_ratio = len(re.findall(r'<[^>]+>', content)) / max(1, len(content.split()))

        return (has_headers or has_paragraphs or has_lists) and html_tag_ratio < 0.1

    def _validate_data_conversion(self, content: str) -> bool:
        """Validate data file conversion (CSV, Excel)."""
        # Data files should convert to tables or structured content
        import re

        # Should have table structure or clear data organization
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
        if len(content.strip()) < 5:
            return False

        # Should not be just error messages or empty content
        error_indicators = ['error', 'failed', 'unable', 'cannot', 'not supported']
        content_lower = content.lower()

        if any(indicator in content_lower for indicator in error_indicators):
            return False

        # Should have some readable text
        words = content.split()
        if len(words) < 3:
            return False

        return True

    def _validate_structured_data_conversion(self, content: str) -> bool:
        """Validate structured data conversion (JSON, XML)."""
        # Should convert to readable markdown format
        import re

        # Should have some structure
        has_headers = bool(re.search(r'^#+ ', content, re.MULTILINE))
        has_structure = bool(re.search(r'^\w+:\s+', content, re.MULTILINE))
        has_code_blocks = '```' in content

        return has_headers or has_structure or has_code_blocks

    def _validate_text_conversion(self, content: str) -> bool:
        """Validate text file conversion."""
        # Text files should preserve content reasonably
        return len(content.strip()) > 0

    def _validate_general_conversion(self, content: str) -> bool:
        """General validation for unknown file types."""
        # Basic check - should have some content and not be obviously broken
        if len(content.strip()) < 5:
            return False

        # Should not be just error messages
        error_indicators = ['error', 'failed', 'unable', 'cannot', 'not supported']
        content_lower = content.lower()

        return not any(indicator in content_lower for indicator in error_indicators)
