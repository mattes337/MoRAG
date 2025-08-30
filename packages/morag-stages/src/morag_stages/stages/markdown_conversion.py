"""Markdown conversion stage implementation."""

import re
from datetime import datetime
from typing import List, Dict, Any, Union
from pathlib import Path
import structlog

from ..models import Stage, StageType, StageStatus, StageResult, StageContext, StageMetadata
from ..exceptions import StageExecutionError, StageValidationError
from ..processors import ProcessorRegistry
from ..utils import detect_content_type, is_content_type

logger = structlog.get_logger(__name__)

# Import core exceptions
try:
    from morag_core.exceptions import ProcessingError
except ImportError:
    class ProcessingError(Exception):  # type: ignore
        pass

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


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize filename for safe filesystem usage.

    Args:
        filename: Original filename
        max_length: Maximum length for filename

    Returns:
        Sanitized filename
    """
    if not filename:
        return "unnamed"

    # Remove or replace invalid characters for Windows/Unix
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f%]'
    sanitized = re.sub(invalid_chars, '_', filename)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')

    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)

    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed"

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')

    return sanitized


class MarkdownConversionStage(Stage):
    """Stage that converts input files to unified markdown format."""

    def __init__(self, stage_type: StageType = StageType.MARKDOWN_CONVERSION):
        """Initialize markdown conversion stage."""
        super().__init__(stage_type)

        if not SERVICES_AVAILABLE:
            logger.warning("MoRAG services not available for markdown conversion")
            self.services = None
        else:
            self.services = MoRAGServices()

        # Initialize YouTube service for fallback processing
        if YOUTUBE_AVAILABLE:
            self.youtube_service = YouTubeService()
        else:
            self.youtube_service = None

        # Initialize processor registry for delegation
        self.processor_registry = ProcessorRegistry()
    
    async def execute(self, 
                     input_files: List[Path], 
                     context: StageContext) -> StageResult:
        """Execute markdown conversion on input files.
        
        Args:
            input_files: List of input file paths
            context: Stage execution context
            
        Returns:
            Stage execution result
        """
        if len(input_files) != 1:
            raise StageValidationError(
                "Markdown conversion stage requires exactly one input file",
                stage_type=self.stage_type.value,
                invalid_files=[str(f) for f in input_files]
            )

        input_file = input_files[0]
        config = context.get_stage_config(self.stage_type)

        # Determine content type using utility
        content_type = detect_content_type(input_file)

        # Check if we have a processor for this content type
        content_type_str = str(content_type).upper() if content_type else "TEXT"
        if not self._should_use_markitdown(input_file, content_type) and not self.processor_registry.supports_content_type(content_type_str):
            # Fallback to services if no processor available
            if not SERVICES_AVAILABLE or self.services is None:
                raise StageExecutionError(
                    f"No processor available for content type {content_type_str} and MoRAG services not available",
                    stage_type=self.stage_type.value
                )

        logger.info("Starting markdown conversion",
                   input_file=str(input_file),
                   config=config)

        try:
            
            # Generate output filename
            output_filename = self._generate_output_filename(input_file, content_type)
            output_file = context.output_dir / output_filename

            context.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if we should use MarkItDown for better quality
            if self._should_use_markitdown(input_file, content_type):
                logger.info("Using MarkItDown for high-quality conversion",
                           input_file=str(input_file),
                           content_type=str(content_type))
                result_data = await self._process_with_markitdown(input_file, output_file, config)
            else:
                # Use processor delegation for specialized processing
                result_data = await self._delegate_processing(input_file, output_file, content_type, config)

            # Check if the output file was renamed (e.g., for YouTube videos with titles)
            final_output_file = result_data.get("final_output_file", output_file)

            # Create metadata
            metadata = StageMetadata(
                execution_time=0.0,  # Will be set by manager
                start_time=datetime.now(),
                input_files=[str(input_file)],
                output_files=[str(final_output_file)],
                config_used=config,
                metrics={
                    "content_type": str(content_type) if content_type else "unknown",
                    "input_size_bytes": input_file.stat().st_size if input_file.exists() else 0,
                    "output_size_bytes": final_output_file.stat().st_size if final_output_file.exists() else 0,
                    **result_data.get("metrics", {})
                }
            )

            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[final_output_file],
                metadata=metadata,
                data=result_data
            )
            
        except Exception as e:
            logger.error("Markdown conversion failed", 
                        input_file=str(input_file), 
                        error=str(e))
            raise StageExecutionError(
                f"Markdown conversion failed: {e}",
                stage_type=self.stage_type.value,
                original_error=e
            )
    
    def validate_inputs(self, input_files: List[Path]) -> bool:
        """Validate input files for markdown conversion.

        Args:
            input_files: List of input file paths

        Returns:
            True if inputs are valid
        """
        logger.debug("Validating markdown conversion inputs",
                    input_count=len(input_files),
                    input_files=[str(f) for f in input_files])

        if len(input_files) != 1:
            logger.error("Invalid input count for markdown conversion",
                        expected=1,
                        actual=len(input_files),
                        files=[str(f) for f in input_files])
            return False

        input_file = input_files[0]
        file_str = str(input_file)

        logger.debug("Validating input file",
                    file_path=file_str,
                    file_type=type(input_file).__name__)

        # Check if file exists (for local files)
        # Handle URLs that may have been converted to Windows paths
        is_url = (
            file_str.startswith(('http://', 'https://')) or
            file_str.replace('\\', '/').startswith(('http://', 'https://')) or
            ('http:' in file_str and ('www.' in file_str or '.com' in file_str or '.org' in file_str or '.net' in file_str)) or
            ('https:' in file_str and ('www.' in file_str or '.com' in file_str or '.org' in file_str or '.net' in file_str))
        )

        logger.debug("URL detection result",
                    file_path=file_str,
                    is_url=is_url)

        if not is_url:
            file_exists = input_file.exists()
            logger.debug("Local file existence check",
                        file_path=file_str,
                        exists=file_exists)
            if not file_exists:
                logger.error("Local file does not exist", file_path=file_str)
                return False

        # Check if file type is supported
        content_type = detect_content_type(input_file)
        logger.debug("Content type detection result",
                    file_path=file_str,
                    content_type=str(content_type) if content_type else None,
                    is_supported=content_type is not None)

        if content_type is None:
            logger.error("Unsupported content type", file_path=file_str)
            return False

        logger.debug("Input validation successful",
                    file_path=file_str,
                    content_type=content_type.value)
        return True

    async def _delegate_processing(
        self,
        input_file: Path,
        output_file: Path,
        content_type: Union[str, object],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delegate processing to appropriate processor.

        Args:
            input_file: Input file path
            output_file: Output file path
            content_type: Content type
            config: Processing configuration

        Returns:
            Processing result data
        """
        content_type_str = str(content_type).upper() if content_type else "TEXT"

        # Try to get processor from registry
        processor = self.processor_registry.get_processor(content_type_str)

        if processor:
            logger.info("Using specialized processor",
                       content_type=content_type_str,
                       processor=type(processor).__name__)

            try:
                result = await processor.process(input_file, output_file, config)

                return {
                    "content_type": content_type_str.lower(),
                    "title": result.metadata.get('title', input_file.stem),
                    "metadata": result.metadata,
                    "final_output_file": result.final_output_file or output_file,
                    "metrics": result.metrics
                }

            except Exception as e:
                logger.warning("Specialized processor failed, falling back to services",
                             content_type=content_type_str, error=str(e))

        # Fallback to legacy processing methods
        if is_content_type(content_type, "VIDEO"):
            return await self._process_video(input_file, output_file, config)
        elif is_content_type(content_type, "AUDIO"):
            return await self._process_audio(input_file, output_file, config)
        elif is_content_type(content_type, "DOCUMENT"):
            return await self._process_document(input_file, output_file, config)
        elif is_content_type(content_type, "WEB"):
            return await self._process_web(input_file, output_file, config)
        elif is_content_type(content_type, "YOUTUBE"):
            return await self._process_youtube(input_file, output_file, config)
        else:
            return await self._process_text(input_file, output_file, config)

    def get_dependencies(self) -> List[StageType]:
        """Get stage dependencies.
        
        Returns:
            Empty list - this is the first stage
        """
        return []
    
    def get_expected_outputs(self, input_files: List[Path], context: StageContext) -> List[Path]:
        """Get expected output file paths.

        Args:
            input_files: List of input file paths
            context: Stage execution context

        Returns:
            List of expected output file paths
        """
        if len(input_files) != 1:
            return []

        input_file = input_files[0]
        content_type = detect_content_type(input_file)
        output_filename = self._generate_output_filename(input_file, content_type)
        output_file = context.output_dir / output_filename
        return [output_file]
    


    def _is_content_type(self, content_type, expected_type: str) -> bool:
        """Check if content type matches expected type, handling both enum and string types.

        Args:
            content_type: Content type (ContentType enum or string)
            expected_type: Expected type as string (e.g., "VIDEO", "AUDIO")

        Returns:
            True if content type matches expected type
        """
        return is_content_type(content_type, expected_type)

    def _generate_output_filename(self, input_file: Path, content_type, metadata: Dict[str, Any] = None) -> str:
        """Generate appropriate output filename based on input type.

        Args:
            input_file: Input file path or URL
            content_type: Detected content type
            metadata: Optional metadata containing title information

        Returns:
            Output filename with .md extension
        """

        if self._is_content_type(content_type, "YOUTUBE"):
            # Try to use video title from metadata first
            if metadata and metadata.get('title'):
                title = metadata['title']
                # Sanitize title for filename
                sanitized_title = sanitize_filename(title, max_length=100)
                return f"{sanitized_title}.md"

            # Try to extract metadata to get title for filename
            try:
                # Fix URL mangling from Windows path conversion
                url_str = str(input_file)
                # Convert backslashes to forward slashes
                url_str = url_str.replace('\\', '/')
                # Fix common URL mangling patterns
                if url_str.startswith('https:') and not url_str.startswith('https://'):
                    url_str = url_str.replace('https:/', 'https://')
                elif url_str.startswith('http:') and not url_str.startswith('http://'):
                    url_str = url_str.replace('http:/', 'http://')

                # Try to get title from YouTube service if available
                # Note: We skip async metadata extraction here to avoid event loop issues
                # The filename will be updated later during processing if needed

                if YOUTUBE_AVAILABLE and self.youtube_service:
                    # Fallback to video ID if metadata extraction fails
                    video_id = self.youtube_service.transcript_service.extract_video_id(url_str)
                    return f"{video_id}.md"
                else:
                    # Fallback: extract video ID using regex
                    patterns = [
                        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
                        r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
                        r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, url_str)
                        if match:
                            video_id = match.group(1)
                            return f"{video_id}.md"

                    # If no pattern matches, fall back to sanitized stem
                    sanitized_name = sanitize_filename(input_file.stem)
                    return f"{sanitized_name}.md"
            except Exception:
                # If video ID extraction fails, fall back to sanitized stem
                sanitized_name = sanitize_filename(input_file.stem)
                return f"{sanitized_name}.md"
        elif self._is_content_type(content_type, "WEB"):
            # Try to use website title from metadata first
            if metadata and metadata.get('title'):
                title = metadata['title']
                # Sanitize title for filename
                sanitized_title = sanitize_filename(title, max_length=100)
                return f"{sanitized_title}.md"

            # Fallback to domain name from URL
            try:
                # Fix URL mangling from Windows path conversion
                url_str = str(input_file)
                # Convert backslashes to forward slashes
                url_str = url_str.replace('\\', '/')
                # Fix common URL mangling patterns
                if url_str.startswith('https:') and not url_str.startswith('https://'):
                    url_str = url_str.replace('https:/', 'https://')
                elif url_str.startswith('http:') and not url_str.startswith('http://'):
                    url_str = url_str.replace('http:/', 'http://')

                # Extract domain name for fallback filename
                from urllib.parse import urlparse
                parsed = urlparse(url_str)
                domain = parsed.netloc or parsed.path.split('/')[0]
                if domain:
                    # Remove www. prefix and sanitize
                    domain = domain.replace('www.', '')
                    sanitized_domain = sanitize_filename(domain, max_length=50)
                    return f"{sanitized_domain}.md"
                else:
                    # If domain extraction fails, fall back to sanitized stem
                    sanitized_name = sanitize_filename(input_file.stem)
                    return f"{sanitized_name}.md"
            except Exception:
                # If URL parsing fails, fall back to sanitized stem
                sanitized_name = sanitize_filename(input_file.stem)
                return f"{sanitized_name}.md"
        else:
            # For other content types, use sanitized stem
            sanitized_name = sanitize_filename(input_file.stem)
            return f"{sanitized_name}.md"

    def _should_use_markitdown(self, file_path: Path, content_type) -> bool:
        """Check if MarkItDown should be used for this file type.

        Args:
            file_path: File path to check
            content_type: Detected content type (ContentType enum or string)

        Returns:
            True if MarkItDown should be used
        """
        # MarkItDown supported extensions (based on documentation)
        markitdown_extensions = {
            # Document formats
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            # Text formats
            '.html', '.htm', '.txt', '.md', '.rst',
            # Data formats
            '.json', '.xml', '.csv',
            # Image formats (with LLM support)
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'
        }

        file_ext = file_path.suffix.lower()

        # Use MarkItDown for supported file extensions
        if file_ext in markitdown_extensions:
            return True

        # Handle both ContentType enums and string content types
        content_type_str = str(content_type).upper() if content_type else ""

        # Don't use MarkItDown for audio/video files (they need specialized processing)
        if SERVICES_AVAILABLE:
            if content_type in [ContentType.AUDIO, ContentType.VIDEO]:
                return False
        else:
            if content_type_str in ["AUDIO", "VIDEO"]:
                return False

        # Don't use MarkItDown for web URLs (we have custom web processing)
        if SERVICES_AVAILABLE:
            if content_type == ContentType.WEB:
                return False
        else:
            if content_type_str == "WEB":
                return False

        return False

    def _validate_conversion_quality(self, content: str, file_path: Path) -> bool:
        """Validate that conversion produced proper markdown for any supported file type.

        Args:
            content: Converted content
            file_path: Original file path

        Returns:
            True if conversion appears successful, False otherwise
        """
        if not content or len(content.strip()) < 5:
            return False

        file_ext = file_path.suffix.lower()

        # Format-specific validation
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

    async def _process_with_markitdown(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process file using MarkItDown for high-quality conversion.

        Args:
            input_file: Input file path
            output_file: Output markdown file path
            config: Stage configuration

        Returns:
            Processing result data
        """
        try:
            from markitdown import MarkItDown

            logger.info("Processing with MarkItDown", input_file=str(input_file))

            # Initialize MarkItDown with optional LLM support for images
            md_converter = MarkItDown()

            # Convert file to markdown
            result = md_converter.convert(str(input_file))

            if not result or not result.text_content:
                raise ProcessingError(f"MarkItDown failed to extract content from {input_file}")

            # Quality validation for all supported file types
            logger.debug("Validating conversion quality",
                       file_type=input_file.suffix.lower(),
                       content_preview=result.text_content[:200] + "..." if len(result.text_content) > 200 else result.text_content)
            if not self._validate_conversion_quality(result.text_content, input_file):
                logger.error("Conversion quality validation failed",
                           file_type=input_file.suffix.lower(),
                           content_preview=result.text_content[:200] + "..." if len(result.text_content) > 200 else result.text_content)
                raise ProcessingError(f"Conversion failed for {input_file.suffix.lower()} - output does not appear to be proper markdown: {input_file}")

            # Create markdown with metadata header
            markdown_content = self._create_markdown_with_metadata(
                content=result.text_content,
                metadata={
                    "title": getattr(result, 'title', None) or input_file.stem,
                    "source": str(input_file),
                    "type": "document",
                    "file_extension": input_file.suffix,
                    "processed_with": "markitdown",
                    "created_at": datetime.now().isoformat(),
                    # Add any additional metadata from MarkItDown result
                    **(getattr(result, 'metadata', {}) or {})
                }
            )

            # Write to file
            output_file.write_text(markdown_content, encoding='utf-8')

            return {
                "content_type": "document",
                "title": getattr(result, 'title', None) or input_file.stem,
                "metadata": getattr(result, 'metadata', {}) or {},
                "metrics": {
                    "file_path": str(input_file),
                    "file_size": input_file.stat().st_size,
                    "content_length": len(result.text_content),
                    "processor": "markitdown"
                }
            }

        except Exception as e:
            error_msg = str(e).lower()

            # Check for specific corruption indicators
            if any(indicator in error_msg for indicator in [
                "data-loss while decompressing corrupted data",
                "corrupted data",
                "decompression failed",
                "invalid pdf",
                "pdf parsing error"
            ]):
                logger.warning("PDF appears corrupted, attempting fallback processing",
                             input_file=str(input_file), error=str(e))

                # Try fallback to document service if available
                if self.services:
                    try:
                        logger.info("Attempting fallback to document service", input_file=str(input_file))
                        return await self._process_with_document_service(input_file, output_file, config)
                    except Exception as fallback_error:
                        logger.error("Fallback processing also failed",
                                   input_file=str(input_file),
                                   fallback_error=str(fallback_error))

                # If no fallback available or fallback failed, provide helpful error
                raise ProcessingError(
                    f"PDF file appears to be corrupted or damaged: {input_file}. "
                    f"Please try with a different PDF file or check if the file is valid. "
                    f"Original error: {e}"
                )

            logger.error(f"MarkItDown processing failed: {e}", input_file=str(input_file))
            raise ProcessingError(f"MarkItDown processing failed for {input_file}: {e}")
    
    async def _process_video(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process video file to markdown.
        
        Args:
            input_file: Input video file
            output_file: Output markdown file
            config: Stage configuration
            
        Returns:
            Processing result data
        """
        logger.info("Processing video file", input_file=str(input_file))
        
        # Prepare options for video service
        options = {
            'include_timestamps': config.get('include_timestamps', True),
            'speaker_diarization': config.get('speaker_diarization', True),
            'topic_segmentation': config.get('topic_segmentation', True),
            'extract_thumbnails': config.get('extract_thumbnails', False)
        }

        # Use video service
        if not self.services:
            raise ProcessingError("MoRAG services not available")
        result = await self.services.process_video(str(input_file), options)
        
        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=result.text_content or "",
            metadata={
                "title": result.metadata.get('title') or input_file.stem,
                "source": str(input_file),
                "type": "video",
                "duration": result.metadata.get('duration'),
                "language": result.metadata.get('language'),
                "created_at": datetime.now().isoformat(),
                **result.metadata
            }
        )

        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')

        return {
            "content_type": "video",
            "title": result.metadata.get('title'),
            "metadata": result.metadata,
            "metrics": {
                "duration": result.metadata.get('duration', 0),
                "transcript_length": len(result.text_content or ""),
                "has_timestamps": config.get('include_timestamps', True)
            }
        }

    async def _process_audio(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio file to markdown.

        Args:
            input_file: Input audio file
            output_file: Output markdown file
            config: Stage configuration

        Returns:
            Processing result data
        """
        logger.info("Processing audio file", input_file=str(input_file))

        # Prepare options for audio service
        options = {
            'include_timestamps': config.get('include_timestamps', True),
            'speaker_diarization': config.get('speaker_diarization', True),
            'topic_segmentation': config.get('topic_segmentation', True)
        }

        # Use audio service
        if not self.services:
            raise ProcessingError("MoRAG services not available")
        result = await self.services.process_audio(str(input_file), options)

        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=result.text_content or "",
            metadata={
                "title": result.metadata.get('title') or input_file.stem,
                "source": str(input_file),
                "type": "audio",
                "duration": result.metadata.get('duration'),
                "language": result.metadata.get('language'),
                "created_at": datetime.now().isoformat(),
                **result.metadata
            }
        )

        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')

        return {
            "content_type": "audio",
            "title": result.metadata.get('title'),
            "metadata": result.metadata,
            "metrics": {
                "duration": result.metadata.get('duration', 0),
                "transcript_length": len(result.text_content or ""),
                "has_timestamps": config.get('include_timestamps', True)
            }
        }

    async def _process_document(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process document file to markdown.

        Args:
            input_file: Input document file
            output_file: Output markdown file
            config: Stage configuration

        Returns:
            Processing result data
        """
        logger.info("Processing document file", input_file=str(input_file))

        # Prepare options for document service
        options = {
            'preserve_formatting': config.get('preserve_formatting', True),
            'extract_tables': config.get('extract_tables', True),
            'extract_images': config.get('extract_images', False),
            'ocr_enabled': config.get('ocr_enabled', True)
        }

        # Use document service
        if not self.services:
            raise ProcessingError("MoRAG services not available")
        result = await self.services.process_document(str(input_file), options)

        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=result.text_content or "",
            metadata={
                "title": result.metadata.get('title') or input_file.stem,
                "source": str(input_file),
                "type": "document",
                "pages": result.metadata.get('pages'),
                "language": result.metadata.get('language'),
                "created_at": datetime.now().isoformat(),
                **result.metadata
            }
        )

        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')

        return {
            "content_type": "document",
            "title": result.metadata.get('title'),
            "metadata": result.metadata,
            "metrics": {
                "pages": result.metadata.get('pages', 0),
                "content_length": len(result.text_content or ""),
                "has_tables": config.get('extract_tables', True)
            }
        }

    async def _process_with_document_service(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing using document service when MarkItDown fails.

        Args:
            input_file: Input file path
            output_file: Output file path
            config: Processing configuration

        Returns:
            Processing result data
        """
        logger.info("Processing with document service fallback", input_file=str(input_file))

        # Use the existing document processing method
        return await self._process_document(input_file, output_file, config)

    async def _process_web(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process web URL to markdown.

        Args:
            input_file: Input URL (as Path object)
            output_file: Output markdown file
            config: Stage configuration

        Returns:
            Processing result data
        """
        # Convert Path back to URL string and fix Windows path conversion issues
        url = str(input_file)

        # Handle Windows path conversion issue - Path() mangles URLs
        if not url.startswith(('http://', 'https://')):
            # Convert backslashes to forward slashes
            url = url.replace('\\', '/')

            # Fix common URL mangling patterns
            if url.startswith('https:') and not url.startswith('https://'):
                # Pattern: https:/www.example.com -> https://www.example.com
                url = url.replace('https:/', 'https://')
            elif url.startswith('http:') and not url.startswith('http://'):
                # Pattern: http:/www.example.com -> http://www.example.com
                url = url.replace('http:/', 'http://')

            # Handle case where the URL got completely mangled
            if ('www.' in url or '.com' in url or '.org' in url or '.net' in url) and not url.startswith(('http://', 'https://')):
                # Try to reconstruct from fragments - default to https
                if not url.startswith(('http', 'www')):
                    url = 'https://' + url
                elif url.startswith('www'):
                    url = 'https://' + url

        logger.info("Processing web URL", url=url)

        # Use web service
        if not self.services:
            raise ProcessingError("MoRAG services not available")
        result = await self.services.process_url(
            url,
            {
                'follow_links': config.get('follow_links', False),
                'max_depth': config.get('max_depth', 1)
            }
        )

        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=result.text_content or "",
            metadata={
                "title": result.metadata.get('title', "Web Content"),
                "source": url,
                "type": "web",
                "url": url,
                "language": result.metadata.get('language'),
                "created_at": datetime.now().isoformat(),
                **result.metadata
            }
        )

        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')

        # Try to rename the output file to use the website title if available
        final_output_file = output_file
        if result.metadata.get('title') and result.metadata.get('title') != "Web Content":
            try:
                title = result.metadata['title']
                sanitized_title = sanitize_filename(title, max_length=100)
                new_filename = f"{sanitized_title}.md"
                new_output_file = output_file.parent / new_filename

                # Only rename if the new filename is different and doesn't already exist
                if new_output_file != output_file and not new_output_file.exists():
                    output_file.rename(new_output_file)
                    final_output_file = new_output_file
                    logger.info("Renamed output file to use website title",
                               old_name=output_file.name,
                               new_name=new_output_file.name)
            except Exception as e:
                logger.warning("Failed to rename output file with website title",
                              error=str(e),
                              title=result.metadata.get('title'))
                # Continue with original filename

        return {
            "content_type": "web",
            "title": result.metadata.get('title', "Web Content"),
            "metadata": result.metadata,
            "final_output_file": final_output_file,  # Include the final file path
            "metrics": {
                "url": url,
                "content_length": len(result.text_content or ""),
                "links_followed": config.get('follow_links', False)
            }
        }

    async def _process_youtube(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process YouTube URL to markdown.

        Args:
            input_file: Input URL (as Path object)
            output_file: Output markdown file
            config: Stage configuration

        Returns:
            Processing result data
        """
        # Convert Path back to URL string and fix Windows path conversion issues
        url = str(input_file)

        # Handle Windows path conversion issue - Path() mangles URLs
        if not url.startswith(('http://', 'https://')):
            # Convert backslashes to forward slashes
            url = url.replace('\\', '/')

            # Fix common URL mangling patterns
            if url.startswith('https:') and not url.startswith('https://'):
                # Pattern: https:/www.youtube.com -> https://www.youtube.com
                url = url.replace('https:/', 'https://')
            elif url.startswith('http:') and not url.startswith('http://'):
                # Pattern: http:/www.youtube.com -> http://www.youtube.com
                url = url.replace('http:/', 'http://')

            # Handle case where the URL got completely mangled
            if 'youtube.com' in url and not url.startswith(('http://', 'https://')):
                # Try to reconstruct from fragments
                if 'https' in url:
                    url = 'https://www.youtube.com' + url.split('youtube.com')[-1]
                else:
                    url = 'https://www.youtube.com' + url.split('youtube.com')[-1]

        logger.info("Processing YouTube URL", url=url)

        # Try MoRAG services first, fallback to YouTube service if not available
        if self.services:
            # Configure YouTube processing options from stage config
            youtube_options = {
                'transcript_only': config.get('transcript_only', False),  # Default to full processing
                'transcript_language': config.get('transcript_language', None),
                'extract_transcript': True,
                'extract_metadata_only': False,
                'extract_audio': not config.get('transcript_only', False),  # Only extract audio if not transcript-only
                'download_subtitles': False,
                'download_thumbnails': False,
                'quality': 'worst'  # Use lowest quality for faster download if needed
            }

            result = await self.services.process_youtube(url, youtube_options)
        elif self.youtube_service:
            # Fallback to direct YouTube service
            logger.info("Using fallback YouTube service for transcript extraction")

            # Use transcript-only mode for fallback to avoid downloading video
            transcript_result = await self.youtube_service.extract_transcript(
                url=url,
                language=config.get('transcript_language', None),
                transcript_only=True
            )

            # Create a result object compatible with the rest of the method
            class FallbackResult:
                def __init__(self, transcript_data):
                    self.text_content = transcript_data.get('transcript_text', '')
                    self.metadata = {
                        'title': f"YouTube Video {transcript_data.get('video_id', '')}",
                        'video_id': transcript_data.get('video_id'),
                        'language': transcript_data.get('language'),
                        'duration': transcript_data.get('duration', 0),
                        'uploader': 'Unknown',
                        'method': 'fallback_transcript_only'
                    }

            result = FallbackResult(transcript_result)
        else:
            raise ProcessingError("Neither MoRAG services nor YouTube fallback service are available")

        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=result.text_content or "",
            metadata={
                "title": result.metadata.get('title', "YouTube Video"),
                "source": url,
                "type": "youtube",
                "url": url,
                "video_id": result.metadata.get('video_id'),
                "uploader": result.metadata.get('uploader'),
                "duration": result.metadata.get('duration'),
                "language": result.metadata.get('language'),
                "created_at": datetime.now().isoformat(),
                **result.metadata
            }
        )

        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')

        # Try to rename the output file to use the video title if available
        final_output_file = output_file
        if result.metadata.get('title'):
            try:
                title = result.metadata['title']
                sanitized_title = sanitize_filename(title, max_length=100)
                new_filename = f"{sanitized_title}.md"
                new_output_file = output_file.parent / new_filename

                # Only rename if the new filename is different and doesn't already exist
                if new_output_file != output_file and not new_output_file.exists():
                    output_file.rename(new_output_file)
                    final_output_file = new_output_file
                    logger.info("Renamed output file to use video title",
                               old_name=output_file.name,
                               new_name=new_output_file.name)
            except Exception as e:
                logger.warning("Failed to rename output file with video title",
                              error=str(e),
                              title=result.metadata.get('title'))
                # Continue with original filename

        return {
            "content_type": "youtube",
            "title": result.metadata.get('title', "YouTube Video"),
            "metadata": result.metadata,
            "final_output_file": final_output_file,  # Include the final file path
            "metrics": {
                "url": url,
                "video_id": result.metadata.get('video_id'),
                "content_length": len(result.text_content or ""),
                "transcript_only": config.get('transcript_only', True)
            }
        }

    async def _process_text(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process text file to markdown.

        Args:
            input_file: Input text file
            output_file: Output markdown file
            config: Stage configuration

        Returns:
            Processing result data
        """
        logger.info("Processing text file", input_file=str(input_file))

        # Read text content
        content = input_file.read_text(encoding='utf-8')

        # Create markdown with metadata header
        markdown_content = self._create_markdown_with_metadata(
            content=content,
            metadata={
                "title": input_file.stem,
                "source": str(input_file),
                "type": "text",
                "language": "en",  # Default language
                "created_at": datetime.now().isoformat()
            }
        )

        # Write to file
        output_file.write_text(markdown_content, encoding='utf-8')

        return {
            "content_type": "text",
            "title": input_file.stem,
            "metadata": {},
            "metrics": {
                "content_length": len(content),
                "original_format": input_file.suffix
            }
        }

    def _create_markdown_with_metadata(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create markdown content with metadata header.

        Args:
            content: Main content
            metadata: Metadata dictionary

        Returns:
            Formatted markdown with metadata header
        """
        content_type = metadata.get('type', 'document')

        if content_type == 'youtube':
            return self._create_youtube_markdown(content, metadata)
        elif content_type in ['audio', 'video']:
            return self._create_media_markdown(content, metadata)
        else:
            return self._create_document_markdown(content, metadata)

    def _create_media_markdown(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create markdown for audio/video files following example.md format."""
        content_type = metadata.get('type', 'audio')
        title = metadata.get('title', metadata.get('source_name', 'Unknown'))

        # Start with title
        markdown_lines = [f"# {content_type.title()} Analysis: {title}", ""]

        # Add media information section
        markdown_lines.append(f"## {content_type.title()} Information")
        markdown_lines.append("")

        # Add relevant metadata based on content type
        if content_type == 'audio':
            if metadata.get('duration'):
                duration = metadata['duration']
                # Handle both numeric and string duration formats
                if isinstance(duration, str):
                    # If it's already formatted (e.g., "01:11:57"), use as-is
                    if ':' in duration:
                        markdown_lines.append(f"- **Duration**: {duration}")
                    else:
                        # Try to parse as numeric string
                        try:
                            duration = float(duration)
                            minutes = int(duration // 60)
                            seconds = int(duration % 60)
                            markdown_lines.append(f"- **Duration**: {minutes:02d}:{seconds:02d}")
                        except ValueError:
                            markdown_lines.append(f"- **Duration**: {duration}")
                else:
                    # Numeric duration
                    minutes = int(duration // 60)
                    seconds = int(duration % 60)
                    markdown_lines.append(f"- **Duration**: {minutes:02d}:{seconds:02d}")

            if metadata.get('channels'):
                markdown_lines.append(f"- **Channels**: {metadata['channels']}")
            if metadata.get('sample_rate'):
                markdown_lines.append(f"- **Sample Rate**: {metadata['sample_rate']}")
            if metadata.get('bit_depth'):
                markdown_lines.append(f"- **Bit Depth**: {metadata['bit_depth']}")

        elif content_type == 'video':
            if metadata.get('duration'):
                duration = metadata['duration']
                # Handle both numeric and string duration formats
                if isinstance(duration, str):
                    # If it's already formatted (e.g., "01:11:57"), use as-is
                    if ':' in duration:
                        markdown_lines.append(f"- **Duration**: {duration}")
                    else:
                        # Try to parse as numeric string
                        try:
                            duration = float(duration)
                            minutes = int(duration // 60)
                            seconds = int(duration % 60)
                            markdown_lines.append(f"- **Duration**: {minutes:02d}:{seconds:02d}")
                        except ValueError:
                            markdown_lines.append(f"- **Duration**: {duration}")
                else:
                    # Numeric duration
                    minutes = int(duration // 60)
                    seconds = int(duration % 60)
                    markdown_lines.append(f"- **Duration**: {minutes:02d}:{seconds:02d}")

            # Add video-specific metadata if available
            if metadata.get('resolution'):
                markdown_lines.append(f"- **Resolution**: {metadata['resolution']}")
            if metadata.get('fps'):
                markdown_lines.append(f"- **FPS**: {metadata['fps']}")
            if metadata.get('format'):
                markdown_lines.append(f"- **Format**: {metadata['format']}")
            if metadata.get('video_codec'):
                markdown_lines.append(f"- **Video Codec**: {metadata['video_codec']}")
            if metadata.get('audio_codec'):
                markdown_lines.append(f"- **Audio**: Yes ({metadata['audio_codec']})")

        markdown_lines.append("")
        markdown_lines.append("")

        # Add transcript section
        markdown_lines.append("## Transcript")
        markdown_lines.append("")

        # Add the actual transcript content
        if content:
            # If content already contains timestamps, use it directly
            if '[' in content and ']' in content:
                markdown_lines.append(content)
            else:
                # If no timestamps, add as plain text
                markdown_lines.append(content)
        else:
            markdown_lines.append("*No transcript available*")

        return "\n".join(markdown_lines)

    def _create_youtube_markdown(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create markdown for YouTube videos with comprehensive metadata."""
        title = metadata.get('title', 'YouTube Video')

        # Start with title
        markdown_lines = [f"# Youtube Analysis: {title}", ""]

        # Add YouTube Information section
        markdown_lines.append("## Youtube Information")
        markdown_lines.append("")

        # Format duration in a user-friendly way
        if metadata.get('duration'):
            duration = metadata['duration']
            if isinstance(duration, (int, float)):
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                if hours > 0:
                    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    duration_str = f"{minutes:02d}:{seconds:02d}"
                markdown_lines.append(f"- **Duration**: {duration_str}")

        # Add channel information
        if metadata.get('uploader'):
            markdown_lines.append(f"- **Channel**: {metadata['uploader']}")

        # Add language if available
        if metadata.get('language'):
            markdown_lines.append(f"- **Language**: {metadata['language']}")

        # Add upload date if available
        if metadata.get('upload_date'):
            upload_date = metadata['upload_date']
            # Format upload_date if it's in YYYYMMDD format
            if isinstance(upload_date, str) and len(upload_date) == 8 and upload_date.isdigit():
                formatted_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                markdown_lines.append(f"- **Upload Date**: {formatted_date}")
            else:
                markdown_lines.append(f"- **Upload Date**: {upload_date}")

        # Add view count if available
        if metadata.get('view_count'):
            view_count = metadata['view_count']
            if isinstance(view_count, int):
                # Format with commas for readability
                formatted_views = f"{view_count:,}"
                markdown_lines.append(f"- **Views**: {formatted_views}")

        # Add like count if available
        if metadata.get('like_count'):
            like_count = metadata['like_count']
            if isinstance(like_count, int):
                formatted_likes = f"{like_count:,}"
                markdown_lines.append(f"- **Likes**: {formatted_likes}")

        # Add comment count if available
        if metadata.get('comment_count'):
            comment_count = metadata['comment_count']
            if isinstance(comment_count, int):
                formatted_comments = f"{comment_count:,}"
                markdown_lines.append(f"- **Comments**: {formatted_comments}")

        # Add video ID
        if metadata.get('video_id'):
            markdown_lines.append(f"- **Video ID**: {metadata['video_id']}")

        # Add categories if available
        if metadata.get('categories') and isinstance(metadata['categories'], list):
            categories = ", ".join(metadata['categories'])
            markdown_lines.append(f"- **Categories**: {categories}")

        # Add tags if available (limit to first 10 for readability)
        if metadata.get('tags') and isinstance(metadata['tags'], list):
            tags = metadata['tags'][:10]  # Limit to first 10 tags
            tags_str = ", ".join(tags)
            if len(metadata['tags']) > 10:
                tags_str += f" (and {len(metadata['tags']) - 10} more)"
            markdown_lines.append(f"- **Tags**: {tags_str}")

        # Add description if available (truncated for readability)
        if metadata.get('description'):
            description = metadata['description']
            if isinstance(description, str) and description.strip():
                # Truncate description if it's too long
                if len(description) > 300:
                    description = description[:300] + "..."
                # Replace newlines with spaces for better formatting
                description = description.replace('\n', ' ').replace('\r', ' ')
                markdown_lines.append(f"- **Description**: {description}")

        # Add processing timestamp
        if metadata.get('created_at'):
            markdown_lines.append(f"- **Created At**: {metadata['created_at']}")

        # Add source URL
        if metadata.get('source'):
            markdown_lines.append(f"- **Source**: {metadata['source']}")

        markdown_lines.extend(["", ""])

        # Add content section
        markdown_lines.append("## Content")
        markdown_lines.append("")

        # Add the actual transcript content
        if content:
            # If content already contains timestamps, use it directly
            if '[' in content and ']' in content:
                markdown_lines.append(content)
            else:
                # If no timestamps, add as plain text
                markdown_lines.append(content)
        else:
            markdown_lines.append("*No transcript available*")

        return "\n".join(markdown_lines)

    def _create_document_markdown(self, content: str, metadata: Dict[str, Any]) -> str:
        """Create markdown for document files with H1 title and H2 sections format."""
        content_type = metadata.get('type', 'document')
        title = metadata.get('title', 'Unknown Document')
        source = metadata.get('source', 'Unknown Source')

        # Start with title
        markdown_lines = [f"# {content_type.title()} Analysis: {title}", ""]

        # Add document information section
        markdown_lines.append(f"## {content_type.title()} Information")
        markdown_lines.append("")

        # Add relevant metadata
        if metadata.get('file_extension'):
            markdown_lines.append(f"- **File Extension**: {metadata['file_extension']}")
        if metadata.get('processed_with'):
            markdown_lines.append(f"- **Processed With**: {metadata['processed_with']}")
        if metadata.get('language'):
            markdown_lines.append(f"- **Language**: {metadata['language']}")
        if metadata.get('pages'):
            markdown_lines.append(f"- **Pages**: {metadata['pages']}")
        if metadata.get('created_at'):
            markdown_lines.append(f"- **Created At**: {metadata['created_at']}")
        if source != 'Unknown Source':
            markdown_lines.append(f"- **Source**: {source}")

        markdown_lines.append("")
        markdown_lines.append("")

        # Add content section
        markdown_lines.append("## Content")
        markdown_lines.append("")

        # Add the actual content
        if content:
            markdown_lines.append(content)
        else:
            markdown_lines.append("*No content available*")

        return "\n".join(markdown_lines)
