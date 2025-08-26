"""Markdown conversion stage implementation."""


from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import structlog

from ..models import Stage, StageType, StageStatus, StageResult, StageContext, StageMetadata
from ..exceptions import StageExecutionError, StageValidationError

logger = structlog.get_logger(__name__)

# Import core exceptions
try:
    from morag_core.exceptions import ProcessingError
except ImportError:
    class ProcessingError(Exception):  # type: ignore
        pass

# Import services - these are required for proper operation
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

        # Determine content type
        content_type = self._detect_content_type(input_file)

        # Check if we need MoRAG services (not needed for MarkItDown processing)
        needs_services = not self._should_use_markitdown(input_file, content_type)

        if needs_services and (not SERVICES_AVAILABLE or self.services is None):
            raise StageExecutionError(
                "MoRAG services not available for markdown conversion",
                stage_type=self.stage_type.value
            )

        logger.info("Starting markdown conversion",
                   input_file=str(input_file),
                   config=config)

        try:
            
            # Generate output filename
            sanitized_name = sanitize_filename(input_file.stem)
            output_file = context.output_dir / f"{sanitized_name}.md"
            context.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if we should use MarkItDown for better quality
            if self._should_use_markitdown(input_file, content_type):
                logger.info("Using MarkItDown for high-quality conversion",
                           input_file=str(input_file),
                           content_type=content_type.value if content_type else "unknown")
                result_data = await self._process_with_markitdown(input_file, output_file, config)
            # Otherwise use specialized processors
            elif content_type == ContentType.VIDEO:
                result_data = await self._process_video(input_file, output_file, config)
            elif content_type == ContentType.AUDIO:
                result_data = await self._process_audio(input_file, output_file, config)
            elif content_type == ContentType.DOCUMENT:
                result_data = await self._process_document(input_file, output_file, config)
            elif content_type == ContentType.WEB:
                result_data = await self._process_web(input_file, output_file, config)
            else:
                result_data = await self._process_text(input_file, output_file, config)
            
            # Create metadata
            metadata = StageMetadata(
                execution_time=0.0,  # Will be set by manager
                start_time=datetime.now(),
                input_files=[str(input_file)],
                output_files=[str(output_file)],
                config_used=config,
                metrics={
                    "content_type": content_type.value if content_type else "unknown",
                    "input_size_bytes": input_file.stat().st_size if input_file.exists() else 0,
                    "output_size_bytes": output_file.stat().st_size if output_file.exists() else 0,
                    **result_data.get("metrics", {})
                }
            )
            
            return StageResult(
                stage_type=self.stage_type,
                status=StageStatus.COMPLETED,
                output_files=[output_file],
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
        if len(input_files) != 1:
            return False

        input_file = input_files[0]

        # Check if file exists (for local files)
        # Handle URLs that may have been converted to Windows paths
        file_str = str(input_file)
        is_url = (
            file_str.startswith(('http://', 'https://')) or
            file_str.replace('\\', '/').startswith(('http://', 'https://')) or
            ('http:' in file_str and ('www.' in file_str or '.com' in file_str or '.org' in file_str or '.net' in file_str)) or
            ('https:' in file_str and ('www.' in file_str or '.com' in file_str or '.org' in file_str or '.net' in file_str))
        )

        if not is_url:
            if not input_file.exists():
                return False

        # Check if file type is supported
        content_type = self._detect_content_type(input_file)
        return content_type is not None
    
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
        sanitized_name = sanitize_filename(input_file.stem)
        output_file = context.output_dir / f"{sanitized_name}.md"
        return [output_file]
    
    def _detect_content_type(self, file_path: Path) -> Optional[ContentType]:
        """Detect content type from file path.

        Args:
            file_path: File path to analyze

        Returns:
            Detected content type or None
        """
        if not SERVICES_AVAILABLE:
            return None

        file_str = str(file_path)

        # Web URLs - handle Windows path conversion issue
        # On Windows, Path() converts URLs to backslash format, so we need to check both
        # Also check for the pattern where the URL scheme gets mangled
        is_url = (
            file_str.startswith(('http://', 'https://')) or
            file_str.replace('\\', '/').startswith(('http://', 'https://')) or
            ('http:' in file_str and ('www.' in file_str or '.com' in file_str or '.org' in file_str or '.net' in file_str)) or
            ('https:' in file_str and ('www.' in file_str or '.com' in file_str or '.org' in file_str or '.net' in file_str))
        )

        if is_url:
            return ContentType.WEB
        
        # Video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        if file_path.suffix.lower() in video_extensions:
            return ContentType.VIDEO
        
        # Audio files
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
        if file_path.suffix.lower() in audio_extensions:
            return ContentType.AUDIO
        
        # Document files
        doc_extensions = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'}
        if file_path.suffix.lower() in doc_extensions:
            return ContentType.DOCUMENT
        
        # Text files (default)
        text_extensions = {'.txt', '.md', '.rst', '.html', '.xml', '.json', '.csv'}
        if file_path.suffix.lower() in text_extensions:
            return ContentType.TEXT
        
        # Default to text for unknown types
        return ContentType.TEXT

    def _should_use_markitdown(self, file_path: Path, content_type: ContentType) -> bool:
        """Check if MarkItDown should be used for this file type.

        Args:
            file_path: File path to check
            content_type: Detected content type

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

        # Don't use MarkItDown for audio/video files (they need specialized processing)
        if content_type in [ContentType.AUDIO, ContentType.VIDEO]:
            return False

        # Don't use MarkItDown for web URLs (we have custom web processing)
        if content_type == ContentType.WEB:
            return False

        return False

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

    async def _process_web(self, input_file: Path, output_file: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process web URL to markdown.

        Args:
            input_file: Input URL (as Path object)
            output_file: Output markdown file
            config: Stage configuration

        Returns:
            Processing result data
        """
        # Normalize URL - fix Windows path conversion issue
        url = str(input_file)

        # Check if this looks like a URL that got mangled by Windows Path conversion
        if ('http:' in url or 'https:' in url) and ('\\' in url or not url.startswith(('http://', 'https://'))):
            # Convert backslashes to forward slashes first
            url = url.replace('\\', '/')

            # Then fix the protocol if needed
            if url.startswith('https:/') and not url.startswith('https://'):
                url = url.replace('https:/', 'https://', 1)
            elif url.startswith('http:/') and not url.startswith('http://'):
                url = url.replace('http:/', 'http://', 1)

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

        return {
            "content_type": "web",
            "title": result.metadata.get('title', "Web Content"),
            "metadata": result.metadata,
            "metrics": {
                "url": url,
                "content_length": len(result.text_content or ""),
                "links_followed": config.get('follow_links', False)
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

        if content_type in ['audio', 'video']:
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
