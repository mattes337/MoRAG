"""Markitdown-based document converter implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import structlog
from langdetect import detect

from morag_core.interfaces.converter import (
    BaseConverter,
    ChunkingStrategy,
    ConversionOptions,
    ConversionResult,
    QualityScore,
    ConversionError,
    UnsupportedFormatError,
)
from morag_core.models.document import Document, DocumentMetadata, DocumentType
from morag_core.utils.file_handling import get_file_info, detect_format, get_file_hash
from morag_core.config import get_settings

from ..services.markitdown_service import MarkitdownService
from .document_formatter import DocumentFormatter

logger = structlog.get_logger(__name__)


class MarkitdownConverter(BaseConverter):
    """Base markitdown-based document converter implementation."""

    def __init__(self):
        """Initialize markitdown converter."""
        self.supported_formats: Set[str] = set()
        self._markitdown_service = None
        self.settings = get_settings()
        self.formatter = DocumentFormatter()

    async def _get_markitdown_service(self) -> MarkitdownService:
        """Get or create markitdown service instance."""
        if self._markitdown_service is None:
            self._markitdown_service = MarkitdownService()
        return self._markitdown_service

    async def convert(
        self, file_path: Union[str, Path], options: Optional[ConversionOptions] = None
    ) -> ConversionResult:
        """Convert document to text using markitdown.

        Args:
            file_path: Path to document file
            options: Conversion options

        Returns:
            Conversion result with document

        Raises:
            ConversionError: If conversion fails
            UnsupportedFormatError: If document format is not supported
        """
        file_path = Path(file_path)
        options = options or ConversionOptions()

        # Check if markitdown is enabled
        if not getattr(self.settings, 'markitdown_enabled', True):
            raise ConversionError("Markitdown is disabled in configuration")

        # Validate input
        await self.validate_input(file_path)

        # Detect format if not specified
        format_type = options.format_type or self.detect_format(file_path)

        # Check if format is supported
        if not await self.supports_format(format_type):
            raise UnsupportedFormatError(f"Format '{format_type}' is not supported by this converter")

        try:
            # Get file info
            file_info = get_file_info(file_path)

            # Calculate file checksum
            checksum = get_file_hash(file_path, "sha256")

            # Create document metadata
            metadata = DocumentMetadata(
                source_type=self._map_format_to_document_type(format_type),
                source_name=file_info["file_name"],
                source_path=str(file_path),
                mime_type=file_info["mime_type"],
                file_size=file_info["file_size"],
                checksum=checksum,
            )

            # Create document
            document = Document(metadata=metadata)

            # Convert using markitdown
            markitdown_service = await self._get_markitdown_service()
            raw_markdown_content = await markitdown_service.convert_file(file_path, self._get_markitdown_options(options))

            # Clean up markitdown artifacts
            cleaned_content = self.formatter.clean_markitdown_artifacts(raw_markdown_content)

            # Early quality validation for all file types - check raw content before formatting
            if cleaned_content and not self._validate_conversion_by_format(cleaned_content, format_type, file_path):
                error_msg = f"Conversion failed for {format_type} - raw output does not appear to be proper markdown"
                logger.error("Conversion validation failed",
                           file_path=str(file_path),
                           format_type=format_type,
                           content_preview=cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content)
                raise ConversionError(error_msg)

            # Extract additional metadata from content
            content_metadata = self.formatter.extract_metadata_from_content(cleaned_content)

            # Update document metadata with extracted information
            for key, value in content_metadata.items():
                if not hasattr(metadata, key) or getattr(metadata, key) is None:
                    setattr(metadata, key, value)

            # Format content according to LLM documentation specifications
            formatted_content = self.formatter.format_document_content(
                cleaned_content,
                file_path,
                {
                    'page_count': content_metadata.get('page_count'),
                    'word_count': content_metadata.get('word_count'),
                    'document_type': self._map_format_to_document_type(format_type).value,
                    'language': content_metadata.get('language'),
                    'file_size': file_info.get('file_size'),
                    'sections': content_metadata.get('sections', []),
                    'section_count': content_metadata.get('section_count', 0)
                }
            )

            # Set the formatted content as the document text
            document.raw_text = formatted_content

            # Detect language if not specified
            if not document.metadata.language and document.raw_text:
                try:
                    # Try using the new language detection service first
                    try:
                        from morag_core.services.language_detection import get_language_service
                        language_service = get_language_service()
                        document.metadata.language = language_service.detect_language(document.raw_text[:1000])
                        logger.debug("Language detected using language service",
                                   language=document.metadata.language)
                    except ImportError:
                        # Fallback to langdetect
                        document.metadata.language = detect(document.raw_text[:1000])
                        logger.debug("Language detected using langdetdown fallback",
                                   language=document.metadata.language)
                except Exception as e:
                    logger.warning(
                        "Failed to detect language",
                        error=str(e),
                        error_type=e.__class__.__name__,
                    )

            # Chunk document if requested
            if options.chunking_strategy != ChunkingStrategy.NONE and document.raw_text:
                document = await self._chunk_document(document, options)

            # Assess quality
            quality = await self.assess_quality(document)

            # Check if quality is acceptable
            min_quality_threshold = 0.3  # Minimum acceptable quality score
            if quality.overall_score < min_quality_threshold:
                error_msg = f"Conversion quality too low (score: {quality.overall_score:.2f}, threshold: {min_quality_threshold}). Issues: {', '.join(quality.issues_detected)}"
                logger.error("Conversion failed due to poor quality",
                           file_path=str(file_path),
                           quality_score=quality.overall_score,
                           issues=quality.issues_detected)
                raise ConversionError(error_msg)

            # Additional validation for all file types - check if output looks like proper markdown
            if document.raw_text and not self._validate_conversion_by_format(document.raw_text, format_type, file_path):
                error_msg = f"Conversion failed for {format_type} - output does not appear to be proper markdown"
                logger.error("Conversion validation failed",
                           file_path=str(file_path),
                           format_type=format_type,
                           content_preview=document.raw_text[:200] + "..." if len(document.raw_text) > 200 else document.raw_text)
                raise ConversionError(error_msg)

            return ConversionResult(
                success=True,
                content=document.raw_text or "",
                metadata={"converter": "markitdown", "format": format_type},
                quality_score=quality,
                document=document,
                warnings=[],
            )

        except ConversionError:
            # Re-raise conversion errors
            raise
        except UnsupportedFormatError:
            # Re-raise unsupported format errors
            raise
        except Exception as e:
            # Wrap other exceptions
            logger.error(
                "Markitdown conversion failed",
                error=str(e),
                error_type=e.__class__.__name__,
                file_path=str(file_path),
            )
            raise ConversionError(f"Failed to convert document with markitdown: {str(e)}")

    def _get_markitdown_options(self, options: ConversionOptions) -> Dict[str, Any]:
        """Convert MoRAG conversion options to markitdown options.
        
        Args:
            options: MoRAG conversion options
            
        Returns:
            Dictionary of markitdown options
        """
        markitdown_options = {}
        
        # Map relevant options to markitdown format
        # Note: Specific option mapping will be refined as we implement each converter
        
        return markitdown_options

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        # Check both our supported formats and markitdown's supported formats
        if format_type.lower() in self.supported_formats:
            markitdown_service = await self._get_markitdown_service()
            return await markitdown_service.supports_format(format_type)
        return False

    async def validate_input(self, file_path: Union[str, Path]) -> bool:
        """Validate input file.

        Args:
            file_path: Path to document file

        Returns:
            True if input is valid

        Raises:
            ConversionError: If input is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ConversionError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ConversionError(f"Not a file: {file_path}")

        if file_path.stat().st_size == 0:
            raise ConversionError(f"Empty file: {file_path}")

        return True

    def detect_format(self, file_path: Union[str, Path]) -> str:
        """Detect document format.

        Args:
            file_path: Path to document file

        Returns:
            Detected format string
        """
        return detect_format(file_path)

    def _map_format_to_document_type(self, format_type: str) -> DocumentType:
        """Map format type to document type.

        Args:
            format_type: Format type string

        Returns:
            Document type enum value
        """
        format_mapping = {
            "pdf": DocumentType.PDF,
            "docx": DocumentType.WORD,
            "doc": DocumentType.WORD,
            "xlsx": DocumentType.EXCEL,
            "xls": DocumentType.EXCEL,
            "pptx": DocumentType.POWERPOINT,
            "ppt": DocumentType.POWERPOINT,
            "txt": DocumentType.TEXT,
            "md": DocumentType.MARKDOWN,
            "html": DocumentType.HTML,
            "htm": DocumentType.HTML,
            "jpg": DocumentType.IMAGE,
            "jpeg": DocumentType.IMAGE,
            "png": DocumentType.IMAGE,
            "gif": DocumentType.IMAGE,
            "bmp": DocumentType.IMAGE,
            "tiff": DocumentType.IMAGE,
            "webp": DocumentType.IMAGE,
            "svg": DocumentType.IMAGE,
            "mp3": DocumentType.AUDIO,
            "wav": DocumentType.AUDIO,
            "m4a": DocumentType.AUDIO,
            "flac": DocumentType.AUDIO,
            "aac": DocumentType.AUDIO,
            "ogg": DocumentType.AUDIO,
            "wma": DocumentType.AUDIO,
            "mp4": DocumentType.VIDEO,
            "avi": DocumentType.VIDEO,
            "mov": DocumentType.VIDEO,
            "mkv": DocumentType.VIDEO,
            "zip": DocumentType.ARCHIVE,
            "epub": DocumentType.EBOOK,
            "tar": DocumentType.ARCHIVE,
            "gz": DocumentType.ARCHIVE,
            "rar": DocumentType.ARCHIVE,
            "7z": DocumentType.ARCHIVE,
        }
        return format_mapping.get(format_type.lower(), DocumentType.UNKNOWN)

    async def assess_quality(self, document: Document) -> QualityScore:
        """Assess the quality of the converted document.

        Args:
            document: Document to assess

        Returns:
            Quality score
        """
        score = 1.0
        issues = []

        # Check if document has content
        if not document.raw_text:
            score = 0.0
            issues.append("No text content extracted")
        else:
            text_length = len(document.raw_text.strip())

            # Check text length
            if text_length < 50:
                score = max(0.1, score - 0.3)
                issues.append("Very short text content")
            elif text_length > 1000000:
                score = max(0.1, score - 0.1)
                issues.append("Extremely large text content")

            # Additional quality checks for all document types
            if document.metadata and hasattr(document.metadata, 'source_path'):
                source_path = Path(str(document.metadata.source_path))
                format_type = source_path.suffix.lower().lstrip('.')

                # Check if content looks properly converted for the file type
                if not self._validate_conversion_by_format(document.raw_text, format_type, source_path):
                    score = max(0.1, score - 0.5)
                    issues.append(f"Content does not appear to be properly converted from {format_type} format")

            # Check for chunks
            if not document.chunks:
                score = max(0.1, score - 0.2)
                issues.append("No chunks created")
            else:
                # Check chunk quality
                total_chunks = len(document.chunks)
                empty_chunks = sum(1 for chunk in document.chunks if not chunk.content.strip())
                short_chunks = sum(1 for chunk in document.chunks if 0 < len(chunk.content.strip()) < 50)

                if empty_chunks > 0:
                    score = max(0.1, score - (empty_chunks / total_chunks) * 0.3)
                    issues.append(f"{empty_chunks} empty chunks")

                if short_chunks > total_chunks / 2:
                    score = max(0.1, score - 0.2)
                    issues.append("Majority of chunks are very short")

        return QualityScore(
            overall_score=score,
            issues_detected=issues,
        )

    def _validate_conversion_by_format(self, content: str, format_type: str, file_path: Path) -> bool:
        """Validate conversion quality based on file format.

        Args:
            content: Converted content
            format_type: Original file format
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
        import re

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
        import re

        # HTML should convert to structured markdown
        has_headers = bool(re.search(r'^#+ ', content, re.MULTILINE))
        has_paragraphs = '\n\n' in content
        has_lists = bool(re.search(r'^\* |^\d+\. ', content, re.MULTILINE))

        # Should not contain excessive raw HTML tags
        html_tag_ratio = len(re.findall(r'<[^>]+>', content)) / max(1, len(content.split()))

        return (has_headers or has_paragraphs or has_lists) and html_tag_ratio < 0.1

    def _validate_data_conversion(self, content: str) -> bool:
        """Validate data file conversion (CSV, Excel)."""
        import re

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
        import re

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


    async def _chunk_document(self, document: Document, options: ConversionOptions) -> Document:
        """Chunk document text.

        Args:
            document: Document to chunk
            options: Conversion options

        Returns:
            Chunked document
        """
        if not document.raw_text:
            return document

        import re

        # Get settings for default chunk configuration
        from morag_core.config import validate_configuration_and_log
        settings = validate_configuration_and_log()

        text = document.raw_text
        strategy = options.chunking_strategy
        chunk_size = options.chunk_size or settings.default_chunk_size
        chunk_overlap = options.chunk_overlap or settings.default_chunk_overlap

        # Log chunking operation details
        logger.info("Starting document chunking",
                   strategy=strategy,
                   chunk_size=chunk_size,
                   chunk_overlap=chunk_overlap,
                   text_length=len(text) if text else 0,
                   enable_page_based_chunking=getattr(settings, 'enable_page_based_chunking', False))

        # Apply chunking strategy
        if strategy == ChunkingStrategy.PAGE:
            # Page-based chunking - use configuration settings
            if getattr(settings, 'enable_page_based_chunking', False):
                await self._chunk_by_pages(document, options)
                return document
            else:
                # Fall back to paragraph chunking if page-based is disabled
                strategy = ChunkingStrategy.PARAGRAPH

        if strategy == ChunkingStrategy.CHARACTER:
            # Character-based chunking with word boundary preservation
            for i in range(0, len(text), chunk_size - chunk_overlap):
                end_pos = min(i + chunk_size, len(text))

                # Find word boundary near the end position
                if end_pos < len(text):
                    end_pos = self._find_word_boundary(text, end_pos, direction="backward")

                chunk_text = text[i:end_pos]
                if chunk_text.strip():
                    document.add_chunk(chunk_text)

        elif strategy == ChunkingStrategy.WORD:
            # Enhanced word-based chunking with better overlap
            words = text.split()
            current_chunk = []
            current_size = 0

            for word in words:
                word_length = len(word) + 1  # +1 for space

                # Check if adding this word would exceed chunk size
                if current_size + word_length > chunk_size and current_chunk:
                    # Create chunk from current words
                    chunk_text = " ".join(current_chunk)
                    document.add_chunk(chunk_text)

                    # Calculate overlap in words (more intelligent)
                    overlap_chars = min(chunk_overlap, current_size)
                    overlap_words = []
                    overlap_size = 0

                    # Add words from the end until we reach overlap size
                    for overlap_word in reversed(current_chunk):
                        word_size = len(overlap_word) + 1
                        if overlap_size + word_size <= overlap_chars:
                            overlap_words.insert(0, overlap_word)
                            overlap_size += word_size
                        else:
                            break

                    current_chunk = overlap_words
                    current_size = overlap_size

                current_chunk.append(word)
                current_size += word_length

            # Add final chunk if not empty
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                document.add_chunk(chunk_text)

        elif strategy == ChunkingStrategy.SENTENCE:
            # Enhanced sentence-based chunking with improved boundary detection
            sentence_boundaries = self._detect_sentence_boundaries(text)
            current_chunk = ""
            current_size = 0

            for i in range(len(sentence_boundaries) - 1):
                start_pos = sentence_boundaries[i]
                end_pos = sentence_boundaries[i + 1]
                sentence = text[start_pos:end_pos].strip()

                if not sentence:
                    continue

                sentence_size = len(sentence) + (1 if current_chunk else 0)  # +1 for space if not first

                # If single sentence is too long, split it at word boundaries
                if len(sentence) > chunk_size:
                    if current_chunk:
                        document.add_chunk(current_chunk.strip())
                        current_chunk = ""
                        current_size = 0

                    # Split long sentence at word boundaries
                    for j in range(0, len(sentence), chunk_size - chunk_overlap):
                        end_pos = min(j + chunk_size, len(sentence))
                        if end_pos < len(sentence):
                            end_pos = self._find_word_boundary(sentence, end_pos, direction="backward")

                        sentence_chunk = sentence[j:end_pos].strip()
                        if sentence_chunk:
                            document.add_chunk(sentence_chunk)
                    continue

                # Check if adding this sentence would exceed chunk size
                if current_size + sentence_size > chunk_size and current_chunk:
                    document.add_chunk(current_chunk.strip())
                    current_chunk = sentence
                    current_size = len(sentence)
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                        current_size += sentence_size
                    else:
                        current_chunk = sentence
                        current_size = len(sentence)

            # Add final chunk if not empty
            if current_chunk:
                document.add_chunk(current_chunk.strip())

        elif strategy == ChunkingStrategy.CHAPTER:
            # Chapter-based chunking (fallback for non-PDF documents)
            await self._chunk_by_chapters_fallback(document, options)
            return document

        elif strategy == ChunkingStrategy.PARAGRAPH:
            # Paragraph-based chunking
            paragraphs = re.split(r'\n\s*\n', text)
            current_chunk = []
            current_size = 0

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                if len(paragraph) > chunk_size:
                    # If a single paragraph is too long, add it as its own chunk
                    if current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        document.add_chunk(chunk_text)
                        current_chunk = []
                        current_size = 0

                    # Split the long paragraph into smaller chunks with word boundary preservation
                    for i in range(0, len(paragraph), chunk_size - chunk_overlap):
                        end_pos = min(i + chunk_size, len(paragraph))

                        # Find word boundary near the end position
                        if end_pos < len(paragraph):
                            end_pos = self._find_word_boundary(paragraph, end_pos, direction="backward")

                        chunk_text = paragraph[i:end_pos]
                        if chunk_text.strip():
                            document.add_chunk(chunk_text)
                    continue

                current_chunk.append(paragraph)
                current_size += len(paragraph) + 2  # +2 for newlines

                if current_size >= chunk_size:
                    chunk_text = "\n\n".join(current_chunk)
                    document.add_chunk(chunk_text)
                    current_chunk = []
                    current_size = 0

            # Add final chunk if not empty
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                document.add_chunk(chunk_text)

        return document

    def _find_word_boundary(self, text: str, position: int, direction: str = "backward") -> int:
        """Find the nearest word boundary from a given position.

        Args:
            text: Text to search in
            position: Starting position
            direction: Search direction ("backward" or "forward")

        Returns:
            Position of nearest word boundary
        """
        import re

        if position <= 0:
            return 0
        if position >= len(text):
            return len(text)

        # Enhanced word boundary detection to prevent mid-word splits
        if direction == "backward":
            # Search backward from position to find a safe word boundary
            # Start from the position and move backward until we find whitespace or punctuation
            for i in range(position, -1, -1):
                if i == 0:
                    return 0
                char = text[i]
                # Check if current character is whitespace or punctuation
                if re.match(r'[\s.!?;:,\-\(\)\[\]{}"\'\n\r\t]', char):
                    # Found a boundary, but make sure we're not in the middle of punctuation
                    # Move to the end of the whitespace/punctuation sequence
                    while i < len(text) and re.match(r'[\s\n\r\t]', text[i]):
                        i += 1
                    return i
                # Also check for word boundaries using regex
                if i > 0 and re.match(r'\w', text[i-1]) and not re.match(r'\w', char):
                    return i
            return 0
        else:  # forward
            # Search forward from position to find a safe word boundary
            for i in range(position, len(text)):
                char = text[i]
                # Check if current character is whitespace or punctuation
                if re.match(r'[\s.!?;:,\-\(\)\[\]{}"\'\n\r\t]', char):
                    return i
                # Also check for word boundaries using regex
                if i > 0 and re.match(r'\w', text[i-1]) and not re.match(r'\w', char):
                    return i
            return len(text)

    def _detect_sentence_boundaries(self, text: str) -> List[int]:
        """Detect sentence boundaries using improved regex patterns.

        Args:
            text: Text to analyze

        Returns:
            List of sentence boundary positions
        """
        import re

        # Enhanced sentence boundary detection
        # Handles abbreviations, decimal numbers, and complex punctuation
        sentence_pattern = r'''
            (?<!\w\.\w.)           # Not preceded by word.word.
            (?<![A-Z][a-z]\.)      # Not preceded by abbreviation like Mr.
            (?<!\d\.\d)            # Not preceded by decimal number
            (?<=\.|\!|\?)          # Preceded by sentence ending punctuation
            \s+                    # Followed by whitespace
            (?=[A-Z])              # Followed by capital letter
        '''

        boundaries = [0]  # Start of text
        for match in re.finditer(sentence_pattern, text, re.VERBOSE):
            boundaries.append(match.start())
        boundaries.append(len(text))  # End of text

        return boundaries

    async def _chunk_by_chapters_fallback(self, document: Document, options: ConversionOptions) -> None:
        """Fallback chapter chunking for non-PDF documents.

        Args:
            document: Document to chunk
            options: Conversion options
        """
        import re

        if not document.raw_text:
            return

        text = document.raw_text

        # Chapter detection patterns for general text
        chapter_patterns = [
            r'^Chapter\s+\d+.*$',  # "Chapter 1", "Chapter 2", etc.
            r'^CHAPTER\s+\d+.*$',  # "CHAPTER 1", "CHAPTER 2", etc.
            r'^\d+\.\s+[A-Z][^.]*$',  # "1. Introduction", "2. Methods", etc.
            r'^[A-Z][A-Z\s]{3,}$',  # All caps titles like "INTRODUCTION"
            r'^\d+\s+[A-Z][^.]*$',  # "1 Introduction", "2 Methods", etc.
            r'^#{1,3}\s+.*$',  # Markdown headers "# Title", "## Title", "### Title"
        ]

        lines = text.split('\n')
        current_chapter = ""
        current_chapter_title = ""
        chapter_count = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                current_chapter += "\n"
                continue

            # Check if this line is a chapter header
            is_chapter_header = False
            for pattern in chapter_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_chapter_header = True
                    break

            if is_chapter_header:
                # Save previous chapter if exists
                if current_chapter and current_chapter_title:
                    document.add_chunk(
                        content=current_chapter.strip(),
                        section=current_chapter_title,
                        metadata={
                            "chapter_number": chapter_count,
                            "is_chapter": True
                        }
                    )

                # Start new chapter
                chapter_count += 1
                current_chapter_title = line
                current_chapter = f"{line}\n\n"
            else:
                # Add line to current chapter
                if not current_chapter_title:
                    # First content without chapter header - create default chapter
                    current_chapter_title = "Introduction"
                    chapter_count = 1

                current_chapter += line + "\n"

        # Add final chapter
        if current_chapter and current_chapter_title:
            document.add_chunk(
                content=current_chapter.strip(),
                section=current_chapter_title,
                metadata={
                    "chapter_number": chapter_count,
                    "is_chapter": True
                }
            )

        logger.info(f"Created {chapter_count} chapters using fallback method")

    async def _chunk_by_pages(self, document: Document, options: ConversionOptions) -> None:
        """Chunk document by pages using configuration settings.

        Args:
            document: Document to chunk
            options: Conversion options
        """
        from morag_core.config import get_settings

        settings = get_settings()
        max_page_size = getattr(settings, 'max_page_chunk_size', 4000)

        if not document.raw_text:
            return

        # For documents without page information, fall back to paragraph chunking
        if not hasattr(document, 'pages') or not document.pages:
            logger.info("No page information available, falling back to paragraph chunking")
            await self._chunk_by_paragraphs_with_page_config(document, options, max_page_size)
            return

        # Process each page
        for page_num, page_content in enumerate(document.pages, 1):
            if not page_content.strip():
                continue

            # If page content is within size limit, create single chunk
            if len(page_content) <= max_page_size:
                document.add_chunk(
                    content=page_content,
                    page_number=page_num,
                    metadata={
                        "page_based_chunking": True,
                        "chunk_type": "page",
                        "page_number": page_num
                    }
                )
            else:
                # Split large pages while preserving page context
                await self._split_large_page(document, page_content, page_num, max_page_size)

    async def _chunk_by_paragraphs_with_page_config(self, document: Document, options: ConversionOptions, max_chunk_size: int) -> None:
        """Fallback chunking by paragraphs when page information is not available.

        Args:
            document: Document to chunk
            options: Conversion options
            max_chunk_size: Maximum chunk size from page configuration
        """
        text = document.raw_text
        paragraphs = [p for p in text.split('\n\n') if p.strip()]

        current_chunk = []
        current_size = 0
        chunk_overlap = options.chunk_overlap or 200

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) > max_chunk_size:
                # Add current chunk if exists
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    document.add_chunk(
                        content=chunk_text,
                        metadata={
                            "page_based_chunking": True,
                            "chunk_type": "paragraph_group",
                            "fallback_chunking": True
                        }
                    )
                    current_chunk = []
                    current_size = 0

                # Split the long paragraph with word boundary preservation
                await self._split_long_text(document, paragraph, max_chunk_size, chunk_overlap)
                continue

            current_chunk.append(paragraph)
            current_size += len(paragraph) + 2  # +2 for newlines

            if current_size >= max_chunk_size:
                chunk_text = "\n\n".join(current_chunk)
                document.add_chunk(
                    content=chunk_text,
                    metadata={
                        "page_based_chunking": True,
                        "chunk_type": "paragraph_group",
                        "fallback_chunking": True
                    }
                )
                current_chunk = []
                current_size = 0

        # Add final chunk if not empty
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            document.add_chunk(
                content=chunk_text,
                metadata={
                    "page_based_chunking": True,
                    "chunk_type": "paragraph_group",
                    "fallback_chunking": True
                }
            )

    async def _split_large_page(self, document: Document, page_content: str, page_num: int, max_size: int) -> None:
        """Split a large page into smaller chunks while preserving page context.

        Args:
            document: Document to add chunks to
            page_content: Content of the page
            page_num: Page number
            max_size: Maximum size per chunk
        """
        # Try to split by paragraphs first
        paragraphs = [p for p in page_content.split('\n\n') if p.strip()]

        current_chunk = []
        current_size = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If single paragraph is too large, split it
            if len(paragraph) > max_size:
                # Add current chunk if exists
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    document.add_chunk(
                        content=chunk_text,
                        page_number=page_num,
                        metadata={
                            "page_based_chunking": True,
                            "chunk_type": "page_split",
                            "page_number": page_num,
                            "chunk_index_on_page": chunk_index,
                            "is_partial_page": True
                        }
                    )
                    current_chunk = []
                    current_size = 0
                    chunk_index += 1

                # Split the long paragraph
                await self._split_long_text_with_page_context(document, paragraph, page_num, max_size, chunk_index)
                continue

            # Check if adding this paragraph would exceed size
            if current_size + len(paragraph) + 2 > max_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                document.add_chunk(
                    content=chunk_text,
                    page_number=page_num,
                    metadata={
                        "page_based_chunking": True,
                        "chunk_type": "page_split",
                        "page_number": page_num,
                        "chunk_index_on_page": chunk_index,
                        "is_partial_page": True
                    }
                )
                current_chunk = []
                current_size = 0
                chunk_index += 1

            current_chunk.append(paragraph)
            current_size += len(paragraph) + 2

        # Add final chunk if not empty
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            document.add_chunk(
                content=chunk_text,
                page_number=page_num,
                metadata={
                    "page_based_chunking": True,
                    "chunk_type": "page_split",
                    "page_number": page_num,
                    "chunk_index_on_page": chunk_index,
                    "is_partial_page": True
                }
            )

    async def _split_long_text(self, document: Document, text: str, max_size: int, overlap: int) -> None:
        """Split long text with word boundary preservation.

        Args:
            document: Document to add chunks to
            text: Text to split
            max_size: Maximum chunk size
            overlap: Overlap between chunks
        """
        for i in range(0, len(text), max_size - overlap):
            end_pos = min(i + max_size, len(text))

            # Find word boundary near the end position
            if end_pos < len(text):
                end_pos = self._find_word_boundary(text, end_pos, direction="backward")

            chunk_text = text[i:end_pos]
            if chunk_text.strip():
                document.add_chunk(
                    content=chunk_text,
                    metadata={
                        "page_based_chunking": True,
                        "chunk_type": "split_text",
                        "fallback_chunking": True
                    }
                )

    async def _split_long_text_with_page_context(self, document: Document, text: str, page_num: int, max_size: int, start_chunk_index: int) -> None:
        """Split long text with page context preservation.

        Args:
            document: Document to add chunks to
            text: Text to split
            page_num: Page number
            max_size: Maximum chunk size
            start_chunk_index: Starting chunk index for this page
        """
        chunk_index = start_chunk_index
        overlap = 200  # Fixed overlap for page splitting

        for i in range(0, len(text), max_size - overlap):
            end_pos = min(i + max_size, len(text))

            # Find word boundary near the end position
            if end_pos < len(text):
                end_pos = self._find_word_boundary(text, end_pos, direction="backward")

            chunk_text = text[i:end_pos]
            if chunk_text.strip():
                document.add_chunk(
                    content=chunk_text,
                    page_number=page_num,
                    metadata={
                        "page_based_chunking": True,
                        "chunk_type": "page_split",
                        "page_number": page_num,
                        "chunk_index_on_page": chunk_index,
                        "is_partial_page": True
                    }
                )
                chunk_index += 1
