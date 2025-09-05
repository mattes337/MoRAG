"""Concrete document converter implementations."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
from langdetect import detect

from morag_core.interfaces.converter import (
    ChunkingStrategy,
    ConversionOptions,
    ConversionResult,
    QualityScore,
    ConversionError,
    UnsupportedFormatError,
)
from morag_core.models.document import Document, DocumentMetadata, DocumentType
from morag_core.utils.file_handling import get_file_info, get_file_hash

from .base_converter import BaseMarkitdownConverter
from .chunking_processor import ChunkingProcessor

logger = structlog.get_logger(__name__)


class MarkitdownConverter(BaseMarkitdownConverter):
    """Concrete markitdown-based document converter implementation."""

    def __init__(self):
        """Initialize the converter."""
        super().__init__()
        self.chunking_processor = ChunkingProcessor()

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
                document = await self.chunking_processor.chunk_document(document, options)

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
        """Validate conversion quality based on file format."""
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


__all__ = ["MarkitdownConverter"]