"""Base document converter implementation."""

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
from morag_core.utils.file_handling import get_file_info, detect_format
from morag_core.config import get_settings

logger = structlog.get_logger(__name__)


class DocumentConverter(BaseConverter):
    """Base document converter implementation."""

    def __init__(self):
        """Initialize document converter."""
        self.supported_formats: Set[str] = set()

    async def convert(
        self, file_path: Union[str, Path], options: Optional[ConversionOptions] = None
    ) -> ConversionResult:
        """Convert document to text.

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

            # Create document metadata
            metadata = DocumentMetadata(
                source_type=self._map_format_to_document_type(format_type),
                source_name=file_info["file_name"],
                source_path=str(file_path),
                mime_type=file_info["mime_type"],
                file_size=file_info["file_size"],
            )

            # Create document
            document = Document(metadata=metadata)

            # Extract text and update document
            document = await self._extract_text(file_path, document, options)

            # Detect language if not specified
            if not document.metadata.language and document.raw_text:
                try:
                    document.metadata.language = detect(document.raw_text[:1000])
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

            return ConversionResult(
                success=True,
                content=document.raw_text or "",
                metadata={},
                quality_score=quality,
                document=document,
                warnings=[],
            )

        except ConversionError:
            # Re-raise conversion errors
            raise
        except Exception as e:
            # Wrap other exceptions
            logger.error(
                "Document conversion failed",
                error=str(e),
                error_type=e.__class__.__name__,
                file_path=str(file_path),
            )
            raise ConversionError(f"Failed to convert document: {str(e)}")

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() in self.supported_formats

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
        """Detect format from file.

        Args:
            file_path: Path to document file

        Returns:
            Format type string
        """
        return detect_format(file_path)

    async def assess_quality(self, document: Document) -> QualityScore:
        """Assess document quality.

        Args:
            document: Document to assess

        Returns:
            Quality score
        """
        # Basic quality assessment
        score = 0.5  # Default medium score
        issues = []

        # Check if document has content
        if not document.raw_text:
            score = 0.0
            issues.append("No text content extracted")
        else:
            # Check text length
            text_length = len(document.raw_text)
            if text_length < 100:
                score = max(0.1, score - 0.3)
                issues.append("Very short text content")
            elif text_length > 1000000:
                score = max(0.1, score - 0.1)
                issues.append("Extremely large text content")

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

    async def _extract_text(self, file_path: Path, document: Document, options: ConversionOptions) -> Document:
        """Extract text from document.

        Args:
            file_path: Path to document file
            document: Document to update
            options: Conversion options

        Returns:
            Updated document

        Raises:
            ConversionError: If text extraction fails
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Text extraction not implemented in base converter")

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
        settings = get_settings()

        text = document.raw_text
        strategy = options.chunking_strategy
        chunk_size = options.chunk_size or settings.default_chunk_size
        chunk_overlap = options.chunk_overlap or settings.default_chunk_overlap

        # Apply chunking strategy
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

                    # Split the long paragraph into smaller chunks
                    for i in range(0, len(paragraph), chunk_size - chunk_overlap):
                        chunk_text = paragraph[i:i + chunk_size]
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

    def _map_format_to_document_type(self, format_type: str) -> DocumentType:
        """Map format type to document type.

        Args:
            format_type: Format type string

        Returns:
            Document type enum
        """
        format_map = {
            "pdf": DocumentType.PDF,
            "text": DocumentType.TEXT,
            "markdown": DocumentType.MARKDOWN,
            "html": DocumentType.HTML,
            "word": DocumentType.WORD,
            "excel": DocumentType.EXCEL,
            "powerpoint": DocumentType.POWERPOINT,
            "json": DocumentType.JSON,
            "xml": DocumentType.XML,
            "csv": DocumentType.CSV,
            "audio": DocumentType.AUDIO,
            "video": DocumentType.VIDEO,
            "image": DocumentType.IMAGE,
            "url": DocumentType.URL,
        }

        return format_map.get(format_type.lower(), DocumentType.UNKNOWN)

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

        # Word boundary pattern - matches spaces, punctuation, or start/end of string
        word_boundary_pattern = r'\s+|[.!?;:,\-\(\)\[\]{}"\']'

        if direction == "backward":
            # Search backward from position
            search_text = text[:position]
            matches = list(re.finditer(word_boundary_pattern, search_text))
            if matches:
                # Return position after the last boundary found
                return matches[-1].end()
            else:
                # No boundary found, return start
                return 0
        else:  # forward
            # Search forward from position
            search_text = text[position:]
            match = re.search(word_boundary_pattern, search_text)
            if match:
                # Return position of the boundary
                return position + match.start()
            else:
                # No boundary found, return end
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