"""Document chunking processor for various chunking strategies."""

import re
from typing import List
from pathlib import Path

import structlog

from morag_core.interfaces.converter import ChunkingStrategy, ConversionOptions
from morag_core.models.document import Document
from morag_core.config import get_settings, validate_configuration_and_log

logger = structlog.get_logger(__name__)


class ChunkingProcessor:
    """Handles document chunking with various strategies."""

    async def chunk_document(self, document: Document, options: ConversionOptions) -> Document:
        """Chunk document text.

        Args:
            document: Document to chunk
            options: Conversion options

        Returns:
            Chunked document
        """
        if not document.raw_text:
            return document

        # Get settings for default chunk configuration
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
        """Find the nearest word boundary from a given position."""
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
        """Detect sentence boundaries using improved regex patterns."""
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
        """Fallback chapter chunking for non-PDF documents."""
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
        """Chunk document by pages using configuration settings."""
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
        """Fallback chunking by paragraphs when page information is not available."""
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
        """Split a large page into smaller chunks while preserving page context."""
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
        """Split long text with word boundary preservation."""
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
        """Split long text with page context preservation."""
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


__all__ = ["ChunkingProcessor"]
