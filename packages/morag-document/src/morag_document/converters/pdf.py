"""PDF document converter implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple

import structlog
import pypdf

from morag_core.interfaces.converter import (
    ConversionOptions,
    ConversionError,
)
from morag_core.models.document import Document

from .base import DocumentConverter

logger = structlog.get_logger(__name__)


class PDFConverter(DocumentConverter):
    """PDF document converter implementation."""

    def __init__(self):
        """Initialize PDF converter."""
        super().__init__()
        self.supported_formats = {"pdf"}

    async def _extract_text(self, file_path: Path, document: Document, options: ConversionOptions) -> Document:
        """Extract text from PDF document.

        Args:
            file_path: Path to PDF file
            document: Document to update
            options: Conversion options

        Returns:
            Updated document

        Raises:
            ConversionError: If text extraction fails
        """
        try:
            # Open PDF file
            with open(file_path, "rb") as file:
                pdf = pypdf.PdfReader(file)
                
                # Extract document info
                info = pdf.metadata
                if info:
                    document.metadata.title = info.title
                    document.metadata.author = info.author
                    document.metadata.created_at = info.creation_date
                    document.metadata.modified_at = info.modification_date
                
                # Extract page count
                document.metadata.page_count = len(pdf.pages)
                
                # Extract text from each page
                full_text = ""
                page_texts = []  # Store page texts for chapter processing

                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            page_number = i + 1
                            page_texts.append((page_number, page_text))

                            # Add page text to document if using page chunking
                            if options.chunking_strategy == "page":
                                document.add_chunk(
                                    content=page_text,
                                    page_number=page_number,
                                )

                            full_text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(
                            "Failed to extract text from PDF page",
                            page=i + 1,
                            error=str(e),
                            error_type=e.__class__.__name__,
                        )

                # Handle chapter chunking
                if options.chunking_strategy == "chapter":
                    await self._chunk_by_chapters(document, page_texts)
                
                # Set raw text
                document.raw_text = full_text
                
                # Estimate word count
                document.metadata.word_count = len(full_text.split())
                
                return document
                
        except Exception as e:
            logger.error(
                "PDF text extraction failed",
                error=str(e),
                error_type=e.__class__.__name__,
                file_path=str(file_path),
            )
            raise ConversionError(f"Failed to extract text from PDF: {str(e)}")

    async def _chunk_by_chapters(self, document: Document, page_texts: List[Tuple[int, str]]) -> None:
        """Chunk document by chapters with page numbers.

        Args:
            document: Document to add chunks to
            page_texts: List of (page_number, page_text) tuples
        """
        import re

        # Chapter detection patterns (ordered by priority)
        chapter_patterns = [
            r'^Chapter\s+\d+',  # "Chapter 1", "Chapter 2", etc.
            r'^CHAPTER\s+\d+',  # "CHAPTER 1", "CHAPTER 2", etc.
            r'^\d+\.\s+[A-Z][^.]*$',  # "1. Introduction", "2. Methods", etc.
            r'^[A-Z][A-Z\s]{3,}$',  # All caps titles like "INTRODUCTION"
            r'^\d+\s+[A-Z][^.]*$',  # "1 Introduction", "2 Methods", etc.
        ]

        current_chapter = ""
        current_chapter_title = ""
        current_chapter_start_page = 1
        chapter_count = 0

        for page_number, page_text in page_texts:
            lines = page_text.split('\n')
            chapter_found = False

            # Check first few lines of each page for chapter headers
            for i, line in enumerate(lines[:5]):  # Check first 5 lines
                line = line.strip()
                if not line:
                    continue

                # Check against chapter patterns
                for pattern in chapter_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        # Found a new chapter
                        if current_chapter and current_chapter_title:
                            # Save previous chapter
                            document.add_chunk(
                                content=current_chapter.strip(),
                                section=current_chapter_title,
                                page_number=current_chapter_start_page,
                                metadata={
                                    "chapter_number": chapter_count,
                                    "start_page": current_chapter_start_page,
                                    "end_page": page_number - 1,
                                    "page_count": page_number - current_chapter_start_page
                                }
                            )

                        # Start new chapter
                        chapter_count += 1
                        current_chapter_title = line
                        current_chapter = f"{line}\n\n"
                        current_chapter_start_page = page_number
                        chapter_found = True

                        # Add remaining lines from this page to the new chapter
                        remaining_lines = lines[i+1:]
                        if remaining_lines:
                            current_chapter += '\n'.join(remaining_lines) + "\n\n"
                        break

                if chapter_found:
                    break

            # If no chapter found on this page, add to current chapter
            if not chapter_found:
                if not current_chapter_title:
                    # First page without chapter header - create default chapter
                    current_chapter_title = "Introduction"
                    current_chapter = page_text + "\n\n"
                    current_chapter_start_page = page_number
                    chapter_count = 1
                else:
                    # Add to current chapter
                    current_chapter += page_text + "\n\n"

        # Add final chapter
        if current_chapter and current_chapter_title:
            document.add_chunk(
                content=current_chapter.strip(),
                section=current_chapter_title,
                page_number=current_chapter_start_page,
                metadata={
                    "chapter_number": chapter_count,
                    "start_page": current_chapter_start_page,
                    "end_page": page_texts[-1][0] if page_texts else current_chapter_start_page,
                    "page_count": (page_texts[-1][0] - current_chapter_start_page + 1) if page_texts else 1
                }
            )

        logger.info(f"Created {chapter_count} chapters from PDF document")