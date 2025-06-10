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
        self._docling_available = self._check_docling_availability()

    def _check_docling_availability(self) -> bool:
        """Check if docling is available and safe to use.

        Returns:
            True if docling can be safely used, False otherwise
        """
        # Check if we should disable docling explicitly
        import os
        disable_docling = os.environ.get('MORAG_DISABLE_DOCLING', 'false').lower() == 'true'

        if disable_docling:
            logger.info("Docling explicitly disabled via MORAG_DISABLE_DOCLING")
            return False

        try:
            # Try importing docling
            import docling

            # Try a basic docling operation to ensure it works
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            # Test basic initialization without processing
            pipeline_options = PdfPipelineOptions()

            # Configure for CPU-safe operation
            force_cpu = os.environ.get('MORAG_FORCE_CPU', 'false').lower() == 'true'
            if force_cpu:
                # Use CPU-safe settings
                pipeline_options.do_ocr = False  # Disable OCR to avoid GPU dependencies
                pipeline_options.do_table_structure = False  # Disable table structure to reduce complexity
                logger.info("Configuring docling for CPU-only mode")
            else:
                # Use default settings
                pipeline_options.do_ocr = False  # Still disable for test
                pipeline_options.do_table_structure = False  # Still disable for test

            # This should not crash if PyTorch is compatible
            test_converter = DocumentConverter()

            logger.info("Docling is available and compatible for enhanced PDF processing",
                       cpu_mode=force_cpu)
            return True

        except ImportError as e:
            logger.info("Docling not available, falling back to pypdf for PDF processing", error=str(e))
            return False
        except Exception as e:
            logger.warning("Docling initialization failed, falling back to pypdf for PDF processing",
                         error=str(e), error_type=e.__class__.__name__)
            return False

    async def _extract_text_with_docling(self, file_path: Path, document: Document, options: ConversionOptions) -> Document:
        """Extract text from PDF using docling for better markdown conversion.

        Args:
            file_path: Path to PDF file
            document: Document to update
            options: Conversion options

        Returns:
            Updated document with markdown content

        Raises:
            ConversionError: If text extraction fails
        """
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            logger.info("Starting PDF to markdown conversion using docling", file_path=str(file_path))

            # Report progress if callback available
            progress_callback = getattr(options, 'progress_callback', None)
            if progress_callback:
                progress_callback(0.1, "Initializing PDF conversion with docling")

            # Configure docling for optimal markdown conversion with CPU safety
            pipeline_options = PdfPipelineOptions()

            # Check if we should use safer settings for CPU compatibility
            import os
            force_cpu = os.environ.get('MORAG_FORCE_CPU', 'false').lower() == 'true'

            if force_cpu:
                # Use safer settings for CPU-only mode
                pipeline_options.do_ocr = False  # Disable OCR to avoid GPU dependencies
                pipeline_options.do_table_structure = False  # Disable table structure to reduce complexity
                logger.info("Using CPU-safe docling configuration")
            else:
                pipeline_options.do_ocr = True  # Enable OCR for scanned PDFs
                pipeline_options.do_table_structure = True  # Preserve table structure

            if progress_callback:
                progress_callback(0.3, "Converting PDF structure and extracting text")

            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            # Convert PDF to docling document
            result = doc_converter.convert(file_path)

            if not result.document:
                raise ConversionError("Docling failed to convert PDF document")

            if progress_callback:
                progress_callback(0.7, "Generating markdown from PDF content")

            # Extract markdown content
            markdown_content = result.document.export_to_markdown()

            # Set document content
            document.raw_text = markdown_content

            # Extract metadata from docling result
            if hasattr(result.document, 'meta'):
                meta = result.document.meta
                if hasattr(meta, 'title') and meta.title:
                    document.metadata.title = meta.title
                if hasattr(meta, 'authors') and meta.authors:
                    document.metadata.author = ", ".join(meta.authors)
                if hasattr(meta, 'creation_date') and meta.creation_date:
                    document.metadata.created_at = meta.creation_date

            # Extract page information for page-based chunking
            if hasattr(result.document, 'pages'):
                document.pages = []
                for page in result.document.pages:
                    page_markdown = page.export_to_markdown() if hasattr(page, 'export_to_markdown') else str(page)
                    document.pages.append(page_markdown)

                document.metadata.page_count = len(document.pages)

            # Estimate word count from markdown
            document.metadata.word_count = len(markdown_content.split())

            if progress_callback:
                progress_callback(0.9, "PDF to markdown conversion completed")

            logger.info("Successfully converted PDF to markdown using docling",
                       file_path=str(file_path),
                       page_count=document.metadata.page_count,
                       word_count=document.metadata.word_count)

            return document

        except ImportError:
            logger.warning("Docling not available, falling back to pypdf")
            return await self._extract_text_with_pypdf(file_path, document, options)
        except Exception as e:
            logger.error("Docling PDF conversion failed, falling back to pypdf",
                        error=str(e),
                        error_type=e.__class__.__name__,
                        file_path=str(file_path))
            return await self._extract_text_with_pypdf(file_path, document, options)

    async def _extract_text(self, file_path: Path, document: Document, options: ConversionOptions) -> Document:
        """Extract text from PDF document using docling (preferred) or pypdf (fallback).

        Args:
            file_path: Path to PDF file
            document: Document to update
            options: Conversion options

        Returns:
            Updated document

        Raises:
            ConversionError: If text extraction fails
        """
        # Try docling first for better markdown conversion
        if self._docling_available:
            logger.info("Using docling for PDF processing", file_path=str(file_path))
            return await self._extract_text_with_docling(file_path, document, options)
        else:
            logger.info("Using pypdf for PDF processing", file_path=str(file_path))
            return await self._extract_text_with_pypdf(file_path, document, options)

    async def _extract_text_with_pypdf(self, file_path: Path, document: Document, options: ConversionOptions) -> Document:
        """Extract text from PDF document using pypdf (fallback method).

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
            logger.info("Starting PDF text extraction using pypdf", file_path=str(file_path))

            # Report progress if callback available
            progress_callback = getattr(options, 'progress_callback', None)
            if progress_callback:
                progress_callback(0.1, "Initializing PDF text extraction with pypdf")

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
                total_pages = len(pdf.pages)

                if progress_callback:
                    progress_callback(0.3, f"Extracting text from {total_pages} pages")

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

                            # Report progress every 10 pages or at the end
                            if progress_callback and (page_number % 10 == 0 or page_number == total_pages):
                                progress = 0.3 + (page_number / total_pages) * 0.5  # 30% to 80% for page processing
                                progress_callback(progress, f"Processed {page_number}/{total_pages} pages")

                    except Exception as e:
                        logger.warning(
                            "Failed to extract text from PDF page",
                            page=i + 1,
                            error=str(e),
                            error_type=e.__class__.__name__,
                        )

                # Handle chapter chunking
                if options.chunking_strategy == "chapter":
                    if progress_callback:
                        progress_callback(0.85, "Processing chapters")
                    await self._chunk_by_chapters(document, page_texts)

                # Set raw text
                document.raw_text = full_text

                # Estimate word count
                document.metadata.word_count = len(full_text.split())

                if progress_callback:
                    progress_callback(0.9, "PDF text extraction completed")

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