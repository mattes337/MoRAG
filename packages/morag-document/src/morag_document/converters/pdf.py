"""PDF document converter implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

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
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Add page text to document
                            if options.chunking_strategy == "page":
                                document.add_chunk(
                                    content=page_text,
                                    page_number=i + 1,
                                )
                            
                            full_text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(
                            "Failed to extract text from PDF page",
                            page=i + 1,
                            error=str(e),
                            error_type=e.__class__.__name__,
                        )
                
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