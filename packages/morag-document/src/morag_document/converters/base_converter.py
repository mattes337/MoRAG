"""Abstract base converter interface for document processing."""

import os
from abc import ABC, abstractmethod
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


class BaseMarkitdownConverter(BaseConverter, ABC):
    """Abstract base class for markitdown-based document converters."""

    def __init__(self):
        """Initialize base converter."""
        self.supported_formats: Set[str] = set()
        self._markitdown_service = None
        self.settings = get_settings()
        self.formatter = DocumentFormatter()

    async def _get_markitdown_service(self) -> MarkitdownService:
        """Get or create markitdown service instance."""
        if self._markitdown_service is None:
            self._markitdown_service = MarkitdownService()
        return self._markitdown_service

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

    @abstractmethod
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
        pass

    @abstractmethod
    async def assess_quality(self, document: Document) -> QualityScore:
        """Assess the quality of the converted document.

        Args:
            document: Document to assess

        Returns:
            Quality score
        """
        pass


__all__ = ["BaseMarkitdownConverter"]