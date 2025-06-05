"""Document processor implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import structlog

from morag_core.interfaces.processor import (
    BaseProcessor,
    ProcessingConfig,
    ProcessingResult,
)
from morag_core.interfaces.converter import (
    ChunkingStrategy,
    ConversionOptions,
    ConversionError,
    UnsupportedFormatError,
)
from morag_core.models.document import Document, DocumentType
from morag_core.utils.file_handling import get_file_info, detect_format
from morag_core.exceptions import ValidationError, ProcessingError

from .converters.base import DocumentConverter
from .converters.pdf import PDFConverter
from .converters.word import WordConverter
from .converters.text import TextConverter
from .converters.excel import ExcelConverter
from .converters.presentation import PresentationConverter

logger = structlog.get_logger(__name__)


class DocumentProcessor(BaseProcessor):
    """Document processor implementation."""

    def __init__(self):
        """Initialize document processor."""
        self.converters: Dict[str, DocumentConverter] = {}
        self._register_converters()

    def _register_converters(self) -> None:
        """Register document converters."""
        # Register PDF converter
        pdf_converter = PDFConverter()
        for format_type in pdf_converter.supported_formats:
            self.converters[format_type] = pdf_converter

        # Register Word converter
        word_converter = WordConverter()
        for format_type in word_converter.supported_formats:
            self.converters[format_type] = word_converter

        # Register Text converter
        text_converter = TextConverter()
        for format_type in text_converter.supported_formats:
            self.converters[format_type] = text_converter
            
        # Register Excel converter
        excel_converter = ExcelConverter()
        for format_type in excel_converter.supported_formats:
            self.converters[format_type] = excel_converter
            
        # Register PowerPoint converter
        presentation_converter = PresentationConverter()
        for format_type in presentation_converter.supported_formats:
            self.converters[format_type] = presentation_converter

    async def process(self, config: ProcessingConfig) -> ProcessingResult:
        """Process document.

        Args:
            config: Processing configuration

        Returns:
            Processing result

        Raises:
            ProcessingError: If processing fails
            ValidationError: If input is invalid
        """
        # Validate input
        await self.validate_input(config)

        try:
            import time
            start_time = time.time()

            # Get file path from config
            file_path = config.file_path
            if not file_path:
                raise ValidationError("File path is required")

            # Detect format
            format_type = detect_format(file_path)

            # Check if format is supported
            if not await self.supports_format(format_type):
                raise UnsupportedFormatError(f"Format '{format_type}' is not supported")

            # Get converter for format
            converter = self.converters.get(format_type)
            if not converter:
                raise UnsupportedFormatError(f"No converter found for format '{format_type}'")

            # Create conversion options
            options = ConversionOptions(
                format_type=format_type,
                chunking_strategy=config.chunking_strategy or ChunkingStrategy.PARAGRAPH,
                chunk_size=config.chunk_size or 1000,
                chunk_overlap=config.chunk_overlap or 100,
                extract_metadata=config.extract_metadata or True,
            )

            # Convert document
            conversion_result = await converter.convert(file_path, options)

            processing_time = time.time() - start_time

            # Return processing result
            return ProcessingResult(
                success=True,
                processing_time=processing_time,
                document=conversion_result.document,
                metadata={
                    "quality_score": conversion_result.quality_score.overall_score,
                    "quality_issues": conversion_result.quality_score.issues_detected,
                    "warnings": conversion_result.warnings,
                },
            )

        except (ConversionError, UnsupportedFormatError) as e:
            # Re-raise as processing error
            raise ProcessingError(str(e))
        except Exception as e:
            # Log and wrap other exceptions
            logger.error(
                "Document processing failed",
                error=str(e),
                error_type=e.__class__.__name__,
            )
            raise ProcessingError(f"Failed to process document: {str(e)}")

    async def process_file(self, file_path: Union[str, Path], **kwargs) -> ProcessingResult:
        """Process document file.

        Args:
            file_path: Path to document file
            **kwargs: Additional processing options

        Returns:
            Processing result

        Raises:
            ProcessingError: If processing fails
            ValidationError: If input is invalid
        """
        # Create processing config
        config = ProcessingConfig(
            file_path=str(file_path),
            **kwargs
        )

        # Process document
        return await self.process(config)

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() in self.converters

    async def validate_input(self, config: ProcessingConfig) -> bool:
        """Validate processing input.

        Args:
            config: Processing configuration

        Returns:
            True if input is valid

        Raises:
            ValidationError: If input is invalid
        """
        if not config.file_path:
            raise ValidationError("File path is required")

        file_path = Path(config.file_path)
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValidationError(f"Not a file: {file_path}")

        return True