"""Image document converter using markitdown framework."""

from pathlib import Path
from typing import Set, Union, Optional
import structlog

from morag_core.interfaces.converter import (
    ConversionResult,
    ConversionOptions,
    QualityScore,
    ConversionError,
    UnsupportedFormatError,
)
from morag_core.models.document import Document, DocumentType
from morag_document.services.markitdown_service import MarkitdownService

logger = structlog.get_logger(__name__)


class ImageConverter:
    """Image document converter using markitdown framework."""

    def __init__(self):
        """Initialize Image converter."""
        self.supported_formats: Set[str] = {
            "image", "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg"
        }
        self.markitdown_service = MarkitdownService()

    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported.

        Args:
            format_type: Format type string

        Returns:
            True if format is supported, False otherwise
        """
        return (format_type.lower() in self.supported_formats or
                format_type.lower() == "image")

    async def convert(
        self, file_path: Union[str, Path], options: Optional[ConversionOptions] = None
    ) -> ConversionResult:
        """Convert image file to text using markitdown.

        Args:
            file_path: Path to image file
            options: Conversion options

        Returns:
            Conversion result with document

        Raises:
            ConversionError: If conversion fails
            UnsupportedFormatError: If image format is not supported
        """
        file_path = Path(file_path)
        options = options or ConversionOptions()

        # Validate input
        if not file_path.exists():
            raise ConversionError(f"Image file not found: {file_path}")

        # Detect format if not specified
        format_type = options.format_type or file_path.suffix.lower().lstrip('.')

        # Check if format is supported
        if not await self.supports_format(format_type):
            raise UnsupportedFormatError(f"Format '{format_type}' is not supported by image converter")

        try:
            # Use markitdown for image OCR and description
            logger.info("Converting image file with markitdown", file_path=str(file_path))
            
            result = await self.markitdown_service.convert_file(file_path)
            
            # Create document
            document = Document(
                id=options.document_id,
                title=options.title or file_path.stem,
                raw_text=result.text_content,
                document_type=DocumentType.IMAGE,
                file_path=str(file_path),
                metadata={
                    "file_size": file_path.stat().st_size,
                    "format": format_type,
                    "conversion_method": "markitdown",
                    **result.metadata
                }
            )

            # Calculate quality score
            quality_score = QualityScore(
                overall_score=0.85,  # Good score for markitdown OCR
                text_extraction_score=0.8,
                structure_preservation_score=0.9,
                metadata_extraction_score=0.85,
                issues_detected=[]
            )

            return ConversionResult(
                document=document,
                quality_score=quality_score,
                warnings=[]
            )

        except Exception as e:
            logger.error("Image conversion failed", error=str(e), file_path=str(file_path))
            raise ConversionError(f"Failed to convert image file: {e}") from e
