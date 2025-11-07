"""Base interfaces for document conversion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..exceptions import MoRAGException


class ConversionError(MoRAGException):
    """Raised when document conversion fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=422, error_type="conversion_error")


class UnsupportedFormatError(ConversionError):
    """Raised when document format is not supported."""

    def __init__(self, format_type: str):
        super().__init__(f"Unsupported format: {format_type}")
        self.format_type = format_type


class ChunkingStrategy(str, Enum):
    """Strategy for chunking converted content."""
    PAGE = "page"  # Chunk by page boundaries
    CHAPTER = "chapter"  # Chunk by chapter boundaries with page numbers
    PARAGRAPH = "paragraph"  # Chunk by paragraphs
    SENTENCE = "sentence"  # Chunk by sentences
    WORD = "word"  # Chunk by words
    CHARACTER = "character"  # Chunk by characters
    FIXED_SIZE = "fixed_size"  # Chunk by fixed token/character size
    SEMANTIC = "semantic"  # Chunk by semantic boundaries
    HYBRID = "hybrid"  # Combination of strategies
    NONE = "none"  # No chunking


@dataclass
class ConversionOptions:
    """Options for document conversion."""
    # General options
    format_type: Optional[str] = None
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.PARAGRAPH
    chunk_size: Optional[int] = None  # Will use settings default if None
    chunk_overlap: Optional[int] = None  # Will use settings default if None
    extract_metadata: bool = True
    extract_images: bool = True
    extract_tables: bool = True
    extract_code_blocks: bool = True
    extract_math: bool = True
    extract_links: bool = True

    # Quality options
    quality_threshold: float = 0.7
    validate_output: bool = True
    fix_encoding_issues: bool = True

    # Progress callback for long-running operations
    progress_callback: Optional[callable] = None

    # Output options
    include_page_numbers: bool = True
    include_line_numbers: bool = False
    include_headers_footers: bool = True
    preserve_formatting: bool = True

    # Format-specific options
    format_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_format(cls, format_type: str) -> 'ConversionOptions':
        """Create format-specific conversion options."""
        options = cls()

        if format_type == 'pdf':
            options.extract_images = True
            options.extract_tables = True
            options.include_page_numbers = True
        elif format_type in ['audio', 'video']:
            options.chunking_strategy = ChunkingStrategy.SEMANTIC
            options.extract_metadata = True
            options.extract_images = False
            options.extract_tables = False
        elif format_type in ['word', 'excel', 'powerpoint']:
            options.extract_images = True
            options.extract_tables = True
        elif format_type == 'web':
            options.extract_images = True
            options.extract_tables = True
            options.extract_links = True

        return options


@dataclass
class QualityScore:
    """Quality assessment of conversion result."""
    overall_score: float = 0.0
    text_quality: float = 0.0
    structure_preservation: float = 0.0
    metadata_quality: float = 0.0
    image_quality: float = 0.0
    table_quality: float = 0.0
    issues_detected: List[str] = field(default_factory=list)


@dataclass
class ConversionResult:
    """Result of document conversion."""
    success: bool
    content: str
    metadata: Dict[str, Any]
    quality_score: Optional[QualityScore] = None
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    converter_used: Optional[str] = None
    fallback_used: bool = False
    original_format: Optional[str] = None
    images: List[Dict[str, Any]] = field(default_factory=list)
    word_count: int = 0
    document: Optional[Any] = None  # For document processing results


class BaseConverter(ABC):
    """Base class for document converters."""

    @abstractmethod
    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert document to markdown.

        Args:
            file_path: Path to document to convert
            options: Conversion options

        Returns:
            ConversionResult with markdown content and metadata

        Raises:
            ConversionError: If conversion fails
            UnsupportedFormatError: If format is not supported
        """
        pass

    @abstractmethod
    def supports_format(self, format_type: str) -> bool:
        """Check if converter supports the given format.

        Args:
            format_type: Format type to check

        Returns:
            True if format is supported, False otherwise
        """
        pass

    def assess_quality(self, content: str, metadata: Dict[str, Any]) -> QualityScore:
        """Assess quality of conversion result.

        Args:
            content: Converted content
            metadata: Extracted metadata

        Returns:
            QualityScore with quality assessment
        """
        # Basic quality assessment
        score = QualityScore()

        # Text quality based on content length and word count
        if content:
            word_count = len(content.split())
            score.text_quality = min(1.0, word_count / 1000)

        # Metadata quality based on number of metadata fields
        if metadata:
            score.metadata_quality = min(1.0, len(metadata) / 10)

        # Overall score is average of individual scores
        score.overall_score = (score.text_quality + score.metadata_quality) / 2

        return score

    def detect_format(self, file_path: Union[str, Path]) -> str:
        """Detect format from file extension.

        Args:
            file_path: Path to document

        Returns:
            Format type string
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower().lstrip('.')

        # Map common extensions to format types
        format_map = {
            # Documents
            'pdf': 'pdf',
            'txt': 'text',
            'md': 'markdown',
            'html': 'html',
            'htm': 'html',
            'xml': 'xml',
            'json': 'json',
            'csv': 'csv',

            # Office
            'doc': 'word',
            'docx': 'word',
            'xls': 'excel',
            'xlsx': 'excel',
            'ppt': 'powerpoint',
            'pptx': 'powerpoint',

            # Audio
            'mp3': 'audio',
            'wav': 'audio',
            'ogg': 'audio',
            'flac': 'audio',
            'm4a': 'audio',

            # Video
            'mp4': 'video',
            'avi': 'video',
            'mov': 'video',
            'mkv': 'video',
            'webm': 'video',

            # Images
            'jpg': 'image',
            'jpeg': 'image',
            'png': 'image',
            'gif': 'image',
            'bmp': 'image',
            'webp': 'image',
        }

        return format_map.get(extension, 'unknown')

    def validate_input(self, file_path: Union[str, Path]) -> None:
        """Validate input file.

        Args:
            file_path: Path to document

        Raises:
            ConversionError: If file does not exist or is not a file
            UnsupportedFormatError: If format is not supported
        """
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            raise ConversionError(f"File not found: {file_path}")

        # Check if path is a file
        if not file_path.is_file():
            raise ConversionError(f"Not a file: {file_path}")

        # Check if format is supported
        format_type = self.detect_format(file_path)
        if not self.supports_format(format_type):
            raise UnsupportedFormatError(format_type)
