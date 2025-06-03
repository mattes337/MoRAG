"""Base classes and interfaces for document conversion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import time
from enum import Enum


class ConversionError(Exception):
    """Base exception for conversion errors."""
    pass


class UnsupportedFormatError(ConversionError):
    """Raised when a format is not supported by a converter."""
    pass


class ChunkingStrategy(Enum):
    """Supported chunking strategies."""
    PAGE = "page"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


@dataclass
class ConversionOptions:
    """Configuration options for document conversion."""
    
    # General options
    preserve_formatting: bool = True
    extract_images: bool = True
    include_metadata: bool = True
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.PAGE
    
    # Quality options
    min_quality_threshold: float = 0.7
    enable_fallback: bool = True
    
    # Output options
    include_toc: bool = False
    clean_whitespace: bool = True
    max_content_length: Optional[int] = None
    
    # Format-specific options
    format_options: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def for_format(cls, format_type: str, **kwargs) -> 'ConversionOptions':
        """Create options optimized for a specific format."""
        options = cls(**kwargs)
        
        # Format-specific defaults
        if format_type.lower() == 'pdf':
            options.format_options.update({
                'use_ocr': True,
                'extract_tables': True,
                'preserve_layout': False
            })
        elif format_type.lower() in ['audio', 'video']:
            options.format_options.update({
                'enable_diarization': True,
                'include_timestamps': True,
                'confidence_threshold': 0.8
            })
        elif format_type.lower() == 'web':
            options.format_options.update({
                'follow_redirects': True,
                'extract_main_content': True,
                'include_navigation': False
            })
        
        return options


@dataclass
class QualityScore:
    """Quality assessment for conversion results."""
    
    overall_score: float
    completeness_score: float
    readability_score: float
    structure_score: float
    metadata_preservation: float
    
    def __post_init__(self):
        """Validate quality scores."""
        for score_name, score_value in self.__dict__.items():
            if not 0.0 <= score_value <= 1.0:
                raise ValueError(f"{score_name} must be between 0.0 and 1.0, got {score_value}")


@dataclass
class ConversionResult:
    """Result of document conversion."""
    
    content: str
    metadata: Dict[str, Any]
    quality_score: Optional[QualityScore] = None
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    
    # Conversion details
    original_format: Optional[str] = None
    target_format: str = "markdown"
    converter_used: Optional[str] = None
    fallback_used: bool = False
    
    @property
    def is_high_quality(self) -> bool:
        """Check if conversion meets quality standards."""
        if not self.quality_score:
            return False
        return self.quality_score.overall_score >= 0.8
    
    @property
    def word_count(self) -> int:
        """Get word count of converted content."""
        return len(self.content.split()) if self.content else 0


class BaseConverter(ABC):
    """Abstract base class for all document converters."""
    
    def __init__(self, name: str):
        self.name = name
        self.supported_formats: List[str] = []
        self.quality_threshold = 0.7
    
    @abstractmethod
    async def convert(self, file_path: Union[str, Path], options: ConversionOptions) -> ConversionResult:
        """Convert document to markdown format.
        
        Args:
            file_path: Path to the document to convert
            options: Conversion configuration options
            
        Returns:
            ConversionResult with converted content and metadata
            
        Raises:
            ConversionError: If conversion fails
            UnsupportedFormatError: If format is not supported
        """
        pass
    
    @abstractmethod
    def supports_format(self, format_type: str) -> bool:
        """Check if converter supports the given format.
        
        Args:
            format_type: File format (e.g., 'pdf', 'docx', 'mp3')
            
        Returns:
            True if format is supported, False otherwise
        """
        pass
    
    def get_quality_score(self, result: ConversionResult) -> float:
        """Calculate quality score for conversion result.
        
        Args:
            result: Conversion result to assess
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if result.quality_score:
            return result.quality_score.overall_score
        
        # Basic quality assessment based on content length and success
        if not result.success or not result.content:
            return 0.0
        
        # Simple heuristic: longer content generally indicates better extraction
        content_score = min(len(result.content) / 1000, 1.0)  # Normalize to 1000 chars
        
        # Penalize if warnings exist
        warning_penalty = len(result.warnings) * 0.1
        
        return max(0.0, min(1.0, content_score - warning_penalty))
    
    def detect_format(self, file_path: Union[str, Path]) -> str:
        """Detect document format from file extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detected format (e.g., 'pdf', 'docx')
        """
        path = Path(file_path)
        extension = path.suffix.lower().lstrip('.')
        
        # Map common extensions to standard format names
        format_mapping = {
            'pdf': 'pdf',
            'doc': 'word',
            'docx': 'word', 
            'xls': 'excel',
            'xlsx': 'excel',
            'ppt': 'powerpoint',
            'pptx': 'powerpoint',
            'mp3': 'audio',
            'wav': 'audio',
            'm4a': 'audio',
            'flac': 'audio',
            'mp4': 'video',
            'avi': 'video',
            'mov': 'video',
            'mkv': 'video',
            'html': 'web',
            'htm': 'web',
            'txt': 'text',
            'md': 'markdown',
            'markdown': 'markdown'
        }
        
        return format_mapping.get(extension, extension)
    
    async def validate_input(self, file_path: Union[str, Path]) -> None:
        """Validate input file before conversion.
        
        Args:
            file_path: Path to validate
            
        Raises:
            ConversionError: If validation fails
        """
        path = Path(file_path)
        
        if not path.exists():
            raise ConversionError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ConversionError(f"Path is not a file: {file_path}")
        
        format_type = self.detect_format(path)
        if not self.supports_format(format_type):
            raise UnsupportedFormatError(f"Format '{format_type}' not supported by {self.name}")
