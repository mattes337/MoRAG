"""Base interfaces for content processors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..exceptions import ProcessingError


@dataclass
class ProcessingConfig:
    """Base configuration for content processing."""
    # File path (required for document processing)
    file_path: Optional[str] = None

    # Common configuration options for all processors
    max_file_size: Optional[int] = None
    quality_threshold: float = 0.7
    extract_metadata: bool = True

    # Document-specific options
    chunking_strategy: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


@dataclass
class ProcessingResult:
    """Base result for content processing."""
    success: bool
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    document: Optional[Any] = None  # For document processing results


class BaseProcessor(ABC):
    """Base class for content processors."""
    
    @abstractmethod
    async def process(self, file_path: Union[str, Path], config: Optional[ProcessingConfig] = None) -> ProcessingResult:
        """Process content from file.
        
        Args:
            file_path: Path to file to process
            config: Processing configuration
            
        Returns:
            ProcessingResult with processed content and metadata
            
        Raises:
            ProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def supports_format(self, format_type: str) -> bool:
        """Check if processor supports the given format.
        
        Args:
            format_type: Format type to check
            
        Returns:
            True if format is supported, False otherwise
        """
        pass
    
    def validate_input(self, file_path: Union[str, Path], config: ProcessingConfig) -> None:
        """Validate input file.
        
        Args:
            file_path: Path to file
            config: Processing configuration
            
        Raises:
            ProcessingError: If file does not exist, is not a file, or exceeds size limit
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")
        
        # Check if path is a file
        if not file_path.is_file():
            raise ProcessingError(f"Not a file: {file_path}")
        
        # Check file size if limit is specified
        if config.max_file_size is not None:
            file_size = file_path.stat().st_size
            if file_size > config.max_file_size:
                raise ProcessingError(
                    f"File too large: {file_size} bytes (max: {config.max_file_size})"
                )