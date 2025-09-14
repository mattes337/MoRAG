"""Base interfaces for content processors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..exceptions import ProcessingError


class ProcessingConfig:
    """Base configuration for content processing."""

    def __init__(self, **kwargs):
        """Initialize processing config with flexible parameter handling."""
        # File path (required for document processing)
        self.file_path: Optional[str] = kwargs.get('file_path')

        # Common configuration options for all processors
        self.max_file_size: Optional[int] = kwargs.get('max_file_size')
        self.quality_threshold: float = kwargs.get('quality_threshold', 0.7)
        self.extract_metadata: bool = kwargs.get('extract_metadata', True)

        # Document-specific options
        self.chunking_strategy: Optional[str] = kwargs.get('chunking_strategy')
        self.chunk_size: Optional[int] = kwargs.get('chunk_size')
        self.chunk_overlap: Optional[int] = kwargs.get('chunk_overlap')

        # Additional options that may be passed but should be ignored by this config
        # These are handled at higher levels (service/task level)
        self.webhook_url: Optional[str] = kwargs.get('webhook_url')
        self.metadata: Optional[Dict[str, Any]] = kwargs.get('metadata')
        self.use_docling: Optional[bool] = kwargs.get('use_docling')
        self.store_in_vector_db: Optional[bool] = kwargs.get('store_in_vector_db')
        self.generate_embeddings: Optional[bool] = kwargs.get('generate_embeddings')

        # Document management options (handled at service level)
        self.document_id: Optional[str] = kwargs.get('document_id')
        self.replace_existing: Optional[bool] = kwargs.get('replace_existing')

        # Remote processing options (handled at service level)
        self.remote: Optional[bool] = kwargs.get('remote')

        # Progress callback for long-running operations
        self.progress_callback: Optional[callable] = kwargs.get('progress_callback')

        # Store any additional unknown parameters for potential use by converters
        self.additional_options: Dict[str, Any] = {
            k: v for k, v in kwargs.items()
            if k not in {
                'file_path', 'max_file_size', 'quality_threshold', 'extract_metadata',
                'chunking_strategy', 'chunk_size', 'chunk_overlap', 'webhook_url',
                'metadata', 'use_docling', 'store_in_vector_db', 'generate_embeddings',
                'document_id', 'replace_existing', 'remote', 'progress_callback'
            }
        }


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


class IContentProcessor(ABC):
    """Interface for content processors."""

    @abstractmethod
    async def process(self, content: Any, options: Dict) -> ProcessingResult:
        """Process content with the given options.

        Args:
            content: Content to process (file path, text, etc.)
            options: Processing options

        Returns:
            ProcessingResult with processed content
        """
        pass


class IServiceCoordinator(ABC):
    """Interface for service coordination."""

    @abstractmethod
    async def get_service(self, service_type: str) -> Any:
        """Get a service instance by type.

        Args:
            service_type: Type of service to retrieve

        Returns:
            Service instance
        """
        pass

    @abstractmethod
    async def initialize_services(self) -> None:
        """Initialize all required services."""
        pass

    @abstractmethod
    async def cleanup_services(self) -> None:
        """Cleanup and dispose of services."""
        pass