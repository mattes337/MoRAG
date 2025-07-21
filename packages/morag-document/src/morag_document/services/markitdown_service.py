"""Markitdown service wrapper for document conversion."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Union
import structlog

from morag_core.exceptions import ConversionError, UnsupportedFormatError
from morag_core.config import get_settings

logger = structlog.get_logger(__name__)


class MarkitdownService:
    """Service wrapper for Microsoft's markitdown library."""
    
    def __init__(self):
        """Initialize markitdown service."""
        self._markitdown = None
        self._initialized = False
        self.settings = get_settings()
        
    async def _initialize(self) -> None:
        """Initialize markitdown client lazily."""
        if self._initialized:
            return
            
        try:
            # Import markitdown dynamically to handle optional dependency
            from markitdown import MarkItDown
            
            # Initialize markitdown with configuration
            self._markitdown = MarkItDown()
            
            # Configure markitdown options based on settings
            await self._configure_markitdown()
            
            self._initialized = True
            logger.info("Markitdown service initialized successfully")
            
        except ImportError as e:
            raise ConversionError(
                "Markitdown is not installed. Please install with: pip install markitdown"
            ) from e
        except Exception as e:
            raise ConversionError(f"Failed to initialize markitdown service: {e}") from e
    
    async def _configure_markitdown(self) -> None:
        """Configure markitdown based on settings."""
        if not self._markitdown:
            return
            
        # Configure Azure Document Intelligence if enabled
        if hasattr(self.settings, 'markitdown_use_azure_doc_intel') and self.settings.markitdown_use_azure_doc_intel:
            if hasattr(self.settings, 'markitdown_azure_endpoint') and self.settings.markitdown_azure_endpoint:
                logger.info("Configuring markitdown with Azure Document Intelligence")
                # Note: Azure DI configuration will be implemented in Phase 3

        # Configure LLM-based image description if enabled
        if hasattr(self.settings, 'markitdown_use_llm_image_description') and self.settings.markitdown_use_llm_image_description:
            logger.info("Configuring markitdown with LLM image description")
            # Note: LLM image description will be implemented in Phase 3
    
    async def convert_file(
        self, 
        file_path: Union[str, Path], 
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Convert file to markdown using markitdown.
        
        Args:
            file_path: Path to file to convert
            options: Optional conversion options
            
        Returns:
            Markdown content as string
            
        Raises:
            ConversionError: If conversion fails
            UnsupportedFormatError: If format is not supported
        """
        await self._initialize()
        
        file_path = Path(file_path)
        options = options or {}
        
        # Validate file exists
        if not file_path.exists():
            raise ConversionError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ConversionError(f"Not a file: {file_path}")
        
        try:
            logger.info("Converting file with markitdown", file_path=str(file_path))
            
            # Run markitdown conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._convert_sync, 
                str(file_path), 
                options
            )
            
            if not result or not result.text_content:
                raise ConversionError(f"Markitdown returned empty content for file: {file_path}")
            
            logger.info(
                "File converted successfully", 
                file_path=str(file_path),
                content_length=len(result.text_content)
            )
            
            return result.text_content
            
        except Exception as e:
            if isinstance(e, (ConversionError, UnsupportedFormatError)):
                raise
            logger.error("Markitdown conversion failed", file_path=str(file_path), error=str(e))
            raise ConversionError(f"Failed to convert file with markitdown: {e}") from e
    
    def _convert_sync(self, file_path: str, options: Dict[str, Any]):
        """Synchronous conversion method for thread pool execution."""
        try:
            return self._markitdown.convert(file_path)
        except Exception as e:
            # Check if it's an unsupported format error
            if "not supported" in str(e).lower() or "unsupported" in str(e).lower():
                raise UnsupportedFormatError(f"Format not supported by markitdown: {e}")
            raise ConversionError(f"Markitdown conversion failed: {e}")
    
    async def get_supported_formats(self) -> list[str]:
        """Get list of formats supported by markitdown.
        
        Returns:
            List of supported file extensions
        """
        await self._initialize()
        
        # Markitdown supports these formats (as of version 0.0.1a2)
        return [
            'pdf', 'docx', 'pptx', 'xlsx', 'xls', 'doc', 'ppt',
            'html', 'htm', 'xml', 'csv', 'json', 'txt', 'md',
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp',
            'mp3', 'wav', 'mp4', 'avi', 'mov', 'wmv', 'flv',
            'zip', 'epub', 'ipynb'
        ]
    
    async def supports_format(self, format_type: str) -> bool:
        """Check if format is supported by markitdown.
        
        Args:
            format_type: File format/extension to check
            
        Returns:
            True if format is supported
        """
        supported_formats = await self.get_supported_formats()
        return format_type.lower().lstrip('.') in supported_formats
    
    async def get_conversion_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about what markitdown would extract from a file.
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            Dictionary with conversion information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConversionError(f"File not found: {file_path}")
        
        format_type = file_path.suffix.lower().lstrip('.')
        is_supported = await self.supports_format(format_type)
        
        return {
            'file_path': str(file_path),
            'format': format_type,
            'supported': is_supported,
            'file_size': file_path.stat().st_size,
            'service': 'markitdown'
        }
