"""Web service for high-level operations."""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio
import structlog

from morag_core.exceptions import ProcessingError
from morag_core.interfaces.service import BaseService
from morag_core.models.document import DocumentChunk

from .processor import WebProcessor, WebScrapingConfig, WebScrapingResult

logger = structlog.get_logger(__name__)


class WebService(BaseService):
    """Web service for high-level operations."""

    def __init__(self):
        """Initialize web service."""
        self.processor = WebProcessor()

    async def initialize(self) -> bool:
        """Initialize the service.

        Returns:
            True if initialization was successful
        """
        return True

    async def shutdown(self) -> None:
        """Shutdown the service and release resources."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Dictionary with health status information
        """
        return {"status": "healthy", "processor": "ready"}
    
    async def process_url(self, url: str, config_options: Optional[Dict[str, Any]] = None) -> WebScrapingResult:
        """Process a single URL with the given configuration options.
        
        Args:
            url: URL to process
            config_options: Optional configuration options
            
        Returns:
            WebScrapingResult with processed content
        """
        # Create config from options
        config = WebScrapingConfig()
        if config_options:
            for key, value in config_options.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return await self.processor.process_url(url, config)
    
    async def process_multiple_urls(
        self,
        urls: List[str],
        config_options: Optional[Dict[str, Any]] = None,
        concurrency_limit: int = 5
    ) -> Dict[str, WebScrapingResult]:
        """Process multiple URLs with concurrency limit.
        
        Args:
            urls: List of URLs to process
            config_options: Optional configuration options
            concurrency_limit: Maximum number of concurrent requests
            
        Returns:
            Dictionary mapping URLs to their processing results
        """
        # Create config from options
        config = WebScrapingConfig()
        if config_options:
            for key, value in config_options.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Process URLs with concurrency limit
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def process_with_semaphore(url: str) -> tuple[str, WebScrapingResult]:
            async with semaphore:
                result = await self.processor.process_url(url, config)
                return url, result
        
        # Create tasks for all URLs
        tasks = [process_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        output = {}
        for url, result in results:
            if isinstance(result, Exception):
                # Handle exceptions
                logger.error("Error processing URL", url=url, error=str(result))
                output[url] = WebScrapingResult(
                    url=url,
                    content=None,  # type: ignore
                    chunks=[],
                    processing_time=0.0,
                    success=False,
                    error_message=f"Error: {str(result)}"
                )
            else:
                output[url] = result
        
        return output
    
    async def extract_content_from_url(self, url: str) -> str:
        """Extract main content from URL as plain text.
        
        Args:
            url: URL to process
            
        Returns:
            Extracted text content
            
        Raises:
            ProcessingError: If processing fails
        """
        result = await self.process_url(url)
        
        if not result.success or not result.content:
            raise ProcessingError(f"Failed to extract content: {result.error_message}")
        
        return result.content.content
    
    async def extract_markdown_from_url(self, url: str) -> str:
        """Extract content from URL as markdown.
        
        Args:
            url: URL to process
            
        Returns:
            Extracted markdown content
            
        Raises:
            ProcessingError: If processing fails
        """
        result = await self.process_url(
            url,
            {"convert_to_markdown": True}
        )
        
        if not result.success or not result.content:
            raise ProcessingError(f"Failed to extract markdown: {result.error_message}")
        
        return result.content.markdown_content
    
    async def extract_metadata_from_url(self, url: str) -> Dict[str, Any]:
        """Extract metadata from URL.
        
        Args:
            url: URL to process
            
        Returns:
            Extracted metadata
            
        Raises:
            ProcessingError: If processing fails
        """
        result = await self.process_url(url)
        
        if not result.success or not result.content:
            raise ProcessingError(f"Failed to extract metadata: {result.error_message}")
        
        return result.content.metadata
    
    async def extract_links_from_url(self, url: str) -> List[str]:
        """Extract links from URL.
        
        Args:
            url: URL to process
            
        Returns:
            List of extracted links
            
        Raises:
            ProcessingError: If processing fails
        """
        result = await self.process_url(
            url,
            {"extract_links": True}
        )
        
        if not result.success or not result.content:
            raise ProcessingError(f"Failed to extract links: {result.error_message}")
        
        return result.content.links
    
    async def extract_images_from_url(self, url: str) -> List[str]:
        """Extract image URLs from URL.
        
        Args:
            url: URL to process
            
        Returns:
            List of extracted image URLs
            
        Raises:
            ProcessingError: If processing fails
        """
        result = await self.process_url(url)
        
        if not result.success or not result.content:
            raise ProcessingError(f"Failed to extract images: {result.error_message}")
        
        return result.content.images