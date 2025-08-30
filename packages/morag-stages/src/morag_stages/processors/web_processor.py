"""Web processor wrapper for stage processing."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import structlog

from .interface import StageProcessor, ProcessorResult

logger = structlog.get_logger(__name__)

# Import core exceptions
try:
    from morag_core.exceptions import ProcessingError
except ImportError:
    class ProcessingError(Exception):  # type: ignore
        pass


class WebStageProcessor(StageProcessor):
    """Stage processor for web content using morag_web package."""
    
    def __init__(self):
        """Initialize web stage processor."""
        self._web_processor = None
        self._services = None
    
    def _get_web_processor(self):
        """Get or create web processor instance."""
        if self._web_processor is None:
            try:
                from morag_web import WebProcessor
                self._web_processor = WebProcessor()
            except ImportError as e:
                raise ProcessingError(f"Web processor not available: {e}")
        return self._web_processor
    
    def _get_services(self):
        """Get MoRAG services for web processing."""
        if self._services is None:
            try:
                from morag_services import MoRAGServices
                self._services = MoRAGServices()
            except ImportError as e:
                raise ProcessingError(f"MoRAG services not available: {e}")
        return self._services
    
    def supports_content_type(self, content_type: str) -> bool:
        """Check if this processor supports the given content type."""
        return content_type.upper() == "WEB"
    
    async def process(
        self, 
        input_file: Path, 
        output_file: Path, 
        config: Dict[str, Any]
    ) -> ProcessorResult:
        """Process web URL to markdown."""
        # Convert Path back to URL string
        url = str(input_file)
        if url.startswith(('C:', 'D:', '/', '\\')):
            # Handle Windows path conversion issues
            url = url.replace('\\', '/')
            if not url.startswith(('http://', 'https://')):
                raise ProcessingError(f"Invalid web URL: {url}")
        
        logger.info("Processing web URL", url=url)
        
        try:
            # Try using morag_web processor first
            try:
                processor = self._get_web_processor()
                
                # Convert config to WebConfig
                from morag_web import WebConfig
                web_config = WebConfig(
                    extract_links=config.get('extract_links', False),
                    follow_links=config.get('follow_links', False),
                    max_depth=config.get('max_depth', 1),
                    clean_content=config.get('clean_content', True),
                    convert_to_markdown=config.get('convert_to_markdown', True),
                    timeout=config.get('timeout', 30)
                )
                
                result = await processor.process_url(url, web_config)
                
                metadata = {
                    "title": result.content.title,
                    "source": url,
                    "type": "web",
                    "url": url,
                    "content_type": result.content.content_type,
                    "content_length": result.content.content_length,
                    "extraction_time": result.content.extraction_time,
                    "created_at": datetime.now().isoformat(),
                    **result.content.metadata
                }
                
                # Use markdown content if available, otherwise plain content
                content = result.content.markdown_content or result.content.content
                
                markdown_content = self.create_markdown_with_metadata(content, metadata)
                output_file.write_text(markdown_content, encoding='utf-8')
                
                return ProcessorResult(
                    content=content,
                    metadata=metadata,
                    metrics={
                        "url": url,
                        "content_length": len(content),
                        "links_followed": config.get('follow_links', False)
                    },
                    final_output_file=output_file
                )
                
            except Exception as web_error:
                logger.warning("morag_web processor failed, trying MoRAG services", error=str(web_error))
                
                # Fallback to MoRAG services
                services = self._get_services()
                
                # Prepare options for web service
                options = {
                    'extract_links': config.get('extract_links', False),
                    'follow_links': config.get('follow_links', False),
                    'max_depth': config.get('max_depth', 1),
                    'timeout': config.get('timeout', 30)
                }
                
                # Use web service
                result = await services.process_web(url, options)
                
                metadata = {
                    "title": result.metadata.get('title', "Web Content"),
                    "source": url,
                    "type": "web",
                    "url": url,
                    "created_at": datetime.now().isoformat(),
                    **result.metadata
                }
                
                content = result.text_content or ""
                markdown_content = self.create_markdown_with_metadata(content, metadata)
                output_file.write_text(markdown_content, encoding='utf-8')
                
                return ProcessorResult(
                    content=content,
                    metadata=metadata,
                    metrics={
                        "url": url,
                        "content_length": len(content),
                        "links_followed": config.get('follow_links', False)
                    },
                    final_output_file=output_file
                )
                
        except Exception as e:
            logger.error("Web processing failed", url=url, error=str(e))
            raise ProcessingError(f"Web processing failed for {url}: {e}")
