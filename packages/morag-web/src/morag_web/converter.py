"""Enhanced Web content to Markdown converter with dynamic content extraction."""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import structlog
from morag_core.exceptions import ProcessingError, ValidationError
from morag_core.interfaces.converter import (
    BaseConverter,
    ConversionOptions,
    ConversionResult,
    QualityScore,
)

from .processor import WebProcessor, WebScrapingConfig
from .web_formatter import WebFormatter

logger = structlog.get_logger(__name__)

try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available, dynamic content extraction disabled")

try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logger.warning("Trafilatura not available, advanced content extraction disabled")

try:
    from readability import Document

    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    logger.warning("Readability not available, content cleaning disabled")

try:
    from newspaper import Article

    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logger.warning("Newspaper3k not available, article extraction disabled")


class WebConverter(BaseConverter):
    """Enhanced Web content to Markdown converter with dynamic content extraction."""

    def __init__(self):
        super().__init__()
        self.name = "Enhanced MoRAG Web Converter"
        self.supported_formats = ["web", "html", "htm", "url"]
        self.formatter = WebFormatter()

        # Initialize extraction methods
        self.extraction_methods = []
        if PLAYWRIGHT_AVAILABLE:
            self.extraction_methods.append("playwright")
        if TRAFILATURA_AVAILABLE:
            self.extraction_methods.append("trafilatura")
        if NEWSPAPER_AVAILABLE:
            self.extraction_methods.append("newspaper")
        if READABILITY_AVAILABLE:
            self.extraction_methods.append("readability")

        # Fallback to basic web processor
        self.extraction_methods.append("basic_web_processor")

        # Initialize web processor
        self.web_processor = WebProcessor()

    def supports_format(self, format_type: str) -> bool:
        """Check if this converter supports the given format."""
        return format_type.lower() in self.supported_formats

    async def convert(
        self, file_path: Union[str, Path], options: ConversionOptions
    ) -> ConversionResult:
        """Convert web content to structured markdown.

        Args:
            file_path: URL or path to HTML file
            options: Conversion options

        Returns:
            ConversionResult with markdown content
        """
        start_time = time.time()

        # Handle both URLs and file paths
        if isinstance(file_path, Path) or (
            isinstance(file_path, str)
            and not file_path.startswith(("http://", "https://"))
        ):
            # Local HTML file
            file_path = Path(file_path)
            await self.validate_input(file_path)
            url = f"file://{file_path.absolute()}"
            is_local_file = True
        else:
            # URL
            url = str(file_path)
            is_local_file = False

        logger.info(
            "Starting web content conversion",
            url=url,
            is_local_file=is_local_file,
            extract_main_content=options.format_options.get(
                "extract_main_content", True
            ),
        )

        try:
            if is_local_file:
                # Read local HTML file
                with open(file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                # Create a simple web result for local files
                web_result = type(
                    "WebResult",
                    (),
                    {
                        "content": html_content,
                        "metadata": {
                            "url": url,
                            "title": "Local HTML File",
                            "filename": file_path.name,
                        },
                        "success": True,
                        "links": [],
                        "images": [],
                    },
                )()
            else:
                # Create web scraping config from conversion options
                web_config = WebScrapingConfig(
                    convert_to_markdown=True,
                    clean_content=options.format_options.get("clean_content", True),
                    extract_links=options.format_options.get("include_links", True),
                    remove_navigation=options.format_options.get(
                        "remove_navigation", True
                    ),
                    remove_footer=options.format_options.get("remove_footer", True),
                )

                # Use web processor
                web_result = await self.web_processor.process_url(url, web_config)

                # Convert WebScrapingResult to expected format
                web_result = type(
                    "WebResult",
                    (),
                    {
                        "content": web_result.content.markdown_content
                        or web_result.content.content,
                        "metadata": {
                            "url": url,
                            "title": web_result.content.title,
                            "extraction_method": "MoRAG Web Processor",
                            **web_result.content.metadata,
                        },
                        "success": web_result.success,
                        "links": [
                            {"url": link, "text": "Link"}
                            for link in web_result.content.links
                        ],
                        "images": [
                            {"src": img, "alt": "Image"}
                            for img in web_result.content.images
                        ],
                    },
                )()

            # Convert to structured markdown
            markdown_content = await self._create_structured_markdown(
                web_result, options
            )

            # Calculate quality score
            quality_score = self._calculate_quality_score(
                markdown_content, web_result.metadata
            )

            processing_time = time.time() - start_time

            result = ConversionResult(
                content=markdown_content,
                metadata=self._enhance_metadata(web_result.metadata, file_path),
                quality_score=quality_score,
                processing_time=processing_time,
                success=web_result.success,
                original_format="web",
                converter_used=self.name,
            )

            logger.info(
                "Web content conversion completed",
                processing_time=processing_time,
                quality_score=quality_score.overall_score,
                word_count=len(markdown_content.split()),
                url=url,
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Web content conversion failed: {str(e)}"

            logger.error(
                "Web content conversion failed",
                error=str(e),
                error_type=type(e).__name__,
                processing_time=processing_time,
                url=url,
            )

            return ConversionResult(
                content="",
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=error_msg,
                original_format="web",
                converter_used=self.name,
            )

    def _calculate_quality_score(self, content: str, metadata: dict) -> QualityScore:
        """Calculate quality score for web conversion."""
        word_count = len(content.split())

        # Base score calculation
        completeness = min(1.0, word_count / 100)  # Assume 100 words is complete
        accuracy = 0.9  # Assume high accuracy for web content
        formatting = 0.8 if "# " in content else 0.5  # Check for headers

        overall_score = (completeness + accuracy + formatting) / 3

        return QualityScore(
            overall_score=overall_score,
            completeness=completeness,
            accuracy=accuracy,
            formatting=formatting,
        )

    async def _create_structured_markdown(
        self, web_result, options: ConversionOptions
    ) -> str:
        """Create structured markdown from web scraping result using formatter."""
        # Clean the raw content
        raw_content = web_result.content if hasattr(web_result, "content") else ""
        cleaned_content = self.formatter.clean_web_content(raw_content)

        # Extract additional metadata from content
        url = web_result.metadata.get("url", "")
        content_metadata = self.formatter.extract_web_metadata(url, cleaned_content)

        # Merge metadata
        combined_metadata = {**web_result.metadata, **content_metadata}

        # Add links to metadata if available
        if hasattr(web_result, "links") and web_result.links:
            combined_metadata["links"] = web_result.links

        # Format content according to LLM documentation specifications
        formatted_content = self.formatter.format_web_content(
            cleaned_content, url, combined_metadata
        )

        return formatted_content

    def _enhance_metadata(
        self, original_metadata: dict, file_path: Union[str, Path]
    ) -> dict:
        """Enhance metadata with additional information."""
        enhanced = original_metadata.copy()

        # Add conversion information
        enhanced.update(
            {"conversion_format": "web_to_markdown", "converter_version": "1.0.0"}
        )

        # Add file information if it's a local file
        if isinstance(file_path, Path):
            enhanced.update(
                {
                    "original_filename": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "file_extension": file_path.suffix.lower(),
                }
            )
        else:
            enhanced["source_url"] = str(file_path)

        return enhanced
