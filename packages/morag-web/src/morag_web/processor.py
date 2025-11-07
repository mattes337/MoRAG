"""Web content processing and scraping."""

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
import structlog
from bs4 import BeautifulSoup, Comment, PageElement, Tag
from markdownify import markdownify
from morag_core.exceptions import ProcessingError, ValidationError
from morag_core.interfaces.processor import (
    BaseProcessor,
    ProcessingConfig,
    ProcessingResult,
)
from morag_core.models.document import DocumentChunk

logger = structlog.get_logger(__name__)


@dataclass
class WebScrapingConfig(ProcessingConfig):
    """Configuration for web scraping operations."""

    timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    user_agent: str = "MoRAG/1.0 (+https://github.com/yourusername/morag)"
    max_content_length: int = 10 * 1024 * 1024  # 10MB
    allowed_content_types: List[str] = field(
        default_factory=lambda: ["text/html", "application/xhtml+xml", "text/plain"]
    )
    extract_links: bool = True
    convert_to_markdown: bool = True
    clean_content: bool = True
    remove_navigation: bool = True
    remove_footer: bool = True
    preserve_tables: bool = True
    preserve_lists: bool = True


@dataclass
class WebContent:
    """Extracted web content."""

    url: str
    title: str
    content: str
    markdown_content: str
    metadata: Dict[str, Any]
    links: List[str]
    images: List[str]
    extraction_time: float
    content_length: int
    content_type: str


@dataclass
class WebScrapingResult(ProcessingResult):
    """Result of web scraping operation."""

    url: Optional[str] = None
    content: Optional[WebContent] = None
    chunks: List[DocumentChunk] = field(default_factory=list)


class WebProcessor(BaseProcessor):
    """Web content processor for scraping and extracting content."""

    def __init__(self):
        """Initialize web processor."""
        self._last_request_time = 0.0

    def _validate_url(self, url: str) -> str:
        """Validate and normalize URL."""
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")

        # Add protocol if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Parse and validate URL
        parsed = urlparse(url)
        if not parsed.netloc or not parsed.scheme:
            raise ValidationError(f"Invalid URL: {url}")

        # Additional validation for domain format
        if (
            "." not in parsed.netloc
            or parsed.netloc.startswith(".")
            or parsed.netloc.endswith(".")
        ):
            raise ValidationError(f"Invalid domain in URL: {url}")

        # Reconstruct URL to normalize it
        return urlunparse(parsed)

    async def _rate_limit(self, config: WebScrapingConfig) -> None:
        """Apply rate limiting to avoid overloading servers."""
        if config.rate_limit_delay > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < config.rate_limit_delay:
                delay = config.rate_limit_delay - elapsed
                logger.debug(f"Rate limiting: waiting {delay:.2f}s")
                await asyncio.sleep(delay)

        self._last_request_time = time.time()

    async def _fetch_content(
        self, url: str, config: WebScrapingConfig
    ) -> tuple[str, str, Dict[str, str]]:
        """Fetch content from URL with retries."""
        headers = {
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            # Remove explicit Accept-Encoding to let httpx handle compression automatically
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

        retry_count = 0
        last_error = None

        while retry_count <= config.max_retries:
            try:
                # Apply rate limiting
                await self._rate_limit(config)

                async with httpx.AsyncClient(
                    timeout=config.timeout, follow_redirects=True
                ) as client:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()

                    # Check content type
                    content_type = (
                        response.headers.get("content-type", "")
                        .split(";")[0]
                        .strip()
                        .lower()
                    )
                    if content_type not in config.allowed_content_types:
                        raise ProcessingError(
                            f"Unsupported content type: {content_type}. "
                            f"Allowed types: {', '.join(config.allowed_content_types)}"
                        )

                    # Check content length
                    content_length = len(response.content)
                    if content_length > config.max_content_length:
                        raise ProcessingError(
                            f"Content too large: {content_length} bytes "
                            f"(max: {config.max_content_length})"
                        )

                    # Handle text decoding more robustly
                    try:
                        # Log response details for debugging
                        logger.debug(
                            "Response details",
                            url=url,
                            status_code=response.status_code,
                            content_encoding=response.headers.get("content-encoding"),
                            content_type=response.headers.get("content-type"),
                            content_length=len(response.content),
                        )

                        # First try the response's automatic text decoding
                        text_content = response.text

                        # Validate that the content is actually readable text
                        # Check for excessive binary characters which indicate corruption
                        binary_chars = sum(
                            1
                            for c in text_content[:1000]
                            if ord(c) < 32 and c not in "\n\r\t"
                        )
                        if (
                            binary_chars > 50
                        ):  # More than 50 binary chars in first 1000 suggests corruption
                            logger.warning(
                                f"Content appears corrupted, binary chars: {binary_chars}",
                                url=url,
                            )
                            raise UnicodeDecodeError(
                                "corruption", b"", 0, 1, "Content appears corrupted"
                            )

                        return text_content, content_type, dict(response.headers)

                    except UnicodeDecodeError as e:
                        logger.warning(f"Text decoding failed: {e}", url=url)
                        # If automatic decoding fails, try different encodings
                        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
                            try:
                                text_content = response.content.decode(
                                    encoding, errors="ignore"
                                )
                                logger.warning(
                                    f"Used fallback encoding: {encoding}", url=url
                                )

                                # Validate the fallback content too
                                binary_chars = sum(
                                    1
                                    for c in text_content[:1000]
                                    if ord(c) < 32 and c not in "\n\r\t"
                                )
                                if binary_chars > 50:
                                    continue  # Try next encoding

                                return (
                                    text_content,
                                    content_type,
                                    dict(response.headers),
                                )
                            except UnicodeDecodeError:
                                continue

                        # If all encodings fail, raise an error
                        raise ProcessingError(f"Could not decode content from {url}")

            except httpx.HTTPStatusError as e:
                last_error = ProcessingError(
                    f"HTTP error: {e.response.status_code} - {e.response.reason_phrase}"
                )
            except httpx.RequestError as e:
                last_error = ProcessingError(f"Request error: {str(e)}")
            except Exception as e:
                last_error = ProcessingError(f"Unexpected error: {str(e)}")

            retry_count += 1
            if retry_count <= config.max_retries:
                wait_time = 2**retry_count  # Exponential backoff
                logger.warning(
                    f"Fetch failed, retrying ({retry_count}/{config.max_retries}) "
                    f"in {wait_time}s",
                    url=url,
                    error=str(last_error),
                )
                await asyncio.sleep(wait_time)

        # If we get here, all retries failed
        raise last_error or ProcessingError("Failed to fetch content after retries")

    def _clean_html(
        self, soup: BeautifulSoup, config: WebScrapingConfig
    ) -> BeautifulSoup:
        """Clean HTML content by removing unwanted elements."""
        # Make a copy to avoid modifying the original
        soup = BeautifulSoup(str(soup), "lxml")

        # Remove script and style elements
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove navigation elements if configured
        if config.remove_navigation:
            nav_selectors = [
                "nav",
                "header",
                ".navigation",
                ".nav",
                ".navbar",
                "#nav",
                "#menu",
            ]
            for selector in nav_selectors:
                for element in soup.select(selector):
                    element.decompose()

        # Remove footer elements if configured
        if config.remove_footer:
            footer_selectors = ["footer", ".footer", "#footer"]
            for selector in footer_selectors:
                for element in soup.select(selector):
                    element.decompose()

        # Remove other common non-content elements
        noise_selectors = [
            ".ad",
            ".ads",
            ".advertisement",
            ".banner",
            ".cookie-notice",
            ".popup",
            ".modal",
            ".sidebar",
            ".widget",
            ".social-media",
            ".share",
            ".sharing",
            ".related-posts",
            ".comments",
            "#comments",
            ".comment-form",
        ]

        for selector in noise_selectors:
            for element in soup.select(selector):
                element.decompose()

        return soup

    def _extract_main_content(
        self, soup: BeautifulSoup
    ) -> Union[Tag, PageElement, BeautifulSoup]:
        """Extract main content from the page."""
        # Try to find main content area
        main_selectors = [
            "main",
            "article",
            '[role="main"]',
            ".content",
            ".main-content",
            ".post-content",
            "#content",
            "#main-content",
            "#post-content",
        ]

        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                return main_element

        # If no main content found, try to find the largest text block
        text_elements = soup.find_all(["div", "section", "article"])
        if text_elements:
            # Find element with most text content
            best_element = max(text_elements, key=lambda x: len(x.get_text()))
            return best_element

        # If all else fails, return the body
        body = soup.find("body")
        return body or soup

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {
            "url": url,
            "domain": urlparse(url).netloc,
        }

        # Extract title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()

        # Extract meta tags
        for meta in soup.find_all("meta"):
            if isinstance(meta, Tag):
                name_attr = meta.get("name", "")
                property_attr = meta.get("property", "")
                content_attr = meta.get("content", "")

                name = str(name_attr).lower() if name_attr else ""
                property_name = str(property_attr).lower() if property_attr else ""
                content = str(content_attr) if content_attr else ""

                if name and content:
                    metadata[f"meta_{name}"] = content
                elif property_name and content:
                    metadata[f"meta_{property_name}"] = content

        # Extract Open Graph metadata
        for meta in soup.find_all("meta", property=re.compile(r"^og:")):
            if isinstance(meta, Tag):
                property_attr = meta.get("property", "")
                content_attr = meta.get("content", "")

                property_name = (
                    str(property_attr).lower().replace("og:", "og_")
                    if property_attr
                    else ""
                )
                content = str(content_attr) if content_attr else ""

                if property_name and content:
                    metadata[property_name] = content

        # Extract Twitter Card metadata
        for meta in soup.find_all("meta", attrs={"name": re.compile(r"^twitter:")}):
            if isinstance(meta, Tag):
                name_attr = meta.get("name", "")
                content_attr = meta.get("content", "")

                name = (
                    str(name_attr).lower().replace("twitter:", "twitter_")
                    if name_attr
                    else ""
                )
                content = str(content_attr) if content_attr else ""

                if name and content:
                    metadata[name] = content

        return metadata

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and normalize links from HTML."""
        links = []

        for a_tag in soup.find_all("a", href=True):
            if isinstance(a_tag, Tag):
                href_attr = a_tag.get("href", "")
                href = str(href_attr).strip() if href_attr else ""

            # Skip empty links and javascript/mailto links
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue

            # Normalize relative URLs
            try:
                absolute_url = urljoin(base_url, href)
                links.append(absolute_url)
            except Exception:
                # Skip malformed URLs
                continue

        return list(set(links))  # Remove duplicates

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract and normalize image URLs from HTML."""
        images = []

        for img_tag in soup.find_all("img", src=True):
            if isinstance(img_tag, Tag):
                src_attr = img_tag.get("src", "")
                src = str(src_attr).strip() if src_attr else ""

            # Skip empty sources and data URIs
            if not src or src.startswith("data:"):
                continue

            # Normalize relative URLs
            try:
                absolute_url = urljoin(base_url, src)
                images.append(absolute_url)
            except Exception:
                # Skip malformed URLs
                continue

        return list(set(images))  # Remove duplicates

    def _convert_to_markdown(self, html: str, config: WebScrapingConfig) -> str:
        """Convert HTML to Markdown using MarkItDown for better content extraction."""
        try:
            from io import BytesIO

            from markitdown import MarkItDown

            # Initialize MarkItDown
            md_converter = MarkItDown()

            # Create a BytesIO object to simulate a file-like object for HTML content
            html_bytes = html.encode("utf-8")
            html_stream = BytesIO(html_bytes)

            # Convert HTML to markdown using MarkItDown
            # MarkItDown automatically handles content extraction and removes clutter
            result = md_converter.convert_stream(html_stream, file_extension=".html")

            return result.text_content if result and result.text_content else ""

        except Exception as e:
            logger.warning(
                f"MarkItDown conversion failed, falling back to basic conversion: {e}"
            )

            # Fallback to basic HTML cleaning if MarkItDown fails
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script, style, nav, header, footer, and other clutter tags
            for tag in soup(
                ["script", "style", "nav", "header", "footer", "aside", "menu"]
            ):
                tag.decompose()

            # Try to find main content area
            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.find("div", class_=["content", "main", "article"])
            )
            if main_content:
                soup = main_content

            # Extract text content
            return soup.get_text(separator="\n", strip=True)

    def _create_chunks(
        self, content: str, metadata: Dict[str, Any], document_id: str
    ) -> List[DocumentChunk]:
        """Create document chunks from content."""
        # Simple chunking by paragraphs for now
        paragraphs = [p for p in content.split("\n\n") if p.strip()]

        chunks = []
        for i, paragraph in enumerate(paragraphs):
            chunk = DocumentChunk(
                document_id=document_id,
                content=paragraph,
                metadata={**metadata, "chunk_index": i, "chunk_type": "paragraph"},
            )
            chunks.append(chunk)

        return chunks

    async def process(
        self, file_path: Union[str, Path], config: Optional[ProcessingConfig] = None
    ) -> ProcessingResult:
        """Process content from file.

        This method is required by the BaseProcessor interface but is not applicable
        for web processing. Use process_url instead.

        Raises:
            ProcessingError: Always raises this error as this method is not supported
        """
        raise ProcessingError(
            "WebProcessor does not support file processing. Use process_url instead."
        )

    def supports_format(self, format_type: str) -> bool:
        """Check if processor supports the given format.

        Args:
            format_type: Format type to check

        Returns:
            True if format is supported, False otherwise
        """
        return format_type.lower() in ["html", "htm", "xhtml", "web"]

    async def process_url(
        self, url: str, config: Optional[WebScrapingConfig] = None
    ) -> WebScrapingResult:
        """Process a single URL and extract content."""
        start_time = time.time()
        config = config or WebScrapingConfig()

        try:
            # Validate URL
            normalized_url = self._validate_url(url)

            logger.info("Starting web content processing", url=normalized_url)

            # Fetch content
            html_content, content_type, headers = await self._fetch_content(
                normalized_url, config
            )

            # Parse HTML
            soup = BeautifulSoup(html_content, "lxml")

            # Extract metadata
            metadata = self._extract_metadata(soup, normalized_url)

            # Clean HTML if configured
            if config.clean_content:
                soup = self._clean_html(soup, config)

            # Extract main content
            main_content = self._extract_main_content(soup)

            # Extract links if configured
            links = []
            if config.extract_links:
                links = self._extract_links(soup, normalized_url)

            # Extract images
            images = self._extract_images(soup, normalized_url)

            # Convert to markdown if configured
            markdown_content = ""
            if config.convert_to_markdown:
                markdown_content = self._convert_to_markdown(str(main_content), config)

            # Create content object
            extraction_time = time.time() - start_time
            content = WebContent(
                url=normalized_url,
                title=metadata.get("title", "") or metadata.get("meta_og_title", ""),
                content=main_content.get_text("\n", strip=True),
                markdown_content=markdown_content,
                metadata=metadata,
                links=links,
                images=images,
                extraction_time=extraction_time,
                content_length=len(html_content),
                content_type=content_type,
            )

            # Create chunks
            chunks = self._create_chunks(content.content, metadata, normalized_url)

            # Calculate total processing time
            processing_time = time.time() - start_time

            return WebScrapingResult(
                url=normalized_url,
                content=content,
                chunks=chunks,
                processing_time=processing_time,
                success=True,
            )

        except (ValidationError, ProcessingError) as e:
            # Known errors
            logger.error("Web processing error", url=url, error=str(e))
            processing_time = time.time() - start_time
            return WebScrapingResult(
                url=url,
                content=None,
                chunks=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e),
            )
        except Exception as e:
            # Unexpected errors
            logger.exception("Unexpected error in web processing", url=url)
            processing_time = time.time() - start_time
            return WebScrapingResult(
                url=url,
                content=None,
                chunks=[],
                processing_time=processing_time,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )
