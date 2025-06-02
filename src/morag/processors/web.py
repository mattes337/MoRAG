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
from bs4 import BeautifulSoup, Comment
from markdownify import markdownify

from ..core.exceptions import ProcessingError, ValidationError
from ..services.chunking import ChunkingService
from ..services.content_converter import ContentConverter, ConversionConfig, ConversionOptions
from .document import DocumentChunk

logger = structlog.get_logger(__name__)


@dataclass
class WebScrapingConfig:
    """Configuration for web scraping operations."""
    timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    user_agent: str = "MoRAG/1.0 (+https://github.com/yourusername/morag)"
    max_content_length: int = 10 * 1024 * 1024  # 10MB
    allowed_content_types: List[str] = field(default_factory=lambda: [
        'text/html', 'application/xhtml+xml', 'text/plain'
    ])
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
class WebScrapingResult:
    """Result of web scraping operation."""
    url: str
    content: WebContent
    chunks: List[DocumentChunk]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class WebProcessor:
    """Web content processor for scraping and extracting content."""
    
    def __init__(
        self,
        chunking_service: Optional[ChunkingService] = None,
        content_converter: Optional[ContentConverter] = None
    ):
        """Initialize web processor."""
        self.chunking_service = chunking_service or ChunkingService()
        self.content_converter = content_converter or ContentConverter()
        self._last_request_time = 0.0
        
    def _validate_url(self, url: str) -> str:
        """Validate and normalize URL."""
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")

        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Parse and validate URL
        parsed = urlparse(url)
        if not parsed.netloc or not parsed.scheme:
            raise ValidationError(f"Invalid URL: {url}")

        # Additional validation for domain format
        if '.' not in parsed.netloc or parsed.netloc.startswith('.') or parsed.netloc.endswith('.'):
            raise ValidationError(f"Invalid domain in URL: {url}")

        # Reconstruct URL to normalize it
        return urlunparse(parsed)
    
    async def _rate_limit(self, delay: float) -> None:
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < delay:
            await asyncio.sleep(delay - time_since_last)
        
        self._last_request_time = time.time()
    
    async def _fetch_content(
        self, 
        url: str, 
        config: WebScrapingConfig
    ) -> tuple[str, str, Dict[str, str]]:
        """Fetch content from URL with retries."""
        await self._rate_limit(config.rate_limit_delay)
        
        headers = {
            'User-Agent': config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        async with httpx.AsyncClient(
            timeout=config.timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        ) as client:
            
            for attempt in range(config.max_retries + 1):
                try:
                    logger.info("Fetching URL", url=url, attempt=attempt + 1)
                    
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(ct in content_type for ct in config.allowed_content_types):
                        raise ValidationError(f"Unsupported content type: {content_type}")
                    
                    # Check content length
                    content_length = len(response.content)
                    if content_length > config.max_content_length:
                        raise ValidationError(f"Content too large: {content_length} bytes")
                    
                    return response.text, content_type, dict(response.headers)
                    
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (404, 403, 401):
                        # Don't retry for these errors
                        raise ProcessingError(f"HTTP {e.response.status_code}: {url}")
                    
                    if attempt == config.max_retries:
                        raise ProcessingError(f"HTTP error after {config.max_retries + 1} attempts: {e}")
                    
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
                except (httpx.RequestError, httpx.TimeoutException) as e:
                    if attempt == config.max_retries:
                        raise ProcessingError(f"Network error after {config.max_retries + 1} attempts: {e}")
                    
                    await asyncio.sleep(2 ** attempt)
        
        raise ProcessingError("Failed to fetch content")
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc,
            'extracted_at': time.time()
        }
        
        # Basic meta tags
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Meta description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag:
            metadata['description'] = desc_tag.get('content', '').strip()
        
        # Meta keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            metadata['keywords'] = keywords_tag.get('content', '').strip()
        
        # Author
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag:
            metadata['author'] = author_tag.get('content', '').strip()
        
        # Open Graph data
        og_tags = soup.find_all('meta', attrs={'property': re.compile(r'^og:')})
        for tag in og_tags:
            prop = tag.get('property', '').replace('og:', '')
            content = tag.get('content', '').strip()
            if prop and content:
                metadata[f'og_{prop}'] = content
        
        # Twitter Card data
        twitter_tags = soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')})
        for tag in twitter_tags:
            name = tag.get('name', '').replace('twitter:', '')
            content = tag.get('content', '').strip()
            if name and content:
                metadata[f'twitter_{name}'] = content
        
        return metadata
    
    def _clean_html(self, soup: BeautifulSoup, config: WebScrapingConfig) -> BeautifulSoup:
        """Clean HTML content by removing unwanted elements."""
        # Remove script and style elements
        for element in soup(['script', 'style', 'noscript']):
            element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove navigation elements if configured
        if config.remove_navigation:
            for nav in soup.find_all(['nav', 'header']):
                nav.decompose()
            
            # Remove elements with navigation-related classes/ids
            nav_selectors = [
                '[class*="nav"]', '[id*="nav"]', '[class*="menu"]', 
                '[id*="menu"]', '[class*="sidebar"]', '[id*="sidebar"]'
            ]
            for selector in nav_selectors:
                for element in soup.select(selector):
                    element.decompose()
        
        # Remove footer elements if configured
        if config.remove_footer:
            for footer in soup.find_all('footer'):
                footer.decompose()
        
        return soup
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from the page."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            if href and not href.startswith('#'):
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)
                if absolute_url not in links:
                    links.append(absolute_url)
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all image URLs from the page."""
        images = []
        
        for img in soup.find_all('img', src=True):
            src = img['src'].strip()
            if src:
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, src)
                if absolute_url not in images:
                    images.append(absolute_url)
        
        return images

    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract main content from the page."""
        # Try to find main content area
        main_selectors = [
            'main', 'article', '[role="main"]',
            '.content', '.main-content', '.post-content',
            '#content', '#main-content', '#post-content'
        ]

        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                return main_element

        # If no main content found, try to find the largest text block
        text_elements = soup.find_all(['div', 'section', 'article'])
        if text_elements:
            # Find element with most text content
            best_element = max(text_elements, key=lambda x: len(x.get_text()))
            return best_element

        # Fallback to body
        return soup.find('body') or soup

    async def _convert_to_markdown(self, html_content: str, config: WebScrapingConfig) -> str:
        """Convert HTML content to Markdown using ContentConverter."""
        # Configure conversion options
        conversion_options = ConversionOptions(
            heading_style='ATX',
            bullet_style='-',
            link_style='inline',
            clean_whitespace=config.clean_content
        )

        # Use ContentConverter for advanced conversion
        result = await self.content_converter.html_to_markdown(html_content, conversion_options)

        if result.success:
            return result.content
        else:
            # Fallback to basic markdownify if ContentConverter fails
            logger.warning("ContentConverter failed, falling back to markdownify", error=result.error_message)
            options = {
                'heading_style': 'ATX',
                'bullets': '-',
                'strip': ['script', 'style', 'nav', 'footer'],
            }
            markdown = markdownify(html_content, **options)
            if config.clean_content:
                markdown = self._clean_markdown(markdown)
            return markdown

    def _clean_markdown(self, markdown: str) -> str:
        """Clean up markdown content."""
        # Remove excessive line breaks
        markdown = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown)

        # Remove trailing spaces
        lines = [line.rstrip() for line in markdown.split('\n')]
        markdown = '\n'.join(lines)

        # Remove empty links
        markdown = re.sub(r'\[\]\([^)]*\)', '', markdown)

        # Clean up spacing around headers
        markdown = re.sub(r'\n\n(#+\s)', r'\n\1', markdown)

        return markdown.strip()

    async def process_url(
        self,
        url: str,
        config: Optional[WebScrapingConfig] = None
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
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract metadata
            metadata = self._extract_metadata(soup, normalized_url)

            # Clean HTML
            cleaned_soup = self._clean_html(soup, config)

            # Extract main content
            main_content = self._extract_main_content(cleaned_soup)

            # Extract text content
            text_content = main_content.get_text(separator='\n', strip=True)

            # Convert to markdown if requested
            markdown_content = ""
            if config.convert_to_markdown:
                markdown_content = await self._convert_to_markdown(str(main_content), config)

            # Extract links and images
            links = []
            images = []
            if config.extract_links:
                links = self._extract_links(soup, normalized_url)
                images = self._extract_images(soup, normalized_url)

            # Create web content object
            web_content = WebContent(
                url=normalized_url,
                title=metadata.get('title', ''),
                content=text_content,
                markdown_content=markdown_content,
                metadata=metadata,
                links=links,
                images=images,
                extraction_time=time.time() - start_time,
                content_length=len(text_content),
                content_type=content_type
            )

            # Create chunks using the chunking service
            chunks = []
            if text_content.strip():
                chunk_infos = await self.chunking_service.chunk_with_metadata(text_content)

                # Convert ChunkInfo objects to DocumentChunk objects
                for i, chunk_info in enumerate(chunk_infos):
                    chunk = DocumentChunk(
                        text=chunk_info.text,
                        chunk_type=chunk_info.chunk_type,
                        page_number=1,  # Web pages are single page
                        element_id=f"web_chunk_{i}",
                        metadata={
                            **metadata,
                            'source_type': 'web',
                            'url': normalized_url,
                            'content_type': content_type,
                            'start_char': chunk_info.start_char,
                            'end_char': chunk_info.end_char,
                            'sentence_count': chunk_info.sentence_count,
                            'word_count': chunk_info.word_count,
                            'entities': chunk_info.entities,
                            'topics': chunk_info.topics
                        }
                    )
                    chunks.append(chunk)

            processing_time = time.time() - start_time

            logger.info(
                "Web content processing completed",
                url=normalized_url,
                content_length=len(text_content),
                chunks_created=len(chunks),
                processing_time=processing_time
            )

            return WebScrapingResult(
                url=normalized_url,
                content=web_content,
                chunks=chunks,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process URL {url}: {str(e)}"

            logger.error(
                "Web content processing failed",
                url=url,
                error=str(e),
                processing_time=processing_time
            )

            return WebScrapingResult(
                url=url,
                content=None,
                chunks=[],
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )

    async def process_urls(
        self,
        urls: List[str],
        config: Optional[WebScrapingConfig] = None
    ) -> List[WebScrapingResult]:
        """Process multiple URLs concurrently."""
        config = config or WebScrapingConfig()

        logger.info("Starting batch web content processing", url_count=len(urls))

        # Process URLs with controlled concurrency
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def process_with_semaphore(url: str) -> WebScrapingResult:
            async with semaphore:
                return await self.process_url(url, config)

        results = await asyncio.gather(
            *[process_with_semaphore(url) for url in urls],
            return_exceptions=True
        )

        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(WebScrapingResult(
                    url=urls[i],
                    content=None,
                    chunks=[],
                    processing_time=0.0,
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)

        successful = len([r for r in processed_results if r.success])
        logger.info(
            "Batch web content processing completed",
            total_urls=len(urls),
            successful=successful,
            failed=len(urls) - successful
        )

        return processed_results
