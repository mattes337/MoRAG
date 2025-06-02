"""Advanced content conversion service for HTML to Markdown and vice versa."""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import bleach
import html2text
import markdown
import structlog
from bs4 import BeautifulSoup, Comment, NavigableString
from markdownify import MarkdownConverter

from ..core.exceptions import ProcessingError, ValidationError

logger = structlog.get_logger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for content conversion operations."""
    preserve_tables: bool = True
    preserve_code_blocks: bool = True
    preserve_images: bool = True
    reference_style_links: bool = False
    clean_whitespace: bool = True
    remove_empty_elements: bool = True
    custom_selectors: Dict[str, str] = field(default_factory=dict)
    sanitize_html: bool = True
    allowed_tags: List[str] = field(default_factory=lambda: [
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'p', 'br', 'div', 'span',
        'strong', 'b', 'em', 'i', 'u', 's', 'del', 'ins',
        'ul', 'ol', 'li', 'dl', 'dt', 'dd',
        'table', 'thead', 'tbody', 'tr', 'th', 'td',
        'blockquote', 'pre', 'code',
        'a', 'img', 'figure', 'figcaption',
        'hr', 'sup', 'sub'
    ])
    allowed_attributes: Dict[str, List[str]] = field(default_factory=lambda: {
        'a': ['href', 'title'],
        'img': ['src', 'alt', 'title', 'width', 'height'],
        'table': ['class'],
        'th': ['scope', 'colspan', 'rowspan'],
        'td': ['colspan', 'rowspan'],
        'code': ['class'],
        'pre': ['class']
    })


@dataclass
class ConversionOptions:
    """Options for specific conversion operations."""
    heading_style: str = "ATX"  # ATX (#) or SETEXT (===)
    bullet_style: str = "-"     # -, *, +
    code_fence_style: str = "```"  # ``` or ~~~
    emphasis_style: str = "*"   # * or _
    strong_style: str = "**"    # ** or __
    link_style: str = "inline"  # inline or reference
    wrap_width: int = 0  # 0 = no wrapping
    escape_misc: bool = True
    body_width: int = 0
    unicode_snob: bool = True
    ignore_links: bool = False
    ignore_images: bool = False
    ignore_emphasis: bool = False
    mark_code: bool = True


@dataclass
class ConversionResult:
    """Result of a content conversion operation."""
    content: str
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class CustomMarkdownConverter(MarkdownConverter):
    """Custom MarkdownConverter with enhanced features."""
    
    def __init__(self, **options):
        super().__init__(**options)
        self.options = options
    
    def convert_table(self, el, text, parent_tags):
        """Enhanced table conversion with better formatting."""
        if not self.options.get('preserve_tables', True):
            return super().convert_table(el, text, parent_tags)
        
        # Extract table structure
        rows = []
        headers = []
        
        # Find header row
        thead = el.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text(strip=True))
        
        # Find body rows
        tbody = el.find('tbody') or el
        for tr in tbody.find_all('tr'):
            if tr.parent.name == 'thead':
                continue
            row = []
            for td in tr.find_all(['td', 'th']):
                row.append(td.get_text(strip=True))
            if row:
                rows.append(row)
        
        if not headers and rows:
            headers = [f"Column {i+1}" for i in range(len(rows[0]))]
        
        if not headers or not rows:
            return super().convert_table(el, text, parent_tags)
        
        # Build markdown table
        result = []
        
        # Header row
        result.append('| ' + ' | '.join(headers) + ' |')
        
        # Separator row
        result.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        
        # Data rows
        for row in rows:
            # Pad row to match header length
            while len(row) < len(headers):
                row.append('')
            result.append('| ' + ' | '.join(row[:len(headers)]) + ' |')
        
        return '\n' + '\n'.join(result) + '\n\n'
    
    def convert_pre(self, el, text, parent_tags):
        """Enhanced code block conversion."""
        if not self.options.get('preserve_code_blocks', True):
            return super().convert_pre(el, text, parent_tags)
        
        code_el = el.find('code')
        if code_el:
            # Extract language from class attribute
            lang = ''
            class_attr = code_el.get('class', [])
            if isinstance(class_attr, list):
                for cls in class_attr:
                    if cls.startswith('language-'):
                        lang = cls.replace('language-', '')
                        break
                    elif cls.startswith('lang-'):
                        lang = cls.replace('lang-', '')
                        break
            
            fence = self.options.get('code_fence_style', '```')
            code_text = code_el.get_text()
            return f'\n{fence}{lang}\n{code_text}\n{fence}\n\n'
        
        return super().convert_pre(el, text, parent_tags)
    
    def convert_img(self, el, text, parent_tags):
        """Enhanced image conversion with figure support."""
        if not self.options.get('preserve_images', True):
            return ''
        
        src = el.get('src', '')
        alt = el.get('alt', '')
        title = el.get('title', '')
        
        # Check if image is inside a figure with caption
        figure = el.find_parent('figure')
        if figure:
            figcaption = figure.find('figcaption')
            if figcaption:
                caption = figcaption.get_text(strip=True)
                if caption and not alt:
                    alt = caption
        
        if title:
            return f'![{alt}]({src} "{title}")'
        else:
            return f'![{alt}]({src})'


class ContentConverter:
    """Advanced content converter for HTML to Markdown and vice versa."""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """Initialize content converter."""
        self.config = config or ConversionConfig()
        
        # Initialize html2text converter
        self.html2text_converter = html2text.HTML2Text()
        self._configure_html2text()
        
        # Initialize markdown converter
        self.markdown_converter = markdown.Markdown(
            extensions=['tables', 'fenced_code', 'toc']
        )
    
    def _configure_html2text(self):
        """Configure html2text converter with default settings."""
        h = self.html2text_converter
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = 0  # No line wrapping
        h.unicode_snob = True
        h.mark_code = True
        # Only set attributes that exist
        if hasattr(h, 'wrap_links'):
            h.wrap_links = False
        if hasattr(h, 'inline_links'):
            h.inline_links = True
        if hasattr(h, 'protect_links'):
            h.protect_links = True
        if hasattr(h, 'skip_internal_links'):
            h.skip_internal_links = False
    
    def _sanitize_html(self, html: str) -> str:
        """Sanitize HTML content to remove potentially harmful elements."""
        if not self.config.sanitize_html:
            return html
        
        return bleach.clean(
            html,
            tags=self.config.allowed_tags,
            attributes=self.config.allowed_attributes,
            strip=True
        )
    
    def _clean_content(self, content: str, content_type: str) -> str:
        """Clean and normalize content."""
        if not self.config.clean_whitespace:
            return content
        
        if content_type == 'markdown':
            # Clean markdown-specific issues
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Remove excessive line breaks
            content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)  # Remove trailing spaces
            content = re.sub(r'\[\]\([^)]*\)', '', content)  # Remove empty links
            content = re.sub(r'\n\n(#+\s)', r'\n\1', content)  # Clean spacing around headers
        elif content_type == 'html':
            # Clean HTML-specific issues
            content = re.sub(r'>\s+<', '><', content)  # Remove whitespace between tags
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        
        return content.strip()

    def extract_main_content(self, html: str, selectors: Optional[List[str]] = None) -> str:
        """Extract main content from HTML using CSS selectors."""
        soup = BeautifulSoup(html, 'lxml')

        # Use custom selectors if provided
        if selectors:
            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    return str(element)

        # Default main content selectors
        main_selectors = [
            'main', 'article', '[role="main"]',
            '.content', '.main-content', '.post-content', '.entry-content',
            '#content', '#main-content', '#post-content', '#entry-content'
        ]

        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                return str(element)

        # Remove navigation, header, footer, sidebar elements
        for unwanted in soup.select('nav, header, footer, aside, .sidebar, .navigation'):
            unwanted.decompose()

        # Find the largest text block
        text_elements = soup.find_all(['div', 'section', 'article'])
        if text_elements:
            best_element = max(text_elements, key=lambda x: len(x.get_text()))
            return str(best_element)

        # Fallback to body
        body = soup.find('body')
        return str(body) if body else html

    async def html_to_markdown(
        self,
        html: str,
        options: Optional[ConversionOptions] = None
    ) -> ConversionResult:
        """Convert HTML to Markdown with advanced options."""
        start_time = time.time()
        options = options or ConversionOptions()
        warnings = []

        try:
            logger.info("Starting HTML to Markdown conversion")

            # Sanitize HTML if configured
            if self.config.sanitize_html:
                html = self._sanitize_html(html)

            # Extract main content if configured
            if self.config.custom_selectors:
                html = self.extract_main_content(html, list(self.config.custom_selectors.values()))

            # Choose conversion method based on options
            if options.link_style == "reference" or self.config.reference_style_links:
                # Use html2text for reference-style links
                markdown_content = await self._convert_with_html2text(html, options)
            else:
                # Use markdownify for inline links
                markdown_content = await self._convert_with_markdownify(html, options)

            # Clean the content
            markdown_content = self._clean_content(markdown_content, 'markdown')

            # Extract metadata
            metadata = self._extract_conversion_metadata(html, markdown_content)

            processing_time = time.time() - start_time

            logger.info(
                "HTML to Markdown conversion completed",
                content_length=len(markdown_content),
                processing_time=processing_time
            )

            return ConversionResult(
                content=markdown_content,
                metadata=metadata,
                processing_time=processing_time,
                success=True,
                warnings=warnings
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"HTML to Markdown conversion failed: {str(e)}"

            logger.error(
                "HTML to Markdown conversion failed",
                error=str(e),
                processing_time=processing_time
            )

            return ConversionResult(
                content="",
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )

    async def _convert_with_markdownify(self, html: str, options: ConversionOptions) -> str:
        """Convert HTML to Markdown using markdownify."""
        # Use basic markdownify options that are known to work
        converter_options = {
            'heading_style': options.heading_style,
            'bullets': options.bullet_style,
            'strip': ['script', 'style', 'nav', 'footer'] if self.config.remove_empty_elements else []
        }

        # Use the basic markdownify function instead of custom converter for now
        from markdownify import markdownify
        return markdownify(html, **converter_options)

    async def _convert_with_html2text(self, html: str, options: ConversionOptions) -> str:
        """Convert HTML to Markdown using html2text."""
        # Configure html2text based on options
        h = self.html2text_converter
        h.body_width = options.wrap_width
        h.unicode_snob = options.unicode_snob
        h.ignore_links = options.ignore_links
        h.ignore_images = options.ignore_images
        h.ignore_emphasis = options.ignore_emphasis
        h.mark_code = options.mark_code

        # Only set attributes that exist
        if hasattr(h, 'escape_misc'):
            h.escape_misc = options.escape_misc

        return h.handle(html)

    def _extract_conversion_metadata(self, html: str, markdown: str) -> Dict[str, Any]:
        """Extract metadata from the conversion process."""
        soup = BeautifulSoup(html, 'lxml')

        metadata = {
            'conversion_time': time.time(),
            'html_length': len(html),
            'markdown_length': len(markdown),
            'compression_ratio': len(markdown) / len(html) if html else 0
        }

        # Count elements
        metadata['element_counts'] = {
            'headings': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
            'paragraphs': len(soup.find_all('p')),
            'links': len(soup.find_all('a')),
            'images': len(soup.find_all('img')),
            'tables': len(soup.find_all('table')),
            'code_blocks': len(soup.find_all(['pre', 'code'])),
            'lists': len(soup.find_all(['ul', 'ol']))
        }

        return metadata

    async def markdown_to_html(
        self,
        markdown_content: str,
        options: Optional[ConversionOptions] = None
    ) -> ConversionResult:
        """Convert Markdown to HTML."""
        start_time = time.time()
        options = options or ConversionOptions()
        warnings = []

        try:
            logger.info("Starting Markdown to HTML conversion")

            # Convert markdown to HTML
            html_content = self.markdown_converter.convert(markdown_content)

            # Sanitize HTML if configured
            if self.config.sanitize_html:
                html_content = self._sanitize_html(html_content)

            # Clean the content
            html_content = self._clean_content(html_content, 'html')

            # Extract metadata
            metadata = {
                'conversion_time': time.time(),
                'markdown_length': len(markdown_content),
                'html_length': len(html_content),
                'expansion_ratio': len(html_content) / len(markdown_content) if markdown_content else 0
            }

            processing_time = time.time() - start_time

            logger.info(
                "Markdown to HTML conversion completed",
                content_length=len(html_content),
                processing_time=processing_time
            )

            return ConversionResult(
                content=html_content,
                metadata=metadata,
                processing_time=processing_time,
                success=True,
                warnings=warnings
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Markdown to HTML conversion failed: {str(e)}"

            logger.error(
                "Markdown to HTML conversion failed",
                error=str(e),
                processing_time=processing_time
            )

            return ConversionResult(
                content="",
                metadata={},
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )

    def clean_content(self, content: str, content_type: str) -> str:
        """Public method to clean content."""
        return self._clean_content(content, content_type)

    async def convert_content(
        self,
        content: str,
        source_format: str,
        target_format: str,
        options: Optional[ConversionOptions] = None
    ) -> ConversionResult:
        """Generic content conversion method."""
        if source_format.lower() == 'html' and target_format.lower() == 'markdown':
            return await self.html_to_markdown(content, options)
        elif source_format.lower() == 'markdown' and target_format.lower() == 'html':
            return await self.markdown_to_html(content, options)
        else:
            raise ValidationError(f"Unsupported conversion: {source_format} to {target_format}")


# Specialized processors for complex content types

class TableProcessor:
    """Specialized processor for HTML tables."""

    @staticmethod
    def convert_table(table_element: BeautifulSoup) -> str:
        """Convert HTML table to Markdown table."""
        rows = []
        headers = []

        # Extract headers
        thead = table_element.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text(strip=True))

        # Extract data rows
        tbody = table_element.find('tbody') or table_element
        for tr in tbody.find_all('tr'):
            if tr.parent.name == 'thead':
                continue
            row = []
            for td in tr.find_all(['td', 'th']):
                cell_text = td.get_text(strip=True)
                # Handle cell spanning
                colspan = int(td.get('colspan', 1))
                rowspan = int(td.get('rowspan', 1))

                if colspan > 1:
                    cell_text += f" (spans {colspan} columns)"
                if rowspan > 1:
                    cell_text += f" (spans {rowspan} rows)"

                row.append(cell_text)
            if row:
                rows.append(row)

        if not headers and rows:
            headers = [f"Column {i+1}" for i in range(len(rows[0]))]

        if not headers or not rows:
            return ""

        # Build markdown table
        result = []

        # Header row
        result.append('| ' + ' | '.join(headers) + ' |')

        # Separator row
        result.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')

        # Data rows
        for row in rows:
            # Pad row to match header length
            while len(row) < len(headers):
                row.append('')
            result.append('| ' + ' | '.join(row[:len(headers)]) + ' |')

        return '\n'.join(result)

    @staticmethod
    def handle_complex_tables(table: BeautifulSoup) -> str:
        """Handle complex tables with merged cells and nested structures."""
        # For now, fall back to basic table conversion
        # This could be enhanced to handle more complex scenarios
        return TableProcessor.convert_table(table)


class CodeBlockProcessor:
    """Specialized processor for code blocks."""

    @staticmethod
    def extract_code_blocks(soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract all code blocks from HTML."""
        code_blocks = []

        for pre in soup.find_all('pre'):
            code_el = pre.find('code')
            if code_el:
                # Extract language from class
                lang = ''
                class_attr = code_el.get('class', [])
                if isinstance(class_attr, list):
                    for cls in class_attr:
                        if cls.startswith(('language-', 'lang-')):
                            lang = cls.replace('language-', '').replace('lang-', '')
                            break

                code_blocks.append({
                    'language': lang,
                    'code': code_el.get_text(),
                    'raw_html': str(pre)
                })

        return code_blocks

    @staticmethod
    def preserve_syntax_highlighting(code_element: BeautifulSoup) -> str:
        """Preserve syntax highlighting information in markdown."""
        lang = ''
        class_attr = code_element.get('class', [])
        if isinstance(class_attr, list):
            for cls in class_attr:
                if cls.startswith(('language-', 'lang-')):
                    lang = cls.replace('language-', '').replace('lang-', '')
                    break

        code_text = code_element.get_text()
        return f'```{lang}\n{code_text}\n```'


class ImageProcessor:
    """Specialized processor for images."""

    @staticmethod
    def process_images(soup: BeautifulSoup, base_url: str = '') -> List[Dict[str, str]]:
        """Process all images in HTML."""
        images = []

        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            title = img.get('title', '')

            # Handle relative URLs
            if base_url and src and not src.startswith(('http://', 'https://', 'data:')):
                src = urljoin(base_url, src)

            # Check for figure caption
            figure = img.find_parent('figure')
            caption = ''
            if figure:
                figcaption = figure.find('figcaption')
                if figcaption:
                    caption = figcaption.get_text(strip=True)

            images.append({
                'src': src,
                'alt': alt or caption,
                'title': title,
                'caption': caption,
                'markdown': ImageProcessor.to_markdown(src, alt or caption, title)
            })

        return images

    @staticmethod
    def to_markdown(src: str, alt: str = '', title: str = '') -> str:
        """Convert image to markdown format."""
        if title:
            return f'![{alt}]({src} "{title}")'
        else:
            return f'![{alt}]({src})'

    @staticmethod
    def handle_figure_captions(figure_element: BeautifulSoup) -> str:
        """Handle figure elements with captions."""
        img = figure_element.find('img')
        figcaption = figure_element.find('figcaption')

        if not img:
            return ''

        src = img.get('src', '')
        alt = img.get('alt', '')
        title = img.get('title', '')

        if figcaption:
            caption = figcaption.get_text(strip=True)
            if not alt:
                alt = caption

            # Create markdown with caption
            img_md = ImageProcessor.to_markdown(src, alt, title)
            return f'{img_md}\n\n*{caption}*'

        return ImageProcessor.to_markdown(src, alt, title)
