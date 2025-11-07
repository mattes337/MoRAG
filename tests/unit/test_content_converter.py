"""Unit tests for content converter service."""

import pytest
from unittest.mock import patch

from src.morag.services.content_converter import (
    ContentConverter,
    ConversionConfig,
    ConversionOptions,
    CustomMarkdownConverter,
    TableProcessor,
    CodeBlockProcessor,
    ImageProcessor
)
from src.morag.core.exceptions import ValidationError


class TestConversionConfig:
    """Test ConversionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConversionConfig()

        assert config.preserve_tables is True
        assert config.preserve_code_blocks is True
        assert config.preserve_images is True
        assert config.reference_style_links is False
        assert config.clean_whitespace is True
        assert config.remove_empty_elements is True
        assert config.sanitize_html is True
        assert isinstance(config.allowed_tags, list)
        assert 'h1' in config.allowed_tags
        assert 'table' in config.allowed_tags

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConversionConfig(
            preserve_tables=False,
            sanitize_html=False,
            allowed_tags=['p', 'div']
        )

        assert config.preserve_tables is False
        assert config.sanitize_html is False
        assert config.allowed_tags == ['p', 'div']


class TestConversionOptions:
    """Test ConversionOptions dataclass."""

    def test_default_options(self):
        """Test default option values."""
        options = ConversionOptions()

        assert options.heading_style == "ATX"
        assert options.bullet_style == "-"
        assert options.code_fence_style == "```"
        assert options.emphasis_style == "*"
        assert options.strong_style == "**"
        assert options.link_style == "inline"
        assert options.wrap_width == 0
        assert options.unicode_snob is True

    def test_custom_options(self):
        """Test custom option values."""
        options = ConversionOptions(
            heading_style="SETEXT",
            bullet_style="*",
            link_style="reference"
        )

        assert options.heading_style == "SETEXT"
        assert options.bullet_style == "*"
        assert options.link_style == "reference"


class TestContentConverter:
    """Test ContentConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a ContentConverter instance for testing."""
        return ContentConverter()

    @pytest.fixture
    def sample_html(self):
        """Sample HTML for testing."""
        return """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Heading</h1>
            <p>This is a <strong>test</strong> paragraph with <em>emphasis</em>.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
            <table>
                <thead>
                    <tr><th>Name</th><th>Age</th></tr>
                </thead>
                <tbody>
                    <tr><td>John</td><td>30</td></tr>
                    <tr><td>Jane</td><td>25</td></tr>
                </tbody>
            </table>
            <pre><code class="language-python">print("Hello, World!")</code></pre>
            <img src="test.jpg" alt="Test Image" title="Test Title">
        </body>
        </html>
        """

    @pytest.fixture
    def sample_markdown(self):
        """Sample Markdown for testing."""
        return """# Main Heading

This is a **test** paragraph with *emphasis*.

- Item 1
- Item 2

| Name | Age |
|------|-----|
| John | 30  |
| Jane | 25  |

```python
print("Hello, World!")
```

![Test Image](test.jpg "Test Title")"""

    def test_init_default(self):
        """Test ContentConverter initialization with defaults."""
        converter = ContentConverter()

        assert converter.config is not None
        assert converter.html2text_converter is not None
        assert converter.markdown_converter is not None

    def test_init_custom_config(self):
        """Test ContentConverter initialization with custom config."""
        config = ConversionConfig(preserve_tables=False)
        converter = ContentConverter(config)

        assert converter.config.preserve_tables is False

    def test_sanitize_html(self, converter):
        """Test HTML sanitization."""
        html = '<script>alert("xss")</script><p>Safe content</p>'
        sanitized = converter._sanitize_html(html)

        assert '<script>' not in sanitized
        assert 'Safe content' in sanitized

    def test_sanitize_html_disabled(self):
        """Test HTML sanitization when disabled."""
        config = ConversionConfig(sanitize_html=False)
        converter = ContentConverter(config)

        html = '<script>alert("xss")</script><p>Safe content</p>'
        sanitized = converter._sanitize_html(html)

        assert sanitized == html  # No sanitization

    def test_clean_markdown_content(self, converter):
        """Test markdown content cleaning."""
        markdown = "# Header\n\n\n\nParagraph   \n\n\n[]() empty link\n\n\n# Another"
        cleaned = converter._clean_content(markdown, 'markdown')

        assert '\n\n\n\n' not in cleaned
        assert '[]()' not in cleaned
        assert cleaned.strip() == cleaned

    def test_clean_html_content(self, converter):
        """Test HTML content cleaning."""
        html = "<p>  Text  </p>  <div>  More  </div>"
        cleaned = converter._clean_content(html, 'html')

        assert '  ' not in cleaned
        assert cleaned.strip() == cleaned

    def test_extract_main_content_with_selectors(self, converter, sample_html):
        """Test main content extraction with custom selectors."""
        selectors = ['body']
        result = converter.extract_main_content(sample_html, selectors)

        assert '<body>' in result
        assert 'Main Heading' in result

    def test_extract_main_content_default(self, converter):
        """Test main content extraction with default selectors."""
        html = '<html><body><main><p>Main content</p></main><nav>Navigation</nav></body></html>'
        result = converter.extract_main_content(html)

        assert 'Main content' in result
        assert '<main>' in result

    @pytest.mark.asyncio
    async def test_html_to_markdown_success(self, converter, sample_html):
        """Test successful HTML to Markdown conversion."""
        result = await converter.html_to_markdown(sample_html)

        assert result.success is True
        assert result.error_message is None
        assert '# Main Heading' in result.content
        assert '**test**' in result.content
        assert '*emphasis*' in result.content
        assert '- Item 1' in result.content
        assert '| Name | Age |' in result.content
        assert '```' in result.content  # Code blocks (language may not be preserved)
        assert 'print("Hello, World!")' in result.content  # Code content
        assert '![Test Image]' in result.content
        assert result.processing_time > 0
        assert result.metadata['html_length'] > 0
        assert result.metadata['markdown_length'] > 0

    @pytest.mark.asyncio
    async def test_html_to_markdown_with_options(self, converter, sample_html):
        """Test HTML to Markdown conversion with custom options."""
        options = ConversionOptions(
            heading_style="SETEXT",
            bullet_style="*",
            link_style="reference"
        )

        result = await converter.html_to_markdown(sample_html, options)

        assert result.success is True
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_html_to_markdown_error_handling(self, converter):
        """Test HTML to Markdown conversion error handling."""
        # Test with invalid HTML that might cause issues
        with patch.object(converter, '_convert_with_markdownify', side_effect=Exception("Test error")):
            with patch.object(converter, '_convert_with_html2text', side_effect=Exception("Test error")):
                result = await converter.html_to_markdown("<invalid>")

                assert result.success is False
                assert result.error_message is not None
                assert "Test error" in result.error_message

    @pytest.mark.asyncio
    async def test_markdown_to_html_success(self, converter, sample_markdown):
        """Test successful Markdown to HTML conversion."""
        result = await converter.markdown_to_html(sample_markdown)

        assert result.success is True
        assert result.error_message is None
        assert '<h1>' in result.content
        assert '<strong>' in result.content
        assert '<em>' in result.content
        assert '<ul>' in result.content
        assert '<table>' in result.content
        assert '<pre>' in result.content
        assert '<img' in result.content
        assert result.processing_time > 0
        assert result.metadata['markdown_length'] > 0
        assert result.metadata['html_length'] > 0

    @pytest.mark.asyncio
    async def test_markdown_to_html_error_handling(self, converter):
        """Test Markdown to HTML conversion error handling."""
        with patch.object(converter.markdown_converter, 'convert', side_effect=Exception("Test error")):
            result = await converter.markdown_to_html("# Test")

            assert result.success is False
            assert result.error_message is not None
            assert "Test error" in result.error_message

    def test_clean_content_public_method(self, converter):
        """Test public clean_content method."""
        content = "  Test content  \n\n\n"
        cleaned = converter.clean_content(content, 'markdown')

        assert cleaned.strip() == "Test content"

    @pytest.mark.asyncio
    async def test_convert_content_html_to_markdown(self, converter, sample_html):
        """Test generic convert_content method for HTML to Markdown."""
        result = await converter.convert_content(sample_html, 'html', 'markdown')

        assert result.success is True
        assert '# Main Heading' in result.content

    @pytest.mark.asyncio
    async def test_convert_content_markdown_to_html(self, converter, sample_markdown):
        """Test generic convert_content method for Markdown to HTML."""
        result = await converter.convert_content(sample_markdown, 'markdown', 'html')

        assert result.success is True
        assert '<h1>' in result.content

    @pytest.mark.asyncio
    async def test_convert_content_unsupported_format(self, converter):
        """Test convert_content with unsupported format combination."""
        with pytest.raises(ValidationError) as exc_info:
            await converter.convert_content("test", 'xml', 'json')

        assert "Unsupported conversion" in str(exc_info.value)


class TestCustomMarkdownConverter:
    """Test CustomMarkdownConverter class."""

    def test_init(self):
        """Test CustomMarkdownConverter initialization."""
        options = {'preserve_tables': True}
        converter = CustomMarkdownConverter(**options)

        assert converter.options == options

    def test_convert_table_basic(self):
        """Test basic table conversion."""
        from bs4 import BeautifulSoup

        html = """
        <table>
            <thead><tr><th>Name</th><th>Age</th></tr></thead>
            <tbody>
                <tr><td>John</td><td>30</td></tr>
                <tr><td>Jane</td><td>25</td></tr>
            </tbody>
        </table>
        """

        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')

        converter = CustomMarkdownConverter(preserve_tables=True)
        result = converter.convert_table(table, '', [])

        assert '| Name | Age |' in result
        assert '| --- | --- |' in result
        assert '| John | 30 |' in result
        assert '| Jane | 25 |' in result

    def test_convert_pre_with_language(self):
        """Test code block conversion with language."""
        from bs4 import BeautifulSoup

        html = '<pre><code class="language-python">print("hello")</code></pre>'
        soup = BeautifulSoup(html, 'html.parser')
        pre = soup.find('pre')

        converter = CustomMarkdownConverter(preserve_code_blocks=True, code_fence_style='```')
        result = converter.convert_pre(pre, '', [])

        assert '```python' in result
        assert 'print("hello")' in result
        assert result.count('```') == 2

    def test_convert_img_with_title(self):
        """Test image conversion with title."""
        from bs4 import BeautifulSoup

        html = '<img src="test.jpg" alt="Test" title="Test Title">'
        soup = BeautifulSoup(html, 'html.parser')
        img = soup.find('img')

        converter = CustomMarkdownConverter(preserve_images=True)
        result = converter.convert_img(img, '', [])

        assert result == '![Test](test.jpg "Test Title")'

    def test_convert_img_with_figure_caption(self):
        """Test image conversion with figure caption."""
        from bs4 import BeautifulSoup

        html = '<figure><img src="test.jpg" alt=""><figcaption>Test Caption</figcaption></figure>'
        soup = BeautifulSoup(html, 'html.parser')
        img = soup.find('img')

        converter = CustomMarkdownConverter(preserve_images=True)
        result = converter.convert_img(img, '', [])

        assert result == '![Test Caption](test.jpg)'


class TestTableProcessor:
    """Test TableProcessor class."""

    def test_convert_table_basic(self):
        """Test basic table conversion."""
        from bs4 import BeautifulSoup

        html = """
        <table>
            <thead><tr><th>Name</th><th>Age</th></tr></thead>
            <tbody>
                <tr><td>John</td><td>30</td></tr>
                <tr><td>Jane</td><td>25</td></tr>
            </tbody>
        </table>
        """

        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')

        result = TableProcessor.convert_table(table)

        assert '| Name | Age |' in result
        assert '| --- | --- |' in result
        assert '| John | 30 |' in result
        assert '| Jane | 25 |' in result

    def test_convert_table_with_colspan(self):
        """Test table conversion with colspan."""
        from bs4 import BeautifulSoup

        html = """
        <table>
            <tr><th>Name</th><th>Details</th></tr>
            <tr><td>John</td><td colspan="2">Age: 30, City: NYC</td></tr>
        </table>
        """

        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')

        result = TableProcessor.convert_table(table)

        assert 'spans 2 columns' in result

    def test_convert_table_no_headers(self):
        """Test table conversion without headers."""
        from bs4 import BeautifulSoup

        html = """
        <table>
            <tr><td>John</td><td>30</td></tr>
            <tr><td>Jane</td><td>25</td></tr>
        </table>
        """

        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')

        result = TableProcessor.convert_table(table)

        assert '| Column 1 | Column 2 |' in result
        assert '| John | 30 |' in result

    def test_convert_empty_table(self):
        """Test conversion of empty table."""
        from bs4 import BeautifulSoup

        html = '<table></table>'
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')

        result = TableProcessor.convert_table(table)

        assert result == ""

    def test_handle_complex_tables(self):
        """Test complex table handling."""
        from bs4 import BeautifulSoup

        html = """
        <table>
            <tr><th>Name</th><th>Age</th></tr>
            <tr><td>John</td><td>30</td></tr>
        </table>
        """

        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')

        result = TableProcessor.handle_complex_tables(table)

        # Should fall back to basic conversion for now
        assert '| Name | Age |' in result


class TestCodeBlockProcessor:
    """Test CodeBlockProcessor class."""

    def test_extract_code_blocks(self):
        """Test code block extraction."""
        from bs4 import BeautifulSoup

        html = """
        <pre><code class="language-python">print("hello")</code></pre>
        <pre><code class="lang-javascript">console.log("world");</code></pre>
        <pre><code>plain code</code></pre>
        """

        soup = BeautifulSoup(html, 'html.parser')
        code_blocks = CodeBlockProcessor.extract_code_blocks(soup)

        assert len(code_blocks) == 3
        assert code_blocks[0]['language'] == 'python'
        assert code_blocks[0]['code'] == 'print("hello")'
        assert code_blocks[1]['language'] == 'javascript'
        assert code_blocks[2]['language'] == ''  # No language specified

    def test_preserve_syntax_highlighting(self):
        """Test syntax highlighting preservation."""
        from bs4 import BeautifulSoup

        html = '<code class="language-python">print("hello")</code>'
        soup = BeautifulSoup(html, 'html.parser')
        code = soup.find('code')

        result = CodeBlockProcessor.preserve_syntax_highlighting(code)

        assert result == '```python\nprint("hello")\n```'

    def test_preserve_syntax_highlighting_no_language(self):
        """Test syntax highlighting preservation without language."""
        from bs4 import BeautifulSoup

        html = '<code>plain code</code>'
        soup = BeautifulSoup(html, 'html.parser')
        code = soup.find('code')

        result = CodeBlockProcessor.preserve_syntax_highlighting(code)

        assert result == '```\nplain code\n```'


class TestImageProcessor:
    """Test ImageProcessor class."""

    def test_process_images_basic(self):
        """Test basic image processing."""
        from bs4 import BeautifulSoup

        html = """
        <img src="test1.jpg" alt="Test 1" title="Title 1">
        <img src="test2.png" alt="Test 2">
        <img src="test3.gif">
        """

        soup = BeautifulSoup(html, 'html.parser')
        images = ImageProcessor.process_images(soup)

        assert len(images) == 3
        assert images[0]['src'] == 'test1.jpg'
        assert images[0]['alt'] == 'Test 1'
        assert images[0]['title'] == 'Title 1'
        assert images[0]['markdown'] == '![Test 1](test1.jpg "Title 1")'
        assert images[1]['markdown'] == '![Test 2](test2.png)'
        assert images[2]['alt'] == ''

    def test_process_images_with_base_url(self):
        """Test image processing with base URL."""
        from bs4 import BeautifulSoup

        html = '<img src="/images/test.jpg" alt="Test">'
        soup = BeautifulSoup(html, 'html.parser')
        images = ImageProcessor.process_images(soup, 'https://example.com')

        assert images[0]['src'] == 'https://example.com/images/test.jpg'

    def test_process_images_with_figure_caption(self):
        """Test image processing with figure caption."""
        from bs4 import BeautifulSoup

        html = """
        <figure>
            <img src="test.jpg" alt="">
            <figcaption>Test Caption</figcaption>
        </figure>
        """

        soup = BeautifulSoup(html, 'html.parser')
        images = ImageProcessor.process_images(soup)

        assert len(images) == 1
        assert images[0]['alt'] == 'Test Caption'
        assert images[0]['caption'] == 'Test Caption'

    def test_to_markdown_with_title(self):
        """Test markdown conversion with title."""
        result = ImageProcessor.to_markdown('test.jpg', 'Alt text', 'Title text')
        assert result == '![Alt text](test.jpg "Title text")'

    def test_to_markdown_without_title(self):
        """Test markdown conversion without title."""
        result = ImageProcessor.to_markdown('test.jpg', 'Alt text')
        assert result == '![Alt text](test.jpg)'

    def test_handle_figure_captions(self):
        """Test figure caption handling."""
        from bs4 import BeautifulSoup

        html = """
        <figure>
            <img src="test.jpg" alt="Test" title="Title">
            <figcaption>Figure Caption</figcaption>
        </figure>
        """

        soup = BeautifulSoup(html, 'html.parser')
        figure = soup.find('figure')

        result = ImageProcessor.handle_figure_captions(figure)

        # The function uses the original alt text, not the caption for the alt attribute
        assert '![Test](test.jpg "Title")' in result
        assert '*Figure Caption*' in result

    def test_handle_figure_no_image(self):
        """Test figure handling without image."""
        from bs4 import BeautifulSoup

        html = '<figure><figcaption>Caption only</figcaption></figure>'
        soup = BeautifulSoup(html, 'html.parser')
        figure = soup.find('figure')

        result = ImageProcessor.handle_figure_captions(figure)

        assert result == ''
