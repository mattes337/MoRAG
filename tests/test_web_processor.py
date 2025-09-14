"""Tests for web processor."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import httpx
from bs4 import BeautifulSoup

from src.morag.processors.web import (
    WebProcessor, 
    WebScrapingConfig, 
    WebContent, 
    WebScrapingResult
)
from src.morag.core.exceptions import ValidationError, ProcessingError


class TestWebProcessor:
    """Test cases for WebProcessor."""
    
    @pytest.fixture
    def web_processor(self):
        """Create a web processor instance."""
        return WebProcessor()
    
    @pytest.fixture
    def sample_html(self):
        """Sample HTML content for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="A test page for web scraping">
            <meta name="keywords" content="test, web, scraping">
            <meta property="og:title" content="Test Page OG">
            <meta name="twitter:card" content="summary">
        </head>
        <body>
            <nav>
                <a href="/home">Home</a>
                <a href="/about">About</a>
            </nav>
            <main>
                <h1>Main Title</h1>
                <p>This is the main content of the page.</p>
                <p>It contains multiple paragraphs with <strong>important</strong> information.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
                <table>
                    <tr><th>Header</th><th>Value</th></tr>
                    <tr><td>Row 1</td><td>Data 1</td></tr>
                </table>
                <img src="/image.jpg" alt="Test image">
                <a href="https://example.com">External Link</a>
            </main>
            <footer>
                <p>Footer content</p>
            </footer>
        </body>
        </html>
        """
    
    def test_validate_url_valid(self, web_processor):
        """Test URL validation with valid URLs."""
        # Test with protocol
        assert web_processor._validate_url("https://example.com") == "https://example.com"
        assert web_processor._validate_url("http://example.com") == "http://example.com"
        
        # Test without protocol (should add https)
        assert web_processor._validate_url("example.com") == "https://example.com"
        
        # Test with path
        assert web_processor._validate_url("example.com/path") == "https://example.com/path"
    
    def test_validate_url_invalid(self, web_processor):
        """Test URL validation with invalid URLs."""
        with pytest.raises(ValidationError):
            web_processor._validate_url("")
        
        with pytest.raises(ValidationError):
            web_processor._validate_url(None)
        
        with pytest.raises(ValidationError):
            web_processor._validate_url("not-a-url")
        
        with pytest.raises(ValidationError):
            web_processor._validate_url("://invalid")
    
    def test_extract_metadata(self, web_processor, sample_html):
        """Test metadata extraction from HTML."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        metadata = web_processor._extract_metadata(soup, "https://example.com")
        
        assert metadata['url'] == "https://example.com"
        assert metadata['domain'] == "example.com"
        assert metadata['title'] == "Test Page"
        assert metadata['description'] == "A test page for web scraping"
        assert metadata['keywords'] == "test, web, scraping"
        assert metadata['og_title'] == "Test Page OG"
        assert metadata['twitter_card'] == "summary"
        assert 'extracted_at' in metadata
    
    def test_clean_html(self, web_processor, sample_html):
        """Test HTML cleaning functionality."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        config = WebScrapingConfig(remove_navigation=True, remove_footer=True)
        
        cleaned_soup = web_processor._clean_html(soup, config)
        
        # Navigation should be removed
        assert not cleaned_soup.find('nav')
        
        # Footer should be removed
        assert not cleaned_soup.find('footer')
        
        # Main content should remain
        assert cleaned_soup.find('main')
        assert cleaned_soup.find('h1')
    
    def test_extract_links(self, web_processor, sample_html):
        """Test link extraction from HTML."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        links = web_processor._extract_links(soup, "https://example.com")
        
        expected_links = [
            "https://example.com/home",
            "https://example.com/about", 
            "https://example.com"  # External link should be converted to absolute
        ]
        
        # Check that all expected links are found
        for link in expected_links:
            assert any(link in found_link for found_link in links)
    
    def test_extract_images(self, web_processor, sample_html):
        """Test image extraction from HTML."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        images = web_processor._extract_images(soup, "https://example.com")
        
        assert "https://example.com/image.jpg" in images
    
    def test_extract_main_content(self, web_processor, sample_html):
        """Test main content extraction."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        main_content = web_processor._extract_main_content(soup)
        
        # Should find the main tag
        assert main_content.name == 'main'
        assert main_content.find('h1')
        assert "Main Title" in main_content.get_text()
    
    def test_convert_to_markdown(self, web_processor):
        """Test HTML to Markdown conversion."""
        html_content = """
        <h1>Title</h1>
        <p>This is a <strong>paragraph</strong> with <em>emphasis</em>.</p>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
        </ul>
        """
        
        config = WebScrapingConfig(preserve_lists=True)
        markdown = web_processor._convert_to_markdown(html_content, config)
        
        assert "# Title" in markdown
        assert "**paragraph**" in markdown
        assert "*emphasis*" in markdown
        assert "- Item 1" in markdown
        assert "- Item 2" in markdown
    
    def test_clean_markdown(self, web_processor):
        """Test markdown cleaning functionality."""
        messy_markdown = """
        # Title
        
        
        
        This is content.   
        
        
        []()
        
        
        # Another Title
        """
        
        cleaned = web_processor._clean_markdown(messy_markdown)
        
        # Should remove excessive line breaks
        assert "\n\n\n" not in cleaned
        
        # Should remove empty links
        assert "[]()" not in cleaned
        
        # Should preserve proper structure
        assert "# Title" in cleaned
        assert "This is content." in cleaned
    
    @pytest.mark.asyncio
    async def test_rate_limit(self, web_processor):
        """Test rate limiting functionality."""
        import time
        
        start_time = time.time()
        await web_processor._rate_limit(0.1)
        await web_processor._rate_limit(0.1)
        end_time = time.time()
        
        # Should have waited at least 0.1 seconds
        assert end_time - start_time >= 0.1
    
    @pytest.mark.asyncio
    async def test_fetch_content_success(self, web_processor, sample_html):
        """Test successful content fetching."""
        config = WebScrapingConfig()
        
        # Mock httpx response
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = sample_html.encode()
        mock_response.raise_for_status = Mock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            content, content_type, headers = await web_processor._fetch_content(
                "https://example.com", config
            )
            
            assert content == sample_html
            assert "text/html" in content_type
            assert headers['content-type'] == 'text/html'
    
    @pytest.mark.asyncio
    async def test_fetch_content_http_error(self, web_processor):
        """Test content fetching with HTTP error."""
        config = WebScrapingConfig()
        
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("Not Found", request=Mock(), response=mock_response)
            )
            
            with pytest.raises(ProcessingError):
                await web_processor._fetch_content("https://example.com", config)
    
    @pytest.mark.asyncio
    async def test_fetch_content_unsupported_type(self, web_processor):
        """Test content fetching with unsupported content type."""
        config = WebScrapingConfig()
        
        # Mock response with unsupported content type
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.content = b"PDF content"
        mock_response.raise_for_status = Mock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            with pytest.raises(ValidationError, match="Unsupported content type"):
                await web_processor._fetch_content("https://example.com", config)
    
    @pytest.mark.asyncio
    async def test_fetch_content_too_large(self, web_processor):
        """Test content fetching with content too large."""
        config = WebScrapingConfig(max_content_length=100)
        
        # Mock response with large content
        large_content = "x" * 200
        mock_response = Mock()
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = large_content.encode()
        mock_response.raise_for_status = Mock()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            with pytest.raises(ValidationError, match="Content too large"):
                await web_processor._fetch_content("https://example.com", config)

    @pytest.mark.asyncio
    async def test_process_url_success(self, web_processor, sample_html):
        """Test successful URL processing."""
        config = WebScrapingConfig()

        # Mock httpx response
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = sample_html.encode()
        mock_response.raise_for_status = Mock()

        # Mock chunking service
        mock_chunk_infos = [
            Mock(
                text="chunk1",
                start_char=0,
                end_char=6,
                sentence_count=1,
                word_count=1,
                entities=[],
                topics=[],
                chunk_type="text"
            ),
            Mock(
                text="chunk2",
                start_char=7,
                end_char=13,
                sentence_count=1,
                word_count=1,
                entities=[],
                topics=[],
                chunk_type="text"
            )
        ]
        web_processor.chunking_service.chunk_with_metadata = AsyncMock(return_value=mock_chunk_infos)

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            result = await web_processor.process_url("https://example.com", config)

            assert result.success
            assert result.url == "https://example.com"
            assert result.content.title == "Test Page"
            assert result.content.content_length > 0
            assert len(result.chunks) == 2
            assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_process_url_failure(self, web_processor):
        """Test URL processing failure."""
        config = WebScrapingConfig()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.RequestError("Network error")
            )

            result = await web_processor.process_url("https://example.com", config)

            assert not result.success
            assert "Network error" in result.error_message
            assert result.content is None
            assert len(result.chunks) == 0

    @pytest.mark.asyncio
    async def test_process_urls_batch(self, web_processor, sample_html):
        """Test batch URL processing."""
        config = WebScrapingConfig()
        urls = ["https://example1.com", "https://example2.com"]

        # Mock httpx response
        mock_response = Mock()
        mock_response.text = sample_html
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = sample_html.encode()
        mock_response.raise_for_status = Mock()

        # Mock chunking service
        mock_chunk_infos = [Mock(
            text="chunk1",
            start_char=0,
            end_char=6,
            sentence_count=1,
            word_count=1,
            entities=[],
            topics=[],
            chunk_type="text"
        )]
        web_processor.chunking_service.chunk_with_metadata = AsyncMock(return_value=mock_chunk_infos)

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            results = await web_processor.process_urls(urls, config)

            assert len(results) == 2
            assert all(r.success for r in results)
            assert results[0].url == "https://example1.com"
            assert results[1].url == "https://example2.com"

    @pytest.mark.asyncio
    async def test_process_urls_mixed_results(self, web_processor, sample_html):
        """Test batch URL processing with mixed success/failure."""
        config = WebScrapingConfig()
        urls = ["https://example1.com", "https://example2.com"]

        # Mock responses - first succeeds, second fails
        mock_success_response = Mock()
        mock_success_response.text = sample_html
        mock_success_response.headers = {'content-type': 'text/html'}
        mock_success_response.content = sample_html.encode()
        mock_success_response.raise_for_status = Mock()

        responses = [mock_success_response, httpx.RequestError("Network error")]

        # Mock chunking service
        mock_chunks = [Mock(text="chunk1")]
        web_processor.chunking_service.chunk_text = AsyncMock(return_value=mock_chunks)

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=responses)

            results = await web_processor.process_urls(urls, config)

            assert len(results) == 2
            assert results[0].success
            assert not results[1].success
            assert results[1].error_message is not None
            assert "https://example2.com" in results[1].error_message


class TestWebScrapingConfig:
    """Test cases for WebScrapingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WebScrapingConfig()

        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.rate_limit_delay == 1.0
        assert "MoRAG" in config.user_agent
        assert config.max_content_length == 10 * 1024 * 1024
        assert 'text/html' in config.allowed_content_types
        assert config.extract_links is True
        assert config.convert_to_markdown is True
        assert config.clean_content is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = WebScrapingConfig(
            timeout=60,
            max_retries=5,
            rate_limit_delay=2.0,
            extract_links=False
        )

        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.rate_limit_delay == 2.0
        assert config.extract_links is False


class TestWebContent:
    """Test cases for WebContent data class."""

    def test_web_content_creation(self):
        """Test WebContent creation."""
        content = WebContent(
            url="https://example.com",
            title="Test Page",
            content="Test content",
            markdown_content="# Test content",
            metadata={"key": "value"},
            links=["https://link1.com"],
            images=["https://image1.jpg"],
            extraction_time=1.5,
            content_length=100,
            content_type="text/html"
        )

        assert content.url == "https://example.com"
        assert content.title == "Test Page"
        assert content.content == "Test content"
        assert content.markdown_content == "# Test content"
        assert content.metadata == {"key": "value"}
        assert content.links == ["https://link1.com"]
        assert content.images == ["https://image1.jpg"]
        assert content.extraction_time == 1.5
        assert content.content_length == 100
        assert content.content_type == "text/html"


class TestWebScrapingResult:
    """Test cases for WebScrapingResult data class."""

    def test_successful_result(self):
        """Test successful WebScrapingResult."""
        content = WebContent(
            url="https://example.com",
            title="Test",
            content="Content",
            markdown_content="# Content",
            metadata={},
            links=[],
            images=[],
            extraction_time=1.0,
            content_length=50,
            content_type="text/html"
        )

        result = WebScrapingResult(
            url="https://example.com",
            content=content,
            chunks=[],
            processing_time=2.0,
            success=True
        )

        assert result.success is True
        assert result.url == "https://example.com"
        assert result.content == content
        assert result.chunks == []
        assert result.processing_time == 2.0
        assert result.error_message is None

    def test_failed_result(self):
        """Test failed WebScrapingResult."""
        result = WebScrapingResult(
            url="https://example.com",
            content=None,
            chunks=[],
            processing_time=1.0,
            success=False,
            error_message="Network error"
        )

        assert result.success is False
        assert result.url == "https://example.com"
        assert result.content is None
        assert result.chunks == []
        assert result.processing_time == 1.0
        assert result.error_message == "Network error"
