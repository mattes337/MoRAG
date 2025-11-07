"""Integration tests for web scraping functionality."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from src.morag.processors.web import WebProcessor, WebScrapingConfig


class TestWebIntegration:
    """Integration tests for web scraping."""

    @pytest.fixture
    def sample_html_page(self):
        """Sample HTML page for testing."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Sample Article - Tech Blog</title>
            <meta name="description" content="A comprehensive guide to web scraping with Python">
            <meta name="keywords" content="web scraping, python, automation">
            <meta name="author" content="John Doe">
            <meta property="og:title" content="Sample Article - Tech Blog">
            <meta property="og:description" content="Learn web scraping techniques">
            <meta name="twitter:card" content="summary_large_image">
        </head>
        <body>
            <header>
                <nav>
                    <a href="/">Home</a>
                    <a href="/about">About</a>
                    <a href="/contact">Contact</a>
                </nav>
            </header>

            <main>
                <article>
                    <h1>The Complete Guide to Web Scraping</h1>

                    <p>Web scraping is a powerful technique for extracting data from websites.
                    In this comprehensive guide, we'll explore various methods and best practices.</p>

                    <h2>Getting Started</h2>
                    <p>Before diving into web scraping, it's important to understand the basics:</p>

                    <ul>
                        <li>HTML structure and DOM</li>
                        <li>HTTP requests and responses</li>
                        <li>CSS selectors and XPath</li>
                        <li>Rate limiting and ethical considerations</li>
                    </ul>

                    <h2>Tools and Libraries</h2>
                    <p>There are several excellent tools for web scraping:</p>

                    <table>
                        <thead>
                            <tr>
                                <th>Tool</th>
                                <th>Language</th>
                                <th>Best For</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>BeautifulSoup</td>
                                <td>Python</td>
                                <td>Simple HTML parsing</td>
                            </tr>
                            <tr>
                                <td>Scrapy</td>
                                <td>Python</td>
                                <td>Large-scale scraping</td>
                            </tr>
                        </tbody>
                    </table>

                    <p>For more advanced scenarios, consider using headless browsers like
                    <strong>Selenium</strong> or <em>Playwright</em>.</p>

                    <img src="/images/scraping-diagram.png" alt="Web scraping workflow diagram">

                    <p>Remember to always respect robots.txt and implement proper rate limiting.</p>

                    <blockquote>
                        "With great power comes great responsibility" - applies to web scraping too!
                    </blockquote>
                </article>
            </main>

            <footer>
                <p>&copy; 2024 Tech Blog. All rights reserved.</p>
                <p>Contact us at <a href="mailto:info@techblog.com">info@techblog.com</a></p>
            </footer>
        </body>
        </html>
        """

    @pytest.mark.asyncio
    async def test_complete_web_scraping_workflow(self, sample_html_page):
        """Test the complete web scraping workflow."""
        # Create web processor
        web_processor = WebProcessor()

        # Configure scraping
        config = WebScrapingConfig(
            timeout=10,
            max_retries=1,
            rate_limit_delay=0.1,
            extract_links=True,
            convert_to_markdown=True,
            clean_content=True,
        )

        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = sample_html_page
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.content = sample_html_page.encode("utf-8")
        mock_response.raise_for_status = Mock()

        # Mock chunking service to return simple chunks
        mock_chunk_infos = [
            Mock(
                text="The Complete Guide to Web Scraping",
                start_char=0,
                end_char=35,
                sentence_count=1,
                word_count=6,
                entities=[],
                topics=["web scraping"],
                chunk_type="title",
            ),
            Mock(
                text="Web scraping is a powerful technique for extracting data from websites.",
                start_char=36,
                end_char=106,
                sentence_count=1,
                word_count=11,
                entities=[],
                topics=["web scraping", "data extraction"],
                chunk_type="text",
            ),
        ]
        web_processor.chunking_service.chunk_text = AsyncMock(
            return_value=mock_chunk_infos
        )

        # Mock HTTP client
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            # Process the URL
            result = await web_processor.process_url(
                "https://techblog.com/web-scraping-guide", config
            )

            # Verify the result
            assert result.success is True
            assert result.url == "https://techblog.com/web-scraping-guide"
            assert result.content is not None

            # Check extracted content
            content = result.content
            assert content.title == "Sample Article - Tech Blog"
            assert "web scraping" in content.content.lower()
            assert content.content_type == "text/html; charset=utf-8"
            assert len(content.content) > 0

            # Check metadata extraction
            metadata = content.metadata
            assert metadata["title"] == "Sample Article - Tech Blog"
            assert (
                metadata["description"]
                == "A comprehensive guide to web scraping with Python"
            )
            assert metadata["keywords"] == "web scraping, python, automation"
            assert metadata["author"] == "John Doe"
            assert metadata["og_title"] == "Sample Article - Tech Blog"
            assert metadata["twitter_card"] == "summary_large_image"
            assert metadata["domain"] == "techblog.com"

            # Check images extraction (navigation links are removed during cleaning)
            assert len(content.images) > 0
            assert any("scraping-diagram.png" in img for img in content.images)

            # Check markdown conversion
            assert len(content.markdown_content) > 0
            assert "# The Complete Guide to Web Scraping" in content.markdown_content
            assert "## Getting Started" in content.markdown_content
            assert "- HTML structure and DOM" in content.markdown_content

            # Check chunks creation
            assert len(result.chunks) == 2
            assert result.chunks[0].text == "The Complete Guide to Web Scraping"
            assert result.chunks[0].chunk_type == "title"
            assert result.chunks[1].chunk_type == "text"

            # Check processing time
            assert result.processing_time > 0

            # Verify HTTP client was called correctly
            mock_client.return_value.__aenter__.return_value.get.assert_called_once()
            call_args = mock_client.return_value.__aenter__.return_value.get.call_args
            assert call_args[0][0] == "https://techblog.com/web-scraping-guide"

            # Check headers were set correctly
            headers = call_args[1]["headers"]
            assert "User-Agent" in headers
            assert "MoRAG" in headers["User-Agent"]
            assert (
                headers["Accept"]
                == "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            )

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self):
        """Test error handling in web scraping workflow."""
        web_processor = WebProcessor()
        config = WebScrapingConfig(max_retries=1)

        # Mock HTTP error
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )

            # Process URL that will fail
            result = await web_processor.process_url("https://nonexistent.com", config)

            # Verify error handling
            assert result.success is False
            assert result.content is None
            assert len(result.chunks) == 0
            assert "Connection failed" in result.error_message
            assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, sample_html_page):
        """Test batch processing of multiple URLs."""
        web_processor = WebProcessor()
        config = WebScrapingConfig(rate_limit_delay=0.1)

        urls = ["https://example1.com", "https://example2.com", "https://example3.com"]

        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = sample_html_page
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = sample_html_page.encode()
        mock_response.raise_for_status = Mock()

        # Mock chunking service
        mock_chunk_info = Mock(
            text="Sample content",
            start_char=0,
            end_char=14,
            sentence_count=1,
            word_count=2,
            entities=[],
            topics=[],
            chunk_type="text",
        )
        web_processor.chunking_service.chunk_text = AsyncMock(
            return_value=[mock_chunk_info]
        )

        # Mock HTTP client
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            # Process URLs in batch
            results = await web_processor.process_urls(urls, config)

            # Verify results
            assert len(results) == 3
            assert all(r.success for r in results)
            assert results[0].url == "https://example1.com"
            assert results[1].url == "https://example2.com"
            assert results[2].url == "https://example3.com"

            # Verify all URLs were processed
            assert mock_client.return_value.__aenter__.return_value.get.call_count == 3
