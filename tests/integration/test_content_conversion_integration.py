"""Integration tests for content conversion with web processor."""

import pytest
from unittest.mock import Mock, patch

from src.morag.processors.web import WebProcessor, WebScrapingConfig
from src.morag.services.content_converter import ContentConverter, ConversionConfig, ConversionOptions
from src.morag.services.chunking import ChunkingService


class TestWebProcessorContentConversionIntegration:
    """Test integration between WebProcessor and ContentConverter."""
    
    @pytest.fixture
    def chunking_service(self):
        """Mock chunking service."""
        service = Mock(spec=ChunkingService)
        service.chunk_with_metadata.return_value = []
        return service
    
    @pytest.fixture
    def content_converter(self):
        """Create ContentConverter for testing."""
        return ContentConverter()
    
    @pytest.fixture
    def web_processor(self, chunking_service, content_converter):
        """Create WebProcessor with ContentConverter."""
        return WebProcessor(
            chunking_service=chunking_service,
            content_converter=content_converter
        )
    
    @pytest.fixture
    def sample_html_response(self):
        """Sample HTML response for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Article</title>
            <meta name="description" content="A test article for conversion">
            <meta name="author" content="Test Author">
        </head>
        <body>
            <nav>
                <ul>
                    <li><a href="/home">Home</a></li>
                    <li><a href="/about">About</a></li>
                </ul>
            </nav>
            <main>
                <article>
                    <h1>Main Article Title</h1>
                    <p>This is the <strong>main content</strong> of the article with <em>emphasis</em>.</p>
                    
                    <h2>Features</h2>
                    <ul>
                        <li>Feature 1</li>
                        <li>Feature 2</li>
                        <li>Feature 3</li>
                    </ul>
                    
                    <h2>Data Table</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Value</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Item 1</td>
                                <td>100</td>
                                <td>First item</td>
                            </tr>
                            <tr>
                                <td>Item 2</td>
                                <td>200</td>
                                <td>Second item</td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <h2>Code Example</h2>
                    <pre><code class="language-python">
def hello_world():
    print("Hello, World!")
    return True
                    </code></pre>
                    
                    <h2>Images</h2>
                    <figure>
                        <img src="https://example.com/image.jpg" alt="Example Image" title="Example Title">
                        <figcaption>This is an example image</figcaption>
                    </figure>
                    
                    <p>More content with <a href="https://example.com">external link</a>.</p>
                </article>
            </main>
            <footer>
                <p>&copy; 2024 Test Site</p>
            </footer>
        </body>
        </html>
        """
    
    @pytest.mark.asyncio
    async def test_web_processor_with_content_converter(self, web_processor, sample_html_response):
        """Test WebProcessor using ContentConverter for markdown conversion."""
        # Mock the HTTP request
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.text = sample_html_response
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.content = sample_html_response.encode()
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            # Configure web scraping
            config = WebScrapingConfig(
                convert_to_markdown=True,
                clean_content=True,
                remove_navigation=True,
                remove_footer=True
            )
            
            # Process URL
            result = await web_processor.process_url("https://example.com/test", config)
            
            assert result.success is True
            assert result.content is not None
            
            # Check that markdown conversion was applied
            markdown_content = result.content.markdown_content
            assert markdown_content is not None
            assert len(markdown_content) > 0
            
            # Verify markdown structure
            assert '# Main Article Title' in markdown_content
            assert '## Features' in markdown_content
            assert '## Data Table' in markdown_content
            assert '## Code Example' in markdown_content
            assert '## Images' in markdown_content
            
            # Verify markdown formatting
            assert '**main content**' in markdown_content
            assert '*emphasis*' in markdown_content
            assert '- Feature 1' in markdown_content
            assert '| Name | Value | Description |' in markdown_content
            assert '```python' in markdown_content
            assert 'def hello_world():' in markdown_content
            assert '![Example Image]' in markdown_content
            assert '[external link](https://example.com)' in markdown_content
            
            # Verify navigation and footer were removed
            assert 'Home' not in markdown_content
            assert 'About' not in markdown_content
            assert '© 2024 Test Site' not in markdown_content
    
    @pytest.mark.asyncio
    async def test_content_converter_fallback(self, web_processor, sample_html_response):
        """Test fallback to markdownify when ContentConverter fails."""
        # Mock ContentConverter to fail
        with patch.object(web_processor.content_converter, 'html_to_markdown') as mock_convert:
            mock_convert.return_value.success = False
            mock_convert.return_value.error_message = "Test error"
            
            # Mock the HTTP request
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = Mock()
                mock_response.text = sample_html_response
                mock_response.headers = {'content-type': 'text/html'}
                mock_response.content = sample_html_response.encode()
                mock_response.raise_for_status.return_value = None
                
                mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
                
                config = WebScrapingConfig(convert_to_markdown=True)
                result = await web_processor.process_url("https://example.com/test", config)
                
                assert result.success is True
                assert result.content.markdown_content is not None
                # Should still have some markdown content from fallback
                assert len(result.content.markdown_content) > 0
    
    @pytest.mark.asyncio
    async def test_different_conversion_options(self, web_processor, sample_html_response):
        """Test different conversion options through WebProcessor."""
        # Test with reference-style links
        converter_config = ConversionConfig(reference_style_links=True)
        content_converter = ContentConverter(converter_config)
        web_processor.content_converter = content_converter
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.text = sample_html_response
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.content = sample_html_response.encode()
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            config = WebScrapingConfig(convert_to_markdown=True)
            result = await web_processor.process_url("https://example.com/test", config)
            
            assert result.success is True
            assert result.content.markdown_content is not None
    
    @pytest.mark.asyncio
    async def test_content_converter_metadata_extraction(self, web_processor, sample_html_response):
        """Test that ContentConverter metadata is properly extracted."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.text = sample_html_response
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.content = sample_html_response.encode()
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            config = WebScrapingConfig(convert_to_markdown=True)
            result = await web_processor.process_url("https://example.com/test", config)
            
            assert result.success is True
            
            # Check that web content metadata includes conversion info
            metadata = result.content.metadata
            assert 'title' in metadata
            assert 'description' in metadata
            assert 'author' in metadata
            assert metadata['title'] == 'Test Article'
            assert metadata['description'] == 'A test article for conversion'
            assert metadata['author'] == 'Test Author'
    
    @pytest.mark.asyncio
    async def test_complex_html_structures(self, web_processor):
        """Test conversion of complex HTML structures."""
        complex_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Complex Test</title></head>
        <body>
            <main>
                <h1>Complex Document</h1>
                
                <!-- Nested lists -->
                <ul>
                    <li>Item 1
                        <ul>
                            <li>Subitem 1.1</li>
                            <li>Subitem 1.2</li>
                        </ul>
                    </li>
                    <li>Item 2</li>
                </ul>
                
                <!-- Complex table with merged cells -->
                <table>
                    <tr>
                        <th rowspan="2">Name</th>
                        <th colspan="2">Details</th>
                    </tr>
                    <tr>
                        <th>Age</th>
                        <th>City</th>
                    </tr>
                    <tr>
                        <td>John</td>
                        <td>30</td>
                        <td>NYC</td>
                    </tr>
                </table>
                
                <!-- Blockquote -->
                <blockquote>
                    <p>This is a quoted text with <strong>emphasis</strong>.</p>
                    <cite>- Author Name</cite>
                </blockquote>
                
                <!-- Multiple code blocks -->
                <pre><code class="language-javascript">
function test() {
    return "JavaScript";
}
                </code></pre>
                
                <pre><code class="language-css">
.class {
    color: red;
}
                </code></pre>
            </main>
        </body>
        </html>
        """
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.text = complex_html
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.content = complex_html.encode()
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            config = WebScrapingConfig(convert_to_markdown=True)
            result = await web_processor.process_url("https://example.com/complex", config)
            
            assert result.success is True
            markdown = result.content.markdown_content
            
            # Check complex structures are preserved
            assert '# Complex Document' in markdown
            assert '- Item 1' in markdown
            assert '- Subitem 1.1' in markdown  # Nested list
            assert '| Name |' in markdown  # Table
            assert 'spans 2 rows' in markdown or 'spans 2 columns' in markdown  # Merged cells
            assert '> This is a quoted text' in markdown  # Blockquote
            assert '```javascript' in markdown
            assert '```css' in markdown
            assert 'function test()' in markdown
            assert '.class {' in markdown
    
    @pytest.mark.asyncio
    async def test_performance_with_large_content(self, web_processor):
        """Test performance with large HTML content."""
        # Generate large HTML content
        large_content_parts = ['<!DOCTYPE html><html><head><title>Large Test</title></head><body><main>']
        
        # Add many paragraphs
        for i in range(100):
            large_content_parts.append(f'<p>This is paragraph {i} with some <strong>bold</strong> and <em>italic</em> text.</p>')
        
        # Add large table
        large_content_parts.append('<table><thead><tr><th>ID</th><th>Name</th><th>Description</th></tr></thead><tbody>')
        for i in range(50):
            large_content_parts.append(f'<tr><td>{i}</td><td>Item {i}</td><td>Description for item {i}</td></tr>')
        large_content_parts.append('</tbody></table>')
        
        large_content_parts.append('</main></body></html>')
        large_html = ''.join(large_content_parts)
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = Mock()
            mock_response.text = large_html
            mock_response.headers = {'content-type': 'text/html'}
            mock_response.content = large_html.encode()
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            config = WebScrapingConfig(convert_to_markdown=True)
            result = await web_processor.process_url("https://example.com/large", config)
            
            assert result.success is True
            assert result.processing_time < 10.0  # Should complete within 10 seconds
            assert len(result.content.markdown_content) > 1000  # Should have substantial content
            
            # Verify structure is preserved
            markdown = result.content.markdown_content
            assert 'This is paragraph 0' in markdown
            assert 'This is paragraph 99' in markdown
            assert '| ID | Name | Description |' in markdown
            assert '| 0 | Item 0 | Description for item 0 |' in markdown
            assert '| 49 | Item 49 | Description for item 49 |' in markdown
