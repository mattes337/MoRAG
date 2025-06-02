#!/usr/bin/env python3
"""Test script for content conversion functionality."""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag.services.content_converter import (
    ContentConverter,
    ConversionConfig,
    ConversionOptions,
    TableProcessor,
    CodeBlockProcessor,
    ImageProcessor
)
from morag.processors.web import WebProcessor, WebScrapingConfig


async def test_basic_html_to_markdown():
    """Test basic HTML to Markdown conversion."""
    print("🔄 Testing basic HTML to Markdown conversion...")
    
    converter = ContentConverter()
    
    html = """
    <html>
    <head><title>Test Document</title></head>
    <body>
        <h1>Main Title</h1>
        <p>This is a <strong>test</strong> paragraph with <em>emphasis</em>.</p>
        <ul>
            <li>First item</li>
            <li>Second item</li>
        </ul>
        <blockquote>
            <p>This is a quote.</p>
        </blockquote>
    </body>
    </html>
    """
    
    result = await converter.html_to_markdown(html)
    
    if result.success:
        print("✅ Basic conversion successful!")
        print(f"📊 Processing time: {result.processing_time:.3f}s")
        print(f"📏 HTML length: {result.metadata['html_length']}")
        print(f"📏 Markdown length: {result.metadata['markdown_length']}")
        print(f"📉 Compression ratio: {result.metadata['compression_ratio']:.2f}")
        print("\n📝 Converted Markdown:")
        print("-" * 50)
        print(result.content)
        print("-" * 50)
    else:
        print(f"❌ Basic conversion failed: {result.error_message}")
    
    return result.success


async def test_advanced_html_features():
    """Test advanced HTML features conversion."""
    print("\n🔄 Testing advanced HTML features...")
    
    converter = ContentConverter()
    
    html = """
    <html>
    <body>
        <h1>Advanced Features Test</h1>
        
        <h2>Data Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Age</th>
                    <th>City</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>John Doe</td>
                    <td>30</td>
                    <td>New York</td>
                </tr>
                <tr>
                    <td>Jane Smith</td>
                    <td>25</td>
                    <td>Los Angeles</td>
                </tr>
            </tbody>
        </table>
        
        <h2>Code Example</h2>
        <pre><code class="language-python">
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
        </code></pre>
        
        <h2>Images</h2>
        <figure>
            <img src="https://example.com/chart.png" alt="Data Chart" title="Sales Data">
            <figcaption>Monthly sales data for Q1 2024</figcaption>
        </figure>
        
        <h2>Links</h2>
        <p>Visit our <a href="https://example.com" title="Homepage">website</a> for more information.</p>
    </body>
    </html>
    """
    
    result = await converter.html_to_markdown(html)
    
    if result.success:
        print("✅ Advanced features conversion successful!")
        print(f"📊 Processing time: {result.processing_time:.3f}s")
        
        markdown = result.content
        
        # Check for specific features
        features_found = []
        if '| Name | Age | City |' in markdown:
            features_found.append("✅ Table headers")
        if '| John Doe | 30 | New York |' in markdown:
            features_found.append("✅ Table data")
        if '```python' in markdown:
            features_found.append("✅ Code blocks with syntax")
        if 'def fibonacci(n):' in markdown:
            features_found.append("✅ Code content")
        if '![Data Chart]' in markdown:
            features_found.append("✅ Images with alt text")
        if '[website](https://example.com' in markdown:
            features_found.append("✅ Links")
        
        print("\n🎯 Features detected:")
        for feature in features_found:
            print(f"  {feature}")
        
        print("\n📝 Converted Markdown:")
        print("-" * 50)
        print(result.content)
        print("-" * 50)
    else:
        print(f"❌ Advanced features conversion failed: {result.error_message}")
    
    return result.success


async def test_markdown_to_html():
    """Test Markdown to HTML conversion."""
    print("\n🔄 Testing Markdown to HTML conversion...")
    
    converter = ContentConverter()
    
    markdown = """
# Main Title

This is a **bold** text and *italic* text.

## Features List

- Feature 1
- Feature 2
- Feature 3

## Data Table

| Product | Price | Stock |
|---------|-------|-------|
| Widget A | $10.99 | 50 |
| Widget B | $15.99 | 30 |

## Code Example

```python
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
```

## Image

![Sample Image](https://example.com/image.jpg "Sample")

Visit [our website](https://example.com) for more info.
    """
    
    result = await converter.markdown_to_html(markdown)
    
    if result.success:
        print("✅ Markdown to HTML conversion successful!")
        print(f"📊 Processing time: {result.processing_time:.3f}s")
        print(f"📏 Markdown length: {result.metadata['markdown_length']}")
        print(f"📏 HTML length: {result.metadata['html_length']}")
        print(f"📈 Expansion ratio: {result.metadata['expansion_ratio']:.2f}")
        
        html = result.content
        
        # Check for specific HTML elements
        features_found = []
        if '<h1>' in html:
            features_found.append("✅ Headings")
        if '<strong>' in html and '<em>' in html:
            features_found.append("✅ Text formatting")
        if '<ul>' in html and '<li>' in html:
            features_found.append("✅ Lists")
        if '<table>' in html and '<th>' in html:
            features_found.append("✅ Tables")
        if '<pre>' in html and '<code>' in html:
            features_found.append("✅ Code blocks")
        if '<img' in html:
            features_found.append("✅ Images")
        if '<a href=' in html:
            features_found.append("✅ Links")
        
        print("\n🎯 HTML elements generated:")
        for feature in features_found:
            print(f"  {feature}")
        
        print("\n📝 Generated HTML:")
        print("-" * 50)
        print(result.content)
        print("-" * 50)
    else:
        print(f"❌ Markdown to HTML conversion failed: {result.error_message}")
    
    return result.success


async def test_web_processor_integration():
    """Test integration with WebProcessor."""
    print("\n🔄 Testing WebProcessor integration...")
    
    # Create a mock HTML response
    mock_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Article</title>
        <meta name="description" content="Test article description">
    </head>
    <body>
        <nav><a href="/home">Home</a></nav>
        <main>
            <article>
                <h1>Article Title</h1>
                <p>This is the main content with <strong>important</strong> information.</p>
                <ul>
                    <li>Point 1</li>
                    <li>Point 2</li>
                </ul>
            </article>
        </main>
        <footer>Copyright 2024</footer>
    </body>
    </html>
    """
    
    # Test the conversion directly
    converter = ContentConverter()
    
    # Extract main content (simulate what WebProcessor does)
    main_content = converter.extract_main_content(mock_html, ['main'])
    
    result = await converter.html_to_markdown(main_content)
    
    if result.success:
        print("✅ WebProcessor integration test successful!")
        print(f"📊 Processing time: {result.processing_time:.3f}s")
        
        markdown = result.content
        
        # Check that navigation and footer are excluded
        if 'Home' not in markdown and 'Copyright 2024' not in markdown:
            print("✅ Navigation and footer properly excluded")
        else:
            print("⚠️  Navigation or footer not properly excluded")
        
        if '# Article Title' in markdown and '**important**' in markdown:
            print("✅ Main content properly converted")
        else:
            print("⚠️  Main content conversion issues")
        
        print("\n📝 Extracted and converted content:")
        print("-" * 50)
        print(result.content)
        print("-" * 50)
    else:
        print(f"❌ WebProcessor integration test failed: {result.error_message}")
    
    return result.success


def test_specialized_processors():
    """Test specialized processors."""
    print("\n🔄 Testing specialized processors...")
    
    from bs4 import BeautifulSoup
    
    # Test TableProcessor
    print("\n📊 Testing TableProcessor...")
    table_html = """
    <table>
        <thead>
            <tr><th>Product</th><th>Price</th><th>Stock</th></tr>
        </thead>
        <tbody>
            <tr><td>Widget A</td><td>$10.99</td><td>50</td></tr>
            <tr><td>Widget B</td><td colspan="2">Out of stock</td></tr>
        </tbody>
    </table>
    """
    
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    table_md = TableProcessor.convert_table(table)
    
    if '| Product | Price | Stock |' in table_md:
        print("✅ TableProcessor working correctly")
        print(f"📝 Table markdown:\n{table_md}")
    else:
        print("❌ TableProcessor failed")
    
    # Test CodeBlockProcessor
    print("\n💻 Testing CodeBlockProcessor...")
    code_html = '<pre><code class="language-python">print("Hello, World!")</code></pre>'
    soup = BeautifulSoup(code_html, 'html.parser')
    code_blocks = CodeBlockProcessor.extract_code_blocks(soup)
    
    if code_blocks and code_blocks[0]['language'] == 'python':
        print("✅ CodeBlockProcessor working correctly")
        print(f"📝 Code block: {code_blocks[0]}")
    else:
        print("❌ CodeBlockProcessor failed")
    
    # Test ImageProcessor
    print("\n🖼️  Testing ImageProcessor...")
    img_html = '<img src="test.jpg" alt="Test Image" title="Test Title">'
    soup = BeautifulSoup(img_html, 'html.parser')
    images = ImageProcessor.process_images(soup)
    
    if images and images[0]['markdown'] == '![Test Image](test.jpg "Test Title")':
        print("✅ ImageProcessor working correctly")
        print(f"📝 Image markdown: {images[0]['markdown']}")
    else:
        print("❌ ImageProcessor failed")
    
    return True


async def main():
    """Run all content conversion tests."""
    print("🚀 Starting Content Conversion Tests")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        test_basic_html_to_markdown(),
        test_advanced_html_features(),
        test_markdown_to_html(),
        test_web_processor_integration(),
    ]
    
    # Run async tests
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    # Run sync tests
    sync_result = test_specialized_processors()
    results.append(sync_result)
    
    # Calculate results
    successful = sum(1 for result in results if result is True)
    total = len(results)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"✅ Successful tests: {successful}/{total}")
    print(f"⏱️  Total time: {total_time:.3f}s")
    
    if successful == total:
        print("🎉 All content conversion tests passed!")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
