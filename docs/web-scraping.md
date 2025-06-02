# Web Scraping in MoRAG

MoRAG includes comprehensive web scraping capabilities for extracting and processing content from websites. This document provides a complete guide to using the web scraping features.

## Overview

The web scraping module provides:
- **URL validation and normalization**
- **Content extraction with BeautifulSoup4**
- **HTML to Markdown conversion**
- **Metadata extraction**
- **Link and image discovery**
- **Content cleaning and text processing**
- **Rate limiting and respectful scraping**
- **Error handling and retries**
- **Async processing for performance**

## Quick Start

```python
import asyncio
from morag.processors.web import WebProcessor, WebScrapingConfig

async def scrape_website():
    # Create processor
    processor = WebProcessor()
    
    # Configure scraping
    config = WebScrapingConfig(
        timeout=30,
        extract_links=True,
        convert_to_markdown=True
    )
    
    # Process URL
    result = await processor.process_url("https://example.com", config)
    
    if result.success:
        print(f"Title: {result.content.title}")
        print(f"Content: {result.content.content}")
        print(f"Chunks: {len(result.chunks)}")
    else:
        print(f"Error: {result.error_message}")

# Run the scraper
asyncio.run(scrape_website())
```

## Configuration Options

### WebScrapingConfig

```python
@dataclass
class WebScrapingConfig:
    timeout: int = 30                    # Request timeout in seconds
    max_retries: int = 3                 # Maximum retry attempts
    rate_limit_delay: float = 1.0        # Delay between requests
    user_agent: str = "MoRAG/1.0"       # User agent string
    max_content_length: int = 10MB       # Maximum content size
    allowed_content_types: List[str]     # Allowed MIME types
    extract_links: bool = True           # Extract page links
    convert_to_markdown: bool = True     # Convert HTML to Markdown
    clean_content: bool = True           # Clean extracted content
    remove_navigation: bool = True       # Remove nav elements
    remove_footer: bool = True           # Remove footer elements
    preserve_tables: bool = True         # Keep table formatting
    preserve_lists: bool = True          # Keep list formatting
```

### Example Configurations

```python
# Fast scraping for simple content
fast_config = WebScrapingConfig(
    timeout=10,
    max_retries=1,
    rate_limit_delay=0.5,
    convert_to_markdown=False
)

# Comprehensive extraction
detailed_config = WebScrapingConfig(
    timeout=60,
    max_retries=5,
    extract_links=True,
    convert_to_markdown=True,
    clean_content=True,
    preserve_tables=True,
    preserve_lists=True
)

# Respectful scraping
respectful_config = WebScrapingConfig(
    rate_limit_delay=2.0,
    user_agent="MoRAG Bot (+https://yoursite.com/bot)"
)
```

## Processing Single URLs

```python
async def process_single_url():
    processor = WebProcessor()
    config = WebScrapingConfig()
    
    result = await processor.process_url("https://example.com", config)
    
    if result.success:
        content = result.content
        
        # Basic information
        print(f"URL: {content.url}")
        print(f"Title: {content.title}")
        print(f"Content Type: {content.content_type}")
        print(f"Content Length: {content.content_length}")
        
        # Extracted content
        print(f"Text Content: {content.content}")
        print(f"Markdown: {content.markdown_content}")
        
        # Metadata
        print(f"Description: {content.metadata.get('description')}")
        print(f"Keywords: {content.metadata.get('keywords')}")
        print(f"Author: {content.metadata.get('author')}")
        
        # Links and images
        print(f"Links found: {len(content.links)}")
        print(f"Images found: {len(content.images)}")
        
        # Processing info
        print(f"Extraction time: {content.extraction_time:.2f}s")
        print(f"Total processing time: {result.processing_time:.2f}s")
        print(f"Chunks created: {len(result.chunks)}")
```

## Batch Processing

```python
async def process_multiple_urls():
    processor = WebProcessor()
    config = WebScrapingConfig(rate_limit_delay=1.0)
    
    urls = [
        "https://example1.com",
        "https://example2.com", 
        "https://example3.com"
    ]
    
    results = await processor.process_urls(urls, config)
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"Processed {len(urls)} URLs")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    for result in failed:
        print(f"Failed {result.url}: {result.error_message}")
```

## Content Extraction Features

### Metadata Extraction

The scraper automatically extracts:
- **Basic meta tags**: title, description, keywords, author
- **Open Graph data**: og:title, og:description, og:image, etc.
- **Twitter Card data**: twitter:card, twitter:title, etc.
- **Structured data**: JSON-LD, microdata (future enhancement)

### Content Cleaning

The processor removes:
- Script and style elements
- HTML comments
- Navigation elements (optional)
- Footer elements (optional)
- Boilerplate content

### HTML to Markdown Conversion

Features include:
- Clean heading conversion (# ## ###)
- List preservation (- * 1.)
- Table formatting
- Link preservation
- Image alt text
- Code block detection
- Blockquote formatting

## Integration with Tasks

### Celery Task Integration

```python
from morag.tasks.web_tasks import process_web_url

# Queue a web scraping task
task = process_web_url.delay(
    url="https://example.com",
    config={"timeout": 30, "convert_to_markdown": True},
    task_id="web_scrape_001"
)

# Check task status
result = task.get()
print(result)
```

### Batch Task Processing

```python
from morag.tasks.web_tasks import process_web_urls_batch

# Queue batch processing
task = process_web_urls_batch.delay(
    urls=["https://site1.com", "https://site2.com"],
    config={"rate_limit_delay": 2.0},
    task_id="batch_scrape_001"
)

result = task.get()
print(f"Processed {result['total_urls']} URLs")
print(f"Successful: {result['successful']}")
print(f"Failed: {result['failed']}")
```

## Error Handling

The scraper handles various error conditions:

### Network Errors
- Connection timeouts
- DNS resolution failures
- SSL certificate errors
- HTTP status errors (404, 500, etc.)

### Content Errors
- Unsupported content types
- Content too large
- Malformed HTML
- Encoding issues

### Rate Limiting
- Automatic delays between requests
- Exponential backoff on retries
- Respectful crawling practices

```python
async def handle_errors():
    processor = WebProcessor()
    
    try:
        result = await processor.process_url("https://invalid-url.com")
        
        if not result.success:
            if "timeout" in result.error_message.lower():
                print("Request timed out - try increasing timeout")
            elif "404" in result.error_message:
                print("Page not found")
            elif "content type" in result.error_message.lower():
                print("Unsupported content type")
            else:
                print(f"Other error: {result.error_message}")
                
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## Best Practices

### Respectful Scraping
1. **Check robots.txt** before scraping
2. **Use appropriate delays** between requests
3. **Set a descriptive User-Agent** string
4. **Respect rate limits** and server capacity
5. **Handle errors gracefully**

### Performance Optimization
1. **Use batch processing** for multiple URLs
2. **Configure appropriate timeouts**
3. **Limit concurrent requests**
4. **Cache results** when appropriate
5. **Monitor memory usage** for large content

### Content Quality
1. **Enable content cleaning** for better text quality
2. **Use markdown conversion** for structured content
3. **Extract metadata** for better context
4. **Validate extracted content** before processing
5. **Handle different content types** appropriately

## Testing

Run the web scraping tests:

```bash
# Unit tests
pytest tests/test_web_processor.py -v

# Integration tests  
pytest tests/test_web_integration.py -v

# Task tests
pytest tests/test_web_tasks.py -v

# All web scraping tests
pytest tests/ -k "web" -v
```

## Examples

See the complete demo script:
```bash
python examples/web_scraping_demo.py
```

This demonstrates:
- Basic URL processing
- Batch processing
- Configuration options
- Error handling
- Content extraction features

## Troubleshooting

### Common Issues

**Import Errors**
```python
# Make sure dependencies are installed
pip install beautifulsoup4 markdownify httpx
```

**Timeout Errors**
```python
# Increase timeout for slow sites
config = WebScrapingConfig(timeout=60)
```

**Content Type Errors**
```python
# Add custom content types
config = WebScrapingConfig(
    allowed_content_types=['text/html', 'application/xhtml+xml', 'text/plain']
)
```

**Rate Limiting**
```python
# Increase delay between requests
config = WebScrapingConfig(rate_limit_delay=3.0)
```

For more help, check the logs or create an issue on GitHub.
