# MoRAG Web

Web scraping and processing capabilities for the MoRAG (Multimodal RAG Ingestion Pipeline) system.

## Features

- **URL validation and normalization**
- **Content extraction with BeautifulSoup4**
- **HTML to Markdown conversion**
- **Metadata extraction**
- **Link and image discovery**
- **Content cleaning and text processing**
- **Rate limiting and respectful scraping**
- **Error handling and retries**
- **Async processing for performance**

## Installation

```bash
pip install morag-web
```

## Usage

```python
import asyncio
from morag_web import WebProcessor, WebScrapingConfig

async def scrape_website():
    # Create processor
    processor = WebProcessor()

    # Configure scraping
    config = WebScrapingConfig(
        timeout=10,
        max_retries=2,
        rate_limit_delay=1.0,
        extract_links=True,
        convert_to_markdown=True,
        clean_content=True
    )

    # Process URL
    result = await processor.process_url("https://example.com", config)

    if result.success:
        content = result.content
        print(f"Title: {content.title}")
        print(f"Content: {content.content[:100]}...")
        print(f"Markdown: {content.markdown_content[:100]}...")
        print(f"Links found: {len(content.links)}")

# Run the example
asyncio.run(scrape_website())
```

## License

MIT
