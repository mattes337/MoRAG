"""Example usage of the morag-web package."""

import asyncio
import sys
from typing import List

from morag_web import WebProcessor, WebScrapingConfig
from morag_web.service import WebService


async def basic_scraping_example(url: str) -> None:
    """Demonstrate basic web scraping functionality."""
    print(f"\n🌐 Processing URL: {url}")
    print("=" * 50)
    
    # Create web processor
    web_processor = WebProcessor()
    
    # Configure scraping
    config = WebScrapingConfig(
        timeout=10,
        max_retries=2,
        rate_limit_delay=1.0,
        extract_links=True,
        convert_to_markdown=True,
        clean_content=True,
        remove_navigation=True,
        remove_footer=True
    )
    
    # Process URL
    result = await web_processor.process_url(url, config)
    
    if result.success:
        content = result.content
        
        # Print basic information
        print(f"\n📄 Title: {content.title}")
        print(f"📊 Content Length: {content.content_length} bytes")
        print(f"⏱️ Processing Time: {result.processing_time:.2f}s")
        
        # Print content preview
        print("\n📝 Content Preview:")
        preview = content.content[:300] + "..." if len(content.content) > 300 else content.content
        print(f"{preview}")
        
        # Print markdown preview
        print("\n📝 Markdown Preview:")
        md_preview = content.markdown_content[:300] + "..." if len(content.markdown_content) > 300 else content.markdown_content
        print(f"{md_preview}")
        
        # Print links and images
        print(f"\n🔗 Links Found: {len(content.links)}")
        if content.links:
            for i, link in enumerate(content.links[:5]):
                print(f"  {i+1}. {link}")
            if len(content.links) > 5:
                print(f"  ... and {len(content.links) - 5} more")
        
        print(f"\n🖼️ Images Found: {len(content.images)}")
        if content.images:
            for i, image in enumerate(content.images[:3]):
                print(f"  {i+1}. {image}")
            if len(content.images) > 3:
                print(f"  ... and {len(content.images) - 3} more")
        
        # Print metadata
        print("\n📋 Metadata:")
        for key, value in list(content.metadata.items())[:10]:
            print(f"  {key}: {value}")
        if len(content.metadata) > 10:
            print(f"  ... and {len(content.metadata) - 10} more fields")
        
        # Print chunks
        print(f"\n📚 Chunks Created: {len(result.chunks)}")
    else:
        print(f"❌ Error: {result.error_message}")


async def service_example(urls: List[str]) -> None:
    """Demonstrate using the WebService for multiple URLs."""
    print("\n🌐 Web Service Example")
    print("=" * 50)
    
    # Create web service
    service = WebService()
    
    # Process multiple URLs
    results = await service.process_multiple_urls(
        urls,
        {"timeout": 10, "max_retries": 1, "rate_limit_delay": 1.0},
        concurrency_limit=2
    )
    
    # Print results summary
    print(f"\n📊 Processed {len(results)} URLs:")
    for url, result in results.items():
        status = "✅ Success" if result.success else f"❌ Failed: {result.error_message}"
        print(f"  {url}: {status}")
        if result.success and result.content:
            print(f"    Title: {result.content.title}")
            print(f"    Processing Time: {result.processing_time:.2f}s")
            print(f"    Content Length: {len(result.content.content)} chars")
            print(f"    Links: {len(result.content.links)}")
            print()


async def main() -> None:
    """Run the examples."""
    # Default URLs to test
    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]
    
    # Use command line arguments if provided
    if len(sys.argv) > 1:
        test_urls = sys.argv[1:]
    
    # Run basic example with first URL
    await basic_scraping_example(test_urls[0])
    
    # Run service example with all URLs
    if len(test_urls) > 1:
        await service_example(test_urls)


if __name__ == "__main__":
    asyncio.run(main())