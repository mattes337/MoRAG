#!/usr/bin/env python3
"""
Web Scraping Demo for MoRAG

This script demonstrates the web scraping capabilities of MoRAG.
It shows how to extract content from web pages, convert to markdown,
and process the content for RAG applications.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add the src directory to the path so we can import MoRAG modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morag_web import WebProcessor, WebScrapingConfig


async def demo_basic_scraping():
    """Demonstrate basic web scraping functionality."""
    print("🌐 MoRAG Web Scraping Demo")
    print("=" * 50)
    
    # Create web processor
    web_processor = WebProcessor()
    
    # Configure scraping with conservative settings
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
    
    # Example URLs to scrape (using public, scraping-friendly sites)
    test_urls = [
        "https://httpbin.org/html",  # Simple test HTML
        "https://example.com",       # Basic example site
    ]
    
    print(f"📋 Configuration:")
    print(f"   • Timeout: {config.timeout}s")
    print(f"   • Max retries: {config.max_retries}")
    print(f"   • Rate limit: {config.rate_limit_delay}s")
    print(f"   • Extract links: {config.extract_links}")
    print(f"   • Convert to markdown: {config.convert_to_markdown}")
    print()
    
    for i, url in enumerate(test_urls, 1):
        print(f"🔍 Processing URL {i}/{len(test_urls)}: {url}")
        print("-" * 50)
        
        try:
            # Process the URL
            result = await web_processor.process_url(url, config)
            
            if result.success:
                content = result.content
                
                print(f"✅ Success!")
                print(f"   • Title: {content.title}")
                print(f"   • Content length: {content.content_length} characters")
                print(f"   • Content type: {content.content_type}")
                print(f"   • Links found: {len(content.links)}")
                print(f"   • Images found: {len(content.images)}")
                print(f"   • Chunks created: {len(result.chunks)}")
                print(f"   • Processing time: {result.processing_time:.2f}s")
                print()
                
                # Show metadata
                print("📊 Metadata:")
                for key, value in content.metadata.items():
                    if key not in ['extracted_at']:  # Skip timestamp
                        print(f"   • {key}: {value}")
                print()
                
                # Show first 200 characters of content
                print("📄 Content preview:")
                preview = content.content[:200]
                if len(content.content) > 200:
                    preview += "..."
                print(f"   {preview}")
                print()
                
                # Show markdown preview if available
                if content.markdown_content:
                    print("📝 Markdown preview:")
                    md_preview = content.markdown_content[:200]
                    if len(content.markdown_content) > 200:
                        md_preview += "..."
                    print(f"   {md_preview}")
                    print()
                
                # Show chunk information
                if result.chunks:
                    print("🧩 Chunks:")
                    for j, chunk in enumerate(result.chunks[:3]):  # Show first 3 chunks
                        chunk_preview = chunk.text[:100]
                        if len(chunk.text) > 100:
                            chunk_preview += "..."
                        print(f"   • Chunk {j+1}: {chunk_preview}")
                    if len(result.chunks) > 3:
                        print(f"   • ... and {len(result.chunks) - 3} more chunks")
                    print()
                
            else:
                print(f"❌ Failed: {result.error_message}")
                print()
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
            print()
        
        # Add delay between requests to be respectful
        if i < len(test_urls):
            print("⏳ Waiting before next request...")
            await asyncio.sleep(2)
            print()


async def demo_batch_processing():
    """Demonstrate batch processing of multiple URLs."""
    print("🚀 Batch Processing Demo")
    print("=" * 50)
    
    web_processor = WebProcessor()
    config = WebScrapingConfig(
        timeout=5,
        rate_limit_delay=0.5,
        convert_to_markdown=True
    )
    
    # Multiple URLs for batch processing
    urls = [
        "https://httpbin.org/html",
        "https://example.com",
        "https://httpbin.org/robots.txt",  # This will fail due to content type
    ]
    
    print(f"📋 Processing {len(urls)} URLs in batch...")
    print()
    
    try:
        # Process all URLs
        results = await web_processor.process_urls(urls, config)
        
        # Summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"📊 Batch Results:")
        print(f"   • Total URLs: {len(urls)}")
        print(f"   • Successful: {len(successful)}")
        print(f"   • Failed: {len(failed)}")
        print()
        
        # Show successful results
        if successful:
            print("✅ Successful extractions:")
            for result in successful:
                print(f"   • {result.url}")
                print(f"     - Title: {result.content.title}")
                print(f"     - Content: {result.content.content_length} chars")
                print(f"     - Chunks: {len(result.chunks)}")
                print(f"     - Time: {result.processing_time:.2f}s")
            print()
        
        # Show failed results
        if failed:
            print("❌ Failed extractions:")
            for result in failed:
                print(f"   • {result.url}")
                print(f"     - Error: {result.error_message}")
            print()
            
    except Exception as e:
        print(f"💥 Batch processing failed: {str(e)}")
        print()


async def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("⚙️  Configuration Options Demo")
    print("=" * 50)
    
    web_processor = WebProcessor()
    url = "https://httpbin.org/html"
    
    # Test different configurations
    configs = [
        ("Minimal extraction", WebScrapingConfig(
            extract_links=False,
            convert_to_markdown=False,
            clean_content=False
        )),
        ("Full extraction", WebScrapingConfig(
            extract_links=True,
            convert_to_markdown=True,
            clean_content=True,
            remove_navigation=True,
            remove_footer=True
        )),
        ("Fast processing", WebScrapingConfig(
            timeout=3,
            max_retries=1,
            rate_limit_delay=0.1
        ))
    ]
    
    for name, config in configs:
        print(f"🔧 Testing: {name}")
        print("-" * 30)
        
        try:
            result = await web_processor.process_url(url, config)
            
            if result.success:
                content = result.content
                print(f"   • Processing time: {result.processing_time:.2f}s")
                print(f"   • Content length: {content.content_length}")
                print(f"   • Has markdown: {bool(content.markdown_content)}")
                print(f"   • Links found: {len(content.links)}")
                print(f"   • Chunks: {len(result.chunks)}")
            else:
                print(f"   • Failed: {result.error_message}")
                
        except Exception as e:
            print(f"   • Exception: {str(e)}")
        
        print()
        await asyncio.sleep(1)  # Brief delay


async def main():
    """Run all demos."""
    print("🎯 MoRAG Web Scraping Demonstration")
    print("=" * 60)
    print()
    
    try:
        # Run demos
        await demo_basic_scraping()
        await demo_batch_processing()
        await demo_configuration_options()
        
        print("🎉 Demo completed successfully!")
        print()
        print("💡 Next steps:")
        print("   • Try scraping your own URLs")
        print("   • Integrate with embedding services")
        print("   • Store results in vector database")
        print("   • Build RAG applications with the extracted content")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
