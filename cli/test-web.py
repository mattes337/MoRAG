#!/usr/bin/env python3
"""
MoRAG Web Processing Test Script

Supports both processing (immediate results) and ingestion (background + storage) modes.

Usage:
    python test-web.py <url> [options]

Processing Mode (immediate results):
    python test-web.py https://example.com
    python test-web.py https://en.wikipedia.org/wiki/Python
    python test-web.py https://github.com/your-repo

Ingestion Mode (background processing + storage):
    python test-web.py https://example.com --ingest
    python test-web.py https://news-site.com/article --ingest --metadata '{"category": "news"}'
    python test-web.py https://docs.site.com --ingest --webhook-url https://my-app.com/webhook

Options:
    --ingest                    Enable ingestion mode (background processing + storage)
    --webhook-url URL          Webhook URL for completion notifications (ingestion mode only)
    --metadata JSON            Additional metadata as JSON string (ingestion mode only)
    --help                     Show this help message
"""

import sys
import asyncio
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import urllib.parse
import requests

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from morag_web import WebProcessor, WebScrapingConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-web")
    sys.exit(1)


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def print_result(key: str, value: str, indent: int = 0):
    """Print a formatted key-value result."""
    spaces = "  " * indent
    print(f"{spaces}📋 {key}: {value}")


def validate_url(url: str) -> bool:
    """Validate URL format."""
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


async def test_web_processing(url: str) -> bool:
    """Test web processing functionality."""
    print_header("MoRAG Web Processing Test")
    
    if not validate_url(url):
        print(f"❌ Error: Invalid URL format: {url}")
        return False
    
    print_result("Target URL", url)
    
    try:
        # Initialize web processor
        processor = WebProcessor()
        print_result("Web Processor", "✅ Initialized successfully")

        # Create web scraping configuration
        config = WebScrapingConfig(
            timeout=30,
            max_retries=2,
            rate_limit_delay=1.0,
            extract_links=True,
            convert_to_markdown=True,
            clean_content=True,
            remove_navigation=True,
            remove_footer=True
        )
        print_result("Web Scraping Config", "✅ Created successfully")

        print_section("Processing Web Content")
        print("🔄 Starting web content extraction...")
        print("   This may take a while depending on the website...")

        # Process the URL
        result = await processor.process_url(url, config)
        
        if result.success:
            print("✅ Web processing completed successfully!")

            print_section("Processing Results")
            print_result("Status", "✅ Success")
            print_result("Processing Time", f"{result.processing_time:.2f} seconds")
            print_result("URL", result.url or url)

            if result.content:
                web_content = result.content
                print_section("Web Content Information")
                print_result("Title", web_content.title)
                print_result("Content Type", web_content.content_type)
                print_result("Content Length", f"{web_content.content_length} characters")
                print_result("Links Found", f"{len(web_content.links)}")
                print_result("Images Found", f"{len(web_content.images)}")
                print_result("Extraction Time", f"{web_content.extraction_time:.2f} seconds")

                if web_content.metadata:
                    print_section("Page Metadata")
                    for key, value in web_content.metadata.items():
                        if isinstance(value, (dict, list)):
                            print_result(key, json.dumps(value, indent=2))
                        else:
                            print_result(key, str(value))

                if web_content.content:
                    print_section("Content Preview")
                    content_preview = web_content.content[:500] + "..." if len(web_content.content) > 500 else web_content.content
                    print(f"📄 Raw Content ({len(web_content.content)} characters):")
                    print(content_preview)

                if web_content.markdown_content:
                    print_section("Markdown Preview")
                    markdown_preview = web_content.markdown_content[:500] + "..." if len(web_content.markdown_content) > 500 else web_content.markdown_content
                    print(f"📄 Markdown Content ({len(web_content.markdown_content)} characters):")
                    print(markdown_preview)

                if web_content.links:
                    print_section("Links Found (first 5)")
                    for i, link in enumerate(web_content.links[:5]):
                        print_result(f"Link {i+1}", link)

            if result.chunks:
                print_section("Document Chunks")
                print_result("Chunks Count", f"{len(result.chunks)}")
                for i, chunk in enumerate(result.chunks[:3]):
                    print(f"  Chunk {i+1}: {chunk.content[:100]}{'...' if len(chunk.content) > 100 else ''}")

            # Create safe filename from URL
            safe_filename = urllib.parse.quote(url, safe='').replace('%', '_')[:50]

            # Save results to file
            output_file = Path(f"uploads/web_{safe_filename}_test_result.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'processing',
                    'url': result.url,
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'content': {
                        'title': web_content.title if result.content else None,
                        'content_type': web_content.content_type if result.content else None,
                        'content_length': web_content.content_length if result.content else 0,
                        'links_count': len(web_content.links) if result.content else 0,
                        'images_count': len(web_content.images) if result.content else 0,
                        'extraction_time': web_content.extraction_time if result.content else 0,
                        'metadata': web_content.metadata if result.content else {},
                        'content': web_content.content if result.content else None,
                        'markdown_content': web_content.markdown_content if result.content else None,
                        'links': web_content.links if result.content else [],
                        'images': web_content.images if result.content else []
                    },
                    'chunks_count': len(result.chunks),
                    'error_message': result.error_message
                }, f, indent=2, ensure_ascii=False)

            # Also save markdown content if available
            if result.content and result.content.markdown_content:
                markdown_file = Path(f"uploads/web_{safe_filename}_converted.md")
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(result.content.markdown_content)
                print_result("Markdown saved to", str(markdown_file))

            print_section("Output")
            print_result("Results saved to", str(output_file))

            return True

        else:
            print("❌ Web processing failed!")
            print_result("Error", result.error_message or "Unknown error")
            return False

    except Exception as e:
        print(f"❌ Error during web processing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_web_ingestion(url: str, webhook_url: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Test web ingestion functionality."""
    print_header("MoRAG Web Ingestion Test")

    if not validate_url(url):
        print(f"❌ Error: Invalid URL format: {url}")
        return False

    print_result("Target URL", url)
    print_result("Webhook URL", webhook_url or "None")
    print_result("Metadata", json.dumps(metadata, indent=2) if metadata else "None")

    try:
        print_section("Submitting Ingestion Task")
        print("🔄 Starting web ingestion...")

        # Prepare request data
        data = {
            'source_type': 'web',
            'url': url
        }

        if webhook_url:
            data['webhook_url'] = webhook_url
        if metadata:
            data['metadata'] = metadata

        # Submit to ingestion API
        response = requests.post(
            'http://localhost:8000/api/v1/ingest/url',
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Web ingestion task submitted successfully!")

            print_section("Ingestion Results")
            print_result("Status", "✅ Success")
            print_result("Task ID", result.get('task_id', 'Unknown'))
            print_result("Message", result.get('message', 'Task created'))
            print_result("Estimated Time", f"{result.get('estimated_time', 'Unknown')} seconds")

            # Create safe filename from URL
            safe_filename = urllib.parse.quote(url, safe='').replace('%', '_')[:50]

            # Save ingestion result
            output_file = Path(f"uploads/web_{safe_filename}_ingest_result.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'ingestion',
                    'task_id': result.get('task_id'),
                    'status': result.get('status'),
                    'message': result.get('message'),
                    'estimated_time': result.get('estimated_time'),
                    'webhook_url': webhook_url,
                    'metadata': metadata,
                    'url': url
                }, f, indent=2, ensure_ascii=False)

            print_section("Output")
            print_result("Ingestion result saved to", str(output_file))
            print_result("Monitor task status", f"curl http://localhost:8000/api/v1/status/{result.get('task_id')}")

            return True
        else:
            print("❌ Web ingestion failed!")
            print_result("Status Code", str(response.status_code))
            print_result("Error", response.text)
            return False

    except Exception as e:
        print(f"❌ Error during web ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MoRAG Web Processing Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Processing Mode (immediate results):
    python test-web.py https://example.com
    python test-web.py https://en.wikipedia.org/wiki/Python
    python test-web.py https://github.com/your-repo

  Ingestion Mode (background processing + storage):
    python test-web.py https://example.com --ingest
    python test-web.py https://news-site.com/article --ingest --metadata '{"category": "news"}'
    python test-web.py https://docs.site.com --ingest --webhook-url https://my-app.com/webhook

Note: Make sure the URL is accessible and includes the protocol (http:// or https://)
        """
    )

    parser.add_argument('url', help='URL to process')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--qdrant', action='store_true',
                       help='Store in Qdrant vector database (ingestion mode only)')
    parser.add_argument('--neo4j', action='store_true',
                       help='Store in Neo4j graph database (ingestion mode only)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')

    args = parser.parse_args()

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in metadata: {e}")
            sys.exit(1)

    try:
        if args.ingest:
            # Ingestion mode
            success = asyncio.run(test_web_ingestion(
                args.url,
                webhook_url=args.webhook_url,
                metadata=metadata
            ))
            if success:
                print("\n🎉 Web ingestion test completed successfully!")
                print("💡 Use the task ID to monitor progress and retrieve results.")
                sys.exit(0)
            else:
                print("\n💥 Web ingestion test failed!")
                sys.exit(1)
        else:
            # Processing mode
            success = asyncio.run(test_web_processing(args.url))
            if success:
                print("\n🎉 Web processing test completed successfully!")
                sys.exit(0)
            else:
                print("\n💥 Web processing test failed!")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
