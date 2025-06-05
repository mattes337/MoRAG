#!/usr/bin/env python3
"""
MoRAG Web Processing Test Script

Usage: python test-web.py <url>

Examples:
    python test-web.py https://example.com
    python test-web.py https://en.wikipedia.org/wiki/Python
    python test-web.py https://github.com/your-repo
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Optional
import urllib.parse

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from morag_web import WebProcessor
    from morag_services import ServiceConfig, ContentType
    from morag_core.models import ProcessingConfig
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
        # Initialize configuration
        config = ServiceConfig()
        print_result("Configuration", "✅ Loaded successfully")

        # Initialize web processor
        processor = WebProcessor(config)
        print_result("Web Processor", "✅ Initialized successfully")

        # Create processing configuration
        processing_config = ProcessingConfig(
            max_file_size=10 * 1024 * 1024,  # 10MB
            timeout=60.0,
            extract_metadata=True
        )
        print_result("Processing Config", "✅ Created successfully")
        
        print_section("Processing Web Content")
        print("🔄 Starting web content extraction...")
        print("   This may take a while depending on the website...")
        
        # Process the URL
        result = await processor.process_url(url, processing_config)
        
        if result.success:
            print("✅ Web processing completed successfully!")
            
            print_section("Processing Results")
            print_result("Status", "✅ Success")
            print_result("Content Type", result.content_type)
            print_result("Processing Time", f"{result.processing_time:.2f} seconds")
            
            if result.metadata:
                print_section("Metadata")
                for key, value in result.metadata.items():
                    if isinstance(value, (dict, list)):
                        print_result(key, json.dumps(value, indent=2))
                    else:
                        print_result(key, str(value))
            
            if result.content:
                print_section("Content Preview")
                content_preview = result.content[:1000] + "..." if len(result.content) > 1000 else result.content
                print(f"📄 Content ({len(result.content)} characters):")
                print(content_preview)
                
                # Check for specific content sections
                if "## Page Title" in result.content:
                    print_result("Page Title", "✅ Found")
                if "## Main Content" in result.content:
                    print_result("Main Content", "✅ Found")
                if "## Links" in result.content:
                    print_result("Links", "✅ Found")
            
            if result.summary:
                print_section("Summary")
                print(f"📝 {result.summary}")
            
            # Create safe filename from URL
            safe_filename = urllib.parse.quote(url, safe='').replace('%', '_')[:50]
            
            # Save results to file
            output_file = Path(f"web_{safe_filename}_test_result.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'url': url,
                    'success': result.success,
                    'content_type': result.content_type,
                    'processing_time': result.processing_time,
                    'metadata': result.metadata,
                    'content': result.content,
                    'summary': result.summary,
                    'error': result.error
                }, f, indent=2, ensure_ascii=False)
            
            # Also save markdown content
            markdown_file = Path(f"web_{safe_filename}_converted.md")
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(result.content)
            
            print_section("Output")
            print_result("Results saved to", str(output_file))
            print_result("Markdown saved to", str(markdown_file))
            
            return True
            
        else:
            print("❌ Web processing failed!")
            print_result("Error", result.error or "Unknown error")
            return False
            
    except Exception as e:
        print(f"❌ Error during web processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test-web.py <url>")
        print()
        print("Examples:")
        print("  python test-web.py https://example.com")
        print("  python test-web.py https://en.wikipedia.org/wiki/Python")
        print("  python test-web.py https://github.com/your-repo")
        print()
        print("Note: Make sure the URL is accessible and includes the protocol (http:// or https://)")
        sys.exit(1)
    
    url = sys.argv[1]
    
    try:
        success = asyncio.run(test_web_processing(url))
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
