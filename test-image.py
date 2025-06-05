#!/usr/bin/env python3
"""
MoRAG Image Processing Test Script

Usage: python test-image.py <image_file>

Examples:
    python test-image.py my-image.jpg
    python test-image.py screenshot.png
    python test-image.py diagram.gif
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Optional

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from morag_image import ImageProcessor
    from morag_services import ServiceConfig, ContentType
    from morag_core.models import ProcessingConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed the MoRAG packages:")
    print("  pip install -e packages/morag-core")
    print("  pip install -e packages/morag-image")
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


async def test_image_processing(image_file: Path) -> bool:
    """Test image processing functionality."""
    print_header("MoRAG Image Processing Test")
    
    if not image_file.exists():
        print(f"❌ Error: Image file not found: {image_file}")
        return False
    
    print_result("Input File", str(image_file))
    print_result("File Size", f"{image_file.stat().st_size / 1024:.2f} KB")
    print_result("File Extension", image_file.suffix.lower())
    
    try:
        # Initialize configuration
        config = ServiceConfig()
        print_result("Configuration", "✅ Loaded successfully")

        # Initialize image processor
        processor = ImageProcessor(config)
        print_result("Image Processor", "✅ Initialized successfully")

        # Create processing configuration
        processing_config = ProcessingConfig(
            max_file_size=50 * 1024 * 1024,  # 50MB
            timeout=120.0,
            extract_metadata=True
        )
        print_result("Processing Config", "✅ Created successfully")
        
        print_section("Processing Image File")
        print("🔄 Starting image processing...")
        
        # Process the image file
        result = await processor.process_file(image_file, processing_config)
        
        if result.success:
            print("✅ Image processing completed successfully!")
            
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
                
                # Check for specific content types
                if "## Image Description" in result.content:
                    print_result("Image Description", "✅ Found")
                if "## OCR Text" in result.content:
                    print_result("OCR Text", "✅ Found")
                if "## Visual Elements" in result.content:
                    print_result("Visual Elements", "✅ Found")
            
            if result.summary:
                print_section("Summary")
                print(f"📝 {result.summary}")
            
            # Save results to file
            output_file = image_file.parent / f"{image_file.stem}_test_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'success': result.success,
                    'content_type': result.content_type,
                    'processing_time': result.processing_time,
                    'metadata': result.metadata,
                    'content': result.content,
                    'summary': result.summary,
                    'error': result.error
                }, f, indent=2, ensure_ascii=False)
            
            # Also save markdown content
            markdown_file = image_file.parent / f"{image_file.stem}_converted.md"
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(result.content)
            
            print_section("Output")
            print_result("Results saved to", str(output_file))
            print_result("Markdown saved to", str(markdown_file))
            
            return True
            
        else:
            print("❌ Image processing failed!")
            print_result("Error", result.error or "Unknown error")
            return False
            
    except Exception as e:
        print(f"❌ Error during image processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test-image.py <image_file>")
        print()
        print("Examples:")
        print("  python test-image.py my-image.jpg")
        print("  python test-image.py screenshot.png")
        print("  python test-image.py diagram.gif")
        print("  python test-image.py chart.bmp")
        sys.exit(1)
    
    image_file = Path(sys.argv[1])
    
    try:
        success = asyncio.run(test_image_processing(image_file))
        if success:
            print("\n🎉 Image processing test completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Image processing test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
