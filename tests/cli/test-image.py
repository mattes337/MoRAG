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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from morag_image import ImageProcessor
    from morag_image.processor import ImageConfig
    from morag_core.interfaces.processor import ProcessingConfig
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
        # Initialize image processor (no API key for basic functionality)
        processor = ImageProcessor()
        print_result("Image Processor", "✅ Initialized successfully")

        # Create image configuration (disable OCR since tesseract is not installed)
        image_config = ImageConfig(
            generate_caption=False,  # Requires API key
            extract_text=False,  # Requires tesseract
            extract_metadata=True,
            resize_max_dimension=1024
        )
        print_result("Image Config", "✅ Created successfully")

        print_section("Processing Image File")
        print("🔄 Starting image processing...")

        # Process the image file
        result = await processor.process_image(image_file, image_config)

        print("✅ Image processing completed successfully!")

        print_section("Processing Results")
        print_result("Status", "✅ Success")
        print_result("Processing Time", f"{result.processing_time:.2f} seconds")

        print_section("Image Information")
        if result.caption:
            print_result("Caption", result.caption)
        else:
            print_result("Caption", "Not generated (requires API key)")

        if result.extracted_text:
            print_result("Extracted Text", result.extracted_text)
        else:
            print_result("Extracted Text", "None found (OCR disabled)")

        print_section("Image Metadata")
        metadata = result.metadata
        print_result("Width", f"{metadata.width}px")
        print_result("Height", f"{metadata.height}px")
        print_result("Format", metadata.format)
        print_result("Mode", metadata.mode)
        print_result("File Size", f"{metadata.file_size / 1024:.2f} KB")
        print_result("Has EXIF", "✅ Yes" if metadata.has_exif else "❌ No")

        if metadata.camera_make:
            print_result("Camera Make", metadata.camera_make)
        if metadata.camera_model:
            print_result("Camera Model", metadata.camera_model)
        if metadata.creation_time:
            print_result("Creation Time", metadata.creation_time)

        if result.confidence_scores:
            print_section("Confidence Scores")
            for key, score in result.confidence_scores.items():
                print_result(key, f"{score:.3f}")

        # Save results to file
        output_file = image_file.parent / f"{image_file.stem}_test_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'processing_time': result.processing_time,
                'caption': result.caption,
                'extracted_text': result.extracted_text,
                'metadata': {
                    'width': metadata.width,
                    'height': metadata.height,
                    'format': metadata.format,
                    'mode': metadata.mode,
                    'file_size': metadata.file_size,
                    'has_exif': metadata.has_exif,
                    'exif_data': metadata.exif_data,
                    'creation_time': metadata.creation_time,
                    'camera_make': metadata.camera_make,
                    'camera_model': metadata.camera_model
                },
                'confidence_scores': result.confidence_scores,
                'temp_files': [str(f) for f in result.temp_files]
            }, f, indent=2, ensure_ascii=False)

        print_section("Output")
        print_result("Results saved to", str(output_file))

        return True
            
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
