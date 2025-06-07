#!/usr/bin/env python3
"""
MoRAG Image Processing Test Script

Supports both processing (immediate results) and ingestion (background + storage) modes.

Usage:
    python test-image.py <image_file> [options]

Processing Mode (immediate results):
    python test-image.py my-image.jpg
    python test-image.py screenshot.png
    python test-image.py diagram.gif

Ingestion Mode (background processing + storage):
    python test-image.py my-image.jpg --ingest
    python test-image.py screenshot.png --ingest --metadata '{"type": "screenshot"}'
    python test-image.py diagram.png --ingest --webhook-url https://my-app.com/webhook

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
import requests

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
                'mode': 'processing',
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


async def test_image_ingestion(image_file: Path, webhook_url: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Test image ingestion functionality."""
    print_header("MoRAG Image Ingestion Test")

    if not image_file.exists():
        print(f"❌ Error: Image file not found: {image_file}")
        return False

    print_result("Input File", str(image_file))
    print_result("File Size", f"{image_file.stat().st_size / 1024:.2f} KB")
    print_result("File Extension", image_file.suffix.lower())
    print_result("Webhook URL", webhook_url or "None")
    print_result("Metadata", json.dumps(metadata, indent=2) if metadata else "None")

    try:
        print_section("Submitting Ingestion Task")
        print("🔄 Starting image ingestion...")

        # Prepare form data
        files = {'file': open(image_file, 'rb')}
        data = {'source_type': 'image'}

        if webhook_url:
            data['webhook_url'] = webhook_url
        if metadata:
            data['metadata'] = json.dumps(metadata)

        # Submit to ingestion API
        response = requests.post(
            'http://localhost:8000/api/v1/ingest/file',
            files=files,
            data=data,
            timeout=30
        )

        files['file'].close()

        if response.status_code == 200:
            result = response.json()
            print("✅ Image ingestion task submitted successfully!")

            print_section("Ingestion Results")
            print_result("Status", "✅ Success")
            print_result("Task ID", result.get('task_id', 'Unknown'))
            print_result("Message", result.get('message', 'Task created'))
            print_result("Estimated Time", f"{result.get('estimated_time', 'Unknown')} seconds")

            # Save ingestion result
            output_file = image_file.parent / f"{image_file.stem}_ingest_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'mode': 'ingestion',
                    'task_id': result.get('task_id'),
                    'status': result.get('status'),
                    'message': result.get('message'),
                    'estimated_time': result.get('estimated_time'),
                    'webhook_url': webhook_url,
                    'metadata': metadata,
                    'file_path': str(image_file)
                }, f, indent=2, ensure_ascii=False)

            print_section("Output")
            print_result("Ingestion result saved to", str(output_file))
            print_result("Monitor task status", f"curl http://localhost:8000/api/v1/status/{result.get('task_id')}")

            return True
        else:
            print("❌ Image ingestion failed!")
            print_result("Status Code", str(response.status_code))
            print_result("Error", response.text)
            return False

    except Exception as e:
        print(f"❌ Error during image ingestion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MoRAG Image Processing Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Processing Mode (immediate results):
    python test-image.py my-image.jpg
    python test-image.py screenshot.png
    python test-image.py diagram.gif

  Ingestion Mode (background processing + storage):
    python test-image.py my-image.jpg --ingest
    python test-image.py screenshot.png --ingest --metadata '{"type": "screenshot"}'
    python test-image.py diagram.png --ingest --webhook-url https://my-app.com/webhook
        """
    )

    parser.add_argument('image_file', help='Path to image file')
    parser.add_argument('--ingest', action='store_true',
                       help='Enable ingestion mode (background processing + storage)')
    parser.add_argument('--webhook-url', help='Webhook URL for completion notifications (ingestion mode only)')
    parser.add_argument('--metadata', help='Additional metadata as JSON string (ingestion mode only)')

    args = parser.parse_args()

    image_file = Path(args.image_file)

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
            success = asyncio.run(test_image_ingestion(
                image_file,
                webhook_url=args.webhook_url,
                metadata=metadata
            ))
            if success:
                print("\n🎉 Image ingestion test completed successfully!")
                print("💡 Use the task ID to monitor progress and retrieve results.")
                sys.exit(0)
            else:
                print("\n💥 Image ingestion test failed!")
                sys.exit(1)
        else:
            # Processing mode
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
