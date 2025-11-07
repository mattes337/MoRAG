"""Simple example of using morag-image package."""

import asyncio
import os
from pathlib import Path
import json

from morag_image.processor import ImageProcessor, ImageConfig
from morag_image.service import ImageService

async def process_single_image():
    """Process a single image file."""
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Captioning will not work.")

    # Create processor
    processor = ImageProcessor(api_key=api_key)

    # Define image path (replace with your own image)
    image_path = Path("path/to/your/image.jpg")
    if not image_path.exists():
        print(f"Error: Image file {image_path} does not exist.")
        return

    # Create configuration
    config = ImageConfig(
        generate_caption=True,
        extract_text=True,
        extract_metadata=True,
        ocr_engine="tesseract",  # or "easyocr"
        resize_max_dimension=1024
    )

    # Process image
    print(f"Processing image: {image_path}")
    result = await processor.process_image(image_path, config)

    # Print results
    print("\nProcessing Results:")
    print(f"Caption: {result.caption}")
    print(f"Extracted Text: {result.extracted_text}")
    print(f"Image Size: {result.metadata.width}x{result.metadata.height}")
    print(f"Format: {result.metadata.format}")
    print(f"Processing Time: {result.processing_time:.2f} seconds")

async def process_multiple_images():
    """Process multiple image files using the service."""
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY environment variable not set. Captioning will not work.")

    # Create service
    service = ImageService(api_key=api_key)

    # Define image directory (replace with your own directory)
    image_dir = Path("path/to/your/images")
    if not image_dir.is_dir():
        print(f"Error: Directory {image_dir} does not exist.")
        return

    # Find image files
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"Error: No image files found in {image_dir}")
        return

    print(f"Found {len(image_files)} image files")

    # Create configuration
    config = {
        "generate_caption": True,
        "extract_text": True,
        "extract_metadata": True,
        "ocr_engine": "tesseract",
        "resize_max_dimension": 1024
    }

    # Process images
    print("Processing images...")
    results = await service.process_batch(image_files, config, max_concurrency=3)

    # Print results
    print("\nProcessing Results:")
    for i, result in enumerate(results):
        print(f"\nImage {i+1}: {image_files[i].name}")
        print(f"Caption: {result['caption']}")
        print(f"Extracted Text: {result['extracted_text']}")
        print(f"Image Size: {result['metadata']['width']}x{result['metadata']['height']}")

    # Save results to JSON file
    output_file = image_dir / "image_processing_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    # Choose which example to run
    # asyncio.run(process_single_image())
    asyncio.run(process_multiple_images())
