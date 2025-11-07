"""Command-line interface for morag-image package."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import structlog

from .processor import ImageProcessor, ImageConfig
from .service import ImageService

logger = structlog.get_logger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MoRAG Image Processing Tool")

    # Main arguments
    parser.add_argument(
        "input",
        help="Path to image file or directory containing images"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file for results (JSON format)"
    )

    # Processing options
    parser.add_argument(
        "--caption",
        action="store_true",
        help="Generate image captions"
    )

    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Extract text using OCR"
    )

    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Extract image metadata"
    )

    parser.add_argument(
        "--ocr-engine",
        choices=["tesseract", "easyocr"],
        default="tesseract",
        help="OCR engine to use"
    )

    parser.add_argument(
        "--api-key",
        help="API key for Gemini vision model"
    )

    parser.add_argument(
        "--max-dimension",
        type=int,
        default=1024,
        help="Maximum image dimension for processing"
    )

    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum number of concurrent processing tasks"
    )

    return parser.parse_args()

async def process_single_image(service: ImageService, file_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single image file."""
    try:
        result = await service.process_image(file_path, config)
        result["file_path"] = str(file_path)
        return result
    except Exception as e:
        logger.error("Failed to process image", file_path=str(file_path), error=str(e))
        return {"file_path": str(file_path), "error": str(e)}

async def process_directory(service: ImageService, dir_path: Path, config: Dict[str, Any], max_concurrency: int) -> List[Dict[str, Any]]:
    """Process all images in a directory."""
    # Find all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(dir_path.glob(f"**/*{ext}"))
        image_files.extend(dir_path.glob(f"**/*{ext.upper()}"))

    if not image_files:
        logger.warning("No image files found in directory", directory=str(dir_path))
        return []

    logger.info("Processing images in directory",
               directory=str(dir_path),
               image_count=len(image_files))

    # Process images in batches
    return await service.process_batch(image_files, config, max_concurrency=max_concurrency)

async def main():
    """Main entry point for CLI."""
    args = parse_args()

    # Create config from arguments
    config = {
        "generate_caption": args.caption,
        "extract_text": args.ocr,
        "extract_metadata": args.metadata,
        "resize_max_dimension": args.max_dimension,
        "ocr_engine": args.ocr_engine
    }

    # If no specific processing is requested, enable all
    if not (args.caption or args.ocr or args.metadata):
        config["generate_caption"] = True
        config["extract_text"] = True
        config["extract_metadata"] = True

    # Create service
    service = ImageService(api_key=args.api_key)

    # Process input (file or directory)
    input_path = Path(args.input)

    if input_path.is_file():
        results = [await process_single_image(service, input_path, config)]
    elif input_path.is_dir():
        results = await process_directory(service, input_path, config, args.max_concurrency)
    else:
        logger.error("Input path does not exist", input_path=str(input_path))
        sys.exit(1)

    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to file", output_file=str(output_path))
    else:
        # Print results to console
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ]
    )

    # Run main function
    asyncio.run(main())
