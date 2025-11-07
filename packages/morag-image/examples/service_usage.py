"""Example of using the morag-image service interface."""

import asyncio
import json
import os
from pathlib import Path

from morag_image.service import ImageService


async def main():
    """Demonstrate the use of ImageService."""
    # Get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "Warning: GOOGLE_API_KEY environment variable not set. Captioning will not work."
        )

    # Create service
    service = ImageService(api_key=api_key)

    # Example 1: Process a single image
    print("\nExample 1: Process a single image")
    image_path = Path("path/to/your/image.jpg")  # Replace with your image path

    # Skip if file doesn't exist
    if not image_path.exists():
        print(f"Image file {image_path} does not exist. Skipping Example 1.")
    else:
        # Define configuration
        config = {
            "generate_caption": True,
            "extract_text": True,
            "extract_metadata": True,
            "ocr_engine": "tesseract",
        }

        # Process image
        try:
            result = await service.process_image(image_path, config)

            # Print results
            print(f"Caption: {result['caption']}")
            print(f"Extracted Text: {result['extracted_text']}")
            print(
                f"Image Size: {result['metadata']['width']}x{result['metadata']['height']}"
            )
            print(f"Format: {result['metadata']['format']}")
            print(f"Processing Time: {result['processing_time']:.2f} seconds")
        except Exception as e:
            print(f"Error processing image: {e}")

    # Example 2: Process multiple images with different configurations
    print("\nExample 2: Process multiple images with different configurations")

    # Define image directory
    image_dir = Path("path/to/your/images")  # Replace with your directory path

    # Skip if directory doesn't exist
    if not image_dir.is_dir():
        print(f"Directory {image_dir} does not exist. Skipping Example 2.")
    else:
        # Find image files
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

        if not image_files:
            print(f"No image files found in {image_dir}. Skipping Example 2.")
        else:
            print(f"Found {len(image_files)} image files")

            # Create different configurations for different processing needs
            configs = [
                # Config 1: Full processing
                {
                    "generate_caption": True,
                    "extract_text": True,
                    "extract_metadata": True,
                    "ocr_engine": "tesseract",
                    "resize_max_dimension": 1024,
                },
                # Config 2: Metadata only
                {
                    "generate_caption": False,
                    "extract_text": False,
                    "extract_metadata": True,
                },
                # Config 3: OCR only with EasyOCR
                {
                    "generate_caption": False,
                    "extract_text": True,
                    "extract_metadata": False,
                    "ocr_engine": "easyocr",
                },
            ]

            # Process each image with each configuration
            all_results = []

            for i, image_file in enumerate(image_files[:3]):  # Process up to 3 images
                print(f"\nProcessing image: {image_file.name}")

                # Use a different config for each image (cycling through configs)
                config = configs[i % len(configs)]
                config_type = ["Full processing", "Metadata only", "OCR only"][
                    i % len(configs)
                ]

                print(f"Using configuration: {config_type}")

                try:
                    result = await service.process_image(image_file, config)

                    # Add file name to result
                    result["file_name"] = image_file.name
                    result["config_type"] = config_type

                    # Print key results
                    if config["generate_caption"]:
                        print(f"Caption: {result['caption']}")
                    if config["extract_text"]:
                        print(
                            f"Extracted Text: {result['extracted_text'][:100]}..."
                            if len(result["extracted_text"]) > 100
                            else f"Extracted Text: {result['extracted_text']}"
                        )
                    if config["extract_metadata"]:
                        print(
                            f"Image Size: {result['metadata']['width']}x{result['metadata']['height']}"
                        )

                    all_results.append(result)
                except Exception as e:
                    print(f"Error processing image: {e}")

            # Save all results to a JSON file
            if all_results:
                output_file = image_dir / "service_example_results.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)
                print(f"\nResults saved to {output_file}")

    # Example 3: Batch processing with the same configuration
    print("\nExample 3: Batch processing with the same configuration")

    # Skip if directory doesn't exist or no images were found
    if not image_dir.is_dir() or not image_files:
        print(
            f"Directory {image_dir} does not exist or no images found. Skipping Example 3."
        )
    else:
        # Use a single configuration for all images
        config = {
            "generate_caption": True,
            "extract_text": True,
            "extract_metadata": True,
            "ocr_engine": "tesseract",
            "resize_max_dimension": 800,
        }

        try:
            # Process all images in batch
            print(f"Processing {len(image_files)} images in batch...")
            results = await service.process_batch(
                image_files, config, max_concurrency=3
            )

            # Print summary
            print(f"Processed {len(results)} images successfully")

            # Save batch results to a JSON file
            output_file = image_dir / "batch_processing_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error in batch processing: {e}")


if __name__ == "__main__":
    asyncio.run(main())
