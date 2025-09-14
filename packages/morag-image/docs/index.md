# Morag Image Package Documentation

Welcome to the documentation for the `morag-image` package, a component of the Morag project that provides image processing capabilities including metadata extraction, OCR (text extraction), and image captioning.

## Documentation Contents

- [Usage Guide](usage.md) - Comprehensive guide on how to use the package
- [API Reference](#api-reference) - Detailed API documentation
- [Examples](../examples/) - Code examples demonstrating package usage

## Overview

The `morag-image` package provides tools for extracting information from images:

- **Metadata Extraction**: Extract technical metadata from images (dimensions, format, EXIF data)
- **Optical Character Recognition (OCR)**: Extract text from images using Tesseract or EasyOCR
- **Image Captioning**: Generate descriptive captions using Google's Gemini Vision API
- **Image Preprocessing**: Resize and optimize images for processing

## Quick Start

### Installation

```bash
pip install morag-image
```

See the [Usage Guide](usage.md#installation) for detailed installation instructions, including system dependencies.

### Basic Usage

```python
import asyncio
from morag_image.processor import ImageProcessor, ImageConfig
import os

async def process_image():
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Create processor with API key
    processor = ImageProcessor(api_key=api_key)
    
    # Configure processing options
    config = ImageConfig(
        extract_metadata=True,
        extract_text=True,
        generate_caption=True
    )
    
    # Process image
    result = await processor.process_image("path/to/image.jpg", config)
    
    # Print results
    print(f"Caption: {result.caption}")
    print(f"Extracted text: {result.text}")
    print(f"Metadata: {result.metadata}")

# Run the async function
asyncio.run(process_image())
```

## Command Line Interface

The package also provides a command-line interface:

```bash
python -m morag_image.cli path/to/image.jpg -o results.json --caption --ocr --metadata
```

See the [CLI Usage Guide](../examples/cli_usage.md) for more details.

## API Reference

### Main Classes

- **ImageProcessor**: Core class for processing images
- **ImageService**: Service class for integrating with the Morag framework
- **ImageConfig**: Configuration class for customizing processing options
- **ImageMetadata**: Dataclass for storing image metadata
- **ImageProcessingResult**: Dataclass for storing processing results

For detailed API documentation, see the docstrings in the source code.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.