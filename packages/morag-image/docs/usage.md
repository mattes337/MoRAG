# Morag Image Package Usage Guide

## Table of Contents

1. [Installation](#installation)
   - [Python Package Installation](#python-package-installation)
   - [System Dependencies](#system-dependencies)
   - [Docker Installation](#docker-installation)
2. [Configuration](#configuration)
   - [Google API Key](#google-api-key)
   - [OCR Engines](#ocr-engines)
3. [Basic Usage](#basic-usage)
   - [Command Line Interface](#command-line-interface)
   - [Python API](#python-api)
4. [Advanced Usage](#advanced-usage)
   - [Custom Configuration](#custom-configuration)
   - [Batch Processing](#batch-processing)
   - [Error Handling](#error-handling)
5. [Output Format](#output-format)
6. [Performance Considerations](#performance-considerations)
7. [Troubleshooting](#troubleshooting)

## Installation

### Python Package Installation

The morag-image package can be installed using pip:

```bash
pip install morag-image
```

For development installation:

```bash
git clone https://github.com/yourusername/morag.git
cd morag/packages/morag-image
pip install -e .
```

### System Dependencies

The package requires several system dependencies:

- **Tesseract OCR**: For text extraction
- **OpenCV dependencies**: For image processing

#### Linux (Ubuntu/Debian)

Use the provided installation script:

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

Or manually install:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
sudo apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx
sudo apt-get install -y python3-dev
```

#### Windows

Use the provided PowerShell script (run as Administrator):

```powershell
.\install_dependencies.ps1
```

Or manually install:

1. Install [Chocolatey](https://chocolatey.org/install)
2. Install Tesseract OCR: `choco install tesseract -y`
3. Install Visual C++ Redistributable: `choco install vcredist140 -y`

### Docker Installation

For a containerized setup with all dependencies included:

```bash
# Clone the repository (if not already done)
git clone https://github.com/yourusername/morag.git
cd morag/packages/morag-image

# Build the Docker image
docker build -t morag-image .

# Run with a single image
docker run -v /path/to/local/dir:/data -e GOOGLE_API_KEY=your-key morag-image \
  python -m morag_image.cli /data/image.jpg -o /data/output.json --caption --ocr
```

Alternatively, use docker-compose:

```bash
# Create a .env file with your Google API key
echo "GOOGLE_API_KEY=your-api-key-here" > .env

# Create a data directory
mkdir -p data

# Run with docker-compose
docker-compose run --rm morag-image \
  python -m morag_image.cli /data/image.jpg -o /data/output.json --caption --ocr
```

## Configuration

### Google API Key

To use the image captioning feature, you need a Google API key with access to the Gemini Vision API:

1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set it as an environment variable:

```bash
# Linux/macOS
export GOOGLE_API_KEY=your-api-key

# Windows (PowerShell)
$env:GOOGLE_API_KEY = "your-api-key"
```

### OCR Engines

The package supports two OCR engines:

1. **Tesseract** (default): Good for clear text in common languages
2. **EasyOCR**: Better for complex scenarios but slower

Select the engine using the `--ocr-engine` flag in CLI or the `ocr_engine` parameter in the API.

## Basic Usage

### Command Line Interface

Process a single image:

```bash
python -m morag_image.cli path/to/image.jpg -o results.json --caption --ocr --metadata
```

Process a directory of images:

```bash
python -m morag_image.cli path/to/images/ -o results.json --caption --ocr --metadata
```

Options:

- `--caption`: Generate image captions using Gemini Vision
- `--ocr`: Extract text from images
- `--metadata`: Extract image metadata
- `--ocr-engine`: Choose OCR engine (tesseract or easyocr)
- `--max-dimension`: Resize images to this maximum dimension
- `--max-concurrency`: Maximum number of concurrent image processing tasks

See the full CLI documentation in the [examples/cli_usage.md](../examples/cli_usage.md) file.

### Python API

Using the ImageProcessor directly:

```python
import asyncio
from morag_image.processor import ImageProcessor, ImageConfig
import os

async def process_image():
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Create processor
    processor = ImageProcessor(api_key=api_key)
    
    # Configure processing options
    config = ImageConfig(
        extract_metadata=True,
        extract_text=True,
        generate_caption=True,
        ocr_engine="tesseract",
        max_dimension=1024
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

Using the ImageService for batch processing:

```python
import asyncio
from morag_image.service import ImageService
from morag_core.config import ServiceConfig
import os
import json

async def process_batch():
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Create service config
    service_config = ServiceConfig(
        name="image_service",
        config={
            "api_key": api_key,
            "extract_metadata": True,
            "extract_text": True,
            "generate_caption": True,
            "ocr_engine": "tesseract",
            "max_dimension": 1024,
            "max_concurrency": 4
        }
    )
    
    # Initialize service
    service = ImageService(service_config)
    
    # Process a batch of images
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    results = await service.process_batch(image_paths)
    
    # Save results to file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

# Run the async function
asyncio.run(process_batch())
```

See more examples in the [examples](../examples/) directory.

## Advanced Usage

### Custom Configuration

You can customize the processing configuration for each image:

```python
from morag_image.processor import ImageConfig

# Create custom configurations
config1 = ImageConfig(
    extract_metadata=True,
    extract_text=False,
    generate_caption=True,
    ocr_engine="tesseract",
    max_dimension=800
)

config2 = ImageConfig(
    extract_metadata=True,
    extract_text=True,
    generate_caption=False,
    ocr_engine="easyocr",
    max_dimension=1200
)

# Process images with different configurations
result1 = await processor.process_image("image1.jpg", config1)
result2 = await processor.process_image("image2.jpg", config2)
```

### Batch Processing

For efficient batch processing with different configurations:

```python
from morag_image.service import ImageService
from morag_image.processor import ImageConfig

# Create configurations for each image
configs = {
    "image1.jpg": ImageConfig(extract_metadata=True, generate_caption=True),
    "image2.jpg": ImageConfig(extract_text=True, ocr_engine="easyocr"),
    "image3.jpg": ImageConfig(extract_metadata=True, extract_text=True)
}

# Process batch with individual configurations
results = await service.process_batch(list(configs.keys()), configs)
```

### Error Handling

Handle processing errors gracefully:

```python
from morag_image.processor import ProcessingError

try:
    result = await processor.process_image("image.jpg", config)
    print(f"Caption: {result.caption}")
except ProcessingError as e:
    print(f"Error processing image: {e}")
except FileNotFoundError:
    print("Image file not found")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Output Format

The processing results are returned as `ImageProcessingResult` objects with the following structure:

```python
class ImageProcessingResult:
    image_path: str  # Path to the processed image
    metadata: Optional[Dict[str, Any]]  # Image metadata (if extracted)
    text: Optional[str]  # Extracted text (if OCR was performed)
    caption: Optional[str]  # Generated caption (if captioning was enabled)
    error: Optional[str]  # Error message (if processing failed)
```

When saved to JSON (via CLI or service), the format is:

```json
{
  "image_path": "path/to/image.jpg",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "format": "JPEG",
    "mode": "RGB",
    "exif": { ... }
  },
  "text": "Extracted text from the image",
  "caption": "A generated caption describing the image content",
  "error": null
}
```

## Performance Considerations

- **Image Resizing**: Large images are automatically resized to improve processing speed. Use the `max_dimension` parameter to control this.
- **OCR Engine Selection**: Tesseract is faster but may be less accurate than EasyOCR for complex scenarios.
- **Concurrency**: Batch processing uses asyncio for concurrent processing. Control the concurrency level with `max_concurrency`.
- **Memory Usage**: Processing many large images simultaneously can consume significant memory. Adjust `max_concurrency` accordingly.

## Troubleshooting

### Common Issues

1. **Missing API Key**:
   - Error: "Google API key not provided"
   - Solution: Set the `GOOGLE_API_KEY` environment variable

2. **Tesseract Not Found**:
   - Error: "Tesseract not installed or not in PATH"
   - Solution: Install Tesseract and ensure it's in your system PATH

3. **EasyOCR Installation Issues**:
   - Error: "No module named 'easyocr'"
   - Solution: Install with `pip install easyocr`

4. **Memory Errors**:
   - Error: "MemoryError" or process killed
   - Solution: Reduce `max_concurrency` or process smaller batches

5. **Slow Processing**:
   - Issue: Processing takes too long
   - Solution: Use Tesseract instead of EasyOCR, reduce image dimensions, or disable unneeded features

### Getting Help

If you encounter issues not covered here, please:

1. Check the [examples](../examples/) directory for working code samples
2. Review the [test files](../tests/) to understand expected behavior
3. Open an issue on the GitHub repository with details about your problem