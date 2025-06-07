# MoRAG Image Processing

Image processing capabilities for the MoRAG (Multimodal RAG Ingestion Pipeline) system.

## Overview

The `morag-image` package provides comprehensive image processing capabilities for the MoRAG system, including:

- Image metadata extraction
- OCR (Optical Character Recognition) for text extraction
- Image captioning using vision models
- Image preprocessing and optimization

## Features

- **Metadata Extraction**: Extract comprehensive metadata from images including EXIF data
- **OCR Processing**: Extract text from images using Tesseract or EasyOCR
- **Image Captioning**: Generate descriptive captions using Gemini Pro Vision
- **Image Preprocessing**: Resize, optimize, and prepare images for processing

## Dependencies

- morag-core: Core components for MoRAG
- Pillow: Image processing library
- pytesseract: OCR text extraction
- easyocr: Alternative OCR engine
- opencv-python: Computer vision operations
- google-generativeai: Vision capabilities for image captioning

## System Requirements

### OCR Dependencies

For Tesseract OCR functionality:
- tesseract-ocr: OCR engine
- tesseract-ocr-eng: English language data (additional language packs can be installed as needed)

### OpenCV Dependencies

For OpenCV functionality:
- libgl1-mesa-glx: OpenGL support
- libglib2.0-0: GLib library
- libsm6: Session Management library
- libxext6: X11 extension library
- libxrender-dev: X Rendering Extension library

## Usage

```python
from pathlib import Path
from morag_image.processor import ImageProcessor, ImageConfig

# Create processor instance
processor = ImageProcessor()

# Configure processing options
config = ImageConfig(
    generate_caption=True,
    extract_text=True,
    extract_metadata=True,
    ocr_engine="tesseract"  # or "easyocr"
)

# Process an image
async def process_image():
    result = await processor.process_image(
        file_path=Path("path/to/image.jpg"),
        config=config
    )
    
    # Access results
    print(f"Caption: {result.caption}")
    print(f"Extracted text: {result.extracted_text}")
    print(f"Image dimensions: {result.metadata.width}x{result.metadata.height}")
```

## License

MIT