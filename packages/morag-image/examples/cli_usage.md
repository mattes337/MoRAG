# Using the morag-image CLI

The `morag-image` package provides a command-line interface (CLI) for processing images without writing Python code. This document demonstrates how to use the CLI for various image processing tasks.

## Prerequisites

1. Install the `morag-image` package:
   ```bash
   pip install morag-image
   ```

2. Set up the Google API key for captioning (optional):
   ```bash
   # On Linux/macOS
   export GOOGLE_API_KEY="your-api-key"
   
   # On Windows (Command Prompt)
   set GOOGLE_API_KEY=your-api-key
   
   # On Windows (PowerShell)
   $env:GOOGLE_API_KEY="your-api-key"
   ```

## Basic Usage

Process a single image with default settings (extract metadata, OCR text, and generate caption):

```bash
python -m morag_image.cli path/to/image.jpg
```

This will output the results in JSON format to the console.

## Saving Results to a File

Save the processing results to a JSON file:

```bash
python -m morag_image.cli path/to/image.jpg --output results.json
```

## Selective Processing

You can choose which processing features to enable:

```bash
# Only extract metadata
python -m morag_image.cli path/to/image.jpg --metadata

# Only perform OCR
python -m morag_image.cli path/to/image.jpg --ocr

# Only generate caption
python -m morag_image.cli path/to/image.jpg --caption

# Combine multiple options
python -m morag_image.cli path/to/image.jpg --metadata --ocr
```

## Choosing OCR Engine

Select which OCR engine to use:

```bash
# Use Tesseract OCR (default)
python -m morag_image.cli path/to/image.jpg --ocr --ocr-engine tesseract

# Use EasyOCR
python -m morag_image.cli path/to/image.jpg --ocr --ocr-engine easyocr
```

## Processing Multiple Images

Process all images in a directory:

```bash
python -m morag_image.cli path/to/images/ --output results.json
```

## Advanced Options

Adjust image preprocessing and concurrency:

```bash
# Set maximum image dimension for preprocessing
python -m morag_image.cli path/to/images/ --max-dimension 800

# Set maximum number of concurrent processing tasks
python -m morag_image.cli path/to/images/ --max-concurrency 5
```

## Complete Example

Process all images in a directory, extract metadata and text using EasyOCR, generate captions, resize images to 800px max dimension, process up to 4 images concurrently, and save results to a JSON file:

```bash
python -m morag_image.cli path/to/images/ \
  --metadata \
  --ocr \
  --caption \
  --ocr-engine easyocr \
  --max-dimension 800 \
  --max-concurrency 4 \
  --output results.json
```

## Viewing Results

The output JSON file contains an array of processing results, one for each image. Each result includes:

- `file_path`: Path to the processed image
- `caption`: Generated image caption (if enabled)
- `extracted_text`: Text extracted from the image (if enabled)
- `metadata`: Image metadata including dimensions, format, etc.
- `processing_time`: Time taken to process the image
- `confidence_scores`: Confidence scores for OCR and other processes