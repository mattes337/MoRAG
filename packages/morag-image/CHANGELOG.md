# Changelog

All notable changes to the `morag-image` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2023-07-15

### Added

- Initial release of the morag-image package
- Core functionality for image processing:
  - Metadata extraction from images
  - OCR text extraction using Tesseract and EasyOCR
  - Image captioning using Google's Gemini Vision API
  - Image preprocessing (resizing)
- Command-line interface for processing single images and directories
- ImageService class for integration with the Morag framework
- Comprehensive documentation and examples
- Docker support for containerized deployment
- Installation scripts for system dependencies

### Dependencies

- Python 3.8+
- morag-core
- Pillow for image processing
- pytesseract and EasyOCR for text extraction
- google-generativeai for image captioning
- opencv-python for image preprocessing
- structlog for logging
- aiofiles for asynchronous file operations

## [Unreleased]

### Planned Features

- Support for additional OCR engines
- Image classification capabilities
- Object detection in images
- Face detection and recognition
- Image similarity comparison
- Batch processing improvements for large datasets
- Performance optimizations
- Additional output formats beyond JSON
