# MoRAG Document Processing

## Overview

The MoRAG Document Processing package is a component of the Modular Retrieval Augmented Generation (MoRAG) system. It provides functionality for processing various document types, extracting text, and preparing documents for embedding and retrieval using Microsoft's markitdown framework for optimal LLM compatibility.

## Features

- **Universal Document Support** (47+ formats) powered by markitdown:
  - **Document Formats**: PDF, Word (DOC/DOCX), Excel (XLS/XLSX), PowerPoint (PPT/PPTX)
  - **Text Formats**: HTML, Markdown, Plain text, CSV, JSON, XML
  - **Image Formats**: JPG, PNG, GIF, BMP, TIFF, WEBP, SVG (with OCR)
  - **Audio Formats**: MP3, WAV, M4A, FLAC, AAC, OGG (with transcription)
  - **Video Formats**: MP4, AVI, MOV, MKV, WEBM, FLV, WMV (with transcription)
  - **Archive Formats**: ZIP, EPUB, TAR, GZ, RAR, 7Z
- **LLM-Optimized Output**: Markdown format optimized for language models
- **High-Quality Conversion**: Advanced table extraction, OCR, structure preservation
- **Async Processing**: Full asynchronous support for scalable processing
- **Quality Assessment**: Automated quality scoring and validation
- **Flexible Chunking**: Configurable chunking strategies for optimal retrieval
- **Language Detection**: Automatic language detection and processing
- **Metadata Extraction**: Comprehensive metadata preservation

## Installation

```bash
pip install morag-document
```

## Usage

### Basic Usage

```python
from morag_document.processor import DocumentProcessor
from morag_core.config import Settings

# Initialize settings
settings = Settings()

# Create document processor
document_processor = DocumentProcessor()

# Process a PDF document
result = await document_processor.process_file("path/to/document.pdf")
print(f"Processed document with {len(result.document.chunks)} chunks")

# Process a document with custom chunking strategy
from morag_core.interfaces.converter import ChunkingStrategy

result = await document_processor.process_file(
    "path/to/document.docx",
    chunking_strategy=ChunkingStrategy.PARAGRAPH
)

# Access extracted text and metadata
document = result.document
print(f"Document title: {document.metadata.title}")
print(f"Document author: {document.metadata.author}")
print(f"Document language: {document.metadata.language}")
print(f"Total chunks: {len(document.chunks)}")

# Access individual chunks
for i, chunk in enumerate(document.chunks):
    print(f"Chunk {i}: {chunk.content[:100]}...")
```

## Architecture

### Markitdown Integration

This package uses Microsoft's markitdown framework as the core conversion engine, providing:

- **Universal Format Support**: Single framework handles 47+ file formats
- **LLM-Optimized Output**: Native Markdown output optimized for language models
- **High Quality**: Advanced table extraction, OCR, and structure preservation
- **Token Efficiency**: Markdown conventions are highly token-efficient for LLMs
- **Reliability**: Production-ready framework from Microsoft

### Converter Architecture

| Format Category | Converter | Supported Formats | Features |
|----------------|-----------|-------------------|----------|
| Documents | PDFConverter | PDF | Advanced table extraction, OCR |
| Documents | WordConverter | DOC, DOCX | Structure preservation, tables |
| Documents | ExcelConverter | XLS, XLSX | Spreadsheet to markdown conversion |
| Documents | PresentationConverter | PPT, PPTX | Slide content extraction |
| Text | TextConverter | TXT, MD, HTML, HTM | Encoding detection, structure preservation |
| Archives | ArchiveConverter | ZIP, EPUB, TAR, GZ, RAR, 7Z | Content extraction |
| Images | ImageConverter | JPG, PNG, GIF, BMP, TIFF, WEBP, SVG | OCR, metadata extraction |
| Audio | AudioConverter | MP3, WAV, M4A, FLAC, AAC, OGG | Transcription, metadata |
| Video | VideoConverter | MP4, AVI, MOV, MKV, WEBM, FLV, WMV | Audio extraction, transcription |

## Dependencies

### Core Dependencies
- **markitdown**: Universal document conversion framework
- **morag-core**: Core components and interfaces
- **morag-image**: Image processing capabilities
- **morag-audio**: Audio processing and transcription
- **morag-video**: Video processing and transcription

### Supporting Libraries
- **structlog**: Structured logging
- **spacy**: Advanced NLP capabilities
- **langdetect**: Language detection
- **beautifulsoup4**: HTML processing (via markitdown)
- **aiofiles**: Asynchronous file operations

## License

MIT
