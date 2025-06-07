# MoRAG Document Processing

## Overview

The MoRAG Document Processing package is a component of the Modular Retrieval Augmented Generation (MoRAG) system. It provides functionality for processing various document types, extracting text, and preparing documents for embedding and retrieval.

## Features

- Support for multiple document formats:
  - PDF documents
  - Microsoft Word documents (.docx)
  - Microsoft Excel spreadsheets (.xlsx)
  - Microsoft PowerPoint presentations (.pptx)
  - HTML documents
  - Markdown files
  - Plain text files
  - CSV files
- Text extraction with metadata preservation
- Document chunking with configurable strategies
- Language detection
- Quality assessment
- Document summarization

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

## Dependencies

- morag-core: Core components for MoRAG
- morag-embedding: Embedding service for MoRAG
- pypdf: PDF processing
- python-docx: Word document processing
- openpyxl: Excel spreadsheet processing
- python-pptx: PowerPoint presentation processing
- beautifulsoup4: HTML processing
- markdown: Markdown processing
- nltk: Natural language processing
- spacy: Advanced NLP capabilities
- langdetect: Language detection
- aiohttp: Asynchronous HTTP client/server
- structlog: Structured logging

## License

MIT