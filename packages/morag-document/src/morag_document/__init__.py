"""MoRAG Document Processing.

MoRAG Document Processing provides comprehensive support for converting various
document formats to unified markdown. This package handles PDF, Word, Excel,
PowerPoint, text files, and other document formats with advanced features like
OCR, table extraction, and structure preservation.

## Supported Formats

### Document Types
- **PDF**: Text extraction, OCR for scanned documents, table detection
- **Microsoft Office**: Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- **Text Files**: Plain text, markdown, RTF, CSV
- **Archives**: ZIP, TAR with recursive extraction
- **Presentations**: PowerPoint with slide-by-slide conversion

### Processing Features
- **OCR Support**: Automatic OCR for scanned PDFs and images
- **Table Extraction**: Intelligent table detection and formatting
- **Structure Preservation**: Headers, lists, formatting maintained
- **Metadata Extraction**: Document properties, creation dates
- **Error Recovery**: Fallback mechanisms for corrupted files

## Usage Examples

### Basic Document Processing
```python
from morag_document import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process a PDF document
result = await processor.process_file(
    file_path="document.pdf",
    output_format="markdown"
)

print(f"Converted content length: {len(result.content)}")
print(f"Extracted metadata: {result.metadata}")
```

### Service-Based Processing
```python
from morag_document import DocumentService

# Initialize service
service = DocumentService()

# Process with custom configuration
config = {
    "extract_tables": True,
    "preserve_formatting": True,
    "enable_ocr": True
}

result = await service.convert_to_markdown(
    input_path="presentation.pptx",
    config=config
)
```

### Advanced Processing Options
```python
from morag_document import DocumentProcessor

processor = DocumentProcessor()

# Configure processing options
options = {
    "ocr_enabled": True,
    "table_extraction": True,
    "preserve_images": True,
    "extract_metadata": True,
    "chunk_size": 2000
}

result = await processor.process_document(
    file_path="complex_document.pdf",
    options=options
)

# Access structured results
for page in result.pages:
    print(f"Page {page.number}: {page.content[:100]}...")

for table in result.tables:
    print(f"Table: {table.rows}x{table.columns}")
```

### Batch Processing
```python
from morag_document import DocumentService
from pathlib import Path

service = DocumentService()

# Process multiple documents
document_paths = [
    "report1.pdf",
    "spreadsheet.xlsx",
    "presentation.pptx"
]

results = []
for path in document_paths:
    result = await service.convert_to_markdown(path)
    results.append(result)
    print(f"Processed: {Path(path).name}")
```

## Configuration

### Environment Variables
```bash
# OCR Configuration
MORAG_OCR_ENABLED=true
MORAG_OCR_LANGUAGE=eng

# Processing Configuration
MORAG_EXTRACT_TABLES=true
MORAG_PRESERVE_FORMATTING=true
MORAG_MAX_FILE_SIZE=100MB

# Fallback Configuration
MORAG_ENABLE_FALLBACK=true
MORAG_MARKITDOWN_ENABLED=true
```

### Programmatic Configuration
```python
from morag_document import DocumentProcessor

config = {
    "ocr_enabled": True,
    "extract_tables": True,
    "preserve_formatting": True,
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "supported_formats": [".pdf", ".docx", ".xlsx", ".pptx"]
}

processor = DocumentProcessor(config=config)
```

## Converters

### PDF Converter
Advanced PDF processing with OCR support:
```python
from morag_document.converters import PDFConverter

converter = PDFConverter(enable_ocr=True)
result = await converter.convert("scanned_document.pdf")

# Access page-level results
for page in result.pages:
    print(f"Page {page.number}: {page.text_confidence}")
```

### Excel Converter
Spreadsheet processing with sheet detection:
```python
from morag_document.converters import ExcelConverter

converter = ExcelConverter()
result = await converter.convert("data.xlsx")

# Access sheet-level results
for sheet in result.sheets:
    print(f"Sheet '{sheet.name}': {sheet.row_count} rows")
```

### Word Converter
Document processing with style preservation:
```python
from morag_document.converters import WordConverter

converter = WordConverter(preserve_formatting=True)
result = await converter.convert("report.docx")

# Access structured content
print(f"Headers found: {len(result.headers)}")
print(f"Tables found: {len(result.tables)}")
```

## Error Handling

The package provides comprehensive error handling:
```python
from morag_document import DocumentProcessor
from morag_document.exceptions import DocumentProcessingError

processor = DocumentProcessor()

try:
    result = await processor.process_file("document.pdf")
except DocumentProcessingError as e:
    print(f"Processing failed: {e.message}")
    print(f"Error code: {e.code}")

    # Try fallback converter
    if e.fallback_available:
        result = await processor.process_with_fallback("document.pdf")
```

## Integration with MoRAG Stages

The document processor integrates seamlessly with the stage-based system:
```python
# Via CLI
python cli/morag-stages.py stage markdown-conversion document.pdf

# Via Python API
from morag_stages import StageManager

manager = StageManager()
result = await manager.execute_stage(
    "markdown-conversion",
    input_path="document.pdf",
    output_dir="./output"
)
```

## CLI Usage

```bash
# Convert single document
python -m morag_document convert document.pdf --output output.md

# Batch convert directory
python -m morag_document batch /path/to/documents --output-dir ./converted

# Convert with OCR enabled
python -m morag_document convert scanned.pdf --ocr --output text.md

# Extract tables to CSV
python -m morag_document extract-tables report.pdf --format csv
```

## Performance Optimization

### Memory Management
- Streaming processing for large files
- Page-by-page processing for PDFs
- Automatic garbage collection

### Processing Speed
- Parallel processing for batch operations
- Caching of expensive operations
- Fast path for text-based documents

### Resource Usage
- Configurable memory limits
- CPU usage optimization
- Disk space management

## Installation

```bash
pip install morag-document
```

Or as part of the full MoRAG system:
```bash
pip install packages/morag/
```

## Dependencies

- MarkItDown for universal document conversion
- pytesseract for OCR capabilities
- python-docx for Word documents
- openpyxl for Excel files
- PyPDF2/pdfplumber for PDF processing

## Version

Current version: {__version__}
"""

from .processor import DocumentProcessor
from .service import DocumentService

__version__ = "0.1.0"

__all__ = ["DocumentProcessor", "DocumentService"]