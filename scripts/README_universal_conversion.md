# Universal Document Conversion Test Scripts

This directory contains test scripts for the universal document conversion pipeline implemented in Task 24.

## Scripts

### `test_universal_conversion.py`

A comprehensive test script that converts any supported document format to markdown with detailed output and quality metrics.

**Usage:**
```bash
python test_universal_conversion.py <file_path> [options]
```

**Examples:**
```bash
# Convert a PDF with metadata
python test_universal_conversion.py document.pdf --include-metadata --verbose

# Convert text with sentence-level chunking
python test_universal_conversion.py document.txt --chunking-strategy sentence

# Convert with custom output directory and suffix
python test_universal_conversion.py presentation.pptx --output-dir ./converted --output-suffix _processed
```

**Options:**
- `--output-suffix`: Suffix for output filename (default: `_converted`)
- `--chunking-strategy`: Chunking strategy (`page`, `sentence`, `paragraph`, `semantic`)
- `--include-metadata`: Include metadata in conversion
- `--extract-images`: Extract images during conversion
- `--quality-threshold`: Minimum quality threshold (default: 0.7)
- `--verbose`: Enable verbose output
- `--output-dir`: Output directory (default: same as input file)

**Output:**
- Creates a markdown file with the specified suffix
- Creates a JSON metadata file with conversion details and quality metrics
- Displays comprehensive conversion statistics

### `demo_universal_conversion.py`

A demonstration script that shows various features of the universal conversion system.

**Usage:**
```bash
python demo_universal_conversion.py
```

**Features Demonstrated:**
- Format detection for various file types
- Available converters and supported formats
- Text conversion with different chunking strategies
- Quality assessment examples
- PDF conversion (if files available)

## Supported Formats

The universal conversion pipeline supports:

| Format | Extensions | Converter | Features |
|--------|------------|-----------|----------|
| PDF | `.pdf` | Enhanced PDF Converter | Docling integration, OCR, table extraction |
| Text | `.txt`, `.md` | Basic Text Converter | Multiple chunking strategies |
| Office | `.docx`, `.xlsx`, `.pptx` | Office Converter | Word, Excel, PowerPoint support |
| Audio | `.mp3`, `.wav`, `.m4a` | Audio Converter | Whisper transcription, speaker diarization |
| Video | `.mp4`, `.avi`, `.mov` | Video Converter | Audio extraction, keyframe analysis |
| Web | `.html`, `.htm` | Web Converter | Content extraction, cleaning |

## Quality Metrics

Each conversion is assessed on:

- **Completeness** (0.0-1.0): How much of the original content was preserved
- **Readability** (0.0-1.0): How well-structured the output markdown is
- **Structure** (0.0-1.0): Whether document hierarchy is maintained
- **Metadata Preservation** (0.0-1.0): How much metadata was extracted
- **Overall Score** (0.0-1.0): Weighted average of all metrics

Quality interpretation:
- 🟢 **High Quality** (≥0.8): Excellent conversion
- 🟡 **Acceptable Quality** (0.6-0.8): Good conversion with minor issues
- 🔴 **Low Quality** (<0.6): Poor conversion, may need manual review

## Chunking Strategies

The system supports different chunking strategies:

- **Page**: Split content by pages (default for PDFs)
- **Paragraph**: Split by paragraph breaks
- **Sentence**: Split by sentence boundaries
- **Semantic**: AI-powered semantic chunking (advanced)

## Output Format

The generated markdown follows a consistent structure:

```markdown
# Document Title

## Document Information
**Source**: original_file.pdf
**Format**: PDF → Markdown
**Pages**: 15
**Quality Score**: 0.85

## Content
### Page 1
[Content from page 1...]

### Page 2
[Content from page 2...]

## Processing Details
**Conversion Method**: Enhanced PDF Processing
**Chunking Strategy**: page
```

## Error Handling

The scripts include comprehensive error handling:

- File validation (existence, format support)
- Conversion error recovery with fallback converters
- Quality threshold enforcement
- Detailed error reporting with suggestions

## Integration

These scripts demonstrate the universal conversion API that can be integrated into:

- MoRAG ingestion pipeline
- Batch document processing workflows
- Web applications via REST API
- Command-line tools and automation scripts

## Dependencies

The conversion system requires:

- Core: `structlog`, `pathlib`
- PDF: `docling` (preferred), `pymupdf`, `pdfplumber`
- Audio: `whisper`, `pydub`, `ffmpeg`
- Video: `opencv-python`, `ffmpeg`
- Office: `python-docx`, `openpyxl`, `python-pptx`
- Web: `playwright`, `trafilatura`, `newspaper3k`
- Text: Built-in Python libraries

## Performance

Typical processing times:
- Text files: <1 second
- PDF files: 10-60 seconds (depending on size and complexity)
- Audio files: 2-5x real-time duration
- Video files: 1-3x real-time duration
- Office documents: 5-30 seconds

## Troubleshooting

Common issues and solutions:

1. **"Unsupported format" error**: Check if the file extension is supported
2. **"Quality below threshold" warning**: Lower the quality threshold or check input quality
3. **"Conversion failed" error**: Check file integrity and dependencies
4. **Slow processing**: Large files may take time; consider chunking or compression

For more details, see the main documentation in `docs/UNIVERSAL_DOCUMENT_CONVERSION.md`.
