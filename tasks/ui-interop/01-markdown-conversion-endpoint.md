# Markdown Conversion Endpoint

## Overview
Implement a lightweight REST endpoint that converts input files to intermediate markdown format without running the full processing pipeline.

## Endpoint Specification
**Endpoint**: `POST /api/convert/markdown`

**Purpose**: Convert input files to intermediate markdown format for UI preview functionality.

## Input Requirements
- File upload (multipart/form-data)
- Optional parameters for conversion settings
- Support for all current input formats (PDF, audio, video, text, etc.)

## Output Requirements
- Markdown content (text/plain or application/json with metadata)
- Conversion metadata including:
  - File type and format
  - Processing time
  - File size
  - Page count (for PDFs)
  - Duration (for audio/video)

## Implementation Requirements

### File Format Support
- **PDF**: Use existing docling for PDF-to-markdown conversion
- **Audio/Video**: Use existing transcription services to generate markdown with timestamps
- **Text**: Direct conversion with minimal processing
- **Other formats**: Extend as needed based on existing MoRAG processors

### Performance Requirements
- Fast response time optimized for UI preview functionality
- No chunking or graph extraction processing
- Minimal memory footprint
- Streaming response for large files

### API Response Format
```json
{
  "markdown": "# Document Title\n\nContent...",
  "metadata": {
    "original_format": "pdf|mp4|mp3|txt|...",
    "file_size_bytes": 1048576,
    "processing_time_ms": 1500,
    "page_count": 10,
    "duration_seconds": 3600,
    "language": "en"
  }
}
```

## Error Handling
- File format validation
- File size limits
- Processing timeout handling
- Clear error messages for unsupported formats

## Security Considerations
- File type validation
- Size limits per file type
- Rate limiting per client
- Input sanitization

## Testing Requirements
- Unit tests for each file format conversion
- Performance benchmarks
- Error scenario testing
- Large file handling tests

## Dependencies
- Existing MoRAG file processors
- Docling for PDF processing
- Audio/video transcription services
- FastAPI/Flask framework
