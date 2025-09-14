# Universal Document Conversion Framework

## Overview

The Universal Document Conversion Framework is a comprehensive system for converting various document formats to structured markdown. It provides a unified API for handling PDFs, audio files, videos, office documents, and web content with consistent quality assessment and metadata extraction.

## Architecture

### Core Components

1. **BaseConverter**: Abstract base class for all converters
2. **DocumentConverter**: Registry and orchestrator for format-specific converters
3. **ConversionOptions**: Configuration for conversion behavior
4. **ConversionResult**: Standardized result format with quality metrics
5. **QualityValidator**: Assessment of conversion quality

### Supported Formats

| Format | Converter | Status | Features |
|--------|-----------|--------|----------|
| PDF | PDFConverter | ✅ Active | Docling integration, OCR, table extraction |
| Audio | AudioConverter | ✅ Active | Whisper transcription, speaker diarization |
| Video | VideoConverter | ✅ Active | Audio extraction, keyframe analysis |
| Office | OfficeConverter | 🔄 Placeholder | Word, Excel, PowerPoint support |
| Web | WebConverter | ⚠️ Limited | HTML parsing, content extraction |

## Quick Start

### Basic Usage

```python
from morag.converters import DocumentConverter, ConversionOptions

# Initialize converter
converter = DocumentConverter()

# Create options
options = ConversionOptions.for_format('pdf')
options.chunking_strategy = ChunkingStrategy.PAGE
options.include_metadata = True

# Convert document
result = await converter.convert_to_markdown('document.pdf', options)

if result.success:
    print(f"Converted {result.word_count} words")
    print(f"Quality score: {result.quality_score.overall_score:.2f}")
    print(result.content)
```

### Using the Service Layer

```python
from morag.services.universal_converter import universal_converter_service

# Convert with embeddings
result = await universal_converter_service.convert_document(
    'document.pdf',
    generate_embeddings=True
)

print(f"Generated {len(result['chunks'])} chunks")
print(f"Generated {len(result['embeddings'])} embeddings")
```

## Configuration

### Default Configuration

```yaml
default_options:
  preserve_formatting: true
  extract_images: true
  include_metadata: true
  chunking_strategy: "page"
  min_quality_threshold: 0.7
  enable_fallback: true

format_specific:
  pdf:
    use_docling: true
    use_ocr: true
    extract_tables: true
  
  audio:
    enable_diarization: false
    include_timestamps: true
    confidence_threshold: 0.8
  
  video:
    extract_keyframes: true
    include_audio: true
```

### Chunking Strategies

- **PAGE**: One chunk per page (default for documents)
- **SENTENCE**: Sentence-based chunking
- **PARAGRAPH**: Paragraph-based chunking  
- **SEMANTIC**: AI-powered semantic chunking

## Quality Assessment

The system provides comprehensive quality scoring:

```python
quality_score = result.quality_score

print(f"Overall: {quality_score.overall_score:.2f}")
print(f"Completeness: {quality_score.completeness_score:.2f}")
print(f"Readability: {quality_score.readability_score:.2f}")
print(f"Structure: {quality_score.structure_score:.2f}")
print(f"Metadata: {quality_score.metadata_preservation:.2f}")
```

### Quality Thresholds

- **0.8+**: High quality conversion
- **0.6-0.8**: Acceptable quality
- **<0.6**: Low quality, may trigger fallback

## API Endpoints

### Convert Document

```http
POST /api/v1/process
Content-Type: multipart/form-data

file: document.pdf
request_data: {
  "mode": "convert",
  "source_type": "file",
  "processing_options": {
    "chunking_strategy": "page",
    "include_metadata": true
  }
}
```

### Process Document with Full Pipeline

```http
POST /api/v1/process
Content-Type: multipart/form-data

file: document.pdf
request_data: {
  "mode": "process",
  "source_type": "file",
  "processing_options": {
    "chunking_strategy": "page",
    "include_metadata": true
  }
}
```

### Batch Processing

```http
POST /api/v1/process
Content-Type: multipart/form-data

request_data: {
  "mode": "process",
  "source_type": "batch",
  "items": [
    {"url": "https://example.com/doc1.pdf", "type": "document"},
    {"url": "https://example.com/doc2.pdf", "type": "document"}
  ]
}
```

### Health Check

```http
GET /health
```

## Format-Specific Features

### PDF Conversion

- **Docling Integration**: Advanced PDF parsing with layout preservation
- **OCR Support**: Text extraction from images and scanned documents
- **Table Extraction**: Structured table data preservation
- **Page-level Chunking**: Maintains document structure
- **Fallback Support**: Unstructured.io as backup parser

### Audio Conversion

- **Whisper Integration**: High-quality speech-to-text
- **Speaker Diarization**: Identify different speakers
- **Timestamp Preservation**: Maintain temporal information
- **Multiple Formats**: MP3, WAV, M4A, FLAC support

### Video Conversion

- **Audio Extraction**: Extract and transcribe audio track
- **Keyframe Analysis**: Visual content description
- **Scene Detection**: Automatic scene segmentation
- **Timeline Synchronization**: Align audio and visual content

## Error Handling

The framework provides robust error handling with fallback mechanisms:

```python
try:
    result = await converter.convert_to_markdown(file_path, options)
    if not result.success:
        print(f"Conversion failed: {result.error_message}")
        print(f"Warnings: {result.warnings}")
except UnsupportedFormatError:
    print("Format not supported")
except ValidationError:
    print("Invalid input file")
```

## Performance Considerations

### Optimization Settings

```python
# Performance configuration
config.performance_settings = {
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'timeout_seconds': 300,  # 5 minutes
    'max_concurrent_conversions': 5,
    'enable_caching': True,
    'cache_ttl_seconds': 3600
}
```

### Processing Times

| Format | Small (<1MB) | Medium (1-10MB) | Large (10-100MB) |
|--------|--------------|-----------------|------------------|
| PDF | <5s | 10-30s | 1-5min |
| Audio | <10s | 30s-2min | 2-10min |
| Video | <30s | 2-5min | 5-20min |

## Testing

### Running Tests

```bash
# Core framework tests
pytest tests/test_universal_converter.py -v

# Integration tests
pytest tests/test_universal_converter_integration.py -v

# Performance tests
pytest tests/test_universal_converter_integration.py::TestPerformance -v
```

### Test Coverage

Current test coverage focuses on:
- ✅ Core converter functionality
- ✅ Quality assessment
- ✅ Configuration management
- ✅ Error handling
- ✅ Service integration
- 🔄 Format-specific converters (in progress)

## Extending the Framework

### Adding New Converters

```python
from morag.converters.base import BaseConverter

class CustomConverter(BaseConverter):
    def __init__(self):
        super().__init__("Custom Converter")
        self.supported_formats = ['custom']
    
    def supports_format(self, format_type: str) -> bool:
        return format_type.lower() in self.supported_formats
    
    async def convert(self, file_path, options):
        # Implementation here
        return ConversionResult(...)

# Register the converter
converter = DocumentConverter()
converter.register_converter('custom', CustomConverter())
```

### Custom Quality Validators

```python
from morag.converters.quality import ConversionQualityValidator

class CustomQualityValidator(ConversionQualityValidator):
    def validate_conversion(self, original_file, result):
        # Custom quality assessment logic
        return QualityScore(...)
```

## Roadmap

### Completed (Task 24) ✅
- Universal conversion framework
- PDF, Audio, Video converters
- Quality assessment system
- Configuration management
- API integration
- Comprehensive testing

### Planned Enhancements (Tasks 25-29)
- **Task 25**: Enhanced PDF conversion with advanced docling features
- **Task 26**: Audio conversion with speaker diarization
- **Task 27**: Video conversion with keyframe extraction
- **Task 28**: Full Office document support
- **Task 29**: Enhanced web content extraction

### Future Improvements
- Real-time conversion streaming
- Advanced AI-powered content analysis
- Multi-language support
- Cloud storage integration
- Conversion result caching

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Format Not Supported**: Check supported formats list
3. **Quality Too Low**: Adjust quality thresholds or enable fallback
4. **Performance Issues**: Reduce file size or increase timeout

### Debug Mode

```python
import structlog
structlog.configure(level="DEBUG")

# Enable detailed logging
result = await converter.convert_to_markdown(file_path, options)
```

## Contributing

When contributing to the universal converter framework:

1. Follow the existing converter patterns
2. Add comprehensive tests
3. Update documentation
4. Ensure quality assessment integration
5. Test with various file formats and sizes

## License

This framework is part of the MoRAG project and follows the same licensing terms.
