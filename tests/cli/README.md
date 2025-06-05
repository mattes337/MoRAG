# MoRAG CLI Testing Scripts

This directory contains command-line testing scripts for validating individual MoRAG components and the complete system.

## Quick Start

### System Validation
```bash
# Quick system check (recommended first step)
python tests/cli/test-simple.py

# Comprehensive system test with detailed report
python tests/cli/test-all.py
```

### Individual Component Testing

#### Audio Processing
```bash
python tests/cli/test-audio.py uploads/audio.mp3
python tests/cli/test-audio.py uploads/recording.wav
python tests/cli/test-audio.py uploads/video.mp4  # Extract audio from video
```

#### Document Processing
```bash
python tests/cli/test-document.py uploads/document.pdf
python tests/cli/test-document.py uploads/presentation.pptx
python tests/cli/test-document.py uploads/spreadsheet.xlsx
python tests/cli/test-document.py uploads/document.docx
```

#### Video Processing
```bash
python tests/cli/test-video.py uploads/video.mp4
python tests/cli/test-video.py uploads/recording.avi
python tests/cli/test-video.py uploads/presentation.mov
```

#### Image Processing
```bash
python tests/cli/test-image.py uploads/image.jpg
python tests/cli/test-image.py uploads/screenshot.png
python tests/cli/test-image.py uploads/diagram.gif
```

#### Web Content Processing
```bash
python tests/cli/test-web.py https://example.com
python tests/cli/test-web.py https://en.wikipedia.org/wiki/Python
python tests/cli/test-web.py https://github.com/your-repo
```

#### YouTube Processing
```bash
python tests/cli/test-youtube.py https://www.youtube.com/watch?v=VIDEO_ID
python tests/cli/test-youtube.py https://youtu.be/VIDEO_ID
```

## Available Scripts

### `test-simple.py`
**Purpose**: Quick system validation and health check  
**Usage**: `python tests/cli/test-simple.py`  
**Features**:
- Tests package imports
- Validates basic functionality
- Checks for sample files
- Verifies Docker configuration
- Confirms documentation presence

### `test-all.py`
**Purpose**: Comprehensive system testing with detailed reporting  
**Usage**: `python tests/cli/test-all.py`  
**Features**:
- Component initialization tests
- Configuration validation
- Processing configuration tests
- Sample file detection
- Detailed JSON report generation

### Individual Component Scripts

#### `test-audio.py`
- Tests audio processing functionality
- Supports MP3, WAV, M4A, and video files (audio extraction)
- Validates transcription, speaker diarization, and topic segmentation
- Generates markdown output with timestamps and speaker labels

#### `test-document.py`
- Tests document processing functionality
- Supports PDF, DOCX, PPTX, XLSX, TXT, MD formats
- Validates text extraction and page-level chunking
- Generates structured markdown output

#### `test-video.py`
- Tests video processing functionality
- Supports MP4, AVI, MOV, and other video formats
- Validates audio extraction, transcription, and visual analysis
- May take several minutes for large files

#### `test-image.py`
- Tests image processing functionality
- Supports JPG, PNG, GIF, BMP formats
- Validates OCR text extraction and AI-powered descriptions
- Generates structured markdown with visual elements

#### `test-web.py`
- Tests web content processing functionality
- Validates content extraction and cleaning
- Supports various website structures
- Requires valid URLs with http:// or https://

#### `test-youtube.py`
- Tests YouTube video processing functionality
- Validates video download, transcription, and metadata extraction
- Supports standard YouTube URL formats
- May take several minutes depending on video length

## Output Files

All test scripts generate output files in the same directory as the input file:

- `{filename}_test_result.json` - Detailed processing results
- `{filename}_converted.md` - Markdown-formatted content
- For web/YouTube: `web_{safe_url}_*` or `youtube_{video_id}_*`

## Prerequisites

### Required Packages
```bash
pip install -e packages/morag-core
pip install -e packages/morag-audio
pip install -e packages/morag-document
pip install -e packages/morag-video
pip install -e packages/morag-image
pip install -e packages/morag-web
pip install -e packages/morag-youtube
pip install -e packages/morag-services
```

### Optional Dependencies
Some features require additional packages:
- **Speaker Diarization**: `pip install pyannote.audio`
- **Advanced Topic Segmentation**: `pip install sentence-transformers`
- **Dynamic Web Content**: `pip install playwright`
- **Advanced Web Extraction**: `pip install trafilatura readability newspaper3k`

### Environment Configuration
Create a `.env` file with required API keys:
```bash
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
```

## Troubleshooting

### Import Errors
If you see import errors, ensure all packages are installed:
```bash
pip install -e .
```

### Missing Dependencies
The scripts will show warnings for missing optional dependencies. Install them as needed:
```bash
pip install pyannote.audio sentence-transformers playwright trafilatura
```

### File Not Found
Ensure your test files exist and paths are correct:
```bash
ls -la uploads/  # Check available files
```

### Processing Failures
Check the generated JSON files for detailed error information and logs.

## Integration with Main Testing

These CLI scripts complement the main test suite:
- **Unit Tests**: `pytest tests/unit/`
- **Integration Tests**: `pytest tests/integration/`
- **Manual Tests**: `python tests/manual/test_*.py`
- **CLI Tests**: `python tests/cli/test-*.py` (this directory)

## Development

When adding new CLI test scripts:
1. Follow the existing naming pattern: `test-{component}.py`
2. Include comprehensive error handling and user feedback
3. Generate both JSON and markdown output files
4. Add usage examples to this README
5. Test with various file types and edge cases

For more information, see the main project documentation in `/docs/`.
