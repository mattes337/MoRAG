# Manual Testing Scripts

This directory contains manual testing scripts for specific components and fixes in the MoRAG system. These scripts are designed for manual execution during development, debugging, and validation of specific features.

## Purpose

Manual testing scripts serve different purposes than automated unit/integration tests:
- **Component validation**: Test specific components in isolation
- **Fix verification**: Validate that specific bug fixes work correctly
- **Performance testing**: Measure performance characteristics
- **Connection testing**: Verify external service connections
- **Feature demonstration**: Show how specific features work

## Available Scripts

### Database & Storage Testing

#### `test_qdrant_connection.py`
Tests Qdrant vector database connection and basic operations.

**Purpose**: Verify Qdrant connectivity, authentication, and basic CRUD operations
**Usage**: `python test_qdrant_connection.py`
**Tests**:
- Connection establishment
- Collection listing and info
- Collection creation
- Authentication (if configured)

#### `test_qdrant_auth.py`
Tests Qdrant authentication mechanisms.

**Purpose**: Validate API key authentication and security settings
**Usage**: `python test_qdrant_auth.py`

#### `test_qdrant_network.py`
Tests Qdrant network connectivity and configuration.

**Purpose**: Diagnose network-related connection issues
**Usage**: `python test_qdrant_network.py`

### Audio Processing Testing

#### `test_audio_transcription_fixes.py`
Tests specific fixes for audio transcription quality and format issues.

**Purpose**: Validate transcription quality improvements and format fixes
**Usage**: `python test_audio_transcription_fixes.py <audio_file>`
**Features**:
- Topic timestamp format validation
- Speaker diarization accuracy
- STT quality assessment
- German language support testing

#### `test_audio_format_fix.py`
Tests audio format conversion and compatibility fixes.

**Purpose**: Validate audio format handling and conversion fixes
**Usage**: `python test_audio_format_fix.py <audio_file>`

#### `test_audio_extraction_optimization.py`
Tests audio extraction optimization features.

**Purpose**: Measure and validate audio extraction performance improvements
**Usage**: `python test_audio_extraction_optimization.py <video_file>`
**Features**:
- Processing time measurement
- File size optimization validation
- Codec selection testing

### Video Processing Testing

#### `test_video_format_fix.py`
Tests video format handling and processing fixes.

**Purpose**: Validate video processing improvements and error handling
**Usage**: `python test_video_format_fix.py <video_file>`

### Document Processing Testing

#### `test_pdf_parsing.py`
Tests PDF parsing and text extraction functionality.

**Purpose**: Validate PDF processing quality and encoding fixes
**Usage**: `python test_pdf_parsing.py <pdf_file>`

#### `test_universal_conversion.py`
Tests the universal document conversion framework.

**Purpose**: Comprehensive testing of document conversion capabilities
**Usage**: `python test_universal_conversion.py <document_file> [options]`
**Features**:
- Format detection
- Conversion quality assessment
- Multiple output formats
- Metadata extraction

### System Integration Testing

#### `test_summarization_fix.py`
Tests document summarization improvements.

**Purpose**: Validate summarization quality and fix effectiveness
**Usage**: `python test_summarization_fix.py <document_file>`

#### `test_webhook_demo.py`
Tests webhook functionality and integration.

**Purpose**: Validate webhook delivery and processing
**Usage**: `python test_webhook_demo.py`

## Usage Guidelines

### Running Manual Tests

1. **Set up environment**:
   ```bash
   # Ensure MoRAG is properly installed
   pip install -e ".[dev,audio,video,image,docling]"
   
   # Start required services
   docker-compose -f docker/docker-compose.redis.yml up -d
   docker-compose -f docker/docker-compose.qdrant.yml up -d
   ```

2. **Run specific tests**:
   ```bash
   cd tests/manual
   python test_qdrant_connection.py
   python test_audio_transcription_fixes.py /path/to/audio.mp3
   ```

3. **Review output**:
   - Check console output for test results
   - Look for generated files (if applicable)
   - Verify performance metrics

### When to Use Manual Tests

- **During development**: Test specific components you're working on
- **Bug investigation**: Isolate and reproduce specific issues
- **Performance analysis**: Measure processing times and resource usage
- **Integration validation**: Test connections to external services
- **Feature verification**: Confirm new features work as expected

### Test Categories

#### Connection Tests
Test external service connectivity and configuration:
- Database connections
- API authentication
- Network connectivity

#### Processing Tests
Test core processing functionality:
- Audio/video processing
- Document conversion
- Text extraction and analysis

#### Fix Validation Tests
Verify specific bug fixes:
- Encoding issues
- Format compatibility
- Performance optimizations

#### Integration Tests
Test component integration:
- Webhook delivery
- Pipeline processing
- End-to-end workflows

## Test Data

### Sample Files
Some tests require sample files. You can use:
- Audio: MP3, WAV, M4A files
- Video: MP4, AVI, MOV files
- Documents: PDF, DOCX, TXT files
- Web: HTML files or URLs

### Test Environment
- Ensure test files are available in a known location
- Use small files for faster testing
- Keep test data separate from production data

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure you're running from the correct directory
   - Check that MoRAG is properly installed
   - Verify Python path includes src/

2. **Service Connection Errors**:
   - Check that required services are running
   - Verify configuration in .env file
   - Test network connectivity

3. **File Not Found Errors**:
   - Verify file paths are correct
   - Ensure test files exist and are readable
   - Check file permissions

### Getting Help

- Check the main README.md for setup instructions
- Review logs in the logs/ directory
- Consult the API documentation at http://localhost:8000/docs
- Check TASKS.md for known issues and implementation status

## Contributing

When adding new manual tests:

1. **Follow naming convention**: `test_<component>_<purpose>.py`
2. **Include documentation**: Add clear docstrings and usage instructions
3. **Add to this README**: Document the new test's purpose and usage
4. **Test thoroughly**: Ensure the test works in different environments
5. **Handle errors gracefully**: Include proper error handling and user feedback
