# MoRAG Test Suite

This directory contains the comprehensive test suite for the MoRAG (Multi-modal Retrieval Augmented Generation) system. The tests are organized into different categories to ensure thorough coverage of all system components.

## Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── manual/         # Manual testing scripts for specific scenarios
├── fixtures/       # Test data and fixtures
├── conftest.py     # Pytest configuration and shared fixtures
└── test_*.py       # Main test files organized by task/feature
```

## Test Categories

### Unit Tests (`unit/`)
Test individual components in isolation:
- **Audio Processing**: Whisper service, audio processors, converters
- **Video Processing**: FFmpeg service, video processors, keyframe extraction
- **Document Processing**: PDF parsing, text extraction, OCR
- **Image Processing**: Vision services, image captioning
- **Web Processing**: Content extraction, HTML parsing
- **Storage Services**: Qdrant integration, vector operations
- **Core Services**: Configuration, logging, error handling

### Integration Tests (`integration/`)
Test component interactions and complete workflows:
- **Audio Pipeline**: End-to-end audio processing with diarization
- **Video Pipeline**: Complete video processing with audio extraction
- **Document Pipeline**: Document ingestion and conversion
- **Web Pipeline**: Web scraping and content processing
- **API Integration**: REST API endpoints and task processing
- **Celery Integration**: Task queue and worker functionality

### Manual Tests (`manual/`)
Scripts for manual testing and validation:
- **Connection Testing**: Database and service connectivity
- **Performance Testing**: Processing time and resource usage
- **Fix Validation**: Specific bug fix verification
- **Feature Demonstration**: New feature validation

### Main Test Files
Task-based tests organized by implementation order:
- `test_01_*`: Project setup and configuration
- `test_02_*`: API framework and health checks
- `test_03_*`: Database setup and performance
- `test_15_*`: Vector storage integration
- `test_16_*`: Metadata management
- `test_17_*`: Ingestion API functionality

## Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install -e ".[dev,audio,video,image,docling]"

# Start required services
docker-compose -f docker/docker-compose.redis.yml up -d
docker-compose -f docker/docker-compose.qdrant.yml up -d

# Initialize database
python scripts/init_db.py
```

### Run All Tests
```bash
# Run the complete test suite
pytest tests/

# Run with coverage report
pytest tests/ --cov=src/morag --cov-report=html

# Run with verbose output
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/test_01_project_setup.py

# Specific test function
pytest tests/unit/test_audio_processor.py::test_audio_config
```

### Run Manual Tests
```bash
# Manual tests are run individually
cd tests/manual
python test_qdrant_connection.py
python test_audio_transcription_fixes.py /path/to/audio.mp3
```

### CLI Tests
```bash
# Quick system validation
python tests/cli/test-simple.py

# Individual component tests
python tests/cli/test-audio.py uploads/audio.mp3
python tests/cli/test-document.py uploads/document.pdf
python tests/cli/test-video.py uploads/video.mp4
python tests/cli/test-image.py uploads/image.jpg
python tests/cli/test-web.py https://example.com
python tests/cli/test-youtube.py https://youtube.com/watch?v=VIDEO_ID

# Comprehensive system test
python tests/cli/test-all.py
```

## Test Configuration

### Environment Variables
Tests use the same configuration as the main application:
```env
# Required for integration tests
GEMINI_API_KEY=your_api_key_here
REDIS_URL=redis://localhost:6379/0
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Optional test-specific settings
TEST_MODE=true
LOG_LEVEL=DEBUG
```

### Test Data
- **Fixtures**: Located in `tests/fixtures/`
- **Sample Files**: Small test files for processing
- **Mock Data**: Generated test data for unit tests
- **External Data**: Real files for integration testing

## Test Guidelines

### Writing Tests
1. **Follow naming conventions**: `test_<component>_<functionality>.py`
2. **Use descriptive test names**: `test_audio_processor_handles_invalid_format`
3. **Include docstrings**: Explain what the test validates
4. **Use appropriate fixtures**: Leverage shared test data and setup
5. **Test edge cases**: Include error conditions and boundary cases

### Test Organization
- **Unit tests**: One test file per component
- **Integration tests**: One test file per workflow/pipeline
- **Test isolation**: Each test should be independent
- **Resource cleanup**: Ensure tests clean up after themselves

### Performance Considerations
- **Use small test files**: Keep test execution fast
- **Mock external services**: When possible, avoid real API calls
- **Parallel execution**: Tests should support parallel running
- **Resource limits**: Set appropriate timeouts and limits

## Coverage Requirements

The project maintains high test coverage standards:
- **Unit Tests**: >95% coverage required
- **Integration Tests**: >90% coverage required
- **Critical Paths**: 100% coverage for core functionality
- **Error Handling**: All exception paths must be tested

### Checking Coverage
```bash
# Generate coverage report
pytest tests/ --cov=src/morag --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

## Continuous Integration

Tests are automatically run on:
- **Pull Requests**: All tests must pass
- **Main Branch**: Full test suite with coverage reporting
- **Releases**: Complete test suite plus manual validation

### Test Automation
- **GitHub Actions**: Automated test execution
- **Coverage Reporting**: Automatic coverage analysis
- **Performance Monitoring**: Track test execution times
- **Dependency Testing**: Test with different dependency versions

## Troubleshooting

### Common Issues

1. **Service Connection Errors**:
   - Ensure Redis and Qdrant are running
   - Check network connectivity and ports
   - Verify configuration in .env file

2. **Import Errors**:
   - Ensure MoRAG is installed in development mode
   - Check Python path includes src/
   - Verify all dependencies are installed

3. **Test Timeouts**:
   - Large files may cause timeouts
   - Check system resources
   - Consider using smaller test files

4. **Flaky Tests**:
   - Network-dependent tests may be unstable
   - Use mocking for external services
   - Add appropriate retry mechanisms

### Getting Help
- **Documentation**: Check component-specific docs
- **Logs**: Review test output and application logs
- **Issues**: Check TASKS.md for known issues
- **Community**: Consult project README and documentation

## Contributing

When adding new tests:
1. **Follow the existing structure**: Place tests in appropriate directories
2. **Update this README**: Document new test categories or requirements
3. **Ensure coverage**: New code should include comprehensive tests
4. **Test your tests**: Verify tests work in clean environments
5. **Performance**: Keep test execution time reasonable

For more information about specific test categories, see:
- `unit/README.md` - Unit testing guidelines
- `integration/README.md` - Integration testing guidelines  
- `manual/README.md` - Manual testing procedures
