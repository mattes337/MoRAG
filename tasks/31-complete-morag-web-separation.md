# Task 31: Complete MoRAG Web Package Separation

## Overview

Complete the separation of web processing functionality from the main MoRAG codebase into the standalone `morag-web` package. The package structure already exists, but the implementation needs to be moved from `src/morag` to `packages/morag-web`.

## Current State

### Completed
- ✅ Package structure created in `packages/morag-web/`
- ✅ Basic package configuration (`pyproject.toml`, `README.md`)
- ✅ Web scraping functionality implemented in main codebase
- ✅ Content extraction and cleaning implemented
- ✅ HTML to Markdown conversion implemented

### Remaining Work
- [ ] Move web processor implementation from `src/morag/processors/web.py`
- [ ] Move web converter implementation from `src/morag/converters/web.py`
- [ ] Move web tasks implementation from `src/morag/tasks/web_tasks.py`
- [ ] Update package dependencies and imports
- [ ] Create comprehensive tests for the separated package
- [ ] Update main codebase to use the new package

## Implementation Steps

### Step 1: Move Web Processor Implementation

**Files to move:**
- `src/morag/processors/web.py` → `packages/morag-web/src/morag_web/processor.py`

**Actions:**
1. Copy the WebProcessor class and related functionality
2. Update imports to use morag-core interfaces
3. Ensure compatibility with the package structure
4. Add proper error handling and logging

### Step 2: Move Web Converter Implementation

**Files to move:**
- `src/morag/converters/web.py` → `packages/morag-web/src/morag_web/converter.py`

**Actions:**
1. Copy the WebConverter class and related functionality
2. Update imports to use morag-core and morag-web modules
3. Ensure Playwright integration works correctly
4. Add comprehensive error handling

### Step 3: Move Web Tasks Implementation

**Files to move:**
- `src/morag/tasks/web_tasks.py` → `packages/morag-web/src/morag_web/tasks.py`

**Actions:**
1. Copy Celery task definitions
2. Update imports and dependencies
3. Ensure task registration works with the new package
4. Add proper error handling and retry logic

### Step 4: Update Package Configuration

**Files to update:**
- `packages/morag-web/pyproject.toml`
- `packages/morag-web/src/morag_web/__init__.py`

**Actions:**
1. Add missing dependencies (playwright, beautifulsoup4, etc.)
2. Update package exports and imports
3. Ensure version compatibility with other MoRAG packages
4. Add optional dependencies for enhanced features

### Step 5: Create Comprehensive Tests

**Files to create:**
- `packages/morag-web/tests/test_processor.py`
- `packages/morag-web/tests/test_converter.py`
- `packages/morag-web/tests/test_tasks.py`
- `packages/morag-web/tests/test_integration.py`

**Test Coverage:**
1. Web scraping functionality
2. Content extraction and cleaning
3. HTML to Markdown conversion
4. Error handling and edge cases
5. Integration with other MoRAG components

### Step 6: Update Main Codebase

**Files to update:**
- `src/morag/processors/__init__.py`
- `src/morag/converters/__init__.py`
- `src/morag/tasks/__init__.py`
- `pyproject.toml` (add morag-web dependency)

**Actions:**
1. Remove old implementations
2. Add imports from morag-web package
3. Update dependency configuration
4. Ensure backward compatibility

## Dependencies

### Required Packages
- `morag-core>=0.1.0` - Core interfaces and utilities
- `httpx>=0.24.0` - HTTP client for web requests
- `beautifulsoup4>=4.12.0` - HTML parsing
- `markdownify>=0.11.0` - HTML to Markdown conversion
- `lxml>=4.9.0` - XML/HTML processing
- `structlog>=24.4.0` - Structured logging
- `aiofiles>=23.2.1` - Async file operations

### Optional Packages
- `playwright>=1.40.0` - Dynamic content extraction
- `selenium>=4.15.0` - Alternative browser automation

## Testing Requirements

### Unit Tests
- [ ] Test web scraping with various URL types
- [ ] Test content extraction from different HTML structures
- [ ] Test HTML to Markdown conversion accuracy
- [ ] Test error handling for invalid URLs and network issues
- [ ] Test rate limiting and respectful scraping

### Integration Tests
- [ ] Test integration with morag-core interfaces
- [ ] Test Celery task execution
- [ ] Test end-to-end web processing pipeline
- [ ] Test with real websites (using test fixtures)

### Performance Tests
- [ ] Test processing speed for large web pages
- [ ] Test memory usage with multiple concurrent requests
- [ ] Test rate limiting effectiveness

## Success Criteria

1. **Functional Separation**: Web processing works entirely through the morag-web package
2. **No Breaking Changes**: Existing functionality continues to work without modification
3. **Test Coverage**: >95% unit test coverage, >90% integration test coverage
4. **Documentation**: Complete API documentation and usage examples
5. **Performance**: No significant performance degradation compared to current implementation

## Validation Steps

1. **Package Installation**: Verify morag-web can be installed independently
2. **Functionality Test**: Process various web content types successfully
3. **Integration Test**: Verify integration with main MoRAG system
4. **Performance Test**: Ensure processing speed meets requirements
5. **Error Handling**: Verify robust error handling for edge cases

## Notes

- Maintain backward compatibility during transition
- Use feature flags if needed for gradual migration
- Ensure proper logging and monitoring integration
- Consider Docker container optimization for web processing
- Plan for future enhancements (e.g., JavaScript rendering, form handling)
