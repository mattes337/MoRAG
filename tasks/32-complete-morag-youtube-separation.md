# Task 32: Complete MoRAG YouTube Package Separation

## Overview

Complete the separation of YouTube processing functionality from the main MoRAG codebase into the standalone `morag-youtube` package. The package structure already exists, but the implementation needs to be moved from `src/morag` to `packages/morag-youtube`.

## Current State

### Completed
- ✅ Package structure created in `packages/morag-youtube/`
- ✅ Basic package configuration (`pyproject.toml`, `README.md`)
- ✅ YouTube video downloading functionality implemented in main codebase
- ✅ Metadata extraction implemented
- ✅ Caption extraction implemented

### Remaining Work
- [ ] Move YouTube processor implementation from `src/morag/processors/youtube.py`
- [ ] Move YouTube tasks implementation from `src/morag/tasks/youtube_tasks.py`
- [ ] Update package dependencies and imports
- [ ] Create comprehensive tests for the separated package
- [ ] Update main codebase to use the new package
- [ ] Integrate with morag-video package for video processing

## Implementation Steps

### Step 1: Move YouTube Processor Implementation

**Files to move:**
- `src/morag/processors/youtube.py` → `packages/morag-youtube/src/morag_youtube/processor.py`

**Actions:**
1. Copy the YouTubeProcessor class and related functionality
2. Update imports to use morag-core interfaces
3. Ensure yt-dlp integration works correctly
4. Add proper error handling and logging
5. Integrate with morag-video for video processing

### Step 2: Move YouTube Tasks Implementation

**Files to move:**
- `src/morag/tasks/youtube_tasks.py` → `packages/morag-youtube/src/morag_youtube/tasks.py`

**Actions:**
1. Copy Celery task definitions
2. Update imports and dependencies
3. Ensure task registration works with the new package
4. Add proper error handling and retry logic
5. Integrate video processing pipeline

### Step 3: Create YouTube Service Layer

**Files to create:**
- `packages/morag-youtube/src/morag_youtube/service.py`

**Actions:**
1. Create high-level service interface
2. Implement video download and processing pipeline
3. Add metadata extraction and enrichment
4. Integrate with morag-video for transcription
5. Add progress tracking and status updates

### Step 4: Update Package Configuration

**Files to update:**
- `packages/morag-youtube/pyproject.toml`
- `packages/morag-youtube/src/morag_youtube/__init__.py`

**Actions:**
1. Add missing dependencies (yt-dlp, morag-video, etc.)
2. Update package exports and imports
3. Ensure version compatibility with other MoRAG packages
4. Add optional dependencies for enhanced features

### Step 5: Create Comprehensive Tests

**Files to create:**
- `packages/morag-youtube/tests/test_processor.py`
- `packages/morag-youtube/tests/test_service.py`
- `packages/morag-youtube/tests/test_tasks.py`
- `packages/morag-youtube/tests/test_integration.py`

**Test Coverage:**
1. YouTube URL validation and parsing
2. Video download functionality
3. Metadata extraction
4. Caption extraction
5. Integration with video processing
6. Error handling for various edge cases

### Step 6: Update Main Codebase

**Files to update:**
- `src/morag/processors/__init__.py`
- `src/morag/tasks/__init__.py`
- `pyproject.toml` (add morag-youtube dependency)

**Actions:**
1. Remove old implementations
2. Add imports from morag-youtube package
3. Update dependency configuration
4. Ensure backward compatibility

## Dependencies

### Required Packages
- `morag-core>=0.1.0` - Core interfaces and utilities
- `morag-video>=0.1.0` - Video processing integration
- `yt-dlp>=2023.10.13` - YouTube video downloading
- `structlog>=24.4.0` - Structured logging
- `aiofiles>=23.2.1` - Async file operations
- `pydantic>=2.0.0` - Data validation

### Optional Packages
- `morag-audio>=0.1.0` - Enhanced audio processing
- `ffmpeg-python>=0.2.0` - Video processing utilities

## Testing Requirements

### Unit Tests
- [ ] Test YouTube URL validation and parsing
- [ ] Test video download with various quality settings
- [ ] Test metadata extraction accuracy
- [ ] Test caption extraction and processing
- [ ] Test error handling for invalid URLs and network issues
- [ ] Test rate limiting and respectful downloading

### Integration Tests
- [ ] Test integration with morag-core interfaces
- [ ] Test integration with morag-video package
- [ ] Test Celery task execution
- [ ] Test end-to-end YouTube processing pipeline
- [ ] Test with real YouTube videos (using test fixtures)

### Performance Tests
- [ ] Test download speed for various video sizes
- [ ] Test memory usage with large videos
- [ ] Test concurrent download handling

## Success Criteria

1. **Functional Separation**: YouTube processing works entirely through the morag-youtube package
2. **No Breaking Changes**: Existing functionality continues to work without modification
3. **Test Coverage**: >95% unit test coverage, >90% integration test coverage
4. **Documentation**: Complete API documentation and usage examples
5. **Performance**: No significant performance degradation compared to current implementation
6. **Integration**: Seamless integration with morag-video for enhanced processing

## Validation Steps

1. **Package Installation**: Verify morag-youtube can be installed independently
2. **Functionality Test**: Download and process various YouTube videos successfully
3. **Integration Test**: Verify integration with main MoRAG system and morag-video
4. **Performance Test**: Ensure download and processing speed meets requirements
5. **Error Handling**: Verify robust error handling for edge cases

## Special Considerations

### YouTube API Compliance
- Respect YouTube's Terms of Service
- Implement proper rate limiting
- Handle API changes gracefully
- Add user agent and request headers appropriately

### Video Processing Integration
- Seamless handoff to morag-video package
- Preserve metadata through processing pipeline
- Handle various video formats and qualities
- Optimize for processing efficiency

### Error Handling
- Network connectivity issues
- Invalid or private videos
- Age-restricted content
- Geo-blocked content
- API rate limits

## Notes

- Maintain backward compatibility during transition
- Use feature flags if needed for gradual migration
- Ensure proper logging and monitoring integration
- Consider Docker container optimization for YouTube processing
- Plan for future enhancements (e.g., playlist support, live stream handling)
- Monitor yt-dlp updates and compatibility
