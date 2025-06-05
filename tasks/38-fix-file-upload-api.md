# Task 38: Fix File Upload API Endpoint

## Overview
**Priority**: HIGH  
**Status**: IN_PROGRESS  
**Estimated Effort**: 1-2 days  
**Dependencies**: None  

## Objective
Fix critical issues in the `/process/file` API endpoint that prevent proper file upload handling, particularly on Windows systems and cross-platform environments.

## Problem Statement

### Current Issues
1. **Platform-specific path issues**: Hardcoded `/tmp/` path doesn't exist on Windows
2. **No temporary directory management**: Missing proper temp directory creation and cleanup
3. **Security vulnerabilities**: No file validation, size limits, or type checking
4. **Poor error handling**: Generic exceptions without specific file handling errors
5. **Resource leaks**: Incomplete cleanup on processing failures

### Error Symptoms
- "No such file or directory" errors when processing uploaded files
- API returns 500 errors for valid file uploads
- Files not accessible to processing services
- Inconsistent behavior across operating systems

## Root Cause Analysis

### Current Implementation Issues
Located in `packages/morag/src/morag/server.py` lines 115-156:

```python
# Problematic code:
temp_path = Path(f"/tmp/{file.filename}")  # Hardcoded Unix path
with open(temp_path, "wb") as f:
    content = await file.read()
    f.write(content)
```

### Problems:
1. **Hardcoded path**: `/tmp/` doesn't exist on Windows
2. **No directory creation**: Assumes temp directory exists
3. **Filename conflicts**: No unique naming for concurrent uploads
4. **No validation**: Accepts any file type/size
5. **Security risk**: Direct filename usage allows path traversal

## Implementation Plan

### Phase 1: Cross-Platform Temporary File Handling
- [ ] Replace hardcoded `/tmp/` with `tempfile.mkdtemp()`
- [ ] Implement unique temporary file naming
- [ ] Add proper directory creation and cleanup
- [ ] Handle concurrent uploads safely

### Phase 2: File Validation and Security
- [ ] Add file size limits (configurable)
- [ ] Implement file type validation
- [ ] Sanitize uploaded filenames
- [ ] Add virus scanning hooks (optional)
- [ ] Prevent path traversal attacks

### Phase 3: Enhanced Error Handling
- [ ] Specific error types for file operations
- [ ] Detailed logging for debugging
- [ ] Graceful degradation on failures
- [ ] Proper HTTP status codes

### Phase 4: Testing and Validation
- [ ] Unit tests for file upload handling
- [ ] Integration tests with various file types
- [ ] Cross-platform testing (Windows/Linux)
- [ ] Performance testing with large files
- [ ] Concurrent upload testing

## Technical Specifications

### New File Upload Handler
```python
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
import aiofiles
import magic  # For file type detection

class FileUploadHandler:
    def __init__(self, max_file_size: int = 100 * 1024 * 1024):  # 100MB default
        self.max_file_size = max_file_size
        self.temp_dir = Path(tempfile.mkdtemp(prefix="morag_uploads_"))
        
    async def save_upload(self, file: UploadFile) -> Path:
        # Validate file
        await self._validate_file(file)
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{self._sanitize_filename(file.filename)}"
        temp_path = self.temp_dir / unique_filename
        
        # Save file
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
            
        return temp_path
```

### Configuration Options
```python
class FileUploadConfig:
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = ['.pdf', '.txt', '.docx', '.mp3', '.mp4', '.jpg', '.png']
    temp_dir_prefix: str = "morag_uploads_"
    cleanup_timeout: int = 3600  # 1 hour
```

## Implementation Steps

### Step 1: Create File Upload Handler
Create `packages/morag/src/morag/utils/file_upload.py`:
- Cross-platform temporary file handling
- File validation and security checks
- Proper cleanup mechanisms

### Step 2: Update Server Endpoint
Modify `packages/morag/src/morag/server.py`:
- Replace hardcoded temp file handling
- Integrate new FileUploadHandler
- Add proper error handling

### Step 3: Add Configuration
Update service configuration:
- File upload limits and restrictions
- Temporary directory settings
- Security options

### Step 4: Create Tests
Add comprehensive test suite:
- Unit tests for file upload handler
- Integration tests for API endpoint
- Cross-platform compatibility tests

## Testing Strategy

### Unit Tests
- File validation logic
- Temporary file creation/cleanup
- Error handling scenarios
- Security checks

### Integration Tests
- Full API endpoint testing
- Various file types and sizes
- Concurrent upload handling
- Cross-platform compatibility

### Performance Tests
- Large file uploads
- Multiple concurrent uploads
- Memory usage monitoring
- Cleanup efficiency

## Success Criteria

### Functional Requirements
- [ ] File uploads work on Windows, Linux, and macOS
- [ ] Proper temporary file handling with cleanup
- [ ] File validation and security measures
- [ ] Comprehensive error handling and logging

### Performance Requirements
- [ ] Handle files up to 100MB efficiently
- [ ] Support 10+ concurrent uploads
- [ ] Memory usage remains stable
- [ ] Cleanup completes within 1 hour

### Security Requirements
- [ ] No path traversal vulnerabilities
- [ ] File type validation enforced
- [ ] Size limits respected
- [ ] Temporary files properly secured

## Deliverables

1. **FileUploadHandler class** - Cross-platform file upload handling
2. **Updated server.py** - Fixed API endpoint implementation
3. **Configuration updates** - File upload settings and limits
4. **Comprehensive tests** - Unit, integration, and performance tests
5. **Documentation** - API usage and configuration guide

## Dependencies

### Internal
- MoRAG core configuration system
- Existing API framework
- Processing orchestrator

### External
- `aiofiles` for async file operations
- `python-magic` for file type detection
- `tempfile` for cross-platform temp handling

## Risks and Mitigation

### Risk: Performance impact with large files
**Mitigation**: Streaming file handling, configurable size limits

### Risk: Disk space exhaustion
**Mitigation**: Automatic cleanup, monitoring, size limits

### Risk: Security vulnerabilities
**Mitigation**: Comprehensive validation, sandboxed temp directories

## Timeline

- **Day 1**: Implement FileUploadHandler and update server endpoint
- **Day 2**: Add tests, documentation, and validation

## Notes

This task is critical for system functionality and must be completed before comprehensive system testing (Task 39) can proceed.
