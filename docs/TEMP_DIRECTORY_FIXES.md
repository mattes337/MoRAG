# Temp Directory and Volume Mapping Fixes

## Overview

This document describes the fixes implemented to resolve volume mapping issues and ensure consistent temporary directory usage across the MoRAG system.

## Problems Identified

### 1. Redundant Volume Mappings
- Both `/app/temp` and `/app/uploads` volumes were mapped in Docker compose files
- Only `/app/temp` was actually used by the application
- This created confusion and unnecessary complexity

### 2. Inconsistent Directory Usage
- API server would fall back to system `/tmp` if `/app/temp` was not accessible
- Worker containers wouldn't know about this fallback
- This could cause "File not found" errors when workers tried to access uploaded files

### 3. No Early Failure Detection
- Directory permission issues were only discovered during runtime
- No validation during application startup
- Led to confusing error messages during file processing

### 4. Unclear Error Messages
- Users didn't know if temp directory was properly configured
- No warnings when using problematic system temp directories

## Solutions Implemented

### 1. Removed Redundant Volume Mappings

**Files Modified:**
- `docker-compose.yml`
- `docker-compose.prod.yml`
- `docker-compose.microservices.yml`
- `Dockerfile`
- `Dockerfile.worker`
- `scripts/deploy.sh`

**Changes:**
- Removed `/app/uploads` volume mappings from all Docker compose files
- Removed uploads directory creation from Dockerfiles and deployment scripts
- Simplified volume configuration to only use `/app/temp`

### 2. Enhanced Directory Validation

**File:** `packages/morag/src/morag/utils/file_upload.py`

**Changes:**
- Enhanced `_try_create_dir()` to test write permissions by creating/deleting a test file
- Improved error handling and logging in temp directory creation
- Added clear warnings when falling back to system temp directories
- Made system temp usage more explicit about container compatibility issues

### 3. Added Startup Validation

**Files Modified:**
- `packages/morag/src/morag/utils/file_upload.py` - Added `validate_temp_directory_access()`
- `packages/morag/src/morag/server.py` - Added startup validation call

**Features:**
- `validate_temp_directory_access()` function validates temp directory during startup
- Tests directory existence and write permissions
- Fails fast if temp directory is not accessible
- Provides clear error messages about configuration issues
- Warns about system temp usage in container environments

### 4. Improved Error Messages and Logging

**Changes:**
- Clear distinction between shared volume, local development, and system temp usage
- Explicit warnings about container compatibility issues
- Better error messages when temp directory validation fails
- Startup logging shows which temp directory is being used

## Technical Details

### Directory Priority Order

1. **`/app/temp`** (Docker shared volume) - Preferred for container deployments
2. **`./temp`** (Local directory) - Used for local development
3. **System temp** (e.g., `/tmp/`) - Last resort, with warnings

### Startup Validation Process

1. Server startup calls `validate_temp_directory_access()`
2. Function gets global upload handler (creates temp directory if needed)
3. Tests directory existence and write permissions
4. Warns if using system temp (problematic in containers)
5. Fails with clear error message if validation fails

### Write Permission Testing

The system now tests write permissions by:
1. Creating a test file in the temp directory
2. Writing content to the test file
3. Deleting the test file
4. Only considering the directory valid if all steps succeed

## Benefits

### 1. Simplified Configuration
- Single temp volume instead of redundant mappings
- Clearer Docker compose files
- Less confusion about which directories are used

### 2. Early Problem Detection
- Startup failures instead of runtime errors
- Clear error messages during server startup
- No more mysterious "File not found" errors during processing

### 3. Better Developer Experience
- Immediate feedback if temp directory is misconfigured
- Clear warnings about potential container issues
- Consistent behavior across all deployment scenarios

### 4. Improved Reliability
- All containers use the same temp directory location
- Guaranteed file sharing between API server and workers
- Reduced race conditions and file access issues

## Testing

### Unit Tests
- `tests/test_temp_directory_fixes.py` - Comprehensive unit tests for validation logic

### Integration Tests
- `tests/cli/test-temp-directory-validation.py` - End-to-end validation of fixes

### Test Coverage
- Temp directory validation success/failure scenarios
- Write permission testing
- Startup validation integration
- File upload and worker access validation

## Usage

### For Developers

The system now provides clear feedback during startup:

```bash
# Successful startup
2025-01-06 14:24:11 [info] Using shared Docker temp directory temp_dir=/app/temp/morag_uploads_cd30e3d7
2025-01-06 14:24:11 [info] Temp directory validation successful

# Startup failure
2025-01-06 14:24:11 [error] STARTUP FAILURE: Temp directory validation failed
RuntimeError: Cannot start server: Temp directory is not writable: /app/temp
```

### For Deployment

Ensure the temp directory is properly mounted:

```yaml
volumes:
  - ./temp:/app/temp  # Required for file sharing between containers
```

### For Troubleshooting

If you see warnings about system temp usage:

```
[warning] STARTUP WARNING: Using system temp directory - this may cause issues in container environments
```

This indicates the `/app/temp` directory is not accessible. Check:
1. Volume mapping in docker-compose.yml
2. Directory permissions on the host
3. Docker volume mount configuration

## Migration Notes

### Existing Deployments

1. Remove any references to `/app/uploads` volume mappings
2. Ensure `/app/temp` volume is properly mapped
3. Remove any uploads directories from host filesystem (optional cleanup)

### Development Environments

No changes required - the system will automatically use `./temp` for local development.

## Future Considerations

1. **Monitoring**: Consider adding metrics for temp directory usage and cleanup
2. **Configuration**: Could make temp directory location configurable via environment variables
3. **Cleanup**: Enhanced cleanup strategies based on disk usage patterns
4. **Security**: Additional validation for temp directory security in multi-tenant environments
