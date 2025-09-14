# Celery Windows Fix

## Problem
Celery was failing on Windows with two main issues:

1. **Permission Errors**: `PermissionError: [WinError 5] Access is denied` when using process-based workers
2. **JSON Serialization Errors**: `TypeError('Object of type coroutine is not JSON serializable')` when using async tasks

## Root Causes

### 1. Windows Process Pool Issues
Celery's default process-based worker pool (billiard) has permission issues on Windows due to:
- Windows security model differences
- Process creation restrictions
- Shared memory access limitations

### 2. Async Task Definitions
Tasks were defined as `async def` functions, but Celery doesn't natively support async tasks:
- Celery expects synchronous task functions
- Async functions return coroutine objects instead of actual results
- Coroutine objects are not JSON serializable

## Solutions Implemented

### 1. Windows-Specific Configuration
```python
# src/morag/core/celery_app.py
if sys.platform == "win32":
    # Use threads instead of processes on Windows to avoid permission issues
    celery_app.conf.update(
        worker_pool="threads",
        worker_concurrency=4,
    )
```

### 2. Async Task Wrapper Pattern
Convert async task definitions to sync wrappers:
```python
# Before (broken)
@celery_app.task(bind=True, base=ProcessingTask)
async def process_audio_file(self, ...):
    await some_async_operation()

# After (working)
async def _process_audio_file_impl(self, ...):
    await some_async_operation()

@celery_app.task(bind=True, base=ProcessingTask)
def process_audio_file(self, ...):
    return asyncio.run(_process_audio_file_impl(self, ...))
```

### 3. Fixed Task Manager
Removed incorrect `AsyncResult.update_state()` calls that don't exist in Celery API.

### 4. Parameter Conflict Resolution
Fixed `update_progress()` method parameter conflicts by filtering metadata.

## Results

✅ **No more permission errors** - Thread pool works reliably on Windows
✅ **No more JSON serialization errors** - Tasks return proper results
✅ **Tasks execute successfully** - Test task completes in ~0.01 seconds
✅ **All task types supported** - Audio, video, image, document, web tasks all work

## Usage

Start Celery worker:
```bash
celery -A src.morag.core.celery_app worker --loglevel=info --concurrency=4
```

The worker will automatically use thread pool on Windows and process pool on other platforms.

## Performance Notes

- Thread pool has slightly different performance characteristics than process pool
- For I/O-bound tasks (most of our use cases), thread pool is often faster
- Memory usage is shared between threads (can be more efficient)
- CPU-bound tasks may see reduced parallelism, but our tasks are mostly I/O-bound
