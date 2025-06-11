# MoRAG Progress Reporting System

This document describes the progress reporting system that allows workers (both local and remote) to report their progress back to job entities in the MoRAG system.

## Overview

The progress reporting system consists of several components:

1. **Progress Event Parser** - Parses progress information from log messages
2. **Progress Handler** - Processes progress events and updates job entities
3. **Remote Job API** - Provides endpoints for remote workers to report progress
4. **Integration with Workers** - Local and remote workers report progress automatically

## Architecture

```
Remote Worker → Progress Logs → Progress Parser → Progress Handler → Job Entity
     ↓                                                    ↑
API Progress Updates ────────────────────────────────────┘

Celery Task → Progress Callback → Progress Handler → Job Entity
```

## Components

### Progress Event Parser

The `ProgressEventParser` class can parse progress information from various log formats:

```python
from morag_core.jobs import ProgressEventParser

parser = ProgressEventParser()

# Parse JSON log
log_line = '{"event": "Processing progress: Audio processing (75%)", "timestamp": "2025-06-11T11:26:50.780725Z"}'
event = parser.parse_json_log(log_line)

# Parse plain text log
event = parser.parse_plain_text_log("Processing... 75%")

# Get latest progress from multiple logs
events = parser.parse_log_stream(log_lines)
latest = parser.get_latest_progress(log_lines)
```

**Supported Progress Formats:**
- `Processing progress: Audio processing: Initializing (52%)`
- `Processing... 75%`
- `Progress: 45% - Converting video`
- `Stage: transcription (67%)`
- `Audio processing: 80% complete`
- `[75%] Processing audio content`

### Progress Handler

The `ProgressHandler` class manages progress events and updates job entities:

```python
from morag_core.jobs import ProgressHandler

handler = ProgressHandler()

# Register job mapping for workers
handler.register_job_mapping("worker-1", "job-123")

# Process log lines
handler.process_log_line(log_line, job_id="job-123")

# Handle remote worker progress
handler.process_remote_worker_progress("worker-1", 75, "Processing audio")

# Handle Celery task progress
handler.process_celery_task_progress("task-123", 50, "Processing document")

# Handle job completion
handler.handle_job_completion("job-123", True, "Processing completed")
```

## Remote Worker Integration

### API Endpoint

Remote workers can report progress using the `/api/v1/remote-jobs/{job_id}/progress` endpoint:

```python
import requests

# Report progress
response = requests.put(
    f"{api_base_url}/api/v1/remote-jobs/{job_id}/progress",
    json={
        "percentage": 75,
        "message": "Transcribing audio content",
        "timestamp": "2025-06-11T11:26:50.780725Z"
    },
    headers={"Authorization": f"Bearer {api_key}"}
)
```

### Remote Converter Integration

The remote converter automatically reports progress:

```python
# In remote_converter.py
def progress_callback(progress: float, message: str = None):
    percentage = int(progress * 100)
    logger.info(f"Processing progress: {message} ({percentage}%)")
    
    # Report progress to API
    asyncio.create_task(self._report_progress(job_id, percentage, message))
```

## Local Worker Integration

### Celery Task Integration

Local Celery tasks can report progress using callbacks:

```python
# In ingest_tasks.py
def progress_callback(progress: float, message: str = None):
    if job_tracker and job_id:
        percentage = int(progress * 100)
        mapped_percentage = 10 + int(progress * 60)  # Map to processing stage
        job_tracker.update_progress(
            job_id, 
            mapped_percentage, 
            None, 
            message or f"Processing: {mapped_percentage}%", 
            user_id
        )

# Add to processing options
options['progress_callback'] = progress_callback
result = await api.process_file(file_path, content_type, options)
```

## Job Entity Updates

Progress updates automatically update the job entity fields:

- **percentage** - Progress percentage (0-100)
- **status** - Job status based on percentage:
  - `PENDING` (0%)
  - `PROCESSING` (1-99%)
  - `FINISHED` (100%)
- **summary** - Progress message/description
- **updated_at** - Timestamp of last update

## Usage Examples

### Example 1: Remote Worker Progress

```python
# Remote worker reports progress
worker_id = "remote-worker-gpu-01"
job_id = "job-audio-123"

# Register mapping
handler.register_job_mapping(worker_id, job_id)

# Simulate progress updates
progress_updates = [
    (10, "Initializing audio processing"),
    (52, "Extracting audio metadata"),
    (75, "Transcribing audio content"),
    (100, "Audio processing completed")
]

for percentage, message in progress_updates:
    handler.process_remote_worker_progress(worker_id, percentage, message)
```

### Example 2: Log Parsing

```python
# Parse progress from log messages
log_messages = [
    '{"event": "Processing progress: Video processing (30%)", "timestamp": "2025-06-11T11:26:51.780725Z"}',
    '{"event": "Processing progress: Video processing (60%)", "timestamp": "2025-06-11T11:26:52.780725Z"}',
    '{"event": "Processing progress: Video processing (100%)", "timestamp": "2025-06-11T11:26:54.780725Z"}',
]

for log_message in log_messages:
    handler.process_log_line(log_message, job_id="job-video-456")
```

### Example 3: Mixed Sources

```python
# Handle progress from multiple sources
handler.process_remote_worker_progress("remote-worker-1", 25, "Remote processing")
handler.process_celery_task_progress("celery-task-1", 50, "Celery processing")

log_line = '{"event": "Processing progress: Direct processing (75%)"}'
handler.process_log_line(log_line, job_id="direct-job-1")
```

## Error Handling

The system handles errors gracefully:

- Invalid log formats are ignored
- Missing job mappings are logged as warnings
- Database errors are caught and logged
- Progress updates continue even if some fail

## Testing

Run the progress reporting tests:

```bash
# Unit tests
python -m pytest tests/test_progress_reporting.py -v

# Integration tests
python -m pytest tests/test_progress_integration.py -v

# Demo script
python examples/progress_reporting_demo.py
```

## Configuration

No additional configuration is required. The system uses the existing job tracking infrastructure and database connections.

## Monitoring

Progress can be monitored through:

1. **Job API endpoints** - Get job status with progress information
2. **Database queries** - Query job entities directly
3. **Log monitoring** - Watch for progress log messages
4. **Webhooks** - Receive notifications on job completion

## Best Practices

1. **Consistent Progress Reporting** - Use standardized progress message formats
2. **Meaningful Messages** - Provide descriptive progress messages
3. **Regular Updates** - Report progress at reasonable intervals (not too frequent)
4. **Error Handling** - Always handle progress reporting failures gracefully
5. **Job Mapping** - Ensure proper job ID mapping for remote workers
