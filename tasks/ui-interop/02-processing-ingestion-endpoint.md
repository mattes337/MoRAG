# Processing/Ingestion Endpoint with Webhooks

## Overview
Implement a comprehensive document processing endpoint that runs the full MoRAG pipeline with real-time progress updates via webhooks.

## Endpoint Specification
**Endpoint**: `POST /api/process/ingest`

**Purpose**: Complete document processing with real-time progress updates via webhooks.

## Input Requirements
- File upload (multipart/form-data)
- Optional document ID (string) for deduplication
- Webhook URL for progress notifications
- Processing configuration options

## Webhook Progress Updates
Each processing step should trigger a webhook POST to the provided URL with:

```json
{
  "document_id": "optional-user-provided-id",
  "step": "markdown_conversion|metadata_extraction|chunking|graph_extraction|ingestion",
  "status": "started|completed|failed",
  "progress_percent": 0-100,
  "timestamp": "ISO8601",
  "data": {
    // Step-specific data (see below)
  }
}
```

## Step-Specific Webhook Data

### 1. Markdown Conversion Complete
```json
{
  "markdown_file_url": "/api/files/temp/{id}/markdown.md",
  "conversion_metadata": {
    "original_format": "pdf|mp4|mp3|txt|...",
    "page_count": 123,
    "duration_seconds": 3600,
    "file_size_bytes": 1048576
  }
}
```

### 2. Metadata Extraction Complete
```json
{
  "metadata_file_url": "/api/files/temp/{id}/metadata.json",
  "metadata": {
    "format": "video/mp4",
    "codecs": ["h264", "aac"],
    "resolution": "1920x1080",
    "duration": 3600,
    "bitrate": 2000000
  }
}
```

### 3. Summary and Analysis Complete
```json
{
  "summary": "Document summary text...",
  "detected_topics": ["topic1", "topic2", "topic3"],
  "detected_speakers": ["Speaker A", "Speaker B"],
  "language": "en"
}
```

### 4. Ingestion Complete
```json
{
  "facts_count": 156,
  "keywords_count": 89,
  "top_keywords": [
    {"keyword": "machine learning", "usage_count": 23},
    {"keyword": "neural networks", "usage_count": 18},
    {"keyword": "data processing", "usage_count": 15},
    {"keyword": "algorithms", "usage_count": 12},
    {"keyword": "optimization", "usage_count": 10},
    {"keyword": "training", "usage_count": 9},
    {"keyword": "validation", "usage_count": 8},
    {"keyword": "performance", "usage_count": 7},
    {"keyword": "accuracy", "usage_count": 6},
    {"keyword": "evaluation", "usage_count": 5}
  ],
  "database_collection": "documents_en",
  "processing_time_seconds": 45.2
}
```

## Implementation Requirements

### Webhook Delivery System
- Reliable webhook delivery with retry logic
- Configurable timeout settings
- Fallback polling endpoint for progress status
- Comprehensive logging of all webhook delivery attempts

### Error Handling
- Send webhook notifications for failed steps with error details
- Include error messages and stack traces in failure notifications
- Graceful degradation when webhook delivery fails
- Recovery mechanisms for partial processing failures

### Security Considerations
- Validate webhook URLs (no localhost/internal IPs unless in dev mode)
- Implement webhook authentication (optional bearer token)
- Rate limiting for processing endpoints
- File size limits and type validation

### Configuration Options
- Webhook timeout settings
- Retry attempts and backoff strategies
- Maximum file sizes per endpoint
- Supported file types per endpoint

## Testing Requirements
- Integration tests with mock webhook servers
- Error scenario testing (network failures, invalid webhooks, etc.)
- Load testing for concurrent processing requests
- Webhook delivery reliability tests

## Dependencies
- Existing MoRAG processing services
- Webhook delivery system (consider using Celery or similar)
- Background task queue for long-running operations
- Temporary file storage system
