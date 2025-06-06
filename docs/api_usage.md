# MoRAG Ingestion API Usage Guide

## Overview

The MoRAG Ingestion API provides RESTful endpoints for processing multimodal content including documents, audio, video, images, web pages, and YouTube videos. All content is processed asynchronously and stored in a vector database for retrieval.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. This will be added in future versions.

For production deployments, authentication can be configured through the service configuration.

## Endpoints

### Health Check

Check if the API is running:

```bash
GET /health/
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {}
}
```

### File Upload Ingestion

Upload and process files (documents, audio, video, images):

```bash
POST /api/v1/ingest/file
```

**Parameters:**
- `source_type` (form): `document`, `audio`, `video`, or `image`
- `file` (file): The file to upload
- `webhook_url` (form, optional): URL to notify when processing completes
- `metadata` (form, optional): JSON string with additional metadata
- `use_docling` (form, optional): Use docling for PDF parsing (default: false)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -F "source_type=document" \
  -F "file=@document.pdf" \
  -F "metadata={\"tags\": [\"important\"], \"priority\": 1}"
```

**Response:**
```json
{
  "task_id": "abc123-def456-ghi789",
  "status": "pending",
  "message": "File ingestion started for document.pdf",
  "estimated_time": 60
}
```

### URL Ingestion

Process content from URLs (web pages, YouTube videos):

```bash
POST /api/v1/ingest/url
```

**Body:**
```json
{
  "source_type": "web",  // or "youtube"
  "url": "https://example.com",
  "webhook_url": "https://your-webhook.com/notify",  // optional
  "metadata": {  // optional
    "tags": ["web", "article"],
    "category": "news"
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "web",
    "url": "https://example.com/article",
    "metadata": {"category": "news"}
  }'
```

### Batch Ingestion

Process multiple URLs in a single request:

```bash
POST /api/v1/ingest/batch
```

**Body:**
```json
{
  "items": [
    {
      "source_type": "web",
      "url": "https://example.com/page1"
    },
    {
      "source_type": "youtube",
      "url": "https://www.youtube.com/watch?v=VIDEO_ID"
    }
  ],
  "webhook_url": "https://your-webhook.com/batch-notify"  // optional
}
```

**Response:**
```json
{
  "batch_id": "batch-abc123",
  "task_ids": ["task-1", "task-2"],
  "total_items": 2,
  "message": "Batch ingestion started with 2 tasks"
}
```

### Task Status

Check the status of a processing task:

```bash
GET /api/v1/status/{task_id}
```

**Response:**
```json
{
  "task_id": "abc123-def456-ghi789",
  "status": "PROGRESS",
  "progress": 0.75,
  "message": "Processing chunks",
  "result": null,
  "error": null,
  "created_at": "2024-01-01T12:00:00",
  "started_at": "2024-01-01T12:00:05",
  "completed_at": null,
  "estimated_time_remaining": 30
}
```

**Status Values:**
- `PENDING`: Task is queued
- `PROGRESS`: Task is being processed
- `SUCCESS`: Task completed successfully
- `FAILURE`: Task failed
- `REVOKED`: Task was cancelled

### List Active Tasks

Get all currently active tasks:

```bash
GET /api/v1/status/
```

**Response:**
```json
{
  "active_tasks": ["task-1", "task-2", "task-3"],
  "count": 3
}
```

### Queue Statistics

Get processing queue statistics:

```bash
GET /api/v1/status/stats/queues
```

**Response:**
```json
{
  "pending": 5,
  "active": 2,
  "completed": 100,
  "failed": 3
}
```

### Cancel Task

Cancel a running or pending task:

```bash
DELETE /api/v1/ingest/{task_id}
```

**Response:**
```json
{
  "message": "Task abc123-def456-ghi789 cancelled successfully"
}
```

## Supported File Types

### Documents
- PDF (`.pdf`)
- Microsoft Word (`.docx`, `.doc`)
- Markdown (`.md`)
- Plain text (`.txt`)

### Audio
- MP3 (`.mp3`)
- WAV (`.wav`)
- M4A (`.m4a`)
- OGG (`.ogg`)
- FLAC (`.flac`)

### Video
- MP4 (`.mp4`)
- MOV (`.mov`)
- AVI (`.avi`)
- WebM (`.webm`)
- MKV (`.mkv`)

### Images
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- GIF (`.gif`)
- BMP (`.bmp`)
- TIFF (`.tiff`)
- WebP (`.webp`)

## File Size Limits

- Documents: 100MB (configurable via settings)
- Audio: 2GB (configurable via settings)
- Video: 5GB (configurable via settings)
- Images: 50MB (configurable via settings)

## Metadata Schema

You can include custom metadata with your ingestion requests:

```json
{
  "tags": ["tag1", "tag2"],
  "categories": ["category1"],
  "custom_fields": {
    "author": "John Doe",
    "department": "Research"
  },
  "notes": "Important document for project X",
  "priority": 3  // 1-5 scale
}
```

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (missing or invalid API key)
- `422`: Validation Error (invalid data format)
- `500`: Internal Server Error

Error responses include details:

```json
{
  "detail": "File too large: 150MB (max: 100MB)"
}
```

## Rate Limiting

- Maximum 50 items per batch request
- File uploads are limited by size, not count
- No explicit rate limiting on API calls

## Webhooks

When processing completes, the system can notify your webhook URL:

**Webhook Payload:**
```json
{
  "task_id": "abc123-def456-ghi789",
  "status": "SUCCESS",
  "result": {
    "chunks_processed": 15,
    "total_text_length": 5000,
    "metadata": {...}
  },
  "completed_at": "2024-01-01T12:05:30"
}
```

## Examples

See `examples/api_demo.py` for a complete Python example demonstrating all API features.

## Interactive Documentation

When the API is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI Schema: `http://localhost:8000/openapi.json`
