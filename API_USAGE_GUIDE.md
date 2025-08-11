# MoRAG API Usage Guide

This guide provides comprehensive API usage examples for all MoRAG content types using the unified processing endpoint.

## 🚀 Unified Processing Endpoint

**MoRAG provides a single, unified endpoint for all processing needs:**

`POST /api/v1/process` - Handles all content types and processing modes

**Three Processing Modes:**
- **Convert Mode** (`mode=convert`) - Fast markdown conversion for UI preview
- **Process Mode** (`mode=process`) - Full processing with immediate results
- **Ingest Mode** (`mode=ingest`) - Full processing + vector storage (background)

**Three Source Types:**
- **File Upload** (`source_type=file`) - Upload files via multipart form
- **URL Processing** (`source_type=url`) - Process web content and URLs
- **Batch Processing** (`source_type=batch`) - Process multiple items

## 📊 Dual Format Output

**MoRAG provides optimized output formats:**

- **API Responses**: Structured JSON format for easy integration and webhooks
- **Vector Storage**: Markdown format for optimal vector search and retrieval

This ensures that:
- API consumers receive well-structured JSON data
- Vector storage maintains human-readable markdown for better search quality
- Both formats are generated efficiently during processing

## 📚 API Usage Examples

### 1. Convert Mode - Fast Markdown Conversion

```bash
# Convert PDF to markdown for UI preview
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@document.pdf" \
  -F 'request_data={"mode":"convert","source_type":"file"}'

# Convert image with text extraction
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@image.png" \
  -F 'request_data={"mode":"convert","source_type":"file","content_type":"image"}'
```

### 2. Process Mode - Full Processing with Immediate Results

```bash
# Process web page with immediate results
curl -X POST "http://localhost:8000/api/v1/process" \
  -F 'request_data={
    "mode": "process",
    "source_type": "url",
    "url": "https://example.com",
    "content_type": "web"
  }'

# Process YouTube video with immediate results
curl -X POST "http://localhost:8000/api/v1/process" \
  -F 'request_data={
    "mode": "process",
    "source_type": "url",
    "url": "https://youtube.com/watch?v=VIDEO_ID",
    "content_type": "youtube"
  }'

# Process uploaded file with immediate results
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@audio.mp3" \
  -F 'request_data={
    "mode": "process",
    "source_type": "file",
    "content_type": "audio"
  }'
```

### 3. Ingest Mode - Background Processing + Vector Storage

```bash
# Ingest document with webhook notifications
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@document.pdf" \
  -F 'request_data={
    "mode": "ingest",
    "source_type": "file",
    "webhook_config": {
      "url": "https://your-app.com/webhook"
    },
    "document_id": "doc-123"
  }'

# Ingest URL content with background processing
curl -X POST "http://localhost:8000/api/v1/process" \
  -F 'request_data={
    "mode": "ingest",
    "source_type": "url",
    "url": "https://example.com/article",
    "webhook_config": {
      "url": "https://your-app.com/webhook"
    }
  }'
```

### 4. Audio Processing with Structured JSON Output

```bash
# Process audio file with full features
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@audio.mp3" \
  -F 'request_data={
    "mode": "process",
    "source_type": "file",
    "content_type": "audio",
    "processing_options": {
      "language": "en",
      "include_metadata": true
    }
  }'
```

**Expected JSON Response Format:**
```json
{
  "success": true,
  "content": {
    "title": "audio",
    "filename": "audio.mp3",
    "metadata": {
      "duration": 120.5,
      "language": "en",
      "num_speakers": 2,
      "segment_count": 15
    },
    "topics": [
      {
        "timestamp": 0,
        "sentences": [
          {
            "timestamp": 0,
            "speaker": 1,
            "text": "Hello, welcome to our discussion."
          },
          {
            "timestamp": 5,
            "speaker": 2,
            "text": "Thank you for having me."
          }
        ]
      },
      {
        "timestamp": 60,
        "sentences": [
          {
            "timestamp": 60,
            "speaker": 1,
            "text": "Let's move to the next topic."
          }
        ]
      }
    ]
  },
  "metadata": {},
  "processing_time": 15.2
}
```

### 5. Video Processing with Structured JSON Output

```bash
# Process video file with full features
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@video.mp4" \
  -F 'request_data={
    "mode": "process",
    "source_type": "file",
    "content_type": "video",
    "processing_options": {
      "include_thumbnails": true,
      "include_metadata": true
    }
  }'
```

**Expected JSON Response Format:**
```json
{
  "success": true,
  "content": {
    "title": "video",
    "filename": "video.mp4",
    "metadata": {
      "duration": 300.0,
      "resolution": "1920x1080",
      "fps": 30.0,
      "format": "mp4",
      "has_audio": true,
      "transcript_length": 1500,
      "segments_count": 25,
      "has_speaker_diarization": true,
      "has_topic_segmentation": true
    },
    "topics": [
      {
        "timestamp": 0,
        "sentences": [
          {
            "timestamp": 0,
            "speaker": 1,
            "text": "Welcome to this video tutorial."
          }
        ]
      }
    ]
  },
  "metadata": {},
  "processing_time": 45.8
}
```

### 6. Document Processing with Chapter Splitting

```bash
# Process PDF document with chapter splitting
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@document.pdf" \
  -F 'request_data={
    "mode": "process",
    "source_type": "file",
    "content_type": "document",
    "processing_options": {
      "chunking_strategy": "chapter",
      "include_metadata": true
    }
  }'
```

**Expected JSON Response Format:**
```json
{
  "success": true,
  "content": {
    "title": "Document Title",
    "filename": "document.pdf",
    "metadata": {
      "source_type": "pdf",
      "page_count": 50,
      "word_count": 15000,
      "author": "Author Name",
      "created_at": "2024-01-01T00:00:00",
      "quality_score": 0.95,
      "chunks_count": 5,
      "processing_time": 12.3
    },
    "chapters": [
      {
        "title": "Chapter 1: Introduction",
        "content": "This is the introduction chapter content...",
        "page_number": 1,
        "chapter_index": 0,
        "metadata": {
          "chapter_number": 1,
          "start_page": 1,
          "end_page": 8,
          "page_count": 8
        }
      },
      {
        "title": "Chapter 2: Methods",
        "content": "This chapter describes the methods...",
        "page_number": 9,
        "chapter_index": 1,
        "metadata": {
          "chapter_number": 2,
          "start_page": 9,
          "end_page": 20,
          "page_count": 12
        }
      }
    ]
  },
  "metadata": {},
  "processing_time": 12.3
}
```

### 7. Batch Processing

```bash
# Process multiple items in one request
curl -X POST "http://localhost:8000/api/v1/process" \
  -F 'request_data={
    "mode": "ingest",
    "source_type": "batch",
    "items": [
      {"url": "https://example.com/article1", "type": "web"},
      {"url": "https://youtube.com/watch?v=abc", "type": "youtube"}
    ],
    "webhook_config": {
      "url": "https://your-app.com/webhook"
    }
  }'
```

## 🔄 Unified Endpoint Summary

| Mode | Purpose | Response | Storage |
|------|---------|----------|---------|
| `convert` | Fast markdown conversion | Immediate markdown | None |
| `process` | Full processing | Immediate structured JSON | None |
| `ingest` | Full processing + storage | Task ID | Vector database |

| Source Type | Input Method | Supported Content |
|-------------|--------------|-------------------|
| `file` | Multipart upload | Documents, audio, video, images |
| `url` | JSON request | Web pages, YouTube videos |
| `batch` | JSON array | Multiple mixed items |

## 🎯 Processing Options

### Audio Processing Options
```json
{
  "processing_options": {
    "language": "en",                 // Language hint or auto-detect
    "chunking_strategy": "topic",     // topic, time, sentence
    "chunk_size": 4000,              // Characters per chunk
    "chunk_overlap": 200,            // Overlap between chunks
    "include_metadata": true         // Include processing metadata
  }
}
```

### Video Processing Options
```json
{
  "processing_options": {
    "include_thumbnails": false,     // Generate video thumbnails
    "language": "en",                // Language hint or auto-detect
    "chunking_strategy": "topic",    // topic, time, sentence
    "chunk_size": 4000,             // Characters per chunk
    "include_metadata": true        // Include processing metadata
  }
}
```

### Document Processing Options
```json
{
  "processing_options": {
    "chunking_strategy": "chapter",  // chapter, page, paragraph, semantic
    "chunk_size": 4000,             // Characters per chunk (non-chapter)
    "chunk_overlap": 200,           // Overlap between chunks
    "language": "en",               // Language hint
    "include_metadata": true        // Include document metadata
  }
}
```

### Database Configuration (for Ingest Mode)
```json
{
  "database_config": {
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "password",
    "qdrant_url": "http://localhost:6333",
    "qdrant_api_key": "optional-api-key",
    "collection_name": "my_documents"
  }
}
```

### Webhook Configuration (for Ingest Mode)
```json
{
  "webhook_config": {
    "url": "https://your-app.com/webhook",
    "auth_token": "optional-bearer-token"
  }
}
```

**Available Chunking Strategies:**
- `chapter`: Split by detected chapters with page numbers (documents)
- `page`: Split by page boundaries (PDFs)
- `paragraph`: Split by paragraphs (articles)
- `semantic`: Smart semantic splitting (recommended)
- `topic`: Split by topic changes (audio/video)
- `time`: Split by time intervals (audio/video)

## 🚀 Python API Usage

```python
import asyncio
import httpx
import json

async def main():
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        # Convert mode - fast markdown conversion
        with open("document.pdf", "rb") as f:
            response = await client.post(
                f"{base_url}/api/v1/process",
                files={"file": f},
                data={"request_data": json.dumps({
                    "mode": "convert",
                    "source_type": "file"
                })}
            )
        convert_result = response.json()

        # Process mode - full processing
        response = await client.post(
            f"{base_url}/api/v1/process",
            data={"request_data": json.dumps({
                "mode": "process",
                "source_type": "url",
                "url": "https://example.com",
                "content_type": "web"
            })}
        )
        process_result = response.json()

        # Ingest mode - background processing + storage
        with open("audio.mp3", "rb") as f:
            response = await client.post(
                f"{base_url}/api/v1/process",
                files={"file": f},
                data={"request_data": json.dumps({
                    "mode": "ingest",
                    "source_type": "file",
                    "webhook_config": {
                        "url": "https://your-app.com/webhook"
                    }
                })}
            )
        ingest_result = response.json()

        print(f"Convert: {convert_result['success']}")
        print(f"Process: {process_result['success']}")
        print(f"Ingest Task ID: {ingest_result['task_id']}")

asyncio.run(main())
```

## 🗄️ Task Management (for Ingest Mode)

When using `mode=ingest`, MoRAG processes content in the background and provides task tracking capabilities.

### Automatic Content Type Detection

MoRAG automatically detects content types based on file extensions and URL patterns:

- **Files**: Detects based on file extension (.pdf → document, .mp3 → audio, .mp4 → video, etc.)
- **URLs**: Detects YouTube URLs, web pages, and other URL patterns
- **Fallback**: You can still specify `content_type` explicitly if needed

Supported auto-detection:
- Documents: `.pdf`, `.docx`, `.txt`, `.md`, `.html`, etc.
- Audio: `.mp3`, `.wav`, `.flac`, `.m4a`, etc.
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, etc.
- Images: `.jpg`, `.png`, `.gif`, `.webp`, etc.
- YouTube: `youtube.com` and `youtu.be` URLs
- Web: Other HTTP/HTTPS URLs

### Task Status Monitoring

```bash
# Check task status (when using mode=ingest)
curl "http://localhost:8000/api/v1/status/abc123-def456-ghi789"

# Response
{
  "task_id": "abc123-def456-ghi789",
  "status": "PROGRESS",
  "progress": 0.75,
  "message": "Processing chunks",
  "result": null,
  "error": null
}

# List active tasks
curl "http://localhost:8000/api/v1/status/"

# Get queue statistics
curl "http://localhost:8000/api/v1/status/stats/queues"

# Cancel a task
curl -X DELETE "http://localhost:8000/api/v1/status/abc123-def456-ghi789"
```

### Search Stored Content

```bash
# Search the vector database (content stored via mode=ingest)
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "limit": 5,
    "filters": {"category": "research"}
  }'
```

## 🔄 Processing Mode Comparison

| Mode | Response | Storage | Use Case | Background |
|------|----------|---------|----------|------------|
| `convert` | Immediate markdown | None | UI preview | Synchronous |
| `process` | Immediate structured JSON | None | One-time analysis | Synchronous |
| `ingest` | Task ID | Vector database | Searchable knowledge base | Asynchronous |
