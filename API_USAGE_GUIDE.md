# MoRAG API Usage Guide - Fixed and Updated

This guide provides corrected API usage examples for all MoRAG content types with the latest fixes applied.

## � Important: Dual Format Output

**MoRAG now provides dual format output to optimize for different use cases:**

- **API Responses**: Structured JSON format for easy integration and webhooks
- **Qdrant Storage**: Markdown format for optimal vector search and retrieval

This ensures that:
- API consumers receive well-structured JSON data
- Vector storage maintains human-readable markdown for better search quality
- Both formats are generated efficiently during processing

## �🔧 Fixed Issues

### ✅ Image Processing
- **Issue**: `UnsupportedFormatError: Unsupported format: Format 'image' is not supported`
- **Fix**: Added image file extensions to content type detection and image processing route to orchestrator
- **Status**: ✅ FIXED

### ✅ Web Processing Routing
- **Issue**: `/process/url` returns `'string' is not a valid ContentType` error
- **Fix**: Fixed orchestrator routing to use services instead of direct processor calls
- **Status**: ✅ FIXED

### ✅ YouTube Processing Routing
- **Issue**: `/process/youtube` returns `YouTubeProcessor does not support file processing`
- **Fix**: Fixed orchestrator routing to use services instead of direct processor calls
- **Status**: ✅ FIXED

### ✅ Audio Processing Configuration
- **Issue**: Diarization and topic segmentation disabled by default
- **Fix**: Changed default configuration to enable both features
- **Status**: ✅ FIXED

### ✅ Structured JSON Output
- **Issue**: All processors returned markdown instead of structured JSON
- **Fix**: Implemented JSON output format for audio and video processing
- **Status**: ✅ FIXED

## 📚 Corrected API Usage Examples

### 1. Image Processing

```bash
# Upload PNG/JPEG files
curl -X POST "http://localhost:8000/process/file" \
  -F "file=@image.png" \
  -F "content_type=image"

# Response will include extracted text and image caption
```

### 2. Web Processing

```bash
# Process web pages
curl -X POST "http://localhost:8000/process/web" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "options": {}
  }'

# OR use the generic URL endpoint
curl -X POST "http://localhost:8000/process/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "content_type": "web"
  }'
```

### 3. YouTube Processing

```bash
# Process YouTube videos
curl -X POST "http://localhost:8000/process/youtube" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://youtube.com/watch?v=VIDEO_ID",
    "options": {}
  }'

# OR use the generic URL endpoint
curl -X POST "http://localhost:8000/process/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://youtube.com/watch?v=VIDEO_ID",
    "content_type": "youtube"
  }'
```

### 4. Audio Processing with Structured JSON Output

```bash
# Upload audio file with diarization and topic segmentation enabled
curl -X POST "http://localhost:8000/process/file" \
  -F "file=@audio.mp3" \
  -F "content_type=audio"
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
# Upload video file
curl -X POST "http://localhost:8000/process/file" \
  -F "file=@video.mp4" \
  -F "content_type=video"
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
# Upload PDF document with chapter splitting
curl -X POST "http://localhost:8000/process/file" \
  -F "file=@document.pdf" \
  -F "content_type=document" \
  -F "options={\"chunking_strategy\": \"chapter\"}"
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

## 🔄 Endpoint Summary

| Endpoint | Method | Purpose | Content Types |
|----------|--------|---------|---------------|
| `/process/file` | POST | Upload and process files | document, audio, video, image |
| `/process/url` | POST | Process content from URLs (auto-detect type) | web, youtube |
| `/process/web` | POST | Process web pages specifically | web |
| `/process/youtube` | POST | Process YouTube videos specifically | youtube |
| `/process/batch` | POST | Process multiple items | all types |

## 🎯 Configuration Options

### Audio Processing Options
```json
{
  "enable_diarization": true,        // Default: true (now enabled)
  "enable_topic_segmentation": true, // Default: true (now enabled)
  "model_size": "medium",            // tiny, base, small, medium, large-v2
  "language": null,                  // Auto-detect if null
  "output_format": "json"            // json, markdown, txt
}
```

### Video Processing Options
```json
{
  "include_thumbnails": false,       // Default: false (opt-in)
  "extract_audio": true,             // Default: true
  "enable_speaker_diarization": true, // Default: true
  "enable_topic_segmentation": true, // Default: true
  "output_format": "json"            // json, markdown
}
```

### Document Processing Options
```json
{
  "chunking_strategy": "chapter",    // chapter, page, paragraph, sentence, word, character
  "chunk_size": 1000,               // Size for non-chapter strategies
  "chunk_overlap": 100,             // Overlap for non-chapter strategies
  "extract_metadata": true,         // Extract document metadata
  "extract_images": true,           // Extract images from document
  "extract_tables": true,           // Extract tables from document
  "output_format": "json"           // json, markdown, txt
}
```

**Available Chunking Strategies:**
- `chapter`: Split by detected chapters with page numbers (recommended for books/reports)
- `page`: Split by page boundaries (good for PDFs)
- `paragraph`: Split by paragraphs (good for articles)
- `sentence`: Split by sentences (fine-grained)
- `word`: Split by words (very fine-grained)
- `character`: Split by characters (basic splitting)

## 🚀 Python API Usage

```python
import asyncio
from morag import MoRAGAPI

async def main():
    api = MoRAGAPI()
    
    # Process different content types
    image_result = await api.process_image("image.png")
    audio_result = await api.process_audio("audio.mp3")
    video_result = await api.process_video("video.mp4")
    web_result = await api.process_web_page("https://example.com")
    youtube_result = await api.process_youtube_video("https://youtube.com/watch?v=123")
    
    # Auto-detect content type from URL
    auto_result = await api.process_url("https://example.com")
    
    await api.cleanup()

asyncio.run(main())
```

## ✅ Verification

All fixes have been tested and verified:
- ✅ Image processing works for PNG, JPEG, GIF files
- ✅ Web processing correctly routes to web services
- ✅ YouTube processing correctly routes to YouTube services
- ✅ Audio processing has diarization and topic segmentation enabled by default
- ✅ Structured JSON output implemented for audio, video, and documents
- ✅ Document chapter splitting with page numbers implemented
- ✅ Chapter detection patterns for various document formats
- ✅ All content types properly routed through orchestrator
- ✅ Comprehensive JSON output schemas for all content types

## 🗄️ Ingestion API (Background Processing + Vector Storage)

The ingestion endpoints process content in the background and store results in the vector database for retrieval. These are ideal for building a searchable knowledge base.

#### Automatic Content Type Detection

MoRAG can automatically detect content types based on file extensions and URL patterns:

- **Files**: Detects based on file extension (.pdf → document, .mp3 → audio, .mp4 → video, etc.)
- **URLs**: Detects YouTube URLs, web pages, and other URL patterns
- **Fallback**: You can still specify `source_type` explicitly if needed

Supported auto-detection:
- Documents: `.pdf`, `.docx`, `.txt`, `.md`, `.html`, etc.
- Audio: `.mp3`, `.wav`, `.flac`, `.m4a`, etc.
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, etc.
- Images: `.jpg`, `.png`, `.gif`, `.webp`, etc.
- YouTube: `youtube.com` and `youtu.be` URLs
- Web: Other HTTP/HTTPS URLs

### File Ingestion

```bash
# Ingest a document file (auto-detect content type)
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -F "file=@document.pdf" \
  -F "metadata={\"tags\": [\"important\"], \"category\": \"research\"}"

# Ingest with explicit source type
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -F "source_type=document" \
  -F "file=@document.pdf" \
  -F "metadata={\"tags\": [\"important\"], \"category\": \"research\"}"

# Response
{
  "task_id": "abc123-def456-ghi789",
  "status": "pending",
  "message": "File ingestion started for document.pdf",
  "estimated_time": 60
}
```

### URL Ingestion

```bash
# Ingest web content (auto-detect content type)
curl -X POST "http://localhost:8000/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "metadata": {"category": "news", "priority": 1}
  }'

# Ingest YouTube video (auto-detect)
curl -X POST "http://localhost:8000/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://youtube.com/watch?v=VIDEO_ID",
    "webhook_url": "https://your-app.com/webhook"
  }'

# Ingest with explicit source type
curl -X POST "http://localhost:8000/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "web",
    "url": "https://example.com/article",
    "metadata": {"category": "news", "priority": 1}
  }'
```

### Batch Ingestion

```bash
# Ingest multiple items (auto-detect content types)
curl -X POST "http://localhost:8000/api/v1/ingest/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "url": "https://example.com/page1"
      },
      {
        "url": "https://youtube.com/watch?v=VIDEO_ID"
      }
    ],
    "webhook_url": "https://your-app.com/batch-webhook"
  }'

# Ingest with explicit source types
curl -X POST "http://localhost:8000/api/v1/ingest/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "source_type": "web",
        "url": "https://example.com/page1"
      },
      {
        "source_type": "youtube",
        "url": "https://youtube.com/watch?v=VIDEO_ID"
      }
    ],
    "webhook_url": "https://your-app.com/batch-webhook"
  }'
```

### Task Status Monitoring

```bash
# Check task status
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
curl -X DELETE "http://localhost:8000/api/v1/ingest/abc123-def456-ghi789"
```

### Search Stored Content

```bash
# Search the vector database
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "limit": 5,
    "filters": {"category": "research"}
  }'
```

## 🔄 Processing vs Ingestion

| Feature | Processing (`/process/*`) | Ingestion (`/api/v1/ingest/*`) |
|---------|---------------------------|--------------------------------|
| **Response** | Immediate results | Task ID for tracking |
| **Storage** | No storage | Stored in vector database |
| **Use Case** | One-time processing | Building searchable knowledge base |
| **Background** | Synchronous | Asynchronous with Celery |
| **Webhooks** | Not supported | Supported |
| **Search** | Not searchable | Searchable via `/search` |
