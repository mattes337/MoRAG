# MoRAG Metadata Documentation

This document describes the metadata returned by MoRAG for each source file type during ingestion and processing, including REST API endpoints and UI component guidance.

## REST API Endpoints for Metadata

### File Metadata Endpoints

- `GET /api/v1/files/info/{file_path}` - Get detailed metadata for a specific file
- `GET /api/v1/files/list` - List files with metadata (supports filtering by stage type and extension)
- `GET /api/v1/files/stats` - Get aggregated statistics about processed files
- `GET /api/v1/stages/status` - Get processing status and available files with metadata

### Stage Execution Endpoints

- `POST /api/v1/stages/execute/{stage_type}` - Execute a processing stage and get metadata
- `POST /api/v1/stages/chain` - Execute multiple stages and get comprehensive metadata
- `GET /api/v1/stages/info` - Get information about available processing stages

## Core Metadata Models

### StageFileMetadata (Base File Metadata)

**Returned by:** All file endpoints
**Structure:**
```json
{
  "filename": "document.pdf",
  "file_path": "/path/to/document.pdf",
  "file_size": 2048576,
  "created_at": "2024-01-15T10:30:00Z",
  "stage_type": "markdown-conversion",
  "content_type": "application/pdf",
  "checksum": "sha256:abc123...",
  "content": "base64_encoded_content_if_requested"
}
```

### DocumentMetadata (Core Document Model)

**Returned by:** Document processing stages
**Structure:**
```json
{
  "source_type": "pdf",
  "source_name": "document.pdf",
  "source_path": "/path/to/document.pdf",
  "source_url": null,
  "mime_type": "application/pdf",
  "file_size": 2048576,
  "checksum": "sha256:abc123...",
  "created_at": "2024-01-15T10:30:00Z",
  "modified_at": "2024-01-15T10:30:00Z",
  "author": "Author Name",
  "title": "Document Title",
  "language": "en",
  "page_count": 15,
  "word_count": 2500,
  "custom": {
    "extraction_quality": 0.95,
    "has_images": true,
    "has_tables": true,
    "processing_method": "docling"
  }
}
```

## Input Type-Specific Metadata

### Document Files

#### PDF Documents

**Processing Stage:** `markdown-conversion`
**Metadata Structure:**
```json
{
  "filename": "document.pdf",
  "file_size": 2048576,
  "file_extension": "pdf",
  "page_count": 15,
  "title": "Document Title",
  "author": "Author Name",
  "subject": "Document Subject",
  "creator": "PDF Creator Application",
  "producer": "PDF Producer",
  "creation_date": "2024-01-15T10:30:00Z",
  "modification_date": "2024-01-15T10:30:00Z",
  "language": "en",
  "processing_method": "docling",
  "extraction_quality": 0.95,
  "has_images": true,
  "has_tables": true,
  "text_length": 45000,
  "chunk_count": 12,
  "quality_score": 0.95,
  "quality_issues": [],
  "warnings": []
}
```

#### Microsoft Word Documents (.docx, .doc)

**Processing Stage:** `markdown-conversion`
**Metadata Structure:**
```json
{
  "filename": "document.docx",
  "file_size": 1024000,
  "file_extension": "docx",
  "title": "Document Title",
  "author": "Author Name",
  "subject": "Document Subject",
  "keywords": ["keyword1", "keyword2"],
  "creation_date": "2024-01-15T10:30:00Z",
  "modification_date": "2024-01-15T10:30:00Z",
  "language": "en",
  "word_count": 2500,
  "paragraph_count": 45,
  "page_count": 8,
  "has_images": false,
  "has_tables": true,
  "text_length": 15000,
  "chunk_count": 8,
  "quality_score": 0.92,
  "quality_issues": [],
  "warnings": []
}
```

#### Text Files (.txt, .md)

**Processing Stage:** `markdown-conversion`
**Metadata Structure:**
```json
{
  "filename": "document.txt",
  "file_size": 8192,
  "file_extension": "txt",
  "encoding": "utf-8",
  "line_count": 150,
  "language": "en",
  "text_length": 7500,
  "chunk_count": 3,
  "quality_score": 1.0,
  "quality_issues": [],
  "warnings": []
}
```

### Audio Files

#### AudioProcessingResult

**Processing Stage:** Audio transcription and analysis
**Metadata Structure:**
```json
{
  "filename": "audio.mp3",
  "file_size": 5242880,
  "file_extension": "mp3",
  "duration": 300.5,
  "sample_rate": 44100,
  "channels": 2,
  "bit_depth": 16,
  "bitrate": 128000,
  "format": "MP3",
  "language": "en",
  "model_used": "whisper-large-v3",
  "processing_time": 45.2,
  "word_count": 1250,
  "segment_count": 25,
  "has_speaker_info": true,
  "has_topic_info": true,
  "num_speakers": 2,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "num_topics": 5,
  "topics": [
    {
      "id": 0,
      "title": "Introduction",
      "summary": "Opening remarks and agenda overview",
      "start_time": 0.0,
      "end_time": 45.2,
      "duration": 45.2,
      "keywords": ["introduction", "agenda", "overview"],
      "speaker_distribution": {"SPEAKER_00": 0.8, "SPEAKER_01": 0.2}
    }
  ],
  "confidence_score": 0.92,
  "metadata_extraction_error": null
}
```

**UI Components:** Audio player with transcript, speaker timeline, topic segments, confidence indicators

### Video Files

#### VideoProcessingResult

**Processing Stage:** Video analysis with audio extraction
**Metadata Structure:**
```json
{
  "filename": "video.mp4",
  "file_size": 104857600,
  "file_extension": "mp4",
  "duration": 600.0,
  "width": 1920,
  "height": 1080,
  "fps": 30.0,
  "codec": "h264",
  "bitrate": 5000000,
  "format": "mp4",
  "has_audio": true,
  "audio_codec": "aac",
  "creation_time": "2024-01-15T14:30:00Z",
  "processing_time": 120.5,
  "audio_processing_result": {
    "transcript": "Full transcript text...",
    "segments": [...],
    "metadata": {
      "processing_time": 85.3,
      "word_count": 2500,
      "segment_count": 45,
      "has_speaker_info": true,
      "has_topic_info": true,
      "num_speakers": 1,
      "speakers": ["SPEAKER_00"],
      "num_topics": 8
    }
  },
  "thumbnails": ["/path/to/thumbnail1.jpg"],
  "keyframes": ["/path/to/keyframe1.jpg"],
  "ocr_results": {
    "text_detected": true,
    "extracted_text": "Text found in video frames",
    "confidence": 0.87
  },
  "transcript_length": 15000,
  "segments_count": 45,
  "has_speaker_diarization": true,
  "has_topic_segmentation": true
}
```

**UI Components:** Video player with transcript overlay, thumbnail gallery, keyframe navigation, OCR text display

### Image Files

#### ImageProcessingResult

**Processing Stage:** Image analysis with OCR and captioning
**Metadata Structure:**
```json
{
  "filename": "image.jpg",
  "file_size": 2048000,
  "file_extension": "jpg",
  "width": 2048,
  "height": 1536,
  "format": "JPEG",
  "mode": "RGB",
  "has_exif": true,
  "exif_data": {
    "camera_make": "Canon",
    "camera_model": "EOS R5",
    "creation_time": "2024-01-15T14:30:00Z",
    "gps_coordinates": [40.7128, -74.0060],
    "orientation": 1,
    "color_space": "sRGB",
    "dpi": [300, 300]
  },
  "creation_time": "2024-01-15T14:30:00Z",
  "camera_make": "Canon",
  "camera_model": "EOS R5",
  "caption": "A beautiful sunset over the mountains",
  "extracted_text": "Sign: Welcome to National Park",
  "processing_time": 3.2,
  "confidence_scores": {
    "caption_confidence": 0.94,
    "ocr_confidence": 0.87,
    "overall_confidence": 0.90
  }
}
```

**UI Components:** Image viewer with EXIF panel, caption display, OCR text overlay, confidence indicators

### Web Pages

#### WebContent Processing Result

**Processing Stage:** Web scraping and content extraction
**Metadata Structure:**
```json
{
  "url": "https://example.com/article",
  "title": "Article Title",
  "content": "Extracted main content...",
  "markdown_content": "# Article Title\n\nExtracted content in markdown...",
  "metadata": {
    "description": "Article description from meta tags",
    "author": "Author Name",
    "publication_date": "2024-01-15T10:00:00Z",
    "language": "en",
    "domain": "example.com",
    "content_type": "text/html",
    "status_code": 200,
    "word_count": 1500,
    "paragraph_count": 25,
    "heading_count": 8,
    "has_structured_data": true,
    "schema_types": ["Article", "Organization"]
  },
  "links": [
    "https://example.com/related1",
    "https://example.com/related2"
  ],
  "images": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.png"
  ],
  "extraction_time": 2.1,
  "content_length": 8500,
  "content_type": "text/html"
}
```

**UI Components:** Web page preview, link explorer, image gallery, metadata panel, content structure view

### YouTube Videos

#### YouTubeDownloadResult

**Processing Stage:** YouTube download and transcript extraction
**Metadata Structure:**
```json
{
  "video_path": "/path/to/downloaded_video.mp4",
  "audio_path": "/path/to/extracted_audio.mp3",
  "subtitle_paths": ["/path/to/subtitles.vtt"],
  "thumbnail_paths": ["/path/to/thumbnail.jpg"],
  "transcript_path": "/path/to/transcript.txt",
  "transcript_text": "Full transcript content...",
  "transcript_language": "en",
  "file_size": 104857600,
  "metadata": {
    "id": "example123",
    "title": "Video Title",
    "description": "Video description",
    "uploader": "Channel Name",
    "upload_date": "2024-01-15",
    "duration": 900.0,
    "view_count": 50000,
    "like_count": 1200,
    "comment_count": 85,
    "tags": ["tag1", "tag2", "tag3"],
    "categories": ["Education"],
    "thumbnail_url": "https://img.youtube.com/vi/example/maxresdefault.jpg",
    "webpage_url": "https://youtube.com/watch?v=example",
    "channel_id": "UC1234567890",
    "channel_url": "https://youtube.com/channel/UC1234567890",
    "playlist_id": null,
    "playlist_title": null,
    "playlist_index": null
  },
  "processing_time": 180.3,
  "success": true,
  "error_message": null
}
```

**UI Components:** YouTube player embed, channel info panel, transcript viewer, download status, metadata display

## Stage Execution Metadata

### StageExecutionResponse

**Returned by:** All stage execution endpoints
**Structure:**
```json
{
  "success": true,
  "stage_type": "markdown-conversion",
  "status": "completed",
  "output_files": [
    {
      "filename": "document.md",
      "file_path": "/output/document.md",
      "file_size": 52428,
      "created_at": "2024-01-15T10:31:00Z",
      "stage_type": "markdown-conversion",
      "content_type": "text/markdown",
      "checksum": "sha256:def456...",
      "content": null
    }
  ],
  "metadata": {
    "execution_time": 12.5,
    "start_time": "2024-01-15T10:30:45Z",
    "end_time": "2024-01-15T10:30:57Z",
    "input_files": ["/input/document.pdf"],
    "config_used": {
      "language": "en",
      "include_metadata": true
    },
    "warnings": []
  },
  "error_message": null,
  "webhook_sent": false
}
```

## Processing Pipeline Metadata

### Graph Processing Results

**Processing Stage:** `fact-generator`
**Structure:**
```json
{
  "success": true,
  "entities_count": 45,
  "relations_count": 78,
  "chunks_processed": 12,
  "processing_time": 25.3,
  "error_message": null,
  "metadata": {
    "model_used": "gemini-1.5-flash",
    "extraction_method": "llm_based",
    "confidence_threshold": 0.7
  },
  "database_results": [
    {
      "database_type": "neo4j",
      "entities_stored": 45,
      "relations_stored": 78,
      "success": true
    }
  ],
  "openie_enabled": false,
  "openie_relations_count": 0
}
```

## Common Metadata Fields

### Base Fields (All Content Types)

- **filename/url**: Source identifier
- **file_size**: Size in bytes (for files)
- **processing_time**: Time taken to process in seconds
- **success**: Whether processing completed successfully
- **error_message**: Error description if processing failed
- **created_at/modified_at**: File timestamps
- **content_type/mime_type**: MIME type of the content

### Quality Indicators

- **confidence_score**: Overall confidence in extraction quality (0.0-1.0)
- **quality_score**: Quality assessment of processing (0.0-1.0)
- **quality_issues**: Array of identified quality problems
- **warnings**: Array of non-fatal processing warnings

### Content Analysis

- **language**: Detected or specified language code
- **word_count**: Number of words in extracted text
- **text_length**: Length of extracted text in characters
- **chunk_count**: Number of chunks created during processing

## UI Component Guidelines

### Metadata Display Components

1. **File Info Panel**
   - Display basic file metadata (size, type, dates)
   - Show processing status and quality indicators
   - Include download/view actions

2. **Processing Status Indicator**
   - Visual progress for multi-stage processing
   - Error state display with retry options
   - Success confirmation with quality metrics

3. **Content-Specific Viewers**
   - **Audio**: Waveform with transcript overlay, speaker labels
   - **Video**: Player with keyframe navigation, OCR overlay
   - **Images**: Viewer with EXIF panel, caption display
   - **Documents**: Preview with quality indicators
   - **Web**: Structured content view with link exploration

4. **Metadata Explorer**
   - Expandable tree view of all metadata fields
   - Search and filter capabilities
   - Export functionality for metadata

### Error Handling in UI

When processing fails, display:
- Clear error message from `error_message` field
- Retry button for transient failures
- Partial results if `partial_processing` is true
- Contact support option for persistent errors

### Performance Considerations

- Use `processing_time` to show performance metrics
- Display `file_size` for upload/download estimates
- Show `chunk_count` for processing complexity indication
- Use `confidence_score` for result reliability indication

## Enhanced Query API Metadata

### Query Response Metadata

**Endpoint:** `POST /api/v2/query`
**Response includes metadata about:**

```json
{
  "success": true,
  "response": "Generated response text...",
  "results": [
    {
      "content": "Retrieved content chunk...",
      "metadata": {
        "source_file": "/path/to/document.pdf",
        "chunk_index": 5,
        "confidence_score": 0.89,
        "location_reference": "Page 3, Paragraph 2",
        "location_type": "page_paragraph"
      },
      "entities": [...],
      "relations": [...]
    }
  ],
  "graph_context": {
    "entities_found": 12,
    "relations_traversed": 8,
    "expansion_depth": 2
  },
  "processing_metadata": {
    "query_time": 1.25,
    "retrieval_time": 0.85,
    "generation_time": 0.40,
    "total_chunks_considered": 150,
    "chunks_used": 8
  }
}
```

### Entity Query Metadata

**Endpoint:** `POST /api/v2/entity/query`
**Returns detailed entity metadata:**

```json
{
  "entities": [
    {
      "id": "entity_123",
      "name": "Entity Name",
      "type": "PERSON",
      "confidence": 0.95,
      "properties": {
        "description": "Entity description",
        "aliases": ["Alternative Name"]
      },
      "source_documents": [
        {
          "document_id": "doc_456",
          "source_file": "/path/to/document.pdf",
          "mentions": 5,
          "first_mention_location": "Page 1"
        }
      ]
    }
  ]
}
```

**UI Components:** Entity explorer, relationship graph, source document navigator, confidence indicators
