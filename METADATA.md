# MoRAG Metadata Documentation

This document describes the metadata returned by MoRAG for each source file type during ingestion and processing.

## Document Files

### PDF Documents

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
  "chunk_count": 12
}
```

### Microsoft Word Documents (.docx, .doc)

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
  "chunk_count": 8
}
```

### Text Files (.txt, .md)

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
  "chunk_count": 3
}
```

## Audio Files

### Audio Processing Results

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
  "speaker_count": 2,
  "topic_count": 5,
  "word_count": 1250,
  "confidence_score": 0.92,
  "has_music": false,
  "noise_level": "low",
  "processing_time": 45.2
}
```

## Video Files

### Video Processing Results

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
  "video_codec": "h264",
  "audio_codec": "aac",
  "audio_sample_rate": 44100,
  "audio_channels": 2,
  "has_audio": true,
  "audio_duration": 600.0,
  "language": "en",
  "model_used": "whisper-large-v3",
  "speaker_count": 1,
  "topic_count": 8,
  "word_count": 2500,
  "confidence_score": 0.89,
  "frame_count": 18000,
  "thumbnail_generated": false,
  "processing_time": 120.5
}
```

## Image Files

### Image Processing Results

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
  "camera_make": "Canon",
  "camera_model": "EOS R5",
  "creation_date": "2024-01-15T14:30:00Z",
  "gps_coordinates": [40.7128, -74.0060],
  "orientation": 1,
  "color_space": "sRGB",
  "dpi": [300, 300],
  "caption": "A beautiful sunset over the mountains",
  "extracted_text": "Sign: Welcome to National Park",
  "objects_detected": ["mountain", "sky", "trees", "sign"],
  "text_regions": 1,
  "confidence_score": 0.94,
  "processing_time": 3.2
}
```

## Web Pages

### Web Content Processing Results

**Metadata Structure:**
```json
{
  "url": "https://example.com/article",
  "title": "Article Title",
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
  "link_count": 15,
  "image_count": 5,
  "has_structured_data": true,
  "schema_types": ["Article", "Organization"],
  "text_length": 8500,
  "chunk_count": 6,
  "processing_time": 2.1
}
```

## YouTube Videos

### YouTube Processing Results

**Metadata Structure:**
```json
{
  "url": "https://youtube.com/watch?v=example",
  "video_id": "example123",
  "title": "Video Title",
  "description": "Video description",
  "channel": "Channel Name",
  "channel_id": "UC1234567890",
  "upload_date": "2024-01-15",
  "duration": 900,
  "view_count": 50000,
  "like_count": 1200,
  "language": "en",
  "category": "Education",
  "tags": ["tag1", "tag2", "tag3"],
  "thumbnail_url": "https://img.youtube.com/vi/example/maxresdefault.jpg",
  "audio_extracted": true,
  "transcript_available": true,
  "auto_generated_captions": false,
  "speaker_count": 1,
  "topic_count": 5,
  "word_count": 3000,
  "confidence_score": 0.91,
  "processing_time": 180.3
}
```

## Common Fields

All metadata objects include these common fields:

- **filename/url**: Source identifier
- **file_size**: Size in bytes (for files)
- **processing_time**: Time taken to process in seconds
- **text_length**: Length of extracted text in characters
- **chunk_count**: Number of chunks created during processing
- **language**: Detected or specified language code
- **creation_date**: When the content was created (if available)
- **modification_date**: When the content was last modified (if available)

## Processing Quality Indicators

- **confidence_score**: Overall confidence in extraction quality (0.0-1.0)
- **extraction_quality**: Quality of text extraction (0.0-1.0)
- **model_used**: AI model used for processing (e.g., "whisper-large-v3", "gemini-pro-vision")

## Error Handling

When processing fails, metadata includes:

```json
{
  "error": "Error description",
  "error_type": "ProcessingError",
  "partial_processing": true,
  "recovered_content": "Any content that was successfully extracted"
}
```
