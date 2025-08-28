# MoRAG YouTube

YouTube video transcription capabilities for the MoRAG (Multimodal RAG Ingestion Pipeline) system using external transcription services.

## Features

- **YouTube video transcription** using external WhisperX API
- **Metadata extraction** from external service
- **Optional video downloading** (opt-in only)
- **Multiple language support**
- **Batch processing** for multiple videos
- **Health monitoring** for external service
- **Async processing** for performance
- **Extended timeout support** for long transcriptions

## Installation

```bash
pip install morag-youtube
```

## Configuration

Set the external service URL (optional, defaults to localhost):

```bash
export YOUTUBE_SERVICE_URL="http://localhost:8000"
```

## Usage

### Basic Transcription

```python
import asyncio
from morag_youtube import YouTubeProcessor, YouTubeConfig

async def transcribe_youtube_video():
    # Create processor
    processor = YouTubeProcessor()

    # Configure processing
    config = YouTubeConfig(
        service_url="http://localhost:8000",  # Optional, uses env var or default
        service_timeout=300,  # 5 minutes timeout
        download_video=False  # Only transcribe, don't download video
    )

    # Process YouTube URL
    result = await processor.process_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ", config)

    if result.success:
        # Access metadata
        print(f"Title: {result.metadata.title}")
        print(f"Uploader: {result.metadata.uploader}")
        print(f"Duration: {result.metadata.duration} seconds")

        # Access transcription results
        if result.transcript:
            print(f"Transcript available with {len(result.transcript.get('entries', []))} segments")
            print(f"Languages: {[lang['language'] for lang in result.transcript_languages]}")

        # Access video file if downloaded
        if result.video_path:
            print(f"Video downloaded to: {result.video_path}")

# Run the example
asyncio.run(transcribe_youtube_video())
```

### With Video Download

```python
async def transcribe_and_download():
    processor = YouTubeProcessor()

    config = YouTubeConfig(
        download_video=True,  # Enable video download
        output_dir=Path("./downloads")  # Specify download directory
    )

    result = await processor.process_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ", config)

    if result.success:
        print(f"Transcription completed!")
        if result.video_path:
            print(f"Video saved to: {result.video_path}")
        print(f"Transcript segments: {len(result.transcript.get('entries', []))}")

asyncio.run(transcribe_and_download())
```

## External Service

This package requires an external YouTube transcription service (Modal WhisperX API) to be running. See `YOUTUBE.md` for details on the service API.

### Service Health Check

```python
from morag_youtube import YouTubeService

async def check_service():
    service = YouTubeService()
    health = await service.health_check()

    if health["status"] == "healthy":
        print("✅ Service is ready")
    else:
        print("❌ Service is unavailable")

asyncio.run(check_service())
```

## Batch Processing

```python
async def process_multiple_videos():
    service = YouTubeService()

    urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=oHg5SJYRHA0"
    ]

    results = await service.process_videos(urls)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Failed: {urls[i]} - {str(result)}")
        else:
            print(f"✅ Success: {urls[i]} - {result.metadata.title}")

asyncio.run(process_multiple_videos())
```

## CLI Usage

The CLI provides simple commands for transcribing YouTube videos using the external service:

```bash
# Check external service health
python -m morag_youtube.cli health

# Transcribe a single video (transcription only)
python -m morag_youtube.cli transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Transcribe and download video file
python -m morag_youtube.cli transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --download-video

# Specify custom service URL and timeout
python -m morag_youtube.cli --service-url "http://my-service:8000" --timeout 600 transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Batch process multiple videos
python -m morag_youtube.cli batch "https://www.youtube.com/watch?v=dQw4w9WgXcQ" "https://www.youtube.com/watch?v=oHg5SJYRHA0"

# Save results in text format instead of JSON
python -m morag_youtube.cli transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output-format text

# Specify output directory for downloaded videos
python -m morag_youtube.cli transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --download-video -o ./downloads
```

### Environment Variables

```bash
# Set external service URL (optional, defaults to localhost:8000)
export YOUTUBE_SERVICE_URL="http://localhost:8000"

# Then use CLI without --service-url
python -m morag_youtube.cli transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Configuration Options

```python
config = YouTubeConfig(
    # External service configuration
    service_url="http://localhost:8000",  # External service URL
    service_timeout=300,                  # Request timeout in seconds (5 minutes)

    # Video download options (opt-in only)
    download_video=False,                 # Whether to download video file
    output_dir=Path("./downloads"),       # Directory for downloaded files

    # Legacy options (kept for backward compatibility but not used)
    quality="best",
    extract_audio=True,
    download_subtitles=True,
    # ... other legacy options
)
```

### External Service Requirements

This package requires an external YouTube transcription service to be running. The service should:

- Provide a `/v1/youtube/transcribe` endpoint
- Accept POST requests with `{"url": "...", "download_video": true/false}`
- Return transcription results and metadata
- Support long-running requests (several minutes for transcription)
- Optionally provide video download with base64 encoding

See `YOUTUBE.md` for complete API specification.

## License

MIT