# MoRAG YouTube

YouTube video processing capabilities for the MoRAG (Multimodal RAG Ingestion Pipeline) system.

## Features

- **YouTube video downloading**
- **Metadata extraction**
- **Caption/subtitle extraction**
- **Thumbnail downloading**
- **Audio extraction**
- **Playlist processing**
- **Format selection and quality control**
- **Async processing for performance**

## Installation

```bash
pip install morag-youtube
```

## Usage

```python
import asyncio
from morag_youtube import YouTubeProcessor, YouTubeConfig

async def process_youtube_video():
    # Create processor
    processor = YouTubeProcessor()
    
    # Configure processing
    config = YouTubeConfig(
        quality="best",
        extract_audio=True,
        download_subtitles=True,
        subtitle_languages=["en"],
        download_thumbnails=True
    )
    
    # Process YouTube URL
    result = await processor.process_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ", config)
    
    if result.success:
        # Access metadata
        print(f"Title: {result.metadata.title}")
        print(f"Uploader: {result.metadata.uploader}")
        print(f"Duration: {result.metadata.duration} seconds")
        
        # Access downloaded files
        if result.video_path:
            print(f"Video downloaded to: {result.video_path}")
        if result.audio_path:
            print(f"Audio extracted to: {result.audio_path}")
        if result.subtitle_paths:
            print(f"Subtitles downloaded to: {', '.join(str(p) for p in result.subtitle_paths)}")
        if result.transcript_path:
            print(f"Transcript extracted to: {result.transcript_path}")
            print(f"Transcript language: {result.transcript_language}")

# Run the example
asyncio.run(process_youtube_video())
```

## Transcript Functionality

The package includes comprehensive YouTube transcript extraction with intelligent fallback strategy:

### Features

- **Intelligent Fallback Strategy**:
  - When cookies are available: Downloads video/audio and transcribes locally for higher accuracy
  - When cookies unavailable or download fails: Falls back to direct YouTube transcript API
- **Original Language Detection**: Automatically uses the video's original language when no language is specified
- **Multiple Format Support**: Export transcripts as text, SRT, or WebVTT formats
- **Language Fallback**: Intelligent language selection with fallback to available languages
- **Manual vs Auto-generated**: Prefers manually created transcripts over auto-generated ones
- **Cookie Support**: Supports YouTube cookies for accessing restricted content

### Usage Examples

#### Extract Transcript in Original Language

```python
from morag_youtube import YouTubeService

async def extract_transcript():
    service = YouTubeService()

    # Extract transcript in original language (MVP requirement)
    result = await service.extract_transcript(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )

    print(f"Language: {result['language']}")
    print(f"Transcript: {result['transcript_text'][:200]}...")
```

#### Check Available Languages

```python
async def check_languages():
    service = YouTubeService()

    languages = await service.get_available_transcript_languages(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )

    for lang_code, info in languages.items():
        status = "Manual" if not info['is_generated'] else "Auto-generated"
        print(f"{lang_code}: {info['language']} ({status})")
```

#### CLI Usage

```bash
# List available transcript languages
python -m morag_youtube.cli transcript-langs "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Extract transcript in original language (default behavior)
python -m morag_youtube.cli transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Extract transcript with cookies for better access (enables audio transcription fallback)
python -m morag_youtube.cli --cookies /path/to/cookies.txt transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Extract transcript using only transcript API (faster, no video download)
python -m morag_youtube.cli transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --transcript-only

# Extract transcript in specific language
python -m morag_youtube.cli transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --language en

# Export as SRT format
python -m morag_youtube.cli transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --format srt

# Environment variable for cookies (alternative to --cookies)
export YOUTUBE_COOKIES_FILE=/path/to/cookies.txt
python -m morag_youtube.cli transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Configuration Options

```python
config = YouTubeConfig(
    extract_transcript=True,           # Enable transcript extraction
    transcript_language=None,          # None = use original language (MVP)
    transcript_format="text",          # "text", "srt", "vtt"
    prefer_audio_transcription=True,   # Prefer audio transcription when cookies available
    cookies_file="/path/to/cookies.txt", # Path to cookies file (overrides env var)
    transcript_only=False              # True = skip video download, use only transcript API
)
```

### Transcript Extraction Modes

The system provides three modes for transcript extraction:

1. **Full Processing Mode (default)**:
   - Downloads video/audio when cookies available
   - Transcribes locally using Whisper for higher accuracy
   - Falls back to YouTube Transcript API if download fails
   - Best accuracy but slower

2. **Transcript-Only Mode (`--transcript-only` or `transcript_only=True`)**:
   - Skips video download entirely
   - Uses only YouTube Transcript API
   - Faster but limited to videos with existing transcripts
   - Ideal for quick transcript extraction

3. **Cookie-Enhanced Mode (with `--cookies` or `cookies_file`)**:
   - Downloads video/audio using yt-dlp with cookies
   - Supports restricted/private videos accessible with cookies
   - Transcribes locally using Whisper
   - Falls back to transcript API if needed

### Fallback Strategy

The system uses an intelligent fallback strategy:
- **Primary**: Local transcription (when cookies available and not transcript-only)
- **Fallback**: Direct YouTube Transcript API
- **Final**: Comprehensive error reporting if all methods fail

This ensures maximum compatibility and accuracy while providing user control over processing speed vs. accuracy.

## License

MIT