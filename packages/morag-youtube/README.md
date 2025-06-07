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

# Run the example
asyncio.run(process_youtube_video())
```

## License

MIT