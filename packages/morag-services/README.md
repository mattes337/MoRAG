# MoRAG Services

The `morag-services` package provides a unified service layer for the MoRAG (Modular Retrieval Augmented Generation) system. It integrates all specialized processing packages into a cohesive API, making it easy to work with multiple content types through a single interface.

## Features

- **Unified API**: Access all MoRAG capabilities through a consistent interface
- **Service Registry**: Dynamically register and discover available services
- **Content Pipeline**: Process content through multiple services in sequence
- **Automatic Content Routing**: Route content to appropriate processors based on type
- **Parallel Processing**: Process multiple content items concurrently
- **Error Handling**: Consistent error handling across all services
- **Configuration Management**: Centralized configuration for all services

## Installation

```bash
pip install morag-services
```

## Usage

### Basic Usage

```python
import asyncio
from morag_services import MoRAGServices

async def main():
    # Initialize the services with default configuration
    services = MoRAGServices()

    # Process a document
    document_result = await services.process_document("path/to/document.pdf")

    # Process a web page
    web_result = await services.process_url("https://example.com")

    # Process an image
    image_result = await services.process_image("path/to/image.jpg")

    # Process multiple items concurrently
    results = await services.process_batch([
        "path/to/document.pdf",
        "https://example.com",
        "path/to/image.jpg",
        "path/to/audio.mp3",
        "https://www.youtube.com/watch?v=example"
    ])

    # Access results
    for item, result in results.items():
        print(f"Processed {item}: {result.success}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Configuration

```python
from morag_services import MoRAGServices, ServiceConfig
from morag_core.models import ProcessingConfig

# Configure services with custom settings
config = ServiceConfig(
    document_config=ProcessingConfig(chunk_size=500),
    web_config=ProcessingConfig(max_depth=2),
    image_config=ProcessingConfig(extract_text=True),
    audio_config=ProcessingConfig(language="en"),
    video_config=ProcessingConfig(extract_frames=True),
    youtube_config=ProcessingConfig(download_subtitles=True),
    embedding_config=ProcessingConfig(model="text-embedding-ada-002"),
    max_concurrent_tasks=5
)

# Initialize services with custom configuration
services = MoRAGServices(config=config)
```

### Content Pipeline

```python
from morag_services import MoRAGServices, Pipeline

# Create a processing pipeline
pipeline = Pipeline()

# Add processing steps
pipeline.add_step("extract", lambda content: services.extract_text(content))
pipeline.add_step("chunk", lambda text: services.chunk_text(text))
pipeline.add_step("embed", lambda chunks: services.embed_text(chunks))

# Process content through the pipeline
result = await pipeline.process("path/to/document.pdf")
```

## Available Services

The `morag-services` package integrates the following specialized services:

- **Document Processing**: Extract text and metadata from documents (PDF, DOCX, etc.)
- **Web Processing**: Scrape and extract content from web pages
- **Image Processing**: Analyze and extract information from images
- **Audio Processing**: Transcribe and analyze audio files
- **Video Processing**: Extract frames, audio, and metadata from videos
- **YouTube Processing**: Download and extract content from YouTube videos
- **Embedding Generation**: Generate vector embeddings for content

## License

MIT
