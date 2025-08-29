# Mock Data Directory

This directory contains pre-generated mock outputs for all MoRAG stages when mock mode is enabled.

## Structure

Each stage has its own subdirectory containing sample outputs for different content types:

- `markdown-conversion/` - Sample markdown conversion outputs
- `markdown-optimizer/` - Sample optimized markdown outputs
- `chunker/` - Sample chunking and summary outputs
- `fact-generator/` - Sample fact extraction outputs
- `ingestor/` - Sample ingestion results

## Content Types Supported

Mock data is provided for multiple content types to ensure comprehensive testing:

### Document Processing (PDF, Word, etc.)
- `document.md` - Business analysis document
- `document.optimized.md` - Optimized version
- `document.chunks.json` - Chunked content
- `document.facts.json` - Extracted entities and relations
- `document.ingestion.json` - Database ingestion results

### Video Processing (MP4, AVI, etc.)
- `video.md` - Product demo transcript
- `video.optimized.md` - Enhanced transcript
- `video.chunks.json` - Temporal-semantic chunks
- `video.facts.json` - Speaker and content analysis
- `video.ingestion.json` - Multimedia ingestion results

### Audio Processing (MP3, WAV, etc.)
- `audio.md` - Podcast interview transcript
- `audio.optimized.md` - Cleaned transcript
- `audio.chunks.json` - Speaker-segmented chunks
- `audio.facts.json` - Speaker identification and topics
- `audio.ingestion.json` - Audio content ingestion

### Web Content Processing
- `web.md` - Article extraction
- `web.optimized.md` - Structured web content
- `web.chunks.json` - Article sections
- `web.facts.json` - Web content entities
- `web.ingestion.json` - Web data ingestion

### YouTube Processing
- `youtube.md` - Tutorial transcript
- `youtube.optimized.md` - Enhanced tutorial content
- `youtube.chunks.json` - Educational content chunks
- `youtube.facts.json` - Tutorial concepts and entities
- `youtube.ingestion.json` - YouTube content ingestion

## Usage

Set `MORAG_MOCK_MODE=true` in your .env file to enable mock mode. When enabled, all stages will return the pre-generated outputs from this directory instead of performing actual processing.

## File Naming Convention

Mock files follow the pattern: `{content_type}.{stage_output_extension}`

The system automatically selects the appropriate mock file based on the detected or specified content type.

## Mock Mode Benefits

- Fast testing and development
- Consistent outputs for testing
- No dependency on external services (LLMs, databases)
- Predictable behavior for CI/CD pipelines
