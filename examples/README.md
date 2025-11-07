# MoRAG Examples

This directory contains comprehensive examples demonstrating the MoRAG (Multi-modal Retrieval Augmented Generation) system capabilities. Each example showcases different aspects of the system and provides practical usage patterns.

## 📋 Available Examples

### 1. API Demo (`api_demo.py`)
**Purpose**: Demonstrates the complete MoRAG ingestion API functionality
**What it shows**:
- Health check endpoints
- File upload and processing
- URL ingestion for web content
- Batch processing capabilities
- Task status monitoring
- Queue statistics

**Prerequisites**:
- MoRAG API server running on `http://localhost:8000`
- Valid API key (set in the script)
- Redis and Qdrant services running

**Usage**:
```bash
cd examples
python api_demo.py
```

**Expected Output**:
- Health status confirmation
- Task IDs for submitted jobs
- Progress tracking information
- Queue statistics and active task counts

---

### 2. Universal Converter Demo (`universal_converter_demo.py`)
**Purpose**: Showcases the universal document conversion framework
**What it shows**:
- Supported document formats
- Conversion options and configuration
- Quality assessment features
- Format-specific processing options
- Chunking strategies (page, sentence, semantic)

**Prerequisites**:
- MoRAG system installed with converter dependencies
- Sample documents (optional - creates test content)

**Usage**:
```bash
cd examples
python universal_converter_demo.py
```

**Expected Output**:
- List of supported formats (PDF, DOCX, audio, video, web)
- Conversion configuration examples
- Quality assessment scores
- Format-specific options demonstration

---

### 3. Page-Based Chunking Demo (`page_based_chunking_demo.py`)
**Purpose**: Demonstrates page-level document chunking strategy
**What it shows**:
- Page-based vs sentence-based chunking comparison
- Vector storage optimization
- Metadata preservation during chunking
- Performance benefits of page-level chunking

**Prerequisites**:
- MoRAG system with document processing capabilities
- Sample PDF or multi-page document

**Usage**:
```bash
cd examples
python page_based_chunking_demo.py
```

**Expected Output**:
- Chunking strategy comparisons
- Vector count differences
- Processing time comparisons
- Metadata structure examples

---

### 4. Web Scraping Demo (`web_scraping_demo.py`)
**Purpose**: Shows web content extraction and processing capabilities
**What it shows**:
- Website content extraction
- HTML to Markdown conversion
- Dynamic content handling with Playwright
- Metadata extraction from web pages
- Content cleaning and formatting

**Prerequisites**:
- MoRAG system with web processing dependencies
- Internet connection for live web scraping
- Playwright browser dependencies (for dynamic content)

**Usage**:
```bash
cd examples
python web_scraping_demo.py
```

**Expected Output**:
- Extracted web content in Markdown format
- Page metadata (title, description, etc.)
- Content structure preservation
- Dynamic content extraction results

---

### 5. Enhanced Audio Processing Demo (`enhanced_audio_processing_demo.py`)
**Purpose**: Demonstrates advanced audio processing with speaker diarization and topic segmentation
**What it shows**:
- Basic audio transcription with Whisper
- Speaker diarization and identification
- Topic segmentation and analysis
- Enhanced markdown conversion with conversational format
- Performance metrics and quality assessment

**Prerequisites**:
- MoRAG system with audio processing dependencies
- Audio file for testing (MP3, WAV, M4A)
- Optional: pyannote.audio for advanced speaker diarization

**Usage**:
```bash
cd examples
python enhanced_audio_processing_demo.py /path/to/audio.mp3
python enhanced_audio_processing_demo.py /path/to/audio.mp3 --basic-only
python enhanced_audio_processing_demo.py /path/to/audio.mp3 --output-json results.json
```

**Expected Output**:
- Speaker identification and timeline
- Topic segmentation with summaries
- Conversational format markdown
- Processing performance metrics

---

### 6. Video Audio Integration Demo (`video_audio_integration_demo.py`)
**Purpose**: Shows automatic audio processing integration for video files
**What it shows**:
- Video audio extraction and processing
- Enhanced audio processing pipeline integration
- Conversational format markdown with timestamps
- Speaker-aware topic boundaries
- Complete video-to-markdown conversion

**Prerequisites**:
- MoRAG system with video and audio processing dependencies
- Video file for testing (MP4, AVI, MOV)
- FFmpeg for video processing

**Usage**:
```bash
cd examples
python video_audio_integration_demo.py /path/to/video.mp4
```

**Expected Output**:
- Video metadata and processing info
- Audio extraction and transcription
- Topic headers with timestamps
- Speaker dialogue format
- Enhanced markdown output

---

### 7. Universal Conversion Demo (Alternative) (`universal_conversion_demo_alt.py`)
**Purpose**: Alternative demonstration of universal document conversion
**What it shows**:
- Format detection and converter selection
- Quality assessment and validation
- Different conversion strategies
- Fallback mechanisms

**Prerequisites**:
- MoRAG system with conversion dependencies
- Sample documents of various formats

**Usage**:
```bash
cd examples
python universal_conversion_demo_alt.py
```

**Expected Output**:
- Supported format listing
- Conversion examples
- Quality metrics demonstration

---

### 8. Transcription Fixes Demo (Alternative) (`transcription_fixes_demo_alt.py`)
**Purpose**: Demonstrates specific transcription quality fixes and improvements
**What it shows**:
- Audio transcription quality improvements
- Speaker diarization fixes
- Topic timestamp formatting
- German language support enhancements

**Prerequisites**:
- MoRAG system with enhanced audio processing
- Audio files with multiple speakers (optional)

**Usage**:
```bash
cd examples
python transcription_fixes_demo_alt.py
```

**Expected Output**:
- Before/after transcription quality comparison
- Speaker identification improvements
- Timestamp format examples

## 🚀 Getting Started

### Quick Start
1. **Ensure MoRAG is running**:
   ```bash
   # Start the debug session (includes all services)
   ./scripts/debug-session.ps1

   # Or manually start services
   docker-compose -f docker/docker-compose.redis.yml up -d
   docker-compose -f docker/docker-compose.qdrant.yml up -d
   python scripts/init_db.py
   celery worker -A morag.core.celery_app:app --loglevel=info &
   uvicorn morag.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Set up environment variables**:
   ```bash
   # Create .env file with required settings
   GEMINI_API_KEY=your_gemini_api_key_here
   API_HOST=0.0.0.0
   API_PORT=8000
   REDIS_URL=redis://localhost:6379/0
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```

3. **Install dependencies**:
   ```bash
   pip install -e ".[dev,audio,video,image,docling]"
   ```

4. **Run any example**:
   ```bash
   cd examples
   python <example_name>.py
   ```

### Example Workflow
For a complete demonstration of MoRAG capabilities:

1. **Start with API Demo** to verify system health
2. **Run Universal Converter Demo** to understand format support
3. **Try Page-Based Chunking Demo** to see optimization benefits
4. **Explore Web Scraping Demo** for web content processing

## 📚 Understanding the Examples

### API Integration Patterns
The examples demonstrate several key integration patterns:

- **Synchronous API calls** for immediate results
- **Asynchronous task submission** for long-running processes
- **Status polling** for progress tracking
- **Batch processing** for multiple items
- **Error handling** and retry mechanisms

### Configuration Examples
Each example shows different configuration approaches:

- **Default configurations** for quick start
- **Custom configurations** for specific use cases
- **Format-specific options** for optimal processing
- **Quality thresholds** for conversion validation

### Output Formats
Examples demonstrate various output formats:

- **Structured Markdown** with metadata preservation
- **JSON responses** with detailed processing information
- **Conversational formats** for audio/video transcripts
- **Topic-based organization** for long-form content

## 🔧 Customization

### Modifying Examples
Each example is designed to be easily customizable:

1. **Change input sources**: Update URLs, file paths, or content
2. **Adjust processing options**: Modify conversion settings
3. **Add custom metadata**: Include additional context information
4. **Integrate with your workflow**: Use examples as templates

### Adding New Examples
To create new examples:

1. Follow the existing structure and naming conventions
2. Include comprehensive documentation and comments
3. Add error handling and user-friendly output
4. Update this README with the new example description

## 🐛 Troubleshooting

### Common Issues

1. **Connection Errors**:
   - Ensure MoRAG API server is running
   - Check Redis and Qdrant services are accessible
   - Verify network connectivity

2. **Authentication Errors**:
   - Check API key configuration
   - Verify environment variables are set
   - Ensure API key has proper permissions

3. **Processing Errors**:
   - Check input file formats are supported
   - Verify file sizes are within limits
   - Ensure required dependencies are installed

4. **Import Errors**:
   - Run examples from the examples/ directory
   - Ensure MoRAG is properly installed
   - Check Python path configuration

### Getting Help

- **Check logs**: Look in `logs/` directory for detailed error information
- **API documentation**: Visit `http://localhost:8000/docs` for interactive API docs
- **Test endpoints**: Use the health check endpoint to verify system status
- **Review tasks**: Check `TASKS.md` for implementation status and known issues

## 📈 Next Steps

After exploring these examples:

1. **Integrate with your application**: Use the patterns shown in your own code
2. **Explore advanced features**: Check the full API documentation
3. **Contribute improvements**: Submit enhancements or new examples
4. **Scale your usage**: Consider batch processing and optimization strategies

For more detailed information about specific components, see:
- `docs/api_usage.md` - Complete API reference
- `docs/UNIVERSAL_DOCUMENT_CONVERSION.md` - Conversion framework details
- `scripts/README.md` - Development and debugging tools
- `TASKS.md` - Implementation progress and features
