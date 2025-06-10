# MoRAG Remote Converter

A standalone remote worker application that polls the MoRAG API for pending remote jobs, processes them using existing MoRAG conversion components, and submits results back to the API.

## Features

- **Horizontal Scaling**: Distribute processing across multiple machines
- **GPU Support**: Leverage remote GPUs for faster audio/video processing
- **Automatic Job Polling**: Continuously polls for available jobs
- **Concurrent Processing**: Handle multiple jobs simultaneously
- **Robust Error Handling**: Comprehensive error handling and retry logic
- **Secure File Transfer**: Safe file download and result submission
- **Flexible Configuration**: Environment variables and YAML configuration
- **Health Monitoring**: Built-in connection testing and health checks

## Supported Content Types

- **Audio**: MP3, WAV, M4A, FLAC, OGG
- **Video**: MP4, AVI, MKV, MOV, WebM
- **Document**: PDF, DOCX, PPTX, XLSX, TXT, MD (optional)
- **Image**: JPG, PNG, GIF, BMP, WebP, TIFF, SVG (optional)
- **Web**: HTML pages, web scraping (optional)
- **YouTube**: Video downloads and processing (optional)

## Quick Start

### 1. Installation

```bash
# Navigate to the remote converter directory
cd tools/remote-converter

# Run the installation script (Linux/macOS)
./install.sh

# Or install manually
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip

# Install MoRAG packages
pip install -e ../../packages/morag-core
pip install -e ../../packages/morag-audio
pip install -e ../../packages/morag-video

# Install additional dependencies
pip install requests pyyaml python-dotenv structlog
```

### 2. Configuration

Create a configuration file:

```bash
# Create sample configuration
python cli.py --create-config

# Copy and edit the configuration
cp remote_converter_config.yaml.example remote_converter_config.yaml
# Edit remote_converter_config.yaml with your settings
```

Example configuration:

```yaml
worker_id: "gpu-worker-01"
api_base_url: "https://your-morag-server.com"
api_key: "your-api-key-here"  # Optional
content_types: ["audio", "video"]
poll_interval: 10
max_concurrent_jobs: 2
log_level: "INFO"
temp_dir: "/tmp/morag_remote"
```

### 3. Model Configuration (Optional)

The remote converter supports overriding AI models through environment variables:

#### Audio Models

**Whisper Model Override:**
```bash
# Use large-v3 model for better accuracy (requires more GPU memory)
WHISPER_MODEL_SIZE=large-v3

# Alternative variable name (both are supported)
MORAG_WHISPER_MODEL_SIZE=large-v3
```

Available Whisper models:
- `tiny` - Fastest, least accurate (~39 MB)
- `base` - Good balance (~74 MB)
- `small` - Better accuracy (~244 MB)
- `medium` - High accuracy (~769 MB) - **Default**
- `large-v2` - Highest accuracy (~1550 MB)
- `large-v3` - Latest, highest accuracy (~1550 MB) - **Recommended for GPU workers**

**SpaCy Model Override:**
```bash
# Use German model for topic segmentation
MORAG_SPACY_MODEL=de_core_news_sm

# Use larger English model
MORAG_SPACY_MODEL=en_core_web_md
```

Common SpaCy models:
- `en_core_web_sm` - English small model - **Default**
- `en_core_web_md` - English medium model
- `de_core_news_sm` - German small model
- `de_core_news_md` - German medium model
- `fr_core_news_sm` - French small model

#### Audio Processing Features

```bash
# Enable/disable speaker diarization (who spoke when)
MORAG_ENABLE_SPEAKER_DIARIZATION=true

# Enable/disable topic segmentation (automatic topic detection)
MORAG_ENABLE_TOPIC_SEGMENTATION=true

# Force specific language (optional, auto-detect if not set)
MORAG_AUDIO_LANGUAGE=de

# Force specific device
MORAG_AUDIO_DEVICE=cuda  # auto, cpu, cuda
```

### 4. Environment Variables (Optional)

```bash
# Copy environment template
cp .env.example .env
# Edit .env with your settings
```

Environment variables:

```bash
# Worker Configuration
MORAG_WORKER_ID=gpu-worker-01
MORAG_API_BASE_URL=https://your-morag-server.com
MORAG_WORKER_CONTENT_TYPES=audio,video
MORAG_WORKER_POLL_INTERVAL=10
MORAG_WORKER_MAX_CONCURRENT_JOBS=2
MORAG_API_KEY=your-api-key-here
MORAG_LOG_LEVEL=INFO
MORAG_TEMP_DIR=/tmp/morag_remote

# Audio Processing Configuration
WHISPER_MODEL_SIZE=large-v3                    # Whisper model: tiny, base, small, medium, large-v2, large-v3
MORAG_AUDIO_LANGUAGE=en                        # Audio language (optional, auto-detect if not set)
MORAG_AUDIO_DEVICE=auto                        # Device: auto, cpu, cuda
MORAG_ENABLE_SPEAKER_DIARIZATION=true          # Enable speaker diarization
MORAG_ENABLE_TOPIC_SEGMENTATION=true           # Enable topic segmentation
MORAG_SPACY_MODEL=de_core_news_sm              # SpaCy model for topic segmentation
```

### 5. Test Connection

```bash
# Test connection to MoRAG API
python cli.py --test-connection

# Show current configuration
python cli.py --show-config
```

### 6. Start the Remote Converter

```bash
# Start with configuration file
python cli.py --config remote_converter_config.yaml

# Start with command line options
python cli.py \
    --worker-id gpu-worker-01 \
    --api-url https://your-morag-server.com \
    --content-types audio,video \
    --max-jobs 2

# Start with startup script (after installation)
./start.sh
```

## Command Line Options

```bash
python cli.py [OPTIONS]

Options:
  --config, -c PATH           Configuration file path
  --worker-id TEXT           Unique worker identifier
  --api-url TEXT             MoRAG API base URL
  --api-key TEXT             API authentication key
  --content-types TEXT       Comma-separated list of content types
  --poll-interval INTEGER    Polling interval in seconds
  --max-jobs INTEGER         Maximum concurrent jobs
  --log-level LEVEL          Log level (DEBUG, INFO, WARNING, ERROR)
  --temp-dir PATH            Temporary directory for file processing
  --create-config            Create sample configuration file
  --test-connection          Test API connection and exit
  --show-config              Show current configuration and exit
  --help                     Show help message
```

## System Service Installation

### Linux (systemd)

```bash
# Copy service file
sudo cp morag-remote-converter.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable morag-remote-converter
sudo systemctl start morag-remote-converter

# Monitor service
sudo systemctl status morag-remote-converter
sudo journalctl -u morag-remote-converter -f
```

### Windows

Create a Windows service using tools like NSSM or run as a scheduled task.

## Monitoring and Troubleshooting

### Check Worker Status

```bash
# Test API connection
python cli.py --test-connection

# Show configuration
python cli.py --show-config

# Check logs (when running as service)
sudo journalctl -u morag-remote-converter -f
```

### Common Issues

#### Connection Failed
- Check API URL and network connectivity
- Verify API key if authentication is required
- Ensure MoRAG server is running and accessible

#### No Jobs Available
- Verify content types match server requirements
- Check if other workers are already processing jobs
- Ensure remote processing is enabled on the server

#### Processing Failures
- Check temp directory permissions
- Verify MoRAG packages are properly installed
- Review worker logs for detailed error messages

#### Import Errors
```bash
# Reinstall MoRAG packages
pip install -e ../../packages/morag-core
pip install -e ../../packages/morag-audio
pip install -e ../../packages/morag-video
```

#### Wrong Whisper Model Used for Video Processing
If you see the remote worker initializing with the correct model (e.g., `large-v3`) but then using a different model (e.g., `base`) when processing video files, this indicates the environment variables are properly set. The VideoProcessor now correctly inherits the Whisper model configuration from environment variables.

**Verification**: Check the logs - you should see:
```
{"model_size": "large-v3", "event": "Initializing Whisper model", "logger": "morag_audio.processor"}
```
Both at startup and during video processing.

## Architecture

The remote converter consists of:

1. **Main Loop**: Continuously polls for jobs and manages processing
2. **Job Processor**: Downloads files, processes them, and submits results
3. **Configuration Manager**: Handles configuration loading and validation
4. **Connection Manager**: Manages API communication and file transfers
5. **Error Handler**: Comprehensive error handling and recovery

## Security Considerations

- **API Authentication**: Use API keys for secure communication
- **File Isolation**: Temporary files are isolated per job
- **Network Security**: All communication over HTTPS recommended
- **Resource Limits**: Configurable limits on concurrent jobs and temp storage

## Performance Tuning

- **Concurrent Jobs**: Adjust `max_concurrent_jobs` based on hardware
- **Poll Interval**: Lower values for faster job pickup, higher for less load
- **Temp Directory**: Use fast storage (SSD) for temporary files
- **Network**: Ensure good bandwidth for file transfers

## Development

### Running Tests

```bash
# Test the remote converter implementation
python -m pytest tests/ -v

# Test with sample jobs
python cli.py --test-connection
```

### Adding New Content Types

1. Install the corresponding MoRAG package
2. Add the content type to your configuration
3. Ensure the processor is available in the remote converter

For more information, see the main [MoRAG documentation](../../README.md).
