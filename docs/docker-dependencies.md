# Docker System Dependencies

This document outlines all system-level dependencies required for the MoRAG application to function properly in Docker containers.

## Overview

The MoRAG application requires various system libraries and tools to support its comprehensive document processing capabilities including:

- Audio/Video processing with FFmpeg
- OCR (Optical Character Recognition) with Tesseract
- Image processing with OpenCV
- Web scraping with Playwright browsers
- Audio processing with librosa/soundfile
- Various file format support

## System Dependencies

### Core Dependencies
- **curl** - HTTP client for health checks and API calls
- **git** - Version control system for packages that install from repositories
- **build-essential** - Compilation tools (builder stage only)

### Audio/Video Processing
- **ffmpeg** - Audio and video processing toolkit
  - Required for: Audio extraction from video, format conversion, keyframe extraction
  - Used by: pydub, moviepy, ffmpeg-python packages
- **libsndfile1** - Audio file format support
  - Required for: librosa, soundfile packages
  - Provides: WAV, FLAC, OGG, and other audio format support
- **libgomp1** - OpenMP support for parallel processing
  - Required for: Audio processing libraries that use parallel computation

### OCR (Optical Character Recognition)
- **tesseract-ocr** - OCR engine
  - Required for: pytesseract package
  - Provides: Text extraction from images
- **tesseract-ocr-eng** - English language data for Tesseract
  - Required for: English text recognition
  - Additional language packs can be added as needed

### Image Processing (OpenCV)
- **libgl1-mesa-glx** - OpenGL support
- **libglib2.0-0** - GLib library
- **libsm6** - Session Management library
- **libxext6** - X11 extension library
- **libxrender-dev** - X Rendering Extension library
  - All required for: opencv-python package
  - Provides: Image processing, computer vision capabilities

### Web Scraping (Playwright Browsers)
- **libnss3** - Network Security Services
- **libnspr4** - Netscape Portable Runtime
- **libatk1.0-0** - Accessibility Toolkit
- **libatk-bridge2.0-0** - ATK bridge library
- **libcups2** - Common UNIX Printing System
- **libdrm2** - Direct Rendering Manager
- **libxss1** - X11 Screen Saver extension
- **libgtk-3-0** - GTK+ 3.0 library
- **libxrandr2** - X11 RandR extension
- **libasound2** - ALSA sound library
- **libpangocairo-1.0-0** - Pango Cairo rendering
- **libcairo-gobject2** - Cairo GObject bindings
- **libgdk-pixbuf2.0-0** - GDK Pixbuf library
  - All required for: Playwright Chromium browser
  - Provides: Dynamic web content extraction, JavaScript rendering

## Installation Commands

### Main Dockerfile (Production)
```dockerfile
# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    git \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxss1 \
    libgtk-3-0 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright browsers (optional, for web scraping with dynamic content)
RUN python -m playwright install chromium || echo "Playwright browser installation failed, will use fallback"
```

### Worker Dockerfile
```dockerfile
# Install system dependencies (includes build-essential for compilation)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    git \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxss1 \
    libgtk-3-0 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright browsers (optional, for web scraping with dynamic content)
RUN python -m playwright install chromium || echo "Playwright browser installation failed, will use fallback"
```

## Fallback Mechanisms

The application is designed with fallback mechanisms for optional dependencies:

1. **Playwright**: Falls back to basic HTTP scraping if browser installation fails
2. **Tesseract**: Falls back to EasyOCR if Tesseract is not available
3. **FFmpeg**: Falls back to librosa/soundfile for audio processing
4. **spaCy models**: Falls back to basic text processing if model download fails

## Troubleshooting

### Common Issues

1. **Playwright browser installation fails**
   - The application will fall back to basic web scraping
   - Check if all browser dependencies are installed
   - Verify network connectivity during build

2. **OpenCV import errors**
   - Ensure all libgl/libglib dependencies are installed
   - Check for missing X11 libraries

3. **Audio processing errors**
   - Verify ffmpeg and libsndfile1 are installed
   - Check audio file format support

4. **OCR not working**
   - Ensure tesseract-ocr and language packs are installed
   - Verify image file accessibility

## Security Considerations

- All packages are installed from official Debian repositories
- Package lists are cleaned after installation to reduce image size
- Non-root user is used for running the application
- Optional dependencies fail gracefully without breaking the application

## Image Size Optimization

- Multi-stage build separates build dependencies from runtime
- Package cache is cleaned after installation
- Only necessary language packs are installed for Tesseract
- Chromium browser is the only Playwright browser installed (not all browsers)
