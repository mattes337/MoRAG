# Alpine Install Script Updates

## Overview

The `alpine-install.sh` script has been updated to reflect the changes that made the Alpine Dockerfile work successfully. These changes ensure better compatibility with Alpine Linux 3.21 and resolve dependency installation issues.

## Key Changes Made

### 1. Build Tools Simplification
- **Removed**: `linux-headers`, `musl-dev`, `unzip` from build tools
- **Added**: `autoconf`, `automake`, `libtool` for better package compilation
- **Result**: More streamlined build environment that matches working Docker setup

### 2. System Dependencies Overhaul
- **Replaced** generic system libraries with comprehensive image processing libraries
- **Added**: Complete set of image processing dependencies:
  - `jpeg-dev`, `libpng-dev`, `freetype-dev`, `lcms2-dev`
  - `openjpeg-dev`, `tiff-dev`, `tk-dev`, `tcl-dev`
  - `harfbuzz-dev`, `fribidi-dev`, `libimagequant-dev`
  - `libxcb-dev`, `libxml2-dev`, `libxslt-dev`
- **Result**: Better support for image processing and PDF handling

### 3. Media Dependencies Simplified
- **Removed**: Complex fallback logic for FFmpeg installation
- **Removed**: Audio processing libraries (libsndfile, portaudio, alsa)
- **Simplified**: Direct installation of `ffmpeg`, `ffmpeg-dev`, `imagemagick`, `imagemagick-dev`
- **Result**: Cleaner installation process with fewer potential failure points

### 4. OCR Dependencies Streamlined
- **Removed**: Complex package name detection logic
- **Simplified**: Direct installation of tesseract packages with multiple language support
- **Added**: `poppler-utils` and `poppler-dev` for PDF processing
- **Result**: More reliable OCR setup

### 5. Web Dependencies Simplified
- **Removed**: Fallback logic for Chromium installation
- **Simplified**: Direct installation of `chromium` and `chromium-chromedriver`
- **Result**: Consistent web scraping capabilities

### 6. Python Installation Modernized
- **Removed**: Complex package name detection and fallback logic
- **Simplified**: Direct installation of Python 3.11 with standard packages
- **Added**: Automatic symlink creation for `python` command
- **Result**: More reliable Python environment setup

### 7. Redis Installation Simplified
- **Removed**: Valkey/Redis detection logic
- **Simplified**: Direct Redis installation and service management
- **Result**: Consistent task queue setup

### 8. Python Build Dependencies Enhanced
- **Replaced**: Rust/Cargo dependencies with scientific computing packages
- **Added**: `cython`, `py3-numpy`, `blas-dev`, `lapack-dev`, `gfortran`
- **Result**: Better support for scientific Python packages

### 9. Dependency Installation Strategy Changed
- **Key Change**: Now excludes both `qdrant-client` AND `docling` from dependencies
- **Simplified**: Uses basic pip install without complex feature sets
- **Removed**: Individual feature set installations (docling, audio, image, office)
- **Removed**: Manual web scraping package installation
- **Removed**: CPU-only PyTorch and TensorFlow installation
- **Removed**: spaCy model downloads
- **Result**: Faster, more reliable installation with fewer potential conflicts

### 10. Alpine-Specific Issues Handling Removed
- **Removed**: `handle_alpine_issues()` function entirely
- **Removed**: Forced rebuilding of lxml and Pillow from source
- **Result**: Relies on system packages and standard pip installation

## Benefits of These Changes

1. **Faster Installation**: Fewer packages to compile and install
2. **More Reliable**: Eliminates complex fallback logic that could fail
3. **Better Compatibility**: Matches the proven working Docker setup
4. **Simpler Maintenance**: Fewer conditional installations to maintain
5. **Reduced Conflicts**: Excludes problematic packages (docling) that cause build issues

## What's Excluded

The updated script intentionally excludes some features to ensure reliability:

- **Docling**: PDF processing library that has compilation issues on Alpine
- **Advanced Audio Processing**: Complex audio libraries that may not be needed
- **Heavy ML Libraries**: PyTorch, TensorFlow (can be added manually if needed)
- **spaCy Models**: Large language models (can be downloaded separately)

## Usage

The script maintains the same usage pattern:

```bash
# Make executable
chmod +x alpine-install.sh

# Run (as root or with sudo)
./alpine-install.sh
```

## Post-Installation

After installation, users can manually add any excluded features they need:

```bash
# Activate virtual environment
source venv/bin/activate

# Install additional packages as needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu
python -m spacy download en_core_web_sm
```

## Testing

The updated script should be tested on a fresh Alpine Linux 3.21 system to ensure all changes work correctly in a native (non-Docker) environment.
