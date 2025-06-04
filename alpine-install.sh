#!/bin/bash

# MoRAG Alpine Linux Installation Script (CPU Only, No Qdrant)
# This script installs MoRAG on Alpine Linux without Qdrant vector database

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root. Please run as a regular user with sudo privileges."
        exit 1
    fi
}

# Check Alpine Linux version
check_alpine_version() {
    if ! grep -q "Alpine Linux" /etc/os-release; then
        log_error "This script is designed for Alpine Linux only."
        exit 1
    fi
    
    local version=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
    log_info "Detected Alpine Linux version: $version"
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    sudo apk update
    sudo apk upgrade
    log_success "System packages updated"
}

# Install essential build tools
install_build_tools() {
    log_info "Installing essential build tools..."
    sudo apk add --no-cache \
        build-base \
        linux-headers \
        musl-dev \
        gcc \
        g++ \
        make \
        cmake \
        pkgconfig \
        git \
        curl \
        wget \
        unzip
    log_success "Build tools installed"
}

# Install system dependencies
install_system_dependencies() {
    log_info "Installing core system libraries..."
    sudo apk add --no-cache \
        glib-dev \
        cairo-dev \
        pango-dev \
        gdk-pixbuf-dev \
        libffi-dev \
        openssl-dev \
        zlib-dev \
        bzip2-dev \
        xz-dev \
        sqlite-dev \
        readline-dev
    log_success "Core system libraries installed"
}

# Install media processing dependencies
install_media_dependencies() {
    log_info "Installing media processing dependencies..."
    sudo apk add --no-cache \
        ffmpeg \
        ffmpeg-dev \
        libsndfile \
        libsndfile-dev \
        portaudio \
        portaudio-dev \
        alsa-lib \
        alsa-lib-dev
    log_success "Media processing dependencies installed"
}

# Install OCR and image processing
install_ocr_dependencies() {
    log_info "Installing OCR and image processing dependencies..."
    sudo apk add --no-cache \
        tesseract-ocr \
        tesseract-ocr-data-eng \
        tesseract-ocr-data-deu \
        tesseract-ocr-dev \
        jpeg-dev \
        tiff-dev \
        freetype-dev \
        lcms2-dev \
        openjpeg-dev \
        libwebp-dev \
        zlib-dev
    log_success "OCR and image processing dependencies installed"
}

# Install web scraping dependencies
install_web_dependencies() {
    log_info "Installing web scraping dependencies..."
    sudo apk add --no-cache \
        libxml2-dev \
        libxslt-dev \
        chromium \
        chromium-chromedriver
    log_success "Web scraping dependencies installed"
}

# Install Python and development tools
install_python() {
    log_info "Installing Python and development tools..."
    sudo apk add --no-cache \
        python3 \
        python3-dev \
        py3-pip \
        py3-virtualenv \
        py3-wheel \
        py3-setuptools \
        py3-numpy \
        py3-scipy \
        py3-pillow \
        py3-lxml \
        py3-cryptography \
        py3-cffi
    
    # Verify Python version
    local python_version=$(python3 --version)
    log_info "Python version: $python_version"
    log_success "Python and development tools installed"
}

# Install and configure Redis
install_redis() {
    log_info "Installing and configuring Redis..."
    sudo apk add --no-cache redis
    
    # Enable Redis service
    sudo rc-update add redis default
    
    # Start Redis
    sudo service redis start
    
    # Test Redis connection
    if redis-cli ping > /dev/null 2>&1; then
        log_success "Redis installed and running"
    else
        log_warning "Redis installed but not responding to ping"
    fi
}

# Install additional build dependencies for Python packages
install_python_build_deps() {
    log_info "Installing additional Python build dependencies..."
    sudo apk add --no-cache \
        rust \
        cargo \
        libffi-dev \
        openssl-dev
    
    # Set environment variable for Rust packages
    export CARGO_NET_GIT_FETCH_WITH_CLI=true
    log_success "Python build dependencies installed"
}

# Clone MoRAG repository
clone_repository() {
    log_info "Cloning MoRAG repository..."
    
    if [ -d "morag" ]; then
        log_warning "MoRAG directory already exists. Skipping clone."
        cd morag
    else
        # Note: Update this URL to the actual repository
        git clone https://github.com/yourusername/morag.git
        cd morag
        log_success "Repository cloned"
    fi
}

# Create Python virtual environment
create_virtualenv() {
    log_info "Creating Python virtual environment..."
    
    if [ -d "venv" ]; then
        log_warning "Virtual environment already exists. Skipping creation."
    else
        python3 -m venv venv
        log_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools
    log_success "Virtual environment activated and pip upgraded"
}

# Install MoRAG dependencies (excluding Qdrant)
install_morag_dependencies() {
    log_info "Installing MoRAG dependencies (CPU-only, no Qdrant)..."

    # Ensure virtual environment is activated
    source venv/bin/activate

    # Create a temporary requirements file without qdrant-client
    log_info "Creating temporary requirements without Qdrant..."
    python -c "
import re
with open('pyproject.toml', 'r') as f:
    content = f.read()

# Extract dependencies, excluding qdrant-client
deps = re.findall(r'\"([^\"]+)\"', content.split('dependencies = [')[1].split(']')[0])
filtered_deps = [dep for dep in deps if not dep.startswith('qdrant-client')]

with open('requirements_no_qdrant.txt', 'w') as f:
    for dep in filtered_deps:
        f.write(dep + '\n')
"

    # Install base dependencies without qdrant
    log_info "Installing base dependencies (excluding Qdrant)..."
    pip install -r requirements_no_qdrant.txt

    # Install feature sets individually
    log_info "Installing docling for PDF processing..."
    pip install -e ".[docling]"

    log_info "Installing audio processing dependencies..."
    pip install -e ".[audio]"

    log_info "Installing image processing dependencies..."
    pip install -e ".[image]"

    log_info "Installing office document dependencies..."
    pip install -e ".[office]"

    # Install web dependencies manually (excluding playwright for Alpine compatibility)
    log_info "Installing web scraping dependencies..."
    pip install beautifulsoup4 markdownify html2text lxml bleach trafilatura readability-lxml newspaper3k

    # Note: No vector database packages needed - using external Qdrant server

    # Force CPU-only versions for key packages
    log_info "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Install TensorFlow CPU version
    log_info "Installing TensorFlow CPU..."
    pip install tensorflow-cpu

    # Download spaCy language models
    log_info "Downloading spaCy language models..."
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm

    # Clean up temporary file
    rm -f requirements_no_qdrant.txt

    log_success "MoRAG dependencies installed"
}

# Handle Alpine-specific package issues
handle_alpine_issues() {
    log_info "Handling Alpine-specific package compatibility..."
    
    # Ensure virtual environment is activated
    source venv/bin/activate
    
    # Install packages that commonly have issues on Alpine
    log_info "Rebuilding lxml from source..."
    pip install --no-binary=:all: lxml
    
    log_info "Rebuilding Pillow from source..."
    pip install --no-binary=:all: pillow
    
    log_success "Alpine-specific issues handled"
}

# Create environment configuration
create_environment_config() {
    log_info "Creating environment configuration..."
    
    if [ -f ".env" ]; then
        log_warning ".env file already exists. Creating .env.alpine as backup."
        cp .env.example .env.alpine
    else
        cp .env.example .env
    fi
    
    # Update .env for Alpine Linux (external Qdrant server)
    cat >> .env << 'EOF'

# Alpine Linux specific settings
PREFERRED_DEVICE=cpu
FORCE_CPU=true

# External Qdrant server configuration
# IMPORTANT: Update these values with your actual Qdrant server details
QDRANT_HOST=your_qdrant_server_ip_here
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=morag_documents
QDRANT_API_KEY=your_qdrant_api_key_if_needed

# Web scraping settings (Alpine compatibility)
ENABLE_DYNAMIC_WEB_SCRAPING=false
WEB_SCRAPING_FALLBACK_ONLY=true

# Conservative resource limits for Alpine
MAX_CONCURRENT_TASKS=2
CELERY_WORKER_CONCURRENCY=1
MAX_FILE_SIZE=50MB
WHISPER_MODEL_SIZE=base
EOF
    
    log_success "Environment configuration created"
    log_warning "Please edit .env file and add your GEMINI_API_KEY"
}

# Note: No local vector database installation needed
# User has Qdrant running on external server

# Create required directories
create_directories() {
    log_info "Creating required directories..."
    mkdir -p uploads temp logs
    chmod 755 uploads temp logs
    log_success "Required directories created"
}

# Main installation function
main() {
    log_info "Starting MoRAG Alpine Linux installation (CPU-only, external Qdrant server)..."

    check_root
    check_alpine_version
    update_system
    install_build_tools
    install_system_dependencies
    install_media_dependencies
    install_ocr_dependencies
    install_web_dependencies
    install_python
    install_redis
    install_python_build_deps
    clone_repository
    create_virtualenv
    install_morag_dependencies
    handle_alpine_issues
    create_environment_config
    create_directories

    log_success "MoRAG installation completed successfully!"
    echo
    log_info "Next steps:"
    echo "1. Edit .env file and configure:"
    echo "   - Add your GEMINI_API_KEY"
    echo "   - Update QDRANT_HOST with your Qdrant server IP/hostname"
    echo "   - Update QDRANT_API_KEY if your Qdrant server requires authentication"
    echo "2. Verify services are running:"
    echo "   - Redis: redis-cli ping"
    echo "   - External Qdrant: curl http://your_qdrant_server:6333/health"
    echo "3. Initialize the database: source venv/bin/activate && python scripts/init_db.py"
    echo "4. Start the Celery worker: source venv/bin/activate && python scripts/start_worker.py"
    echo "5. Start the API server: source venv/bin/activate && uvicorn src.morag.api.main:app --host 0.0.0.0 --port 8000"
    echo "6. Test the installation: curl http://localhost:8000/health/"
    echo
    log_info "Services installed locally:"
    echo "- Redis (task queue): localhost:6379"
    echo "- MoRAG API (when started): localhost:8000"
    echo
    log_info "External services (configure in .env):"
    echo "- Qdrant (vector database): your_qdrant_server:6333"
    echo
    log_warning "IMPORTANT: Make sure to update .env file with your external Qdrant server details!"
}

# Run main function
main "$@"
