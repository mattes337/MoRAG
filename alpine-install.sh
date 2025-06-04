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

# Check if running as root (Alpine allows root execution)
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. This is acceptable on Alpine Linux."
        USE_SUDO=""
    else
        log_info "Running as regular user. Will use sudo for system operations."
        USE_SUDO="sudo"
        # Check if sudo exists (not default on Alpine)
        if ! command -v sudo &> /dev/null; then
            log_error "sudo not found. Please run as root or install sudo package."
            log_info "To install sudo: apk add sudo"
            exit 1
        fi
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
    $USE_SUDO apk update
    $USE_SUDO apk upgrade

    # Enable community repository for additional packages (like FFmpeg)
    log_info "Enabling community repository..."
    if ! grep -q "community" /etc/apk/repositories; then
        echo "http://dl-cdn.alpinelinux.org/alpine/v$(cat /etc/alpine-release | cut -d'.' -f1-2)/community" | $USE_SUDO tee -a /etc/apk/repositories
        $USE_SUDO apk update
    fi

    log_success "System packages updated and community repository enabled"
}

# Install essential build tools
install_build_tools() {
    log_info "Installing essential build tools..."
    $USE_SUDO apk add --no-cache \
        build-base \
        gcc \
        g++ \
        make \
        cmake \
        pkgconfig \
        autoconf \
        automake \
        libtool \
        git \
        curl \
        wget
    log_success "Build tools installed"
}

# Install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    $USE_SUDO apk add --no-cache \
        linux-headers \
        musl-dev \
        libffi-dev \
        openssl-dev \
        zlib-dev \
        jpeg-dev \
        libpng-dev \
        freetype-dev \
        lcms2-dev \
        openjpeg-dev \
        tiff-dev \
        tk-dev \
        tcl-dev \
        harfbuzz-dev \
        fribidi-dev \
        libimagequant-dev \
        libxcb-dev \
        libxml2-dev \
        libxslt-dev
    log_success "System dependencies installed"
}

# Install media processing dependencies
install_media_dependencies() {
    log_info "Installing media processing dependencies..."
    $USE_SUDO apk add --no-cache \
        ffmpeg \
        ffmpeg-dev \
        imagemagick \
        imagemagick-dev
    log_success "Media processing dependencies installed"
}

# Install OCR dependencies
install_ocr_dependencies() {
    log_info "Installing OCR dependencies..."
    $USE_SUDO apk add --no-cache \
        tesseract-ocr \
        tesseract-ocr-data-eng \
        tesseract-ocr-data-deu \
        tesseract-ocr-data-fra \
        tesseract-ocr-data-spa \
        poppler-utils \
        poppler-dev
    log_success "OCR dependencies installed"
}

# Install web dependencies
install_web_dependencies() {
    log_info "Installing web dependencies..."
    $USE_SUDO apk add --no-cache \
        chromium \
        chromium-chromedriver
    log_success "Web dependencies installed"
}

# Install Python
install_python() {
    log_info "Installing Python 3.11..."
    $USE_SUDO apk add --no-cache \
        python3 \
        python3-dev \
        py3-pip \
        py3-virtualenv

    # Create symlinks for compatibility
    if [ ! -f /usr/bin/python ]; then
        $USE_SUDO ln -sf /usr/bin/python3 /usr/bin/python
    fi

    log_success "Python 3.11 installed"
}

# Install Redis
install_redis() {
    log_info "Installing Redis..."
    $USE_SUDO apk add --no-cache redis

    # Enable and start service
    $USE_SUDO rc-update add redis default
    $USE_SUDO service redis start

    log_success "Redis installed and started"
}

# Install Python build dependencies
install_python_build_deps() {
    log_info "Installing Python build dependencies..."
    $USE_SUDO apk add --no-cache \
        py3-wheel \
        py3-setuptools \
        cython \
        py3-numpy \
        blas-dev \
        lapack-dev \
        gfortran
    log_success "Python build dependencies installed"
}

# Check if we're already in MoRAG repository
check_repository() {
    log_info "Checking if we're in MoRAG repository..."

    # Check if we're already in the MoRAG repository by looking for key files
    if [ -f "pyproject.toml" ] && [ -f ".env.example" ] && [ -d "src/morag" ]; then
        log_success "Already in MoRAG repository directory"
        return 0
    elif [ -d "morag" ]; then
        log_info "Found morag subdirectory, entering it..."
        cd morag
        if [ -f "pyproject.toml" ] && [ -f ".env.example" ] && [ -d "src/morag" ]; then
            log_success "Entered MoRAG repository directory"
            return 0
        else
            log_error "morag directory exists but doesn't contain MoRAG files"
            return 1
        fi
    else
        log_error "Not in MoRAG repository and no morag subdirectory found"
        log_error "Please run this script from within the cloned MoRAG repository"
        log_info "To clone the repository: git clone https://github.com/yourusername/morag.git"
        return 1
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

# Install MoRAG dependencies (Alpine-optimized)
install_morag_dependencies() {
    log_info "Installing MoRAG dependencies (Alpine-optimized)..."

    source venv/bin/activate

    # Extract dependencies from pyproject.toml, excluding problematic ones
    python3 -c "
import re
try:
    with open('pyproject.toml', 'r') as f:
        content = f.read()

    # Extract dependencies, excluding qdrant-client and docling
    deps = re.findall(r'\"([^\"]+)\"', content.split('dependencies = [')[1].split(']')[0])
    filtered_deps = [dep for dep in deps if not dep.startswith('qdrant-client') and not dep.startswith('docling')]

    # Write filtered requirements
    with open('requirements_alpine.txt', 'w') as f:
        for dep in filtered_deps:
            f.write(dep + '\n')

    print(f'Created requirements_alpine.txt with {len(filtered_deps)} dependencies')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"

    # Install dependencies
    pip install -r requirements_alpine.txt

    # Install the package itself in development mode (without docling extra)
    pip install -e .

    # Clean up temporary file
    rm -f requirements_alpine.txt

    log_success "MoRAG dependencies installed"
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

    # Update .env for Alpine Linux
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
    check_repository
    create_virtualenv
    install_morag_dependencies
    create_environment_config
    create_directories

    log_success "MoRAG Alpine Linux installation completed successfully!"
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
