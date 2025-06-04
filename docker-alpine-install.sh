#!/bin/bash

# Docker-friendly Alpine Linux install script for MoRAG
# This script is optimized for Docker containers and skips service management

set -e

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

# Update system packages
update_system() {
    log_info "Updating Alpine package index..."
    apk update
    log_success "Package index updated"
}

# Install build tools
install_build_tools() {
    log_info "Installing build tools..."
    apk add --no-cache \
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
    apk add --no-cache \
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
    apk add --no-cache \
        ffmpeg \
        ffmpeg-dev \
        imagemagick \
        imagemagick-dev
    log_success "Media processing dependencies installed"
}

# Install OCR dependencies
install_ocr_dependencies() {
    log_info "Installing OCR dependencies..."
    apk add --no-cache \
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
    apk add --no-cache \
        chromium \
        chromium-chromedriver
    log_success "Web dependencies installed"
}

# Install Python
install_python() {
    log_info "Installing Python 3.11..."
    apk add --no-cache \
        python3 \
        python3-dev \
        py3-pip \
        py3-virtualenv
    
    # Create symlinks for compatibility
    if [ ! -f /usr/bin/python ]; then
        ln -sf /usr/bin/python3 /usr/bin/python
    fi
    
    log_success "Python 3.11 installed"
}

# Install Redis
install_redis() {
    log_info "Installing Redis..."
    apk add --no-cache redis
    log_success "Redis installed (service management skipped in Docker)"
}

# Install Python build dependencies
install_python_build_deps() {
    log_info "Installing Python build dependencies..."
    apk add --no-cache \
        py3-wheel \
        py3-setuptools \
        cython \
        py3-numpy \
        blas-dev \
        lapack-dev \
        gfortran
    log_success "Python build dependencies installed"
}

# Check repository
check_repository() {
    log_info "Checking MoRAG repository..."
    
    if [ ! -f "pyproject.toml" ]; then
        log_error "pyproject.toml not found. Are you in the MoRAG repository?"
        exit 1
    fi
    
    if [ ! -d "src/morag" ]; then
        log_error "src/morag directory not found. Invalid repository structure."
        exit 1
    fi
    
    log_success "Repository structure verified"
}

# Create virtual environment
create_virtualenv() {
    log_info "Creating Python virtual environment..."
    
    if [ -d "venv" ]; then
        log_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment created and activated"
}

# Install MoRAG dependencies (simplified for Docker)
install_morag_dependencies() {
    log_info "Installing MoRAG dependencies (Docker-optimized)..."
    
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
    with open('requirements_docker.txt', 'w') as f:
        for dep in filtered_deps:
            f.write(dep + '\n')
    
    print(f'Created requirements_docker.txt with {len(filtered_deps)} dependencies')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"
    
    # Install dependencies
    pip install -r requirements_docker.txt
    
    # Install the package itself in development mode (without docling extra)
    pip install -e .
    
    log_success "MoRAG dependencies installed"
}

# Create environment configuration
create_environment_config() {
    log_info "Creating environment configuration..."
    
    if [ -f ".env" ]; then
        log_warning ".env file already exists. Creating .env.docker as backup."
        cp .env.example .env.docker
    else
        cp .env.example .env
    fi
    
    # Update .env for Docker/Alpine Linux
    cat >> .env << 'EOF'

# Docker/Alpine Linux specific settings
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

# Conservative resource limits for Docker
MAX_CONCURRENT_TASKS=2
CELERY_WORKER_CONCURRENCY=1
MAX_FILE_SIZE=50MB
WHISPER_MODEL_SIZE=base
EOF
    
    log_success "Environment configuration created"
}

# Create required directories
create_directories() {
    log_info "Creating required directories..."
    mkdir -p uploads temp logs
    chmod 755 uploads temp logs
    log_success "Required directories created"
}

# Main installation function
main() {
    log_info "Starting MoRAG Docker Alpine Linux installation..."

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

    log_success "MoRAG Docker installation completed successfully!"
    echo
    log_info "Container is ready. Configure .env file with your settings."
}

# Run main function
main "$@"
