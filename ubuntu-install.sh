#!/bin/bash

# MoRAG Ubuntu Linux Installation Script
# This script installs MoRAG on Ubuntu Linux with full feature support

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
        log_warning "Running as root. This is not recommended for Ubuntu."
        log_warning "Consider running as a regular user with sudo privileges."
        USE_SUDO=""
    else
        log_info "Running as regular user. Will use sudo for system operations."
        USE_SUDO="sudo"
        # Check if user has sudo privileges
        if ! sudo -n true 2>/dev/null; then
            log_error "User does not have sudo privileges. Please run with sudo or as root."
            exit 1
        fi
    fi
}

# Check Ubuntu version
check_ubuntu_version() {
    if ! grep -q "Ubuntu" /etc/os-release; then
        log_error "This script is designed for Ubuntu Linux only."
        log_info "For other distributions, please adapt the package installation commands."
        exit 1
    fi
    
    local version=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
    local version_major=$(echo $version | cut -d'.' -f1)
    
    log_info "Detected Ubuntu version: $version"
    
    # Check for supported versions (18.04+)
    if [ "$version_major" -lt 18 ]; then
        log_error "Ubuntu 18.04 or later is required. Detected version: $version"
        exit 1
    fi
    
    # Set version-specific variables
    PYTHON_VERSION="python3"
    PYTHON_DEV="python3-dev"
    
    log_info "Will use Python version: $PYTHON_VERSION"
}

# Update system packages
update_system() {
    log_info "Updating system packages..."
    $USE_SUDO apt update
    $USE_SUDO apt upgrade -y
    log_success "System packages updated"
}

# Install essential build tools
install_build_tools() {
    log_info "Installing essential build tools..."
    $USE_SUDO apt install -y \
        build-essential \
        gcc \
        g++ \
        make \
        cmake \
        pkg-config \
        autoconf \
        automake \
        libtool \
        git \
        curl \
        wget \
        unzip \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release
    log_success "Build tools installed"
}

# Install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    $USE_SUDO apt install -y \
        linux-headers-$(uname -r) \
        libffi-dev \
        libssl-dev \
        zlib1g-dev \
        libjpeg-dev \
        libpng-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libopenjp2-7-dev \
        libtiff5-dev \
        tk-dev \
        tcl-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libxcb1-dev \
        libxml2-dev \
        libxslt1-dev \
        libcairo2-dev \
        libgirepository1.0-dev \
        libglib2.0-dev
    log_success "System dependencies installed"
}

# Install media processing dependencies
install_media_dependencies() {
    log_info "Installing media processing dependencies..."
    $USE_SUDO apt install -y \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        imagemagick \
        libmagickwand-dev
    log_success "Media processing dependencies installed"
}

# Install OCR dependencies
install_ocr_dependencies() {
    log_info "Installing OCR dependencies..."
    $USE_SUDO apt install -y \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-deu \
        tesseract-ocr-fra \
        tesseract-ocr-spa \
        libtesseract-dev \
        poppler-utils \
        libpoppler-dev \
        libpoppler-cpp-dev
    log_success "OCR dependencies installed"
}

# Install web dependencies
install_web_dependencies() {
    log_info "Installing web dependencies..."
    
    # Install Chrome/Chromium
    if command -v google-chrome &> /dev/null; then
        log_info "Google Chrome already installed"
    elif command -v chromium-browser &> /dev/null; then
        log_info "Chromium already installed"
    else
        log_info "Installing Chromium browser..."
        $USE_SUDO apt install -y chromium-browser chromium-chromedriver
    fi
    
    log_success "Web dependencies installed"
}

# Install Python
install_python() {
    log_info "Installing Python and related packages..."
    
    # Install Python and development packages
    $USE_SUDO apt install -y \
        $PYTHON_VERSION \
        $PYTHON_DEV \
        python3-pip \
        python3-venv \
        python3-wheel \
        python3-setuptools
    
    # Create symlinks for compatibility if needed
    if [ ! -f /usr/bin/python ]; then
        $USE_SUDO ln -sf /usr/bin/python3 /usr/bin/python
    fi
    
    # Upgrade pip
    python3 -m pip install --user --upgrade pip --break-system-packages
    
    log_success "Python installed and configured"
}

# Install Redis
install_redis() {
    log_info "Installing Redis..."
    $USE_SUDO apt install -y redis-server
    
    # Configure Redis to start on boot
    $USE_SUDO systemctl enable redis-server
    $USE_SUDO systemctl start redis-server
    
    # Test Redis connection
    if redis-cli ping | grep -q "PONG"; then
        log_success "Redis installed and running"
    else
        log_warning "Redis installed but may not be running properly"
    fi
}

# Install Python build dependencies
install_python_build_deps() {
    log_info "Installing Python build dependencies..."
    $USE_SUDO apt install -y \
        python3-numpy \
        python3-scipy \
        libatlas-base-dev \
        liblapack-dev \
        libblas-dev \
        gfortran \
        libhdf5-dev \
        libnetcdf-dev
    log_success "Python build dependencies installed"
}

# Install GPU support (optional)
install_gpu_support() {
    log_info "Checking for GPU support..."

    # Check if NVIDIA GPU is present
    if lspci | grep -i nvidia &> /dev/null; then
        log_info "NVIDIA GPU detected. Installing CUDA support..."

        # Add NVIDIA package repository
        if [ ! -f /etc/apt/sources.list.d/cuda.list ]; then
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d '.')/x86_64/cuda-keyring_1.0-1_all.deb
            $USE_SUDO dpkg -i cuda-keyring_1.0-1_all.deb
            $USE_SUDO apt update
        fi

        # Install CUDA toolkit (minimal)
        $USE_SUDO apt install -y cuda-toolkit-12-2

        log_success "CUDA support installed"
        log_info "You may need to reboot for GPU support to be fully available"
    else
        log_info "No NVIDIA GPU detected. Skipping CUDA installation."
        log_info "CPU-only mode will be used for AI processing."
    fi
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

    # Upgrade pip and install wheel
    pip install --upgrade pip wheel setuptools
    log_success "Virtual environment activated and pip upgraded"
}

# Install Ubuntu-compatible whisper alternatives
install_whisper_alternatives() {
    log_info "Installing whisper alternatives for speech recognition..."

    # Try to install whisper alternatives in order of preference
    local whisper_installed=false

    # Option 1: Try faster-whisper (preferred on Ubuntu)
    if ! $whisper_installed; then
        log_info "Attempting to install faster-whisper..."
        if pip install faster-whisper; then
            log_success "faster-whisper installed successfully"
            whisper_installed=true
        else
            log_warning "faster-whisper installation failed"
        fi
    fi

    # Option 2: Try Vosk (lightweight alternative)
    if ! $whisper_installed; then
        log_info "Attempting to install Vosk..."
        if pip install vosk; then
            log_success "Vosk installed successfully"
            whisper_installed=true
        else
            log_warning "Vosk installation failed"
        fi
    fi

    # Option 3: Try whispercpp (C++ implementation)
    if ! $whisper_installed; then
        log_info "Attempting to install whispercpp..."
        if pip install whispercpp; then
            log_success "whispercpp installed successfully"
            whisper_installed=true
        else
            log_warning "whispercpp installation failed"
        fi
    fi

    # Option 4: Try pywhispercpp (alternative C++ bindings)
    if ! $whisper_installed; then
        log_info "Attempting to install pywhispercpp..."
        if pip install pywhispercpp; then
            log_success "pywhispercpp installed successfully"
            whisper_installed=true
        else
            log_warning "pywhispercpp installation failed"
        fi
    fi

    # Option 5: Try whisper-cpp-python (another alternative)
    if ! $whisper_installed; then
        log_info "Attempting to install whisper-cpp-python..."
        if pip install whisper-cpp-python; then
            log_success "whisper-cpp-python installed successfully"
            whisper_installed=true
        else
            log_warning "whisper-cpp-python installation failed"
        fi
    fi

    # Option 6: Try original OpenAI whisper (last resort)
    if ! $whisper_installed; then
        log_info "Attempting to install openai-whisper as last resort..."
        if pip install openai-whisper; then
            log_success "openai-whisper installed successfully"
            whisper_installed=true
        else
            log_warning "openai-whisper installation failed"
        fi
    fi

    if ! $whisper_installed; then
        log_error "Failed to install any whisper alternative. Audio transcription will not be available."
        log_info "You can manually install a whisper library later if needed."
    else
        log_success "Whisper alternative installed successfully"
    fi
}

# Install MoRAG dependencies
install_morag_dependencies() {
    log_info "Installing MoRAG dependencies..."

    source venv/bin/activate

    # Extract dependencies from pyproject.toml, excluding problematic ones for Ubuntu
    python3 -c "
import re
try:
    with open('pyproject.toml', 'r') as f:
        content = f.read()

    # Extract dependencies (Ubuntu can handle most dependencies)
    deps = re.findall(r'\"([^\"]+)\"', content.split('dependencies = [')[1].split(']')[0])

    # For Ubuntu, we might exclude fewer dependencies than Alpine
    # Only exclude if there are known issues
    filtered_deps = [dep for dep in deps if not dep.startswith('qdrant-client')]  # Keep docling on Ubuntu

    # Write filtered requirements
    with open('requirements_ubuntu.txt', 'w') as f:
        for dep in filtered_deps:
            f.write(dep + '\n')

    print(f'Created requirements_ubuntu.txt with {len(filtered_deps)} dependencies')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"

    # Install dependencies
    pip install -r requirements_ubuntu.txt

    # Install the package itself in development mode
    pip install -e .

    # Install whisper alternatives
    install_whisper_alternatives

    # Clean up temporary file
    rm -f requirements_ubuntu.txt

    # Manual re-install the packages
    pip install pydub librosa mutagen ffmpeg-python Pillow opencv-python google.generativeai
    pip install protobuf==5.29.5

    log_success "MoRAG dependencies installed"
}

# Install Qdrant vector database (optional)
install_qdrant() {
    log_info "Installing Qdrant vector database..."

    read -p "Do you want to install Qdrant locally? (y/N): " install_qdrant_choice

    if [[ $install_qdrant_choice =~ ^[Yy]$ ]]; then
        log_info "Installing Qdrant locally..."

        # Install Docker if not present
        if ! command -v docker &> /dev/null; then
            log_info "Installing Docker..."
            curl -fsSL https://get.docker.com -o get-docker.sh
            $USE_SUDO sh get-docker.sh
            $USE_SUDO usermod -aG docker $USER
            rm get-docker.sh
            log_info "Docker installed. You may need to log out and back in for group changes to take effect."
        fi

        # Install Docker Compose if not present
        if ! command -v docker-compose &> /dev/null; then
            log_info "Installing Docker Compose..."
            $USE_SUDO curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            $USE_SUDO chmod +x /usr/local/bin/docker-compose
        fi

        # Create Qdrant Docker Compose file
        cat > docker-compose.qdrant.yml << 'EOF'
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: morag-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

volumes:
  qdrant_storage:
EOF

        # Start Qdrant
        docker-compose -f docker-compose.qdrant.yml up -d

        log_success "Qdrant installed and started locally"
        QDRANT_HOST="localhost"
        QDRANT_PORT="6333"
    else
        log_info "Skipping local Qdrant installation. You can configure an external Qdrant server in .env"
        QDRANT_HOST="your_qdrant_server_ip_here"
        QDRANT_PORT="6333"
    fi
}

# Create environment configuration
create_environment_config() {
    log_info "Creating environment configuration..."

    if [ -f ".env" ]; then
        log_warning ".env file already exists. Creating .env.ubuntu as backup."
        cp .env.example .env.ubuntu
    else
        cp .env.example .env
    fi

    # Detect GPU availability
    local device_setting="auto"
    local force_cpu="false"

    if lspci | grep -i nvidia &> /dev/null && command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected and drivers available"
        device_setting="auto"
        force_cpu="false"
    else
        log_info "No GPU detected or drivers not available, using CPU"
        device_setting="cpu"
        force_cpu="true"
    fi

    # Update .env for Ubuntu Linux
    cat >> .env << EOF

# Ubuntu Linux specific settings
PREFERRED_DEVICE=$device_setting
FORCE_CPU=$force_cpu

# Qdrant server configuration
QDRANT_HOST=$QDRANT_HOST
QDRANT_PORT=$QDRANT_PORT
QDRANT_COLLECTION_NAME=morag_documents
QDRANT_API_KEY=

# Web scraping settings (Ubuntu supports dynamic scraping)
ENABLE_DYNAMIC_WEB_SCRAPING=true
WEB_SCRAPING_FALLBACK_ONLY=false

# Resource limits for Ubuntu (can be higher than Alpine)
MAX_CONCURRENT_TASKS=4
CELERY_WORKER_CONCURRENCY=2
MAX_FILE_SIZE=100MB
WHISPER_MODEL_SIZE=base

# Whisper backend configuration (Ubuntu supports all backends)
# Leave empty for auto-selection of best available backend
WHISPER_BACKEND=
# Available backends: faster_whisper, vosk, whispercpp, pywhispercpp, whisper_cpp_python, openai_whisper
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

# Test installation
test_installation() {
    log_info "Testing installation..."

    source venv/bin/activate

    # Test Python imports
    python3 -c "
import sys
print(f'Python version: {sys.version}')

# Test basic imports
try:
    import morag
    print('✅ MoRAG package imported successfully')
except ImportError as e:
    print(f'❌ Failed to import MoRAG: {e}')
    sys.exit(1)

# Test whisper backends
try:
    from morag.services.whisper_backends import WhisperBackendFactory
    available_backends = WhisperBackendFactory.get_available_backends()
    print(f'✅ Available whisper backends: {available_backends}')
    if not available_backends:
        print('⚠️  No whisper backends available')
except Exception as e:
    print(f'❌ Whisper backend test failed: {e}')

# Test Redis connection
try:
    import redis
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')

print('Installation test completed')
"

    log_success "Installation test completed"
}

# Main installation function
main() {
    log_info "Starting MoRAG Ubuntu Linux installation..."

    check_root
    check_ubuntu_version
    update_system
    install_build_tools
    install_system_dependencies
    install_media_dependencies
    install_ocr_dependencies
    install_web_dependencies
    install_python
    install_redis
    install_python_build_deps
    install_gpu_support
    check_repository
    create_virtualenv
    install_morag_dependencies
    install_qdrant
    create_environment_config
    create_directories
    test_installation

    log_success "MoRAG Ubuntu Linux installation completed successfully!"
    echo
    log_info "Next steps:"
    echo "1. Edit .env file and configure:"
    echo "   - Add your GEMINI_API_KEY"
    if [ "$QDRANT_HOST" = "your_qdrant_server_ip_here" ]; then
        echo "   - Update QDRANT_HOST with your Qdrant server IP/hostname"
        echo "   - Update QDRANT_API_KEY if your Qdrant server requires authentication"
    fi
    echo "2. Verify services are running:"
    echo "   - Redis: redis-cli ping"
    if [ "$QDRANT_HOST" = "localhost" ]; then
        echo "   - Qdrant: curl http://localhost:6333/health"
    else
        echo "   - External Qdrant: curl http://$QDRANT_HOST:$QDRANT_PORT/health"
    fi
    echo "3. Initialize the database: source venv/bin/activate && python scripts/init_db.py"
    echo "4. Start the Celery worker: source venv/bin/activate && python scripts/start_worker.py"
    echo "5. Start the API server: source venv/bin/activate && uvicorn src.morag.api.main:app --host 0.0.0.0 --port 8000"
    echo "6. Test the installation: curl http://localhost:8000/health/"
    echo
    log_info "Services installed:"
    echo "- Redis (task queue): localhost:6379"
    if [ "$QDRANT_HOST" = "localhost" ]; then
        echo "- Qdrant (vector database): localhost:6333"
    fi
    echo "- MoRAG API (when started): localhost:8000"
    echo
    if [ "$QDRANT_HOST" != "localhost" ]; then
        log_info "External services (configure in .env):"
        echo "- Qdrant (vector database): $QDRANT_HOST:$QDRANT_PORT"
        echo
        log_warning "IMPORTANT: Make sure to update .env file with your external Qdrant server details!"
    fi

    # GPU-specific instructions
    if lspci | grep -i nvidia &> /dev/null; then
        echo
        log_info "GPU Support:"
        if command -v nvidia-smi &> /dev/null; then
            echo "✅ NVIDIA GPU and drivers detected"
            echo "- GPU acceleration should be available for AI processing"
            echo "- Whisper models will automatically use GPU when beneficial"
        else
            echo "⚠️  NVIDIA GPU detected but drivers may not be properly installed"
            echo "- Install NVIDIA drivers: sudo ubuntu-drivers autoinstall"
            echo "- Reboot after driver installation"
        fi
    fi

    echo
    log_info "Testing whisper backends:"
    echo "- Run: python tests/manual/test_whisper_backends.py"
    echo "- Or: python scripts/test_alpine_whisper_fix.py"
}

# Run main function
main "$@"
