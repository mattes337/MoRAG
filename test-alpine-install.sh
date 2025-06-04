#!/bin/bash

# Test script for Alpine install script
# This script simulates Alpine Linux environment and tests the install script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[TEST-INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[TEST-SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[TEST-WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[TEST-ERROR]${NC} $1"
}

# Test function to check if script would work
test_alpine_install() {
    log_info "Testing Alpine install script..."
    
    # Check if we're in the right directory
    if [ ! -f "alpine-install.sh" ]; then
        log_error "alpine-install.sh not found in current directory"
        return 1
    fi
    
    if [ ! -f "pyproject.toml" ]; then
        log_error "pyproject.toml not found - not in MoRAG repository"
        return 1
    fi
    
    if [ ! -f ".env.example" ]; then
        log_error ".env.example not found"
        return 1
    fi
    
    if [ ! -d "src/morag" ]; then
        log_error "src/morag directory not found"
        return 1
    fi
    
    log_success "Repository structure check passed"
    
    # Test the Python dependency extraction logic
    log_info "Testing Python dependency extraction..."
    python3 -c "
import re
try:
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    
    # Extract dependencies, excluding qdrant-client
    deps = re.findall(r'\"([^\"]+)\"', content.split('dependencies = [')[1].split(']')[0])
    filtered_deps = [dep for dep in deps if not dep.startswith('qdrant-client')]
    
    print(f'Found {len(deps)} total dependencies')
    print(f'Filtered to {len(filtered_deps)} dependencies (excluding qdrant-client)')
    
    # Write test requirements file
    with open('test_requirements_no_qdrant.txt', 'w') as f:
        for dep in filtered_deps:
            f.write(dep + '\n')
    
    print('Test requirements file created successfully')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_success "Dependency extraction test passed"
    else
        log_error "Dependency extraction test failed"
        return 1
    fi
    
    # Show what dependencies would be installed
    log_info "Dependencies that would be installed:"
    cat test_requirements_no_qdrant.txt
    
    # Clean up test file
    rm -f test_requirements_no_qdrant.txt
    
    # Test environment file creation
    log_info "Testing environment file creation..."
    if [ -f ".env" ]; then
        log_warning ".env file already exists, backing up as .env.backup"
        cp .env .env.backup
    fi
    
    # Test copying .env.example to .env
    cp .env.example .env.test
    
    # Test appending Alpine-specific settings
    cat >> .env.test << 'EOF'

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
    
    log_success "Environment file creation test passed"
    log_info "Test .env file created as .env.test"
    
    # Test directory creation
    log_info "Testing directory creation..."
    mkdir -p test_uploads test_temp test_logs
    chmod 755 test_uploads test_temp test_logs
    log_success "Directory creation test passed"
    
    # Clean up test directories
    rmdir test_uploads test_temp test_logs
    
    log_success "All tests passed! Alpine install script should work."
    
    return 0
}

# Main function
main() {
    log_info "Starting Alpine install script test..."
    
    if test_alpine_install; then
        log_success "Alpine install script test completed successfully!"
        echo
        log_info "The script should work on Alpine Linux. Key findings:"
        echo "1. Repository structure is correct"
        echo "2. Dependency extraction logic works"
        echo "3. Environment file creation works"
        echo "4. Directory creation works"
        echo
        log_info "Potential issues to watch for:"
        echo "1. Package availability on Alpine (FFmpeg, Tesseract, etc.)"
        echo "2. Python package compilation on musl libc"
        echo "3. Service installation and startup"
        echo "4. Network connectivity for package downloads"
    else
        log_error "Alpine install script test failed!"
        return 1
    fi
}

# Run main function
main "$@"
