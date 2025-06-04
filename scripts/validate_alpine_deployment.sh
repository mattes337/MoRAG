#!/bin/bash

# Alpine Container Deployment Validation Script
# This script validates that the Alpine container is working correctly

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

# Test functions
test_container_environment() {
    log_info "Testing container environment..."
    
    # Check if we're in Alpine
    if [ -f /etc/alpine-release ]; then
        log_success "✓ Running in Alpine Linux $(cat /etc/alpine-release)"
    else
        log_warning "⚠ Not running in Alpine Linux"
    fi
    
    # Check Python version
    python_version=$(python3 --version 2>&1)
    log_info "Python version: $python_version"
    
    # Check if virtual environment is activated
    if [ -n "$VIRTUAL_ENV" ]; then
        log_success "✓ Virtual environment activated: $VIRTUAL_ENV"
    else
        log_warning "⚠ Virtual environment not activated"
    fi
}

test_system_dependencies() {
    log_info "Testing system dependencies..."
    
    dependencies=(
        "redis-cli"
        "tesseract"
        "ffmpeg"
        "curl"
        "chromium-browser"
    )
    
    for dep in "${dependencies[@]}"; do
        if command -v "$dep" >/dev/null 2>&1; then
            log_success "✓ $dep is available"
        else
            log_error "✗ $dep is missing"
        fi
    done
}

test_python_packages() {
    log_info "Testing Python packages..."
    
    # Activate virtual environment if not already activated
    if [ -z "$VIRTUAL_ENV" ] && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    # Test critical packages
    packages=(
        "fastapi"
        "uvicorn"
        "celery"
        "redis"
        "requests"
        "numpy"
        "pandas"
    )
    
    for package in "${packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            log_success "✓ $package is available"
        else
            log_error "✗ $package is missing"
        fi
    done
}

test_morag_import() {
    log_info "Testing MoRAG import..."
    
    # Activate virtual environment if not already activated
    if [ -z "$VIRTUAL_ENV" ] && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    if python3 -c "import morag; print('MoRAG version:', getattr(morag, '__version__', 'unknown'))" 2>/dev/null; then
        log_success "✓ MoRAG import successful"
    else
        log_error "✗ MoRAG import failed"
        return 1
    fi
}

test_redis_connection() {
    log_info "Testing Redis connection..."
    
    # Check if Redis is running locally
    if redis-cli ping >/dev/null 2>&1; then
        log_success "✓ Local Redis is responding"
    else
        log_warning "⚠ Local Redis not responding (may be using external Redis)"
    fi
    
    # Test Redis connection via Python
    if [ -z "$VIRTUAL_ENV" ] && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    python3 -c "
import redis
import os
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
try:
    r = redis.from_url(redis_url)
    r.ping()
    print('✓ Redis connection via Python successful')
except Exception as e:
    print(f'✗ Redis connection failed: {e}')
    exit(1)
" && log_success "Redis Python connection works" || log_error "Redis Python connection failed"
}

test_api_startup() {
    log_info "Testing API startup..."
    
    # Activate virtual environment if not already activated
    if [ -z "$VIRTUAL_ENV" ] && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    # Test if API can be imported
    if python3 -c "from morag.api.main import app; print('API app created successfully')" 2>/dev/null; then
        log_success "✓ API can be imported and created"
    else
        log_error "✗ API import/creation failed"
        return 1
    fi
}

test_basic_functionality() {
    log_info "Testing basic functionality..."
    
    # Activate virtual environment if not already activated
    if [ -z "$VIRTUAL_ENV" ] && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    # Run the Python test script
    if python3 scripts/test_alpine_container.py; then
        log_success "✓ Basic functionality tests passed"
    else
        log_error "✗ Basic functionality tests failed"
        return 1
    fi
}

start_api_server() {
    log_info "Starting API server for testing..."
    
    # Activate virtual environment if not already activated
    if [ -z "$VIRTUAL_ENV" ] && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    # Start API server in background
    uvicorn src.morag.api.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    
    # Wait for server to start
    log_info "Waiting for API server to start..."
    sleep 10
    
    # Test if server is responding
    if curl -f http://localhost:8000/health/ >/dev/null 2>&1; then
        log_success "✓ API server is responding"
        
        # Test API endpoints
        log_info "Testing API endpoints..."
        
        # Test health endpoint
        if curl -s http://localhost:8000/health/ | grep -q "status"; then
            log_success "✓ Health endpoint works"
        else
            log_error "✗ Health endpoint failed"
        fi
        
        # Test docs endpoint
        if curl -f http://localhost:8000/docs >/dev/null 2>&1; then
            log_success "✓ API documentation is accessible"
        else
            log_warning "⚠ API documentation may not be accessible"
        fi
        
    else
        log_error "✗ API server is not responding"
    fi
    
    # Stop API server
    kill $API_PID 2>/dev/null || true
    log_info "API server stopped"
}

run_validation() {
    log_info "Starting Alpine Container Validation..."
    echo "============================================================"
    
    # Run all tests
    test_container_environment
    echo "------------------------------------------------------------"
    
    test_system_dependencies
    echo "------------------------------------------------------------"
    
    test_python_packages
    echo "------------------------------------------------------------"
    
    test_morag_import
    echo "------------------------------------------------------------"
    
    test_redis_connection
    echo "------------------------------------------------------------"
    
    test_api_startup
    echo "------------------------------------------------------------"
    
    test_basic_functionality
    echo "------------------------------------------------------------"
    
    start_api_server
    echo "============================================================"
    
    log_success "🎉 Alpine container validation completed!"
    log_info "If all tests passed, your Alpine container is ready for production use."
    log_info "Next steps:"
    log_info "1. Configure your .env file with actual API keys and Qdrant server details"
    log_info "2. Start the container with: docker-compose -f docker-compose.alpine.yml up"
    log_info "3. Access the API at: http://localhost:8000"
    log_info "4. View API documentation at: http://localhost:8000/docs"
}

# Main execution
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Alpine Container Validation Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help, -h    Show this help message"
    echo "  --quick       Run quick validation (skip API server test)"
    echo ""
    echo "This script validates that the Alpine container is working correctly."
    echo "It tests system dependencies, Python packages, MoRAG functionality,"
    echo "and API server startup."
    exit 0
fi

if [ "$1" = "--quick" ]; then
    log_info "Running quick validation (skipping API server test)..."
    test_container_environment
    test_system_dependencies
    test_python_packages
    test_morag_import
    test_redis_connection
    test_api_startup
    log_success "Quick validation completed!"
else
    run_validation
fi
