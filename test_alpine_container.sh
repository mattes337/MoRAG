#!/bin/bash

# Alpine Container Test Script
# This script tests the Alpine Docker container functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Docker is running
check_docker() {
    log_info "Checking Docker..."
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    log_success "Docker is running"
}

# Check if docker-compose is available
check_docker_compose() {
    log_info "Checking Docker Compose..."
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    log_success "Docker Compose is available"
}

# Check if .env file exists
check_env_file() {
    log_info "Checking environment configuration..."
    if [ ! -f ".env" ]; then
        log_warning ".env file not found. Creating from template..."
        if [ -f ".env.alpine" ]; then
            cp .env.alpine .env
            log_warning "Please edit .env file with your actual configuration before running the container."
        else
            log_error ".env.alpine template not found. Please create .env file manually."
            exit 1
        fi
    else
        log_success ".env file found"
    fi
}

# Build the Alpine container
build_container() {
    log_info "Building Alpine container..."
    if docker-compose -f docker-compose.alpine.yml build; then
        log_success "Alpine container built successfully"
    else
        log_error "Failed to build Alpine container"
        exit 1
    fi
}

# Start the container stack
start_containers() {
    log_info "Starting container stack..."
    if docker-compose -f docker-compose.alpine.yml up -d; then
        log_success "Container stack started"
    else
        log_error "Failed to start container stack"
        exit 1
    fi
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for Redis
    log_info "Waiting for Redis..."
    for i in {1..30}; do
        if docker exec morag-redis-alpine redis-cli ping >/dev/null 2>&1; then
            log_success "Redis is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Redis failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
    
    # Wait for API
    log_info "Waiting for API..."
    for i in {1..60}; do
        if curl -f http://localhost:8000/health/ >/dev/null 2>&1; then
            log_success "API is ready"
            break
        fi
        if [ $i -eq 60 ]; then
            log_error "API failed to start within 60 seconds"
            exit 1
        fi
        sleep 1
    done
}

# Test API endpoints
test_api_endpoints() {
    log_info "Testing API endpoints..."
    
    # Test health endpoint
    if curl -s http://localhost:8000/health/ | grep -q "status"; then
        log_success "✓ Health endpoint works"
    else
        log_error "✗ Health endpoint failed"
        return 1
    fi
    
    # Test docs endpoint
    if curl -f http://localhost:8000/docs >/dev/null 2>&1; then
        log_success "✓ API documentation is accessible"
    else
        log_warning "⚠ API documentation may not be accessible"
    fi
    
    # Test ingestion endpoint structure
    if curl -s http://localhost:8000/docs | grep -q "ingestion"; then
        log_success "✓ Ingestion endpoints are available"
    else
        log_warning "⚠ Ingestion endpoints may not be available"
    fi
}

# Run container validation
run_container_validation() {
    log_info "Running container validation..."
    if docker exec morag-api-alpine bash scripts/validate_alpine_deployment.sh --quick; then
        log_success "✓ Container validation passed"
    else
        log_error "✗ Container validation failed"
        return 1
    fi
}

# Test basic functionality
test_basic_functionality() {
    log_info "Testing basic functionality..."
    if docker exec morag-api-alpine python3 scripts/test_alpine_container.py; then
        log_success "✓ Basic functionality tests passed"
    else
        log_error "✗ Basic functionality tests failed"
        return 1
    fi
}

# Show container status
show_status() {
    log_info "Container status:"
    docker-compose -f docker-compose.alpine.yml ps
    
    log_info "Container logs (last 10 lines):"
    echo "--- API Logs ---"
    docker-compose -f docker-compose.alpine.yml logs --tail=10 morag-api
    echo "--- Worker Logs ---"
    docker-compose -f docker-compose.alpine.yml logs --tail=10 morag-worker
}

# Cleanup function
cleanup() {
    if [ "$1" = "--cleanup" ]; then
        log_info "Stopping and removing containers..."
        docker-compose -f docker-compose.alpine.yml down
        log_success "Cleanup completed"
    fi
}

# Main test function
run_tests() {
    log_info "Starting Alpine Container Test Suite..."
    echo "============================================================"
    
    check_docker
    check_docker_compose
    check_env_file
    
    echo "------------------------------------------------------------"
    build_container
    
    echo "------------------------------------------------------------"
    start_containers
    
    echo "------------------------------------------------------------"
    wait_for_services
    
    echo "------------------------------------------------------------"
    test_api_endpoints
    
    echo "------------------------------------------------------------"
    run_container_validation
    
    echo "------------------------------------------------------------"
    test_basic_functionality
    
    echo "------------------------------------------------------------"
    show_status
    
    echo "============================================================"
    log_success "🎉 Alpine container test completed successfully!"
    log_info "Your Alpine container is ready for use."
    log_info ""
    log_info "Access points:"
    log_info "- API: http://localhost:8000"
    log_info "- Documentation: http://localhost:8000/docs"
    log_info "- Health Check: http://localhost:8000/health/"
    log_info ""
    log_info "To stop the containers: docker-compose -f docker-compose.alpine.yml down"
    log_info "To view logs: docker-compose -f docker-compose.alpine.yml logs -f"
}

# Help function
show_help() {
    echo "Alpine Container Test Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help        Show this help message"
    echo "  --cleanup     Stop and remove containers after testing"
    echo "  --build-only  Only build the container, don't run tests"
    echo "  --quick       Run quick tests (skip some validation steps)"
    echo ""
    echo "This script builds and tests the Alpine Docker container."
    echo "It validates that all services start correctly and API endpoints work."
}

# Parse command line arguments
case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    --cleanup)
        cleanup --cleanup
        exit 0
        ;;
    --build-only)
        check_docker
        check_docker_compose
        build_container
        log_success "Build completed. Use 'docker-compose -f docker-compose.alpine.yml up' to start."
        exit 0
        ;;
    --quick)
        log_info "Running quick tests..."
        check_docker
        check_docker_compose
        check_env_file
        start_containers
        wait_for_services
        test_api_endpoints
        log_success "Quick tests completed!"
        exit 0
        ;;
    "")
        run_tests
        ;;
    *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac

# Cleanup on exit if requested
trap 'cleanup $1' EXIT
