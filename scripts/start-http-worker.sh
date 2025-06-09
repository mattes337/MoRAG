#!/bin/bash
# Start HTTP Remote Worker - No Redis Required
# This script starts a remote worker that connects directly to MoRAG server via HTTP

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
WORKER_TYPE="gpu"
POLL_INTERVAL=5
MAX_CONCURRENT=1

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Start HTTP Remote Worker for MoRAG (No Redis Required)

OPTIONS:
    -s, --server-url URL     Main server URL (required)
    -k, --api-key KEY        API key for authentication (required)
    -t, --worker-type TYPE   Worker type: gpu, cpu (default: gpu)
    -i, --poll-interval SEC  Polling interval in seconds (default: 5)
    -c, --max-concurrent N   Max concurrent tasks (default: 1)
    -e, --env-file FILE      Environment file path (default: .env)
    -h, --help               Show this help message

EXAMPLES:
    # Start GPU worker
    $0 --server-url http://main-server:8000 --api-key your-key

    # Start CPU worker with custom settings
    $0 -s http://main-server:8000 -k your-key -t cpu -i 10 -c 2

    # Use environment file
    $0 --env-file configs/http-worker.env

ENVIRONMENT VARIABLES:
    MORAG_SERVER_URL         Main server URL
    MORAG_API_KEY           API key for authentication
    WORKER_TYPE             Worker type (gpu/cpu)
    POLL_INTERVAL           Polling interval in seconds
    MAX_CONCURRENT_TASKS    Maximum concurrent tasks

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -t|--worker-type)
            WORKER_TYPE="$2"
            shift 2
            ;;
        -i|--poll-interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        -c|--max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        -e|--env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Load environment file if it exists
if [[ -f "$ENV_FILE" ]]; then
    print_info "Loading environment from: $ENV_FILE"
    set -a  # automatically export all variables
    source "$ENV_FILE"
    set +a
fi

# Use environment variables if command line options not provided
SERVER_URL="${SERVER_URL:-$MORAG_SERVER_URL}"
API_KEY="${API_KEY:-$MORAG_API_KEY}"
WORKER_TYPE="${WORKER_TYPE:-$WORKER_TYPE}"
POLL_INTERVAL="${POLL_INTERVAL:-$POLL_INTERVAL}"
MAX_CONCURRENT="${MAX_CONCURRENT:-$MAX_CONCURRENT_TASKS}"

# Validate required parameters
if [[ -z "$SERVER_URL" ]]; then
    print_error "Server URL is required. Use --server-url or set MORAG_SERVER_URL"
    show_usage
    exit 1
fi

if [[ -z "$API_KEY" ]]; then
    print_error "API key is required. Use --api-key or set MORAG_API_KEY"
    show_usage
    exit 1
fi

# Validate worker type
if [[ "$WORKER_TYPE" != "gpu" && "$WORKER_TYPE" != "cpu" ]]; then
    print_error "Invalid worker type: $WORKER_TYPE. Must be 'gpu' or 'cpu'"
    exit 1
fi

# Check if Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/start_http_remote_worker.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check Python and dependencies
print_info "Checking Python environment..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in a virtual environment or have the packages installed
if ! python3 -c "import morag" &> /dev/null; then
    print_warning "MoRAG packages not found. Make sure to install them first:"
    echo "  pip install -e packages/morag-core"
    echo "  pip install -e packages/morag"
    echo "  # ... install other packages as needed"
fi

# Generate worker ID
WORKER_ID="http-worker-$(hostname)-$(date +%s)"

# Display configuration
print_info "HTTP Remote Worker Configuration:"
echo "  Server URL: $SERVER_URL"
echo "  Worker Type: $WORKER_TYPE"
echo "  Worker ID: $WORKER_ID"
echo "  Poll Interval: ${POLL_INTERVAL}s"
echo "  Max Concurrent: $MAX_CONCURRENT"
echo "  API Key: ${API_KEY:0:8}..." # Show only first 8 characters
echo ""

# Test server connectivity
print_info "Testing server connectivity..."
if command -v curl &> /dev/null; then
    if curl -s --connect-timeout 10 "$SERVER_URL/health" > /dev/null; then
        print_success "Server is reachable"
    else
        print_warning "Cannot reach server at $SERVER_URL"
        print_warning "Worker will continue trying to connect..."
    fi
else
    print_warning "curl not available, skipping connectivity test"
fi

# Set up signal handling for graceful shutdown
cleanup() {
    print_info "Shutting down worker..."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Change to project directory
cd "$PROJECT_ROOT"

# Start the worker
print_success "Starting HTTP Remote Worker..."
print_info "Press Ctrl+C to stop"
echo ""

exec python3 "$PYTHON_SCRIPT" \
    --server-url "$SERVER_URL" \
    --api-key "$API_KEY" \
    --worker-id "$WORKER_ID" \
    --worker-type "$WORKER_TYPE" \
    --poll-interval "$POLL_INTERVAL" \
    --max-concurrent "$MAX_CONCURRENT"
