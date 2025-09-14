#!/bin/bash
"""
MoRAG Remote Converter Installation Script

This script sets up the remote converter environment and dependencies.
"""

set -e  # Exit on any error

echo "🚀 MoRAG Remote Converter Installation"
echo "=" * 50

# Check if we're in the right directory
if [ ! -f "../../packages/morag-core/pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the tools/remote-converter directory"
    echo "Expected directory structure: MoRAG/tools/remote-converter/"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.9+ required, found Python $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, removing..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install MoRAG packages
echo "📦 Installing MoRAG packages..."
pip install -e ../../packages/morag-core
pip install -e ../../packages/morag-audio
pip install -e ../../packages/morag-video
pip install -e ../../packages/morag-document
pip install -e ../../packages/morag-image
pip install -e ../../packages/morag-web
pip install -e ../../packages/morag-youtube

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install requests pyyaml python-dotenv structlog

# Create configuration file
echo "⚙️ Creating sample configuration..."
python3 cli.py --create-config

# Create systemd service file
echo "🔧 Creating systemd service file..."
cat > morag-remote-converter.service << EOF
[Unit]
Description=MoRAG Remote Converter
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python $(pwd)/cli.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create startup script
echo "📝 Creating startup script..."
cat > start.sh << 'EOF'
#!/bin/bash
# MoRAG Remote Converter Startup Script

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Start the remote converter
python cli.py "$@"
EOF

chmod +x start.sh

# Create environment file template
echo "📝 Creating environment file template..."
cat > .env.example << 'EOF'
# MoRAG Remote Converter Environment Configuration

# Worker Configuration
MORAG_WORKER_ID=gpu-worker-01
MORAG_API_BASE_URL=http://localhost:8000
MORAG_WORKER_CONTENT_TYPES=audio,video
MORAG_WORKER_POLL_INTERVAL=10
MORAG_WORKER_MAX_CONCURRENT_JOBS=2

# Optional API Authentication
# MORAG_API_KEY=your-api-key-here

# Logging and Temp Directory
MORAG_LOG_LEVEL=INFO
MORAG_TEMP_DIR=/tmp/morag_remote

# Audio Processing Configuration
# Override Whisper model size (tiny, base, small, medium, large-v2, large-v3)
WHISPER_MODEL_SIZE=large-v3
# Alternative variable name (both are supported)
# MORAG_WHISPER_MODEL_SIZE=large-v3

# Audio language (optional, auto-detect if not set)
# MORAG_AUDIO_LANGUAGE=en

# Audio device (auto, cpu, cuda)
# MORAG_AUDIO_DEVICE=auto

# Enable/disable features
# MORAG_ENABLE_SPEAKER_DIARIZATION=true
# MORAG_ENABLE_TOPIC_SEGMENTATION=true

# SpaCy model for topic segmentation (en_core_web_sm, de_core_news_sm, etc.)
# MORAG_SPACY_MODEL=de_core_news_sm
EOF

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Configure the remote converter:"
echo "   cp remote_converter_config.yaml.example remote_converter_config.yaml"
echo "   # Edit the configuration file with your settings"
echo ""
echo "2. (Optional) Set up environment variables:"
echo "   cp .env.example .env"
echo "   # Edit the .env file with your settings"
echo ""
echo "3. Test the connection:"
echo "   ./start.sh --test-connection"
echo ""
echo "4. Start the remote converter:"
echo "   ./start.sh"
echo ""
echo "🔧 System service installation (optional):"
echo "   sudo cp morag-remote-converter.service /etc/systemd/system/"
echo "   sudo systemctl enable morag-remote-converter"
echo "   sudo systemctl start morag-remote-converter"
echo ""
echo "📊 Monitor the service:"
echo "   sudo systemctl status morag-remote-converter"
echo "   sudo journalctl -u morag-remote-converter -f"
echo ""
echo "🎉 Happy processing!"
