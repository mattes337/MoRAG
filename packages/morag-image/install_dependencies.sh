#!/bin/bash
# Installation script for morag-image system dependencies

set -e

echo "Installing system dependencies for morag-image package..."

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
else
    echo "Cannot detect operating system. Please install dependencies manually."
    exit 1
fi

# Install dependencies based on OS
if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    echo "Detected Debian/Ubuntu-based system"
    
    # Update package lists
    sudo apt-get update
    
    # Install Tesseract OCR and language data
    echo "Installing Tesseract OCR..."
    sudo apt-get install -y tesseract-ocr
    sudo apt-get install -y tesseract-ocr-eng
    # Add more language packs as needed, e.g.:
    # sudo apt-get install -y tesseract-ocr-fra tesseract-ocr-deu
    
    # Install OpenCV dependencies
    echo "Installing OpenCV dependencies..."
    sudo apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-dri
    
    # Install Python development headers (needed for some packages)
    sudo apt-get install -y python3-dev
    
    echo "System dependencies installed successfully."
    
 elif [[ "$OS" == *"Fedora"* ]] || [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
    echo "Detected Red Hat/Fedora-based system"
    
    # Install Tesseract OCR and language data
    echo "Installing Tesseract OCR..."
    sudo dnf install -y tesseract
    sudo dnf install -y tesseract-langpack-eng
    # Add more language packs as needed
    
    # Install OpenCV dependencies
    echo "Installing OpenCV dependencies..."
    sudo dnf install -y libSM libXext libXrender mesa-libGL
    
    # Install Python development headers
    sudo dnf install -y python3-devel
    
    echo "System dependencies installed successfully."
    
else
    echo "Unsupported operating system: $OS"
    echo "Please install the following dependencies manually:"
    echo "- Tesseract OCR and language data"
    echo "- OpenCV system dependencies (libSM, libXext, libXrender, libGL)"
    echo "- Python development headers"
    exit 1
fi

# Verify Tesseract installation
if command -v tesseract > /dev/null; then
    echo "Tesseract OCR installed successfully:"
    tesseract --version | head -n 1
else
    echo "WARNING: Tesseract OCR installation could not be verified."
    echo "You may need to install it manually or check your PATH."
fi

echo ""
echo "Next steps:"
echo "1. Install the Python package: pip install morag-image"
echo "2. Set up your Google API key for Gemini Vision (if using captioning):"
echo "   export GOOGLE_API_KEY=your-api-key"
echo ""
echo "For more information, see the README.md file."