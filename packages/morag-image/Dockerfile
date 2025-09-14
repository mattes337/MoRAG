# Dockerfile for morag-image package
# Provides a ready-to-use environment with all dependencies installed

FROM python:3.10-slim

LABEL maintainer="Morag Project"
LABEL description="Docker image for morag-image package with all dependencies"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dri \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy package files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create a volume for input/output data
VOLUME ["/data"]

# Set environment variable for Google API key (to be overridden at runtime)
ENV GOOGLE_API_KEY=""

# Default command
CMD ["python", "-m", "morag_image.cli", "--help"]

# Usage instructions as comments
# Build: docker build -t morag-image .
# Run with a single image: 
#   docker run -v /path/to/local/dir:/data -e GOOGLE_API_KEY=your-key morag-image python -m morag_image.cli /data/image.jpg -o /data/output.json --caption --ocr
# Run with a directory of images:
#   docker run -v /path/to/local/dir:/data -e GOOGLE_API_KEY=your-key morag-image python -m morag_image.cli /data/images -o /data/output.json --caption --ocr