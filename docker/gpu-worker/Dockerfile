FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files
COPY packages/ ./packages/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Install Python dependencies
RUN pip3 install -e packages/morag-core && \
    pip3 install -e packages/morag-services && \
    pip3 install -e packages/morag-audio && \
    pip3 install -e packages/morag-video && \
    pip3 install -e packages/morag-document && \
    pip3 install -e packages/morag-image && \
    pip3 install -e packages/morag-web && \
    pip3 install -e packages/morag-youtube && \
    pip3 install -e packages/morag

# Create temp directory
RUN mkdir -p /app/temp && chmod 777 /app/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV TEMP_DIR=/app/temp

# Make scripts executable
RUN chmod +x ./scripts/start-gpu-worker.sh

# Default command
CMD ["./scripts/start-gpu-worker.sh", "/app/configs/gpu-worker.env"]
