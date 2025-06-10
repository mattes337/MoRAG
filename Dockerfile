# Multi-stage build for MoRAG modular system
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (these rarely change, so cache them early)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first
RUN pip install --upgrade pip


# Dependencies stage - install Python packages before copying code
FROM base AS dependencies

# Set working directory for consistent paths
WORKDIR /build

# Copy only requirements.txt first (this layer will be cached unless requirements.txt changes)
COPY requirements.txt ./

# Install PyTorch CPU-only version first for compatibility
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install base requirements first - this is the heavy lifting that we want to cache
RUN pip install -r requirements.txt

# Download spaCy model (optional, will fallback if not available)
RUN python -m spacy download en_core_web_sm || echo "spaCy model download failed, will use fallback"


# Builder stage - install MoRAG packages with full source code
FROM dependencies AS builder

# Copy package files and install MoRAG packages
COPY packages/ ./packages/

# Install MoRAG packages in dependency order (non-editable for Docker)
RUN pip install /build/packages/morag-core && \
    pip install /build/packages/morag-embedding && \
    pip install /build/packages/morag-audio && \
    pip install /build/packages/morag-video && \
    pip install /build/packages/morag-document && \
    pip install /build/packages/morag-image && \
    pip install /build/packages/morag-web && \
    pip install /build/packages/morag-youtube && \
    pip install /build/packages/morag-services && \
    pip install /build/packages/morag



# Runtime base - install system runtime dependencies
FROM python:3.11-slim AS runtime-base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies (this layer is cached unless system deps change)
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    libreoffice \
    poppler-utils \
    chromium \
    chromium-driver \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    git \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxss1 \
    libgtk-3-0 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libcairo-gobject2 \
    libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome/Chromium path for web scraping
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Copy virtual environment from builder (contains all dependencies)
COPY --from=builder /opt/venv /opt/venv

# Install Playwright browsers (optional, for web scraping with dynamic content)
RUN python -m playwright install chromium || echo "Playwright browser installation failed, will use fallback"


# Development stage
FROM runtime-base AS development

# Create app directory
WORKDIR /app

# Copy application code and configuration (this layer changes frequently)
COPY packages/ ./packages/
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY docs/ ./docs/
COPY *.md ./
COPY *.txt ./

# Create necessary directories
RUN mkdir -p uploads temp logs data

# Default command for development - run MoRAG API server with reload
CMD ["python", "-m", "morag.server", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Production stage
FROM runtime-base AS production

# Create app user with home directory
RUN groupadd -r morag && useradd -r -g morag -m -d /home/morag morag

# Create app directory
WORKDIR /app

# Copy application code and configuration (this layer changes frequently)
COPY packages/ ./packages/
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY docs/ ./docs/
COPY *.md ./
COPY *.txt ./

# Create necessary directories including cache directories and remote jobs structure
RUN mkdir -p temp logs data \
    data/remote_jobs/pending \
    data/remote_jobs/processing \
    data/remote_jobs/completed \
    data/remote_jobs/failed \
    data/remote_jobs/timeout \
    data/remote_jobs/cancelled \
    /home/morag/.cache/huggingface \
    /home/morag/.cache/whisper \
    /home/morag/.cache/transformers && \
    chmod -R 755 /home/morag/.cache && \
    chmod -R 755 data && \
    chmod +x scripts/check_cpu_compatibility.py && \
    chmod +x scripts/start_worker_safe.sh && \
    chown -R morag:morag /app /home/morag

# Switch to app user
USER morag

# Run initialization scripts and then start the server
CMD ["sh", "-c", "python scripts/ensure_data_directories.py && python scripts/check_cpu_compatibility.py && python -m morag.server --host 0.0.0.0 --port 8000"]
