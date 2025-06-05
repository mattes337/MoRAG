# Multi-stage build for MoRAG modular system
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package files and install MoRAG packages
COPY packages/ ./packages/
COPY requirements.txt ./

# Install base requirements
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install MoRAG packages in dependency order
RUN pip install -e packages/morag-core && \
    pip install -e packages/morag-embedding && \
    pip install -e packages/morag-audio && \
    pip install -e packages/morag-video && \
    pip install -e packages/morag-document && \
    pip install -e packages/morag-image && \
    pip install -e packages/morag-web && \
    pip install -e packages/morag-youtube && \
    pip install -e packages/morag-services && \
    pip install -e packages/morag

# Download spaCy model (optional, will fallback if not available)
RUN python -m spacy download en_core_web_sm || echo "spaCy model download failed, will use fallback"



# Production stage
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies for MoRAG services
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

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Install Playwright browsers (optional, for web scraping with dynamic content)
RUN python -m playwright install chromium || echo "Playwright browser installation failed, will use fallback"

# Create app user
RUN groupadd -r morag && useradd -r -g morag morag

# Create app directory
WORKDIR /app

# Copy application code and configuration
COPY packages/ ./packages/
COPY scripts/ ./scripts/
COPY examples/ ./examples/
COPY docs/ ./docs/
COPY *.md ./
COPY *.txt ./

# Create necessary directories
RUN mkdir -p uploads temp logs data && \
    chown -R morag:morag /app

# Switch to app user
USER morag

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run MoRAG API server
CMD ["python", "-m", "morag.server", "--host", "0.0.0.0", "--port", "8000"]
