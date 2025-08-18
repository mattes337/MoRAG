# Dockerfile for MoRAG Web Processing Service
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for web processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    chromium \
    chromium-driver \
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
    libgdk-pixbuf-xlib-2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome/Chromium path for Selenium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install web-specific dependencies
RUN pip install \
    selenium \
    beautifulsoup4 \
    lxml \
    requests \
    httpx \
    aiohttp \
    playwright \
    scrapy \
    readability-lxml \
    newspaper3k \
    trafilatura

# Install Playwright browsers
RUN python -m playwright install chromium

# Copy source code
COPY packages/morag-core ./packages/morag-core
COPY packages/morag-embedding ./packages/morag-embedding
COPY packages/morag-web ./packages/morag-web

# Install packages
RUN pip install -e packages/morag-core
RUN pip install -e packages/morag-embedding
RUN pip install -e packages/morag-web

# Create directories
RUN mkdir -p /app/data /app/logs /app/temp

# Set permissions
RUN chmod -R 755 /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import morag_web; print('Web service healthy')" || exit 1

# Default command
CMD ["python", "-m", "morag_web.service"]

# =============================================================================
# Development stage
# =============================================================================
FROM base AS development

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy jupyter

# Set development environment
ENV ENVIRONMENT=development
ENV DEBUG=true

# =============================================================================
# Production stage
# =============================================================================
FROM base AS production

# Remove development tools and clean up
RUN apt-get autoremove -y && apt-get clean

# Set production environment
ENV ENVIRONMENT=production
ENV DEBUG=false

# Use non-root user for security
RUN useradd -m -u 1000 morag && chown -R morag:morag /app
USER morag
