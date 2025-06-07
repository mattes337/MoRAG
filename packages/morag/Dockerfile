# Multi-stage Dockerfile for MoRAG main package
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY packages/morag/pyproject.toml packages/morag/README.md ./
COPY packages/morag-core/ ./packages/morag-core/
COPY packages/morag-services/ ./packages/morag-services/
COPY packages/morag-web/ ./packages/morag-web/
COPY packages/morag-youtube/ ./packages/morag-youtube/

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -e ./packages/morag-core/
RUN pip install -e ./packages/morag-services/
RUN pip install -e ./packages/morag-web/
RUN pip install -e ./packages/morag-youtube/

# Copy source code
COPY packages/morag/src/ ./src/

# Install main package
RUN pip install -e .

# Change ownership to app user
RUN chown -R app:app /app

# Switch to app user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["morag-server", "--host", "0.0.0.0", "--port", "8000"]

# API Server stage
FROM base as api-server
CMD ["morag-server", "--host", "0.0.0.0", "--port", "8000"]

# Worker stage
FROM base as worker
CMD ["morag-worker", "--loglevel", "info", "--concurrency", "2"]

# CLI stage
FROM base as cli
ENTRYPOINT ["morag"]
