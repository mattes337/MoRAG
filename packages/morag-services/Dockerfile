# Dockerfile for MoRAG Services Package
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

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

# Copy requirements and install dependencies
COPY packages/morag-core/ ./packages/morag-core/
COPY packages/morag-services/pyproject.toml packages/morag-services/README.md ./

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -e ./packages/morag-core/

# Copy source code
COPY packages/morag-services/src/ ./src/

# Install package
RUN pip install -e .

# Change ownership to app user
RUN chown -R app:app /app

# Switch to app user
USER app

# Expose port for service
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import morag_services; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "morag_services.cli"]
