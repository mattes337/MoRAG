# Dockerfile for MoRAG Document Processing Service
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for document processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    libreoffice \
    poppler-utils \
    pandoc \
    wkhtmltopdf \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install document-specific dependencies
RUN pip install \
    pypdf2 \
    pymupdf \
    python-docx \
    python-pptx \
    openpyxl \
    xlrd \
    beautifulsoup4 \
    lxml \
    markdown \
    pdfplumber \
    tabula-py \
    camelot-py[cv] \
    pytesseract \
    pillow

# Copy source code
COPY packages/morag-core ./packages/morag-core
COPY packages/morag-embedding ./packages/morag-embedding
COPY packages/morag-document ./packages/morag-document

# Install packages
RUN pip install -e packages/morag-core
RUN pip install -e packages/morag-embedding
RUN pip install -e packages/morag-document

# Create directories
RUN mkdir -p /app/data /app/logs /app/temp

# Set permissions
RUN chmod -R 755 /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import morag_document; print('Document service healthy')" || exit 1

# Default command
CMD ["python", "-m", "morag_document.service"]

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
