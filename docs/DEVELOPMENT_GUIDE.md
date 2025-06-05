# MoRAG Development Guide

## Overview

This guide provides comprehensive instructions for developing with the MoRAG modular architecture. MoRAG has been migrated from a monolithic structure to a modular package-based architecture for better maintainability, scalability, and development experience.

## Architecture Overview

### Modular Package Structure

MoRAG consists of the following packages:

- **`morag-core`** - Core interfaces, models, and base classes
- **`morag-services`** - AI services, vector storage, and processing services
- **`morag-web`** - Web content processing and scraping
- **`morag-youtube`** - YouTube video processing
- **`morag-audio`** - Audio processing and transcription
- **`morag-video`** - Video processing and conversion
- **`morag-document`** - Document processing (PDF, Office, etc.)
- **`morag-image`** - Image processing and OCR
- **`morag`** - Main integration package

### Package Dependencies

```
morag-core (base)
├── morag-services (depends on core)
├── morag-web (depends on core, services)
├── morag-audio (depends on core, services)
├── morag-video (depends on core, services, audio)
├── morag-document (depends on core, services)
├── morag-image (depends on core, services)
├── morag-youtube (depends on core, services, audio, video)
└── morag (depends on all packages)
```

## Development Environment Setup

### Prerequisites

- Python 3.11+ (recommended for Alpine Linux compatibility)
- Git
- Docker (optional, for containerized development)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd MoRAG
   ```

2. **Install in development mode:**
   ```bash
   # Install all packages in development mode
   pip install -e packages/morag-core
   pip install -e packages/morag-services
   pip install -e packages/morag-web
   pip install -e packages/morag-youtube
   pip install -e packages/morag-audio
   pip install -e packages/morag-video
   pip install -e packages/morag-document
   pip install -e packages/morag-image
   pip install -e packages/morag
   ```

3. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

## Import Guidelines

### ✅ Correct Import Patterns

Use the new modular package imports:

```python
# Core functionality
from morag_core.models import Document, DocumentChunk
from morag_core.interfaces.processor import BaseProcessor
from morag_core.config import Settings

# Services
from morag_services.embedding import gemini_service
from morag_services.storage import qdrant_service

# Processors
from morag_audio import AudioProcessor, AudioConfig
from morag_video import VideoProcessor, VideoConfig
from morag_web import WebProcessor, WebScrapingConfig
from morag_document import DocumentProcessor
from morag_image import ImageProcessor, ImageConfig
from morag_youtube import YouTubeProcessor
```

### ❌ Deprecated Import Patterns

Avoid the old monolithic imports:

```python
# DON'T USE - These are deprecated
from morag.processors.audio import AudioProcessor
from morag.services.embedding import gemini_service
from morag.converters.video import VideoConverter
from morag.core.config import Settings
```

### Backward Compatibility

For legacy code, compatibility layers exist in `src/morag/processors/__init__.py` that will redirect imports, but new code should use the modular imports.

## Package Development

### Creating a New Package

1. **Use the package creation script:**
   ```bash
   python scripts/create_package.py --name morag-newfeature
   ```

2. **Manual package structure:**
   ```
   packages/morag-newfeature/
   ├── src/
   │   └── morag_newfeature/
   │       ├── __init__.py
   │       ├── processor.py
   │       ├── converter.py
   │       └── services/
   ├── tests/
   ├── pyproject.toml
   ├── README.md
   └── CHANGELOG.md
   ```

### Package Development Workflow

1. **Make changes** to your package
2. **Run tests** for your package:
   ```bash
   cd packages/morag-yourpackage
   python -m pytest tests/
   ```
3. **Run integration tests:**
   ```bash
   python -m pytest tests/integration/
   ```
4. **Validate architecture compliance:**
   ```bash
   python scripts/validate_cleanup.py
   ```

### Adding Dependencies

Use package managers, not manual editing:

```bash
# For a specific package
cd packages/morag-audio
pip install new-dependency

# Update pyproject.toml accordingly
```

## Testing

### Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for package interactions
│   ├── test_package_independence.py
│   ├── test_architecture_compliance.py
│   └── test_cross_package_integration.py
├── manual/                  # Manual testing scripts
└── fixtures/                # Test data and fixtures
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/integration/test_package_independence.py -v

# Run tests for specific package
cd packages/morag-audio
python -m pytest tests/
```

### Integration Tests

The integration test suite validates:

- **Package Independence** - Each package works in isolation
- **Architecture Compliance** - No deprecated import patterns
- **Cross-Package Integration** - Packages work together correctly

## Code Style and Conventions

### Import Organization

```python
# Standard library imports
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import structlog
import pytest

# MoRAG core imports
from morag_core.models import Document
from morag_core.interfaces.processor import BaseProcessor

# MoRAG package imports
from morag_audio import AudioProcessor
from morag_services.embedding import gemini_service

# Local imports
from .utils import helper_function
```

### Error Handling

Use the centralized error handling from `morag_core`:

```python
from morag_core.exceptions import ProcessingError, ValidationError

try:
    result = await processor.process(document)
except ProcessingError as e:
    logger.error("Processing failed", error=str(e))
    raise
```

### Logging

Use structured logging consistently:

```python
import structlog

logger = structlog.get_logger(__name__)

# Good logging practices
logger.info("Processing started", file_path=str(file_path), file_size=file_size)
logger.error("Processing failed", error=str(e), file_path=str(file_path))
```

## Architecture Validation

### Automated Validation

Run validation scripts regularly:

```bash
# Validate cleanup and migration
python scripts/validate_cleanup.py

# Fix any remaining import issues
python scripts/fix_remaining_imports.py

# Check architecture compliance
python -m pytest tests/integration/test_architecture_compliance.py -v
```

### Manual Validation

1. **Check for circular dependencies:**
   ```bash
   python scripts/check_dependencies.py
   ```

2. **Validate package isolation:**
   ```bash
   python -m pytest tests/integration/test_package_independence.py -v
   ```

3. **Test cross-package integration:**
   ```bash
   python -m pytest tests/integration/test_cross_package_integration.py -v
   ```

## Common Development Tasks

### Adding a New Processor

1. **Create processor in appropriate package:**
   ```python
   # packages/morag-newtype/src/morag_newtype/processor.py
   from morag_core.interfaces.processor import BaseProcessor
   
   class NewTypeProcessor(BaseProcessor):
       async def process(self, input_data):
           # Implementation
           pass
   ```

2. **Export in package __init__.py:**
   ```python
   # packages/morag-newtype/src/morag_newtype/__init__.py
   from .processor import NewTypeProcessor
   
   __all__ = ['NewTypeProcessor']
   ```

3. **Add tests:**
   ```python
   # packages/morag-newtype/tests/test_processor.py
   import pytest
   from morag_newtype import NewTypeProcessor
   
   def test_processor_creation():
       processor = NewTypeProcessor()
       assert processor is not None
   ```

### Adding a New Service

1. **Add to morag-services package:**
   ```python
   # packages/morag-services/src/morag_services/new_service.py
   class NewService:
       def __init__(self):
           pass
   ```

2. **Update package exports:**
   ```python
   # packages/morag-services/src/morag_services/__init__.py
   from .new_service import NewService
   ```

### Updating Dependencies

1. **Use package managers:**
   ```bash
   cd packages/morag-audio
   pip install --upgrade some-dependency
   ```

2. **Update pyproject.toml:**
   ```toml
   [project]
   dependencies = [
       "some-dependency>=2.0.0",
   ]
   ```

3. **Test compatibility:**
   ```bash
   python -m pytest tests/
   ```

## Troubleshooting

### Common Issues

1. **Import errors after migration:**
   - Run `python scripts/fix_remaining_imports.py`
   - Check for typos in import statements
   - Ensure packages are installed in development mode

2. **Circular dependency errors:**
   - Review package dependency graph
   - Move shared functionality to `morag-core`
   - Use dependency injection patterns

3. **Test failures:**
   - Check if packages are properly installed
   - Verify test data and fixtures exist
   - Run tests in isolation to identify issues

### Getting Help

1. **Check validation reports:**
   ```bash
   python scripts/validate_cleanup.py
   ```

2. **Review architecture compliance:**
   ```bash
   python -m pytest tests/integration/test_architecture_compliance.py -v
   ```

3. **Check package structure:**
   ```bash
   python scripts/validate_package.py --package morag-audio
   ```

## Best Practices

1. **Always use modular imports** in new code
2. **Run integration tests** before committing changes
3. **Validate architecture compliance** regularly
4. **Use package managers** for dependency management
5. **Follow the dependency hierarchy** (don't create circular dependencies)
6. **Write tests** for new functionality
7. **Use structured logging** consistently
8. **Handle errors** gracefully with proper exception types

## Migration from Legacy Code

If you have legacy code using old import patterns:

1. **Run the import fixer:**
   ```bash
   python scripts/fix_remaining_imports.py
   ```

2. **Update manually if needed:**
   - Replace `from morag.processors.audio` with `from morag_audio`
   - Replace `from morag.services.embedding` with `from morag_services.embedding`

3. **Test thoroughly:**
   ```bash
   python -m pytest tests/integration/
   ```

4. **Validate compliance:**
   ```bash
   python scripts/validate_cleanup.py
   ```

This guide ensures consistent development practices across the MoRAG modular architecture.
