# MoRAG Core

Core components for the MoRAG (Multimodal RAG Ingestion Pipeline) project.

## Overview

This package contains the essential core components of the MoRAG system, including:

- Base exceptions and error handling
- Configuration management
- Common utilities
- Interface definitions and base classes

## Installation

```bash
pip install morag-core
```

## Usage

```python
from morag_core.exceptions import MoRAGException, ValidationError
from morag_core.config import Settings

# Use core components in your application
settings = Settings()
print(f"API Host: {settings.api_host}")

try:
    # Your code here
    pass
except ValidationError as e:
    print(f"Validation error: {e}")
except MoRAGException as e:
    print(f"General error: {e}")
```

## Dependencies

This package has minimal dependencies to avoid dependency conflicts:

- pydantic
- pydantic-settings
- structlog
- python-dotenv

## License

MIT