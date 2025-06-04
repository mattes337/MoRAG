# MoRAG Embedding Service

## Overview

The MoRAG Embedding Service is a component of the Modular Retrieval Augmented Generation (MoRAG) system. It provides functionality for generating embeddings from text using various embedding models, with a primary focus on Google's Gemini models.

## Features

- Text embedding generation using Google Gemini models
- Batch embedding processing for efficient handling of multiple texts
- Resilient embedding generation with retry mechanisms
- Rate limiting to prevent API quota exhaustion
- Text summarization capabilities

## Installation

```bash
pip install morag-embedding
```

## Usage

### Basic Usage

```python
from morag_embedding.service import GeminiEmbeddingService
from morag_core.config import Settings

# Initialize settings
settings = Settings()

# Create embedding service
embedding_service = GeminiEmbeddingService(api_key=settings.gemini_api_key)

# Generate embedding
result = await embedding_service.generate_embedding("This is a sample text for embedding.")
print(f"Generated embedding with {len(result.embedding)} dimensions")

# Generate batch embeddings
texts = ["First text", "Second text", "Third text"]
batch_result = await embedding_service.generate_batch_embeddings(texts)
print(f"Generated {len(batch_result.embeddings)} embeddings")

# Generate summary
summary_result = await embedding_service.generate_summary(
    "This is a long text that needs to be summarized. It contains multiple sentences and details."
)
print(f"Summary: {summary_result.summary}")
```

## Dependencies

- morag-core: Core components for MoRAG
- google-generativeai: Google's Generative AI Python SDK
- tenacity: Retry library for Python
- numpy: Numerical computing library
- aiohttp: Asynchronous HTTP client/server
- structlog: Structured logging

## License

MIT