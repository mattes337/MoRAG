# MoRAG Stages

Stage-based processing system for MoRAG that provides modular, reusable processing stages for content ingestion and analysis.

## Overview

MoRAG Stages implements a pipeline architecture with five canonical stages:

1. **markdown-conversion**: Convert input files to unified markdown format
2. **markdown-optimizer**: LLM-based text improvement and error correction (optional)
3. **chunker**: Create summary, chunks, and contextual embeddings
4. **fact-generator**: Extract facts, entities, relations, and keywords
5. **ingestor**: Database ingestion and storage

## Features

- **Modular Design**: Each stage can be executed independently
- **File-Based Interface**: Stages communicate through standardized file formats
- **Webhook Support**: Notifications for stage completions
- **Resume Capability**: Skip completed stages using intermediate files
- **Configurable**: Extensive configuration options for each stage
- **Type Safety**: Full type hints and Pydantic models

## Installation

```bash
pip install morag-stages
```

## Quick Start

```python
from morag_stages import StageManager, StageType, StageContext
from pathlib import Path

# Create stage manager
manager = StageManager()

# Create context
context = StageContext(
    source_path=Path("input.mp4"),
    output_dir=Path("./output"),
    config={
        "markdown-conversion": {
            "include_timestamps": True
        },
        "chunker": {
            "chunk_strategy": "semantic",
            "chunk_size": 4000
        }
    }
)

# Execute single stage
result = await manager.execute_stage(
    StageType.MARKDOWN_CONVERSION,
    [Path("input.mp4")],
    context
)

# Execute stage chain
results = await manager.execute_stage_chain(
    [StageType.MARKDOWN_CONVERSION, StageType.CHUNKER],
    [Path("input.mp4")],
    context
)
```

## Stage Types

### StageType.MARKDOWN_CONVERSION
Converts various input formats to markdown:
- Video/Audio files → Transcribed markdown with timestamps
- PDF/Documents → Structured markdown with metadata
- URLs → Scraped content as markdown
- Text files → Normalized markdown

**Output**: `{filename}.md`

### StageType.MARKDOWN_OPTIMIZER
Optional LLM-based improvement:
- Fix transcription errors
- Improve readability and structure
- Preserve timestamps and metadata

**Output**: `{filename}.opt.md`

### StageType.CHUNKER
Create chunks and embeddings:
- Generate document summary
- Split into semantic chunks
- Create contextual embeddings
- Add chunk metadata

**Output**: `{filename}.chunks.json`

### StageType.FACT_GENERATOR
Extract structured knowledge:
- Entity extraction and normalization
- Relationship identification
- Keyword extraction
- Domain-specific fact generation

**Output**: `{filename}.facts.json`

### StageType.INGESTOR
Database storage:
- Multi-database support (Qdrant, Neo4j)
- Deduplication and conflict resolution
- Configurable database targets

**Output**: `{filename}.ingestion.json`

## Configuration

Each stage accepts configuration through the `StageContext.config` dictionary:

```python
config = {
    "markdown-conversion": {
        "include_timestamps": True,
        "preserve_formatting": True,
        "transcription_model": "whisper-large"
    },
    "markdown-optimizer": {
        "model": "gemini-pro",
        "max_tokens": 8192,
        "temperature": 0.1
    },
    "chunker": {
        "chunk_strategy": "semantic",  # or "page-level", "topic-based"
        "chunk_size": 4000,
        "overlap": 200,
        "generate_summary": True
    },
    "fact-generator": {
        "extract_entities": True,
        "extract_relations": True,
        "extract_keywords": True,
        "domain": "general"
    },
    "ingestor": {
        "databases": ["qdrant", "neo4j"],
        "collection_name": "documents",
        "batch_size": 50
    }
}
```

## File Formats

### Markdown Files (.md)
```markdown
---
title: "Document Title"
source: "input.mp4"
type: "video"
duration: 1800
language: "en"
created_at: "2024-01-15T10:30:00Z"
---

# Content

[00:00 - 00:05] Introduction to the topic...
[00:05 - 00:12] Main discussion points...
```

### Chunks File (.chunks.json)
```json
{
    "summary": "Document summary...",
    "chunks": [
        {
            "id": "chunk_001",
            "content": "Chunk content...",
            "metadata": {
                "start_time": "00:00",
                "end_time": "00:30",
                "page": 1,
                "topic": "Introduction"
            },
            "embedding": [0.1, 0.2, ...],
            "context_summary": "Contextual summary..."
        }
    ],
    "metadata": {
        "total_chunks": 25,
        "chunk_strategy": "semantic",
        "embedding_model": "text-embedding-004"
    }
}
```

### Facts File (.facts.json)
```json
{
    "entities": [
        {
            "name": "Entity Name",
            "type": "Person",
            "normalized_name": "entity_name",
            "confidence": 0.95,
            "source_chunks": ["chunk_001", "chunk_003"]
        }
    ],
    "relations": [
        {
            "subject": "Entity A",
            "predicate": "WORKS_FOR",
            "object": "Entity B",
            "confidence": 0.88,
            "source_chunks": ["chunk_002"]
        }
    ],
    "facts": [
        {
            "statement": "Factual statement...",
            "entities": ["Entity A", "Entity B"],
            "source_chunk": "chunk_001",
            "confidence": 0.92
        }
    ],
    "keywords": ["keyword1", "keyword2", "keyword3"]
}
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
isort src tests

# Type checking
mypy src

# Linting
ruff check src tests
```

## License

MIT License - see LICENSE file for details.
