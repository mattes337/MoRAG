"""MoRAG Core - Essential components for the MoRAG system.

MoRAG Core provides the fundamental interfaces, models, and utilities that power
the MoRAG system. This package includes AI agent abstractions, semantic chunking,
and core configuration management.

## Key Components

### AI Agents
Type-safe PydanticAI-powered agents for structured content processing:

```python
from morag_core import create_agent, AgentConfig

# Create a configured agent
config = AgentConfig(model="gemini-1.5-pro", temperature=0.1)
agent = create_agent("content_analysis", config)

# Process content with structured output
result = await agent.run("Analyze this document...")
```

### Semantic Chunking
Intelligent content segmentation based on meaning and context:

```python
from morag_core import create_chunker, ChunkingConfig

# Configure semantic chunking
config = ChunkingConfig(
    strategy="semantic",
    chunk_size=2000,
    overlap_size=200
)
chunker = create_chunker(config)

# Chunk content intelligently
chunks = await chunker.chunk(content, metadata)
```

### Available Agents
- `content_analysis` - General content analysis and summarization
- `entity_extraction` - Named entity recognition and extraction
- `relation_extraction` - Relationship detection between entities
- `semantic_chunking` - Content-aware text segmentation
- `transcript_analysis` - Audio/video transcript processing

### Result Models
Structured output models for consistent data handling:
- `EntityExtractionResult` - Extracted entities with confidence scores
- `RelationExtractionResult` - Entity relationships and types
- `SummaryResult` - Content summaries with key insights
- `SemanticChunkingResult` - Intelligently segmented content
- `ContentAnalysisResult` - Comprehensive content analysis
- `TranscriptAnalysisResult` - Structured transcript processing

## Usage Examples

### Basic Agent Usage
```python
from morag_core import create_agent_with_config

# Quick agent creation with defaults
agent = create_agent_with_config("entity_extraction")
result = await agent.run("Extract entities from this text...")

# Access structured results
for entity in result.entities:
    print(f"{entity.name}: {entity.type} (confidence: {entity.confidence})")
```

### Semantic Chunking
```python
from morag_core import SemanticChunker, ChunkingConfig

# Configure for document processing
config = ChunkingConfig(
    chunk_size=1500,
    overlap_size=150,
    strategy="semantic"
)

chunker = SemanticChunker(config)
chunks = await chunker.chunk(document_text, {"source": "document.pdf"})

# Process chunks
for chunk in chunks:
    print(f"Chunk {chunk.index}: {len(chunk.content)} chars")
```

### Provider Configuration
```python
from morag_core import GeminiProvider, ProviderConfig

# Configure AI provider
config = ProviderConfig(
    api_key="your-api-key",
    model="gemini-1.5-pro",
    temperature=0.1
)

provider = GeminiProvider(config)
# Provider is automatically used by agents
```

## Installation

```bash
pip install morag-core
```

Or as part of the full MoRAG system:

```bash
pip install packages/morag/
```

## Dependencies

- PydanticAI for type-safe AI interactions
- Pydantic for data validation
- Structlog for structured logging

## Version

Current version: {__version__}
"""

from .ai import (
    MoRAGBaseAgent,
    AgentConfig,
    GeminiProvider,
    ProviderConfig,
    AgentFactory,
    create_agent,
    create_agent_with_config,
    EntityExtractionResult,
    RelationExtractionResult,
    SummaryResult,
    SemanticChunkingResult,
    ContentAnalysisResult,
    TranscriptAnalysisResult,
)

from .chunking import (
    ChunkingConfig,
    ChunkingStrategy,
    SemanticChunker,
    ChunkerFactory,
    create_chunker,
)

__version__ = "0.1.0"

__all__ = [
    "MoRAGBaseAgent",
    "AgentConfig",
    "GeminiProvider",
    "ProviderConfig",
    "AgentFactory",
    "create_agent",
    "create_agent_with_config",
    "ChunkingConfig",
    "ChunkingStrategy",
    "SemanticChunker",
    "ChunkerFactory",
    "create_chunker",
    "EntityExtractionResult",
    "RelationExtractionResult",
    "SummaryResult",
    "SemanticChunkingResult",
    "ContentAnalysisResult",
    "TranscriptAnalysisResult",
]
