# Graphiti Integration for MoRAG

This module provides comprehensive integration between MoRAG and Graphiti, a temporal knowledge graph system that uses episodes to represent knowledge with built-in deduplication, temporal queries, and hybrid search capabilities.

## Overview

Graphiti enhances MoRAG with:
- **Temporal Knowledge Graphs**: Episodes represent knowledge with temporal context
- **Automatic Deduplication**: Built-in deduplication of similar content
- **Hybrid Search**: Combines semantic and keyword search capabilities
- **Entity Relationships**: Automatic extraction and linking of entities
- **Temporal Queries**: Query knowledge based on time periods

## Installation

First, install the required dependencies:

```bash
pip install graphiti-core openai
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Neo4j Database Settings
GRAPHITI_NEO4J_URI=bolt://localhost:7687
GRAPHITI_NEO4J_USERNAME=neo4j
GRAPHITI_NEO4J_PASSWORD=password
GRAPHITI_NEO4J_DATABASE=morag_graphiti

# OpenAI API Settings (required for Graphiti)
OPENAI_API_KEY=your_openai_api_key_here

# Graphiti Settings
GRAPHITI_TELEMETRY_ENABLED=false
USE_PARALLEL_RUNTIME=false

# OpenAI Model Configuration
GRAPHITI_OPENAI_MODEL=gpt-4
GRAPHITI_OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# MoRAG Integration Settings
GRAPHITI_ENABLED=true
GRAPHITI_BACKEND_PREFERENCE=graphiti
GRAPHITI_ENABLE_FALLBACK=true
```

### Programmatic Configuration

```python
from morag_graph.graphiti import GraphitiConfig

config = GraphitiConfig(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    neo4j_database="morag_graphiti",
    openai_api_key="your-api-key",
    openai_model="gpt-4",
    openai_embedding_model="text-embedding-3-small"
)
```

## Core Components

### 1. Connection Service

Manages connections to Graphiti and provides basic episode operations:

```python
from morag_graph.graphiti import GraphitiConnectionService

async def example_connection():
    async with GraphitiConnectionService(config) as conn:
        # Create an episode
        success = await conn.create_episode(
            name="example_episode",
            content="This is example content",
            source_description="Example source"
        )
        
        # Search episodes
        results = await conn.search_episodes("example query", limit=10)
```

### 2. Document Episode Mapper

Converts MoRAG documents into Graphiti episodes:

```python
from morag_graph.graphiti import DocumentEpisodeMapper
from morag_core.models import Document

async def map_document():
    mapper = DocumentEpisodeMapper(config)
    
    # Map entire document to episode
    result = await mapper.map_document_to_episode(document)
    
    # Map individual chunks to episodes
    chunk_results = await mapper.map_document_chunks_to_episodes(document)
```

### 3. Search Service

Provides advanced search capabilities using Graphiti:

```python
from morag_graph.graphiti import GraphitiSearchService

async def search_example():
    search_service = GraphitiSearchService(config)
    
    # Perform hybrid search
    results, metrics = await search_service.search(
        query="artificial intelligence",
        limit=10,
        search_type="hybrid"
    )
    
    # Search by document ID
    doc_results, _ = await search_service.search_by_document_id("doc-123")
    
    # Search by metadata
    meta_results, _ = await search_service.search_by_metadata({
        "category": "research",
        "type": "paper"
    })
```

### 4. Hybrid Search Integration

Combines Graphiti with existing MoRAG search with fallback capabilities:

```python
from morag_graph.graphiti import HybridSearchService, GraphitiSearchService

async def hybrid_search_example():
    graphiti_service = GraphitiSearchService(config)
    hybrid_service = HybridSearchService(
        graphiti_service, 
        legacy_service=None,  # Optional legacy search service
        prefer_graphiti=True
    )
    
    # Search with automatic fallback
    result = await hybrid_service.search("query")
    print(f"Method used: {result['method_used']}")
    print(f"Fallback used: {result['fallback_used']}")
    print(f"Results: {len(result['results'])}")
```

## Usage Examples

### Complete Workflow

```python
import asyncio
from morag_graph.graphiti import (
    GraphitiConfig, DocumentEpisodeMapper, GraphitiSearchService
)
from morag_core.models import Document

async def complete_workflow():
    # Configure Graphiti
    config = GraphitiConfig(
        openai_api_key="your-api-key",
        neo4j_uri="bolt://localhost:7687"
    )
    
    # Create services
    mapper = DocumentEpisodeMapper(config)
    search_service = GraphitiSearchService(config)
    
    # Map document to episodes
    document = load_document()  # Your document loading logic
    episode_result = await mapper.map_document_to_episode(document)
    
    if episode_result["success"]:
        print(f"Created episode: {episode_result['episode_name']}")
        
        # Search for related content
        results, metrics = await search_service.search(
            "related content query",
            limit=5
        )
        
        print(f"Found {len(results)} related episodes")
        for result in results:
            print(f"- {result.content[:100]}... (score: {result.score})")

# Run the workflow
asyncio.run(complete_workflow())
```

### Integration with Existing MoRAG Code

```python
from morag_graph import GRAPHITI_AVAILABLE

if GRAPHITI_AVAILABLE:
    from morag_graph.graphiti import HybridSearchService, GraphitiSearchService
    
    # Use Graphiti-enhanced search
    graphiti_service = GraphitiSearchService()
    search_service = HybridSearchService(graphiti_service)
else:
    # Fall back to existing search
    search_service = ExistingMoRAGSearchService()

# Use search_service regardless of Graphiti availability
results = await search_service.search_chunks("query")
```

## Error Handling

The integration includes comprehensive error handling:

```python
from morag_graph.graphiti import GraphitiConnectionService

async def robust_usage():
    service = GraphitiConnectionService()
    
    # Connection will fail gracefully if Graphiti is not available
    if await service.connect():
        print("Connected to Graphiti successfully")
        # Use Graphiti features
    else:
        print("Graphiti not available, using fallback")
        # Use alternative approach
```

## Testing

The integration includes comprehensive tests:

```bash
# Run all Graphiti tests
cd packages/morag-graph
python -m pytest tests/ -k "graphiti" -v

# Run specific test categories
python -m pytest tests/test_graphiti_config.py -v
python -m pytest tests/test_graphiti_connection.py -v
python -m pytest tests/test_graphiti_episode_mapper.py -v
python -m pytest tests/test_graphiti_search_service.py -v
python -m pytest tests/test_graphiti_search_integration.py -v
python -m pytest tests/test_graphiti_integration.py -v
```

## Architecture

The Graphiti integration follows a layered architecture:

1. **Configuration Layer**: `GraphitiConfig` manages all configuration
2. **Connection Layer**: `GraphitiConnectionService` handles low-level connections
3. **Mapping Layer**: `DocumentEpisodeMapper` converts MoRAG documents to episodes
4. **Search Layer**: `GraphitiSearchService` provides search capabilities
5. **Integration Layer**: `HybridSearchService` combines with existing MoRAG search

## Performance Considerations

- **Batch Processing**: Use batch operations for multiple documents
- **Connection Pooling**: Reuse connection services across operations
- **Caching**: Graphiti includes built-in caching for episodes
- **Async Operations**: All operations are async for better performance

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure `graphiti-core` is installed
2. **Connection Failed**: Check Neo4j is running and credentials are correct
3. **API Key Error**: Verify OpenAI API key is set and valid
4. **Search Returns Empty**: Check if episodes have been created

### Debug Mode

Enable debug logging:

```python
import structlog
structlog.configure(level="DEBUG")
```

## Contributing

When contributing to the Graphiti integration:

1. Add tests for new functionality
2. Update this README for new features
3. Ensure backward compatibility
4. Follow the existing error handling patterns
5. Add proper logging for debugging
