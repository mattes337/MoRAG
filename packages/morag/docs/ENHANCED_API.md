# Enhanced API Documentation

## Overview

The MoRAG Enhanced API (v2) provides graph-augmented retrieval capabilities that combine traditional vector search with knowledge graph traversal for more comprehensive and contextual results.

## Key Features

- **Graph-Augmented Retrieval**: Combines vector search with knowledge graph exploration
- **Multiple Query Types**: Support for simple, entity-focused, relation-focused, multi-hop, and analytical queries
- **Flexible Expansion Strategies**: Direct neighbors, breadth-first, shortest path, and adaptive expansion
- **Result Fusion**: Multiple strategies for combining vector and graph results
- **Streaming Support**: Real-time result streaming for large queries
- **Entity Exploration**: Direct entity and relationship queries
- **Graph Analytics**: Statistics and insights about the knowledge graph
- **Backward Compatibility**: Legacy API (v1) support with migration guidance

## API Endpoints

### Enhanced Query Endpoints (v2)

#### POST /api/v2/query
Enhanced query with graph-augmented retrieval.

**Request Body:**
```json
{
  "query": "How does machine learning relate to artificial intelligence?",
  "query_type": "entity_focused",
  "max_results": 10,
  "expansion_strategy": "adaptive",
  "expansion_depth": 2,
  "fusion_strategy": "adaptive",
  "include_graph_context": true,
  "include_reasoning_path": true,
  "enable_multi_hop": true,
  "min_relevance_score": 0.1,
  "entity_types": ["CONCEPT", "TECHNOLOGY"],
  "relation_types": ["SUBSET_OF", "RELATED_TO"],
  "timeout_seconds": 30
}
```

**Response:**
```json
{
  "query_id": "uuid-string",
  "query": "How does machine learning relate to artificial intelligence?",
  "results": [
    {
      "id": "result_1",
      "content": "Machine learning is a subset of artificial intelligence...",
      "relevance_score": 0.95,
      "source_type": "hybrid",
      "document_id": "doc_123",
      "metadata": {},
      "connected_entities": ["machine_learning", "artificial_intelligence"],
      "relation_context": [],
      "reasoning_path": ["entity_extraction", "graph_expansion", "result_fusion"]
    }
  ],
  "graph_context": {
    "entities": {
      "machine_learning": {
        "id": "machine_learning",
        "name": "Machine Learning",
        "type": "CONCEPT",
        "properties": {},
        "relevance_score": 0.9,
        "source_documents": ["doc_123"]
      }
    },
    "relations": [],
    "expansion_path": ["machine_learning", "artificial_intelligence"],
    "reasoning_steps": [
      "1. Analyzed query: 'How does machine learning...' (type: entity_focused)",
      "2. Applied adaptive expansion strategy with depth 2",
      "3. Found 5 vector-based results",
      "4. Found 3 graph-based results",
      "5. Applied adaptive fusion strategy",
      "6. Filtered results by minimum relevance score 0.1",
      "7. Returned top 1 results"
    ]
  },
  "total_results": 1,
  "processing_time_ms": 245.5,
  "fusion_strategy_used": "adaptive",
  "expansion_strategy_used": "adaptive",
  "confidence_score": 0.87,
  "completeness_score": 0.92
}
```

#### POST /api/v2/query/stream
Streaming version of enhanced query for real-time results.

#### POST /api/v2/entity/query
Query specific entities and their relationships.

**Request Body:**
```json
{
  "entity_name": "machine learning",
  "entity_type": "CONCEPT",
  "include_relations": true,
  "relation_depth": 2,
  "max_relations": 50
}
```

#### POST /api/v2/graph/traverse
Perform graph traversal between entities.

**Request Body:**
```json
{
  "start_entity": "entity_1",
  "end_entity": "entity_2",
  "traversal_type": "shortest_path",
  "max_depth": 3,
  "max_paths": 10,
  "relation_filters": ["RELATED_TO", "SUBSET_OF"],
  "entity_filters": ["CONCEPT", "TECHNOLOGY"]
}
```

#### GET /api/v2/graph/analytics
Get graph analytics and statistics.

**Query Parameters:**
- `metric_type`: `overview`, `centrality`, or `communities`

### Legacy Endpoints (v1)

#### POST /api/v1/query
Legacy query endpoint for backward compatibility.

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "max_results": 10,
  "min_score": 0.1
}
```

#### GET /api/v1/migration-guide
Get migration guidance from v1 to v2 API.

#### GET /api/v1/health
Legacy health check with deprecation notice.

## Query Types

### Simple
Basic queries with minimal graph enhancement.
- Best for: General information retrieval
- Features: Basic vector search with light graph context

### Entity Focused
Queries centered around specific entities.
- Best for: Learning about specific concepts, people, or things
- Features: Entity extraction, neighbor expansion, entity-centric results

### Relation Focused
Queries about relationships between entities.
- Best for: Understanding connections and relationships
- Features: Relation extraction, path finding, relationship analysis

### Multi-hop
Complex queries requiring reasoning across multiple entities.
- Best for: Complex analytical questions
- Features: Multi-step reasoning, deep graph traversal, path analysis

### Analytical
Comprehensive analysis queries.
- Best for: Research and deep analysis
- Features: Full graph analysis, statistical insights, comprehensive context

## Expansion Strategies

### Direct Neighbors
Expand to immediate neighbors only.
- **Use case**: Quick context without deep exploration
- **Performance**: Fast
- **Depth**: 1 hop only

### Breadth First
Explore all entities at each depth level.
- **Use case**: Comprehensive local context
- **Performance**: Moderate
- **Depth**: Configurable (1-5)

### Shortest Path
Find shortest paths between query entities.
- **Use case**: Understanding connections
- **Performance**: Moderate to slow
- **Depth**: Path-dependent

### Adaptive
Automatically choose the best strategy based on query.
- **Use case**: General purpose (recommended)
- **Performance**: Variable
- **Depth**: Context-dependent

### None
Disable graph expansion (vector search only).
- **Use case**: Performance-critical scenarios
- **Performance**: Fastest
- **Depth**: 0

## Fusion Strategies

### Weighted
Combine results using weighted scores.
- **Best for**: Balanced results
- **Weights**: Vector: 0.6, Graph: 0.4

### Reciprocal Rank Fusion (RRF)
Combine results using reciprocal rank fusion.
- **Best for**: Diverse result sets
- **Method**: Rank-based combination

### Adaptive
Automatically choose the best fusion strategy.
- **Best for**: General purpose (recommended)
- **Method**: Context-dependent selection

### Vector Only
Use only vector search results.
- **Best for**: Traditional RAG behavior
- **Performance**: Fastest

### Graph Only
Use only graph-based results.
- **Best for**: Relationship-focused queries
- **Performance**: Variable

## Error Handling

### Common Error Codes

- **400 Bad Request**: Invalid query parameters or validation errors
- **408 Request Timeout**: Query exceeded timeout limit
- **500 Internal Server Error**: Server-side processing errors
- **503 Service Unavailable**: Graph components not available

### Graceful Degradation

When graph components are unavailable:
- API falls back to vector-only search
- Graph-specific endpoints return 503 errors
- Legacy endpoints continue to work
- Clear error messages indicate reduced functionality

## Performance Considerations

### Query Complexity
- Simple queries: < 500ms
- Entity-focused queries: < 1s
- Multi-hop queries: < 2s
- Analytical queries: < 5s

### Optimization Tips
1. Use appropriate query types for your use case
2. Limit expansion depth for better performance
3. Use entity/relation type filters to reduce search space
4. Set reasonable timeout values
5. Consider using streaming for large result sets

## Migration from v1 to v2

### Key Changes
1. **Enhanced Request Model**: More configuration options
2. **Graph Context**: Additional context in responses
3. **New Query Types**: Support for different query strategies
4. **Streaming Support**: Real-time result delivery

### Migration Steps
1. Update endpoint from `/api/v1/query` to `/api/v2/query`
2. Update request model to `EnhancedQueryRequest`
3. Handle new response fields in `EnhancedQueryResponse`
4. Test with existing queries
5. Gradually adopt new features

### Backward Compatibility
- v1 endpoints remain available until 2024-12-01
- Deprecation warnings included in responses
- Migration guide available at `/api/v1/migration-guide`
- Feature parity maintained for basic functionality

## Examples

### Basic Query
```python
import requests

response = requests.post("http://localhost:8000/api/v2/query", json={
    "query": "What is machine learning?",
    "query_type": "simple",
    "max_results": 5
})

results = response.json()
print(f"Found {len(results['results'])} results")
```

### Entity-Focused Query
```python
response = requests.post("http://localhost:8000/api/v2/query", json={
    "query": "Tell me about neural networks",
    "query_type": "entity_focused",
    "expansion_strategy": "breadth_first",
    "expansion_depth": 2,
    "include_graph_context": True
})
```

### Multi-hop Reasoning
```python
response = requests.post("http://localhost:8000/api/v2/query", json={
    "query": "How do transformers relate to attention mechanisms in deep learning?",
    "query_type": "multi_hop",
    "expansion_strategy": "adaptive",
    "enable_multi_hop": True,
    "include_reasoning_path": True
})
```

## Support and Troubleshooting

### Common Issues
1. **Graph components not available**: Check if morag-graph package is installed
2. **Slow queries**: Reduce expansion depth or use simpler query types
3. **Empty results**: Check minimum relevance score and query complexity

### Getting Help
- Check the migration guide: `/api/v1/migration-guide`
- Review API documentation: `/docs`
- Monitor logs for detailed error information
