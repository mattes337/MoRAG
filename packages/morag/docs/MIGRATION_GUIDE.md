# Migration Guide: API v1 to v2

## Overview

This guide helps you migrate from the legacy MoRAG API (v1) to the enhanced API (v2) with graph-augmented retrieval capabilities.

## Timeline

- **Deprecation Date**: January 1, 2024
- **End of Support**: December 1, 2024
- **Removal Date**: TBD (will be announced 6 months in advance)

## Key Differences

### Endpoint Changes

| v1 Endpoint | v2 Endpoint | Status |
|-------------|-------------|---------|
| `POST /api/v1/query` | `POST /api/v2/query` | Enhanced |
| `GET /api/v1/health` | `GET /api/v2/health` | New features |
| N/A | `POST /api/v2/entity/query` | New |
| N/A | `POST /api/v2/graph/traverse` | New |
| N/A | `GET /api/v2/graph/analytics` | New |
| N/A | `POST /api/v2/query/stream` | New |

### Request Model Changes

#### v1 Request (Legacy)
```json
{
  "query": "What is machine learning?",
  "max_results": 10,
  "min_score": 0.1,
  "filters": {}
}
```

#### v2 Request (Enhanced)
```json
{
  "query": "What is machine learning?",
  "query_type": "simple",
  "max_results": 10,
  "expansion_strategy": "adaptive",
  "expansion_depth": 2,
  "fusion_strategy": "adaptive",
  "include_graph_context": true,
  "include_reasoning_path": false,
  "enable_multi_hop": true,
  "min_relevance_score": 0.1,
  "timeout_seconds": 30
}
```

### Response Model Changes

#### v1 Response (Legacy)
```json
{
  "query": "What is machine learning?",
  "results": [
    {
      "id": "result_1",
      "content": "Machine learning is...",
      "score": 0.95,
      "metadata": {
        "document_id": "doc_123",
        "source_type": "vector"
      }
    }
  ],
  "total_results": 1,
  "processing_time_ms": 150.5
}
```

#### v2 Response (Enhanced)
```json
{
  "query_id": "uuid-string",
  "query": "What is machine learning?",
  "results": [
    {
      "id": "result_1",
      "content": "Machine learning is...",
      "relevance_score": 0.95,
      "source_type": "hybrid",
      "document_id": "doc_123",
      "metadata": {},
      "connected_entities": ["machine_learning", "ai"],
      "relation_context": [],
      "reasoning_path": null
    }
  ],
  "graph_context": {
    "entities": {},
    "relations": [],
    "expansion_path": [],
    "reasoning_steps": null
  },
  "total_results": 1,
  "processing_time_ms": 150.5,
  "fusion_strategy_used": "adaptive",
  "expansion_strategy_used": "adaptive",
  "confidence_score": 0.87,
  "completeness_score": 0.92
}
```

## Migration Steps

### Step 1: Update Client Code

#### Python Example

**Before (v1):**
```python
import requests

def query_v1(query, max_results=10):
    response = requests.post("http://localhost:8000/api/v1/query", json={
        "query": query,
        "max_results": max_results,
        "min_score": 0.1
    })
    
    data = response.json()
    return [
        {
            "content": result["content"],
            "score": result["score"],
            "document_id": result["metadata"]["document_id"]
        }
        for result in data["results"]
    ]
```

**After (v2):**
```python
import requests

def query_v2(query, max_results=10, query_type="simple"):
    response = requests.post("http://localhost:8000/api/v2/query", json={
        "query": query,
        "query_type": query_type,
        "max_results": max_results,
        "min_relevance_score": 0.1,
        "include_graph_context": True
    })
    
    data = response.json()
    return [
        {
            "content": result["content"],
            "score": result["relevance_score"],
            "document_id": result["document_id"],
            "source_type": result["source_type"],
            "entities": result["connected_entities"]
        }
        for result in data["results"]
    ]
```

### Step 2: Handle New Response Fields

#### Accessing Graph Context
```python
def extract_graph_context(response_data):
    graph_context = response_data.get("graph_context", {})
    
    entities = graph_context.get("entities", {})
    relations = graph_context.get("relations", [])
    reasoning_steps = graph_context.get("reasoning_steps", [])
    
    return {
        "entity_count": len(entities),
        "relation_count": len(relations),
        "has_reasoning": len(reasoning_steps) > 0
    }
```

#### Quality Metrics
```python
def get_quality_metrics(response_data):
    return {
        "confidence": response_data.get("confidence_score", 0.0),
        "completeness": response_data.get("completeness_score", 0.0),
        "processing_time": response_data.get("processing_time_ms", 0.0)
    }
```

### Step 3: Gradual Feature Adoption

#### Phase 1: Basic Migration
Start with minimal changes to maintain compatibility:

```python
def migrate_basic(query, max_results=10):
    # Use v2 endpoint with v1-like parameters
    response = requests.post("http://localhost:8000/api/v2/query", json={
        "query": query,
        "query_type": "simple",
        "max_results": max_results,
        "include_graph_context": False,  # Disable new features initially
        "enable_multi_hop": False
    })
    
    # Extract results in v1 format
    data = response.json()
    return {
        "query": data["query"],
        "results": [
            {
                "id": result["id"],
                "content": result["content"],
                "score": result["relevance_score"],
                "metadata": {
                    "document_id": result["document_id"],
                    "source_type": result["source_type"]
                }
            }
            for result in data["results"]
        ],
        "total_results": data["total_results"],
        "processing_time_ms": data["processing_time_ms"]
    }
```

#### Phase 2: Enable Graph Features
Gradually enable graph-augmented features:

```python
def migrate_enhanced(query, max_results=10):
    response = requests.post("http://localhost:8000/api/v2/query", json={
        "query": query,
        "query_type": "entity_focused",  # Use entity-focused queries
        "max_results": max_results,
        "expansion_strategy": "adaptive",
        "include_graph_context": True,   # Enable graph context
        "enable_multi_hop": True
    })
    
    return response.json()
```

#### Phase 3: Full Feature Utilization
Use advanced features for optimal results:

```python
def migrate_advanced(query, max_results=10):
    # Determine query type based on content
    query_type = "multi_hop" if "how" in query.lower() or "why" in query.lower() else "entity_focused"
    
    response = requests.post("http://localhost:8000/api/v2/query", json={
        "query": query,
        "query_type": query_type,
        "max_results": max_results,
        "expansion_strategy": "adaptive",
        "fusion_strategy": "adaptive",
        "include_graph_context": True,
        "include_reasoning_path": True,
        "enable_multi_hop": True,
        "entity_types": ["CONCEPT", "TECHNOLOGY", "PERSON"],
        "min_relevance_score": 0.2
    })
    
    return response.json()
```

### Step 4: Error Handling Updates

#### v1 Error Handling
```python
def handle_v1_errors(response):
    if response.status_code == 500:
        return {"error": "Server error", "results": []}
    return response.json()
```

#### v2 Error Handling
```python
def handle_v2_errors(response):
    if response.status_code == 400:
        error_detail = response.json().get("detail", "Invalid request")
        return {"error": f"Validation error: {error_detail}", "results": []}
    elif response.status_code == 408:
        return {"error": "Query timeout", "results": []}
    elif response.status_code == 503:
        return {"error": "Graph features unavailable, using vector search only", "results": []}
    elif response.status_code == 500:
        return {"error": "Server error", "results": []}
    return response.json()
```

## Testing Your Migration

### Compatibility Test
```python
def test_compatibility():
    test_queries = [
        "What is machine learning?",
        "How does AI work?",
        "Explain neural networks",
        "What are the applications of deep learning?"
    ]
    
    for query in test_queries:
        try:
            # Test v1 endpoint
            v1_response = requests.post("http://localhost:8000/api/v1/query", json={
                "query": query,
                "max_results": 5
            })
            
            # Test v2 endpoint
            v2_response = requests.post("http://localhost:8000/api/v2/query", json={
                "query": query,
                "query_type": "simple",
                "max_results": 5,
                "include_graph_context": False
            })
            
            print(f"Query: {query}")
            print(f"v1 results: {len(v1_response.json()['results'])}")
            print(f"v2 results: {len(v2_response.json()['results'])}")
            print("---")
            
        except Exception as e:
            print(f"Error testing query '{query}': {e}")
```

### Performance Comparison
```python
import time

def compare_performance():
    query = "What is machine learning?"
    
    # Test v1 performance
    start_time = time.time()
    v1_response = requests.post("http://localhost:8000/api/v1/query", json={
        "query": query,
        "max_results": 10
    })
    v1_time = time.time() - start_time
    
    # Test v2 performance
    start_time = time.time()
    v2_response = requests.post("http://localhost:8000/api/v2/query", json={
        "query": query,
        "query_type": "simple",
        "max_results": 10
    })
    v2_time = time.time() - start_time
    
    print(f"v1 response time: {v1_time:.3f}s")
    print(f"v2 response time: {v2_time:.3f}s")
    print(f"v2 server processing: {v2_response.json()['processing_time_ms']}ms")
```

## Common Migration Issues

### Issue 1: Field Name Changes
**Problem**: `score` field renamed to `relevance_score`
**Solution**: Update field access in your code

### Issue 2: Response Structure Changes
**Problem**: Additional fields in v2 response
**Solution**: Use `.get()` method for optional fields

### Issue 3: Graph Features Not Available
**Problem**: 503 errors when graph components unavailable
**Solution**: Implement fallback to vector-only search

### Issue 4: Performance Differences
**Problem**: v2 queries may be slower due to graph processing
**Solution**: Use appropriate query types and tune parameters

## Best Practices

1. **Gradual Migration**: Migrate incrementally, testing each phase
2. **Feature Flags**: Use configuration to toggle between v1 and v2
3. **Error Handling**: Implement robust error handling for new error codes
4. **Performance Monitoring**: Monitor response times and adjust parameters
5. **Testing**: Thoroughly test with your actual queries and data
6. **Documentation**: Update your API documentation and client libraries

## Support

- **Migration Guide API**: `GET /api/v1/migration-guide`
- **Health Check**: `GET /api/v1/health` (shows deprecation status)
- **API Documentation**: `/docs` (interactive Swagger UI)
- **Legacy Support**: v1 endpoints remain available until end-of-support date

## Rollback Plan

If you encounter issues with v2:

1. **Immediate Rollback**: Switch back to v1 endpoints
2. **Identify Issues**: Use logs and error messages to diagnose problems
3. **Gradual Re-migration**: Address issues and migrate again incrementally
4. **Contact Support**: Reach out for assistance with complex migration issues
