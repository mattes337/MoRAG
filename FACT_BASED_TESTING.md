# Fact-Based System Testing Guide

This document provides instructions for testing the updated MoRAG system with fact-based retrieval capabilities.

## Overview

The MoRAG system has been updated to use a new fact-based retrieval system that provides:

- **Structured fact extraction** from graph traversal
- **Source attribution** with detailed metadata
- **Recursive graph exploration** with LLM-guided decisions
- **Confidence scoring** and relevance decay
- **Comprehensive fact synthesis** into final answers

## Updated Components

### 1. Enhanced Query Endpoint (`/api/v2/query`)
- **New Parameter**: `use_fact_retrieval=true` enables fact-based retrieval
- **Response**: Includes `facts`, `final_answer`, `traversal_steps`, and LLM call counts
- **Backward Compatible**: Traditional retrieval still available when `use_fact_retrieval=false`

### 2. Intelligent Retrieval Endpoint (`/api/v2/intelligent-query/facts`)
- **New Endpoint**: Dedicated fact-based intelligent retrieval
- **Features**: Entity identification, recursive path following, fact extraction
- **Response**: Structured facts with source attribution

### 3. Reasoning Endpoints
- **New Endpoint**: `/reasoning/query/facts` for full fact-based reasoning
- **Simplified Endpoint**: `/reasoning/query/facts/simple` with fewer parameters
- **Features**: Multi-hop reasoning with fact extraction and synthesis

### 4. CLI Tool (`cli/test-prompt.py`)
- **New Flag**: `--use-fact-retrieval` enables fact-based processing
- **Output**: Structured facts with source information and traversal details
- **Examples**: Updated help text with fact-based examples

## Testing Instructions

### Prerequisites

1. **Environment Setup**:
   ```bash
   # Required environment variables
   export GEMINI_API_KEY="your-gemini-api-key"
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USER="neo4j"
   export NEO4J_PASSWORD="your-password"
   export QDRANT_HOST="localhost"
   export QDRANT_PORT="6333"
   export QDRANT_COLLECTION_NAME="morag_documents"
   ```

2. **Database Setup**:
   - Neo4j instance with ingested graph data
   - Qdrant instance with vector embeddings
   - Both databases should contain related content

3. **Server Running**:
   ```bash
   python -m morag.server
   ```

### Automated Testing

#### 1. API Endpoint Testing
```bash
# Run comprehensive endpoint tests
python test_fact_based_endpoints.py
```

This script tests:
- Enhanced query endpoint with fact retrieval
- Intelligent retrieval facts endpoint
- Reasoning fact-based endpoints
- Health check endpoints

#### 2. CLI Tool Testing
```bash
# Run CLI fact retrieval tests
python test_cli_fact_retrieval.py
```

This script tests:
- CLI script with `--use-fact-retrieval` flag
- Output validation and structure
- Fact extraction and source attribution

### Manual Testing

#### 1. Enhanced Query API
```bash
curl -X POST "http://localhost:8000/api/v2/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the benefits of machine learning?",
    "use_fact_retrieval": true,
    "max_depth": 3,
    "max_total_facts": 20,
    "facts_only": false,
    "language": "en"
  }'
```

Expected response structure:
```json
{
  "query_id": "...",
  "query": "What are the benefits of machine learning?",
  "facts": [
    {
      "fact_text": "Machine learning enables automated pattern recognition...",
      "source_node_id": "node_123",
      "score": 0.95,
      "final_decayed_score": 0.85,
      "source_description": "Document: ml_benefits.pdf"
    }
  ],
  "final_answer": "Machine learning provides several key benefits...",
  "total_nodes_explored": 15,
  "gta_llm_calls": 5,
  "fca_llm_calls": 20,
  "final_llm_calls": 1
}
```

#### 2. CLI Tool Testing
```bash
# Test fact-based retrieval
python cli/test-prompt.py --neo4j --qdrant --use-fact-retrieval \
  "How does artificial intelligence relate to machine learning?"

# Test with output file
python cli/test-prompt.py --neo4j --qdrant --use-fact-retrieval \
  --output results.json \
  "What are the applications of deep learning?"
```

Expected output includes:
- Method: Fact-based retrieval
- Facts extracted: [number]
- Nodes explored: [number]
- Confidence score: [0.0-1.0]
- Processing time and LLM call counts

#### 3. Reasoning API Testing
```bash
# Simple fact-based reasoning
curl -X POST "http://localhost:8000/reasoning/query/facts/simple" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the relationship between AI and machine learning?",
    "max_depth": 2,
    "max_facts": 15,
    "facts_only": false,
    "language": "en"
  }'
```

## Validation Criteria

### 1. Functional Requirements
- ✅ Fact extraction produces structured, actionable facts
- ✅ Source attribution includes detailed metadata
- ✅ Graph traversal follows logical paths
- ✅ Final answers synthesize facts coherently
- ✅ Backward compatibility maintained

### 2. Performance Requirements
- ✅ Response times under 30 seconds for typical queries
- ✅ LLM call counts reasonable (< 50 total calls)
- ✅ Memory usage stable during processing
- ✅ Error handling for timeouts and failures

### 3. Quality Requirements
- ✅ Facts are relevant to the query
- ✅ Source information is accurate and traceable
- ✅ Confidence scores reflect actual relevance
- ✅ Final answers are coherent and comprehensive

## Troubleshooting

### Common Issues

1. **Service Unavailable (503)**
   - Check if RecursiveFactRetrievalService dependencies are installed
   - Verify Neo4j and Qdrant connections
   - Check environment variables

2. **Timeout Errors**
   - Reduce `max_depth` or `max_total_facts`
   - Check database performance
   - Verify LLM API availability

3. **Empty Results**
   - Ensure databases contain relevant data
   - Check entity extraction is working
   - Verify graph connectivity

4. **Invalid JSON Responses**
   - Check for model overload errors
   - Verify API key validity
   - Check server logs for detailed errors

### Debug Information

Enable debug information in requests:
```json
{
  "query": "your query",
  "use_fact_retrieval": true,
  "include_debug_info": true
}
```

This provides additional context about:
- Entity identification process
- Graph traversal decisions
- Fact scoring details
- LLM interaction logs

## Performance Benchmarks

Expected performance for typical queries:

| Query Type | Facts | Nodes | Time | LLM Calls |
|------------|-------|-------|------|-----------|
| Simple     | 5-15  | 3-8   | 5-15s| 10-20     |
| Complex    | 15-30 | 8-20  | 15-30s| 20-40     |
| Deep       | 30-50 | 20-50 | 30-60s| 40-80     |

## Next Steps

After successful testing:

1. **Production Deployment**
   - Update environment configurations
   - Monitor performance metrics
   - Set up alerting for failures

2. **User Training**
   - Document new API parameters
   - Provide usage examples
   - Update client applications

3. **Monitoring**
   - Track fact extraction quality
   - Monitor LLM usage costs
   - Analyze user query patterns
