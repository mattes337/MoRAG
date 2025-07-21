# MoRAG Context Generation Testing Script

This script tests the context generation for LLM calls using agentic AI to find entities, navigate through the knowledge graph, and execute prompts with the generated context.

## Overview

The `test-context-generation.py` script implements an agentic AI system that:

1. **Entity Extraction**: Uses LLM-based entity extraction to identify key entities from the input prompt
2. **Graph Navigation**: Intelligently navigates through the knowledge graph to find related entities and relationships
3. **Vector Search**: Searches for relevant documents using vector similarity
4. **Context Scoring**: Evaluates the quality and relevance of the generated context
5. **Response Generation**: Executes the prompt with all gathered context to generate a comprehensive response

## Features

- **Agentic AI Approach**: Uses LLM reasoning to guide entity extraction and graph traversal
- **Multi-Modal Context**: Combines graph knowledge, vector search, and reasoning paths
- **Quality Scoring**: Provides context quality metrics
- **Detailed Logging**: Shows all processing steps when verbose mode is enabled
- **Flexible Output**: Supports JSON export and context visualization

## Usage

### Basic Usage

```bash
# Test with both Neo4j and Qdrant
python test-context-generation.py --neo4j --qdrant "How does nutrition affect ADHD symptoms?"

# Test with only Neo4j (graph operations)
python test-context-generation.py --neo4j "What are the connections between AI and healthcare?"

# Test with only Qdrant (vector search)
python test-context-generation.py --qdrant "Explain machine learning applications"
```

### Advanced Options

```bash
# Verbose mode with context display
python test-context-generation.py --neo4j --qdrant --verbose --show-context "Complex query here"

# Save results to JSON file
python test-context-generation.py --neo4j --qdrant --output results.json "Your query"

# Use specific model and token limit
python test-context-generation.py --neo4j --model gemini-1.5-pro --max-tokens 8000 "Your query"
```

### Command Line Options

- `--neo4j`: Enable Neo4j graph operations
- `--qdrant`: Enable Qdrant vector search
- `--model MODEL`: Specify LLM model (default: from env or gemini-1.5-flash)
- `--max-tokens N`: Maximum tokens for LLM responses (default: 4000)
- `--verbose`: Show detailed processing steps
- `--show-context`: Display the generated context details
- `--output FILE`: Save results to JSON file

## Output

The script provides comprehensive output including:

### Summary Metrics
- Context quality score (0.0-1.0)
- Processing time
- Number of entities extracted
- Number of graph entities and relations found
- Number of vector chunks retrieved

### Context Details (with --show-context)
- Extracted entities with types and relevance
- Graph entities and their relationships
- Vector document chunks
- Reasoning paths

### Final Response
- LLM-generated response using all gathered context

### JSON Export (with --output)
Complete results including all context data, processing steps, and performance metrics.

## How It Works

### 1. Entity Extraction
The script uses an LLM-based approach to extract entities from the input prompt:
- Analyzes the query to identify key concepts
- Extracts 3-7 most relevant entities
- Provides entity types and relevance explanations
- Normalizes entity names using LLM-based normalization

### 2. Graph Navigation
For each extracted entity, the system:
- Searches for matching entities in the Neo4j graph
- Finds related entities and relationships
- Uses LLM reasoning to identify important reasoning paths
- Limits results to maintain performance

### 3. Vector Search
- Searches Qdrant for documents relevant to the prompt
- Retrieves top matching chunks with similarity scores
- Includes metadata and source information

### 4. Context Scoring
- Uses LLM evaluation to score context quality
- Considers relevance, completeness, and usefulness
- Falls back to heuristic scoring if LLM is unavailable

### 5. Response Generation
- Combines all context sources into a comprehensive prompt
- Uses LLM to synthesize information and generate response
- Maintains context quality awareness in the response

## Requirements

- MoRAG components installed
- Neo4j database (optional, for graph operations)
- Qdrant vector database (optional, for vector search)
- Gemini API key for LLM operations
- Environment variables configured

## Environment Setup

Ensure these environment variables are set:
```bash
GEMINI_API_KEY=your_gemini_api_key
MORAG_GEMINI_MODEL=gemini-1.5-flash  # optional
NEO4J_URI=bolt://localhost:7687       # for Neo4j
NEO4J_USERNAME=neo4j                  # for Neo4j
NEO4J_PASSWORD=your_password          # for Neo4j
QDRANT_URL=http://localhost:6333      # for Qdrant
```

## Example Output

```
🚀 Testing context generation for: 'How does nutrition affect ADHD symptoms?'
📊 Databases: Neo4j=True, Qdrant=True
🤖 Model: gemini-1.5-flash
✅ Neo4j connection established
✅ Qdrant connection established

================================================================================
🎯 CONTEXT GENERATION RESULTS
================================================================================
📊 Context Quality Score: 0.85/1.0
⚡ Processing Time: 3.42s
🔍 Entities Extracted: 5
🕸️  Graph Entities: 12
🔗 Graph Relations: 18
📄 Vector Chunks: 7

================================================================================
🎯 FINAL RESPONSE
================================================================================
Nutrition significantly impacts ADHD symptoms through several mechanisms...
[Detailed response with context-based information]

💾 Results saved to: results.json
```

## Error Handling

The script includes comprehensive error handling:
- Database connection failures are handled gracefully
- LLM API errors fall back to alternative approaches
- Individual component failures don't stop the entire process
- Detailed error messages help with troubleshooting

## Performance Considerations

- Entity extraction is limited to 5 entities for performance
- Graph queries are limited to prevent excessive traversal
- Vector search is limited to top 10 results
- Concurrent processing with semaphores for rate limiting
