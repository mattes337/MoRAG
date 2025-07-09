# MoRAG Prompt Testing CLI Script

The `test-prompt.py` script allows testing LLM prompts with multi-hop reasoning and RAG capabilities, showing all LLM interactions step by step.

## Features

- **Detailed LLM Interaction Logging**: Shows all prompts sent to the LLM and responses received
- **Multi-hop Reasoning**: Supports entity extraction and graph traversal for complex queries
- **RAG Integration**: Uses both Neo4j (graph) and Qdrant (vector) databases for retrieval
- **Step-by-Step Execution**: Breaks down the reasoning process into clear steps
- **JSON Output**: Saves detailed results including all LLM interactions to a file
- **Verbose Mode**: Shows real-time LLM prompts and responses during execution

## Usage

### Basic Usage

```bash
# Simple query with vector search
python cli/test-prompt.py --qdrant "What is artificial intelligence?"

# Query with graph database
python cli/test-prompt.py --neo4j "How are companies connected to universities?"

# Full multi-hop reasoning with both databases
python cli/test-prompt.py --neo4j --qdrant --enable-multi-hop "mit welchen lebensmitteln kann ich ADHS eindämmen?"
```

### Advanced Usage

```bash
# Verbose mode to see all LLM interactions in real-time
python cli/test-prompt.py --neo4j --qdrant --enable-multi-hop --verbose "How are Apple's AI efforts connected to universities?"

# Save detailed results to JSON file
python cli/test-prompt.py --qdrant --output results.json "What are the benefits of machine learning?"

# Complex reasoning query
python cli/test-prompt.py --neo4j --qdrant --enable-multi-hop --verbose --output analysis.json "How does climate change affect renewable energy adoption?"
```

## Command Line Options

- `--neo4j`: Use Neo4j for graph operations and entity relationships
- `--qdrant`: Use Qdrant for vector similarity search
- `--enable-multi-hop`: Enable multi-hop reasoning with entity extraction and graph traversal
- `--verbose`: Show detailed LLM interactions in real-time
- `--output FILE`: Save results to JSON file

## Execution Steps

The script follows a structured execution process:

### Step 1: Query Analysis
- Analyzes the prompt to understand intent and type
- Identifies key entities and relationships
- Determines the reasoning strategy needed

### Step 2: Entity Extraction (if multi-hop enabled)
- Extracts 2-5 key entities from the query
- Uses these as starting points for graph traversal

### Step 3: Multi-hop Reasoning (if enabled)
- Finds reasoning paths between entities
- Uses LLM-guided path selection
- Explores relationships in the knowledge graph

### Step 4: Vector Search (if Qdrant enabled)
- Searches for relevant documents using vector similarity
- Retrieves contextual information from the knowledge base

### Step 5: Final Synthesis
- Combines all gathered information
- Generates a comprehensive response using the LLM
- Provides detailed, well-reasoned answers

## Output Format

### Console Output
- Real-time step execution with progress indicators
- LLM prompts and responses (in verbose mode)
- Final comprehensive answer
- Execution summary with timing and interaction counts

### JSON Output (when --output is used)
```json
{
  "results": {
    "prompt": "Your query here",
    "timestamp": "2025-07-09T11:44:27.803992",
    "steps": [
      {
        "step": "query_analysis",
        "result": "Analysis results..."
      }
    ],
    "final_result": "Final comprehensive answer",
    "performance": {
      "total_time_seconds": 22.4,
      "llm_interactions": 2
    }
  },
  "llm_interactions": {
    "total_interactions": 4,
    "total_prompts": 2,
    "total_responses": 2,
    "interactions": [
      {
        "step": 1,
        "type": "prompt",
        "timestamp": "2025-07-09T11:44:28.104843",
        "context": "query_analysis",
        "content": "The actual prompt sent to LLM..."
      },
      {
        "step": 1,
        "type": "response",
        "timestamp": "2025-07-09T11:44:39.128396",
        "context": "query_analysis",
        "content": "The LLM's response..."
      }
    ]
  }
}
```

## Requirements

- MoRAG system components installed
- Environment variables configured:
  - `GEMINI_API_KEY`: Your Gemini API key
  - `MORAG_GEMINI_MODEL`: Model to use (default: gemini-2.5-flash)
  - Database connection settings for Neo4j and/or Qdrant

## Examples

### Example 1: Simple Information Query
```bash
python cli/test-prompt.py --qdrant "What is machine learning?"
```

### Example 2: Multi-hop Reasoning
```bash
python cli/test-prompt.py --neo4j --qdrant --enable-multi-hop "How are technology companies connected to AI research?"
```

### Example 3: German Language Query
```bash
python cli/test-prompt.py --neo4j --qdrant --enable-multi-hop "mit welchen lebensmitteln kann ich ADHS eindämmen?"
```

### Example 4: Complex Analysis with Full Logging
```bash
python cli/test-prompt.py --neo4j --qdrant --enable-multi-hop --verbose --output detailed_analysis.json "How does artificial intelligence impact healthcare innovation?"
```

## Troubleshooting

- Ensure all MoRAG components are properly installed
- Check that database connections are configured correctly
- Verify that the GEMINI_API_KEY environment variable is set
- Use `--verbose` mode to see detailed execution steps for debugging

## Integration with MoRAG

This script demonstrates the full capabilities of the MoRAG system:
- Graph-based knowledge representation (Neo4j)
- Vector similarity search (Qdrant)
- Multi-hop reasoning and path finding
- LLM-guided query processing and synthesis
- Structured output and detailed logging

It serves as both a testing tool and a reference implementation for building applications that leverage MoRAG's advanced reasoning capabilities.
