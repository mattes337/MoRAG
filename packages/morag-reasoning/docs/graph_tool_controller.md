# Graph Tool Controller for Gemini Function Calling

This document describes the implementation of the `GraphToolController` class, which provides a structured interface for Gemini function calling with graph operations in the MORAG system.

## Overview

The `GraphToolController` implements a secure, policy-driven approach to graph operations that can be exposed to Gemini's function calling capabilities. It provides five core graph operations with built-in safety limits and structured citation formats.

## Architecture

### Core Components

1. **ToolCall/ToolResult Data Classes**: Structured request/response handling
2. **Function Specifications**: JSON Schema definitions for Gemini integration
3. **Policy Enforcement**: Whitelist-based tool access control
4. **Limit Enforcement**: Configurable limits for safety and performance
5. **Action Tracing**: Complete audit trail of all operations

### Supported Operations

| Operation | Purpose | Limits |
|-----------|---------|--------|
| `extract_entities` | Extract entities from text | Max entities per call |
| `match_entity` | Find entities in graph | Confidence scoring |
| `expand_neighbors` | Traverse graph relationships | Max depth, max neighbors |
| `fetch_chunk` | Retrieve entity-associated content | Max chunks per entity |
| `extract_facts` | Extract structured facts | Score threshold filtering |

## Usage

### Basic Setup

```python
from morag_reasoning.graph_tool_controller import GraphToolController, ToolCall

# Initialize with custom limits
controller = GraphToolController(
    max_hops=3,
    score_threshold=0.7,
    max_entities_per_call=10,
    max_neighbors_per_entity=5,
    max_chunks_per_entity=3
)

# Get function specifications for Gemini
function_specs = controller.get_function_specs()
```

### Function Specifications

The controller provides JSON Schema specifications that can be directly used with Gemini's function calling:

```python
# Example function spec for extract_entities
{
    "name": "extract_entities",
    "description": "Extract named entities from text using graph-aware fact extraction",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to extract entities from"
            }
        },
        "required": ["text"]
    }
}
```

### Tool Execution

```python
# Execute a tool call
tool_call = ToolCall(
    name="extract_entities",
    args={"text": "Python is a programming language developed by Guido van Rossum"}
)

result = await controller.handle_tool_call(tool_call)

if result.error:
    print(f"Error: {result.error}")
else:
    print(f"Extracted entities: {result.result['entities']}")
    print(f"Count: {result.result['count']}")
```

## Function Details

### 1. Extract Entities

**Purpose**: Extract named entities from text using graph-aware fact extraction.

**Parameters**:
- `text` (string, required): Text to extract entities from

**Response**:
```json
{
    "entities": ["Python", "programming language", "Guido van Rossum"],
    "count": 3,
    "extraction_method": "graph_fact_extractor"
}
```

**Limits**: Respects `max_entities_per_call` setting.

### 2. Match Entity

**Purpose**: Find and match entities in the knowledge graph.

**Parameters**:
- `name` (string, required): Entity name to search for
- `entity_type` (string, optional): Filter by entity type

**Response**:
```json
{
    "entity_id": "python_programming_123",
    "canonical_name": "Python Programming Language",
    "entity_type": "Technology",
    "confidence": 0.95,
    "properties": {"category": "programming", "paradigm": "multi-paradigm"}
}
```

### 3. Expand Neighbors

**Purpose**: Traverse graph relationships to find connected entities.

**Parameters**:
- `entity_id` (string, required): Starting entity ID
- `depth` (integer, optional): Traversal depth (1-3, default: 1)
- `relation_types` (array, optional): Filter by relationship types

**Response**:
```json
{
    "neighbors": [
        {
            "id": "django_framework_456",
            "name": "Django Framework",
            "type": "Framework",
            "relationship": "built_with",
            "distance": 1,
            "properties": {"category": "web_framework"}
        }
    ],
    "count": 1,
    "depth_used": 1,
    "total_explored": 15
}
```

**Limits**:
- Maximum depth limited by `max_hops`
- Results limited by `max_neighbors_per_entity`

### 4. Fetch Chunk

**Purpose**: Retrieve content chunks associated with an entity.

**Parameters**:
- `entity_id` (string, required): Entity to fetch chunks for
- `chunk_types` (array, optional): Filter by chunk types

**Response**:
```json
{
    "entity_name": "Python Programming Language",
    "chunks": [
        {
            "chunk_id": "chunk_789",
            "content": "Python is a high-level programming language...",
            "source": "[document:python_guide.pdf:1:page=5:section=introduction]",
            "chunk_type": "documentation",
            "relevance_score": 0.92
        }
    ],
    "count": 1
}
```

**Limits**: Results limited by `max_chunks_per_entity`.

### 5. Extract Facts

**Purpose**: Extract structured facts from text with confidence scoring.

**Parameters**:
- `text` (string, required): Text to extract facts from
- `min_confidence` (number, optional): Override default score threshold

**Response**:
```json
{
    "facts": [
        {
            "fact_id": "fact_123",
            "subject": "Python",
            "predicate": "is_a",
            "object": "programming language",
            "confidence": 0.95,
            "source": "[document:extracted_text:0:fact_id=fact_123]",
            "metadata": {"domain": "technology"}
        }
    ],
    "total_extracted": 3,
    "filtered_count": 1,
    "score_threshold": 0.8
}
```

**Limits**: Facts filtered by `score_threshold`.

## Citation Format

All operations return structured citations in the format:
```
[source_type:source_name:source_index:additional_metadata]
```

**Examples**:
- `[document:research.pdf:1:page=15:chapter=2.2]`
- `[audio:interview.mp3:3:timecode=00:15:30]`
- `[document:extracted_text:0:fact_id=fact_123]`
- `[web:example.com:2:section=methodology]`

## Security and Policy

### Tool Whitelist

Only the five approved operations are allowed:
- `extract_entities`
- `match_entity`
- `expand_neighbors`
- `fetch_chunk`
- `extract_facts`

Any attempt to call unauthorized tools results in a `ToolCallError`.

### Configurable Limits

```python
controller = GraphToolController(
    max_hops=2,                    # Maximum graph traversal depth
    score_threshold=0.8,           # Minimum confidence for facts
    max_entities_per_call=5,       # Limit entity extraction results
    max_neighbors_per_entity=3,    # Limit neighbor expansion results
    max_chunks_per_entity=2        # Limit chunk retrieval results
)
```

### Error Handling

All operations return structured results with error information:

```python
result = await controller.handle_tool_call(tool_call)

if result.error:
    # Handle error case
    print(f"Operation failed: {result.error}")
else:
    # Process successful result
    process_result(result.result)
```

## Monitoring and Tracing

### Action Traces

All operations are automatically traced:

```python
# Get execution history
traces = controller.get_action_traces()

for trace in traces:
    print(f"Tool: {trace.tool_name}")
    print(f"Args: {trace.args}")
    print(f"Time: {trace.execution_time}s")
    if trace.error:
        print(f"Error: {trace.error}")
```

### Statistics

```python
# Get usage statistics
stats = controller.get_stats()

print(f"Total calls: {stats['total_calls']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Tool usage: {stats['tool_counts']}")
print(f"Total execution time: {stats['total_execution_time']:.2f}s")
```

## Integration with Gemini

### Function Registration

```python
# Get function specs for Gemini
function_specs = controller.get_function_specs()

# Register with Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    tools=function_specs
)
```

### Handling Function Calls

```python
async def handle_gemini_function_call(function_call):
    """Handle function call from Gemini."""
    tool_call = ToolCall(
        name=function_call.name,
        args=dict(function_call.args)
    )

    result = await controller.handle_tool_call(tool_call)

    if result.error:
        return {"error": result.error}
    else:
        return result.result
```

### Example Conversation Flow

```python
# User: "What entities are related to Python programming?"

# 1. Extract entities from query
extract_call = ToolCall(
    name="extract_entities",
    args={"text": "Python programming"}
)
extract_result = await controller.handle_tool_call(extract_call)

# 2. Match main entity
match_call = ToolCall(
    name="match_entity",
    args={"name": "Python"}
)
match_result = await controller.handle_tool_call(match_call)

# 3. Expand to find related entities
if match_result.result["entity_id"]:
    expand_call = ToolCall(
        name="expand_neighbors",
        args={
            "entity_id": match_result.result["entity_id"],
            "depth": 2
        }
    )
    expand_result = await controller.handle_tool_call(expand_call)

    # Return related entities with structured citations
    return expand_result.result["neighbors"]
```

## Testing

Comprehensive tests are provided in `tests/test_graph_tool_controller.py`:

```bash
# Run tests
pytest tests/test_graph_tool_controller.py -v

# Run with coverage
pytest tests/test_graph_tool_controller.py --cov=morag_reasoning.graph_tool_controller
```

## Configuration

### Environment Variables

```bash
# Optional: Override default limits
export MORAG_MAX_GRAPH_HOPS=3
export MORAG_SCORE_THRESHOLD=0.7
export MORAG_MAX_ENTITIES_PER_CALL=10
```

### Initialization Options

```python
# Development configuration (more permissive)
dev_controller = GraphToolController(
    max_hops=3,
    score_threshold=0.6,
    max_entities_per_call=20,
    max_neighbors_per_entity=10,
    max_chunks_per_entity=5
)

# Production configuration (more restrictive)
prod_controller = GraphToolController(
    max_hops=2,
    score_threshold=0.8,
    max_entities_per_call=5,
    max_neighbors_per_entity=3,
    max_chunks_per_entity=2
)
```

## Performance Considerations

1. **Limit Configuration**: Adjust limits based on expected load and response time requirements
2. **Caching**: Consider implementing caching for frequently accessed entities
3. **Async Operations**: All operations are async for better concurrency
4. **Resource Monitoring**: Use action traces to monitor resource usage

## Future Enhancements

1. **Dynamic Limits**: Adjust limits based on system load
2. **Caching Layer**: Add intelligent caching for graph operations
3. **Batch Operations**: Support batch processing for multiple entities
4. **Advanced Filtering**: More sophisticated filtering options
5. **Real-time Updates**: Support for real-time graph updates

## Troubleshooting

### Common Issues

1. **Service Initialization Failures**: Check that required services (graph store, fact extractor) are properly configured
2. **Limit Exceeded Errors**: Adjust limits or optimize queries
3. **Citation Format Issues**: Ensure all sources follow the structured citation format
4. **Performance Issues**: Monitor execution times and adjust limits accordingly

### Debug Mode

```python
# Enable detailed logging
import logging
logging.getLogger('morag_reasoning.graph_tool_controller').setLevel(logging.DEBUG)

# Check action traces for debugging
traces = controller.get_action_traces()
for trace in traces:
    if trace.error:
        print(f"Failed operation: {trace.tool_name} - {trace.error}")
```
