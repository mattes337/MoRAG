# MoRAG Reasoning

Multi-hop reasoning capabilities for MoRAG (Multimodal RAG Ingestion Pipeline).

## Overview

The `morag-reasoning` package provides advanced multi-hop reasoning capabilities that enable the system to perform complex reasoning across multiple entities and relationships in the knowledge graph. It uses LLM-guided path selection and iterative context refinement to answer complex queries that require connecting information from multiple sources.

## Features

- **LLM-Guided Path Selection**: Uses Large Language Models to intelligently select the most relevant reasoning paths
- **Multiple Reasoning Strategies**: Supports forward chaining, backward chaining, and bidirectional search
- **Iterative Context Refinement**: Automatically identifies gaps in context and retrieves additional information
- **Context Gap Analysis**: Analyzes current context to determine sufficiency for answering queries
- **Fallback Mechanisms**: Robust operation with fallback strategies when LLM calls fail

## Installation

```bash
pip install morag-reasoning
```

## Dependencies

- Python 3.11+
- morag-core package
- morag-graph package
- LLM API access (Gemini, OpenAI, etc.)

## Usage

### Basic Multi-Hop Reasoning

```python
from morag_reasoning import PathSelectionAgent, ReasoningPathFinder, IterativeRetriever
from morag_reasoning.llm import LLMClient

# Initialize components
llm_client = LLMClient(provider="gemini", api_key="your-api-key")
path_selector = PathSelectionAgent(llm_client)
path_finder = ReasoningPathFinder(graph_engine, path_selector)
iterative_retriever = IterativeRetriever(llm_client, graph_engine, vector_retriever)

# Find reasoning paths
query = "How are Apple's AI research efforts related to their partnership with universities?"
start_entities = ["Apple Inc.", "AI research"]

reasoning_paths = await path_finder.find_reasoning_paths(
    query, start_entities, strategy="bidirectional"
)

# Refine context iteratively
initial_context = RetrievalContext(
    paths=[path.path for path in reasoning_paths[:5]]
)

refined_context = await iterative_retriever.refine_context(
    query, initial_context
)
```

### Path Selection Strategies

The system supports three main reasoning strategies:

1. **Forward Chaining**: Start from query entities and explore forward
2. **Backward Chaining**: Start from potential answers and work backward  
3. **Bidirectional**: Search from both ends and meet in the middle

```python
# Forward chaining (default)
paths = await path_finder.find_reasoning_paths(
    query, start_entities, strategy="forward_chaining"
)

# Backward chaining
paths = await path_finder.find_reasoning_paths(
    query, start_entities, target_entities=["target1", "target2"], 
    strategy="backward_chaining"
)

# Bidirectional search
paths = await path_finder.find_reasoning_paths(
    query, start_entities, target_entities=["target1", "target2"],
    strategy="bidirectional"
)
```

### Context Refinement

The iterative retriever automatically identifies gaps in context and retrieves additional information:

```python
# Create initial context
context = RetrievalContext(
    entities={"Apple": {"type": "ORG"}},
    documents=[{"id": "doc1", "content": "Apple is a technology company"}]
)

# Refine context iteratively
refined_context = await iterative_retriever.refine_context(
    "What products does Apple make?", context
)

# Check final analysis
final_analysis = refined_context.metadata.get('final_analysis')
print(f"Context sufficient: {final_analysis.is_sufficient}")
print(f"Confidence: {final_analysis.confidence}")
```

## Configuration

### LLM Configuration

```python
from morag_reasoning.llm import LLMConfig, LLMClient

config = LLMConfig(
    provider="gemini",
    model="gemini-1.5-flash",
    api_key="your-api-key",
    temperature=0.1,
    max_tokens=2000,
    max_retries=5
)

llm_client = LLMClient(config)
```

### Reasoning Configuration

```python
# Path selection configuration
path_selector = PathSelectionAgent(
    llm_client=llm_client,
    max_paths=10  # Maximum paths to select
)

# Iterative retrieval configuration
iterative_retriever = IterativeRetriever(
    llm_client=llm_client,
    graph_engine=graph_engine,
    vector_retriever=vector_retriever,
    max_iterations=5,  # Maximum refinement iterations
    sufficiency_threshold=0.8  # Confidence threshold for stopping
)
```

## Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=morag_reasoning

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

## Performance

The system is designed for efficient multi-hop reasoning:

- **Path Selection**: < 3 seconds for 50 candidate paths
- **Context Analysis**: < 2 seconds per iteration
- **Multi-hop Reasoning**: < 10 seconds for complex queries
- **Memory Usage**: < 2GB for large reasoning tasks

## Architecture

The package consists of several key components:

- **LLM Client**: Unified interface for LLM interactions
- **Path Selection Agent**: LLM-guided path selection with multiple strategies
- **Reasoning Path Finder**: Path discovery and selection coordination
- **Iterative Retriever**: Context refinement with gap analysis
- **Models**: Data structures for reasoning contexts and results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.
