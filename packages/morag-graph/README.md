# MoRAG Graph

Graph-augmented RAG components for MoRAG (Multimodal RAG Ingestion Pipeline).

## Overview

The `morag-graph` package provides graph database integration and LLM-based entity and relation extraction capabilities for the MoRAG system. It enables knowledge graph construction from documents and graph-guided retrieval to enhance RAG performance.

## Features

- Neo4J graph database integration
- LLM-based entity and relation extraction
- Dynamic schema evolution
- JSON-first development approach
- Graph construction pipeline
- Graph traversal utilities
- Hybrid retrieval system

## Installation

```bash
pip install morag-graph
```

## Dependencies

- Neo4J (>=5.15.0)
- OpenAI API access for LLM-based extraction
- morag-core package

## Usage

```python
from morag_graph.models import Entity, Relation, Graph
from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.storage import Neo4jStorage

# Extract entities and relations from text
extractor = EntityExtractor()
entities = await extractor.extract("Your text content here")

# Store in Neo4J
storage = Neo4jStorage(uri="neo4j://localhost:7687", auth=("neo4j", "password"))
await storage.store_entities(entities)
```

## License

MIT