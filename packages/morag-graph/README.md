# MoRAG Graph

Graph-augmented RAG components for MoRAG (Multimodal RAG Ingestion Pipeline).

## Overview

The `morag-graph` package provides graph database integration and LLM-based entity and relation extraction capabilities for the MoRAG system. It enables knowledge graph construction from documents and graph-guided retrieval to enhance RAG performance.

This package has been optimized to create **document-agnostic knowledge graphs** by removing document-specific metadata from entities and relations, making the extracted knowledge more generic and reusable across different contexts.

## Features

- Neo4J graph database integration
- LLM-based entity and relation extraction with context-aware relation type detection
- Document-agnostic entity and relation extraction
- Enhanced relation types (PLAYED_ROLE, PORTRAYED, PRACTICES, ENGAGED_IN, STUDIED)
- Dynamic schema evolution
- JSON-first development approach
- Graph construction pipeline
- Graph traversal utilities
- Hybrid retrieval system
- Production-ready ingestion scripts

## Installation

```bash
pip install morag-graph
```

## Dependencies

- Neo4J (>=5.15.0)
- Google Gemini API access for LLM-based extraction
- morag-core package

## Usage

### Basic Entity and Relation Extraction

```python
from morag_graph.models import Entity, Relation, Graph
from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.storage import Neo4jStorage

# Extract entities and relations from text
entity_extractor = EntityExtractor()
relation_extractor = RelationExtractor()

entities = await entity_extractor.extract("Your text content here")
relations = await relation_extractor.extract("Your text content here", entities)

# Store in Neo4J
storage = Neo4jStorage(uri="neo4j://localhost:7687", auth=("neo4j", "password"))
for entity in entities:
    await storage.store_entity(entity)
for relation in relations:
    await storage.store_relation(relation)
```

### Command Line Scripts

#### Extract Knowledge Graph from Text

```bash
# Extract entities and relations from a text file
python run_extraction.py input.txt --output extracted_data.json
```

#### Ingest Data into Neo4j

```bash
# Ingest extracted data into Neo4j
python run_ingestion.py extracted_data.json --neo4j-password your_password --clear
```

#### Convert Legacy CUSTOM Relations

```bash
# Convert CUSTOM relations to meaningful types
python convert_custom_relations.py extracted_data.json
```

## Key Improvements

### Document-Agnostic Extraction

Entities and relations no longer include document-specific attributes like:
- `source_text` (document excerpts)
- `source_doc_id` (document identifiers)
- `chunk_index` and `total_chunks` (document chunking metadata)

This makes the knowledge graph more generic and reusable across different contexts.

### Enhanced Relation Types

New relation types have been added to replace generic `CUSTOM` relations:
- `PLAYED_ROLE`: Person -> Role/Character
- `PORTRAYED`: Person -> Character/Role  
- `PRACTICES`: Person -> Activity/Belief
- `ENGAGED_IN`: Person -> Activity
- `STUDIED`: Person -> Subject/Field

The system now uses context-aware detection to automatically suggest appropriate relation types based on textual context.

### Improved Fallback Strategy

Instead of defaulting to `CUSTOM` relations, the system now:
1. Attempts context-based relation type detection
2. Falls back to `RELATED_TO` for better semantic meaning

## License

MIT