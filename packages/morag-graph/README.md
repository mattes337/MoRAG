# MoRAG Graph

Graph-augmented RAG components for MoRAG (Multimodal RAG Ingestion Pipeline).

## 🚨 Important Migration Notice

**Traditional entity/relation extraction has been completely removed** in favor of Graphiti's superior episode-based knowledge representation.

- ❌ **Removed**: `EntityExtractor`, `RelationExtractor`, `HybridEntityExtractor`
- ❌ **Removed**: Manual entity/relation extraction workflows
- ✅ **New**: Graphiti-based automatic knowledge graph building
- ✅ **New**: Episode-based knowledge representation with temporal context

**Migration**: Replace all traditional extraction code with Graphiti-based ingestion (see examples below).

## Overview

The `morag-graph` package provides graph database integration and LLM-based entity and relation extraction capabilities for the MoRAG system. It enables knowledge graph construction from documents and graph-guided retrieval to enhance RAG performance.

This package has been optimized to create **document-agnostic knowledge graphs** by removing document-specific metadata from entities and relations, making the extracted knowledge more generic and reusable across different contexts.

## Features

- Neo4J graph database integration
- LLM-based entity and relation extraction with context-aware relation type detection
- **Intention-based extraction**: Document intention analysis for guided type abstraction
- Document-agnostic entity and relation extraction
- **Type abstraction**: Automatic reduction of overly specific types for better graph connectivity
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

### Graphiti Knowledge Graph Ingestion (Recommended)

```python
from morag_graph.graphiti import GraphitiConnectionService, GraphitiConfig
from morag_graph.models import Document

# Modern Graphiti-based approach
config = GraphitiConfig(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    openai_api_key="your-api-key"
)

connection_service = GraphitiConnectionService(config)
await connection_service.connect()

# Create episode with automatic entity extraction
await connection_service.create_episode(
    name="Document Title",
    content="Your text content here",
    source_description="Document ingestion"
)

# Enhanced hybrid extraction (AI + Pattern Matching)
hybrid_extractor = HybridEntityExtractor(
    min_confidence=0.7,
    enable_pattern_matching=True,
    pattern_confidence_boost=0.1
)
enhanced_entities = await hybrid_extractor.extract("Your text content here")

# Store in Neo4J
storage = Neo4jStorage(uri="neo4j://localhost:7687", auth=("neo4j", "password"))
for entity in enhanced_entities:
    await storage.store_entity(entity)
for relation in relations:
    await storage.store_relation(relation)
```

### Hybrid Entity Extraction

The package now includes enhanced entity extraction that combines AI-based extraction with pattern matching for improved accuracy:

```python
from morag_graph.extraction import HybridEntityExtractor, EntityPatternMatcher, EntityPattern, PatternType

# Create hybrid extractor
extractor = HybridEntityExtractor(
    min_confidence=0.6,
    enable_pattern_matching=True,
    pattern_confidence_boost=0.1,  # Boost confidence for pattern matches
    ai_confidence_boost=0.0        # Boost confidence for AI matches
)

# Extract entities with enhanced accuracy
text = "I'm using Python and React to build applications for Microsoft."
entities = await extractor.extract(text)

# Add custom patterns
pattern_matcher = EntityPatternMatcher()
custom_pattern = EntityPattern(
    pattern=r"\bCustomTech\b",
    entity_type="TECHNOLOGY",
    pattern_type=PatternType.REGEX,
    confidence=0.9,
    description="Custom technology pattern"
)
pattern_matcher.add_pattern(custom_pattern)

# Get extraction statistics
stats = extractor.get_extraction_stats()
print(f"Using {stats['pattern_count']} patterns")
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

## Key Improvements

### Document-Agnostic Extraction

Entities and relations no longer include document-specific attributes like:
- `source_text` (document excerpts)
- `source_doc_id` (document identifiers)
- `chunk_index` and `total_chunks` (document chunking metadata)

This makes the knowledge graph more generic and reusable across different contexts.

### Dynamic Relation Types

The system uses LLM-based dynamic relation type detection to automatically generate appropriate relation types based on textual context. Common relation types that may be extracted include:
- Employment relations: `WORKS_FOR`, `EMPLOYED_BY`
- Organizational relations: `FOUNDED`, `OWNS`, `PART_OF`
- Location relations: `LOCATED_IN`, `BASED_IN`
- Technical relations: `USES`, `DEVELOPS`, `IMPLEMENTS`
- General relations: `RELATED_TO`, `ASSOCIATED_WITH`

All relation types are determined dynamically by the LLM based on the content being processed.

### Graphiti Episode-Based Knowledge Representation

Graphiti provides a modern approach to knowledge graph construction using episodes:

1. **Episode-Based Storage**: Content is stored as episodes with temporal context
2. **Automatic Entity Extraction**: Built-in LLM-based entity and relation extraction
3. **Deduplication**: Automatic detection and merging of similar content
4. **Hybrid Search**: Combines semantic and keyword search capabilities

**Example Usage:**

```python
from morag_graph.graphiti import GraphitiConnectionService, GraphitiConfig

# Configure Graphiti
config = GraphitiConfig(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
    openai_api_key="your-api-key"
)

# Create connection service
connection_service = GraphitiConnectionService(config)
await connection_service.connect()

# Create episode with automatic entity extraction
await connection_service.create_episode(
    name="Company Structure Document",
    content="John Smith is the CEO of TechCorp...",
    source_description="Organizational chart document"
)
```

**Benefits:**
- Automatic entity and relation extraction
- Temporal knowledge representation
- Built-in deduplication and consistency
- Simplified workflow with single API calls

See the CLI scripts for complete examples of Graphiti usage.

## License

MIT