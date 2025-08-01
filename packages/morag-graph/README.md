# MoRAG Graph

Graph-augmented RAG components for MoRAG (Multimodal RAG Ingestion Pipeline).

## Overview

The `morag-graph` package provides graph database integration and LLM-based entity and relation extraction capabilities for the MoRAG system. It enables knowledge graph construction from documents and graph-guided retrieval to enhance RAG performance.

This package has been optimized to create **document-agnostic knowledge graphs** by removing document-specific metadata from entities and relations, making the extracted knowledge more generic and reusable across different contexts.

## Features

- **LangExtract Integration**: Powered by LangExtract for state-of-the-art entity and relation extraction
- **Domain-Specific Extraction**: Specialized extraction for medical, technical, legal, and other domains
- **Multi-Language Support**: Extract entities and relations from text in multiple languages
- **Source Grounding**: Precise tracking of extraction sources with character-level accuracy
- **Confidence Scoring**: Built-in confidence assessment for all extractions
- **Visualization**: HTML-based visualization of extraction results and knowledge graphs
- Neo4J graph database integration
- **Type abstraction**: Automatic reduction of overly specific types for better graph connectivity
- Dynamic schema evolution
- Graph construction pipeline
- Graph traversal utilities
- Production-ready ingestion scripts

## Installation

```bash
pip install morag-graph
```

## Dependencies

- Neo4J (>=5.15.0)
- LangExtract API access (Google Gemini or compatible LLM)
- morag-core package

## Usage

### Basic Entity and Relation Extraction

```python
from morag_graph.models import Entity, Relation, Graph
from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.storage import Neo4jStorage

# LangExtract-based extraction with domain specialization
entity_extractor = EntityExtractor(domain="medical")
relation_extractor = RelationExtractor(domain="medical")

# Extract entities and relations
text = "Dr. Smith prescribed aspirin to treat the patient's headache."
entities = await entity_extractor.extract(text, source_doc_id="doc_1")
relations = await relation_extractor.extract(text, entities=entities, source_doc_id="doc_1")

# Store in Neo4J
storage = Neo4jStorage(uri="neo4j://localhost:7687", auth=("neo4j", "password"))
await storage.store_entities(entities)
await storage.store_relations(relations)
for relation in relations:
    await storage.store_relation(relation)
```



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

### Intention-Based Extraction

The new intention-based extraction system reduces type fragmentation by:

1. **Document Intention Analysis**: Automatically generates a concise intention summary that captures the document's primary purpose and domain
2. **Guided Type Abstraction**: Uses the intention to guide the LLM toward more abstract, domain-appropriate entity and relation types
3. **Reduced Fragmentation**: Prevents overly specific types like "IS_CEO", "IS_CTO", "IS_CFO" in favor of unified types like "IS_MEMBER"

**Example Usage:**

```python
from morag_graph.extraction import EntityExtractor, RelationExtractor
from morag_graph.extraction.base import LLMConfig

# Configure LLM
config = LLMConfig(provider="gemini", api_key="your-api-key")

# Create extractors
entity_extractor = EntityExtractor(config=config)
relation_extractor = RelationExtractor(config=config)

# Extract with intention context
text = "John Smith is the CEO of TechCorp..."
intention = "Document explaining the structure of the organization/company"

entities = await entity_extractor.extract(text, intention=intention)
relations = await relation_extractor.extract(text, entities=entities, intention=intention)
```

**Benefits:**
- Fewer, more meaningful entity and relation types
- Better graph connectivity through type abstraction
- Domain-aware extraction that respects document context
- Maintained LLM flexibility with better guidance

See `examples/intention_based_extraction.py` for a complete demonstration.

## License

MIT