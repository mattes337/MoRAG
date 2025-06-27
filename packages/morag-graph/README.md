# MoRAG Graph

Graph-augmented RAG components for MoRAG (Multimodal RAG Ingestion Pipeline).

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

### Basic Entity and Relation Extraction

```python
from morag_graph.models import Entity, Relation, Graph
from morag_graph.extraction import EntityExtractor, RelationExtractor, HybridEntityExtractor
from morag_graph.storage import Neo4jStorage

# Standard AI-based extraction
entity_extractor = EntityExtractor()
relation_extractor = RelationExtractor()

entities = await entity_extractor.extract("Your text content here")
relations = await relation_extractor.extract("Your text content here", entities)

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

### Enhanced Relation Types

New relation types have been added to replace generic `CUSTOM` relations:
- `PLAYED_ROLE`: Person -> Role/Character
- `PORTRAYED`: Person -> Character/Role  
- `PRACTICES`: Person -> Activity/Belief
- `ENGAGED_IN`: Person -> Activity
- `STUDIED`: Person -> Subject/Field

The system now uses context-aware detection to automatically suggest appropriate relation types based on textual context.

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