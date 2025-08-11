# MoRAG Graph

Graph-augmented RAG components for MoRAG (Multimodal RAG Ingestion Pipeline).

## Overview

The `morag-graph` package provides graph database integration and LLM-based fact extraction capabilities for the MoRAG system. It enables knowledge graph construction from documents through structured fact extraction and graph-guided retrieval to enhance RAG performance.

This package uses **structured fact extraction** to create comprehensive knowledge graphs with detailed source attribution, making the extracted knowledge actionable and traceable across different contexts.

## Features

- **Structured Fact Extraction**: Advanced LLM-based extraction of structured facts with subject-object-approach-solution patterns
- **Domain-Specific Extraction**: Specialized extraction for research, medical, technical, legal, and other domains
- **Multi-Language Support**: Extract facts from text in multiple languages
- **Detailed Source Attribution**: Comprehensive metadata preservation including timestamps, chapters, pages, and speaker information
- **Confidence Scoring**: Built-in confidence assessment for all extractions
- **Vector and Graph Storage**: Dual storage in both Neo4j graph database and Qdrant vector database
- **Fact Relationships**: Automatic detection and extraction of relationships between facts
- **Citation Generation**: Machine-readable and human-readable source citations
- Neo4j graph database integration
- Qdrant vector database integration
- Dynamic schema evolution
- Graph construction pipeline
- Graph traversal utilities
- Production-ready ingestion scripts

## Installation

```bash
pip install morag-graph
```

## Dependencies

- Neo4j (>=5.15.0)
- Qdrant (>=1.7.0) - optional for vector storage
- Google Gemini API access (or compatible LLM)
- morag-core package
- morag-services package

## Usage

### Basic Fact Extraction

```python
from morag_graph.models.fact import Fact, FactRelation
from morag_graph.extraction.fact_extractor import FactExtractor
from morag_graph.extraction.fact_graph_builder import FactGraphBuilder
from morag_graph.services.fact_extraction_service import FactExtractionService
from morag_graph.storage.neo4j_storage import Neo4jStorage

# Initialize fact extractor with domain specialization
fact_extractor = FactExtractor(
    model_id="gemini-2.0-flash",
    api_key="your-api-key",
    domain="research",
    min_confidence=0.7,
    max_facts_per_chunk=10
)

# Extract facts from text
text = "Dr. Smith prescribed aspirin to treat the patient's headache. The medication was effective in reducing pain within 30 minutes."

facts = await fact_extractor.extract_facts(
    chunk_text=text,
    chunk_id="chunk_1",
    document_id="doc_1",
    context={
        'domain': 'medical',
        'language': 'en',
        'source_file_name': 'medical_notes.txt'
    }
)

# Extract relationships between facts
fact_graph_builder = FactGraphBuilder(
    model_id="gemini-2.0-flash",
    api_key="your-api-key"
)

fact_graph = await fact_graph_builder.build_fact_graph(facts)
relationships = fact_graph.relationships

# Store in Neo4j and Qdrant
storage = Neo4jStorage(uri="neo4j://localhost:7687", auth=("neo4j", "password"))
fact_service = FactExtractionService(
    neo4j_storage=storage,
    enable_vector_storage=True
)

result = await fact_service.extract_and_store_facts([chunk])
```

### Command Line Scripts

#### Extract Facts from Documents

```bash
# Extract facts from a document with default settings
python cli/fact-extraction.py document.md

# Extract facts for specific domain with higher confidence
python cli/fact-extraction.py research_paper.pdf --domain research --min-confidence 0.8

# Store facts in both Neo4j and Qdrant databases
python cli/fact-extraction.py document.txt --neo4j --qdrant --verbose
```

#### Legacy Entity Extraction (Deprecated)

```bash
# Legacy entity extraction (use fact extraction instead)
python run_extraction.py input.txt --output extracted_data.json

# Legacy ingestion (use fact extraction CLI instead)
python run_ingestion.py extracted_data.json --neo4j-password your_password --clear
```

## Key Features

### Structured Fact Extraction

Facts are extracted using a structured approach that captures:
- **Subject**: The main entity or concept
- **Object**: The target entity or concept
- **Approach**: The method, process, or mechanism
- **Solution**: The outcome, result, or resolution
- **Remarks**: Additional context or qualifications

This structure makes facts more actionable and useful for downstream applications.

### Comprehensive Source Attribution

Facts include detailed source metadata for complete traceability:
- `source_file_path` and `source_file_name` for file identification
- `page_number` and `chapter_title` for document navigation
- `timestamp_start` and `timestamp_end` for audio/video content
- `topic_header` and `speaker_label` for structured content
- `source_text_excerpt` for exact source reference

### Dual Storage Architecture

Facts are stored in both graph and vector databases:
- **Neo4j**: For relationship traversal and graph queries
- **Qdrant**: For semantic similarity search and retrieval
- **Unified Interface**: Single service manages both storage types

### Dynamic Fact Relationships

The system automatically detects relationships between facts:
- **Causal relationships**: `CAUSES`, `LEADS_TO`, `RESULTS_IN`
- **Temporal relationships**: `PRECEDES`, `FOLLOWS`, `CONCURRENT`
- **Logical relationships**: `SUPPORTS`, `CONTRADICTS`, `ELABORATES`
- **Hierarchical relationships**: `PART_OF`, `CONTAINS`, `GENERALIZES`

### Machine-Readable Citations

Facts generate both human-readable and machine-readable citations:
- **Human-readable**: "document.pdf | page 5 | chapter 'Introduction' | timestamp 120.5s"
- **Machine-readable**: "[document.pdf:page_5:Introduction]"

**Example Fact Structure:**

```python
from morag_graph.models.fact import Fact

fact = Fact(
    subject="Aspirin",
    object="headache pain",
    approach="oral administration",
    solution="pain reduction within 30 minutes",
    remarks="effective for mild to moderate pain",
    fact_type="treatment",
    domain="medical",
    extraction_confidence=0.95,
    source_file_name="medical_notes.txt",
    page_number=3,
    chapter_title="Pain Management"
)
# Generate citation for the fact
citation = fact.get_citation()
print(citation)  # "medical_notes.txt | page 3 | chapter 'Pain Management'"

# Generate machine-readable source
source = fact.get_machine_readable_source()
print(source)  # "[medical_notes.txt:page_3:Pain_Management]"
```

**Benefits:**
- Structured, actionable facts instead of loose entities
- Complete source traceability with detailed metadata
- Dual storage for both graph traversal and semantic search
- Automatic relationship detection between facts
- Domain-specific extraction with confidence scoring

## Migration from Entity Extraction

If you're migrating from the old entity extraction approach:

1. **Replace EntityExtractor/RelationExtractor** with `FactExtractor`
2. **Update storage calls** to use `FactExtractionService`
3. **Modify CLI scripts** to use `cli/fact-extraction.py`
4. **Update data models** to work with `Fact` and `FactRelation` objects

See the migration guide in `docs/migration.md` for detailed instructions.

## License

MIT