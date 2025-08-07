# MoRAG LLM Development Guide

## Mission Statement

**MoRAG (Modular Retrieval Augmented Generation)** is a comprehensive, domain-agnostic system for processing and indexing multimodal content to enable intelligent retrieval-augmented generation. Our mission is to create a flexible, scalable platform that can adapt to any domain without hardcoded assumptions, using AI-driven dynamic type generation and semantic understanding.

## Vision

To build the world's most adaptable RAG system that:
- **Learns and adapts** to any domain through LLM-driven analysis
- **Processes any content type** (documents, audio, video, web, images) uniformly
- **Scales horizontally** through modular architecture
- **Maintains semantic coherence** across multimodal content
- **Enables intelligent reasoning** through graph-augmented retrieval

## Core Principles

### 1. Domain Agnosticism
- **NO hardcoded domain-specific labels, types, or classifications**
- **LLM determines** all entity types, relationship types, and categorizations dynamically
- **Generic architecture** that adapts to medical, legal, technical, financial, or any other domain
- **Dynamic type generation** based on content analysis, not predefined schemas

### 2. AI-First Design
- **LLM-driven decisions** for entity extraction, relationship determination, and categorization
- **Structured outputs** using Pydantic models for validation and consistency
- **Fallback mechanisms** when AI services are unavailable
- **Confidence scoring** for all AI-generated content

### 3. Modular Architecture
- **Independent packages** for each content type (audio, video, document, web, image)
- **Service-oriented design** with clear interfaces and dependency injection
- **Horizontal scalability** through containerized microservices
- **Loose coupling** between components

### 4. Semantic Intelligence
- **Graph-augmented RAG** combining vector search with knowledge graph traversal
- **Multi-hop reasoning** across entities and relationships
- **Context-aware chunking** with semantic boundaries
- **Source attribution** with detailed metadata tracking

## Development Guidelines

### Entity and Relationship Extraction

#### ✅ DO: Dynamic Type Generation
```python
# LLM determines types based on content
entity_extractor = EntityExtractor(config)  # No predefined types
relation_extractor = RelationExtractor(config)  # No predefined types

# LLM generates context-appropriate types like:
# - PHARMACEUTICAL_DRUG, MEDICAL_CONDITION (medical domain)
# - PROGRAMMING_LANGUAGE, SOFTWARE_FRAMEWORK (technical domain)
# - LEGAL_STATUTE, COURT_CASE (legal domain)
```

#### ❌ DON'T: Hardcoded Domain Types
```python
# NEVER do this - hardcoded domain-specific types
MEDICAL_TYPES = {"DRUG": "...", "DISEASE": "..."}
LEGAL_TYPES = {"STATUTE": "...", "CASE": "..."}
```

### Graph Database Design

#### ✅ DO: Generic Graph Structure
```python
# Generic relationships determined by LLM
Document → CONTAINS → DocumentChunk → MENTIONS → Entity
Entity ← SUBJECT ← Relation → OBJECT → Entity
Relation → EXTRACTED_FROM → DocumentChunk

# LLM-generated relationship types:
# TREATS, CAUSES, PREVENTS (medical)
# IMPLEMENTS, EXTENDS, USES (technical)
# REGULATES, APPLIES_TO, SUPERSEDES (legal)
```

#### ❌ DON'T: Domain-Specific Schema
```python
# NEVER hardcode domain-specific relationships
MEDICAL_RELATIONS = ["TREATS", "CAUSES"]
TECH_RELATIONS = ["IMPLEMENTS", "EXTENDS"]
```

### Content Processing

#### ✅ DO: Uniform Processing Pipeline
```python
# Same pipeline for all content types
async def process_content(content: Any, content_type: ContentType) -> ProcessingResult:
    # 1. Convert to markdown (universal format)
    # 2. Extract entities and relations (LLM-driven)
    # 3. Generate embeddings
    # 4. Store in vector database
    # 5. Build knowledge graph
```

#### ✅ DO: Semantic Chunking
```python
# Intelligent chunking based on content structure
chunking_strategies = {
    "SEMANTIC": "LLM-driven semantic boundaries",
    "CHAPTER": "Document structure-aware",
    "TOPIC": "Topic-based with timestamps (audio/video)",
    "PAGE": "Page-level for PDFs"
}
```

### API Design

#### ✅ DO: Consistent Endpoints
```python
# Processing endpoints (immediate results)
POST /process/file
POST /process/url
POST /process/youtube

# Ingestion endpoints (background + vector storage)
POST /api/v1/ingest/file
POST /api/v1/ingest/url
POST /api/v1/ingest/batch
```

#### ✅ DO: Flexible Configuration
```python
# Support multiple databases and configurations
databases = [
    {"type": "qdrant", "collection": "documents"},
    {"type": "neo4j", "database": "knowledge_graph"}
]
```

## Key Components

### 1. Core Services
- **MoRAGServices**: Unified service orchestration
- **GeminiEmbeddingService**: Text embeddings and AI processing
- **QdrantVectorStorage**: Vector similarity search
- **Neo4jGraphStorage**: Knowledge graph storage

### 2. Content Processors
- **DocumentService**: PDF, DOCX, PPTX, XLSX processing
- **AudioService**: Transcription with speaker diarization
- **VideoService**: Video processing with optional thumbnails
- **WebService**: HTML scraping and processing
- **ImageService**: OCR and vision analysis

### 3. Graph Components
- **EntityExtractor**: Dynamic entity extraction
- **RelationExtractor**: Dynamic relationship extraction
- **FactProcessingService**: Structured fact extraction
- **RecursiveFactRetrieval**: Multi-hop reasoning

## Environment Configuration

### Required Variables
```bash
# AI Services
GEMINI_API_KEY=your_gemini_api_key

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=morag_documents

# Graph Database (optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### Optional Overrides
```bash
# Model Configuration
WHISPER_MODEL_SIZE=large-v3
MORAG_SPACY_MODEL=en_core_web_sm
MORAG_DEFAULT_CHUNK_SIZE=4000

# Feature Toggles
MORAG_ENABLE_SPEAKER_DIARIZATION=true
MORAG_ENABLE_TOPIC_SEGMENTATION=true
```

## Testing Philosophy

### Test-Driven Development
- **Write tests first** for new functionality
- **Validate AI outputs** with structured assertions
- **Test error conditions** and fallback mechanisms
- **Integration tests** for end-to-end workflows

### Example Test Structure
```python
def test_dynamic_entity_extraction():
    """Test LLM generates appropriate entity types."""
    extractor = EntityExtractor(config)  # No predefined types
    
    medical_text = "Aspirin treats headaches and reduces inflammation."
    entities = await extractor.extract(medical_text)
    
    # LLM should generate medical-specific types
    assert any(e.type in ["MEDICATION", "PHARMACEUTICAL_DRUG"] for e in entities)
    assert any(e.type in ["SYMPTOM", "MEDICAL_CONDITION"] for e in entities)
```

## Performance Considerations

### Optimization Strategies
- **Batch processing** for embeddings and AI calls
- **Caching** for repeated content
- **Async processing** for I/O operations
- **GPU acceleration** for audio/video processing

### Scalability Patterns
- **Horizontal scaling** through Docker containers
- **Queue-based processing** with Celery
- **Database sharding** for large datasets
- **CDN integration** for static content

## Error Handling

### Graceful Degradation
- **Fallback to CPU** when GPU unavailable
- **Vector-only search** when graph unavailable
- **Rule-based extraction** when LLM fails
- **Retry mechanisms** with exponential backoff

### Comprehensive Logging
```python
logger.info(f"Processing {content_type} content: {source}")
logger.debug(f"LLM generated {len(entities)} entities, {len(relations)} relations")
logger.warning(f"Fallback to rule-based extraction: {error}")
logger.error(f"Critical failure in {component}: {error}")
```

## Commit Message Guidelines

Always end responses with a git commit message for the complete thread:

```
feat: implement dynamic entity extraction with LLM-driven type generation

- Remove hardcoded entity and relation types
- Enable LLM to determine optimal types based on content
- Add comprehensive test suite for dynamic extraction
- Update documentation with domain-agnostic principles
```

---

**Remember**: MoRAG is designed to be universally adaptable. When in doubt, choose the more generic, LLM-driven approach over hardcoded domain-specific solutions.
