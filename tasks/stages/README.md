# MoRAG Stage-Based Processing Architecture

## Overview

This document outlines the comprehensive refactoring of MoRAG to implement a stage-based processing architecture. The new system uses named, optional, and selectable processing stages that can be executed individually via CLI or REST endpoints.

## Architecture Goals

### Core Principles
- **Modularity**: Each stage is independent and can be executed separately
- **Reusability**: Intermediate files can be reused across multiple processing runs
- **Flexibility**: Stages can be skipped or executed in different combinations
- **Persistence**: Each stage produces downloadable files that can be stored alongside source content
- **Webhook Integration**: All stage outputs can be pushed to webhook URLs
- **Database Flexibility**: Support for multiple database ingestion targets
- **No Legacy Code**: All old endpoints and code will be completely removed - no backwards compatibility
- **Clean Architecture**: Replace everything, never leave obsolete or deprecated code

### Stage-Based Processing Flow

```
Input File → markdown-conversion → markdown-optimizer → chunker → fact-generator → ingestor
             ↓                   ↓                   ↓         ↓               ↓
           .md                 .opt.md            .chunks   .facts        Database
```

## Processing Stages

### markdown-conversion Stage
**Purpose**: Convert input files to unified markdown format
- **Input**: Video/Audio/Document files, URLs, or text
- **Output**: `{filename}.md` with metadata header
- **Services**: VideoService, AudioService, DocumentService, WebService
- **Features**:
  - Transcription for audio/video with timestamps
  - PDF-to-markdown conversion using docling
  - Web scraping and content extraction
  - Metadata extraction and normalization

### markdown-optimizer Stage (Optional)
**Purpose**: LLM-based text improvement and transcription error correction
- **Input**: `{filename}.md` from markdown-conversion
- **Output**: `{filename}.opt.md` (optimized markdown)
- **Features**:
  - Fix transcription errors and improve readability using system/user message pattern
  - Enhance structure and formatting
  - Preserve timestamps and metadata
  - Configurable LLM prompts for different content types

### chunker Stage
**Purpose**: Create summary, chunks, and contextual embeddings
- **Input**: `{filename}.md` or `{filename}.opt.md`
- **Output**: `{filename}.chunks.json`
- **Features**:
  - Document summary generation with contextual understanding
  - Configurable chunking strategies (semantic, page-level, topic-based)
  - Contextual embeddings for each chunk with surrounding context
  - Chunk metadata with source references and contextual summaries

### fact-generator Stage
**Purpose**: Extract facts, entities, relations, and keywords
- **Input**: `{filename}.chunks.json`
- **Output**: `{filename}.facts.json`
- **Features**:
  - Entity extraction and normalization
  - Relationship identification
  - Keyword extraction
  - Domain-specific fact generation
  - Source attribution with chunk references

### ingestor Stage
**Purpose**: Database ingestion and storage
- **Input**: `{filename}.chunks.json` and `{filename}.facts.json`
- **Output**: Database records and `{filename}.ingestion.json` (results)
- **Features**:
  - Multi-database support (Qdrant, Neo4j)
  - Configurable database targets
  - Deduplication and conflict resolution
  - Ingestion result tracking

## Implementation Tasks

### Task 1: Design Stage Models and Interfaces
- Define `Stage` base class and interface using canonical stage names
- Create `StageResult` and `StageContext` models
- Design file naming conventions and metadata structures
- Implement stage dependency management

### Task 2: Implement markdown-conversion Stage
- Completely replace existing services with new stage interface
- Implement unified markdown output format
- Add metadata header standardization
- Create stage-specific configuration options
- Remove all legacy processing code

### Task 3: Implement markdown-optimizer Stage
- Design LLM prompt templates using system/user message pattern
- Implement text optimization pipeline with proper message structure
- Add quality assessment and validation
- Create configurable optimization strategies
- Use system messages for instructions, user messages for content

### Task 4: Implement chunker Stage
- Completely replace chunking services with new stage interface
- Implement contextual summary generation with surrounding context awareness
- Add contextual embedding generation with chunk relationships
- Create chunk metadata with contextual summaries and source tracking
- Generate embeddings that understand chunk context and relationships

### Task 5: Implement fact-generator Stage
- Completely replace fact extraction with new stage interface
- Implement entity normalization and deduplication
- Add relationship extraction and validation
- Create fact metadata and source attribution
- Remove all legacy fact extraction code

### Task 6: Implement ingestor Stage
- Completely replace ingestion coordinator with new stage interface
- Add multi-database configuration support
- Implement ingestion result tracking
- Add conflict resolution and deduplication
- Remove all legacy ingestion code

### Task 7: Update CLI Scripts and Commands
- Create completely new stage-based CLI interface
- Add individual stage execution commands using canonical names
- Implement stage chaining and dependency resolution
- Remove all existing CLI scripts and replace with new ones

### Task 8: Update REST Endpoints
- Design completely new stage-based REST API
- Implement individual stage endpoints using canonical names
- Add file download and management endpoints
- **REMOVE ALL OLD ENDPOINTS COMPLETELY** - no backwards compatibility
- Replace entire API structure

### Task 9: Implement File Management and Webhook Support
- Create file storage and retrieval system
- Implement webhook notification system
- Add file download endpoints
- Create file cleanup and retention policies

### Task 10: Update Tests and Documentation
- Replace all tests with new stage-based architecture tests
- Create comprehensive API documentation for new endpoints only
- Add usage examples and tutorials
- Update deployment and configuration guides
- Remove all documentation for old system

## File Naming Conventions

```
{source_filename}.md           # Stage 1 output
{source_filename}.opt.md       # Stage 2 output (optional)
{source_filename}.chunks.json  # Stage 3 output
{source_filename}.facts.json   # Stage 4 output
{source_filename}.ingestion.json # Stage 5 output
```

## CLI Interface Design

```bash
# Execute individual stages using canonical names
morag markdown-conversion input.mp4 --output-dir ./processed
morag markdown-optimizer input.md --optimize-for transcription
morag chunker input.md --chunk-strategy semantic
morag fact-generator input.chunks.json --domain medical
morag ingestor input.chunks.json input.facts.json --databases qdrant,neo4j

# Execute stage chains using canonical names
morag stages markdown-conversion,chunker,fact-generator input.mp4
morag stages markdown-optimizer,chunker,fact-generator,ingestor input.md --skip-stage fact-generator

# Full pipeline
morag process input.mp4 --stages all
```

## REST API Design (Completely New - All Old Endpoints Removed)

```
POST /api/v1/stages/markdown-conversion/execute    # markdown-conversion
POST /api/v1/stages/markdown-optimizer/execute     # markdown-optimizer
POST /api/v1/stages/chunker/execute                # chunker
POST /api/v1/stages/fact-generator/execute         # fact-generator
POST /api/v1/stages/ingestor/execute               # ingestor

GET  /api/v1/files/{file_id}                       # Download stage output
POST /api/v1/stages/chain                          # Execute multiple stages
GET  /api/v1/stages/status/{job_id}                # Check stage execution status
```

## Implementation Priority

1. **High Priority**: Stage models, interfaces, and Stage 1 implementation
2. **Medium Priority**: Stages 3-5 implementation and CLI updates
3. **Low Priority**: Stage 2 (optional), REST API updates, and advanced features

## Success Criteria

- All existing functionality preserved through stage-based architecture
- Individual stages can be executed independently
- Intermediate files are reusable and downloadable
- Webhook integration works for all stage outputs
- Performance is maintained or improved
- Full backward compatibility with existing CLI and REST interfaces
