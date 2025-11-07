# MoRAG Migration Guide

## Breaking Changes in v2.0

**IMPORTANT**: MoRAG v2.0 is a complete rewrite with **NO backward compatibility**. This migration guide helps you understand the changes and update your code accordingly.

## API Changes

### Complete API Redesign

The entire API has been redesigned. Previous endpoints are **completely removed**:

❌ **REMOVED ENDPOINTS** (no fallback):
- `/api/v1/process` - REMOVED
- `/api/v1/ingest/file` - REMOVED
- `/api/v1/query` - REMOVED
- `/api/v1/extract` - REMOVED
- All legacy processing endpoints

### New Stage-Based API

All processing now uses the stage-based system:

✅ **NEW ENDPOINTS**:
- `/api/v1/stages/{stage}/execute` - Execute single stage
- `/api/v1/stages/chain` - Execute stage chain
- `/api/v1/stages/list` - List available stages
- `/api/v1/files/upload` - File upload for processing
- `/api/v1/files/{file_id}/status` - Check processing status

### Stage-Based Processing Architecture

The new system uses canonical stage names:

1. **`markdown-conversion`** - Convert input files to unified markdown
2. **`markdown-optimizer`** - LLM-based text improvement (optional)
3. **`chunker`** - Create summary, chunks, and contextual embeddings
4. **`fact-generator`** - Extract facts, entities, relations, and keywords
5. **`ingestor`** - Database ingestion and storage

## Migration Examples

### Before (v1.x) - DEPRECATED
```python
# OLD API - NO LONGER WORKS
import requests

# This will fail - endpoint removed
response = requests.post("/api/v1/process", {
    "file_path": "document.pdf",
    "extract_entities": True
})
```

### After (v2.0) - NEW APPROACH
```python
# NEW STAGE-BASED API
import requests

# Upload file first
files = {"file": open("document.pdf", "rb")}
upload_response = requests.post("/api/v1/files/upload", files=files)
file_id = upload_response.json()["file_id"]

# Execute stage chain
chain_response = requests.post("/api/v1/stages/chain", {
    "file_id": file_id,
    "stages": ["markdown-conversion", "chunker", "fact-generator"]
})
```

### CLI Changes

#### Before (v1.x) - DEPRECATED
```bash
# OLD CLI - NO LONGER WORKS
python cli/process.py document.pdf --extract-entities
python cli/ingest.py document.pdf
```

#### After (v2.0) - NEW CLI
```bash
# NEW STAGE-BASED CLI
python cli/morag-stages.py stage markdown-conversion document.pdf
python cli/morag-stages.py stage chunker output/document.md
python cli/morag-stages.py stages "markdown-conversion,chunker,fact-generator" document.pdf

# Complete pipeline
python cli/morag-stages.py process document.pdf --optimize --output-dir ./output
```

## Configuration Changes

### Environment Variables

Updated environment variable names:
```bash
# NEW CONFIGURATION SYSTEM
MORAG_GEMINI_MODEL=gemini-1.5-pro  # Global fallback
MORAG_FACT_EXTRACTION_AGENT_MODEL=gemini-2.0-flash  # Agent-specific
MORAG_CHUNK_SIZE=2000
MORAG_BATCH_SIZE=50
MORAG_MAX_WORKERS=4
```

### Package Structure Changes

The monolithic structure has been replaced with modular packages:

```
OLD (v1.x):           NEW (v2.0):
src/                  packages/
├── morag/           ├── morag-core/         # Core interfaces
├── services/        ├── morag-services/     # AI services
├── processors/      ├── morag-stages/       # Processing stages
└── ...              ├── morag-audio/        # Audio processing
                     ├── morag-document/     # Document processing
                     ├── morag-video/        # Video processing
                     ├── morag-graph/        # Knowledge graph
                     └── morag/              # Main package
```

## Key Architectural Changes

### 1. PydanticAI Integration
- Type-safe, structured AI interactions
- Automatic validation and error handling
- Structured fact extraction with confidence scoring

### 2. Stage-Based Processing
- Modular pipeline with resume capability
- Standardized input/output formats
- Automatic detection of completed stages

### 3. Enhanced Performance
- 4x faster embedding batch processing
- Improved GPU/CPU fallback mechanisms
- Optimized memory usage

### 4. Improved Error Handling
- Comprehensive exception hierarchy
- Structured error responses
- Better debugging information

## Testing Changes

### Before (v1.x)
```bash
# OLD TESTING
python test_system.py
python test_integration.py
```

### After (v2.0)
```bash
# NEW TESTING STRUCTURE
pytest                              # All tests
pytest tests/unit/                 # Unit tests
pytest tests/integration/          # Integration tests
python tests/cli/test-simple.py   # Quick validation
```

## Database Schema Changes

### Neo4j Knowledge Graph
- Updated entity and relation models
- New fact-based storage approach
- Improved relationship detection

### Qdrant Vector Storage
- Unified collection structure
- Enhanced embedding batch processing
- Better similarity search performance

## Breaking Changes Summary

1. **Complete API Redesign**: All endpoints changed
2. **No Backward Compatibility**: Must rewrite all API calls
3. **New Package Structure**: Import paths changed
4. **Configuration Updates**: Environment variables renamed
5. **CLI Commands**: All CLI scripts updated
6. **Database Schema**: Knowledge graph structure changed

## Migration Checklist

- [ ] Update all API endpoint calls to use stage-based system
- [ ] Rewrite CLI scripts to use new `morag-stages.py`
- [ ] Update environment variable names (`MORAG_` prefix)
- [ ] Install new package structure (`pip install packages/morag/`)
- [ ] Update import statements for new package structure
- [ ] Test stage-based processing with your content
- [ ] Verify database connections work with new schema
- [ ] Update documentation and deployment scripts

## Getting Help

If you encounter issues during migration:

1. Check the updated documentation in `/docs/`
2. Run system validation: `python tests/cli/test-simple.py`
3. Review package interfaces in `packages/*/src/*//__init__.py`
4. Check the CLAUDE.md file for current development commands

## Support Timeline

- **v1.x**: No longer supported
- **v2.0**: Current version with active development
- **Migration Support**: Available through Q1 2025
