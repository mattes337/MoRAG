# MoRAG Refactoring - Postponed Tasks

## Overview
This document tracks refactoring tasks that were identified but postponed due to complexity, risk, or time constraints. These tasks remain valuable improvements but require more extensive planning and testing.

---

## File Splitting Tasks

### 1. Complete markdown_conversion.py Refactoring
**File**: `packages/morag-stages/src/morag_stages/stages/markdown_conversion.py`  
**Current Size**: 1627 lines  
**Status**: Partially completed (validators extracted)

#### What Was Done
- ✅ Created `markdown_validators.py` module with all validation functions
- ✅ Extracted validation logic to improve modularity

#### Remaining Work
- **Extract conversion methods** to `markdown_converters.py`:
  - `_process_with_markitdown()`
  - `_process_video()`
  - `_process_audio()`
  - `_process_with_document_service()`
  - Helper methods for different content types
- **Refactor main stage class** to use extracted modules
- **Update imports** and dependencies across the codebase
- **Create comprehensive tests** for split modules

#### Complexity Factors
- Single large class with tightly coupled methods
- Complex dependency management (optional services)
- Integration with multiple processing engines
- Extensive error handling and fallback logic

#### Estimated Effort
**Time**: 4-6 hours  
**Risk**: Medium (integration complexity)

---

### 2. Split markitdown_base.py Implementation
**File**: `packages/morag-document/src/morag_document/converters/markitdown_base.py`  
**Current Size**: 1149 lines  
**Status**: Not started

#### Proposed Split
- **Base converter class** - `markitdown_base.py` (interface and common logic)
- **Specific converters** - `markitdown_pdf.py`, `markitdown_office.py`, `markitdown_image.py`
- **Utilities** - `markitdown_utils.py` (helper functions, validation)

#### Complexity Factors
- Deep integration with MarkItDown library
- Complex inheritance hierarchies
- File format-specific processing logic
- Error handling for corrupted files

#### Estimated Effort
**Time**: 3-4 hours  
**Risk**: Medium-High (breaking changes possible)

---

## Logger Standardization - Remaining Files

### Files Still Using Old Pattern
The following files still use the old `structlog.get_logger(__name__)` pattern and should be updated to use the centralized `get_logger()` utility:

#### Core Agent Files
- `packages/morag-core/src/morag_core/agents/advanced_extraction_agent.py`
- `packages/morag-core/src/morag_core/agents/cfg_extraction_agent.py`
- `packages/morag-core/src/morag_core/ai/base_agent.py`

#### Service Files
- `packages/morag-youtube/src/morag_youtube/transcript.py`
- `packages/morag/src/morag/services/enhanced_webhook_service.py`
- `packages/morag/src/morag/ingest_tasks.py`

#### Estimated Effort
**Time**: 1-2 hours (systematic find/replace)  
**Risk**: Low  
**Pattern**: Replace `import structlog; logger = structlog.get_logger(__name__)` with `from ..utils.logging import get_logger; logger = get_logger(__name__)`

---

## Additional File Size Violations

### Large Files Identified But Not Split

#### 1. services.py (1144 lines)
**Location**: `packages/morag/src/morag/services.py`  
**Proposed Split**:
- `embedding_services.py` - Embedding-related services
- `storage_services.py` - Storage and database services  
- `processing_services.py` - Content processing services

#### 2. webdav-processor.py (1119 lines)
**Location**: `cli/webdav-processor.py`  
**Proposed Split**:
- `webdav_processor.py` - Core processing logic
- `webdav_cli.py` - Command-line interface
- `webdav_config.py` - Configuration handling

#### 3. fact_generator.py (1061 lines)
**Location**: `packages/morag-stages/src/morag_stages/stages/fact_generator.py`  
**Proposed Split**:
- `fact_generator.py` - Main stage logic
- `fact_extraction.py` - Extraction algorithms
- `fact_processing.py` - Post-processing and validation

#### 4. fact_graph_builder.py (1042 lines)
**Location**: `packages/morag-graph/src/morag_graph/fact_graph_builder.py`  
**Proposed Split**:
- `fact_graph_builder.py` - Main builder class
- `graph_operations.py` - Graph manipulation operations
- `graph_analytics.py` - Analysis and metrics

#### 5. API stages.py (1035 lines)
**Location**: `packages/morag/src/morag/api_models/endpoints/stages.py`  
**Proposed Split**:
- `stage_endpoints.py` - Basic stage operations
- `file_endpoints.py` - File processing endpoints
- `status_endpoints.py` - Status and monitoring endpoints

#### 6. qdrant_storage.py (1034 lines)
**Location**: `packages/morag-services/src/morag_services/storage/qdrant_storage.py`  
**Proposed Split**:
- `qdrant_storage.py` - Core storage operations
- `qdrant_documents.py` - Document-specific operations
- `qdrant_vectors.py` - Vector-specific operations

---

## HTTP Client Migration - Remaining Files

### Test Files Still Using Requests
The following test and CLI files still use `requests` and should be migrated to `httpx`:

#### Test Files
- `tests/test_quality_gates_configuration.py`
- `tests/test_confidence_calculation.py`
- `tests/cli/test-url-debug.py`
- `tests/cli/test-document.py`
- `tests/cli/test-web.py`
- `tests/cli/test-youtube.py`
- `tests/cli/test-remote-debug.py`
- And ~20 other test files

#### Tools and Scripts
- `tools/remote-converter/remote_converter_minimal.py`
- `tools/remote-converter/remote_converter.py`
- `scripts/test-docker.py`

#### Estimated Effort
**Time**: 2-3 hours (systematic replacement)  
**Risk**: Low-Medium (test compatibility)

---

## Missing Unit Tests - Critical Gaps

### Current Coverage Issues
- **Overall Coverage**: 66 test files for 321 production files (20.6% coverage)
- **Missing Test Categories**:
  - Core interfaces (`morag_core/interfaces/*.py`) - 0% coverage
  - Storage classes (`*storage.py`) - Partial coverage
  - Service classes (`*service.py`) - Inconsistent coverage
  - Processor classes (`*processor.py`) - Limited coverage

### Immediate Priority Files
1. `packages/morag/src/morag/database_handler.py` (newly created)
2. `packages/morag/src/morag/chunk_processor.py` (newly created)
3. `packages/morag-stages/src/morag_stages/stages/markdown_validators.py` (newly created)
4. Core interfaces in `morag_core/interfaces/`
5. Storage services in `morag_services/storage/`

#### Estimated Effort
**Time**: 8-12 hours for comprehensive test coverage  
**Risk**: Medium (requires understanding complex logic)

---

## Requirements.txt Optimization - Advanced

### Optional Dependency Groups
Create proper optional dependency groups in `setup.py` or `pyproject.toml`:

```toml
[tool.setuptools.extras-require]
audio-ml = [
    "torch>=2.1.0,<2.7.0",
    "torchaudio>=2.1.0,<2.7.0", 
    "pyannote.audio>=3.3.0,<4.0.0",
    "sentence-transformers>=3.0.0,<5.0.0"
]
scientific = [
    "scipy>=1.13.0,<1.15.0",
    "scikit-learn>=1.5.0,<1.6.0"
]
all-extras = ["morag[audio-ml,scientific]"]
```

#### Estimated Effort
**Time**: 1-2 hours  
**Risk**: Low

---

## Documentation Updates Required

### Files That Need Documentation Updates
1. **Installation guides** - Update with new optional dependencies
2. **API documentation** - Reflect split file structure  
3. **Developer guides** - New module organization
4. **Testing documentation** - Updated test commands and coverage expectations

#### Estimated Effort
**Time**: 2-3 hours  
**Risk**: Low

---

## Validation and Testing Strategy

### Integration Testing Needed
- **Database handler integration** - Test with real Qdrant/Neo4j instances
- **Chunk processor integration** - Test with various content types
- **End-to-end pipeline testing** - Ensure split files work together
- **Performance regression testing** - Verify no performance degradation

### Recommended Approach
1. Create comprehensive integration test suite
2. Set up automated testing pipeline  
3. Performance benchmarking before/after
4. Gradual rollout with monitoring

#### Estimated Effort
**Time**: 4-6 hours for comprehensive testing  
**Risk**: Medium

---

## Summary

### Total Postponed Work Estimate
- **High Priority**: 8-12 hours (file splitting, testing)
- **Medium Priority**: 6-8 hours (logger standardization, HTTP migration)  
- **Low Priority**: 3-4 hours (documentation, requirements optimization)
- **Total**: 17-24 hours of additional refactoring work

### Recommendations
1. **Phase 1**: Complete markdown_conversion.py splitting and create tests for new modules
2. **Phase 2**: Address remaining large files (services.py, fact_generator.py)
3. **Phase 3**: Systematic cleanup (logger standardization, HTTP migration)
4. **Phase 4**: Comprehensive testing and documentation

### Risk Assessment
- **Low Risk**: Logger standardization, documentation updates
- **Medium Risk**: File splitting, HTTP client migration  
- **High Risk**: Large architectural changes, dependency restructuring

All postponed tasks are **technically sound** and **beneficial**, but require careful planning and extensive testing to avoid breaking existing functionality.