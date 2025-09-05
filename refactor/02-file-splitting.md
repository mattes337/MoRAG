# MoRAG Refactoring Task 2: File Splitting Strategy

## Priority: HIGH  
## Estimated Time: 8-12 hours
## Impact: Major maintainability improvement

## Overview
This task addresses the 9 files exceeding 1000 lines by splitting them into focused, maintainable modules. Each resulting file should be ≤500 lines (target) with clear single responsibilities.

## Critical Files Requiring Immediate Splitting

### 1. `ingestion_coordinator.py` (2958 lines) - CRITICAL
**Current Issues**: 
- Massive coordinator class handling database detection, embedding, metadata, initialization, and writing
- Multiple responsibilities violating SRP
- Difficult to test and maintain

**Split Strategy**:
```
ingestion_coordinator.py (2958) → Split into 4 files:

1. ingestion_coordinator.py (400 lines)
   - Main coordination logic only
   - Interface definitions
   - High-level orchestration

2. database_handler.py (800 lines)  
   - Database detection and configuration
   - Collection/database initialization
   - Connection management

3. result_processor.py (700 lines)
   - Embedding generation
   - Metadata processing  
   - Result file creation

4. data_writer.py (600 lines)
   - Database writing operations
   - Batch processing
   - Error handling
```

**Implementation Plan**:
```python
# 1. Create new files structure
packages/morag/src/morag/ingestion/
├── __init__.py
├── coordinator.py          # Main orchestrator (400 lines)
├── database_handler.py     # DB operations (800 lines)
├── result_processor.py     # Result processing (700 lines) 
└── data_writer.py         # Data writing (600 lines)

# 2. Extract classes
class IngestionCoordinator:      # coordinator.py
class DatabaseHandler:           # database_handler.py  
class ResultProcessor:           # result_processor.py
class DataWriter:                # data_writer.py
```

### 2. `markdown_conversion.py` (1627 lines) - HIGH
**Current Issues**:
- Single file handling all markdown conversion logic
- Multiple converter classes mixed together
- Complex stage processing logic

**Split Strategy**:
```
markdown_conversion.py (1627) → Split into 3 files:

1. markdown_conversion_stage.py (300 lines)
   - Stage interface and coordination
   - Pipeline orchestration
   - Error handling

2. converter_factory.py (400 lines)
   - Converter selection logic
   - Factory pattern implementation
   - Content type detection

3. conversion_processors.py (600 lines)
   - Individual converter implementations
   - Processing utilities
   - Quality assessment
```

### 3. `markitdown_base.py` (1149 lines) - HIGH
**Current Issues**:
- Base converter class with too many responsibilities
- Mixed abstract base and concrete implementations

**Split Strategy**:
```
markitdown_base.py (1149) → Split into 2 files:

1. base_converter.py (300 lines)
   - Abstract base class only
   - Interface definitions
   - Common utilities

2. document_converters.py (600 lines)
   - Concrete converter implementations
   - Document-specific processing
   - Format handlers
```

### 4. `services.py` (1144 lines) - HIGH 
**Current Issues**:
- Multiple service types in single file
- Mixed embedding, storage, and processing services

**Split Strategy**:
```
services.py (1144) → Split into 3 files:

1. embedding_services.py (400 lines)
   - GeminiEmbeddingService
   - Batch embedding logic
   - Embedding utilities

2. storage_services.py (400 lines)
   - QdrantVectorStorage 
   - Storage operations
   - Connection management

3. processing_services.py (350 lines)
   - Processing utilities
   - Service coordination
   - Common interfaces
```

### 5. `fact_generator.py` (1061 lines) - MEDIUM
**Split Strategy**:
```
fact_generator.py (1061) → Split into 2 files:

1. fact_generation_stage.py (350 lines)
   - Stage orchestration
   - Pipeline coordination
   - Configuration management

2. fact_extraction_engine.py (500 lines)
   - Fact extraction logic
   - AI-powered processing
   - Result validation
```

## Remaining Files (500-1000 lines) - MEDIUM Priority

### 6. `webdav-processor.py` (1119 lines)
```
webdav-processor.py → Split into:
1. webdav_cli.py (200 lines) - CLI interface
2. webdav_processor.py (500 lines) - Processing logic
3. webdav_client.py (300 lines) - WebDAV operations
```

### 7. `fact_graph_builder.py` (1042 lines) 
```  
fact_graph_builder.py → Split into:
1. graph_builder_interface.py (200 lines) - Interface
2. fact_graph_operations.py (600 lines) - Core operations
3. graph_utilities.py (200 lines) - Helper functions
```

### 8. `stages.py` (API endpoints) (1035 lines)
```
stages.py → Split into:
1. stages_router.py (200 lines) - FastAPI router setup
2. stage_execution_endpoints.py (400 lines) - Execution endpoints
3. file_management_endpoints.py (300 lines) - File endpoints
```

### 9. `qdrant_storage.py` (1034 lines)
```
qdrant_storage.py → Split into:
1. qdrant_client.py (300 lines) - Connection and client
2. qdrant_operations.py (500 lines) - CRUD operations  
3. qdrant_utilities.py (200 lines) - Helper functions
```

## Implementation Guidelines

### File Structure Standards
```python
# Each split file should follow this structure:
"""
Module docstring explaining purpose and scope.
Maximum 500 lines per file.
Single responsibility principle.
"""

# 1. Imports (grouped logically)
from typing import ...
import standard_lib
import third_party
from morag_core import ...

# 2. Constants and configuration
CONSTANT_VALUE = "value"

# 3. Class definitions (max 2-3 classes per file)
class MainClass:
    """Clear docstring explaining responsibility."""
    pass

# 4. Utility functions (if needed)
def utility_function():
    pass

# 5. Module-level exports
__all__ = ["MainClass", "utility_function"]
```

### Dependency Management
```python
# Each split file should minimize dependencies
# Use dependency injection for complex dependencies
# Maintain clear interfaces between split components

# Example dependency injection pattern:
class IngestionCoordinator:
    def __init__(
        self,
        database_handler: DatabaseHandler,
        result_processor: ResultProcessor, 
        data_writer: DataWriter
    ):
        self._db_handler = database_handler
        self._processor = result_processor
        self._writer = data_writer
```

## Testing Strategy for Split Files

### Unit Test Requirements
Each split file must have corresponding unit tests:
```python
# For each new file created, add:
tests/unit/test_<module_name>.py

# Example:
tests/unit/ingestion/test_coordinator.py
tests/unit/ingestion/test_database_handler.py  
tests/unit/ingestion/test_result_processor.py
tests/unit/ingestion/test_data_writer.py
```

### Integration Tests
```python
# Test split components work together:
tests/integration/test_ingestion_workflow.py
tests/integration/test_markdown_conversion_pipeline.py
```

## Implementation Phase Plan

### Phase 1: Critical Files (6 hours)
**Order of execution**:
1. `ingestion_coordinator.py` → 4 files (2 hours)
2. `markitdown_base.py` → 2 files (1 hour)  
3. `services.py` → 3 files (1.5 hours)
4. Unit tests for split files (1.5 hours)

### Phase 2: Medium Priority (4 hours)
1. `markdown_conversion.py` → 3 files (1.5 hours)
2. `fact_generator.py` → 2 files (1 hour)
3. Unit tests (1.5 hours)

### Phase 3: Remaining Files (4 hours)
1. Split remaining 4 files (2.5 hours)
2. Integration tests (1.5 hours)

## Success Criteria

### File Size Targets
- [ ] No files >1000 lines
- [ ] Target: All files ≤500 lines  
- [ ] Average file size: 200-400 lines

### Code Quality
- [ ] Each file has single responsibility
- [ ] Clear interfaces between split components
- [ ] All imports are necessary and used
- [ ] No circular dependencies introduced

### Testing Coverage
- [ ] Unit tests for each split file (>80% coverage)
- [ ] Integration tests verify component interaction
- [ ] All existing functionality preserved

### Performance
- [ ] No performance regression
- [ ] Import times not significantly increased
- [ ] Memory usage unchanged or improved

## Validation Commands

```bash
# After each split:
# 1. Check file sizes
find packages/ -name "*.py" -exec wc -l {} + | sort -nr | head -20

# 2. Verify no syntax errors
python check_syntax.py --verbose

# 3. Run tests
pytest tests/unit/ -v
pytest tests/integration/ -v

# 4. Check imports
python -c "from morag.ingestion import IngestionCoordinator; print('Import successful')"

# 5. Performance baseline
time python tests/cli/test-simple.py
```

## Risk Mitigation

### Backup Strategy
```bash
# Before starting each split:
git checkout -b refactor/split-<filename>
git add . && git commit -m "Backup before splitting <filename>"
```

### Rollback Plan
```bash
# If issues occur:
git checkout main
git branch -D refactor/split-<filename>
```

### Testing Safety Net
```bash
# Run full test suite after each major split:
python tests/cli/test-all.py
pytest tests/ -x -v
```

## Dependencies for Next Tasks
This task creates the foundation for:
- **03-deduplication.md**: Smaller files easier to deduplicate
- **04-dependency-cleanup.md**: Clear module boundaries for dependency analysis  
- **05-testing-strategy.md**: Focused test creation for split components

## Documentation Updates Required
- Update API documentation for moved classes
- Update import examples in README.md
- Update CLAUDE.md with new file structure
- Add architectural diagrams showing component relationships