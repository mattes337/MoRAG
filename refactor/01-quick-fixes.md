# MoRAG Refactoring Task 1: Quick Fixes

## Priority: HIGH
## Estimated Time: 2-4 hours
## Impact: Immediate code quality improvement

## Overview
This task addresses immediate code quality issues that can be fixed quickly without architectural changes. These fixes will improve maintainability and reduce technical debt before proceeding with larger refactoring efforts.

## Issues Identified

### 1. Import Statement Cleanup
**Problem**: Redundant HTTP client imports - both `requests` and `httpx` are used across the codebase (38 files use requests, 15 use httpx).

**Quick Fixes**:
```bash
# Files using requests (38 files):
- API_USAGE_GUIDE.md, test files, tools/remote-converter, packages/morag-youtube
# Files using httpx (15 files): 
- packages/morag-reasoning, packages/morag-web, packages/morag-stages
```

**Action Items**:
1. **Standardize on httpx**: Replace all `requests` usage with `httpx` for consistency
2. **Remove requests dependency**: Update requirements.txt to remove `requests>=2.32.0`
3. **Update import statements**: 
   ```python
   # Replace:
   import requests
   # With:
   import httpx
   ```

### 2. Unused Heavy Dependencies
**Problem**: Heavy dependencies with minimal usage could be optional or removed entirely.

**Dependencies to Review**:
- `torch` (25 occurrences across 9 files) - Only used in audio processing
- `numpy` (12 occurrences across 11 files) - Light usage, could be reduced
- `scipy` - Listed in requirements but minimal usage found
- `scikit-learn` - Listed in requirements but minimal usage found

**Quick Fixes**:
1. Make PyTorch optional for audio processing:
   ```python
   # Add to morag_core/optional_deps.py
   try:
       import torch
       HAS_TORCH = True
   except ImportError:
       HAS_TORCH = False
       torch = None
   ```

2. Move heavy ML dependencies to optional extras in requirements.txt:
   ```txt
   # Move to [audio-ml] extra:
   torch>=2.1.0,<2.7.0
   torchaudio>=2.1.0,<2.7.0
   pyannote.audio>=3.3.0,<4.0.0
   
   # Move to [scientific] extra:
   scipy>=1.13.0,<1.15.0
   scikit-learn>=1.5.0,<1.6.0
   ```

### 3. Duplicate Logger Initialization
**Problem**: Logger initialization is duplicated across many files (401 `__init__` methods found).

**Quick Fix**:
1. Create a standard logger utility:
   ```python
   # packages/morag-core/src/morag_core/utils/logging.py
   def get_logger(name: str = None):
       import structlog
       return structlog.get_logger(name or __name__)
   ```

2. Standardize logger usage:
   ```python
   # Replace duplicated patterns like:
   logger = structlog.get_logger(__name__)
   # With:
   from morag_core.utils.logging import get_logger
   logger = get_logger(__name__)
   ```

### 4. File Size Violations - Immediate Actions
**Files exceeding 1000 lines** (9 critical files):

| File | Lines | Quick Action |
|------|--------|-------------|
| `ingestion_coordinator.py` | 2958 | Split into 3 files: coordinator, database_handler, result_processor |
| `markdown_conversion.py` | 1627 | Extract converter classes to separate files |
| `markitdown_base.py` | 1149 | Split base class and specific converters |
| `services.py` | 1144 | Split by service type (embedding, storage, processing) |
| `webdav-processor.py` | 1119 | Extract processing logic from CLI interface |
| `fact_generator.py` | 1061 | Split fact extraction and fact processing logic |
| `fact_graph_builder.py` | 1042 | Extract graph operations to separate module |
| `stages.py` (API endpoints) | 1035 | Split by endpoint groups |
| `qdrant_storage.py` | 1034 | Split storage operations by data type |

### 5. Missing Unit Tests - Critical Gaps
**Current Coverage**: 66 test files for 321 production files (20.6% coverage)

**Immediate Priority** (files without tests):
1. Core interfaces (`morag_core/interfaces/*.py`)
2. Storage classes (`*storage.py`)
3. Service classes (`*service.py`)
4. Processor classes (`*processor.py`)

## Implementation Plan

### Phase 1: Import Cleanup (30 minutes)
```bash
# 1. Update all requests imports to httpx
find . -name "*.py" -exec sed -i 's/import requests/import httpx/g' {} \;
find . -name "*.py" -exec sed -i 's/requests\./httpx\./g' {} \;

# 2. Update requirements.txt
# Remove: requests>=2.32.0  
# Keep: httpx==0.28.1

# 3. Test imports
python -c "import httpx; print('httpx import successful')"
```

### Phase 2: Optional Dependencies (45 minutes)
1. Update `packages/morag-core/src/morag_core/optional_deps.py`
2. Modify audio processing files to handle missing torch gracefully
3. Update requirements.txt with optional extras
4. Update installation documentation

### Phase 3: Logger Standardization (30 minutes)
1. Enhance `morag_core/utils/logging.py`
2. Create find/replace script for logger initialization
3. Apply changes across codebase
4. Test logger functionality

### Phase 4: Critical File Size Fixes (2 hours)
**Focus on top 3 largest files only**:
1. `ingestion_coordinator.py` (2958 lines) → Split into 3 files
2. `markdown_conversion.py` (1627 lines) → Extract converters
3. `markitdown_base.py` (1149 lines) → Split base and implementations

## Success Criteria

### Immediate Wins
- [ ] Single HTTP client library (httpx only)
- [ ] Heavy dependencies made optional  
- [ ] Standardized logging across all files
- [ ] Top 3 largest files split to <800 lines each
- [ ] No import errors after changes

### Quality Metrics
- [ ] All tests pass after changes
- [ ] Import time reduced by removing unused dependencies
- [ ] Reduced memory footprint without optional ML dependencies
- [ ] Code maintainability improved with smaller files

## Testing Strategy
```bash
# After each change:
python check_syntax.py --verbose
pytest tests/unit/ -v
python tests/cli/test-simple.py

# Performance test:
time python -c "from morag import api"
```

## Next Steps
After completing quick fixes, proceed to:
1. `02-file-splitting.md` - Address remaining oversized files
2. `03-deduplication.md` - Remove duplicate code patterns  
3. `04-dependency-cleanup.md` - Full dependency audit
4. `05-testing-strategy.md` - Comprehensive test coverage

## Notes
- **Backward Compatibility**: These changes maintain API compatibility
- **Risk Level**: LOW - mostly cosmetic and import changes
- **Dependencies**: No external dependencies required
- **Rollback**: Easy to revert via git if issues occur