# Task 16: Cleanup Files

## Objective
Delete all obsolete files and directories that are no longer needed after the LangExtract migration. This includes old extractors, normalizers, pattern matchers, and related utilities.

## Prerequisites
- All previous tasks completed (1-15)
- LangExtract system fully functional
- Backup created
- All tests passing

## Files and Directories to Delete

### 1. Core Extraction Files

**Directory**: `packages/morag-graph/src/morag_graph/extractors/`
```bash
# DELETE entire directory and contents
rm -rf packages/morag-graph/src/morag_graph/extractors/
```

**Files in this directory**:
- `openie_extractor.py` (already removed in Task 6)
- `__init__.py`
- Any other extractor files

### 2. Old Extraction Components

**Directory**: `packages/morag-graph/src/morag_graph/extraction/`

**Files to DELETE**:
```bash
rm packages/morag-graph/src/morag_graph/extraction/entity_extractor.py
rm packages/morag-graph/src/morag_graph/extraction/relation_extractor.py
rm packages/morag-graph/src/morag_graph/extraction/hybrid_extractor.py
rm packages/morag-graph/src/morag_graph/extraction/spacy_extractor.py
rm packages/morag-graph/src/morag_graph/extraction/pattern_matcher.py
rm packages/morag-graph/src/morag_graph/extraction/base.py
```

**Files to KEEP**:
```bash
# Keep LangExtract files
packages/morag-graph/src/morag_graph/extraction/langextract_entity_extractor.py
packages/morag-graph/src/morag_graph/extraction/langextract_relation_extractor.py
packages/morag-graph/src/morag_graph/extraction/langextract_base.py
packages/morag-graph/src/morag_graph/extraction/__init__.py  # Updated version
```

### 3. Normalization System

**Directory**: `packages/morag-graph/src/morag_graph/normalizers/`
```bash
# DELETE entire directory
rm -rf packages/morag-graph/src/morag_graph/normalizers/
```

**Files in this directory**:
- `entity_normalizer.py`
- `__init__.py`
- Any normalization utilities

### 4. Old AI Agents

**Directory**: `packages/morag-graph/src/morag_graph/ai/`

**Files to DELETE**:
```bash
rm packages/morag-graph/src/morag_graph/ai/entity_agent.py
rm packages/morag-graph/src/morag_graph/ai/relation_agent.py
```

**Files to KEEP**:
```bash
# Keep if used elsewhere
packages/morag-graph/src/morag_graph/ai/__init__.py  # Updated version
```

### 5. Old Services

**Directory**: `packages/morag-graph/src/morag_graph/services/`

**Files to DELETE**:
```bash
rm packages/morag-graph/src/morag_graph/services/openie_service.py  # Already removed
# Remove any other extraction-related services
```

**Files to KEEP**:
```bash
packages/morag-graph/src/morag_graph/services/langextract_service.py
```

### 6. Old Processors

**Directory**: `packages/morag-graph/src/morag_graph/processors/`
```bash
# DELETE entire directory if it exists
rm -rf packages/morag-graph/src/morag_graph/processors/
```

**Files that might be in this directory**:
- `openie_processor.py`
- `triplet_processor.py`
- `sentence_processor.py`
- `entity_processor.py`

### 7. Old Storage Operations

**Directory**: `packages/morag-graph/src/morag_graph/storage/neo4j_operations/`

**Files to DELETE**:
```bash
rm packages/morag-graph/src/morag_graph/storage/neo4j_operations/openie_operations.py  # Already removed
```

### 8. Old Configuration Files

**Directory**: `packages/morag-graph/src/morag_graph/config/`

**Files to DELETE**:
```bash
rm packages/morag-graph/src/morag_graph/config/extraction_config.py  # If exists
rm packages/morag-graph/src/morag_graph/config/normalization_config.py
rm packages/morag-graph/src/morag_graph/config/openie_config.py  # If exists
```

**Files to KEEP**:
```bash
packages/morag-graph/src/morag_graph/config/langextract_config.py
```

### 9. Old Monitoring

**Directory**: `packages/morag-graph/src/morag_graph/monitoring/`

**Files to DELETE**:
```bash
rm packages/morag-graph/src/morag_graph/monitoring/normalization_metrics.py
rm packages/morag-graph/src/morag_graph/monitoring/extraction_metrics.py  # If exists
```

### 10. Test Files

**Directory**: `tests/`

**Files to DELETE**:
```bash
# Entity extraction tests
rm tests/test_entity_extractor.py
rm tests/test_pydantic_ai_extraction.py

# Relation extraction tests  
rm tests/test_relation_extractor.py

# OpenIE tests (already removed)
rm tests/test_openie_*.py

# Hybrid extraction tests
rm tests/test_hybrid_extractor.py

# SpaCy tests
rm tests/test_spacy_extractor.py

# Normalization tests
rm tests/test_entity_normalizer.py
rm tests/test_normalization_*.py

# Pattern matching tests
rm tests/test_pattern_matcher.py

# Integration tests for old systems
rm tests/integration/test_extraction_integration.py
rm tests/integration/test_openie_integration.py
```

**Files to KEEP**:
```bash
tests/test_langextract_*.py  # New LangExtract tests
tests/integration/test_langextract_integration.py
```

### 11. Task Files

**Directory**: `tasks/`

**Directories to DELETE**:
```bash
# Remove old task directories that are no longer relevant
rm -rf tasks/openie/
rm -rf tasks/cole-medin/  # If extraction-related
rm -rf tasks/quick-wins/  # If extraction-related
rm -rf tasks/graph-extension/  # If extraction-related
```

**Files to KEEP**:
```bash
tasks/langextract/  # Keep this directory
```

### 12. Documentation Files

**Files to DELETE**:
```bash
# Remove old extraction documentation
rm docs/extraction_old.md  # If exists
rm docs/openie.md  # If exists
rm docs/normalization.md  # If exists
```

### 13. Example Files

**Directory**: `examples/`

**Files to DELETE**:
```bash
rm examples/entity_extraction_example.py  # If exists
rm examples/openie_example.py  # If exists
rm examples/normalization_example.py  # If exists
```

**Files to KEEP**:
```bash
examples/langextract_example.py  # If created
```

### 14. Scripts

**Directory**: `scripts/`

**Files to DELETE**:
```bash
rm scripts/test_extraction.py  # If exists
rm scripts/benchmark_extractors.py  # If exists
rm scripts/cleanup_openie_schema.cypher  # After running it
```

## Implementation Script

**File**: `scripts/cleanup_obsolete_files.py`
```python
#!/usr/bin/env python3
"""Script to clean up obsolete files after LangExtract migration."""

import os
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Files and directories to delete
FILES_TO_DELETE = [
    # Core extraction files
    "packages/morag-graph/src/morag_graph/extraction/entity_extractor.py",
    "packages/morag-graph/src/morag_graph/extraction/relation_extractor.py", 
    "packages/morag-graph/src/morag_graph/extraction/hybrid_extractor.py",
    "packages/morag-graph/src/morag_graph/extraction/spacy_extractor.py",
    "packages/morag-graph/src/morag_graph/extraction/pattern_matcher.py",
    "packages/morag-graph/src/morag_graph/extraction/base.py",
    
    # AI agents
    "packages/morag-graph/src/morag_graph/ai/entity_agent.py",
    "packages/morag-graph/src/morag_graph/ai/relation_agent.py",
    
    # Configuration
    "packages/morag-graph/src/morag_graph/config/normalization_config.py",
    "packages/morag-graph/src/morag_graph/monitoring/normalization_metrics.py",
    
    # Test files
    "tests/test_entity_extractor.py",
    "tests/test_pydantic_ai_extraction.py",
    "tests/test_relation_extractor.py",
    "tests/test_hybrid_extractor.py",
    "tests/test_spacy_extractor.py",
    "tests/test_entity_normalizer.py",
    "tests/test_pattern_matcher.py",
]

DIRECTORIES_TO_DELETE = [
    "packages/morag-graph/src/morag_graph/extractors",
    "packages/morag-graph/src/morag_graph/normalizers", 
    "packages/morag-graph/src/morag_graph/processors",
    "tasks/openie",
]

def delete_file(file_path: Path) -> bool:
    """Delete a file if it exists."""
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    else:
        logger.info(f"File not found (already deleted?): {file_path}")
        return True

def delete_directory(dir_path: Path) -> bool:
    """Delete a directory if it exists."""
    if dir_path.exists():
        try:
            shutil.rmtree(dir_path)
            logger.info(f"Deleted directory: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete directory {dir_path}: {e}")
            return False
    else:
        logger.info(f"Directory not found (already deleted?): {dir_path}")
        return True

def main():
    """Main cleanup function."""
    logger.info("Starting cleanup of obsolete files...")
    
    success_count = 0
    total_count = 0
    
    # Delete files
    for file_path_str in FILES_TO_DELETE:
        file_path = BASE_DIR / file_path_str
        total_count += 1
        if delete_file(file_path):
            success_count += 1
    
    # Delete directories
    for dir_path_str in DIRECTORIES_TO_DELETE:
        dir_path = BASE_DIR / dir_path_str
        total_count += 1
        if delete_directory(dir_path):
            success_count += 1
    
    logger.info(f"Cleanup completed: {success_count}/{total_count} operations successful")
    
    if success_count == total_count:
        logger.info("All obsolete files and directories removed successfully!")
    else:
        logger.warning(f"{total_count - success_count} operations failed. Check logs above.")

if __name__ == "__main__":
    main()
```

## Execution Steps

### Step 1: Create Backup
```bash
# Create backup branch
git checkout -b backup-before-file-cleanup
git add -A
git commit -m "Backup before file cleanup"
```

### Step 2: Run Cleanup Script
```bash
# Make script executable
chmod +x scripts/cleanup_obsolete_files.py

# Run cleanup
python scripts/cleanup_obsolete_files.py
```

### Step 3: Manual Verification
```bash
# Verify key directories are gone
ls packages/morag-graph/src/morag_graph/extractors/  # Should not exist
ls packages/morag-graph/src/morag_graph/normalizers/  # Should not exist

# Verify key files are gone
ls packages/morag-graph/src/morag_graph/extraction/entity_extractor.py  # Should not exist
```

### Step 4: Update Import Statements
Check and update any remaining import statements that might reference deleted files.

### Step 5: Run Tests
```bash
# Run full test suite to ensure nothing is broken
pytest packages/morag-graph/tests/ -v
pytest tests/ -v
```

### Step 6: Clean Git History
```bash
# Add changes
git add -A

# Commit cleanup
git commit -m "Remove obsolete files after LangExtract migration

- Deleted old entity/relation extractors
- Removed OpenIE system components  
- Cleaned up normalization system
- Removed pattern matching components
- Updated test files
- Cleaned up configuration files"
```

## Verification Steps

1. **File System Check**:
   ```bash
   find packages/morag-graph/src/ -name "*extractor*.py" | grep -v langextract
   find packages/morag-graph/src/ -name "*normaliz*.py"
   find packages/morag-graph/src/ -name "*openie*.py"
   ```

2. **Import Check**:
   ```python
   # Should fail - old imports no longer work
   try:
       from morag_graph.extraction.entity_extractor import EntityExtractor
       print("ERROR: Old imports still work!")
   except ImportError:
       print("SUCCESS: Old imports properly removed")
   ```

3. **Test Check**:
   ```bash
   # Should pass - only new tests remain
   pytest tests/ -v --tb=short
   ```

## Success Criteria

- [ ] All obsolete files deleted
- [ ] All obsolete directories removed
- [ ] No broken imports remaining
- [ ] All tests passing
- [ ] Git history clean
- [ ] No references to deleted components
- [ ] LangExtract system fully functional

## Rollback Plan

If issues arise:
```bash
# Restore from backup
git checkout backup-before-file-cleanup
git checkout main
git merge backup-before-file-cleanup
```

## Next Steps

After completing this task:
1. Move to Task 17: Update tests
2. Verify system functionality
3. Update documentation

## Notes

- This is the final cleanup step
- Be careful not to delete files still in use
- Some files might have been deleted in previous tasks
- Always test after cleanup to ensure system works
- Keep LangExtract-related files intact
