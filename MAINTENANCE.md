# MoRAG Codebase Maintenance Report

**Generated:** 2025-07-22
**Updated:** 2025-07-22
**Scope:** All packages in the MoRAG codebase
**Focus:** Current maintenance issues requiring attention

## Executive Summary

This report tracks ongoing maintenance issues in the MoRAG codebase. Previous critical issues (text converter fixes, obsolete dependencies) have been resolved. Current focus is on large file refactoring and new issues that have emerged.

## 🔍 Current Status

### ✅ Recently Completed
- **Text converter fixes**: Custom markdown logic removed, imports fixed
- **Obsolete dependencies**: Removed 6 unused libraries from morag-document
- **Server.py refactoring**: Reduced from 1,286 lines to 247 lines ✅
- **Neo4j storage refactoring**: Reduced from 1,560 lines to 389 lines ✅

### ⚠️ Open Issues Requiring Attention

## 1. New Large Files Exceeding Limits

### Hard Cap Violations (>1000 lines)
1. **`packages/morag/src/morag/ingestion_coordinator.py`** - **1,500 lines**
   - **Issue**: New large file for ingestion coordination
   - **Impact**: Complex maintenance and testing
   - **Priority**: High - Split into smaller modules

2. **`packages/morag-graph/src/morag_graph/storage/qdrant_storage.py`** - **1,001 lines**
   - **Issue**: Just over the hard cap limit
   - **Impact**: Maintenance complexity
   - **Priority**: Medium - Minor refactoring needed

### Soft Cap Violations (>500 lines)
1. **`packages/morag-services/src/morag_services/services.py`** - **880 lines**
   - **Issue**: Large service coordination file
   - **Priority**: Medium - Consider splitting by service type

2. **`packages/morag-services/src/morag_services/storage.py`** - **857 lines**
   - **Issue**: Large storage abstraction file
   - **Priority**: Medium - Consider splitting by storage type

3. **`packages/morag-graph/src/morag_graph/storage/json_storage.py`** - **834 lines**
   - **Issue**: Large JSON storage implementation
   - **Priority**: Low - Acceptable for storage implementation

4. **`packages/morag/src/morag/ingest_tasks.py`** - **785 lines**
   - **Issue**: Large task coordination file
   - **Priority**: Medium - Consider splitting by task type

5. **`packages/morag-document/src/morag_document/converters/markitdown_base.py`** - **775 lines**
   - **Issue**: Large base converter implementation
   - **Priority**: Low - Acceptable for base class

6. **`packages/morag-video/src/morag_video/processor.py`** - **765 lines**
   - **Issue**: Large video processing file
   - **Priority**: Medium - Consider splitting by processing type

## 2. Root Level Dependencies Still Need Review

### Main Requirements Analysis
**File**: `requirements.txt`

**Office Document Dependencies (Lines 52-53):**
- `python-docx>=1.1.2,<2.0.0` and `openpyxl>=3.1.5,<4.0.0`
- **Status**: May be redundant since markitdown handles these formats
- **Priority**: Low - Test if markitdown coverage is sufficient

## 📋 Current Maintenance Priorities

### Priority 1 (High)
1. **Refactor ingestion_coordinator.py** - Split 1,500-line file into modules
2. **Minor refactor qdrant_storage.py** - Reduce from 1,001 lines to under 1,000

### Priority 2 (Medium)
1. **Review large service files** - Consider splitting services.py and storage.py
2. **Review task coordination** - Consider splitting ingest_tasks.py
3. **Review video processing** - Consider splitting video processor

### Priority 3 (Low)
1. **Test root-level dependencies** - Verify if office document libraries are still needed
2. **Monitor file growth** - Prevent new files from exceeding limits
3. **Standardize dependency management** - Consistent versioning across packages

## 🔧 Recommended Actions

### Immediate Actions (Priority 1)
1. **Split ingestion_coordinator.py**:
   - Extract ingestion strategies to separate modules
   - Extract coordination logic to separate classes
   - Target: Reduce to under 500 lines

2. **Minor refactor qdrant_storage.py**:
   - Extract utility functions to separate module
   - Target: Reduce to under 1,000 lines

### Medium-term Actions (Priority 2)
1. **Review and potentially split**:
   - `morag-services/services.py` (880 lines)
   - `morag-services/storage.py` (857 lines)
   - `morag/ingest_tasks.py` (785 lines)
   - `morag-video/processor.py` (765 lines)

### Long-term Actions (Priority 3)
1. **Dependency cleanup**:
   - Test markitdown-only document conversion
   - Remove unused office document dependencies if possible
2. **Monitoring**:
   - Add file size checks to CI/CD pipeline
   - Regular maintenance reviews

---

## � Maintenance History

### ✅ Completed Fixes (Previous Maintenance Cycles)

1. **Fixed text converter issues** ✅
   - **File**: `packages/morag-document/src/morag_document/converters/text.py`
   - **Action**: Removed custom markdown/HTML processing logic
   - **Result**: Reduced from 186 lines to 34 lines (-82%)
   - **Impact**: Eliminated runtime errors from missing imports

2. **Removed obsolete dependencies** ✅
   - **File**: `packages/morag-document/pyproject.toml`
   - **Removed**: 6 obsolete libraries (`pypdf`, `python-docx`, `openpyxl`, `python-pptx`, `docling`, `markdown`)
   - **Result**: Reduced package dependencies by 55%

3. **Completed server.py refactoring** ✅
   - **Before**: 1,286 lines (monolithic)
   - **After**: 247 lines (modular with endpoint routers)
   - **Improvement**: -81% reduction, much better maintainability

4. **Completed neo4j_storage.py refactoring** ✅
   - **Before**: 1,560 lines (monolithic)
   - **After**: 389 lines (delegated to operation classes)
   - **Improvement**: -75% reduction, modular architecture

### � Overall Progress

| Metric | Previous State | Current State | Improvement |
|--------|---------------|---------------|-------------|
| Files >1000 lines | 2 critical files | 2 new files | Resolved previous, new issues emerged |
| morag-document deps | 11 libraries | 5 libraries | -55% |
| Text converter quality | Runtime errors | Stable | ✅ Fixed |
| Server maintainability | Poor (1,286 lines) | Good (247 lines) | ✅ Excellent |
| Neo4j maintainability | Poor (1,560 lines) | Good (389 lines) | ✅ Excellent |

---

## 🎯 Next Steps

**Immediate Focus**: Address the 2 new large files that have emerged:
1. `ingestion_coordinator.py` (1,500 lines) - **Priority 1**
2. `qdrant_storage.py` (1,001 lines) - **Priority 1**

**Status**: Previous critical issues resolved ✅. New maintenance cycle needed for emerging large files.
