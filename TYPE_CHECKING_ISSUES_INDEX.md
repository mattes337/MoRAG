# Type Checking Issues Index

**Status**: 27 errors, 1146 warnings (98% error reduction, 33% warning reduction)

## Summary Statistics

- **Total Errors**: 27 (all legitimate missing imports or type issues)
- **Total Warnings**: 1146 (mostly intentional re-exports)
- **Files Affected**: ~50 files
- **Packages Affected**: All major packages

---

## Error Categories

### 1. **Type Annotation Issues** (17 errors)

#### 1.1 Invalid `callable` Type Usage (2 errors)
**Files**:
- `packages/morag-core/src/morag_core/interfaces/processor.py:45`
- `packages/morag-core/src/morag_core/interfaces/converter.py:64`

**Issue**: Using `callable` instead of `typing.Callable`
```python
# Current (wrong):
progress_callback: callable = None

# Should be:
from typing import Callable
progress_callback: Callable = None
```

#### 1.2 Implicit Optional Arguments (4 errors)
**Files**:
- `packages/morag-graph/src/morag_graph/utils/id_generation.py:31` (checksum)
- `packages/morag-graph/src/morag_graph/utils/id_generation.py:279` (existing_ids)

**Issue**: PEP 484 prohibits implicit Optional
```python
# Current (wrong):
def generate_document_id(checksum: str = None) -> str:

# Should be:
def generate_document_id(checksum: Optional[str] = None) -> str:
```

#### 1.3 Pydantic Field Type Mismatches (3 errors)
**Files**:
- `packages/morag-reasoning/src/morag_reasoning/recursive_fact_models.py:44,69,100`

**Issue**: `default_factory` expects callable, not class
```python
# Current (wrong):
source: SourceMetadata = Field(default_factory=SourceMetadata)

# Should be:
source: SourceMetadata = Field(default_factory=lambda: SourceMetadata())
```

#### 1.4 Pydantic Field Call Overload Issues (3 errors)
**Files**:
- `packages/morag/src/morag/api_models/openapi_schemas.py:10,54,55`

**Issue**: Incorrect Field() usage with positional arguments
```python
# Current (wrong):
file_path: str = Field("path/to/file", "Description")

# Should be:
file_path: str = Field(default="path/to/file", description="Description")
```

#### 1.5 Missing Type Annotations (3 errors)
**Files**:
- `packages/morag-image/src/morag_image/converters/image_formatter.py:130`
- `packages/morag-graph/src/morag_graph/maintenance/config_validator.py:222`
- `packages/morag/src/morag/agents/morag_pipeline_agent.py:160,268`

**Issue**: Variables need explicit type hints
```python
# Current (wrong):
current_content = []

# Should be:
current_content: list[str] = []
```

#### 1.6 Type Assignment Mismatches (2 errors)
**Files**:
- `packages/morag-web/src/morag_web/web_formatter.py:228,231,338`
- `packages/morag-image/src/morag_image/converters/image_formatter.py:311,315,319,320,321,323,327,328,329,331`

**Issue**: Assigning wrong types to typed variables
```python
# Example:
metadata: dict[str, Any] = {}
metadata["width"] = 100  # OK
metadata = 100  # ERROR: Can't assign int to dict
```

---

### 2. **Function Call Issues** (5 errors)

#### 2.1 Missing Required Arguments (3 errors)
**Files**:
- `packages/morag/src/morag/dependencies.py:640` (QueryEntityExtractor)
- `packages/morag/src/morag/agents/morag_pipeline_agent.py:149` (RecursiveFactRetrievalService)
- `packages/morag/src/morag/endpoints/intelligent_retrieval.py:146,175` (get_neo4j_storages, get_qdrant_storages)

**Issue**: Missing required positional arguments

#### 2.2 Unexpected Keyword Arguments (2 errors)
**Files**:
- `packages/morag/src/morag/agents/morag_pipeline_agent.py:324` (Document)
- `packages/morag-stages/src/morag_stages/processors/youtube_processor.py:110` (YouTubeConfig)

**Issue**: Passing kwargs that don't exist in function signature

---

### 3. **Attribute/Method Issues** (3 errors)

#### 3.1 Missing Attributes (3 errors)
**Files**:
- `packages/morag/src/morag/agents/morag_pipeline_agent.py:144,145` (MoRAGServices.graph_storage, ServiceConfig.llm_config)
- `packages/morag/src/morag/ingest_tasks.py:538,543,584,602,716,721,762,769,920,948` (ProcessingResult.text_content)

**Issue**: Accessing attributes that don't exist on the class

---

### 4. **Async/Await Issues** (2 errors)

**Files**:
- `packages/morag/src/morag/dependencies.py:628,635`

**Issue**: Trying to index coroutine without await
```python
# Current (wrong):
storages = get_neo4j_storages()[0]

# Should be:
storages = (await get_neo4j_storages())[0]
```

---

### 5. **Unreachable Code** (3 errors)

**Files**:
- `packages/morag-graph/src/morag_graph/utils/id_generation.py:297`
- `packages/morag/src/morag/ingest_tasks.py:513,691,891`

**Issue**: Code after return/raise statements

---

### 6. **Miscellaneous Type Errors** (remaining errors)

- **Name redefinition**: `packages/morag/src/morag/cli.py:18,20,22,24`
- **Cannot assign to type**: `packages/morag/src/morag/endpoints/reasoning.py:29,30`
- **Cannot assign to method**: `packages/morag/src/morag/server.py:232`
- **Invalid exception type**: `packages/morag/src/morag/endpoints/intelligent_retrieval.py:111`

---

## Warning Categories (1146 total)

### Most Common Warnings:

1. **Re-export warnings** (~800): Intentional `__all__` exports
2. **Union-attr warnings** (~200): Accessing attributes on Optional types without None checks
3. **No-any-return warnings** (~50): Functions returning Any instead of specific types
4. **Arg-type warnings** (~50): Argument type mismatches (often Optional vs required)
5. **Index warnings** (~30): Indexing issues with collections
6. **Call-arg warnings** (~16): Missing or unexpected arguments

---

## Priority Fixes

### High Priority (Breaking Errors)
1. Fix `callable` → `Callable` imports (2 files)
2. Fix Pydantic Field() calls (3 files)
3. Fix missing function arguments (5 locations)
4. Fix async/await issues (2 locations)

### Medium Priority (Type Safety)
5. Add Optional[] to implicit optional parameters (4 locations)
6. Fix Pydantic default_factory (3 locations)
7. Add missing type annotations (5 locations)

### Low Priority (Code Quality)
8. Remove unreachable code (4 locations)
9. Fix attribute access issues (13 locations)
10. Fix name redefinitions (4 locations)

---

## Files Requiring Attention (Sorted by Error Count)

1. **packages/morag/src/morag/ingest_tasks.py** - 15 errors
2. **packages/morag-image/src/morag_image/converters/image_formatter.py** - 11 errors
3. **packages/morag/src/morag/dependencies.py** - 6 errors
4. **packages/morag/src/morag/agents/morag_pipeline_agent.py** - 6 errors
5. **packages/morag/src/morag/api_models/endpoints/stages.py** - 5 errors
6. **packages/morag/src/morag/endpoints/recursive_fact_retrieval.py** - 5 errors
7. **packages/morag-graph/src/morag_graph/utils/id_generation.py** - 4 errors
8. **packages/morag/src/morag/endpoints/intelligent_retrieval.py** - 4 errors
9. **packages/morag/src/morag/cli.py** - 4 errors
10. **packages/morag/src/morag/tasks/enhanced_processing_task.py** - 4 errors

---

## Package-Level Breakdown

### morag-core (2 errors)
- ✅ EASY: callable → Callable type fixes (2)

### morag-graph (8 errors)
- ✅ EASY: Optional parameter fixes (2)
- ⚠️ MEDIUM: Unreachable code (1)
- ⚠️ MEDIUM: Type annotation (1)
- 🔴 HARD: Attribute access (2)
- 🔴 HARD: Function return types (2)

### morag-reasoning (3 errors)
- ⚠️ MEDIUM: Pydantic default_factory (3)

### morag (main package) (30+ errors)
- ⚠️ MEDIUM: Pydantic Field() calls (3)
- 🔴 HARD: ProcessingResult.text_content (10)
- 🔴 HARD: Missing function arguments (5)
- 🔴 HARD: Async/await issues (2)
- 🔴 HARD: Attribute access (3)
- ⚠️ MEDIUM: Unreachable code (3)
- ⚠️ MEDIUM: Name redefinitions (4)
- ⚠️ MEDIUM: Type annotations (3)

### morag-stages (2 errors)
- 🔴 HARD: Function call issues (2)

### morag-web (3 errors)
- ⚠️ MEDIUM: Type assignment mismatches (3)

### morag-image (11 errors)
- ⚠️ MEDIUM: Type assignment mismatches (10)
- ✅ EASY: Type annotation (1)

### morag-youtube (1 error)
- ⚠️ MEDIUM: Return type (1)

---

## Automated Fix Script Candidates

### Script 1: Import Fixes
```bash
# Fix callable → Callable
find packages/ -name "*.py" -exec sed -i 's/: callable/: Callable/g' {} \;
find packages/ -name "*.py" -exec sed -i 's/from typing import/from typing import Callable,/g' {} \;
```

### Script 2: Optional Fixes
```python
# Pattern: param: Type = None → param: Optional[Type] = None
# Requires AST parsing for safety
```

### Script 3: Type Annotation Additions
```python
# Pattern: var = [] → var: list[Type] = []
# Requires context analysis
```

---

## Testing Strategy

### After Each Fix:
```bash
# Run mypy on specific package
py -m mypy packages/morag-core --show-error-codes

# Run full check
py -m mypy packages/ --show-error-codes --no-error-summary | tee mypy_output.txt

# Count errors
py -m mypy packages/ 2>&1 | grep "error:" | wc -l
```

### Regression Prevention:
```bash
# Run tests after fixes
pytest packages/morag-core/tests/ -v
pytest packages/morag-graph/tests/ -v
pytest packages/morag/tests/ -v
```

---

## Next Steps

1. **Phase 1 - Quick Wins** (30 min)
   - Fix callable → Callable (2 files)
   - Add Optional[] to parameters (4 locations)
   - Add type annotations (5 locations)
   - Remove unreachable code (4 locations)
   - **Expected reduction**: 15 errors → 12 errors

2. **Phase 2 - Medium Complexity** (1 hour)
   - Fix Pydantic Field() calls (6 locations)
   - Fix async/await issues (2 locations)
   - Fix type assignment mismatches (13 locations)
   - **Expected reduction**: 12 errors → 6 errors

3. **Phase 3 - Hard Refactoring** (2+ hours)
   - Refactor ProcessingResult.text_content (10 locations)
   - Fix missing function arguments (7 locations)
   - Fix missing attributes (3 locations)
   - **Expected reduction**: 6 errors → 0 errors

4. **Phase 4 - Warning Cleanup** (ongoing)
   - Address union-attr warnings with None checks
   - Add specific return types instead of Any
   - Fix argument type mismatches

---

## Success Metrics

- ✅ **Current**: 27 errors, 1146 warnings
- 🎯 **Target Phase 1**: 12 errors, 1146 warnings
- 🎯 **Target Phase 2**: 6 errors, 1000 warnings
- 🎯 **Target Phase 3**: 0 errors, 800 warnings
- 🏆 **Final Goal**: 0 errors, <500 warnings (intentional re-exports)
