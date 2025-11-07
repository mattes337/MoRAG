# Type Checking - Detailed Fixes

## Quick Reference

| Category | Count | Difficulty | Time Est. |
|----------|-------|------------|-----------|
| Import fixes (callable → Callable) | 2 | Easy | 5 min |
| Optional parameter fixes | 4 | Easy | 10 min |
| Pydantic Field() fixes | 6 | Medium | 20 min |
| Missing type annotations | 5 | Easy | 10 min |
| Function call fixes | 7 | Hard | 60 min |
| Attribute fixes | 13 | Hard | 90 min |
| Async/await fixes | 2 | Medium | 15 min |
| Unreachable code removal | 4 | Easy | 5 min |
| Miscellaneous | 4 | Medium | 30 min |
| **TOTAL** | **47** | **Mixed** | **~4 hours** |

---

## 1. Import Fixes (2 errors) ⚡ QUICK WIN

### File: `packages/morag-core/src/morag_core/interfaces/processor.py`
**Line 45**: `progress_callback: callable`

```python
# Add to imports at top:
from typing import Callable, Optional

# Change line 45:
- progress_callback: callable = None
+ progress_callback: Optional[Callable] = None
```

### File: `packages/morag-core/src/morag_core/interfaces/converter.py`
**Line 64**: `progress_callback: callable`

```python
# Add to imports at top:
from typing import Callable, Optional

# Change line 64:
- progress_callback: callable = None
+ progress_callback: Optional[Callable] = None
```

---

## 2. Optional Parameter Fixes (4 errors) ⚡ QUICK WIN

### File: `packages/morag-graph/src/morag_graph/utils/id_generation.py`

**Line 31**: `checksum: str = None`
```python
from typing import Optional

# Change:
- def generate_document_id(self, collection_name: str, source_path: str, checksum: str = None) -> str:
+ def generate_document_id(self, collection_name: str, source_path: str, checksum: Optional[str] = None) -> str:
```

**Line 279**: `existing_ids: list[str] = None`
```python
# Change:
- def check_collision(self, new_id: str, existing_ids: list[str] = None) -> CollisionCheckResult:
+ def check_collision(self, new_id: str, existing_ids: Optional[list[str]] = None) -> CollisionCheckResult:
```

**Line 297**: Remove unreachable code
```python
# Find the unreachable statement and remove it
```

---

## 3. Pydantic Field Fixes (6 errors) 🔧 MEDIUM

### File: `packages/morag-reasoning/src/morag_reasoning/recursive_fact_models.py`

**Lines 44, 69, 100**: `default_factory=SourceMetadata`

```python
# Change all three occurrences:
- source: SourceMetadata = Field(default_factory=SourceMetadata)
+ source: SourceMetadata = Field(default_factory=lambda: SourceMetadata())
```

### File: `packages/morag/src/morag/api_models/openapi_schemas.py`

**Line 10**: Field with positional args
```python
# Find the field definition and change to:
- field_name: str = Field("default_value", "Description text")
+ field_name: str = Field(default="default_value", description="Description text")
```

**Lines 54, 55**: Same issue
```python
# Apply same fix pattern
```

---

## 4. Missing Type Annotations (5 errors) ⚡ QUICK WIN

### File: `packages/morag-image/src/morag_image/converters/image_formatter.py`
**Line 130**:
```python
- current_content = []
+ current_content: list[str] = []
```

### File: `packages/morag-graph/src/morag_graph/maintenance/config_validator.py`
**Line 222**:
```python
- warnings = []
+ warnings: list[str] = []
```

### File: `packages/morag/src/morag/agents/morag_pipeline_agent.py`
**Line 160**:
```python
- stage_timings = {}
+ stage_timings: dict[str, float] = {}
```

**Line 268**:
```python
- intermediate_files = []
+ intermediate_files: list[str] = []
```

---

## 5. Function Call Fixes (7 errors) 🔴 HARD - REQUIRES INVESTIGATION

### File: `packages/morag/src/morag/dependencies.py`

**Line 640**: Missing arguments to QueryEntityExtractor
```python
# INVESTIGATE: What are the required arguments?
# Check QueryEntityExtractor.__init__ signature
# Add missing: entity_extractor, graph_storage
```

**Lines 628, 635**: Missing await
```python
# Change:
- neo4j_storages = get_neo4j_storages()[0]
+ neo4j_storages = (await get_neo4j_storages())[0]

- qdrant_storages = get_qdrant_storages()[0]
+ qdrant_storages = (await get_qdrant_storages())[0]
```

### File: `packages/morag/src/morag/agents/morag_pipeline_agent.py`

**Line 149**: Missing arguments to RecursiveFactRetrievalService
```python
# INVESTIGATE: Check RecursiveFactRetrievalService.__init__
# Add missing: llm_client, neo4j_storage, qdrant_storage
```

**Line 324**: Unexpected kwargs to Document
```python
# INVESTIGATE: Check Document.__init__ signature
# Remove or rename: title, content
```

### File: `packages/morag/src/morag/endpoints/intelligent_retrieval.py`

**Lines 146, 175**: Missing database_servers argument
```python
# INVESTIGATE: Where should database_servers come from?
# Add the required argument
```

---

## 6. Attribute Fixes (13 errors) 🔴 HARD - REQUIRES REFACTORING

### File: `packages/morag/src/morag/ingest_tasks.py`

**Lines 538, 543, 584, 602, 716, 721, 762, 769, 920, 948**: ProcessingResult.text_content

**ROOT CAUSE**: ProcessingResult class doesn't have `text_content` attribute

**SOLUTION OPTIONS**:
1. Add `text_content` property to ProcessingResult
2. Change all references to use correct attribute name
3. Create adapter/wrapper

```python
# Option 1: Add to ProcessingResult class
@property
def text_content(self) -> Optional[str]:
    return self.content  # or whatever the correct attribute is
```

### File: `packages/morag/src/morag/agents/morag_pipeline_agent.py`

**Line 144**: MoRAGServices.graph_storage
**Line 145**: ServiceConfig.llm_config

```python
# INVESTIGATE: Check if these attributes exist or need to be added
```

---

## 7. Async/Await Fixes (2 errors) 🔧 MEDIUM

### File: `packages/morag/src/morag/dependencies.py`

**Line 628**:
```python
- neo4j_storages = get_neo4j_storages()[0]
+ neo4j_storages = (await get_neo4j_storages())[0]
```

**Line 635**:
```python
- qdrant_storages = get_qdrant_storages()[0]
+ qdrant_storages = (await get_qdrant_storages())[0]
```

---

## 8. Unreachable Code (4 errors) ⚡ QUICK WIN

### Files to check:
1. `packages/morag-graph/src/morag_graph/utils/id_generation.py:297`
2. `packages/morag/src/morag/ingest_tasks.py:513`
3. `packages/morag/src/morag/ingest_tasks.py:691`
4. `packages/morag/src/morag/ingest_tasks.py:891`

**Action**: Remove or comment out unreachable statements

---

## 9. Miscellaneous (4 errors) 🔧 MEDIUM

### File: `packages/morag/src/morag/cli.py`
**Lines 18, 20, 22, 24**: Name redefinition

```python
# INVESTIGATE: Are these imports conflicting with local definitions?
# Rename local variables or use aliases for imports
```

### File: `packages/morag/src/morag/endpoints/reasoning.py`
**Lines 29, 30**: Cannot assign to type

```python
# INVESTIGATE: Trying to assign None to a type
# This is likely a pattern issue - check what's being attempted
```

### File: `packages/morag/src/morag/server.py`
**Line 232**: Cannot assign to method

```python
# INVESTIGATE: Trying to assign to a method
# This is incorrect - methods can't be reassigned
```

### File: `packages/morag/src/morag/endpoints/intelligent_retrieval.py`
**Line 111**: Invalid exception type

```python
# INVESTIGATE: Exception must derive from BaseException
# Check what's being raised
```

---

## Execution Plan

### Phase 1: Quick Wins (30 minutes)
1. ✅ Fix callable → Callable (2 files)
2. ✅ Add Optional[] to parameters (2 locations)
3. ✅ Add type annotations (5 locations)
4. ✅ Remove unreachable code (4 locations)

### Phase 2: Medium Complexity (1 hour)
5. 🔧 Fix Pydantic Field() calls (3 files, 6 locations)
6. 🔧 Fix async/await issues (2 locations)
7. 🔧 Investigate miscellaneous errors (4 locations)

### Phase 3: Hard Refactoring (2+ hours)
8. 🔴 Fix ProcessingResult.text_content (10 locations)
9. 🔴 Fix missing function arguments (7 locations)
10. 🔴 Fix missing attributes (3 locations)
