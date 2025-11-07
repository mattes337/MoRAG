# Type Checking - Action Plan

## Executive Summary

**Current Status**: 27 errors, 1146 warnings (98% error reduction from initial state!)

**Breakdown**:
- ✅ **15 Quick Wins** (30 minutes) - Simple find/replace or one-line fixes
- ⚠️ **21 Medium Tasks** (1-2 hours) - Requires understanding context
- 🔴 **20 Hard Tasks** (2-4 hours) - Requires refactoring or investigation

**Total Estimated Time**: 4-6 hours to zero errors

---

## Immediate Actions (Do First)

### 1. Create Backup Branch
```bash
git checkout -b fix/type-checking-errors
git push -u origin fix/type-checking-errors
```

### 2. Run Baseline Test
```bash
py -m mypy packages/ --show-error-codes --no-error-summary 2>&1 | tee baseline_errors.txt
grep "error:" baseline_errors.txt | wc -l  # Should show 27
```

---

## Phase 1: Quick Wins (30 minutes) ✅

### Task 1.1: Fix `callable` → `Callable` (5 min)
**Files**: 2
- `packages/morag-core/src/morag_core/interfaces/processor.py:45`
- `packages/morag-core/src/morag_core/interfaces/converter.py:64`

**Commands**:
```bash
# Manual fix recommended (only 2 files)
# Add: from typing import Callable, Optional
# Change: callable → Optional[Callable]
```

**Verification**:
```bash
py -m mypy packages/morag-core --show-error-codes | grep "callable"
# Should return nothing
```

### Task 1.2: Add Optional[] to Parameters (10 min)
**Files**: 1
- `packages/morag-graph/src/morag_graph/utils/id_generation.py:31,279`

**Fix**:
```python
from typing import Optional

# Line 31:
def generate_document_id(..., checksum: Optional[str] = None) -> str:

# Line 279:
def check_collision(..., existing_ids: Optional[list[str]] = None) -> CollisionCheckResult:
```

**Verification**:
```bash
py -m mypy packages/morag-graph/src/morag_graph/utils/id_generation.py
```

### Task 1.3: Add Type Annotations (10 min)
**Files**: 3
- `packages/morag-image/src/morag_image/converters/image_formatter.py:130`
- `packages/morag-graph/src/morag_graph/maintenance/config_validator.py:222`
- `packages/morag/src/morag/agents/morag_pipeline_agent.py:160,268`

**Pattern**:
```python
# Before:
var = []
var = {}

# After:
var: list[str] = []
var: dict[str, float] = {}
```

### Task 1.4: Remove Unreachable Code (5 min)
**Files**: 2
- `packages/morag-graph/src/morag_graph/utils/id_generation.py:297`
- `packages/morag/src/morag/ingest_tasks.py:513,691,891`

**Action**: Comment out or remove the unreachable statements

**Expected Result**: 15 errors fixed → **12 errors remaining**

---

## Phase 2: Medium Complexity (1-2 hours) ⚠️

### Task 2.1: Fix Pydantic Field() Calls (20 min)
**Files**: 2
- `packages/morag-reasoning/src/morag_reasoning/recursive_fact_models.py:44,69,100`
- `packages/morag/src/morag/api_models/openapi_schemas.py:10,54,55`

**Fix Pattern 1** (default_factory):
```python
# Before:
source: SourceMetadata = Field(default_factory=SourceMetadata)

# After:
source: SourceMetadata = Field(default_factory=lambda: SourceMetadata())
```

**Fix Pattern 2** (positional args):
```python
# Before:
field: str = Field("default", "description")

# After:
field: str = Field(default="default", description="description")
```

### Task 2.2: Fix Async/Await (15 min)
**Files**: 1
- `packages/morag/src/morag/dependencies.py:628,635`

**Fix**:
```python
# Before:
neo4j_storages = get_neo4j_storages()[0]

# After:
neo4j_storages = (await get_neo4j_storages())[0]
```

### Task 2.3: Fix Type Assignment Mismatches (30 min)
**Files**: 2
- `packages/morag-web/src/morag_web/web_formatter.py:228,231,338`
- `packages/morag-image/src/morag_image/converters/image_formatter.py:311,315,319,320,321,323,327,328,329,331`

**Investigation Required**: Check what types are expected vs provided

**Expected Result**: 21 errors fixed → **6 errors remaining**

---

## Phase 3: Hard Refactoring (2-4 hours) 🔴

### Task 3.1: Fix ProcessingResult.text_content (90 min)
**Files**: 1 (10 occurrences)
- `packages/morag/src/morag/ingest_tasks.py`

**Root Cause**: ProcessingResult class missing `text_content` attribute

**Investigation Steps**:
1. Find ProcessingResult class definition
2. Check what attribute should be used instead
3. Either add property or refactor all usages

**Options**:
```python
# Option A: Add property to ProcessingResult
@property
def text_content(self) -> Optional[str]:
    return self.content  # or whatever is correct

# Option B: Refactor all call sites
result.text_content → result.content
```

### Task 3.2: Fix Missing Function Arguments (60 min)
**Files**: 4
- `packages/morag/src/morag/dependencies.py:640`
- `packages/morag/src/morag/agents/morag_pipeline_agent.py:149,324`
- `packages/morag/src/morag/endpoints/intelligent_retrieval.py:146,175`
- `packages/morag-stages/src/morag_stages/processors/youtube_processor.py:110`

**Investigation Required**: For each error:
1. Find the function/class definition
2. Check required parameters
3. Determine where to get the missing values
4. Update call site

### Task 3.3: Fix Missing Attributes (30 min)
**Files**: 1
- `packages/morag/src/morag/agents/morag_pipeline_agent.py:144,145`

**Investigation**:
- Check if MoRAGServices should have graph_storage
- Check if ServiceConfig should have llm_config
- Add attributes or refactor code

### Task 3.4: Fix Miscellaneous (30 min)
**Files**: 3
- `packages/morag/src/morag/cli.py:18,20,22,24` (name redefinition)
- `packages/morag/src/morag/endpoints/reasoning.py:29,30` (assign to type)
- `packages/morag/src/morag/server.py:232` (assign to method)
- `packages/morag/src/morag/endpoints/intelligent_retrieval.py:111` (invalid exception)

**Expected Result**: All errors fixed → **0 errors remaining**

---

## Verification Checklist

After each phase:
- [ ] Run mypy on affected packages
- [ ] Count remaining errors
- [ ] Run unit tests
- [ ] Commit changes with descriptive message

Final verification:
- [ ] `py -m mypy packages/ --show-error-codes` shows 0 errors
- [ ] All tests pass: `pytest packages/ -v`
- [ ] No new runtime errors introduced
- [ ] Create PR with summary of changes

---

## Git Commit Strategy

```bash
# After Phase 1:
git add packages/morag-core packages/morag-graph packages/morag-image
git commit -m "fix(types): Phase 1 - Quick wins (callable, Optional, annotations, unreachable code)"

# After Phase 2:
git add packages/morag-reasoning packages/morag
git commit -m "fix(types): Phase 2 - Pydantic Field fixes, async/await, type assignments"

# After Phase 3:
git add packages/morag packages/morag-stages
git commit -m "fix(types): Phase 3 - ProcessingResult refactor, function arguments, attributes"

# Final:
git push origin fix/type-checking-errors
# Create PR
```

---

## Success Criteria

✅ **Phase 1 Complete**: 12 errors remaining (from 27)
✅ **Phase 2 Complete**: 6 errors remaining (from 12)
✅ **Phase 3 Complete**: 0 errors remaining (from 6)
🏆 **Final Goal**: Clean mypy run with only intentional warnings
