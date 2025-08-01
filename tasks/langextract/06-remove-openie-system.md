# Task 6: Remove OpenIE System

## Objective
Completely remove the OpenIE system from MoRAG, including all related files, dependencies, configurations, and database operations. LangExtract will replace OpenIE's relation extraction capabilities.

## Prerequisites
- Tasks 1-5 completed (LangExtract system working)
- Understanding of current OpenIE integration points
- Database backup (recommended)

## OpenIE Components to Remove

### 1. Core OpenIE Files

**Files to Delete**:
```
packages/morag-graph/src/morag_graph/extractors/
├── openie_extractor.py
├── __init__.py (update imports)

packages/morag-graph/src/morag_graph/services/
├── openie_service.py

packages/morag-graph/src/morag_graph/storage/neo4j_operations/
├── openie_operations.py

packages/morag-graph/src/morag_graph/processors/
├── openie_processor.py (if exists)
├── triplet_processor.py (if exists)
├── sentence_processor.py (if exists)

packages/morag-graph/src/morag_graph/models/
├── openie_models.py (if exists)
```

### 2. Remove OpenIE from Graph Builders

**File**: `packages/morag-graph/src/morag_graph/builders/enhanced_graph_builder.py`

Remove OpenIE-related code:
```python
# REMOVE these imports
from ..extractors import OpenIEExtractor  # DELETE

# REMOVE these initialization parameters
def __init__(
    self,
    # ... other params
    openie_enabled: bool = False,  # DELETE
    openie_extractor: Optional[OpenIEExtractor] = None,  # DELETE
    # ... other params
):
    # REMOVE these assignments
    self.openie_enabled = openie_enabled  # DELETE
    self.openie_extractor = openie_extractor  # DELETE

# REMOVE entire OpenIE extraction sections
async def build_graph(self, content: str, document_id: str, **kwargs):
    # ... existing code ...
    
    # DELETE this entire section (lines ~257-282)
    # Step 3: Extract relations using OpenIE (if enabled)
    openie_relations = []
    openie_result = None
    
    if self.openie_enabled and self.openie_extractor:
        try:
            self.logger.debug("Extracting relations with OpenIE...")
            openie_result = await self.openie_extractor.extract_full(
                content,
                entities=entities,
                source_doc_id=document_id
            )
            openie_relations = openie_result.relations
            
            # Store OpenIE triplets in Neo4j if available
            if isinstance(self.storage, Neo4jStorage) and openie_result.triplets:
                await self.storage.store_openie_triplets(
                    openie_result.triplets,
                    openie_result.entity_matches,
                    openie_result.normalized_predicates,
                    document_id
                )
                
        except Exception as e:
            self.logger.error(f"OpenIE extraction failed: {e}")
            # Continue with LLM-only results
    # END DELETE SECTION

    # DELETE similar sections in chunk processing (~412-430)
```

**File**: `packages/morag-graph/src/morag_graph/builders/graph_builder.py`

Remove any OpenIE references if present.

### 3. Remove OpenIE from Storage Layer

**File**: `packages/morag-graph/src/morag_graph/storage/neo4j_storage.py`

Remove OpenIE operations:
```python
# REMOVE these imports
try:
    from .neo4j_operations import OpenIEOperations  # DELETE
    _OPENIE_AVAILABLE = True  # DELETE
except ImportError:  # DELETE
    _OPENIE_AVAILABLE = False  # DELETE
    OpenIEOperations = None  # DELETE

class Neo4jStorage(BaseStorage):
    def __init__(self, config: Neo4jConfig):
        # ... existing code ...
        
        # REMOVE OpenIE operations initialization
        self._openie_ops: Optional[OpenIEOperations] = None  # DELETE

    async def connect(self) -> None:
        # ... existing code ...
        
        # REMOVE OpenIE operations setup
        if _OPENIE_AVAILABLE:  # DELETE
            self._openie_ops = OpenIEOperations(self.driver, self.config.database)  # DELETE

    # DELETE entire method
    async def initialize_openie_schema(self) -> None:
        """Initialize OpenIE schema in Neo4j."""
        if not self._openie_ops:
            if not _OPENIE_AVAILABLE:
                raise RuntimeError("OpenIE operations not available")
            raise RuntimeError("Connection not initialized")
        await self._openie_ops.initialize_schema()

    # DELETE entire method
    async def store_openie_triplets(
        self,
        triplets,
        entity_matches=None,
        normalized_predicates=None,
        source_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Store OpenIE triplets in Neo4j."""
        if not self._openie_ops:
            if not _OPENIE_AVAILABLE:
                raise RuntimeError("OpenIE operations not available")
            raise RuntimeError("Connection not initialized")
        return await self._openie_ops.store_triplets(
            triplets, entity_matches, normalized_predicates, source_doc_id
        )
```

### 4. Remove OpenIE Configuration

**File**: `packages/morag-core/src/morag_core/config.py`

Remove OpenIE configuration:
```python
# DELETE these configuration fields
openie_enabled: bool = Field(
    default=True,
    alias="MORAG_OPENIE_ENABLED",
    description="Enable OpenIE relation extraction"
)

openie_implementation: str = Field(
    default="stanford",
    alias="MORAG_OPENIE_IMPLEMENTATION", 
    description="OpenIE implementation to use (stanford, openie5, etc.)"
)

openie_confidence_threshold: float = Field(
    default=0.7,
    alias="MORAG_OPENIE_CONFIDENCE_THRESHOLD",
    ge=0.0,
    le=1.0,
    description="Minimum confidence threshold for OpenIE triplets"
)

openie_max_triplets_per_sentence: int = Field(
    default=10,
    alias="MORAG_OPENIE_MAX_TRIPLETS_PER_SENTENCE",
    ge=1,
    le=50,
    description="Maximum number of triplets to extract per sentence"
)
```

### 5. Update Import Statements

**File**: `packages/morag-graph/src/morag_graph/extraction/__init__.py`

Remove OpenIE imports:
```python
# DELETE these lines
try:
    from ..extractors import OpenIEExtractor  # DELETE
    _OPENIE_AVAILABLE = True  # DELETE
except ImportError:  # DELETE
    _OPENIE_AVAILABLE = False  # DELETE
    OpenIEExtractor = None  # DELETE
```

**File**: `packages/morag-graph/src/morag_graph/__init__.py`

Remove any OpenIE exports.

### 6. Remove OpenIE Dependencies

**File**: `packages/morag-graph/requirements.txt`

Remove OpenIE-related dependencies:
```txt
# DELETE these if present
stanford-openie
allennlp
allennlp-models
openie
```

**File**: `packages/morag-graph/pyproject.toml`

Remove from dependencies list.

### 7. Clean Up Database Schema

**SQL Script**: `scripts/cleanup_openie_schema.cypher`
```cypher
// Remove OpenIE-specific nodes and relationships
MATCH (t:OpenIETriplet)
DETACH DELETE t;

MATCH (p:OpenIEPredicate)
DETACH DELETE p;

// Remove OpenIE-specific relationships
MATCH ()-[r:OPENIE_SUBJECT|OPENIE_OBJECT|OPENIE_PREDICATE]-()
DELETE r;

// Remove OpenIE-specific indexes
DROP INDEX openie_triplet_id IF EXISTS;
DROP INDEX openie_predicate_normalized IF EXISTS;

// Remove OpenIE-specific constraints
DROP CONSTRAINT openie_triplet_unique IF EXISTS;
```

### 8. Update Tests

**Files to Update/Remove**:
```
tests/test_openie_extractor.py  # DELETE
tests/test_openie_service.py    # DELETE
tests/test_openie_operations.py # DELETE
tests/integration/test_openie_integration.py  # DELETE
```

**Files to Update**:
```
tests/test_enhanced_graph_builder.py  # Remove OpenIE test cases
tests/test_neo4j_storage.py          # Remove OpenIE storage tests
```

### 9. Update Documentation

**Files to Update**:
```
packages/morag-graph/README.md        # Remove OpenIE sections
docs/extraction.md                    # Remove OpenIE documentation
docs/configuration.md                 # Remove OpenIE config docs
```

## Implementation Steps

### Step 1: Backup and Preparation
```bash
# Create backup branch
git checkout -b backup-before-openie-removal

# Create database backup
neo4j-admin dump --database=neo4j --to=backup-before-openie-removal.dump
```

### Step 2: Remove Files
```bash
# Remove core OpenIE files
rm packages/morag-graph/src/morag_graph/extractors/openie_extractor.py
rm packages/morag-graph/src/morag_graph/services/openie_service.py
rm packages/morag-graph/src/morag_graph/storage/neo4j_operations/openie_operations.py

# Remove test files
rm tests/test_openie_*.py
rm tests/integration/test_openie_*.py
```

### Step 3: Update Code Files
Use the code changes specified above to remove OpenIE references from:
- Graph builders
- Storage layer
- Configuration
- Import statements

### Step 4: Clean Database
```bash
# Run cleanup script
cypher-shell -f scripts/cleanup_openie_schema.cypher
```

### Step 5: Update Dependencies
```bash
# Remove from requirements
pip uninstall stanford-openie allennlp allennlp-models

# Update requirements files
# (Remove OpenIE dependencies)
```

### Step 6: Run Tests
```bash
# Run full test suite
pytest packages/morag-graph/tests/ -v

# Run integration tests
pytest tests/integration/ -v
```

## Verification Steps

1. **Code Verification**:
   ```bash
   # Search for remaining OpenIE references
   grep -r "openie\|OpenIE" packages/morag-graph/src/ --exclude-dir=__pycache__
   grep -r "triplet" packages/morag-graph/src/ --exclude-dir=__pycache__
   ```

2. **Import Verification**:
   ```python
   # Should not find any OpenIE imports
   from morag_graph.extraction import *
   # Should not have OpenIEExtractor
   ```

3. **Database Verification**:
   ```cypher
   // Should return 0
   MATCH (t:OpenIETriplet) RETURN count(t);
   MATCH (p:OpenIEPredicate) RETURN count(p);
   ```

4. **Configuration Verification**:
   ```python
   from morag_core.config import get_settings
   settings = get_settings()
   # Should not have openie_* attributes
   ```

## Success Criteria

- [ ] All OpenIE files deleted
- [ ] All OpenIE code references removed
- [ ] All OpenIE dependencies uninstalled
- [ ] All OpenIE database objects removed
- [ ] All OpenIE tests removed/updated
- [ ] No remaining OpenIE imports
- [ ] All tests passing
- [ ] Documentation updated

## Rollback Plan

If issues arise:
1. Restore from backup branch: `git checkout backup-before-openie-removal`
2. Restore database: `neo4j-admin load --from=backup-before-openie-removal.dump`
3. Reinstall dependencies: `pip install -r requirements-backup.txt`

## Next Steps

After completing this task:
1. Move to Task 7: Remove hybrid extractor
2. Verify LangExtract handles all relation extraction needs
3. Monitor performance to ensure no regression

## Notes

- This is a **destructive operation** - ensure backups are created
- OpenIE provided triplet-based relation extraction
- LangExtract's relation extraction should be superior in quality
- Some domain-specific OpenIE configurations may need to be recreated as LangExtract examples
- Database cleanup is optional but recommended for clean state
