# Task 1.1: Unified ID Architecture

## Overview

Implement a unified ID architecture that ensures consistent identification across Neo4j and Qdrant systems. This foundational task establishes the ID generation strategies that will be used throughout the integration.

## Objectives

- Design deterministic ID generation for documents and chunks
- Maintain existing entity ID strategy while enhancing cross-system compatibility
- Create ID validation and collision detection mechanisms
- Establish ID format standards for the entire system

## Current State Analysis

### Existing ID Patterns

**Neo4j**:
- Documents: UUID4 (`str(uuid.uuid4())`)
- DocumentChunks: UUID4 with document_id references
- Entities: SHA256-based deterministic IDs (`hash(name:type:source_doc_id)`)
- Relations: Deterministic based on source/target entities

**Qdrant**:
- Vector points: Auto-generated or custom IDs
- No standardized cross-system linking

## Implementation Plan

### Step 1: Create ID Generation Utilities

Create `src/morag_graph/utils/id_generation.py`:

```python
import hashlib
import uuid
from typing import Optional, Union
from datetime import datetime

class UnifiedIDGenerator:
    """Unified ID generation for cross-system compatibility."""
    
    @staticmethod
    def generate_document_id(source_file: str, checksum: Optional[str] = None) -> str:
        """Generate deterministic document ID based on source file and checksum.
        
        Args:
            source_file: Path to the source document
            checksum: Optional file checksum for uniqueness
            
        Returns:
            Deterministic document ID
        """
        if checksum:
            # Use checksum for deterministic ID if available
            content = f"{source_file}:{checksum}"
            return f"doc_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
        else:
            # Fallback to UUID for new documents
            return f"doc_{str(uuid.uuid4()).replace('-', '')[:16]}"
    
    @staticmethod
    def generate_chunk_id(document_id: str, chunk_index: int) -> str:
        """Generate deterministic chunk ID.
        
        Args:
            document_id: Parent document ID
            chunk_index: Zero-based chunk index
            
        Returns:
            Deterministic chunk ID
        """
        return f"{document_id}:chunk:{chunk_index:04d}"
    
    @staticmethod
    def generate_entity_id(name: str, entity_type: str, source_doc_id: str) -> str:
        """Generate deterministic entity ID (maintains existing strategy).
        
        Args:
            name: Entity name
            entity_type: Entity type (PERSON, ORGANIZATION, etc.)
            source_doc_id: Source document ID
            
        Returns:
            Deterministic entity ID
        """
        content = f"{name}:{entity_type}:{source_doc_id}"
        return f"ent_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    @staticmethod
    def generate_relation_id(source_entity_id: str, target_entity_id: str, 
                           relation_type: str) -> str:
        """Generate deterministic relation ID.
        
        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID
            relation_type: Type of relation
            
        Returns:
            Deterministic relation ID
        """
        # Sort entity IDs to ensure consistent direction
        entities = sorted([source_entity_id, target_entity_id])
        content = f"{entities[0]}:{entities[1]}:{relation_type}"
        return f"rel_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    @staticmethod
    def parse_id_type(id_value: str) -> str:
        """Parse ID type from unified ID format.
        
        Args:
            id_value: Unified ID
            
        Returns:
            ID type (doc, chunk, ent, rel)
        """
        if id_value.startswith('doc_'):
            return 'document'
        elif ':chunk:' in id_value:
            return 'chunk'
        elif id_value.startswith('ent_'):
            return 'entity'
        elif id_value.startswith('rel_'):
            return 'relation'
        else:
            return 'unknown'
    
    @staticmethod
    def extract_document_id_from_chunk(chunk_id: str) -> str:
        """Extract document ID from chunk ID.
        
        Args:
            chunk_id: Chunk ID in format 'doc_xxx:chunk:nnnn'
            
        Returns:
            Document ID
        """
        return chunk_id.split(':chunk:')[0]
    
    @staticmethod
    def extract_chunk_index_from_chunk(chunk_id: str) -> int:
        """Extract chunk index from chunk ID.
        
        Args:
            chunk_id: Chunk ID in format 'doc_xxx:chunk:nnnn'
            
        Returns:
            Chunk index
        """
        return int(chunk_id.split(':chunk:')[1])

class IDValidator:
    """Validation utilities for unified IDs."""
    
    @staticmethod
    def validate_document_id(doc_id: str) -> bool:
        """Validate document ID format."""
        return doc_id.startswith('doc_') and len(doc_id) == 20
    
    @staticmethod
    def validate_chunk_id(chunk_id: str) -> bool:
        """Validate chunk ID format."""
        parts = chunk_id.split(':chunk:')
        if len(parts) != 2:
            return False
        doc_id, chunk_idx = parts
        return (IDValidator.validate_document_id(doc_id) and 
                chunk_idx.isdigit() and len(chunk_idx) == 4)
    
    @staticmethod
    def validate_entity_id(entity_id: str) -> bool:
        """Validate entity ID format."""
        return entity_id.startswith('ent_') and len(entity_id) == 20
    
    @staticmethod
    def validate_relation_id(relation_id: str) -> bool:
        """Validate relation ID format."""
        return relation_id.startswith('rel_') and len(relation_id) == 20

class IDCollisionDetector:
    """Detect and handle ID collisions."""
    
    def __init__(self):
        self.seen_ids = set()
        self.collision_count = 0
    
    def check_collision(self, id_value: str) -> bool:
        """Check if ID already exists.
        
        Args:
            id_value: ID to check
            
        Returns:
            True if collision detected
        """
        if id_value in self.seen_ids:
            self.collision_count += 1
            return True
        self.seen_ids.add(id_value)
        return False
    
    def get_collision_stats(self) -> dict:
        """Get collision statistics."""
        return {
            'total_ids': len(self.seen_ids),
            'collisions': self.collision_count,
            'collision_rate': self.collision_count / max(len(self.seen_ids), 1)
        }
```

### Step 2: Update Document Model

Modify `src/morag_graph/models/document.py`:

```python
# Add to existing Document class
from ..utils.id_generation import UnifiedIDGenerator, IDValidator

class Document(BaseModel):
    # ... existing fields ...
    
    def __init__(self, **data):
        # Generate unified ID if not provided
        if 'id' not in data or not data['id']:
            data['id'] = UnifiedIDGenerator.generate_document_id(
                source_file=data.get('source_file', ''),
                checksum=data.get('checksum')
            )
        super().__init__(**data)
    
    @validator('id')
    def validate_id_format(cls, v):
        if not IDValidator.validate_document_id(v):
            raise ValueError(f"Invalid document ID format: {v}")
        return v
    
    def get_unified_id(self) -> str:
        """Get unified document ID."""
        return self.id
    
    def is_unified_format(self) -> bool:
        """Check if using unified ID format."""
        return IDValidator.validate_document_id(self.id)
```

### Step 3: Update DocumentChunk Model

Modify `src/morag_graph/models/document_chunk.py`:

```python
# Add to existing DocumentChunk class
from ..utils.id_generation import UnifiedIDGenerator, IDValidator

class DocumentChunk(BaseModel):
    # ... existing fields ...
    
    def __init__(self, **data):
        # Generate unified chunk ID if not provided
        if 'id' not in data or not data['id']:
            data['id'] = UnifiedIDGenerator.generate_chunk_id(
                document_id=data['document_id'],
                chunk_index=data['chunk_index']
            )
        super().__init__(**data)
    
    @validator('id')
    def validate_id_format(cls, v):
        if not IDValidator.validate_chunk_id(v):
            raise ValueError(f"Invalid chunk ID format: {v}")
        return v
    
    def get_unified_id(self) -> str:
        """Get unified chunk ID."""
        return self.id
    
    def get_document_id_from_chunk(self) -> str:
        """Extract document ID from chunk ID."""
        return UnifiedIDGenerator.extract_document_id_from_chunk(self.id)
    
    def get_chunk_index_from_id(self) -> int:
        """Extract chunk index from chunk ID."""
        return UnifiedIDGenerator.extract_chunk_index_from_chunk(self.id)
```

### Step 4: Create Migration Utilities

Create `src/morag_graph/utils/id_migration.py`:

```python
import asyncio
from typing import List, Dict, Any
from ..storage.neo4j_storage import Neo4jStorage
from ..storage.qdrant_storage import QdrantStorage
from .id_generation import UnifiedIDGenerator, IDValidator

class IDMigrationService:
    """Service for migrating existing IDs to unified format."""
    
    def __init__(self, neo4j_storage: Neo4jStorage, qdrant_storage: QdrantStorage):
        self.neo4j = neo4j_storage
        self.qdrant = qdrant_storage
        self.migration_log = []
    
    async def migrate_document_ids(self, batch_size: int = 100) -> Dict[str, Any]:
        """Migrate document IDs to unified format.
        
        Args:
            batch_size: Number of documents to process per batch
            
        Returns:
            Migration statistics
        """
        # Get all documents with old ID format
        query = """
        MATCH (d:Document)
        WHERE NOT d.id STARTS WITH 'doc_'
        RETURN d.id as old_id, d.source_file as source_file, 
               d.checksum as checksum
        LIMIT $batch_size
        """
        
        migrated_count = 0
        error_count = 0
        
        while True:
            result = await self.neo4j.execute_query(query, batch_size=batch_size)
            documents = [record.data() for record in result]
            
            if not documents:
                break
            
            for doc in documents:
                try:
                    # Generate new unified ID
                    new_id = UnifiedIDGenerator.generate_document_id(
                        source_file=doc['source_file'],
                        checksum=doc.get('checksum')
                    )
                    
                    # Update document ID in Neo4j
                    await self._update_document_id_neo4j(
                        old_id=doc['old_id'],
                        new_id=new_id
                    )
                    
                    # Update references in Qdrant
                    await self._update_document_id_qdrant(
                        old_id=doc['old_id'],
                        new_id=new_id
                    )
                    
                    migrated_count += 1
                    self.migration_log.append({
                        'type': 'document',
                        'old_id': doc['old_id'],
                        'new_id': new_id,
                        'status': 'success'
                    })
                    
                except Exception as e:
                    error_count += 1
                    self.migration_log.append({
                        'type': 'document',
                        'old_id': doc['old_id'],
                        'error': str(e),
                        'status': 'error'
                    })
        
        return {
            'migrated': migrated_count,
            'errors': error_count,
            'total_processed': migrated_count + error_count
        }
    
    async def _update_document_id_neo4j(self, old_id: str, new_id: str):
        """Update document ID in Neo4j."""
        query = """
        MATCH (d:Document {id: $old_id})
        SET d.id = $new_id
        WITH d
        MATCH (d)-[r]-(related)
        RETURN count(r) as relationships_updated
        """
        await self.neo4j.execute_query(query, old_id=old_id, new_id=new_id)
    
    async def _update_document_id_qdrant(self, old_id: str, new_id: str):
        """Update document ID references in Qdrant."""
        # Search for vectors with old document_id
        search_result = await self.qdrant.client.scroll(
            collection_name=self.qdrant.collection_name,
            scroll_filter={
                "must": [
                    {
                        "key": "document_id",
                        "match": {"value": old_id}
                    }
                ]
            },
            limit=1000
        )
        
        # Update metadata for each vector
        for point in search_result[0]:
            point.payload['document_id'] = new_id
            await self.qdrant.client.upsert(
                collection_name=self.qdrant.collection_name,
                points=[point]
            )
    
    def get_migration_report(self) -> Dict[str, Any]:
        """Generate migration report."""
        successful = [log for log in self.migration_log if log['status'] == 'success']
        errors = [log for log in self.migration_log if log['status'] == 'error']
        
        return {
            'total_migrations': len(self.migration_log),
            'successful': len(successful),
            'errors': len(errors),
            'success_rate': len(successful) / max(len(self.migration_log), 1),
            'error_details': errors
        }
```

## Testing

### Unit Tests

Create `tests/test_unified_id_architecture.py`:

```python
import pytest
from src.morag_graph.utils.id_generation import (
    UnifiedIDGenerator, IDValidator, IDCollisionDetector
)

class TestUnifiedIDGenerator:
    
    def test_document_id_generation(self):
        # Test deterministic generation
        doc_id1 = UnifiedIDGenerator.generate_document_id(
            "test.pdf", "checksum123"
        )
        doc_id2 = UnifiedIDGenerator.generate_document_id(
            "test.pdf", "checksum123"
        )
        assert doc_id1 == doc_id2
        assert doc_id1.startswith('doc_')
        assert len(doc_id1) == 20
    
    def test_chunk_id_generation(self):
        doc_id = "doc_1234567890123456"
        chunk_id = UnifiedIDGenerator.generate_chunk_id(doc_id, 5)
        assert chunk_id == f"{doc_id}:chunk:0005"
    
    def test_entity_id_generation(self):
        entity_id = UnifiedIDGenerator.generate_entity_id(
            "John Doe", "PERSON", "doc_1234567890123456"
        )
        assert entity_id.startswith('ent_')
        assert len(entity_id) == 20
    
    def test_id_parsing(self):
        assert UnifiedIDGenerator.parse_id_type("doc_1234567890123456") == "document"
        assert UnifiedIDGenerator.parse_id_type("doc_123:chunk:0001") == "chunk"
        assert UnifiedIDGenerator.parse_id_type("ent_1234567890123456") == "entity"

class TestIDValidator:
    
    def test_document_id_validation(self):
        assert IDValidator.validate_document_id("doc_1234567890123456")
        assert not IDValidator.validate_document_id("invalid_id")
        assert not IDValidator.validate_document_id("doc_123")  # too short
    
    def test_chunk_id_validation(self):
        assert IDValidator.validate_chunk_id("doc_1234567890123456:chunk:0001")
        assert not IDValidator.validate_chunk_id("invalid:chunk:001")
        assert not IDValidator.validate_chunk_id("doc_123:chunk:1")  # wrong format

class TestIDCollisionDetector:
    
    def test_collision_detection(self):
        detector = IDCollisionDetector()
        
        # First occurrence should not be a collision
        assert not detector.check_collision("test_id_1")
        
        # Second occurrence should be a collision
        assert detector.check_collision("test_id_1")
        
        # Different ID should not be a collision
        assert not detector.check_collision("test_id_2")
        
        stats = detector.get_collision_stats()
        assert stats['total_ids'] == 2
        assert stats['collisions'] == 1
```

### Integration Tests

Create `tests/test_id_migration.py`:

```python
import pytest
import asyncio
from src.morag_graph.utils.id_migration import IDMigrationService
from src.morag_graph.storage.neo4j_storage import Neo4jStorage
from src.morag_graph.storage.qdrant_storage import QdrantStorage

@pytest.mark.asyncio
class TestIDMigration:
    
    async def test_document_id_migration(self, neo4j_storage, qdrant_storage):
        migration_service = IDMigrationService(neo4j_storage, qdrant_storage)
        
        # Create test document with old ID format
        old_doc_id = "old-uuid-format"
        await neo4j_storage.execute_query(
            "CREATE (d:Document {id: $id, source_file: 'test.pdf'})",
            id=old_doc_id
        )
        
        # Run migration
        result = await migration_service.migrate_document_ids(batch_size=10)
        
        assert result['migrated'] >= 1
        assert result['errors'] == 0
        
        # Verify new ID format
        query_result = await neo4j_storage.execute_query(
            "MATCH (d:Document) WHERE d.id STARTS WITH 'doc_' RETURN d.id"
        )
        assert len(query_result) >= 1
```

## Performance Considerations

- **ID Generation**: SHA256 hashing is computationally efficient
- **Collision Rate**: 16-character hash provides ~2^64 unique combinations
- **Migration**: Process in batches to avoid memory issues
- **Validation**: Cache validation results for frequently accessed IDs

## Rollback Strategy

1. **Backup**: Create full database backups before migration
2. **Logging**: Maintain detailed migration logs
3. **Reversal**: Implement reverse migration utilities
4. **Validation**: Verify data integrity after migration

## Success Criteria

- [ ] All new documents use unified ID format
- [ ] Existing documents migrated without data loss
- [ ] ID validation prevents invalid formats
- [ ] Collision detection reports zero conflicts
- [ ] Performance benchmarks meet requirements
- [ ] All tests pass

## Next Steps

After completing this task:
1. Proceed to Task 1.2: Document and Chunk ID Standardization
2. Update all storage classes to use unified IDs
3. Implement comprehensive testing
4. Plan production migration strategy

---

**Estimated Time**: 2-3 days  
**Dependencies**: None  
**Risk Level**: Medium (requires careful migration planning)