# Task 2.1: Bidirectional Reference Storage

## Overview

Implement bidirectional reference storage between Neo4j and Qdrant systems to enable seamless cross-system navigation and data consistency. This task establishes the foundation for cross-system entity linking and metadata synchronization.

## Objectives

- Create bidirectional ID mapping between Neo4j nodes and Qdrant points
- Implement reference storage in both database systems
- Establish consistency validation mechanisms
- Enable efficient cross-system lookups
- Provide reference cleanup and maintenance utilities

## Current State Analysis

### Existing Reference Patterns

**Neo4j**:
- DocumentChunks have no Qdrant point references
- Entities lack vector embedding references
- No cross-system navigation capabilities

**Qdrant**:
- Metadata includes `document_id` but no Neo4j node references
- No entity-level linking to graph database
- Limited cross-system query capabilities

## Implementation Plan

### Step 1: Extend Neo4j Schema

Update Neo4j node properties to include Qdrant references:

```cypher
// Add Qdrant reference properties to DocumentChunk nodes
MATCH (chunk:DocumentChunk)
SET chunk.qdrant_point_id = null,
    chunk.qdrant_collection = 'morag_vectors',
    chunk.vector_status = 'pending'

// Add vector reference properties to Entity nodes
MATCH (entity:Entity)
SET entity.qdrant_point_id = null,
    entity.has_embedding = false,
    entity.embedding_model = null,
    entity.embedding_updated = null

// Create indexes for efficient lookups
CREATE INDEX chunk_qdrant_ref_idx FOR (c:DocumentChunk) ON (c.qdrant_point_id);
CREATE INDEX entity_qdrant_ref_idx FOR (e:Entity) ON (e.qdrant_point_id);
```

### Step 2: Create Reference Management Service

Implement `src/morag_graph/services/reference_manager.py`:

```python
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

from qdrant_client import QdrantClient
from neo4j import AsyncSession

@dataclass
class CrossReference:
    """Represents a cross-system reference."""
    neo4j_id: str
    neo4j_type: str  # 'DocumentChunk', 'Entity', 'Document'
    qdrant_point_id: str
    qdrant_collection: str
    created_at: datetime
    updated_at: datetime
    status: str  # 'active', 'pending', 'broken'

class ReferenceManager:
    """Manages bidirectional references between Neo4j and Qdrant."""
    
    def __init__(self, neo4j_session: AsyncSession, qdrant_client: QdrantClient):
        self.neo4j_session = neo4j_session
        self.qdrant_client = qdrant_client
        self.logger = logging.getLogger(__name__)
    
    async def create_chunk_reference(self, 
                                   chunk_id: str, 
                                   qdrant_point_id: str,
                                   collection_name: str = "morag_vectors") -> bool:
        """Create bidirectional reference for DocumentChunk."""
        try:
            # Update Neo4j chunk with Qdrant reference
            neo4j_query = """
            MATCH (chunk:DocumentChunk {id: $chunk_id})
            SET chunk.qdrant_point_id = $qdrant_point_id,
                chunk.qdrant_collection = $collection_name,
                chunk.vector_status = 'active',
                chunk.reference_updated = datetime()
            RETURN chunk.id as updated_id
            """
            
            neo4j_result = await self.neo4j_session.run(
                neo4j_query,
                chunk_id=chunk_id,
                qdrant_point_id=qdrant_point_id,
                collection_name=collection_name
            )
            
            neo4j_updated = await neo4j_result.single()
            if not neo4j_updated:
                self.logger.error(f"Failed to update Neo4j chunk {chunk_id}")
                return False
            
            # Update Qdrant point with Neo4j reference
            await self.qdrant_client.set_payload(
                collection_name=collection_name,
                points=[qdrant_point_id],
                payload={
                    "neo4j_chunk_id": chunk_id,
                    "neo4j_type": "DocumentChunk",
                    "reference_updated": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Created bidirectional reference: {chunk_id} <-> {qdrant_point_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create chunk reference: {e}")
            return False
    
    async def create_entity_reference(self, 
                                    entity_id: str, 
                                    qdrant_point_id: str,
                                    embedding_model: str,
                                    collection_name: str = "morag_entities") -> bool:
        """Create bidirectional reference for Entity."""
        try:
            # Update Neo4j entity with Qdrant reference
            neo4j_query = """
            MATCH (entity:Entity {id: $entity_id})
            SET entity.qdrant_point_id = $qdrant_point_id,
                entity.qdrant_collection = $collection_name,
                entity.has_embedding = true,
                entity.embedding_model = $embedding_model,
                entity.embedding_updated = datetime()
            RETURN entity.id as updated_id
            """
            
            neo4j_result = await self.neo4j_session.run(
                neo4j_query,
                entity_id=entity_id,
                qdrant_point_id=qdrant_point_id,
                collection_name=collection_name,
                embedding_model=embedding_model
            )
            
            neo4j_updated = await neo4j_result.single()
            if not neo4j_updated:
                self.logger.error(f"Failed to update Neo4j entity {entity_id}")
                return False
            
            # Update Qdrant point with Neo4j reference
            await self.qdrant_client.set_payload(
                collection_name=collection_name,
                points=[qdrant_point_id],
                payload={
                    "neo4j_entity_id": entity_id,
                    "neo4j_type": "Entity",
                    "embedding_model": embedding_model,
                    "reference_updated": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Created entity reference: {entity_id} <-> {qdrant_point_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create entity reference: {e}")
            return False
    
    async def get_qdrant_references(self, neo4j_ids: List[str]) -> Dict[str, Optional[str]]:
        """Get Qdrant point IDs for given Neo4j node IDs."""
        query = """
        MATCH (n)
        WHERE n.id IN $neo4j_ids AND n.qdrant_point_id IS NOT NULL
        RETURN n.id as neo4j_id, n.qdrant_point_id as qdrant_id
        """
        
        result = await self.neo4j_session.run(query, neo4j_ids=neo4j_ids)
        references = {}
        
        async for record in result:
            references[record["neo4j_id"]] = record["qdrant_id"]
        
        # Fill in None for missing references
        for neo4j_id in neo4j_ids:
            if neo4j_id not in references:
                references[neo4j_id] = None
        
        return references
    
    async def get_neo4j_references(self, 
                                 qdrant_point_ids: List[str],
                                 collection_name: str = "morag_vectors") -> Dict[str, Optional[str]]:
        """Get Neo4j node IDs for given Qdrant point IDs."""
        try:
            points = await self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=qdrant_point_ids,
                with_payload=True
            )
            
            references = {}
            for point in points:
                point_id = str(point.id)
                neo4j_id = None
                
                if point.payload:
                    neo4j_id = point.payload.get("neo4j_chunk_id") or point.payload.get("neo4j_entity_id")
                
                references[point_id] = neo4j_id
            
            # Fill in None for missing points
            for point_id in qdrant_point_ids:
                if point_id not in references:
                    references[point_id] = None
            
            return references
            
        except Exception as e:
            self.logger.error(f"Failed to get Neo4j references: {e}")
            return {point_id: None for point_id in qdrant_point_ids}
    
    async def validate_references(self, 
                                collection_name: str = "morag_vectors",
                                batch_size: int = 100) -> Dict[str, Any]:
        """Validate bidirectional reference consistency."""
        validation_results = {
            "total_checked": 0,
            "valid_references": 0,
            "broken_references": [],
            "orphaned_neo4j": [],
            "orphaned_qdrant": []
        }
        
        try:
            # Check Neo4j nodes with Qdrant references
            neo4j_query = """
            MATCH (n)
            WHERE n.qdrant_point_id IS NOT NULL
            RETURN n.id as neo4j_id, n.qdrant_point_id as qdrant_id,
                   labels(n)[0] as node_type
            """
            
            neo4j_result = await self.neo4j_session.run(neo4j_query)
            neo4j_refs = []
            
            async for record in neo4j_result:
                neo4j_refs.append({
                    "neo4j_id": record["neo4j_id"],
                    "qdrant_id": record["qdrant_id"],
                    "node_type": record["node_type"]
                })
            
            # Validate references in batches
            for i in range(0, len(neo4j_refs), batch_size):
                batch = neo4j_refs[i:i + batch_size]
                qdrant_ids = [ref["qdrant_id"] for ref in batch]
                
                try:
                    points = await self.qdrant_client.retrieve(
                        collection_name=collection_name,
                        ids=qdrant_ids,
                        with_payload=True
                    )
                    
                    found_ids = {str(point.id) for point in points}
                    
                    for ref in batch:
                        validation_results["total_checked"] += 1
                        
                        if ref["qdrant_id"] in found_ids:
                            validation_results["valid_references"] += 1
                        else:
                            validation_results["broken_references"].append({
                                "neo4j_id": ref["neo4j_id"],
                                "qdrant_id": ref["qdrant_id"],
                                "node_type": ref["node_type"]
                            })
                
                except Exception as e:
                    self.logger.error(f"Batch validation failed: {e}")
                    for ref in batch:
                        validation_results["broken_references"].append(ref)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Reference validation failed: {e}")
            return validation_results
    
    async def cleanup_broken_references(self, 
                                      validation_results: Dict[str, Any]) -> int:
        """Clean up broken references identified during validation."""
        cleaned_count = 0
        
        try:
            # Clean up broken Neo4j references
            for broken_ref in validation_results["broken_references"]:
                neo4j_query = """
                MATCH (n {id: $neo4j_id})
                SET n.qdrant_point_id = null,
                    n.vector_status = 'broken',
                    n.reference_updated = datetime()
                RETURN n.id as cleaned_id
                """
                
                result = await self.neo4j_session.run(
                    neo4j_query,
                    neo4j_id=broken_ref["neo4j_id"]
                )
                
                if await result.single():
                    cleaned_count += 1
                    self.logger.info(f"Cleaned broken reference for {broken_ref['neo4j_id']}")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Reference cleanup failed: {e}")
            return cleaned_count
```

### Step 3: Create Reference Synchronization Service

Implement `src/morag_graph/services/reference_sync.py`:

```python
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta

from .reference_manager import ReferenceManager, CrossReference

class ReferenceSynchronizer:
    """Synchronizes references between Neo4j and Qdrant systems."""
    
    def __init__(self, reference_manager: ReferenceManager):
        self.reference_manager = reference_manager
        self.logger = logging.getLogger(__name__)
    
    async def sync_all_references(self, 
                                collection_name: str = "morag_vectors",
                                batch_size: int = 100) -> Dict[str, int]:
        """Synchronize all references between systems."""
        sync_stats = {
            "processed": 0,
            "created": 0,
            "updated": 0,
            "failed": 0
        }
        
        try:
            # Get all Neo4j chunks without Qdrant references
            neo4j_query = """
            MATCH (chunk:DocumentChunk)
            WHERE chunk.qdrant_point_id IS NULL
            RETURN chunk.id as chunk_id, chunk.document_id as document_id
            ORDER BY chunk.document_id, chunk.chunk_index
            """
            
            result = await self.reference_manager.neo4j_session.run(neo4j_query)
            chunks_to_sync = []
            
            async for record in result:
                chunks_to_sync.append({
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"]
                })
            
            # Process in batches
            for i in range(0, len(chunks_to_sync), batch_size):
                batch = chunks_to_sync[i:i + batch_size]
                
                for chunk_info in batch:
                    sync_stats["processed"] += 1
                    
                    try:
                        # Check if Qdrant point exists with this chunk_id
                        points = await self.reference_manager.qdrant_client.retrieve(
                            collection_name=collection_name,
                            ids=[chunk_info["chunk_id"]],
                            with_payload=True
                        )
                        
                        if points:
                            # Create reference
                            success = await self.reference_manager.create_chunk_reference(
                                chunk_id=chunk_info["chunk_id"],
                                qdrant_point_id=chunk_info["chunk_id"],
                                collection_name=collection_name
                            )
                            
                            if success:
                                sync_stats["created"] += 1
                            else:
                                sync_stats["failed"] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Failed to sync chunk {chunk_info['chunk_id']}: {e}")
                        sync_stats["failed"] += 1
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            return sync_stats
            
        except Exception as e:
            self.logger.error(f"Reference synchronization failed: {e}")
            return sync_stats
    
    async def sync_entity_references(self, 
                                   collection_name: str = "morag_entities",
                                   batch_size: int = 50) -> Dict[str, int]:
        """Synchronize entity references."""
        sync_stats = {
            "processed": 0,
            "created": 0,
            "failed": 0
        }
        
        try:
            # Get entities without Qdrant references
            neo4j_query = """
            MATCH (entity:Entity)
            WHERE entity.qdrant_point_id IS NULL
            RETURN entity.id as entity_id, entity.name as name, entity.type as type
            """
            
            result = await self.reference_manager.neo4j_session.run(neo4j_query)
            entities_to_sync = []
            
            async for record in result:
                entities_to_sync.append({
                    "entity_id": record["entity_id"],
                    "name": record["name"],
                    "type": record["type"]
                })
            
            # Process in batches
            for i in range(0, len(entities_to_sync), batch_size):
                batch = entities_to_sync[i:i + batch_size]
                
                for entity_info in batch:
                    sync_stats["processed"] += 1
                    
                    try:
                        # Check if Qdrant point exists for this entity
                        points = await self.reference_manager.qdrant_client.retrieve(
                            collection_name=collection_name,
                            ids=[entity_info["entity_id"]],
                            with_payload=True
                        )
                        
                        if points:
                            # Create reference
                            success = await self.reference_manager.create_entity_reference(
                                entity_id=entity_info["entity_id"],
                                qdrant_point_id=entity_info["entity_id"],
                                embedding_model="text-embedding-004",
                                collection_name=collection_name
                            )
                            
                            if success:
                                sync_stats["created"] += 1
                            else:
                                sync_stats["failed"] += 1
                    
                    except Exception as e:
                        self.logger.error(f"Failed to sync entity {entity_info['entity_id']}: {e}")
                        sync_stats["failed"] += 1
                
                await asyncio.sleep(0.1)
            
            return sync_stats
            
        except Exception as e:
            self.logger.error(f"Entity reference synchronization failed: {e}")
            return sync_stats
```

## Testing Strategy

### Unit Tests

Create `tests/test_reference_manager.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from morag_graph.services.reference_manager import ReferenceManager, CrossReference

@pytest.fixture
def mock_neo4j_session():
    session = AsyncMock()
    return session

@pytest.fixture
def mock_qdrant_client():
    client = AsyncMock()
    return client

@pytest.fixture
def reference_manager(mock_neo4j_session, mock_qdrant_client):
    return ReferenceManager(mock_neo4j_session, mock_qdrant_client)

@pytest.mark.asyncio
async def test_create_chunk_reference_success(reference_manager, mock_neo4j_session, mock_qdrant_client):
    # Mock successful Neo4j update
    mock_result = AsyncMock()
    mock_result.single.return_value = {"updated_id": "chunk_123"}
    mock_neo4j_session.run.return_value = mock_result
    
    # Mock successful Qdrant update
    mock_qdrant_client.set_payload.return_value = None
    
    # Test
    success = await reference_manager.create_chunk_reference(
        chunk_id="chunk_123",
        qdrant_point_id="point_456"
    )
    
    assert success is True
    mock_neo4j_session.run.assert_called_once()
    mock_qdrant_client.set_payload.assert_called_once()

@pytest.mark.asyncio
async def test_create_chunk_reference_neo4j_failure(reference_manager, mock_neo4j_session, mock_qdrant_client):
    # Mock Neo4j failure
    mock_result = AsyncMock()
    mock_result.single.return_value = None
    mock_neo4j_session.run.return_value = mock_result
    
    # Test
    success = await reference_manager.create_chunk_reference(
        chunk_id="chunk_123",
        qdrant_point_id="point_456"
    )
    
    assert success is False
    mock_neo4j_session.run.assert_called_once()
    mock_qdrant_client.set_payload.assert_not_called()

@pytest.mark.asyncio
async def test_get_qdrant_references(reference_manager, mock_neo4j_session):
    # Mock Neo4j query result
    mock_result = AsyncMock()
    mock_result.__aiter__.return_value = iter([
        {"neo4j_id": "chunk_1", "qdrant_id": "point_1"},
        {"neo4j_id": "chunk_2", "qdrant_id": "point_2"}
    ])
    mock_neo4j_session.run.return_value = mock_result
    
    # Test
    references = await reference_manager.get_qdrant_references(["chunk_1", "chunk_2", "chunk_3"])
    
    expected = {
        "chunk_1": "point_1",
        "chunk_2": "point_2",
        "chunk_3": None
    }
    
    assert references == expected

@pytest.mark.asyncio
async def test_validate_references(reference_manager, mock_neo4j_session, mock_qdrant_client):
    # Mock Neo4j query result
    mock_result = AsyncMock()
    mock_result.__aiter__.return_value = iter([
        {"neo4j_id": "chunk_1", "qdrant_id": "point_1", "node_type": "DocumentChunk"},
        {"neo4j_id": "chunk_2", "qdrant_id": "point_2", "node_type": "DocumentChunk"}
    ])
    mock_neo4j_session.run.return_value = mock_result
    
    # Mock Qdrant retrieve - only point_1 exists
    mock_point = MagicMock()
    mock_point.id = "point_1"
    mock_qdrant_client.retrieve.return_value = [mock_point]
    
    # Test
    results = await reference_manager.validate_references()
    
    assert results["total_checked"] == 2
    assert results["valid_references"] == 1
    assert len(results["broken_references"]) == 1
    assert results["broken_references"][0]["neo4j_id"] == "chunk_2"
```

### Integration Tests

Create `tests/integration/test_reference_integration.py`:

```python
import pytest
import asyncio
from qdrant_client import QdrantClient
from neo4j import AsyncGraphDatabase

from morag_graph.services.reference_manager import ReferenceManager
from morag_graph.services.reference_sync import ReferenceSynchronizer

@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_reference_management():
    # Setup test databases
    qdrant_client = QdrantClient(":memory:")
    neo4j_driver = AsyncGraphDatabase.driver("bolt://localhost:7687")
    
    async with neo4j_driver.session() as session:
        reference_manager = ReferenceManager(session, qdrant_client)
        synchronizer = ReferenceSynchronizer(reference_manager)
        
        # Create test data
        chunk_id = "test_chunk_123"
        qdrant_point_id = "test_point_456"
        
        # Test reference creation
        success = await reference_manager.create_chunk_reference(
            chunk_id=chunk_id,
            qdrant_point_id=qdrant_point_id
        )
        assert success
        
        # Test reference retrieval
        references = await reference_manager.get_qdrant_references([chunk_id])
        assert references[chunk_id] == qdrant_point_id
        
        # Test validation
        validation_results = await reference_manager.validate_references()
        assert validation_results["total_checked"] >= 1
        
        # Cleanup
        await reference_manager.cleanup_broken_references(validation_results)
    
    await neo4j_driver.close()
```

## Performance Considerations

### Optimization Strategies

1. **Batch Operations**: Process references in batches to reduce database load
2. **Async Processing**: Use asyncio for concurrent operations
3. **Indexing**: Create appropriate indexes for reference lookups
4. **Caching**: Cache frequently accessed references
5. **Connection Pooling**: Use connection pools for database access

### Performance Targets

- Reference creation: < 10ms per reference
- Batch reference sync: < 1s per 100 references
- Reference validation: < 5s per 1000 references
- Cross-system lookup: < 50ms per query

## Success Criteria

- [ ] Bidirectional references created for all document chunks
- [ ] Entity references established for embedded entities
- [ ] Reference validation passes with >99% consistency
- [ ] Cross-system lookups perform within target latency
- [ ] Reference cleanup utilities function correctly
- [ ] Comprehensive test coverage (>90%)

## Risk Assessment

**Risk Level**: Medium

**Key Risks**:
- Reference inconsistency during concurrent operations
- Performance degradation with large datasets
- Database connection failures during sync

**Mitigation Strategies**:
- Implement transaction-based reference creation
- Use batch processing with error handling
- Add retry logic for database operations
- Monitor reference consistency continuously

## Rollback Plan

1. **Immediate Rollback**: Remove Qdrant reference fields from Neo4j
2. **Data Cleanup**: Clear reference metadata from Qdrant payloads
3. **Service Restoration**: Revert to independent system operations
4. **Validation**: Verify system functionality without references

## Next Steps

- **Task 2.2**: [Metadata Synchronization](./task-2.2-metadata-synchronization.md)
- **Task 2.3**: [ID Mapping Utilities](./task-2.3-id-mapping-utilities.md)

## Dependencies

- **Task 1.1**: Unified ID Architecture (must be completed)
- **Task 1.2**: Document and Chunk ID Standardization (must be completed)
- **Task 1.3**: Entity ID Integration (must be completed)

## Estimated Time

**Total**: 3-4 days

- Planning and design: 0.5 days
- Implementation: 2 days
- Testing: 1 day
- Documentation: 0.5 days

## Status

- [ ] Planning
- [ ] Implementation
- [ ] Testing
- [ ] Documentation
- [ ] Review
- [ ] Deployment