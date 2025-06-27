# Task 2.3: ID Mapping Utilities

## Overview

This task focuses on creating comprehensive utilities for managing ID mappings and cross-system references between Neo4j and Qdrant. These utilities will provide the foundation for maintaining data consistency and enabling efficient cross-system queries.

## Objectives

- Create robust ID mapping and validation utilities
- Implement cross-system reference management
- Establish consistency checking mechanisms
- Provide migration and synchronization tools
- Enable efficient cross-system navigation

## Implementation Plan

### 1. Core ID Mapping Service

```python
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from datetime import datetime

class IDType(Enum):
    DOCUMENT = "document"
    CHUNK = "chunk"
    ENTITY = "entity"
    RELATION = "relation"

@dataclass
class IDMapping:
    """Represents a mapping between Neo4j and Qdrant IDs"""
    neo4j_id: str
    qdrant_id: str
    id_type: IDType
    created_at: datetime
    last_verified: Optional[datetime] = None
    is_valid: bool = True
    metadata: Dict = None

class IDMappingService:
    """Service for managing ID mappings between Neo4j and Qdrant"""
    
    def __init__(self, neo4j_storage, qdrant_storage, redis_client=None):
        self.neo4j_storage = neo4j_storage
        self.qdrant_storage = qdrant_storage
        self.redis_client = redis_client  # For caching
        self.logger = logging.getLogger(__name__)
        self._mapping_cache: Dict[str, IDMapping] = {}
    
    async def create_mapping(
        self, 
        neo4j_id: str, 
        qdrant_id: str, 
        id_type: IDType,
        metadata: Dict = None
    ) -> IDMapping:
        """Create a new ID mapping"""
        mapping = IDMapping(
            neo4j_id=neo4j_id,
            qdrant_id=qdrant_id,
            id_type=id_type,
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store in cache
        cache_key = f"{id_type.value}:{neo4j_id}"
        self._mapping_cache[cache_key] = mapping
        
        # Store in Redis if available
        if self.redis_client:
            await self._store_mapping_in_redis(mapping)
        
        self.logger.info(f"Created ID mapping: {neo4j_id} -> {qdrant_id} ({id_type.value})")
        return mapping
    
    async def get_qdrant_id(self, neo4j_id: str, id_type: IDType) -> Optional[str]:
        """Get Qdrant ID from Neo4j ID"""
        cache_key = f"{id_type.value}:{neo4j_id}"
        
        # Check cache first
        if cache_key in self._mapping_cache:
            mapping = self._mapping_cache[cache_key]
            if mapping.is_valid:
                return mapping.qdrant_id
        
        # Check Redis cache
        if self.redis_client:
            mapping = await self._get_mapping_from_redis(neo4j_id, id_type)
            if mapping and mapping.is_valid:
                self._mapping_cache[cache_key] = mapping
                return mapping.qdrant_id
        
        # Query from storage systems
        mapping = await self._discover_mapping(neo4j_id, id_type)
        if mapping:
            self._mapping_cache[cache_key] = mapping
            return mapping.qdrant_id
        
        return None
    
    async def get_neo4j_id(self, qdrant_id: str, id_type: IDType) -> Optional[str]:
        """Get Neo4j ID from Qdrant ID"""
        # Search through cache
        for mapping in self._mapping_cache.values():
            if mapping.qdrant_id == qdrant_id and mapping.id_type == id_type:
                if mapping.is_valid:
                    return mapping.neo4j_id
        
        # Query from Qdrant metadata
        if id_type == IDType.CHUNK:
            point = await self.qdrant_storage.client.retrieve(
                collection_name="morag_vectors",
                ids=[qdrant_id],
                with_payload=True
            )
            if point and len(point) > 0:
                payload = point[0].payload
                neo4j_id = payload.get("neo4j_chunk_id") or payload.get("chunk_id")
                if neo4j_id:
                    # Create mapping for future use
                    await self.create_mapping(neo4j_id, qdrant_id, id_type)
                    return neo4j_id
        
        return None
    
    async def validate_mapping(self, mapping: IDMapping) -> bool:
        """Validate that a mapping is still correct"""
        try:
            if mapping.id_type == IDType.CHUNK:
                # Check Neo4j side
                neo4j_exists = await self._check_neo4j_chunk_exists(mapping.neo4j_id)
                
                # Check Qdrant side
                qdrant_exists = await self._check_qdrant_point_exists(mapping.qdrant_id)
                
                is_valid = neo4j_exists and qdrant_exists
                
                # Update mapping validity
                mapping.is_valid = is_valid
                mapping.last_verified = datetime.utcnow()
                
                return is_valid
            
            elif mapping.id_type == IDType.ENTITY:
                # Check if entity exists in Neo4j
                neo4j_exists = await self._check_neo4j_entity_exists(mapping.neo4j_id)
                
                # Check if entity is referenced in Qdrant metadata
                qdrant_refs = await self._check_qdrant_entity_references(mapping.neo4j_id)
                
                is_valid = neo4j_exists and len(qdrant_refs) > 0
                mapping.is_valid = is_valid
                mapping.last_verified = datetime.utcnow()
                
                return is_valid
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating mapping {mapping.neo4j_id}: {e}")
            mapping.is_valid = False
            return False
    
    async def bulk_validate_mappings(self, id_type: Optional[IDType] = None) -> Dict[str, bool]:
        """Validate multiple mappings in bulk"""
        results = {}
        mappings_to_validate = [
            mapping for mapping in self._mapping_cache.values()
            if id_type is None or mapping.id_type == id_type
        ]
        
        # Process in batches to avoid overwhelming the databases
        batch_size = 50
        for i in range(0, len(mappings_to_validate), batch_size):
            batch = mappings_to_validate[i:i + batch_size]
            
            validation_tasks = [
                self.validate_mapping(mapping) for mapping in batch
            ]
            
            batch_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            for mapping, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Validation error for {mapping.neo4j_id}: {result}")
                    results[mapping.neo4j_id] = False
                else:
                    results[mapping.neo4j_id] = result
        
        return results
    
    async def cleanup_invalid_mappings(self) -> int:
        """Remove invalid mappings from cache and storage"""
        invalid_count = 0
        keys_to_remove = []
        
        for key, mapping in self._mapping_cache.items():
            if not mapping.is_valid:
                keys_to_remove.append(key)
                invalid_count += 1
                
                # Remove from Redis if available
                if self.redis_client:
                    await self._remove_mapping_from_redis(mapping)
        
        # Remove from cache
        for key in keys_to_remove:
            del self._mapping_cache[key]
        
        self.logger.info(f"Cleaned up {invalid_count} invalid mappings")
        return invalid_count
    
    async def get_mapping_statistics(self) -> Dict:
        """Get statistics about ID mappings"""
        total_mappings = len(self._mapping_cache)
        valid_mappings = sum(1 for m in self._mapping_cache.values() if m.is_valid)
        
        type_counts = {}
        for mapping in self._mapping_cache.values():
            type_name = mapping.id_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        return {
            "total_mappings": total_mappings,
            "valid_mappings": valid_mappings,
            "invalid_mappings": total_mappings - valid_mappings,
            "mappings_by_type": type_counts,
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    # Helper methods
    async def _discover_mapping(self, neo4j_id: str, id_type: IDType) -> Optional[IDMapping]:
        """Discover mapping by querying both systems"""
        if id_type == IDType.CHUNK:
            # Query Neo4j for chunk with qdrant_point_id
            query = """
            MATCH (chunk:DocumentChunk {id: $chunk_id})
            RETURN chunk.qdrant_point_id as qdrant_id
            """
            
            result = await self.neo4j_storage.execute_query(query, {"chunk_id": neo4j_id})
            if result and len(result) > 0:
                qdrant_id = result[0].get("qdrant_id")
                if qdrant_id:
                    return IDMapping(
                        neo4j_id=neo4j_id,
                        qdrant_id=qdrant_id,
                        id_type=id_type,
                        created_at=datetime.utcnow()
                    )
        
        return None
    
    async def _check_neo4j_chunk_exists(self, chunk_id: str) -> bool:
        """Check if chunk exists in Neo4j"""
        query = "MATCH (chunk:DocumentChunk {id: $chunk_id}) RETURN count(chunk) as count"
        result = await self.neo4j_storage.execute_query(query, {"chunk_id": chunk_id})
        return result and len(result) > 0 and result[0].get("count", 0) > 0
    
    async def _check_qdrant_point_exists(self, point_id: str) -> bool:
        """Check if point exists in Qdrant"""
        try:
            points = await self.qdrant_storage.client.retrieve(
                collection_name="morag_vectors",
                ids=[point_id]
            )
            return len(points) > 0
        except Exception:
            return False
    
    async def _check_neo4j_entity_exists(self, entity_id: str) -> bool:
        """Check if entity exists in Neo4j"""
        query = "MATCH (entity:Entity {id: $entity_id}) RETURN count(entity) as count"
        result = await self.neo4j_storage.execute_query(query, {"entity_id": entity_id})
        return result and len(result) > 0 and result[0].get("count", 0) > 0
    
    async def _check_qdrant_entity_references(self, entity_id: str) -> List[str]:
        """Check for entity references in Qdrant metadata"""
        try:
            # Search for points that mention this entity
            search_result = await self.qdrant_storage.client.scroll(
                collection_name="morag_vectors",
                scroll_filter={
                    "must": [
                        {
                            "key": "mentioned_entities",
                            "match": {"any": [entity_id]}
                        }
                    ]
                },
                limit=10
            )
            
            return [point.id for point in search_result[0]] if search_result else []
        except Exception:
            return []
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder implementation)"""
        # This would need to be implemented with actual hit/miss tracking
        return 0.85  # Placeholder
    
    async def _store_mapping_in_redis(self, mapping: IDMapping):
        """Store mapping in Redis cache"""
        if not self.redis_client:
            return
        
        key = f"id_mapping:{mapping.id_type.value}:{mapping.neo4j_id}"
        value = {
            "qdrant_id": mapping.qdrant_id,
            "created_at": mapping.created_at.isoformat(),
            "is_valid": mapping.is_valid,
            "metadata": mapping.metadata
        }
        
        await self.redis_client.setex(key, 3600, str(value))  # 1 hour TTL
    
    async def _get_mapping_from_redis(self, neo4j_id: str, id_type: IDType) -> Optional[IDMapping]:
        """Get mapping from Redis cache"""
        if not self.redis_client:
            return None
        
        key = f"id_mapping:{id_type.value}:{neo4j_id}"
        try:
            value = await self.redis_client.get(key)
            if value:
                data = eval(value)  # In production, use json.loads
                return IDMapping(
                    neo4j_id=neo4j_id,
                    qdrant_id=data["qdrant_id"],
                    id_type=id_type,
                    created_at=datetime.fromisoformat(data["created_at"]),
                    is_valid=data["is_valid"],
                    metadata=data.get("metadata", {})
                )
        except Exception as e:
            self.logger.error(f"Error retrieving mapping from Redis: {e}")
        
        return None
    
    async def _remove_mapping_from_redis(self, mapping: IDMapping):
        """Remove mapping from Redis cache"""
        if not self.redis_client:
            return
        
        key = f"id_mapping:{mapping.id_type.value}:{mapping.neo4j_id}"
        await self.redis_client.delete(key)
```

### 2. Cross-System Navigation Utilities

```python
class CrossSystemNavigator:
    """Utilities for navigating between Neo4j and Qdrant data"""
    
    def __init__(self, id_mapping_service: IDMappingService):
        self.id_mapping = id_mapping_service
        self.logger = logging.getLogger(__name__)
    
    async def get_related_chunks_for_entity(self, entity_id: str) -> List[Dict]:
        """Get all chunks that mention a specific entity"""
        # Get chunks from Neo4j
        neo4j_query = """
        MATCH (entity:Entity {id: $entity_id})-[:MENTIONED_IN]->(chunk:DocumentChunk)
        RETURN chunk.id as chunk_id, chunk.text as text, chunk.chunk_index as index
        """
        
        neo4j_results = await self.id_mapping.neo4j_storage.execute_query(
            neo4j_query, {"entity_id": entity_id}
        )
        
        # Enrich with Qdrant data
        enriched_chunks = []
        for result in neo4j_results:
            chunk_id = result["chunk_id"]
            qdrant_id = await self.id_mapping.get_qdrant_id(chunk_id, IDType.CHUNK)
            
            chunk_data = {
                "neo4j_id": chunk_id,
                "qdrant_id": qdrant_id,
                "text": result["text"],
                "chunk_index": result["index"]
            }
            
            # Get vector similarity scores if available
            if qdrant_id:
                vector_data = await self._get_qdrant_point_data(qdrant_id)
                chunk_data.update(vector_data)
            
            enriched_chunks.append(chunk_data)
        
        return enriched_chunks
    
    async def get_entities_for_chunk(self, chunk_id: str) -> List[Dict]:
        """Get all entities mentioned in a specific chunk"""
        # First try to get from Qdrant metadata
        qdrant_id = await self.id_mapping.get_qdrant_id(chunk_id, IDType.CHUNK)
        
        entities = []
        if qdrant_id:
            point_data = await self._get_qdrant_point_data(qdrant_id)
            mentioned_entities = point_data.get("mentioned_entities", [])
            
            # Get detailed entity info from Neo4j
            if mentioned_entities:
                neo4j_query = """
                MATCH (entity:Entity)
                WHERE entity.id IN $entity_ids
                RETURN entity.id as id, entity.name as name, entity.type as type,
                       entity.embedding_vector as embedding
                """
                
                neo4j_results = await self.id_mapping.neo4j_storage.execute_query(
                    neo4j_query, {"entity_ids": mentioned_entities}
                )
                
                entities = [
                    {
                        "id": result["id"],
                        "name": result["name"],
                        "type": result["type"],
                        "has_embedding": result["embedding"] is not None
                    }
                    for result in neo4j_results
                ]
        
        return entities
    
    async def find_similar_chunks_with_entities(
        self, 
        query_vector: List[float], 
        entity_filter: List[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Find similar chunks with entity information"""
        # Build Qdrant filter
        qdrant_filter = None
        if entity_filter:
            qdrant_filter = {
                "must": [
                    {
                        "key": "mentioned_entities",
                        "match": {"any": entity_filter}
                    }
                ]
            }
        
        # Search in Qdrant
        search_results = await self.id_mapping.qdrant_storage.client.search(
            collection_name="morag_vectors",
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True
        )
        
        # Enrich with Neo4j data
        enriched_results = []
        for result in search_results:
            chunk_data = {
                "qdrant_id": result.id,
                "score": result.score,
                "text": result.payload.get("text", ""),
                "document_id": result.payload.get("document_id"),
                "mentioned_entities": result.payload.get("mentioned_entities", [])
            }
            
            # Get Neo4j chunk ID
            neo4j_id = await self.id_mapping.get_neo4j_id(result.id, IDType.CHUNK)
            chunk_data["neo4j_id"] = neo4j_id
            
            # Get entity details
            if chunk_data["mentioned_entities"]:
                entities = await self.get_entities_for_chunk(neo4j_id or result.id)
                chunk_data["entities"] = entities
            
            enriched_results.append(chunk_data)
        
        return enriched_results
    
    async def _get_qdrant_point_data(self, point_id: str) -> Dict:
        """Get point data from Qdrant"""
        try:
            points = await self.id_mapping.qdrant_storage.client.retrieve(
                collection_name="morag_vectors",
                ids=[point_id],
                with_payload=True,
                with_vectors=False
            )
            
            if points and len(points) > 0:
                return points[0].payload
            
        except Exception as e:
            self.logger.error(f"Error retrieving Qdrant point {point_id}: {e}")
        
        return {}
```

## Testing Strategy

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestIDMappingService:
    @pytest.fixture
    async def id_mapping_service(self):
        neo4j_storage = AsyncMock()
        qdrant_storage = AsyncMock()
        redis_client = AsyncMock()
        
        return IDMappingService(neo4j_storage, qdrant_storage, redis_client)
    
    async def test_create_mapping(self, id_mapping_service):
        """Test creating a new ID mapping"""
        mapping = await id_mapping_service.create_mapping(
            neo4j_id="chunk_123",
            qdrant_id="point_456",
            id_type=IDType.CHUNK
        )
        
        assert mapping.neo4j_id == "chunk_123"
        assert mapping.qdrant_id == "point_456"
        assert mapping.id_type == IDType.CHUNK
        assert mapping.is_valid is True
    
    async def test_get_qdrant_id_from_cache(self, id_mapping_service):
        """Test retrieving Qdrant ID from cache"""
        # Create mapping first
        await id_mapping_service.create_mapping(
            neo4j_id="chunk_123",
            qdrant_id="point_456",
            id_type=IDType.CHUNK
        )
        
        # Retrieve from cache
        qdrant_id = await id_mapping_service.get_qdrant_id("chunk_123", IDType.CHUNK)
        assert qdrant_id == "point_456"
    
    async def test_validate_mapping_valid_chunk(self, id_mapping_service):
        """Test validating a valid chunk mapping"""
        # Mock storage responses
        id_mapping_service.neo4j_storage.execute_query.return_value = [{"count": 1}]
        id_mapping_service.qdrant_storage.client.retrieve.return_value = [MagicMock()]
        
        mapping = IDMapping(
            neo4j_id="chunk_123",
            qdrant_id="point_456",
            id_type=IDType.CHUNK,
            created_at=datetime.utcnow()
        )
        
        is_valid = await id_mapping_service.validate_mapping(mapping)
        assert is_valid is True
        assert mapping.is_valid is True
        assert mapping.last_verified is not None
    
    async def test_bulk_validate_mappings(self, id_mapping_service):
        """Test bulk validation of mappings"""
        # Create test mappings
        mappings = [
            await id_mapping_service.create_mapping(f"chunk_{i}", f"point_{i}", IDType.CHUNK)
            for i in range(5)
        ]
        
        # Mock validation responses
        id_mapping_service.neo4j_storage.execute_query.return_value = [{"count": 1}]
        id_mapping_service.qdrant_storage.client.retrieve.return_value = [MagicMock()]
        
        results = await id_mapping_service.bulk_validate_mappings(IDType.CHUNK)
        
        assert len(results) == 5
        assert all(results.values())  # All should be valid

class TestCrossSystemNavigator:
    @pytest.fixture
    async def navigator(self):
        id_mapping_service = AsyncMock()
        return CrossSystemNavigator(id_mapping_service)
    
    async def test_get_related_chunks_for_entity(self, navigator):
        """Test getting related chunks for an entity"""
        # Mock Neo4j response
        navigator.id_mapping.neo4j_storage.execute_query.return_value = [
            {"chunk_id": "chunk_1", "text": "Sample text", "index": 0},
            {"chunk_id": "chunk_2", "text": "Another text", "index": 1}
        ]
        
        # Mock ID mapping
        navigator.id_mapping.get_qdrant_id.side_effect = ["point_1", "point_2"]
        
        chunks = await navigator.get_related_chunks_for_entity("entity_123")
        
        assert len(chunks) == 2
        assert chunks[0]["neo4j_id"] == "chunk_1"
        assert chunks[0]["qdrant_id"] == "point_1"
        assert chunks[1]["neo4j_id"] == "chunk_2"
        assert chunks[1]["qdrant_id"] == "point_2"
```

### Integration Tests

```python
class TestIDMappingIntegration:
    """Integration tests with real Neo4j and Qdrant instances"""
    
    @pytest.fixture
    async def integration_setup(self):
        # Set up real connections to test databases
        neo4j_storage = Neo4jStorage(test_config)
        qdrant_storage = QdrantStorage(test_config)
        
        id_mapping_service = IDMappingService(neo4j_storage, qdrant_storage)
        
        # Clean up test data
        await self._cleanup_test_data(neo4j_storage, qdrant_storage)
        
        yield id_mapping_service
        
        # Clean up after tests
        await self._cleanup_test_data(neo4j_storage, qdrant_storage)
    
    async def test_end_to_end_mapping_workflow(self, integration_setup):
        """Test complete mapping workflow with real databases"""
        id_mapping_service = integration_setup
        
        # Create test document and chunk in Neo4j
        document_id = str(uuid.uuid4())
        chunk_id = f"{document_id}:chunk:0"
        
        await id_mapping_service.neo4j_storage.store_document_chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text="Test chunk text",
            chunk_index=0
        )
        
        # Create test vector in Qdrant
        test_vector = [0.1] * 384
        point_id = chunk_id
        
        await id_mapping_service.qdrant_storage.store_vector(
            point_id=point_id,
            vector=test_vector,
            metadata={"document_id": document_id, "text": "Test chunk text"}
        )
        
        # Create mapping
        mapping = await id_mapping_service.create_mapping(
            neo4j_id=chunk_id,
            qdrant_id=point_id,
            id_type=IDType.CHUNK
        )
        
        # Test retrieval
        retrieved_qdrant_id = await id_mapping_service.get_qdrant_id(chunk_id, IDType.CHUNK)
        assert retrieved_qdrant_id == point_id
        
        retrieved_neo4j_id = await id_mapping_service.get_neo4j_id(point_id, IDType.CHUNK)
        assert retrieved_neo4j_id == chunk_id
        
        # Test validation
        is_valid = await id_mapping_service.validate_mapping(mapping)
        assert is_valid is True
    
    async def _cleanup_test_data(self, neo4j_storage, qdrant_storage):
        """Clean up test data from databases"""
        # Clean Neo4j test data
        await neo4j_storage.execute_query(
            "MATCH (n) WHERE n.id STARTS WITH 'test_' DELETE n"
        )
        
        # Clean Qdrant test data
        try:
            await qdrant_storage.client.delete(
                collection_name="morag_vectors",
                points_selector={"filter": {"must": [{"key": "test", "match": {"value": True}}]}}
            )
        except Exception:
            pass  # Collection might not exist
```

## Performance Considerations

### Optimization Strategies

1. **Caching**:
   - In-memory cache for frequently accessed mappings
   - Redis cache for distributed environments
   - TTL-based cache invalidation

2. **Batch Operations**:
   - Bulk validation of mappings
   - Batch creation and updates
   - Parallel processing where possible

3. **Lazy Loading**:
   - Load mappings on-demand
   - Background validation processes
   - Incremental cache warming

### Performance Targets

- **Mapping Retrieval**: < 10ms for cached mappings
- **Validation**: < 100ms per mapping
- **Bulk Operations**: Process 1000+ mappings per minute
- **Cache Hit Rate**: > 80% for frequently accessed mappings

## Success Criteria

- [ ] ID mapping service handles all entity types (documents, chunks, entities, relations)
- [ ] Cross-system navigation utilities provide seamless data access
- [ ] Mapping validation ensures data consistency
- [ ] Performance targets are met under load
- [ ] Comprehensive test coverage (>90%)
- [ ] Integration with existing MoRAG components
- [ ] Monitoring and alerting for mapping health

## Risk Assessment

**Medium Risk**: Cache consistency and performance under high load

**Mitigation Strategies**:
- Implement cache invalidation strategies
- Use distributed caching for scalability
- Monitor cache performance and hit rates
- Implement circuit breakers for external dependencies

## Rollback Plan

1. **Disable ID mapping service** and fall back to direct database queries
2. **Clear caches** to prevent stale data issues
3. **Restore previous version** of navigation utilities
4. **Monitor system stability** after rollback

## Next Steps

- **Task 3.1**: Neo4j Vector Storage
- **Task 3.2**: Selective Vector Strategy
- **Integration**: Incorporate utilities into existing MoRAG workflows

## Dependencies

- **Task 2.1**: Cross-System Entity Linking (completed)
- **Task 2.2**: Vector Embedding Integration (completed)
- Neo4j and Qdrant storage implementations
- Redis for distributed caching (optional)

## Estimated Time

**4-5 days**

## Status

- [ ] Not Started
- [ ] In Progress
- [ ] Testing
- [ ] Completed
- [ ] Verified