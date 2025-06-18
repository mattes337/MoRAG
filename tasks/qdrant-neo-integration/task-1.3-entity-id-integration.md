# Task 1.3: Entity ID Integration

## Overview

Integrate entity identification across Neo4j and Qdrant systems while maintaining the existing deterministic entity ID strategy. This task ensures entities can be referenced and linked between vector and graph databases.

## Objectives

- Maintain existing entity ID generation strategy
- Store entity references in Qdrant vector metadata
- Create entity-chunk linking mechanisms
- Implement entity-based vector filtering
- Establish entity consistency validation

## Dependencies

- Task 1.1: Unified ID Architecture
- Task 1.2: Document and Chunk ID Standardization
- Existing entity extraction pipeline in `morag-graph`

## Current Entity ID Strategy

The existing entity ID generation uses SHA256 hashing:
```python
entity_id = hash(f"{name}:{type}:{source_doc_id}")
```

This strategy will be **maintained** to ensure:
- Deterministic entity identification
- Deduplication across documents
- Backward compatibility

## Implementation Plan

### Step 1: Enhance Entity Model

Modify `src/morag_graph/models/entity.py`:

```python
# Add imports
from ..utils.id_generation import UnifiedIDGenerator, IDValidator
from typing import List, Optional, Set

class Entity(BaseModel):
    # ... existing fields ...
    
    # New fields for cross-system integration
    mentioned_in_chunks: Optional[List[str]] = Field(default_factory=list)
    qdrant_vector_ids: Optional[List[str]] = Field(default_factory=list)
    embedding_vector: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    last_updated: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    def __init__(self, **data):
        # Ensure ID is generated using existing strategy
        if 'id' not in data or not data['id']:
            data['id'] = self._generate_entity_id(
                name=data['name'],
                entity_type=data['type'],
                source_doc_id=data['source_doc_id']
            )
        super().__init__(**data)
    
    @staticmethod
    def _generate_entity_id(name: str, entity_type: str, source_doc_id: str) -> str:
        """Generate entity ID using existing strategy."""
        return UnifiedIDGenerator.generate_entity_id(name, entity_type, source_doc_id)
    
    def add_chunk_reference(self, chunk_id: str):
        """Add reference to a chunk where this entity is mentioned."""
        if not IDValidator.validate_chunk_id(chunk_id):
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
        
        if chunk_id not in self.mentioned_in_chunks:
            self.mentioned_in_chunks.append(chunk_id)
            self.last_updated = datetime.utcnow()
    
    def add_qdrant_vector_reference(self, vector_id: str):
        """Add reference to a Qdrant vector containing this entity."""
        if vector_id not in self.qdrant_vector_ids:
            self.qdrant_vector_ids.append(vector_id)
            self.last_updated = datetime.utcnow()
    
    def get_document_ids(self) -> Set[str]:
        """Get all document IDs where this entity is mentioned."""
        doc_ids = set()
        for chunk_id in self.mentioned_in_chunks:
            if IDValidator.validate_chunk_id(chunk_id):
                doc_id = UnifiedIDGenerator.extract_document_id_from_chunk(chunk_id)
                doc_ids.add(doc_id)
        return doc_ids
    
    def set_embedding(self, vector: List[float], model: str):
        """Set entity embedding vector."""
        self.embedding_vector = vector
        self.embedding_model = model
        self.last_updated = datetime.utcnow()
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties with enhanced fields."""
        props = super().to_neo4j_properties()
        props.update({
            'mentioned_in_chunks': self.mentioned_in_chunks,
            'qdrant_vector_ids': self.qdrant_vector_ids,
            'embedding_vector': self.embedding_vector,
            'embedding_model': self.embedding_model,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'unified_integration': True
        })
        return props
    
    @classmethod
    def from_neo4j_node(cls, node) -> 'Entity':
        """Create Entity from Neo4j node with enhanced fields."""
        entity = super().from_neo4j_node(node)
        
        # Handle new fields
        entity.mentioned_in_chunks = node.get('mentioned_in_chunks', [])
        entity.qdrant_vector_ids = node.get('qdrant_vector_ids', [])
        entity.embedding_vector = node.get('embedding_vector')
        entity.embedding_model = node.get('embedding_model')
        
        if node.get('last_updated'):
            entity.last_updated = datetime.fromisoformat(node['last_updated'])
        
        return entity
```

### Step 2: Update Neo4j Storage for Entities

Add methods to `src/morag_graph/storage/neo4j_storage.py`:

```python
# Add to Neo4jStorage class

async def store_entity_with_chunk_references(self, entity: Entity, chunk_ids: List[str]) -> str:
    """Store entity with chunk references.
    
    Args:
        entity: Entity instance
        chunk_ids: List of chunk IDs where entity is mentioned
        
    Returns:
        Entity ID
    """
    # Validate chunk IDs
    for chunk_id in chunk_ids:
        if not IDValidator.validate_chunk_id(chunk_id):
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
    
    # Add chunk references to entity
    for chunk_id in chunk_ids:
        entity.add_chunk_reference(chunk_id)
    
    # Store entity
    query = """
    MERGE (e:Entity {id: $id})
    SET e.name = $name,
        e.type = $type,
        e.source_doc_id = $source_doc_id,
        e.mentioned_in_chunks = $mentioned_in_chunks,
        e.qdrant_vector_ids = $qdrant_vector_ids,
        e.embedding_vector = $embedding_vector,
        e.embedding_model = $embedding_model,
        e.last_updated = $last_updated,
        e.unified_integration = true
    RETURN e.id as entity_id
    """
    
    result = await self.execute_query(
        query,
        id=entity.id,
        name=entity.name,
        type=entity.type,
        source_doc_id=entity.source_doc_id,
        mentioned_in_chunks=entity.mentioned_in_chunks,
        qdrant_vector_ids=entity.qdrant_vector_ids,
        embedding_vector=entity.embedding_vector,
        embedding_model=entity.embedding_model,
        last_updated=entity.last_updated.isoformat()
    )
    
    # Create relationships to chunks
    for chunk_id in chunk_ids:
        await self._create_entity_chunk_relationship(entity.id, chunk_id)
    
    return result[0]['entity_id']

async def _create_entity_chunk_relationship(self, entity_id: str, chunk_id: str):
    """Create MENTIONED_IN relationship between entity and chunk."""
    query = """
    MATCH (e:Entity {id: $entity_id})
    MATCH (c:DocumentChunk {id: $chunk_id})
    MERGE (e)-[:MENTIONED_IN]->(c)
    """
    
    await self.execute_query(
        query,
        entity_id=entity_id,
        chunk_id=chunk_id
    )

async def get_entities_by_chunk_id(self, chunk_id: str) -> List[Entity]:
    """Get all entities mentioned in a specific chunk.
    
    Args:
        chunk_id: Unified chunk ID
        
    Returns:
        List of Entity instances
    """
    if not IDValidator.validate_chunk_id(chunk_id):
        raise ValueError(f"Invalid chunk ID format: {chunk_id}")
    
    query = """
    MATCH (e:Entity)-[:MENTIONED_IN]->(c:DocumentChunk {id: $chunk_id})
    RETURN e.id as id,
           e.name as name,
           e.type as type,
           e.source_doc_id as source_doc_id,
           e.mentioned_in_chunks as mentioned_in_chunks,
           e.qdrant_vector_ids as qdrant_vector_ids,
           e.embedding_vector as embedding_vector,
           e.embedding_model as embedding_model,
           e.last_updated as last_updated
    """
    
    result = await self.execute_query(query, chunk_id=chunk_id)
    
    entities = []
    for entity_data in result:
        entity = Entity(
            id=entity_data['id'],
            name=entity_data['name'],
            type=entity_data['type'],
            source_doc_id=entity_data['source_doc_id'],
            mentioned_in_chunks=entity_data.get('mentioned_in_chunks', []),
            qdrant_vector_ids=entity_data.get('qdrant_vector_ids', []),
            embedding_vector=entity_data.get('embedding_vector'),
            embedding_model=entity_data.get('embedding_model')
        )
        
        if entity_data.get('last_updated'):
            entity.last_updated = datetime.fromisoformat(entity_data['last_updated'])
        
        entities.append(entity)
    
    return entities

async def get_entities_by_document_id(self, document_id: str) -> List[Entity]:
    """Get all entities mentioned in a document.
    
    Args:
        document_id: Unified document ID
        
    Returns:
        List of Entity instances
    """
    if not IDValidator.validate_document_id(document_id):
        raise ValueError(f"Invalid document ID format: {document_id}")
    
    query = """
    MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:DocumentChunk)
    MATCH (e:Entity)-[:MENTIONED_IN]->(c)
    RETURN DISTINCT e.id as id,
           e.name as name,
           e.type as type,
           e.source_doc_id as source_doc_id,
           e.mentioned_in_chunks as mentioned_in_chunks,
           e.qdrant_vector_ids as qdrant_vector_ids,
           e.embedding_vector as embedding_vector,
           e.embedding_model as embedding_model,
           e.last_updated as last_updated
    """
    
    result = await self.execute_query(query, document_id=document_id)
    
    entities = []
    for entity_data in result:
        entity = Entity(
            id=entity_data['id'],
            name=entity_data['name'],
            type=entity_data['type'],
            source_doc_id=entity_data['source_doc_id'],
            mentioned_in_chunks=entity_data.get('mentioned_in_chunks', []),
            qdrant_vector_ids=entity_data.get('qdrant_vector_ids', []),
            embedding_vector=entity_data.get('embedding_vector'),
            embedding_model=entity_data.get('embedding_model')
        )
        
        if entity_data.get('last_updated'):
            entity.last_updated = datetime.fromisoformat(entity_data['last_updated'])
        
        entities.append(entity)
    
    return entities

async def update_entity_qdrant_references(self, entity_id: str, qdrant_vector_ids: List[str]):
    """Update entity's Qdrant vector references.
    
    Args:
        entity_id: Entity ID
        qdrant_vector_ids: List of Qdrant vector IDs
    """
    query = """
    MATCH (e:Entity {id: $entity_id})
    SET e.qdrant_vector_ids = $qdrant_vector_ids,
        e.last_updated = $last_updated
    RETURN e.id
    """
    
    await self.execute_query(
        query,
        entity_id=entity_id,
        qdrant_vector_ids=qdrant_vector_ids,
        last_updated=datetime.utcnow().isoformat()
    )
```

### Step 3: Enhance Qdrant Storage with Entity References

Add methods to `src/morag_graph/storage/qdrant_storage.py`:

```python
# Add to QdrantStorage class

async def store_chunk_vector_with_entities(self,
                                         chunk_id: str,
                                         vector: List[float],
                                         entities: List[Entity],
                                         metadata: Dict[str, Any]) -> str:
    """Store chunk vector with entity references.
    
    Args:
        chunk_id: Unified chunk ID
        vector: Dense vector embedding
        entities: List of entities mentioned in chunk
        metadata: Additional metadata
        
    Returns:
        Point ID (same as chunk_id)
    """
    # Validate chunk ID
    if not IDValidator.validate_chunk_id(chunk_id):
        raise ValueError(f"Invalid chunk ID format: {chunk_id}")
    
    # Extract document information
    document_id = UnifiedIDGenerator.extract_document_id_from_chunk(chunk_id)
    chunk_index = UnifiedIDGenerator.extract_chunk_index_from_chunk(chunk_id)
    
    # Prepare entity metadata
    entity_ids = [entity.id for entity in entities]
    entity_names = [entity.name for entity in entities]
    entity_types = [entity.type for entity in entities]
    entity_type_counts = self._count_entity_types(entities)
    
    # Enhanced metadata with entity information
    enhanced_metadata = {
        'document_id': document_id,
        'chunk_id': chunk_id,
        'chunk_index': chunk_index,
        'neo4j_chunk_id': chunk_id,
        'mentioned_entities': entity_ids,
        'entity_names': entity_names,
        'entity_types': entity_types,
        'entity_type_counts': entity_type_counts,
        'entity_count': len(entities),
        'unified_id_format': True,
        'has_entities': len(entities) > 0,
        **metadata
    }
    
    # Create point
    point = PointStruct(
        id=chunk_id,
        vector=vector,
        payload=enhanced_metadata
    )
    
    # Store in Qdrant
    await self.client.upsert(
        collection_name=self.collection_name,
        points=[point]
    )
    
    return chunk_id

def _count_entity_types(self, entities: List[Entity]) -> Dict[str, int]:
    """Count entities by type."""
    type_counts = {}
    for entity in entities:
        type_counts[entity.type] = type_counts.get(entity.type, 0) + 1
    return type_counts

async def search_by_entity_filter(self,
                                query_vector: List[float],
                                entity_ids: Optional[List[str]] = None,
                                entity_types: Optional[List[str]] = None,
                                entity_names: Optional[List[str]] = None,
                                top_k: int = 10) -> List[Dict[str, Any]]:
    """Search vectors with entity-based filtering.
    
    Args:
        query_vector: Query embedding vector
        entity_ids: Filter by specific entity IDs
        entity_types: Filter by entity types
        entity_names: Filter by entity names
        top_k: Number of results to return
        
    Returns:
        List of search results with entity information
    """
    from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
    
    # Build filter conditions
    filter_conditions = []
    
    if entity_ids:
        filter_conditions.append(
            FieldCondition(
                key="mentioned_entities",
                match=MatchAny(any=entity_ids)
            )
        )
    
    if entity_types:
        filter_conditions.append(
            FieldCondition(
                key="entity_types",
                match=MatchAny(any=entity_types)
            )
        )
    
    if entity_names:
        filter_conditions.append(
            FieldCondition(
                key="entity_names",
                match=MatchAny(any=entity_names)
            )
        )
    
    # Create filter
    search_filter = None
    if filter_conditions:
        search_filter = Filter(must=filter_conditions)
    
    # Perform search
    results = await self.client.search(
        collection_name=self.collection_name,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )
    
    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append({
            'id': result.id,
            'score': result.score,
            'chunk_id': result.payload.get('chunk_id'),
            'document_id': result.payload.get('document_id'),
            'mentioned_entities': result.payload.get('mentioned_entities', []),
            'entity_names': result.payload.get('entity_names', []),
            'entity_types': result.payload.get('entity_types', []),
            'entity_count': result.payload.get('entity_count', 0),
            'text': result.payload.get('text', ''),
            'metadata': result.payload
        })
    
    return formatted_results

async def get_entity_co_occurrence_stats(self, entity_id: str) -> Dict[str, Any]:
    """Get co-occurrence statistics for an entity.
    
    Args:
        entity_id: Entity ID to analyze
        
    Returns:
        Co-occurrence statistics
    """
    # Search for chunks containing the entity
    search_filter = Filter(
        must=[
            FieldCondition(
                key="mentioned_entities",
                match=MatchAny(any=[entity_id])
            )
        ]
    )
    
    results = await self.client.scroll(
        collection_name=self.collection_name,
        scroll_filter=search_filter,
        limit=1000,
        with_payload=True
    )
    
    # Analyze co-occurrences
    co_occurring_entities = {}
    co_occurring_types = {}
    document_count = set()
    
    for point in results[0]:
        payload = point.payload
        document_count.add(payload.get('document_id'))
        
        # Count co-occurring entities
        for other_entity in payload.get('mentioned_entities', []):
            if other_entity != entity_id:
                co_occurring_entities[other_entity] = co_occurring_entities.get(other_entity, 0) + 1
        
        # Count co-occurring entity types
        for entity_type in payload.get('entity_types', []):
            co_occurring_types[entity_type] = co_occurring_types.get(entity_type, 0) + 1
    
    return {
        'entity_id': entity_id,
        'total_mentions': len(results[0]),
        'document_count': len(document_count),
        'co_occurring_entities': dict(sorted(co_occurring_entities.items(), key=lambda x: x[1], reverse=True)[:10]),
        'co_occurring_types': dict(sorted(co_occurring_types.items(), key=lambda x: x[1], reverse=True)),
        'avg_mentions_per_document': len(results[0]) / max(len(document_count), 1)
    }
```

### Step 4: Create Entity Integration Service

Create `src/morag_graph/services/entity_integration_service.py`:

```python
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from ..models.entity import Entity
from ..models.document_chunk import DocumentChunk
from ..storage.neo4j_storage import Neo4jStorage
from ..storage.qdrant_storage import QdrantStorage
from ..utils.id_generation import UnifiedIDGenerator

class EntityIntegrationService:
    """Service for integrating entities across Neo4j and Qdrant."""
    
    def __init__(self, neo4j_storage: Neo4jStorage, qdrant_storage: QdrantStorage):
        self.neo4j = neo4j_storage
        self.qdrant = qdrant_storage
    
    async def process_chunk_with_entities(self,
                                        chunk: DocumentChunk,
                                        entities: List[Entity],
                                        vector: List[float],
                                        chunk_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process chunk with entity integration.
        
        Args:
            chunk: DocumentChunk instance
            entities: List of entities mentioned in chunk
            vector: Chunk embedding vector
            chunk_metadata: Additional chunk metadata
            
        Returns:
            Processing result
        """
        # Store chunk in Neo4j
        chunk_id = await self.neo4j.store_chunk_with_unified_id(chunk)
        
        # Store entities with chunk references
        entity_ids = []
        for entity in entities:
            entity_id = await self.neo4j.store_entity_with_chunk_references(
                entity=entity,
                chunk_ids=[chunk_id]
            )
            entity_ids.append(entity_id)
        
        # Store vector in Qdrant with entity references
        vector_id = await self.qdrant.store_chunk_vector_with_entities(
            chunk_id=chunk_id,
            vector=vector,
            entities=entities,
            metadata=chunk_metadata or {}
        )
        
        # Update entity Qdrant references in Neo4j
        for entity in entities:
            await self.neo4j.update_entity_qdrant_references(
                entity_id=entity.id,
                qdrant_vector_ids=[vector_id]
            )
        
        return {
            'chunk_id': chunk_id,
            'entity_ids': entity_ids,
            'vector_id': vector_id,
            'entities_processed': len(entities),
            'integration_complete': True
        }
    
    async def search_with_entity_context(self,
                                       query_vector: List[float],
                                       entity_context: Optional[List[str]] = None,
                                       entity_types: Optional[List[str]] = None,
                                       expand_graph: bool = True,
                                       top_k: int = 10) -> Dict[str, Any]:
        """Search with entity context and optional graph expansion.
        
        Args:
            query_vector: Query embedding vector
            entity_context: Entity IDs for context filtering
            entity_types: Entity types for filtering
            expand_graph: Whether to expand search using graph relationships
            top_k: Number of results to return
            
        Returns:
            Search results with entity context
        """
        # Initial vector search with entity filtering
        vector_results = await self.qdrant.search_by_entity_filter(
            query_vector=query_vector,
            entity_ids=entity_context,
            entity_types=entity_types,
            top_k=top_k
        )
        
        # Optional graph expansion
        expanded_entities = set(entity_context or [])
        if expand_graph and entity_context:
            # Find related entities through graph relationships
            related_entities = await self._find_related_entities(entity_context)
            expanded_entities.update(related_entities)
            
            # Additional search with expanded entity context
            if related_entities:
                expanded_results = await self.qdrant.search_by_entity_filter(
                    query_vector=query_vector,
                    entity_ids=list(expanded_entities),
                    top_k=top_k
                )
                
                # Merge and deduplicate results
                vector_results = self._merge_search_results(vector_results, expanded_results)
        
        # Enrich results with entity information from Neo4j
        enriched_results = await self._enrich_results_with_entity_info(vector_results)
        
        return {
            'results': enriched_results,
            'total_results': len(enriched_results),
            'entity_context_used': list(expanded_entities),
            'graph_expansion_applied': expand_graph and bool(entity_context)
        }
    
    async def _find_related_entities(self, entity_ids: List[str], max_hops: int = 2) -> List[str]:
        """Find entities related through graph relationships."""
        query = """
        MATCH (e:Entity)
        WHERE e.id IN $entity_ids
        CALL {
            WITH e
            MATCH (e)-[r*1..2]-(related:Entity)
            WHERE type(r) IN ['RELATED_TO', 'CO_OCCURS_WITH', 'SIMILAR_TO']
            RETURN DISTINCT related.id as related_id
        }
        RETURN COLLECT(DISTINCT related_id) as related_entities
        """
        
        result = await self.neo4j.execute_query(query, entity_ids=entity_ids)
        
        if result:
            return result[0]['related_entities']
        return []
    
    def _merge_search_results(self, results1: List[Dict], results2: List[Dict]) -> List[Dict]:
        """Merge and deduplicate search results."""
        seen_ids = set()
        merged = []
        
        # Add results from first search
        for result in results1:
            if result['id'] not in seen_ids:
                merged.append(result)
                seen_ids.add(result['id'])
        
        # Add unique results from second search
        for result in results2:
            if result['id'] not in seen_ids:
                merged.append(result)
                seen_ids.add(result['id'])
        
        # Sort by score
        merged.sort(key=lambda x: x['score'], reverse=True)
        
        return merged
    
    async def _enrich_results_with_entity_info(self, results: List[Dict]) -> List[Dict]:
        """Enrich search results with detailed entity information from Neo4j."""
        enriched = []
        
        for result in results:
            chunk_id = result['chunk_id']
            
            # Get detailed entity information from Neo4j
            entities = await self.neo4j.get_entities_by_chunk_id(chunk_id)
            
            # Add entity details to result
            result['entity_details'] = [
                {
                    'id': entity.id,
                    'name': entity.name,
                    'type': entity.type,
                    'has_embedding': entity.embedding_vector is not None
                }
                for entity in entities
            ]
            
            enriched.append(result)
        
        return enriched
    
    async def validate_entity_integration(self) -> Dict[str, Any]:
        """Validate entity integration across systems."""
        # Get entity statistics from Neo4j
        neo4j_stats = await self.neo4j.execute_query(
            """
            MATCH (e:Entity)
            RETURN count(e) as total_entities,
                   count(e.mentioned_in_chunks) as entities_with_chunks,
                   count(e.qdrant_vector_ids) as entities_with_qdrant_refs,
                   count(e.embedding_vector) as entities_with_embeddings
            """
        )
        
        # Get entity statistics from Qdrant
        qdrant_stats = await self.qdrant.client.scroll(
            collection_name=self.qdrant.collection_name,
            limit=1,
            with_payload=True
        )
        
        total_vectors = qdrant_stats[1]  # Total count
        
        # Sample check for entity consistency
        sample_results = await self.qdrant.client.scroll(
            collection_name=self.qdrant.collection_name,
            limit=100,
            with_payload=True
        )
        
        vectors_with_entities = sum(
            1 for point in sample_results[0] 
            if point.payload.get('mentioned_entities')
        )
        
        neo4j_data = neo4j_stats[0] if neo4j_stats else {}
        
        return {
            'neo4j_entities': neo4j_data.get('total_entities', 0),
            'entities_with_chunk_refs': neo4j_data.get('entities_with_chunks', 0),
            'entities_with_qdrant_refs': neo4j_data.get('entities_with_qdrant_refs', 0),
            'entities_with_embeddings': neo4j_data.get('entities_with_embeddings', 0),
            'total_qdrant_vectors': total_vectors,
            'vectors_with_entities_sample': vectors_with_entities,
            'entity_integration_rate': (
                neo4j_data.get('entities_with_qdrant_refs', 0) / 
                max(neo4j_data.get('total_entities', 1), 1)
            ),
            'vector_entity_rate_sample': vectors_with_entities / max(len(sample_results[0]), 1)
        }
```

## Testing

### Unit Tests

Create `tests/test_entity_id_integration.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.morag_graph.models.entity import Entity
from src.morag_graph.models.document_chunk import DocumentChunk
from src.morag_graph.services.entity_integration_service import EntityIntegrationService

@pytest.mark.asyncio
class TestEntityIntegration:
    
    @pytest.fixture
    def sample_entities(self):
        return [
            Entity(
                name="John Doe",
                type="PERSON",
                source_doc_id="doc_1234567890123456"
            ),
            Entity(
                name="Acme Corp",
                type="ORGANIZATION",
                source_doc_id="doc_1234567890123456"
            )
        ]
    
    @pytest.fixture
    def sample_chunk(self):
        return DocumentChunk(
            id="doc_1234567890123456:chunk:0000",
            document_id="doc_1234567890123456",
            chunk_index=0,
            text="John Doe works at Acme Corp."
        )
    
    async def test_entity_id_generation(self, sample_entities):
        entity = sample_entities[0]
        
        # Test deterministic ID generation
        expected_id = Entity._generate_entity_id(
            "John Doe", "PERSON", "doc_1234567890123456"
        )
        assert entity.id == expected_id
        
        # Test ID format
        assert entity.id.startswith('ent_')
        assert len(entity.id) == 20
    
    async def test_chunk_reference_management(self, sample_entities):
        entity = sample_entities[0]
        chunk_id = "doc_1234567890123456:chunk:0000"
        
        # Add chunk reference
        entity.add_chunk_reference(chunk_id)
        assert chunk_id in entity.mentioned_in_chunks
        
        # Test document ID extraction
        doc_ids = entity.get_document_ids()
        assert "doc_1234567890123456" in doc_ids
    
    async def test_entity_integration_service(self, sample_entities, sample_chunk):
        # Mock storage
        neo4j_storage = AsyncMock()
        qdrant_storage = AsyncMock()
        
        service = EntityIntegrationService(neo4j_storage, qdrant_storage)
        
        # Mock storage responses
        neo4j_storage.store_chunk_with_unified_id.return_value = sample_chunk.id
        neo4j_storage.store_entity_with_chunk_references.return_value = sample_entities[0].id
        qdrant_storage.store_chunk_vector_with_entities.return_value = sample_chunk.id
        
        # Process chunk with entities
        result = await service.process_chunk_with_entities(
            chunk=sample_chunk,
            entities=sample_entities,
            vector=[0.1, 0.2, 0.3, 0.4],
            chunk_metadata={'test': True}
        )
        
        # Verify results
        assert result['chunk_id'] == sample_chunk.id
        assert result['entities_processed'] == 2
        assert result['integration_complete'] is True
        
        # Verify storage calls
        neo4j_storage.store_chunk_with_unified_id.assert_called_once()
        assert neo4j_storage.store_entity_with_chunk_references.call_count == 2
        qdrant_storage.store_chunk_vector_with_entities.assert_called_once()

@pytest.mark.asyncio
class TestEntitySearch:
    
    async def test_entity_filtered_search(self):
        qdrant_storage = AsyncMock()
        
        # Mock search results
        mock_results = [
            {
                'id': 'doc_123:chunk:0000',
                'score': 0.95,
                'chunk_id': 'doc_123:chunk:0000',
                'mentioned_entities': ['ent_1', 'ent_2'],
                'entity_names': ['John Doe', 'Acme Corp'],
                'entity_types': ['PERSON', 'ORGANIZATION']
            }
        ]
        
        qdrant_storage.search_by_entity_filter.return_value = mock_results
        
        # Test search
        results = await qdrant_storage.search_by_entity_filter(
            query_vector=[0.1, 0.2, 0.3],
            entity_ids=['ent_1'],
            entity_types=['PERSON'],
            top_k=10
        )
        
        assert len(results) == 1
        assert results[0]['mentioned_entities'] == ['ent_1', 'ent_2']
        assert 'PERSON' in results[0]['entity_types']
```

### Integration Tests

Create `tests/test_entity_cross_system_integration.py`:

```python
import pytest
import asyncio
from src.morag_graph.services.entity_integration_service import EntityIntegrationService
from src.morag_graph.storage.neo4j_storage import Neo4jStorage
from src.morag_graph.storage.qdrant_storage import QdrantStorage
from src.morag_graph.models.entity import Entity
from src.morag_graph.models.document_chunk import DocumentChunk

@pytest.mark.asyncio
@pytest.mark.integration
class TestEntityCrossSystemIntegration:
    
    @pytest.fixture
    async def integration_service(self):
        # Setup real storage connections
        neo4j = Neo4jStorage(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test"
        )
        qdrant = QdrantStorage(
            host="localhost",
            port=6333,
            collection_name="test_entity_integration"
        )
        
        await neo4j.connect()
        await qdrant.ensure_collection()
        
        service = EntityIntegrationService(neo4j, qdrant)
        
        yield service
        
        # Cleanup
        await neo4j.disconnect()
    
    async def test_end_to_end_entity_integration(self, integration_service):
        # Create test data
        chunk = DocumentChunk(
            document_id="doc_test123456789",
            chunk_index=0,
            text="John Doe is the CEO of Acme Corporation."
        )
        
        entities = [
            Entity(
                name="John Doe",
                type="PERSON",
                source_doc_id="doc_test123456789"
            ),
            Entity(
                name="Acme Corporation",
                type="ORGANIZATION",
                source_doc_id="doc_test123456789"
            )
        ]
        
        # Process chunk with entities
        result = await integration_service.process_chunk_with_entities(
            chunk=chunk,
            entities=entities,
            vector=[0.1, 0.2, 0.3, 0.4] * 96,  # 384-dim vector
            chunk_metadata={'test_integration': True}
        )
        
        # Verify processing
        assert result['integration_complete'] is True
        assert result['entities_processed'] == 2
        
        chunk_id = result['chunk_id']
        
        # Test entity-based search
        search_result = await integration_service.search_with_entity_context(
            query_vector=[0.1, 0.2, 0.3, 0.4] * 96,
            entity_context=[entities[0].id],
            top_k=5
        )
        
        assert len(search_result['results']) > 0
        assert search_result['graph_expansion_applied'] is True
        
        # Verify entity details in results
        first_result = search_result['results'][0]
        assert 'entity_details' in first_result
        assert len(first_result['entity_details']) == 2
        
        # Cleanup
        await integration_service.qdrant.delete_vectors_by_document_id("doc_test123456789")
        await integration_service.neo4j.execute_query(
            "MATCH (n) WHERE n.id CONTAINS 'doc_test123456789' DETACH DELETE n"
        )
```

## Performance Considerations

- **Entity ID Caching**: Cache frequently accessed entity IDs
- **Batch Processing**: Process entities in batches for large documents
- **Index Optimization**: Create indexes on entity fields in both systems
- **Memory Management**: Limit entity metadata size in Qdrant payloads

## Success Criteria

- [ ] Entity IDs maintain existing deterministic generation
- [ ] Entities properly linked between Neo4j and Qdrant
- [ ] Entity-based vector filtering works correctly
- [ ] Cross-system entity consistency validation passes
- [ ] Performance benchmarks meet requirements
- [ ] Integration tests pass with real databases
- [ ] Entity co-occurrence analysis functions correctly

## Next Steps

After completing this task:
1. Proceed to Phase 2: Cross-System Linking
2. Implement entity embedding generation and storage
3. Create entity-based query optimization
4. Plan entity relationship extraction enhancements

---

**Estimated Time**: 3-4 days  
**Dependencies**: Tasks 1.1, 1.2  
**Risk Level**: Medium (requires careful entity relationship management)