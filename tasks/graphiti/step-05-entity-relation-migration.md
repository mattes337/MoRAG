# Step 5: Entity and Relation Migration

**Duration**: 3-4 days
**Phase**: Core Integration
**Prerequisites**: Steps 1-4 completed, adapter layer working

## Objective

Replace MoRAG's direct Neo4j entity and relation storage with Graphiti's episode-based approach, while maintaining entity deduplication, relationship integrity, and search functionality.

## Deliverables

1. Graphiti-based entity and relation storage service
2. Entity deduplication using Graphiti's built-in capabilities
3. Relationship management through episode metadata
4. Migration utilities for existing Neo4j data
5. Comprehensive validation and testing framework

## Implementation

### 1. Create Graphiti Entity Storage Service

**File**: `packages/morag-graph/src/morag_graph/graphiti/entity_storage.py`

```python
"""Graphiti-based entity and relation storage service."""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from .config import create_graphiti_instance, GraphitiConfig
from .adapters.entity_adapter import EntityAdapter, RelationAdapter
from morag_graph.models import Entity, Relation, EntityId, RelationId

logger = logging.getLogger(__name__)


@dataclass
class EntityStorageResult:
    """Result of entity storage operation."""
    success: bool
    entity_id: Optional[str] = None
    episode_id: Optional[str] = None
    deduplication_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class RelationStorageResult:
    """Result of relation storage operation."""
    success: bool
    relation_id: Optional[str] = None
    episode_id: Optional[str] = None
    missing_entities: List[str] = None
    error: Optional[str] = None


class GraphitiEntityStorage:
    """Service for storing entities and relations using Graphiti episodes."""

    def __init__(self, config: Optional[GraphitiConfig] = None):
        self.config = config
        self.graphiti = create_graphiti_instance(config)
        self.entity_adapter = EntityAdapter()
        self.relation_adapter = RelationAdapter()

        # Cache for entity deduplication
        self._entity_cache: Dict[str, str] = {}  # entity_name -> episode_id
        self._relation_cache: Dict[str, str] = {}  # relation_key -> episode_id
    
    async def store_entity(
        self, 
        entity: Entity,
        auto_deduplicate: bool = True
    ) -> EntityStorageResult:
        """Store an entity using Graphiti episodes.
        
        Args:
            entity: Entity to store
            auto_deduplicate: Whether to check for existing entities
            
        Returns:
            EntityStorageResult with storage information
        """
        try:
            # Check for existing entity if deduplication is enabled
            existing_episode_id = None
            deduplication_info = None
            
            if auto_deduplicate:
                existing_episode_id, deduplication_info = await self._find_existing_entity(entity)
            
            if existing_episode_id:
                # Update existing entity episode
                episode_id = await self._update_entity_episode(
                    existing_episode_id, entity, deduplication_info
                )
                
                return EntityStorageResult(
                    success=True,
                    entity_id=entity.id,
                    episode_id=episode_id,
                    deduplication_info=deduplication_info
                )
            else:
                # Create new entity episode
                episode_id = await self._create_entity_episode(entity)
                
                # Update cache
                self._entity_cache[entity.name.lower()] = episode_id
                
                return EntityStorageResult(
                    success=True,
                    entity_id=entity.id,
                    episode_id=episode_id
                )
        
        except Exception as e:
            logger.error(f"Failed to store entity {entity.id}: {e}")
            return EntityStorageResult(
                success=False,
                entity_id=entity.id,
                error=str(e)
            )
    
    async def store_relation(
        self, 
        relation: Relation,
        ensure_entities_exist: bool = True
    ) -> RelationStorageResult:
        """Store a relation using Graphiti episodes.
        
        Args:
            relation: Relation to store
            ensure_entities_exist: Whether to verify source/target entities exist
            
        Returns:
            RelationStorageResult with storage information
        """
        try:
            missing_entities = []
            
            # Check if source and target entities exist
            if ensure_entities_exist:
                source_exists = await self._entity_exists(relation.source_entity_id)
                target_exists = await self._entity_exists(relation.target_entity_id)
                
                if not source_exists:
                    missing_entities.append(relation.source_entity_id)
                if not target_exists:
                    missing_entities.append(relation.target_entity_id)
                
                if missing_entities:
                    return RelationStorageResult(
                        success=False,
                        relation_id=relation.id,
                        missing_entities=missing_entities,
                        error=f"Missing entities: {missing_entities}"
                    )
            
            # Create relation episode
            episode_id = await self._create_relation_episode(relation)
            
            # Update cache
            relation_key = f"{relation.source_entity_id}:{relation.target_entity_id}:{relation.relation_type}"
            self._relation_cache[relation_key] = episode_id
            
            return RelationStorageResult(
                success=True,
                relation_id=relation.id,
                episode_id=episode_id
            )
        
        except Exception as e:
            logger.error(f"Failed to store relation {relation.id}: {e}")
            return RelationStorageResult(
                success=False,
                relation_id=relation.id,
                error=str(e)
            )
    
    async def store_entities_batch(
        self, 
        entities: List[Entity],
        auto_deduplicate: bool = True
    ) -> List[EntityStorageResult]:
        """Store multiple entities in batch.
        
        Args:
            entities: List of entities to store
            auto_deduplicate: Whether to deduplicate entities
            
        Returns:
            List of storage results
        """
        results = []
        
        # Pre-populate cache for better deduplication
        if auto_deduplicate:
            await self._refresh_entity_cache()
        
        for entity in entities:
            result = await self.store_entity(entity, auto_deduplicate)
            results.append(result)
            
            # Log progress for large batches
            if len(results) % 100 == 0:
                logger.info(f"Processed {len(results)}/{len(entities)} entities")
        
        return results
    
    async def store_relations_batch(
        self, 
        relations: List[Relation],
        ensure_entities_exist: bool = True
    ) -> List[RelationStorageResult]:
        """Store multiple relations in batch.
        
        Args:
            relations: List of relations to store
            ensure_entities_exist: Whether to verify entities exist
            
        Returns:
            List of storage results
        """
        results = []
        
        for relation in relations:
            result = await self.store_relation(relation, ensure_entities_exist)
            results.append(result)
            
            # Log progress for large batches
            if len(results) % 100 == 0:
                logger.info(f"Processed {len(results)}/{len(relations)} relations")
        
        return results
    
    async def _find_existing_entity(self, entity: Entity) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Find existing entity episode by name.
        
        Args:
            entity: Entity to search for
            
        Returns:
            Tuple of (episode_id, deduplication_info)
        """
        # Check cache first
        cache_key = entity.name.lower()
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key], {"source": "cache"}
        
        # Search Graphiti for existing entity
        search_results = await self.graphiti.search(
            query=entity.name,
            limit=10
        )
        
        for result in search_results:
            metadata = getattr(result, 'metadata', {})
            
            # Check if this is an entity episode
            if (metadata.get('adapter_type') == 'entity' and 
                metadata.get('name', '').lower() == entity.name.lower()):
                
                episode_id = getattr(result, 'episode_id', None)
                if episode_id:
                    # Update cache
                    self._entity_cache[cache_key] = episode_id
                    
                    deduplication_info = {
                        "source": "search",
                        "existing_confidence": metadata.get('confidence', 0.0),
                        "existing_type": metadata.get('type'),
                        "match_score": result.score
                    }
                    
                    return episode_id, deduplication_info
        
        return None, None
    
    async def _create_entity_episode(self, entity: Entity) -> str:
        """Create a new episode for an entity.
        
        Args:
            entity: Entity to create episode for
            
        Returns:
            Episode ID
        """
        # Convert entity to Graphiti format
        conversion_result = self.entity_adapter.to_graphiti(entity)
        if not conversion_result.success:
            raise ValueError(f"Entity conversion failed: {conversion_result.error}")
        
        entity_data = conversion_result.data
        
        # Create episode
        episode_id = await self.graphiti.add_episode(
            name=f"Entity: {entity.name}",
            episode_body=f"Entity '{entity.name}' of type {entity.type} with confidence {entity.confidence}",
            source_description=f"MoRAG Entity | Type: {entity.type} | Doc: {entity.source_doc_id}",
            episode_type=EpisodeType.json,  # Use JSON for structured entity data
            metadata=entity_data
        )
        
        logger.debug(f"Created entity episode {episode_id} for entity {entity.id}")
        return episode_id
    
    async def _update_entity_episode(
        self, 
        episode_id: str, 
        entity: Entity, 
        deduplication_info: Dict[str, Any]
    ) -> str:
        """Update existing entity episode with new information.
        
        Args:
            episode_id: Existing episode ID
            entity: New entity data
            deduplication_info: Information about the existing entity
            
        Returns:
            Episode ID (same as input)
        """
        # For now, Graphiti doesn't have direct episode update methods
        # We would need to implement this based on Graphiti's capabilities
        # This is a placeholder for the update logic
        
        logger.info(
            f"Entity deduplication: {entity.name} matched existing episode {episode_id} "
            f"(confidence: {deduplication_info.get('existing_confidence', 'unknown')})"
        )
        
        # In a full implementation, we might:
        # 1. Retrieve the existing episode
        # 2. Merge the entity data (keeping highest confidence, etc.)
        # 3. Update the episode with merged data
        
        return episode_id
    
    async def _create_relation_episode(self, relation: Relation) -> str:
        """Create a new episode for a relation.
        
        Args:
            relation: Relation to create episode for
            
        Returns:
            Episode ID
        """
        # Convert relation to Graphiti format
        conversion_result = self.relation_adapter.to_graphiti(relation)
        if not conversion_result.success:
            raise ValueError(f"Relation conversion failed: {conversion_result.error}")
        
        relation_data = conversion_result.data
        
        # Create episode
        episode_id = await self.graphiti.add_episode(
            name=f"Relation: {relation.source_entity_id} -> {relation.target_entity_id}",
            episode_body=f"Relation of type {relation.relation_type} between entities {relation.source_entity_id} and {relation.target_entity_id}",
            source_description=f"MoRAG Relation | Type: {relation.relation_type} | Doc: {relation.source_doc_id}",
            episode_type=EpisodeType.json,
            metadata=relation_data
        )
        
        logger.debug(f"Created relation episode {episode_id} for relation {relation.id}")
        return episode_id
    
    async def _entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists in Graphiti.
        
        Args:
            entity_id: Entity ID to check
            
        Returns:
            True if entity exists
        """
        # Search for entity by ID
        search_results = await self.graphiti.search(
            query=entity_id,
            limit=5
        )
        
        for result in search_results:
            metadata = getattr(result, 'metadata', {})
            if (metadata.get('adapter_type') == 'entity' and 
                metadata.get('id') == entity_id):
                return True
        
        return False
    
    async def _refresh_entity_cache(self):
        """Refresh the entity cache by searching for all entity episodes."""
        # This is a simplified implementation
        # In practice, we might want to implement pagination for large datasets
        
        search_results = await self.graphiti.search(
            query="adapter_type:entity",  # Search for entity episodes
            limit=1000  # Adjust based on expected entity count
        )
        
        self._entity_cache.clear()
        
        for result in search_results:
            metadata = getattr(result, 'metadata', {})
            if metadata.get('adapter_type') == 'entity':
                entity_name = metadata.get('name', '').lower()
                episode_id = getattr(result, 'episode_id', None)
                if entity_name and episode_id:
                    self._entity_cache[entity_name] = episode_id
        
        logger.info(f"Refreshed entity cache with {len(self._entity_cache)} entities")


class EntityMigrationService:
    """Service for migrating entities and relations from Neo4j to Graphiti."""
    
    def __init__(
        self, 
        graphiti_storage: GraphitiEntityStorage,
        neo4j_storage: Optional[Any] = None  # Neo4jStorage instance
    ):
        self.graphiti_storage = graphiti_storage
        self.neo4j_storage = neo4j_storage
    
    async def migrate_entities_from_neo4j(
        self, 
        batch_size: int = 100,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Migrate entities from Neo4j to Graphiti.
        
        Args:
            batch_size: Number of entities to process per batch
            dry_run: If True, don't actually store in Graphiti
            
        Returns:
            Migration results
        """
        if not self.neo4j_storage:
            raise ValueError("Neo4j storage not configured for migration")
        
        results = {
            "total_entities": 0,
            "migrated_entities": 0,
            "failed_entities": 0,
            "deduplicated_entities": 0,
            "errors": [],
            "dry_run": dry_run
        }
        
        try:
            # Get all entities from Neo4j
            all_entities = await self.neo4j_storage.get_all_entities()
            results["total_entities"] = len(all_entities)
            
            logger.info(f"Starting entity migration: {len(all_entities)} entities")
            
            # Process in batches
            for i in range(0, len(all_entities), batch_size):
                batch = all_entities[i:i + batch_size]
                
                if not dry_run:
                    batch_results = await self.graphiti_storage.store_entities_batch(batch)
                    
                    for result in batch_results:
                        if result.success:
                            results["migrated_entities"] += 1
                            if result.deduplication_info:
                                results["deduplicated_entities"] += 1
                        else:
                            results["failed_entities"] += 1
                            results["errors"].append(result.error)
                else:
                    # Dry run - just count
                    results["migrated_entities"] += len(batch)
                
                logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} entities")
        
        except Exception as e:
            logger.error(f"Entity migration failed: {e}")
            results["errors"].append(str(e))
        
        return results
    
    async def migrate_relations_from_neo4j(
        self, 
        batch_size: int = 100,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Migrate relations from Neo4j to Graphiti.
        
        Args:
            batch_size: Number of relations to process per batch
            dry_run: If True, don't actually store in Graphiti
            
        Returns:
            Migration results
        """
        if not self.neo4j_storage:
            raise ValueError("Neo4j storage not configured for migration")
        
        results = {
            "total_relations": 0,
            "migrated_relations": 0,
            "failed_relations": 0,
            "missing_entities": 0,
            "errors": [],
            "dry_run": dry_run
        }
        
        try:
            # Get all relations from Neo4j
            all_relations = await self.neo4j_storage.get_all_relations()
            results["total_relations"] = len(all_relations)
            
            logger.info(f"Starting relation migration: {len(all_relations)} relations")
            
            # Process in batches
            for i in range(0, len(all_relations), batch_size):
                batch = all_relations[i:i + batch_size]
                
                if not dry_run:
                    batch_results = await self.graphiti_storage.store_relations_batch(batch)
                    
                    for result in batch_results:
                        if result.success:
                            results["migrated_relations"] += 1
                        else:
                            results["failed_relations"] += 1
                            if result.missing_entities:
                                results["missing_entities"] += len(result.missing_entities)
                            results["errors"].append(result.error)
                else:
                    # Dry run - just count
                    results["migrated_relations"] += len(batch)
                
                logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} relations")
        
        except Exception as e:
            logger.error(f"Relation migration failed: {e}")
            results["errors"].append(str(e))
        
        return results
```

### 2. Create Entity Search and Retrieval Service

**File**: `packages/morag-graph/src/morag_graph/graphiti/entity_search.py`

```python
"""Entity search and retrieval using Graphiti."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .entity_storage import GraphitiEntityStorage
from .adapters.entity_adapter import EntityAdapter, RelationAdapter
from morag_graph.models import Entity, Relation

logger = logging.getLogger(__name__)


@dataclass
class EntitySearchResult:
    """Result of entity search operation."""
    entity: Entity
    episode_id: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class RelationSearchResult:
    """Result of relation search operation."""
    relation: Relation
    episode_id: str
    score: float
    metadata: Dict[str, Any]


class GraphitiEntitySearch:
    """Service for searching entities and relations in Graphiti."""
    
    def __init__(self, entity_storage: GraphitiEntityStorage):
        self.entity_storage = entity_storage
        self.graphiti = entity_storage.graphiti
        self.entity_adapter = EntityAdapter()
        self.relation_adapter = RelationAdapter()
    
    async def search_entities(
        self, 
        query: str, 
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[EntitySearchResult]:
        """Search for entities by name or attributes.
        
        Args:
            query: Search query
            entity_type: Optional entity type filter
            limit: Maximum results
            
        Returns:
            List of entity search results
        """
        # Construct search query
        search_query = query
        if entity_type:
            search_query += f" type:{entity_type}"
        
        # Search Graphiti
        search_results = await self.graphiti.search(
            query=search_query,
            limit=limit * 2  # Get more results to filter
        )
        
        entity_results = []
        
        for result in search_results:
            metadata = getattr(result, 'metadata', {})
            
            # Filter for entity episodes
            if metadata.get('adapter_type') != 'entity':
                continue
            
            # Convert back to Entity
            conversion_result = self.entity_adapter.from_graphiti(metadata)
            if not conversion_result.success:
                logger.warning(f"Failed to convert entity from search result: {conversion_result.error}")
                continue
            
            entity_result = EntitySearchResult(
                entity=conversion_result.data,
                episode_id=getattr(result, 'episode_id', ''),
                score=result.score,
                metadata=metadata
            )
            
            entity_results.append(entity_result)
            
            if len(entity_results) >= limit:
                break
        
        return entity_results
    
    async def get_entity_by_id(self, entity_id: str) -> Optional[EntitySearchResult]:
        """Get entity by ID.
        
        Args:
            entity_id: Entity ID to search for
            
        Returns:
            EntitySearchResult if found, None otherwise
        """
        search_results = await self.search_entities(entity_id, limit=5)
        
        for result in search_results:
            if result.entity.id == entity_id:
                return result
        
        return None
    
    async def get_entities_by_document(self, document_id: str) -> List[EntitySearchResult]:
        """Get all entities associated with a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of entities from the document
        """
        search_results = await self.graphiti.search(
            query=f"source_doc_id:{document_id}",
            limit=100
        )
        
        entity_results = []
        
        for result in search_results:
            metadata = getattr(result, 'metadata', {})
            
            if (metadata.get('adapter_type') == 'entity' and 
                metadata.get('source_doc_id') == document_id):
                
                conversion_result = self.entity_adapter.from_graphiti(metadata)
                if conversion_result.success:
                    entity_result = EntitySearchResult(
                        entity=conversion_result.data,
                        episode_id=getattr(result, 'episode_id', ''),
                        score=result.score,
                        metadata=metadata
                    )
                    entity_results.append(entity_result)
        
        return entity_results
    
    async def search_relations(
        self, 
        source_entity_id: Optional[str] = None,
        target_entity_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: int = 10
    ) -> List[RelationSearchResult]:
        """Search for relations by entity IDs or type.
        
        Args:
            source_entity_id: Source entity ID filter
            target_entity_id: Target entity ID filter
            relation_type: Relation type filter
            limit: Maximum results
            
        Returns:
            List of relation search results
        """
        # Construct search query
        query_parts = ["adapter_type:relation"]
        
        if source_entity_id:
            query_parts.append(f"source_entity_id:{source_entity_id}")
        if target_entity_id:
            query_parts.append(f"target_entity_id:{target_entity_id}")
        if relation_type:
            query_parts.append(f"relation_type:{relation_type}")
        
        search_query = " AND ".join(query_parts)
        
        # Search Graphiti
        search_results = await self.graphiti.search(
            query=search_query,
            limit=limit
        )
        
        relation_results = []
        
        for result in search_results:
            metadata = getattr(result, 'metadata', {})
            
            # Convert back to Relation
            conversion_result = self.relation_adapter.from_graphiti(metadata)
            if not conversion_result.success:
                logger.warning(f"Failed to convert relation from search result: {conversion_result.error}")
                continue
            
            relation_result = RelationSearchResult(
                relation=conversion_result.data,
                episode_id=getattr(result, 'episode_id', ''),
                score=result.score,
                metadata=metadata
            )
            
            relation_results.append(relation_result)
        
        return relation_results
    
    async def get_entity_relations(self, entity_id: str) -> Tuple[List[RelationSearchResult], List[RelationSearchResult]]:
        """Get all relations for an entity (both incoming and outgoing).
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Tuple of (outgoing_relations, incoming_relations)
        """
        # Get outgoing relations (entity as source)
        outgoing = await self.search_relations(source_entity_id=entity_id)
        
        # Get incoming relations (entity as target)
        incoming = await self.search_relations(target_entity_id=entity_id)
        
        return outgoing, incoming
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_graphiti_entity_migration.py`

```python
"""Unit tests for Graphiti entity migration."""

import pytest
from unittest.mock import Mock, AsyncMock
from morag_graph.models import Entity, Relation, EntityType, RelationType
from morag_graph.graphiti.entity_storage import GraphitiEntityStorage, EntityStorageResult
from morag_graph.graphiti.entity_search import GraphitiEntitySearch


class TestGraphitiEntityStorage:
    """Test Graphiti entity storage functionality."""
    
    @pytest.fixture
    def mock_entity_storage(self):
        """Create mock entity storage."""
        storage = GraphitiEntityStorage()
        storage.graphiti = Mock()
        storage.graphiti.add_episode = AsyncMock()
        storage.graphiti.search = AsyncMock()
        return storage
    
    @pytest.fixture
    def sample_entity(self):
        """Create sample entity."""
        return Entity(
            id="entity_123",
            name="John Doe",
            type=EntityType.PERSON,
            confidence=0.95,
            source_doc_id="doc_123"
        )
    
    @pytest.mark.asyncio
    async def test_store_entity_new(self, mock_entity_storage, sample_entity):
        """Test storing a new entity."""
        # Mock no existing entity found
        mock_entity_storage.graphiti.search.return_value = []
        mock_entity_storage.graphiti.add_episode.return_value = "episode_123"
        
        result = await mock_entity_storage.store_entity(sample_entity)
        
        assert result.success is True
        assert result.entity_id == "entity_123"
        assert result.episode_id == "episode_123"
        mock_entity_storage.graphiti.add_episode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_entity_deduplication(self, mock_entity_storage, sample_entity):
        """Test entity deduplication."""
        # Mock existing entity found
        mock_result = Mock()
        mock_result.score = 0.9
        mock_result.metadata = {
            'adapter_type': 'entity',
            'name': 'John Doe',
            'confidence': 0.8
        }
        mock_result.episode_id = "existing_episode"
        
        mock_entity_storage.graphiti.search.return_value = [mock_result]
        
        result = await mock_entity_storage.store_entity(sample_entity)
        
        assert result.success is True
        assert result.episode_id == "existing_episode"
        assert result.deduplication_info is not None
        assert result.deduplication_info["source"] == "search"
    
    @pytest.mark.asyncio
    async def test_batch_entity_storage(self, mock_entity_storage):
        """Test batch entity storage."""
        entities = [
            Entity(id=f"entity_{i}", name=f"Entity {i}", type=EntityType.PERSON)
            for i in range(3)
        ]
        
        mock_entity_storage.graphiti.search.return_value = []
        mock_entity_storage.graphiti.add_episode.return_value = "episode_id"
        
        results = await mock_entity_storage.store_entities_batch(entities)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert mock_entity_storage.graphiti.add_episode.call_count == 3


class TestGraphitiEntitySearch:
    """Test Graphiti entity search functionality."""
    
    @pytest.fixture
    def mock_entity_search(self):
        """Create mock entity search."""
        mock_storage = Mock()
        mock_storage.graphiti = Mock()
        mock_storage.graphiti.search = AsyncMock()
        
        search = GraphitiEntitySearch(mock_storage)
        return search
    
    @pytest.mark.asyncio
    async def test_search_entities(self, mock_entity_search):
        """Test entity search functionality."""
        # Mock search results
        mock_result = Mock()
        mock_result.score = 0.9
        mock_result.episode_id = "episode_123"
        mock_result.metadata = {
            'adapter_type': 'entity',
            'id': 'entity_123',
            'name': 'John Doe',
            'type': 'PERSON',
            'confidence': 0.95,
            'attributes': {},
            'source_doc_id': 'doc_123',
            'mentioned_in_chunks': []
        }
        
        mock_entity_search.entity_storage.graphiti.search.return_value = [mock_result]
        
        results = await mock_entity_search.search_entities("John Doe")
        
        assert len(results) == 1
        assert results[0].entity.name == "John Doe"
        assert results[0].episode_id == "episode_123"
        assert results[0].score == 0.9
    
    @pytest.mark.asyncio
    async def test_get_entity_by_id(self, mock_entity_search):
        """Test getting entity by ID."""
        # Mock search results
        mock_result = Mock()
        mock_result.score = 1.0
        mock_result.episode_id = "episode_123"
        mock_result.metadata = {
            'adapter_type': 'entity',
            'id': 'entity_123',
            'name': 'John Doe',
            'type': 'PERSON',
            'confidence': 0.95,
            'attributes': {},
            'source_doc_id': 'doc_123',
            'mentioned_in_chunks': []
        }
        
        mock_entity_search.entity_storage.graphiti.search.return_value = [mock_result]
        
        result = await mock_entity_search.get_entity_by_id("entity_123")
        
        assert result is not None
        assert result.entity.id == "entity_123"
        assert result.entity.name == "John Doe"


@pytest.mark.integration
class TestEntityMigration:
    """Integration tests for entity migration."""
    
    @pytest.mark.asyncio
    async def test_migration_dry_run(self):
        """Test migration dry run."""
        from morag_graph.graphiti.entity_storage import EntityMigrationService
        
        # Mock components
        mock_graphiti_storage = Mock()
        mock_neo4j_storage = Mock()
        mock_neo4j_storage.get_all_entities = AsyncMock(return_value=[
            Entity(id="e1", name="Entity 1", type=EntityType.PERSON),
            Entity(id="e2", name="Entity 2", type=EntityType.ORGANIZATION)
        ])
        
        migration_service = EntityMigrationService(
            mock_graphiti_storage, 
            mock_neo4j_storage
        )
        
        results = await migration_service.migrate_entities_from_neo4j(dry_run=True)
        
        assert results["dry_run"] is True
        assert results["total_entities"] == 2
        assert results["migrated_entities"] == 2
        assert results["failed_entities"] == 0
```

## Validation Checklist

- [ ] Entity storage using Graphiti episodes works correctly
- [ ] Entity deduplication prevents duplicate storage
- [ ] Relation storage maintains entity references
- [ ] Batch processing handles large datasets efficiently
- [ ] Entity search returns accurate results
- [ ] Relation search supports various filter combinations
- [ ] Migration service can transfer data from Neo4j
- [ ] Error handling for missing entities and failed conversions
- [ ] Performance acceptable for expected data volumes
- [ ] Unit tests cover all major functionality

## Success Criteria

1. **Functional**: Entities and relations store and retrieve correctly
2. **Deduplication**: Automatic entity deduplication prevents duplicates
3. **Performance**: Batch operations handle large datasets efficiently
4. **Search**: Entity and relation search provides accurate results
5. **Migration**: Existing Neo4j data can be migrated successfully

## Next Steps

After completing this step:
1. Test migration with sample Neo4j data
2. Validate entity deduplication accuracy
3. Benchmark performance against direct Neo4j storage
4. Proceed to [Step 6: Ingestion Coordinator Integration](./step-06-coordinator-integration.md)

## Performance Considerations

- Entity cache improves deduplication performance
- Batch processing reduces individual episode creation overhead
- Search indexing in Graphiti affects query performance
- Memory usage scales with entity cache size and batch sizes
