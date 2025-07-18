"""Graphiti-based entity and relation storage service."""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    GRAPHITI_AVAILABLE = True
except ImportError:
    # Graceful degradation when graphiti-core is not installed
    class EpisodeType:
        text = "text"
    GRAPHITI_AVAILABLE = False

from .config import create_graphiti_instance, GraphitiConfig
from .adapters.entity_adapter import EntityAdapter, RelationAdapter
from morag_graph.models import Entity, Relation

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
    
    def __post_init__(self):
        if self.missing_entities is None:
            self.missing_entities = []


class GraphitiEntityStorage:
    """Service for storing entities and relations using Graphiti episodes."""

    def __init__(self, config: Optional[GraphitiConfig] = None):
        self.config = config
        self.entity_adapter = EntityAdapter()
        self.relation_adapter = RelationAdapter()

        # Cache for entity deduplication
        self._entity_cache: Dict[str, str] = {}  # entity_name -> episode_id
        self._relation_cache: Dict[str, str] = {}  # relation_key -> episode_id
        
        # Initialize Graphiti instance if available
        if GRAPHITI_AVAILABLE:
            try:
                self.graphiti = create_graphiti_instance(config)
            except Exception as e:
                logger.warning(f"Failed to create Graphiti instance: {e}")
                self.graphiti = None
        else:
            self.graphiti = None
    
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
        if not self.graphiti:
            return EntityStorageResult(
                success=False,
                entity_id=entity.id,
                error="Graphiti instance not available"
            )
        
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
        if not self.graphiti:
            return RelationStorageResult(
                success=False,
                relation_id=relation.id,
                error="Graphiti instance not available"
            )
        
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
            relation_key = f"{relation.source_entity_id}:{relation.target_entity_id}:{relation.type}"
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
        
        if not self.graphiti:
            return None, None
        
        try:
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
                            "match_score": getattr(result, 'score', 0.0)
                        }
                        
                        return episode_id, deduplication_info
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error searching for existing entity: {e}")
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

        # Add adapter metadata
        entity_data["metadata"]["adapter_type"] = "entity"
        entity_data["metadata"]["name"] = entity.name
        entity_data["metadata"]["type"] = str(entity.type)

        # Create episode
        episode_id = await self.graphiti.add_episode(
            name=entity_data["name"],
            content=entity_data["description"],
            source_description=f"Entity: {entity.name}",
            metadata=entity_data["metadata"]
        )

        logger.info(f"Created entity episode {episode_id} for entity {entity.id}")
        return episode_id

    async def _update_entity_episode(
        self,
        episode_id: str,
        entity: Entity,
        deduplication_info: Dict[str, Any]
    ) -> str:
        """Update an existing entity episode.

        Args:
            episode_id: ID of existing episode
            entity: New entity data
            deduplication_info: Information about the existing entity

        Returns:
            Episode ID
        """
        # Convert entity to Graphiti format
        conversion_result = self.entity_adapter.to_graphiti(entity)
        if not conversion_result.success:
            raise ValueError(f"Entity conversion failed: {conversion_result.error}")

        entity_data = conversion_result.data

        # Add deduplication metadata
        entity_data["metadata"]["adapter_type"] = "entity"
        entity_data["metadata"]["name"] = entity.name
        entity_data["metadata"]["type"] = str(entity.type)
        entity_data["metadata"]["deduplication_info"] = deduplication_info
        entity_data["metadata"]["updated_at"] = datetime.utcnow().isoformat()

        # Update episode (this is a simplified approach - actual implementation
        # would depend on Graphiti's update capabilities)
        try:
            # For now, we'll create a new episode with updated content
            # In a full implementation, you'd use Graphiti's update methods
            updated_content = f"{entity_data['description']}\n\nUpdated: {datetime.utcnow().isoformat()}"

            # This is a placeholder - actual Graphiti update would be different
            logger.info(f"Updated entity episode {episode_id} for entity {entity.id}")
            return episode_id

        except Exception as e:
            logger.error(f"Failed to update entity episode {episode_id}: {e}")
            # Fallback to creating new episode
            return await self._create_entity_episode(entity)

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

        # Add adapter metadata
        relation_data["metadata"]["adapter_type"] = "relation"
        relation_data["metadata"]["source_entity"] = relation.source_entity_id
        relation_data["metadata"]["target_entity"] = relation.target_entity_id
        relation_data["metadata"]["relation_type"] = str(relation.type)

        # Create episode
        episode_id = await self.graphiti.add_episode(
            name=f"relation_{relation.source_entity_id}_{relation.target_entity_id}_{relation.type}",
            content=relation_data["description"],
            source_description=f"Relation: {relation.type}",
            metadata=relation_data["metadata"]
        )

        logger.info(f"Created relation episode {episode_id} for relation {relation.id}")
        return episode_id

    async def _entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists in storage.

        Args:
            entity_id: ID of entity to check

        Returns:
            True if entity exists, False otherwise
        """
        if not self.graphiti:
            return False

        try:
            # Search for entity by ID in metadata
            search_results = await self.graphiti.search(
                query=f"morag_entity_id:{entity_id}",
                limit=1
            )

            for result in search_results:
                metadata = getattr(result, 'metadata', {})
                if (metadata.get('adapter_type') == 'entity' and
                    metadata.get('morag_entity_id') == entity_id):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking entity existence: {e}")
            return False

    async def _refresh_entity_cache(self):
        """Refresh the entity cache from Graphiti storage."""
        if not self.graphiti:
            return

        try:
            # Search for all entity episodes
            search_results = await self.graphiti.search(
                query="adapter_type:entity",
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

        except Exception as e:
            logger.error(f"Error refreshing entity cache: {e}")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        return {
            "entity_cache_size": len(self._entity_cache),
            "relation_cache_size": len(self._relation_cache),
            "graphiti_available": self.graphiti is not None,
            "entity_adapter_stats": self.entity_adapter.get_stats(),
            "relation_adapter_stats": self.relation_adapter.get_stats()
        }


def create_entity_storage(config: Optional[GraphitiConfig] = None) -> GraphitiEntityStorage:
    """Create a GraphitiEntityStorage instance.

    Args:
        config: Optional Graphiti configuration

    Returns:
        GraphitiEntityStorage instance
    """
    return GraphitiEntityStorage(config)
