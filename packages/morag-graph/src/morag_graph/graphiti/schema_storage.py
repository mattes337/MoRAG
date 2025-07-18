"""Schema-aware storage service for custom entity types."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .entity_storage import GraphitiEntityStorage, EntityStorageResult
from .custom_schema import schema_registry, MoragEntityType, MoragRelationType

logger = logging.getLogger(__name__)


class SchemaAwareEntityStorage(GraphitiEntityStorage):
    """Enhanced entity storage with custom schema validation."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.schema_registry = schema_registry
    
    async def store_typed_entity(
        self,
        entity_data: Dict[str, Any],
        auto_deduplicate: bool = True,
        validate_schema: bool = True
    ) -> EntityStorageResult:
        """Store entity with schema validation.
        
        Args:
            entity_data: Raw entity data
            auto_deduplicate: Whether to check for existing entities
            validate_schema: Whether to validate against schema
            
        Returns:
            EntityStorageResult with validation info
        """
        try:
            # Validate against custom schema if requested
            if validate_schema:
                validated_data = self.schema_registry.validate_entity(entity_data)
                logger.debug(f"Entity validated against schema: {validated_data.get('type')}")
            else:
                validated_data = entity_data
            
            # Convert to MoRAG Entity model
            entity = self._dict_to_entity(validated_data)
            
            # Store using parent method
            result = await self.store_entity(entity, auto_deduplicate)
            
            # Add schema validation info to result
            if hasattr(result, 'metadata'):
                result.metadata = result.metadata or {}
                result.metadata['schema_validated'] = validate_schema
                result.metadata['entity_type'] = validated_data.get('type')
            
            return result
            
        except Exception as e:
            logger.error(f"Schema-aware entity storage failed: {e}")
            return EntityStorageResult(
                success=False,
                error=f"Schema validation/storage error: {str(e)}"
            )
    
    async def store_typed_relation(
        self,
        relation_data: Dict[str, Any],
        relation_category: str = "base",
        validate_schema: bool = True
    ) -> Dict[str, Any]:
        """Store relation with schema validation.
        
        Args:
            relation_data: Raw relation data
            relation_category: Category for schema selection
            validate_schema: Whether to validate against schema
            
        Returns:
            Storage result
        """
        try:
            # Validate against custom schema if requested
            if validate_schema:
                validated_data = self.schema_registry.validate_relation(relation_data, relation_category)
                logger.debug(f"Relation validated against {relation_category} schema")
            else:
                validated_data = relation_data
            
            # Convert to MoRAG Relation model
            relation = self._dict_to_relation(validated_data)
            
            # Store using parent method
            result = await self.store_relation(relation)
            
            return {
                'success': result.success,
                'relation_id': result.relation_id,
                'episode_id': result.episode_id,
                'schema_validated': validate_schema,
                'relation_category': relation_category,
                'error': result.error
            }
            
        except Exception as e:
            logger.error(f"Schema-aware relation storage failed: {e}")
            return {
                'success': False,
                'error': f"Schema validation/storage error: {str(e)}"
            }
    
    async def enhance_existing_entities(
        self,
        entity_type_filter: Optional[MoragEntityType] = None,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Enhance existing entities with schema-based validation and enrichment.
        
        Args:
            entity_type_filter: Optional filter for specific entity types
            batch_size: Number of entities to process per batch
            
        Returns:
            Enhancement results
        """
        results = {
            'total_processed': 0,
            'enhanced_entities': 0,
            'validation_errors': 0,
            'errors': []
        }
        
        try:
            # Search for existing entities
            search_query = "adapter_type:entity"
            if entity_type_filter:
                search_query += f" type:{entity_type_filter.value}"
            
            # Use search service if available
            if hasattr(self, 'search_service') and self.search_service:
                search_results, _ = await self.search_service.search(query=search_query, limit=1000)
            else:
                # Fallback to direct Graphiti search
                search_results = await self.graphiti.search(query=search_query, limit=1000)
            
            for i in range(0, len(search_results), batch_size):
                batch = search_results[i:i + batch_size]
                
                for result in batch:
                    try:
                        metadata = getattr(result, 'metadata', {})
                        
                        # Skip if already schema-validated
                        if metadata.get('schema_validated'):
                            continue
                        
                        # Validate and enhance
                        enhanced_data = self.schema_registry.validate_entity(metadata)
                        
                        # Update episode with enhanced data
                        # Note: This would require episode update functionality in Graphiti
                        # For now, we'll log the enhancement
                        logger.info(f"Enhanced entity {metadata.get('id')} with schema validation")
                        
                        results['enhanced_entities'] += 1
                        
                    except Exception as e:
                        results['validation_errors'] += 1
                        results['errors'].append(str(e))
                    
                    results['total_processed'] += 1
                
                logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} entities")
        
        except Exception as e:
            results['errors'].append(f"Enhancement process failed: {str(e)}")
        
        return results
    
    def _dict_to_entity(self, entity_data: Dict[str, Any]):
        """Convert dictionary to MoRAG Entity model."""
        try:
            from morag_graph.models import Entity, EntityType
            
            # Map custom types to MoRAG types
            morag_type = self._map_to_morag_type(entity_data.get('type'))
            
            return Entity(
                id=entity_data['id'],
                name=entity_data['name'],
                type=morag_type,
                confidence=entity_data.get('confidence', 0.5),
                attributes=entity_data.get('metadata', {}),
                source_doc_id=entity_data.get('source_document_id')
            )
        except ImportError:
            # Fallback if MoRAG models not available
            logger.warning("MoRAG models not available, using dict representation")
            return entity_data
    
    def _dict_to_relation(self, relation_data: Dict[str, Any]):
        """Convert dictionary to MoRAG Relation model."""
        try:
            from morag_graph.models import Relation, RelationType
            
            # Map custom types to MoRAG types
            morag_type = self._map_to_morag_relation_type(relation_data.get('relation_type'))
            
            return Relation(
                id=relation_data['id'],
                source_entity_id=relation_data['source_entity_id'],
                target_entity_id=relation_data['target_entity_id'],
                relation_type=morag_type,
                confidence=relation_data.get('confidence', 0.5),
                attributes=relation_data.get('metadata', {}),
                source_doc_id=relation_data.get('source_document_id')
            )
        except ImportError:
            # Fallback if MoRAG models not available
            logger.warning("MoRAG models not available, using dict representation")
            return relation_data
    
    def _map_to_morag_type(self, custom_type: str):
        """Map custom entity type to MoRAG EntityType."""
        try:
            from morag_graph.models import EntityType
            
            mapping = {
                'PERSON': EntityType.PERSON,
                'ORGANIZATION': EntityType.ORGANIZATION,
                'LOCATION': EntityType.LOCATION,
                'TECHNOLOGY': EntityType.TECHNOLOGY,
                'CONCEPT': EntityType.CONCEPT,
                'DOCUMENT': EntityType.DOCUMENT,
            }
            
            return mapping.get(custom_type, EntityType.UNKNOWN)
        except ImportError:
            return custom_type
    
    def _map_to_morag_relation_type(self, custom_type: str):
        """Map custom relation type to MoRAG RelationType."""
        try:
            from morag_graph.models import RelationType
            
            mapping = {
                'MENTIONS': RelationType.MENTIONS,
                'CONTAINS': RelationType.CONTAINS,
                'RELATED_TO': RelationType.RELATED_TO,
                'REFERENCES': RelationType.REFERENCES,
            }
            
            return mapping.get(custom_type, RelationType.UNKNOWN)
        except ImportError:
            return custom_type


class SchemaAwareSearchService:
    """Search service with schema-aware filtering and enhancement."""

    def __init__(self, storage_service: SchemaAwareEntityStorage):
        self.storage_service = storage_service
        self.graphiti = storage_service.graphiti

    async def search_by_entity_type(
        self,
        entity_type: MoragEntityType,
        query: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search entities by specific type.

        Args:
            entity_type: Entity type to search for
            query: Optional additional query
            limit: Maximum results

        Returns:
            List of typed entities
        """
        search_parts = [f"type:{entity_type.value}", "adapter_type:entity"]

        if query:
            search_parts.append(query)

        search_query = " AND ".join(search_parts)

        try:
            # Use search service if available
            if hasattr(self.storage_service, 'search_service') and self.storage_service.search_service:
                results, _ = await self.storage_service.search_service.search(query=search_query, limit=limit)
            else:
                # Fallback to direct Graphiti search
                results = await self.graphiti.search(query=search_query, limit=limit)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        typed_entities = []
        for result in results:
            metadata = getattr(result, 'metadata', {})

            # Validate against schema
            try:
                validated_data = schema_registry.validate_entity(metadata)
                typed_entities.append(validated_data)
            except Exception as e:
                logger.warning(f"Schema validation failed for search result: {e}")
                typed_entities.append(metadata)

        return typed_entities

    async def search_semantic_relations(
        self,
        relation_type: MoragRelationType,
        entity_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for semantic relations of specific type.

        Args:
            relation_type: Relation type to search for
            entity_id: Optional entity ID filter
            limit: Maximum results

        Returns:
            List of semantic relations
        """
        search_parts = [f"relation_type:{relation_type.value}", "adapter_type:relation"]

        if entity_id:
            search_parts.append(f"(source_entity_id:{entity_id} OR target_entity_id:{entity_id})")

        search_query = " AND ".join(search_parts)

        try:
            # Use search service if available
            if hasattr(self.storage_service, 'search_service') and self.storage_service.search_service:
                results, _ = await self.storage_service.search_service.search(query=search_query, limit=limit)
            else:
                # Fallback to direct Graphiti search
                results = await self.graphiti.search(query=search_query, limit=limit)
        except Exception as e:
            logger.error(f"Semantic relation search failed: {e}")
            return []

        semantic_relations = []
        for result in results:
            metadata = getattr(result, 'metadata', {})

            # Validate against semantic relation schema
            try:
                validated_data = schema_registry.validate_relation(metadata, "semantic")
                semantic_relations.append(validated_data)
            except Exception as e:
                logger.warning(f"Semantic relation validation failed: {e}")
                semantic_relations.append(metadata)

        return semantic_relations


# Convenience functions for creating schema-aware services
def create_schema_aware_storage(config=None) -> SchemaAwareEntityStorage:
    """Create a schema-aware entity storage service.

    Args:
        config: Optional Graphiti configuration

    Returns:
        SchemaAwareEntityStorage instance
    """
    return SchemaAwareEntityStorage(config)


def create_schema_aware_search(storage_service: SchemaAwareEntityStorage) -> SchemaAwareSearchService:
    """Create a schema-aware search service.

    Args:
        storage_service: Schema-aware storage service

    Returns:
        SchemaAwareSearchService instance
    """
    return SchemaAwareSearchService(storage_service)
