"""Entity deduplication service for removing duplicate entities from the knowledge graph."""

import asyncio
from typing import Dict, List, Set, Tuple, Optional, Any
import structlog

from .entity_normalizer import EnhancedEntityNormalizer, EntityMergeCandidate
from ..storage.neo4j_storage import Neo4jStorage
from ..models.types import EntityId
from ..config.normalization_config import get_config_for_component

logger = structlog.get_logger(__name__)


class EntityDeduplicator:
    """Service for deduplicating entities in the knowledge graph."""
    
    def __init__(self, neo4j_storage: Neo4jStorage, llm_service=None, config: Optional[Dict[str, Any]] = None):
        """Initialize entity deduplicator.
        
        Args:
            neo4j_storage: Neo4j storage instance
            llm_service: LLM service for normalization (optional)
            config: Optional configuration dictionary
        """
        self.neo4j = neo4j_storage

        # Load configuration from file or use provided config
        if config is None:
            self.config = get_config_for_component('deduplicator')
        else:
            # Merge provided config with defaults
            default_config = get_config_for_component('deduplicator')
            default_config.update(config)
            self.config = default_config

        # Initialize normalizer with configuration
        self.normalizer = EnhancedEntityNormalizer(llm_service, self.config)

        # Configuration
        self.merge_confidence_threshold = self.config.get('merge_confidence_threshold', 0.8)
        self.dry_run = self.config.get('dry_run', False)
        
        logger.info(
            "Entity deduplicator initialized",
            merge_confidence_threshold=self.merge_confidence_threshold,
            dry_run=self.dry_run
        )

    async def deduplicate_entities(self, collection_name: str = None, language: str = None) -> Dict[str, Any]:
        """Deduplicate entities in a collection or entire graph.
        
        Args:
            collection_name: Optional collection name to limit deduplication scope
            language: Optional language context for normalization
            
        Returns:
            Dictionary with deduplication results and statistics
        """
        logger.info(
            "Starting entity deduplication",
            collection_name=collection_name,
            language=language,
            dry_run=self.dry_run
        )
        
        # 1. Get all entities
        entities = await self._get_all_entities(collection_name)
        logger.info("Retrieved entities for deduplication", count=len(entities))
        
        if len(entities) < 2:
            logger.info("Not enough entities for deduplication")
            return {
                'total_entities_before': len(entities),
                'merge_candidates_found': 0,
                'merges_applied': 0,
                'merge_results': []
            }

        # 2. Find merge candidates using LLM-based analysis
        entity_names = [entity['name'] for entity in entities]
        candidates = await self.normalizer.find_merge_candidates(entity_names, language)
        
        logger.info("Found merge candidates", count=len(candidates))

        # 3. Apply merges if not in dry run mode
        merge_results = []
        if not self.dry_run:
            for candidate in candidates:
                if candidate.confidence >= self.merge_confidence_threshold:
                    try:
                        result = await self._merge_entities(candidate, collection_name)
                        merge_results.append(result)
                        logger.info(
                            "Successfully merged entities",
                            entities=candidate.entities,
                            canonical_form=candidate.canonical_form,
                            confidence=candidate.confidence
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to merge entities",
                            entities=candidate.entities,
                            error=str(e)
                        )
        else:
            logger.info("Dry run mode - no merges applied")

        return {
            'total_entities_before': len(entities),
            'merge_candidates_found': len(candidates),
            'merges_applied': len(merge_results),
            'merge_results': merge_results,
            'candidates': [
                {
                    'entities': candidate.entities,
                    'canonical_form': candidate.canonical_form,
                    'confidence': candidate.confidence,
                    'reason': candidate.merge_reason
                }
                for candidate in candidates
            ]
        }

    async def _get_all_entities(self, collection_name: str = None) -> List[Dict[str, Any]]:
        """Get all entity names from the graph.
        
        Args:
            collection_name: Optional collection name to filter entities
            
        Returns:
            List of entity dictionaries with name and metadata
        """
        if collection_name:
            # Get entities for specific collection
            query = """
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:DocumentChunk)
            WHERE c.collection_name = $collection_name
            RETURN DISTINCT e.name as name, e.id as id, e.type as type, e.confidence as confidence
            """
            params = {'collection_name': collection_name}
        else:
            # Get all entities
            query = """
            MATCH (e:Entity)
            RETURN e.name as name, e.id as id, e.type as type, e.confidence as confidence
            """
            params = {}
        
        # Execute query using the storage's query operations
        if hasattr(self.neo4j, '_query_ops') and self.neo4j._query_ops:
            result = await self.neo4j._query_ops._execute_query(query, params)
        else:
            # Fallback to direct driver usage
            async with self.neo4j.driver.session(database=self.neo4j.config.database) as session:
                result = await session.run(query, params)
                records = await result.data()
                return records
        
        return [dict(record) for record in result]

    async def _merge_entities(self, candidate: EntityMergeCandidate, collection_name: str = None) -> Dict[str, Any]:
        """Merge entities in the graph.
        
        Args:
            candidate: Entity merge candidate with entities to merge
            collection_name: Optional collection name for scoping
            
        Returns:
            Dictionary with merge results
        """
        entities_to_merge = candidate.entities
        canonical_form = candidate.canonical_form
        
        logger.debug(
            "Merging entities",
            entities_to_merge=entities_to_merge,
            canonical_form=canonical_form
        )
        
        # Find the canonical entity or use the first one as canonical
        canonical_entity_name = canonical_form
        if canonical_form not in entities_to_merge:
            canonical_entity_name = entities_to_merge[0]
            
        # Merge all relationships to canonical entity
        merge_query = """
        // Find all entities to merge
        MATCH (old:Entity)
        WHERE old.name IN $entities_to_merge 
          AND old.name <> $canonical_form
        
        // Find or create canonical entity
        MERGE (canonical:Entity {name: $canonical_form})
        ON CREATE SET 
            canonical.id = old.id,
            canonical.type = old.type,
            canonical.confidence = old.confidence,
            canonical.created_at = datetime(),
            canonical.updated_at = datetime()
        ON MATCH SET
            canonical.confidence = CASE 
                WHEN old.confidence > canonical.confidence THEN old.confidence 
                ELSE canonical.confidence 
            END,
            canonical.updated_at = datetime()
        
        // Transfer all outgoing relationships
        WITH old, canonical
        OPTIONAL MATCH (old)-[r1]->(other)
        WHERE NOT (canonical)-[r1_type:SAME_TYPE]->(other)
        FOREACH (rel IN CASE WHEN r1 IS NOT NULL THEN [r1] ELSE [] END |
            CREATE (canonical)-[new_rel:SAME_TYPE]->(other)
            SET new_rel = properties(rel)
        )
        
        // Transfer all incoming relationships  
        WITH old, canonical
        OPTIONAL MATCH (other)-[r2]->(old)
        WHERE NOT (other)-[r2_type:SAME_TYPE]->(canonical)
        FOREACH (rel IN CASE WHEN r2 IS NOT NULL THEN [r2] ELSE [] END |
            CREATE (other)-[new_rel:SAME_TYPE]->(canonical)
            SET new_rel = properties(rel)
        )
        
        // Delete old entities and their relationships
        WITH old, canonical
        DETACH DELETE old
        
        RETURN count(old) as merged_count, canonical.name as canonical_name
        """
        
        params = {
            'entities_to_merge': entities_to_merge,
            'canonical_form': canonical_entity_name
        }
        
        # Execute merge query
        if hasattr(self.neo4j, '_query_ops') and self.neo4j._query_ops:
            result = await self.neo4j._query_ops._execute_query(merge_query, params)
        else:
            # Fallback to direct driver usage
            async with self.neo4j.driver.session(database=self.neo4j.config.database) as session:
                result = await session.run(merge_query, params)
                records = await result.data()
                result = records
        
        merged_count = result[0]['merged_count'] if result else 0
        
        return {
            'merged_entities': entities_to_merge,
            'canonical_form': canonical_entity_name,
            'confidence': candidate.confidence,
            'reason': candidate.merge_reason,
            'merged_count': merged_count
        }

    async def get_duplicate_candidates(self, collection_name: str = None, language: str = None) -> List[Dict[str, Any]]:
        """Get duplicate candidates without performing merges (dry run).
        
        Args:
            collection_name: Optional collection name to limit scope
            language: Optional language context
            
        Returns:
            List of merge candidates
        """
        entities = await self._get_all_entities(collection_name)
        entity_names = [entity['name'] for entity in entities]
        candidates = await self.normalizer.find_merge_candidates(entity_names, language)
        
        return [
            {
                'entities': candidate.entities,
                'canonical_form': candidate.canonical_form,
                'confidence': candidate.confidence,
                'reason': candidate.merge_reason
            }
            for candidate in candidates
            if candidate.confidence >= self.merge_confidence_threshold
        ]
