# Task 2.1: Cross-System Entity Linking

## Overview

Establish robust entity linking mechanisms between Neo4j and Qdrant to enable seamless entity-based queries and relationship traversal across both systems. This task creates the foundation for hybrid retrieval by ensuring entities can be efficiently referenced and linked between vector and graph databases.

## Objectives

- Create bidirectional entity reference system
- Implement entity synchronization mechanisms
- Establish entity relationship mapping
- Enable cross-system entity queries
- Create entity consistency monitoring

## Dependencies

- Task 1.1: Unified ID Architecture
- Task 1.2: Document and Chunk ID Standardization
- Task 1.3: Entity ID Integration
- Existing Neo4j relationship model

## Implementation Plan

### Step 1: Entity Reference Manager

Create `src/morag_graph/services/entity_reference_manager.py`:

```python
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from ..models.entity import Entity
from ..models.relation import Relation
from ..storage.neo4j_storage import Neo4jStorage
from ..storage.qdrant_storage import QdrantStorage
from ..utils.id_generation import UnifiedIDGenerator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class EntityReference:
    """Entity reference across systems."""
    entity_id: str
    neo4j_node_id: Optional[str] = None
    qdrant_vector_ids: List[str] = None
    chunk_references: List[str] = None
    last_sync: Optional[datetime] = None
    sync_status: str = "pending"  # pending, synced, error
    
    def __post_init__(self):
        if self.qdrant_vector_ids is None:
            self.qdrant_vector_ids = []
        if self.chunk_references is None:
            self.chunk_references = []

@dataclass
class EntityLinkage:
    """Entity linkage information."""
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence_score: float
    co_occurrence_count: int
    shared_documents: List[str]
    last_updated: datetime

class EntityReferenceManager:
    """Manages entity references across Neo4j and Qdrant systems."""
    
    def __init__(self, neo4j_storage: Neo4jStorage, qdrant_storage: QdrantStorage):
        self.neo4j = neo4j_storage
        self.qdrant = qdrant_storage
        self._reference_cache: Dict[str, EntityReference] = {}
        self._linkage_cache: Dict[str, List[EntityLinkage]] = {}
        self._sync_lock = asyncio.Lock()
    
    async def create_entity_reference(self, entity: Entity, chunk_ids: List[str]) -> EntityReference:
        """Create comprehensive entity reference across systems.
        
        Args:
            entity: Entity instance
            chunk_ids: List of chunk IDs where entity is mentioned
            
        Returns:
            EntityReference instance
        """
        async with self._sync_lock:
            # Store entity in Neo4j and get node ID
            neo4j_node_id = await self.neo4j.store_entity_with_chunk_references(
                entity=entity,
                chunk_ids=chunk_ids
            )
            
            # Get Qdrant vector IDs for chunks containing this entity
            qdrant_vector_ids = []
            for chunk_id in chunk_ids:
                # Chunk ID is the same as vector ID in our unified system
                qdrant_vector_ids.append(chunk_id)
            
            # Create reference
            reference = EntityReference(
                entity_id=entity.id,
                neo4j_node_id=neo4j_node_id,
                qdrant_vector_ids=qdrant_vector_ids,
                chunk_references=chunk_ids,
                last_sync=datetime.utcnow(),
                sync_status="synced"
            )
            
            # Cache reference
            self._reference_cache[entity.id] = reference
            
            # Update entity with cross-system references
            await self._update_entity_cross_references(entity.id, reference)
            
            logger.info(f"Created entity reference for {entity.id} with {len(chunk_ids)} chunk references")
            
            return reference
    
    async def _update_entity_cross_references(self, entity_id: str, reference: EntityReference):
        """Update entity with cross-system reference information."""
        # Update Neo4j entity with Qdrant references
        await self.neo4j.update_entity_qdrant_references(
            entity_id=entity_id,
            qdrant_vector_ids=reference.qdrant_vector_ids
        )
        
        # Update Qdrant vectors with entity reference metadata
        for vector_id in reference.qdrant_vector_ids:
            await self._update_qdrant_vector_entity_metadata(vector_id, entity_id)
    
    async def _update_qdrant_vector_entity_metadata(self, vector_id: str, entity_id: str):
        """Update Qdrant vector with entity reference metadata."""
        try:
            # Get current vector
            result = await self.qdrant.client.retrieve(
                collection_name=self.qdrant.collection_name,
                ids=[vector_id],
                with_payload=True
            )
            
            if result:
                current_payload = result[0].payload
                
                # Add entity reference to metadata
                entity_refs = current_payload.get('neo4j_entity_refs', [])
                if entity_id not in entity_refs:
                    entity_refs.append(entity_id)
                
                # Update payload
                updated_payload = {
                    **current_payload,
                    'neo4j_entity_refs': entity_refs,
                    'cross_system_sync': True,
                    'last_entity_sync': datetime.utcnow().isoformat()
                }
                
                # Update vector
                await self.qdrant.client.set_payload(
                    collection_name=self.qdrant.collection_name,
                    payload=updated_payload,
                    points=[vector_id]
                )
                
        except Exception as e:
            logger.error(f"Failed to update Qdrant vector {vector_id} with entity {entity_id}: {e}")
    
    async def get_entity_reference(self, entity_id: str) -> Optional[EntityReference]:
        """Get entity reference from cache or rebuild.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            EntityReference or None
        """
        # Check cache first
        if entity_id in self._reference_cache:
            return self._reference_cache[entity_id]
        
        # Rebuild from systems
        return await self._rebuild_entity_reference(entity_id)
    
    async def _rebuild_entity_reference(self, entity_id: str) -> Optional[EntityReference]:
        """Rebuild entity reference from Neo4j and Qdrant."""
        try:
            # Get entity from Neo4j
            entity_query = """
            MATCH (e:Entity {id: $entity_id})
            RETURN e.qdrant_vector_ids as qdrant_vector_ids,
                   e.mentioned_in_chunks as chunk_references,
                   e.last_updated as last_sync,
                   id(e) as neo4j_node_id
            """
            
            result = await self.neo4j.execute_query(entity_query, entity_id=entity_id)
            
            if not result:
                return None
            
            entity_data = result[0]
            
            reference = EntityReference(
                entity_id=entity_id,
                neo4j_node_id=str(entity_data['neo4j_node_id']),
                qdrant_vector_ids=entity_data.get('qdrant_vector_ids', []),
                chunk_references=entity_data.get('chunk_references', []),
                last_sync=datetime.fromisoformat(entity_data['last_sync']) if entity_data.get('last_sync') else None,
                sync_status="synced"
            )
            
            # Cache reference
            self._reference_cache[entity_id] = reference
            
            return reference
            
        except Exception as e:
            logger.error(f"Failed to rebuild entity reference for {entity_id}: {e}")
            return None
    
    async def find_entity_linkages(self, entity_id: str, linkage_types: Optional[List[str]] = None) -> List[EntityLinkage]:
        """Find entity linkages based on co-occurrence and relationships.
        
        Args:
            entity_id: Source entity ID
            linkage_types: Types of linkages to find
            
        Returns:
            List of EntityLinkage instances
        """
        # Check cache
        cache_key = f"{entity_id}:{':'.join(linkage_types or [])}"
        if cache_key in self._linkage_cache:
            return self._linkage_cache[cache_key]
        
        linkages = []
        
        # Find co-occurrence based linkages from Qdrant
        co_occurrence_linkages = await self._find_co_occurrence_linkages(entity_id)
        linkages.extend(co_occurrence_linkages)
        
        # Find relationship based linkages from Neo4j
        relationship_linkages = await self._find_relationship_linkages(entity_id, linkage_types)
        linkages.extend(relationship_linkages)
        
        # Merge and deduplicate
        unique_linkages = self._merge_linkages(linkages)
        
        # Cache results
        self._linkage_cache[cache_key] = unique_linkages
        
        return unique_linkages
    
    async def _find_co_occurrence_linkages(self, entity_id: str) -> List[EntityLinkage]:
        """Find entity linkages based on co-occurrence in Qdrant."""
        linkages = []
        
        try:
            # Get co-occurrence statistics from Qdrant
            co_occurrence_stats = await self.qdrant.get_entity_co_occurrence_stats(entity_id)
            
            # Convert to linkages
            for other_entity_id, count in co_occurrence_stats.get('co_occurring_entities', {}).items():
                # Calculate confidence based on co-occurrence frequency
                confidence = min(count / 10.0, 1.0)  # Normalize to 0-1
                
                # Get shared documents
                shared_docs = await self._get_shared_documents(entity_id, other_entity_id)
                
                linkage = EntityLinkage(
                    source_entity_id=entity_id,
                    target_entity_id=other_entity_id,
                    relationship_type="CO_OCCURS_WITH",
                    confidence_score=confidence,
                    co_occurrence_count=count,
                    shared_documents=shared_docs,
                    last_updated=datetime.utcnow()
                )
                
                linkages.append(linkage)
                
        except Exception as e:
            logger.error(f"Failed to find co-occurrence linkages for {entity_id}: {e}")
        
        return linkages
    
    async def _find_relationship_linkages(self, entity_id: str, linkage_types: Optional[List[str]]) -> List[EntityLinkage]:
        """Find entity linkages based on explicit relationships in Neo4j."""
        linkages = []
        
        try:
            # Build relationship type filter
            type_filter = ""
            if linkage_types:
                type_filter = f"AND type(r) IN {linkage_types}"
            
            query = f"""
            MATCH (e1:Entity {{id: $entity_id}})-[r]-(e2:Entity)
            WHERE e1.id <> e2.id {type_filter}
            RETURN e2.id as target_entity_id,
                   type(r) as relationship_type,
                   r.confidence as confidence,
                   r.created_at as created_at
            """
            
            result = await self.neo4j.execute_query(query, entity_id=entity_id)
            
            for rel_data in result:
                # Get shared documents
                shared_docs = await self._get_shared_documents(entity_id, rel_data['target_entity_id'])
                
                linkage = EntityLinkage(
                    source_entity_id=entity_id,
                    target_entity_id=rel_data['target_entity_id'],
                    relationship_type=rel_data['relationship_type'],
                    confidence_score=rel_data.get('confidence', 0.8),
                    co_occurrence_count=0,  # Not applicable for explicit relationships
                    shared_documents=shared_docs,
                    last_updated=datetime.fromisoformat(rel_data['created_at']) if rel_data.get('created_at') else datetime.utcnow()
                )
                
                linkages.append(linkage)
                
        except Exception as e:
            logger.error(f"Failed to find relationship linkages for {entity_id}: {e}")
        
        return linkages
    
    async def _get_shared_documents(self, entity_id1: str, entity_id2: str) -> List[str]:
        """Get documents where both entities are mentioned."""
        try:
            query = """
            MATCH (e1:Entity {id: $entity_id1})-[:MENTIONED_IN]->(c1:DocumentChunk)<-[:HAS_CHUNK]-(d:Document)
            MATCH (e2:Entity {id: $entity_id2})-[:MENTIONED_IN]->(c2:DocumentChunk)<-[:HAS_CHUNK]-(d)
            RETURN DISTINCT d.id as document_id
            """
            
            result = await self.neo4j.execute_query(
                query,
                entity_id1=entity_id1,
                entity_id2=entity_id2
            )
            
            return [row['document_id'] for row in result]
            
        except Exception as e:
            logger.error(f"Failed to get shared documents for {entity_id1} and {entity_id2}: {e}")
            return []
    
    def _merge_linkages(self, linkages: List[EntityLinkage]) -> List[EntityLinkage]:
        """Merge and deduplicate linkages."""
        linkage_map = {}
        
        for linkage in linkages:
            key = f"{linkage.source_entity_id}:{linkage.target_entity_id}:{linkage.relationship_type}"
            
            if key in linkage_map:
                # Merge linkages - take higher confidence and combine data
                existing = linkage_map[key]
                existing.confidence_score = max(existing.confidence_score, linkage.confidence_score)
                existing.co_occurrence_count = max(existing.co_occurrence_count, linkage.co_occurrence_count)
                existing.shared_documents = list(set(existing.shared_documents + linkage.shared_documents))
                existing.last_updated = max(existing.last_updated, linkage.last_updated)
            else:
                linkage_map[key] = linkage
        
        return list(linkage_map.values())
    
    async def sync_entity_references(self, entity_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Synchronize entity references across systems.
        
        Args:
            entity_ids: Specific entity IDs to sync, or None for all
            
        Returns:
            Sync statistics
        """
        async with self._sync_lock:
            if entity_ids is None:
                # Get all entity IDs from Neo4j
                result = await self.neo4j.execute_query(
                    "MATCH (e:Entity) RETURN e.id as entity_id"
                )
                entity_ids = [row['entity_id'] for row in result]
            
            sync_stats = {
                'total_entities': len(entity_ids),
                'synced_successfully': 0,
                'sync_errors': 0,
                'error_details': []
            }
            
            for entity_id in entity_ids:
                try:
                    # Rebuild reference
                    reference = await self._rebuild_entity_reference(entity_id)
                    
                    if reference:
                        # Verify cross-system consistency
                        is_consistent = await self._verify_entity_consistency(entity_id, reference)
                        
                        if is_consistent:
                            reference.sync_status = "synced"
                            sync_stats['synced_successfully'] += 1
                        else:
                            reference.sync_status = "error"
                            sync_stats['sync_errors'] += 1
                            sync_stats['error_details'].append(f"Inconsistency detected for {entity_id}")
                        
                        # Update cache
                        self._reference_cache[entity_id] = reference
                    else:
                        sync_stats['sync_errors'] += 1
                        sync_stats['error_details'].append(f"Failed to rebuild reference for {entity_id}")
                        
                except Exception as e:
                    sync_stats['sync_errors'] += 1
                    sync_stats['error_details'].append(f"Error syncing {entity_id}: {str(e)}")
                    logger.error(f"Error syncing entity {entity_id}: {e}")
            
            logger.info(f"Entity sync completed: {sync_stats['synced_successfully']}/{sync_stats['total_entities']} successful")
            
            return sync_stats
    
    async def _verify_entity_consistency(self, entity_id: str, reference: EntityReference) -> bool:
        """Verify entity consistency across systems."""
        try:
            # Check if all chunk references exist in both systems
            for chunk_id in reference.chunk_references:
                # Check Neo4j
                neo4j_result = await self.neo4j.execute_query(
                    "MATCH (c:DocumentChunk {id: $chunk_id}) RETURN count(c) as count",
                    chunk_id=chunk_id
                )
                
                if not neo4j_result or neo4j_result[0]['count'] == 0:
                    logger.warning(f"Chunk {chunk_id} not found in Neo4j for entity {entity_id}")
                    return False
                
                # Check Qdrant
                qdrant_result = await self.qdrant.client.retrieve(
                    collection_name=self.qdrant.collection_name,
                    ids=[chunk_id]
                )
                
                if not qdrant_result:
                    logger.warning(f"Vector {chunk_id} not found in Qdrant for entity {entity_id}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying consistency for entity {entity_id}: {e}")
            return False
    
    async def get_cross_system_query_plan(self, entity_ids: List[str], query_type: str = "hybrid") -> Dict[str, Any]:
        """Generate query plan for cross-system entity queries.
        
        Args:
            entity_ids: Entity IDs to query
            query_type: Type of query (vector, graph, hybrid)
            
        Returns:
            Query execution plan
        """
        plan = {
            'query_type': query_type,
            'entity_ids': entity_ids,
            'neo4j_operations': [],
            'qdrant_operations': [],
            'fusion_strategy': 'weighted_score',
            'estimated_cost': 0
        }
        
        # Get entity references
        references = []
        for entity_id in entity_ids:
            ref = await self.get_entity_reference(entity_id)
            if ref:
                references.append(ref)
        
        if query_type in ['graph', 'hybrid']:
            # Plan Neo4j operations
            plan['neo4j_operations'] = [
                {
                    'operation': 'entity_expansion',
                    'entity_ids': entity_ids,
                    'max_hops': 2,
                    'relationship_types': ['RELATED_TO', 'CO_OCCURS_WITH']
                },
                {
                    'operation': 'chunk_retrieval',
                    'chunk_ids': [chunk_id for ref in references for chunk_id in ref.chunk_references]
                }
            ]
            plan['estimated_cost'] += len(entity_ids) * 10  # Neo4j query cost
        
        if query_type in ['vector', 'hybrid']:
            # Plan Qdrant operations
            vector_ids = [vector_id for ref in references for vector_id in ref.qdrant_vector_ids]
            
            plan['qdrant_operations'] = [
                {
                    'operation': 'entity_filtered_search',
                    'entity_filter': entity_ids,
                    'vector_ids': vector_ids
                },
                {
                    'operation': 'similarity_search',
                    'top_k': 50,
                    'include_metadata': True
                }
            ]
            plan['estimated_cost'] += len(vector_ids) * 5  # Qdrant search cost
        
        if query_type == 'hybrid':
            plan['fusion_strategy'] = 'reciprocal_rank_fusion'
            plan['estimated_cost'] += 20  # Fusion overhead
        
        return plan
    
    async def clear_cache(self):
        """Clear reference and linkage caches."""
        self._reference_cache.clear()
        self._linkage_cache.clear()
        logger.info("Entity reference caches cleared")
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'reference_cache_size': len(self._reference_cache),
            'linkage_cache_size': len(self._linkage_cache),
            'memory_usage_estimate': (
                len(self._reference_cache) * 1024 +  # Rough estimate
                len(self._linkage_cache) * 2048
            )
        }
```

### Step 2: Cross-System Query Engine

Create `src/morag_graph/services/cross_system_query_engine.py`:

```python
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from .entity_reference_manager import EntityReferenceManager, EntityLinkage
from ..storage.neo4j_storage import Neo4jStorage
from ..storage.qdrant_storage import QdrantStorage
from ..models.entity import Entity
import logging

logger = logging.getLogger(__name__)

class CrossSystemQueryEngine:
    """Engine for executing queries across Neo4j and Qdrant systems."""
    
    def __init__(self, 
                 neo4j_storage: Neo4jStorage, 
                 qdrant_storage: QdrantStorage,
                 entity_reference_manager: EntityReferenceManager):
        self.neo4j = neo4j_storage
        self.qdrant = qdrant_storage
        self.entity_manager = entity_reference_manager
    
    async def execute_hybrid_entity_query(self,
                                         query_vector: List[float],
                                         entity_context: List[str],
                                         expand_entities: bool = True,
                                         max_hops: int = 2,
                                         vector_top_k: int = 50,
                                         final_top_k: int = 10) -> Dict[str, Any]:
        """Execute hybrid query combining vector search and entity graph expansion.
        
        Args:
            query_vector: Query embedding vector
            entity_context: Entity IDs for context
            expand_entities: Whether to expand entity context using graph
            max_hops: Maximum hops for entity expansion
            vector_top_k: Top K for vector search
            final_top_k: Final number of results to return
            
        Returns:
            Hybrid query results
        """
        start_time = datetime.utcnow()
        
        # Phase 1: Entity Context Expansion (if enabled)
        expanded_entities = set(entity_context)
        if expand_entities and entity_context:
            expansion_results = await self._expand_entity_context(
                entity_context, max_hops
            )
            expanded_entities.update(expansion_results['expanded_entity_ids'])
            logger.info(f"Expanded {len(entity_context)} entities to {len(expanded_entities)}")
        
        # Phase 2: Parallel Vector and Graph Queries
        vector_task = asyncio.create_task(
            self._execute_vector_query(query_vector, list(expanded_entities), vector_top_k)
        )
        
        graph_task = asyncio.create_task(
            self._execute_graph_query(list(expanded_entities))
        )
        
        vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
        
        # Phase 3: Result Fusion
        fused_results = await self._fuse_results(
            vector_results, graph_results, final_top_k
        )
        
        # Phase 4: Result Enrichment
        enriched_results = await self._enrich_results(fused_results)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'results': enriched_results,
            'total_results': len(enriched_results),
            'entity_context_original': entity_context,
            'entity_context_expanded': list(expanded_entities),
            'vector_results_count': len(vector_results),
            'graph_results_count': len(graph_results),
            'execution_time_seconds': execution_time,
            'query_metadata': {
                'expand_entities': expand_entities,
                'max_hops': max_hops,
                'vector_top_k': vector_top_k,
                'final_top_k': final_top_k
            }
        }
    
    async def _expand_entity_context(self, entity_ids: List[str], max_hops: int) -> Dict[str, Any]:
        """Expand entity context using graph relationships."""
        expanded_entities = set(entity_ids)
        expansion_paths = []
        
        for hop in range(max_hops):
            current_entities = list(expanded_entities)
            new_entities = set()
            
            # Find linkages for current entities
            for entity_id in current_entities:
                linkages = await self.entity_manager.find_entity_linkages(
                    entity_id, 
                    linkage_types=['RELATED_TO', 'CO_OCCURS_WITH', 'SIMILAR_TO']
                )
                
                for linkage in linkages:
                    if linkage.confidence_score >= 0.5:  # Confidence threshold
                        new_entities.add(linkage.target_entity_id)
                        expansion_paths.append({
                            'hop': hop + 1,
                            'source': entity_id,
                            'target': linkage.target_entity_id,
                            'relationship': linkage.relationship_type,
                            'confidence': linkage.confidence_score
                        })
            
            # Add new entities to expanded set
            before_count = len(expanded_entities)
            expanded_entities.update(new_entities)
            after_count = len(expanded_entities)
            
            logger.info(f"Hop {hop + 1}: Added {after_count - before_count} new entities")
            
            # Stop if no new entities found
            if before_count == after_count:
                break
        
        return {
            'expanded_entity_ids': list(expanded_entities),
            'expansion_paths': expansion_paths,
            'total_hops': min(max_hops, hop + 1)
        }
    
    async def _execute_vector_query(self, 
                                  query_vector: List[float], 
                                  entity_filter: List[str], 
                                  top_k: int) -> List[Dict[str, Any]]:
        """Execute vector search with entity filtering."""
        try:
            results = await self.qdrant.search_by_entity_filter(
                query_vector=query_vector,
                entity_ids=entity_filter,
                top_k=top_k
            )
            
            # Add source information
            for result in results:
                result['source'] = 'vector'
                result['vector_score'] = result['score']
            
            return results
            
        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            return []
    
    async def _execute_graph_query(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """Execute graph-based query for entity-related chunks."""
        try:
            # Get chunks related to entities through graph relationships
            query = """
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:DocumentChunk)
            WHERE e.id IN $entity_ids
            OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(d:Document)
            RETURN DISTINCT c.id as chunk_id,
                   c.text as text,
                   c.chunk_index as chunk_index,
                   d.id as document_id,
                   d.file_name as file_name,
                   collect(DISTINCT e.id) as mentioned_entities,
                   collect(DISTINCT e.name) as entity_names,
                   collect(DISTINCT e.type) as entity_types
            ORDER BY c.chunk_index
            """
            
            result = await self.neo4j.execute_query(query, entity_ids=entity_ids)
            
            # Format results
            graph_results = []
            for row in result:
                graph_results.append({
                    'id': row['chunk_id'],
                    'chunk_id': row['chunk_id'],
                    'text': row['text'],
                    'document_id': row['document_id'],
                    'file_name': row['file_name'],
                    'mentioned_entities': row['mentioned_entities'],
                    'entity_names': row['entity_names'],
                    'entity_types': row['entity_types'],
                    'source': 'graph',
                    'graph_score': 1.0,  # Base score for graph results
                    'chunk_index': row['chunk_index']
                })
            
            return graph_results
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return []
    
    async def _fuse_results(self, 
                          vector_results: List[Dict[str, Any]], 
                          graph_results: List[Dict[str, Any]], 
                          top_k: int) -> List[Dict[str, Any]]:
        """Fuse vector and graph results using reciprocal rank fusion."""
        # Create result map
        result_map = {}
        
        # Process vector results
        for i, result in enumerate(vector_results):
            chunk_id = result['chunk_id']
            if chunk_id not in result_map:
                result_map[chunk_id] = result.copy()
                result_map[chunk_id]['fusion_sources'] = []
            
            # Add vector ranking
            result_map[chunk_id]['vector_rank'] = i + 1
            result_map[chunk_id]['vector_score'] = result['vector_score']
            result_map[chunk_id]['fusion_sources'].append('vector')
        
        # Process graph results
        for i, result in enumerate(graph_results):
            chunk_id = result['chunk_id']
            if chunk_id not in result_map:
                result_map[chunk_id] = result.copy()
                result_map[chunk_id]['fusion_sources'] = []
            
            # Add graph ranking
            result_map[chunk_id]['graph_rank'] = i + 1
            result_map[chunk_id]['graph_score'] = result['graph_score']
            result_map[chunk_id]['fusion_sources'].append('graph')
            
            # Merge entity information if not present
            if 'mentioned_entities' not in result_map[chunk_id]:
                result_map[chunk_id]['mentioned_entities'] = result['mentioned_entities']
                result_map[chunk_id]['entity_names'] = result['entity_names']
                result_map[chunk_id]['entity_types'] = result['entity_types']
        
        # Calculate fusion scores using Reciprocal Rank Fusion
        for chunk_id, result in result_map.items():
            fusion_score = 0.0
            
            # Vector contribution
            if 'vector_rank' in result:
                fusion_score += 1.0 / (60 + result['vector_rank'])  # RRF with k=60
            
            # Graph contribution
            if 'graph_rank' in result:
                fusion_score += 1.0 / (60 + result['graph_rank'])  # RRF with k=60
            
            result['fusion_score'] = fusion_score
            result['is_hybrid'] = len(result['fusion_sources']) > 1
        
        # Sort by fusion score and return top K
        fused_results = sorted(
            result_map.values(), 
            key=lambda x: x['fusion_score'], 
            reverse=True
        )[:top_k]
        
        return fused_results
    
    async def _enrich_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich results with additional entity and document information."""
        enriched = []
        
        for result in results:
            # Get detailed entity information
            entity_details = []
            for entity_id in result.get('mentioned_entities', []):
                entity_ref = await self.entity_manager.get_entity_reference(entity_id)
                if entity_ref:
                    # Get entity linkages
                    linkages = await self.entity_manager.find_entity_linkages(entity_id)
                    
                    entity_details.append({
                        'id': entity_id,
                        'chunk_references_count': len(entity_ref.chunk_references),
                        'qdrant_vector_count': len(entity_ref.qdrant_vector_ids),
                        'linkages_count': len(linkages),
                        'last_sync': entity_ref.last_sync.isoformat() if entity_ref.last_sync else None
                    })
            
            # Add enrichment data
            result['entity_details'] = entity_details
            result['entity_linkage_strength'] = sum(
                detail['linkages_count'] for detail in entity_details
            )
            
            enriched.append(result)
        
        return enriched
    
    async def execute_entity_similarity_query(self, 
                                            entity_id: str, 
                                            similarity_threshold: float = 0.7,
                                            top_k: int = 10) -> Dict[str, Any]:
        """Find entities similar to a given entity across both systems.
        
        Args:
            entity_id: Source entity ID
            similarity_threshold: Minimum similarity score
            top_k: Number of similar entities to return
            
        Returns:
            Similar entities with scores
        """
        # Get entity linkages
        linkages = await self.entity_manager.find_entity_linkages(entity_id)
        
        # Filter by similarity threshold
        similar_entities = [
            linkage for linkage in linkages 
            if linkage.confidence_score >= similarity_threshold
        ]
        
        # Sort by confidence and limit
        similar_entities.sort(key=lambda x: x.confidence_score, reverse=True)
        similar_entities = similar_entities[:top_k]
        
        # Get additional information for similar entities
        enriched_similar = []
        for linkage in similar_entities:
            target_ref = await self.entity_manager.get_entity_reference(linkage.target_entity_id)
            
            enriched_similar.append({
                'entity_id': linkage.target_entity_id,
                'relationship_type': linkage.relationship_type,
                'confidence_score': linkage.confidence_score,
                'co_occurrence_count': linkage.co_occurrence_count,
                'shared_documents': linkage.shared_documents,
                'target_chunk_count': len(target_ref.chunk_references) if target_ref else 0,
                'target_vector_count': len(target_ref.qdrant_vector_ids) if target_ref else 0
            })
        
        return {
            'source_entity_id': entity_id,
            'similar_entities': enriched_similar,
            'total_similar': len(enriched_similar),
            'similarity_threshold': similarity_threshold
        }
    
    async def get_query_performance_stats(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        # Get cache statistics
        cache_stats = await self.entity_manager.get_cache_statistics()
        
        # Get system statistics
        neo4j_stats = await self.neo4j.execute_query(
            """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c:DocumentChunk)
            RETURN count(DISTINCT e) as total_entities,
                   count(DISTINCT c) as total_chunks,
                   avg(size(e.qdrant_vector_ids)) as avg_vector_refs_per_entity
            """
        )
        
        # Get Qdrant collection info
        collection_info = await self.qdrant.client.get_collection(
            collection_name=self.qdrant.collection_name
        )
        
        neo4j_data = neo4j_stats[0] if neo4j_stats else {}
        
        return {
            'cache_statistics': cache_stats,
            'neo4j_statistics': {
                'total_entities': neo4j_data.get('total_entities', 0),
                'total_chunks': neo4j_data.get('total_chunks', 0),
                'avg_vector_refs_per_entity': neo4j_data.get('avg_vector_refs_per_entity', 0)
            },
            'qdrant_statistics': {
                'total_vectors': collection_info.vectors_count,
                'collection_status': collection_info.status.name
            }
        }
```

## Testing

### Unit Tests

Create `tests/test_cross_system_entity_linking.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from src.morag_graph.services.entity_reference_manager import (
    EntityReferenceManager, EntityReference, EntityLinkage
)
from src.morag_graph.services.cross_system_query_engine import CrossSystemQueryEngine
from src.morag_graph.models.entity import Entity

@pytest.mark.asyncio
class TestEntityReferenceManager:
    
    @pytest.fixture
    def mock_storages(self):
        neo4j_storage = AsyncMock()
        qdrant_storage = AsyncMock()
        return neo4j_storage, qdrant_storage
    
    @pytest.fixture
    def entity_manager(self, mock_storages):
        neo4j_storage, qdrant_storage = mock_storages
        return EntityReferenceManager(neo4j_storage, qdrant_storage)
    
    @pytest.fixture
    def sample_entity(self):
        return Entity(
            name="Test Entity",
            type="PERSON",
            source_doc_id="doc_1234567890123456"
        )
    
    async def test_create_entity_reference(self, entity_manager, sample_entity, mock_storages):
        neo4j_storage, qdrant_storage = mock_storages
        
        # Mock storage responses
        neo4j_storage.store_entity_with_chunk_references.return_value = sample_entity.id
        neo4j_storage.update_entity_qdrant_references.return_value = None
        qdrant_storage.client.retrieve.return_value = [MagicMock(payload={})]
        qdrant_storage.client.set_payload.return_value = None
        
        chunk_ids = ["doc_1234567890123456:chunk:0000", "doc_1234567890123456:chunk:0001"]
        
        # Create entity reference
        reference = await entity_manager.create_entity_reference(sample_entity, chunk_ids)
        
        # Verify reference
        assert reference.entity_id == sample_entity.id
        assert reference.chunk_references == chunk_ids
        assert reference.qdrant_vector_ids == chunk_ids
        assert reference.sync_status == "synced"
        
        # Verify storage calls
        neo4j_storage.store_entity_with_chunk_references.assert_called_once_with(
            entity=sample_entity,
            chunk_ids=chunk_ids
        )
        neo4j_storage.update_entity_qdrant_references.assert_called_once()
    
    async def test_find_entity_linkages(self, entity_manager, mock_storages):
        neo4j_storage, qdrant_storage = mock_storages
        entity_id = "ent_test123456789"
        
        # Mock co-occurrence stats
        qdrant_storage.get_entity_co_occurrence_stats.return_value = {
            'co_occurring_entities': {
                'ent_other123456789': 5,
                'ent_another123456': 3
            }
        }
        
        # Mock relationship query
        neo4j_storage.execute_query.return_value = [
            {
                'target_entity_id': 'ent_related123456',
                'relationship_type': 'RELATED_TO',
                'confidence': 0.8,
                'created_at': datetime.utcnow().isoformat()
            }
        ]
        
        # Mock shared documents
        entity_manager._get_shared_documents = AsyncMock(return_value=['doc_123', 'doc_456'])
        
        # Find linkages
        linkages = await entity_manager.find_entity_linkages(entity_id)
        
        # Verify linkages
        assert len(linkages) >= 2  # At least co-occurrence and relationship linkages
        
        # Check co-occurrence linkage
        co_occur_linkages = [l for l in linkages if l.relationship_type == 'CO_OCCURS_WITH']
        assert len(co_occur_linkages) >= 1
        
        # Check relationship linkage
        rel_linkages = [l for l in linkages if l.relationship_type == 'RELATED_TO']
        assert len(rel_linkages) >= 1
    
    async def test_sync_entity_references(self, entity_manager, mock_storages):
        neo4j_storage, qdrant_storage = mock_storages
        
        # Mock entity IDs query
        neo4j_storage.execute_query.return_value = [
            {'entity_id': 'ent_1'},
            {'entity_id': 'ent_2'}
        ]
        
        # Mock rebuild reference
        entity_manager._rebuild_entity_reference = AsyncMock(
            return_value=EntityReference(
                entity_id='ent_1',
                sync_status='synced'
            )
        )
        entity_manager._verify_entity_consistency = AsyncMock(return_value=True)
        
        # Sync references
        stats = await entity_manager.sync_entity_references()
        
        # Verify stats
        assert stats['total_entities'] == 2
        assert stats['synced_successfully'] >= 0
        assert stats['sync_errors'] >= 0

@pytest.mark.asyncio
class TestCrossSystemQueryEngine:
    
    @pytest.fixture
    def mock_components(self):
        neo4j_storage = AsyncMock()
        qdrant_storage = AsyncMock()
        entity_manager = AsyncMock()
        return neo4j_storage, qdrant_storage, entity_manager
    
    @pytest.fixture
    def query_engine(self, mock_components):
        neo4j_storage, qdrant_storage, entity_manager = mock_components
        return CrossSystemQueryEngine(neo4j_storage, qdrant_storage, entity_manager)
    
    async def test_execute_hybrid_entity_query(self, query_engine, mock_components):
        neo4j_storage, qdrant_storage, entity_manager = mock_components
        
        # Mock entity expansion
        entity_manager.find_entity_linkages.return_value = [
            EntityLinkage(
                source_entity_id='ent_1',
                target_entity_id='ent_2',
                relationship_type='RELATED_TO',
                confidence_score=0.8,
                co_occurrence_count=5,
                shared_documents=['doc_1'],
                last_updated=datetime.utcnow()
            )
        ]
        
        # Mock vector search
        qdrant_storage.search_by_entity_filter.return_value = [
            {
                'id': 'chunk_1',
                'chunk_id': 'chunk_1',
                'score': 0.95,
                'mentioned_entities': ['ent_1'],
                'text': 'Test chunk'
            }
        ]
        
        # Mock graph query
        neo4j_storage.execute_query.return_value = [
            {
                'chunk_id': 'chunk_2',
                'text': 'Graph chunk',
                'chunk_index': 0,
                'document_id': 'doc_1',
                'file_name': 'test.txt',
                'mentioned_entities': ['ent_1'],
                'entity_names': ['Test Entity'],
                'entity_types': ['PERSON']
            }
        ]
        
        # Mock entity reference
        entity_manager.get_entity_reference.return_value = EntityReference(
            entity_id='ent_1',
            chunk_references=['chunk_1'],
            qdrant_vector_ids=['chunk_1']
        )
        
        # Execute hybrid query
        result = await query_engine.execute_hybrid_entity_query(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            entity_context=['ent_1'],
            expand_entities=True,
            final_top_k=5
        )
        
        # Verify result structure
        assert 'results' in result
        assert 'total_results' in result
        assert 'entity_context_original' in result
        assert 'entity_context_expanded' in result
        assert 'execution_time_seconds' in result
        
        # Verify entity expansion
        assert len(result['entity_context_expanded']) >= len(result['entity_context_original'])
    
    async def test_entity_similarity_query(self, query_engine, mock_components):
        neo4j_storage, qdrant_storage, entity_manager = mock_components
        
        # Mock entity linkages
        entity_manager.find_entity_linkages.return_value = [
            EntityLinkage(
                source_entity_id='ent_1',
                target_entity_id='ent_2',
                relationship_type='SIMILAR_TO',
                confidence_score=0.9,
                co_occurrence_count=10,
                shared_documents=['doc_1', 'doc_2'],
                last_updated=datetime.utcnow()
            )
        ]
        
        # Mock entity reference
        entity_manager.get_entity_reference.return_value = EntityReference(
            entity_id='ent_2',
            chunk_references=['chunk_1', 'chunk_2'],
            qdrant_vector_ids=['chunk_1', 'chunk_2']
        )
        
        # Execute similarity query
        result = await query_engine.execute_entity_similarity_query(
            entity_id='ent_1',
            similarity_threshold=0.7,
            top_k=5
        )
        
        # Verify result
        assert result['source_entity_id'] == 'ent_1'
        assert len(result['similar_entities']) >= 1
        
        similar_entity = result['similar_entities'][0]
        assert similar_entity['entity_id'] == 'ent_2'
        assert similar_entity['confidence_score'] == 0.9
        assert similar_entity['target_chunk_count'] == 2
```

### Integration Tests

Create `tests/test_cross_system_integration.py`:

```python
import pytest
import asyncio
from src.morag_graph.services.entity_reference_manager import EntityReferenceManager
from src.morag_graph.services.cross_system_query_engine import CrossSystemQueryEngine
from src.morag_graph.storage.neo4j_storage import Neo4jStorage
from src.morag_graph.storage.qdrant_storage import QdrantStorage
from src.morag_graph.models.entity import Entity
from src.morag_graph.models.document_chunk import DocumentChunk

@pytest.mark.asyncio
@pytest.mark.integration
class TestCrossSystemIntegration:
    
    @pytest.fixture
    async def integrated_system(self):
        # Setup real storage connections
        neo4j = Neo4jStorage(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test"
        )
        qdrant = QdrantStorage(
            host="localhost",
            port=6333,
            collection_name="test_cross_system"
        )
        
        await neo4j.connect()
        await qdrant.ensure_collection()
        
        entity_manager = EntityReferenceManager(neo4j, qdrant)
        query_engine = CrossSystemQueryEngine(neo4j, qdrant, entity_manager)
        
        yield {
            'neo4j': neo4j,
            'qdrant': qdrant,
            'entity_manager': entity_manager,
            'query_engine': query_engine
        }
        
        # Cleanup
        await neo4j.disconnect()
    
    async def test_end_to_end_cross_system_linking(self, integrated_system):
        components = integrated_system
        entity_manager = components['entity_manager']
        query_engine = components['query_engine']
        
        # Create test entities
        entity1 = Entity(
            name="Alice Johnson",
            type="PERSON",
            source_doc_id="doc_cross_test_123"
        )
        
        entity2 = Entity(
            name="TechCorp Inc",
            type="ORGANIZATION",
            source_doc_id="doc_cross_test_123"
        )
        
        chunk_ids = ["doc_cross_test_123:chunk:0000"]
        
        # Create entity references
        ref1 = await entity_manager.create_entity_reference(entity1, chunk_ids)
        ref2 = await entity_manager.create_entity_reference(entity2, chunk_ids)
        
        # Verify references created
        assert ref1.sync_status == "synced"
        assert ref2.sync_status == "synced"
        
        # Test entity linkage discovery
        linkages = await entity_manager.find_entity_linkages(entity1.id)
        
        # Should find co-occurrence linkage
        co_occur_linkages = [l for l in linkages if l.target_entity_id == entity2.id]
        assert len(co_occur_linkages) > 0
        
        # Test cross-system query
        query_vector = [0.1, 0.2, 0.3, 0.4] * 96  # 384-dim vector
        
        hybrid_result = await query_engine.execute_hybrid_entity_query(
            query_vector=query_vector,
            entity_context=[entity1.id],
            expand_entities=True,
            final_top_k=5
        )
        
        # Verify hybrid query results
        assert hybrid_result['total_results'] >= 0
        assert entity1.id in hybrid_result['entity_context_original']
        
        # Test entity similarity
        similarity_result = await query_engine.execute_entity_similarity_query(
            entity_id=entity1.id,
            similarity_threshold=0.5
        )
        
        assert similarity_result['source_entity_id'] == entity1.id
        
        # Cleanup
        await components['qdrant'].delete_vectors_by_document_id("doc_cross_test_123")
        await components['neo4j'].execute_query(
            "MATCH (n) WHERE n.id CONTAINS 'doc_cross_test_123' DETACH DELETE n"
        )
```

## Performance Considerations

- **Reference Caching**: Cache entity references to reduce database queries
- **Batch Operations**: Process entity linkages in batches
- **Async Processing**: Use asyncio for parallel cross-system operations
- **Index Optimization**: Create appropriate indexes in both systems
- **Connection Pooling**: Use connection pools for database access

## Success Criteria

- [ ] Entity references successfully created across both systems
- [ ] Bidirectional entity linking works correctly
- [ ] Cross-system entity queries execute efficiently
- [ ] Entity consistency validation passes
- [ ] Performance benchmarks meet requirements
- [ ] Integration tests pass with real databases
- [ ] Entity linkage discovery functions correctly
- [ ] Hybrid query fusion produces relevant results

## Next Steps

After completing this task:
1. Proceed to Task 2.2: Vector Embedding Integration
2. Implement sparse vector support
3. Create advanced query optimization
4. Plan relationship extraction enhancements

---

**Estimated Time**: 4-5 days  
**Dependencies**: Tasks 1.1, 1.2, 1.3  
**Risk Level**: High (complex cross-system coordination required)