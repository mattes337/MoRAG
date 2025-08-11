"""Query optimization utilities for maintenance jobs.

Provides optimized Neo4j queries and query execution patterns for better performance
in maintenance operations.

IMPORTANT: All queries use relationship-agnostic patterns (e.g., [r] instead of [r:ABOUT])
because MoRAG uses LLM-generated dynamic relationship types that vary by domain
(TREATS, CAUSES, SOLVES, INFLUENCES, etc.). Hardcoding specific relationship types
would miss most actual relationships in the graph.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class QueryStats:
    """Statistics for query execution."""
    query_hash: str
    execution_time: float
    result_count: int
    parameters: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class QueryOptimizer:
    """Optimized queries for maintenance operations."""
    
    # Optimized query templates with proper indexing hints
    QUERIES = {
        'find_entities_by_fact_count': """
            MATCH (e:Entity)
            OPTIONAL MATCH (f:Fact)-[r]->(e)
            WITH e, count(DISTINCT f) AS fact_count
            WHERE fact_count >= $min_facts
            RETURN e.id AS k_id, e.name AS k_name, fact_count
            ORDER BY fact_count DESC
            LIMIT $limit
        """,

        'find_cooccurring_entities': """
            MATCH (e1:Entity {id: $entity_id})
            MATCH (f:Fact)-[r1]->(e1)
            MATCH (f)-[r2]->(e2:Entity)
            WHERE e2 <> e1
            WITH e2, count(DISTINCT f) AS cofacts, $total_facts AS total
            WITH e2, cofacts, toFloat(cofacts) / total AS share
            WHERE share >= $min_share
            RETURN e2.id AS entity_id, e2.name AS entity_name, cofacts, share
            ORDER BY cofacts DESC
            LIMIT $limit
        """,
        
        'find_duplicate_relationships': """
            MATCH (a:Entity)-[r1]->(b:Entity)
            MATCH (a)-[r2]->(b)
            WHERE id(r1) < id(r2) AND type(r1) = type(r2)
            RETURN id(r1) AS primary_id, id(r2) AS duplicate_id, 
                   type(r1) AS rel_type, a.name AS source, b.name AS target
            LIMIT $limit
        """,
        
        'find_orphaned_relationships': """
            MATCH ()-[r]->()
            WHERE NOT EXISTS {
                MATCH (start)-[r]->(end)
                WHERE start IS NOT NULL AND end IS NOT NULL
            }
            RETURN id(r) AS rel_id, type(r) AS rel_type
            LIMIT $limit
        """,
        
        'find_self_referential_relationships': """
            MATCH (e:Entity)-[r]->(e)
            RETURN id(r) AS rel_id, type(r) AS rel_type, e.name AS entity_name
            LIMIT $limit
        """,
        
        'get_relationship_type_summary': """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
        """,

        'get_fact_relationship_types': """
            MATCH (f:Fact)-[r]->(e:Entity)
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
        """,


        
        'find_entities_by_similarity': """
            MATCH (e1:Entity), (e2:Entity)
            WHERE e1.name <> e2.name 
            AND e1.type = e2.type
            AND id(e1) < id(e2)
            WITH e1, e2, 
                 apoc.text.levenshteinSimilarity(e1.name, e2.name) AS similarity
            WHERE similarity >= $similarity_threshold
            RETURN e1.id AS entity1_id, e1.name AS entity1_name,
                   e2.id AS entity2_id, e2.name AS entity2_name,
                   similarity
            ORDER BY similarity DESC
            LIMIT $limit
        """,
        
        'batch_merge_entities': """
            UNWIND $merge_pairs AS pair
            MATCH (primary:Entity {id: pair.primary_id})
            MATCH (duplicate:Entity {id: pair.duplicate_id})
            
            // Move all relationships from duplicate to primary
            OPTIONAL MATCH (duplicate)-[r1]->(target:Entity)
            WHERE target <> primary
            MERGE (primary)-[new_r1:${rel_type}]->(target)
            ON CREATE SET new_r1 = properties(r1)
            DELETE r1
            
            OPTIONAL MATCH (source:Entity)-[r2]->(duplicate)
            WHERE source <> primary
            MERGE (source)-[new_r2:${rel_type}]->(primary)
            ON CREATE SET new_r2 = properties(r2)
            DELETE r2
            
            // Move all facts from duplicate to primary
            OPTIONAL MATCH (f:Fact)-[fr]->(duplicate)
            MERGE (f)-[new_fr:${fact_rel_type}]->(primary)
            ON CREATE SET new_fr = properties(fr)
            DELETE fr
            
            // Delete the duplicate entity
            DELETE duplicate
            
            RETURN pair.primary_id AS merged_into, pair.duplicate_id AS removed
        """,
        
        'batch_delete_relationships': """
            UNWIND $relationship_ids AS rel_id
            MATCH ()-[r]->()
            WHERE id(r) = rel_id
            DELETE r
            RETURN rel_id AS deleted_id
        """,
        
        'update_relationship_types': """
            UNWIND $type_mappings AS mapping
            MATCH (a)-[r]->(b)
            WHERE type(r) = mapping.old_type
            CREATE (a)-[new_r:${new_type}]->(b)
            SET new_r = properties(r)
            DELETE r
            RETURN mapping.old_type AS old_type, mapping.new_type AS new_type, count(*) AS updated_count
        """
    }
    
    def __init__(self, storage):
        self.storage = storage
        self.query_stats: List[QueryStats] = []
    
    async def execute_optimized_query(
        self, 
        query_name: str, 
        params: Dict[str, Any],
        **template_params
    ) -> List[Dict[str, Any]]:
        """Execute an optimized query with performance monitoring."""
        if query_name not in self.QUERIES:
            raise ValueError(f"Unknown query: {query_name}")
        
        query_template = self.QUERIES[query_name]
        
        # Apply template parameters if provided
        if template_params:
            query = query_template.format(**template_params)
        else:
            query = query_template
        
        return await self._execute_with_stats(query, params, query_name)
    
    async def _execute_with_stats(
        self, 
        query: str, 
        params: Dict[str, Any],
        query_name: str = "custom"
    ) -> List[Dict[str, Any]]:
        """Execute query with performance statistics collection."""
        query_hash = f"{query_name}_{hash(query)}"
        start_time = time.time()
        
        try:
            result = await self.storage._connection_ops._execute_query(query, params)
            execution_time = time.time() - start_time
            
            stats = QueryStats(
                query_hash=query_hash,
                execution_time=execution_time,
                result_count=len(result),
                parameters=params,
                success=True
            )
            
            self.query_stats.append(stats)
            
            logger.info("Query executed successfully",
                       query_name=query_name,
                       execution_time=execution_time,
                       result_count=len(result))
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            stats = QueryStats(
                query_hash=query_hash,
                execution_time=execution_time,
                result_count=0,
                parameters=params,
                success=False,
                error=str(e)
            )
            
            self.query_stats.append(stats)
            
            logger.error("Query execution failed",
                        query_name=query_name,
                        execution_time=execution_time,
                        error=str(e),
                        query_preview=query[:200])
            raise
    
    async def find_entities_by_fact_count(
        self, 
        min_facts: int = 10, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Find entities with at least min_facts attached facts."""
        return await self.execute_optimized_query(
            'find_entities_by_fact_count',
            {'min_facts': min_facts, 'limit': limit}
        )
    
    async def find_cooccurring_entities(
        self, 
        entity_id: str, 
        total_facts: int,
        min_share: float = 0.18,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Find entities that co-occur with the given entity in facts."""
        return await self.execute_optimized_query(
            'find_cooccurring_entities',
            {
                'entity_id': entity_id,
                'total_facts': max(1, total_facts),
                'min_share': min_share,
                'limit': limit
            }
        )
    
    async def find_duplicate_relationships(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Find exact duplicate relationships."""
        return await self.execute_optimized_query(
            'find_duplicate_relationships',
            {'limit': limit}
        )
    
    async def find_orphaned_relationships(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Find relationships pointing to non-existent entities."""
        return await self.execute_optimized_query(
            'find_orphaned_relationships',
            {'limit': limit}
        )
    
    async def get_relationship_type_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all relationship types and their counts."""
        return await self.execute_optimized_query(
            'get_relationship_type_summary',
            {}
        )

    async def get_fact_relationship_types(self) -> List[Dict[str, Any]]:
        """Get summary of relationship types between Facts and Entities.

        This is useful for understanding what dynamic relationship types
        the LLM has generated in your specific domain.
        """
        return await self.execute_optimized_query(
            'get_fact_relationship_types',
            {}
        )
    
    async def batch_merge_entities(
        self, 
        merge_pairs: List[Dict[str, str]],
        rel_type: str = "MERGED_FROM",
        fact_rel_type: str = "ABOUT"
    ) -> List[Dict[str, Any]]:
        """Merge entities in batch for better performance."""
        return await self.execute_optimized_query(
            'batch_merge_entities',
            {'merge_pairs': merge_pairs},
            rel_type=rel_type,
            fact_rel_type=fact_rel_type
        )
    
    async def batch_delete_relationships(self, relationship_ids: List[int]) -> List[Dict[str, Any]]:
        """Delete relationships in batch for better performance."""
        return await self.execute_optimized_query(
            'batch_delete_relationships',
            {'relationship_ids': relationship_ids}
        )
    
    def get_query_performance_summary(self) -> Dict[str, Any]:
        """Get summary of query performance statistics."""
        if not self.query_stats:
            return {"total_queries": 0}
        
        successful_queries = [s for s in self.query_stats if s.success]
        failed_queries = [s for s in self.query_stats if not s.success]
        
        avg_execution_time = sum(s.execution_time for s in successful_queries) / len(successful_queries) if successful_queries else 0
        total_results = sum(s.result_count for s in successful_queries)
        
        return {
            "total_queries": len(self.query_stats),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "average_execution_time": avg_execution_time,
            "total_results_returned": total_results,
            "success_rate": len(successful_queries) / len(self.query_stats) if self.query_stats else 0
        }
