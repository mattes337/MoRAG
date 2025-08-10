"""Keyword Deduplication Maintenance Job

Performs intelligent deduplication of similar keywords/entities using LLM-based viability analysis.
Combines semantically related entities while preserving meaningful distinctions.

Examples of merges:
- "Omega 3" + "Omega-3" + "Omega-3 Wert" + "Omega-3 Blutwert" → "Omega-3"
- Plurals vs singulars, formatting variations, semantic extensions

Defaults to dry-run for safety. Uses LLM to evaluate merge viability based on:
- Semantic similarity of entities
- Fact count distribution
- Retrieval impact assessment
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import structlog

from morag_graph.storage import Neo4jStorage, Neo4jConfig
from morag_graph.extraction.systematic_deduplicator import (
    EntitySimilarityCalculator,
    SystematicDeduplicator,
    MergeCandidate
)
from morag_graph.models.entity import Entity

logger = structlog.get_logger(__name__)


@dataclass
class KeywordDeduplicationConfig:
    similarity_threshold: float = 0.75  # Higher threshold for keyword deduplication
    max_cluster_size: int = 8           # Max entities per merge cluster
    min_fact_threshold: int = 1         # Min facts to consider for deduplication (removed restrictive filter)
    preserve_high_confidence: float = 0.95  # Don't merge high-confidence entities
    semantic_similarity_weight: float = 0.6  # Weight for embedding vs string similarity
    batch_size: int = 50                # Batch size for merge operations
    dry_run: bool = True                # Preview merges without applying
    job_tag: str = ""                   # Job tag for tracking
    limit_entities: int = 100           # Max entities to process per run
    enable_rotation: bool = True        # Enable rotation to prevent starvation
    process_all_if_small: bool = True   # Process all entities if total < limit_entities

    def ensure_defaults(self) -> None:
        if not self.job_tag:
            # Use date-based tag for more predictable rotation across deployments
            # This allows multiple runs per day while maintaining rotation consistency
            import datetime
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            self.job_tag = f"kw_dedup_{date_str}"


class KeywordDeduplicationService:
    """Intelligent keyword deduplication using LLM-based viability analysis."""

    def __init__(self, storage: Neo4jStorage, config: Optional[KeywordDeduplicationConfig] = None):
        self.storage = storage
        self.config = config or KeywordDeduplicationConfig()
        self.config.ensure_defaults()
        self._llm_client = None
        
        # Use existing deduplication infrastructure
        self.deduplicator = SystematicDeduplicator(
            similarity_threshold=self.config.similarity_threshold,
            merge_confidence_threshold=0.7,  # Lower threshold for keyword merging
            enable_llm_validation=True
        )

    async def _get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client
        try:
            from morag_reasoning.llm import LLMClient  # type: ignore
            self._llm_client = LLMClient(None)
            return self._llm_client
        except Exception as e:
            logger.warning("LLM client unavailable for keyword deduplication", error=str(e))
            return None

    async def run(self) -> Dict[str, Any]:
        """Run keyword deduplication job."""
        start_time = time.time()
        
        # Get candidate entities for deduplication
        entities = await self._get_deduplication_candidates()
        
        if not entities:
            return {
                "job_tag": self.config.job_tag,
                "processed": 0,
                "merges_performed": 0,
                "processing_time": time.time() - start_time,
                "message": "No entities found for deduplication"
            }

        # Find merge candidates using enhanced similarity
        merge_candidates = await self._find_keyword_merge_candidates(entities)
        
        # Validate merges with LLM viability analysis
        confirmed_merges = await self._validate_merges_with_viability_analysis(merge_candidates)
        
        # Apply merges if not dry run
        merges_applied = 0
        if not self.config.dry_run and confirmed_merges:
            merges_applied = await self._apply_keyword_merges(confirmed_merges)

        processing_time = time.time() - start_time
        
        return {
            "job_tag": self.config.job_tag,
            "processed": len(entities),
            "candidates_found": len(merge_candidates),
            "merges_confirmed": len(confirmed_merges),
            "merges_applied": merges_applied,
            "processing_time": processing_time,
            "dry_run": self.config.dry_run,
            "merge_details": [
                {
                    "primary": merge.primary_entity.name,
                    "duplicates": [e.name for e in merge.duplicate_entities],
                    "similarity": merge.similarity_score,
                    "confidence": merge.merge_confidence
                }
                for merge in confirmed_merges
            ]
        }

    async def _get_deduplication_candidates(self) -> List[Entity]:
        """Get entities that are candidates for deduplication using rotation to prevent starvation."""
        # Get total count of all entities and eligible entities
        total_count_query = """
        MATCH (e:Entity)
        RETURN count(e) AS total_count
        """

        eligible_count_query = """
        MATCH (e:Entity)
        WHERE e.confidence < $max_confidence
        RETURN count(e) AS eligible_count
        """

        count_params = {
            "max_confidence": self.config.preserve_high_confidence
        }

        total_result = await self.storage._connection_ops._execute_query(total_count_query, {})
        eligible_result = await self.storage._connection_ops._execute_query(eligible_count_query, count_params)

        total_entities_all = total_result[0]["total_count"] if total_result else 0
        total_entities = eligible_result[0]["eligible_count"] if eligible_result else 0

        # If total entities is small or rotation is disabled, process all
        if (not self.config.enable_rotation or
            (self.config.process_all_if_small and total_entities <= self.config.limit_entities)):
            offset = 0
            limit = total_entities if self.config.process_all_if_small else self.config.limit_entities
            logger.info(f"Processing all {total_entities} eligible entities (no rotation needed, {total_entities_all} total entities)")
        else:
            # Calculate rotation offset based on job_tag for deterministic but varied selection
            import hashlib
            tag_hash = int(hashlib.md5(self.config.job_tag.encode()).hexdigest()[:8], 16)

            # Calculate number of possible batches based on ALL entities, not just eligible ones
            # This ensures we cycle through the entire entity space over multiple runs
            num_batches = max(1, (total_entities_all + self.config.limit_entities - 1) // self.config.limit_entities)
            batch_index = tag_hash % num_batches
            offset = batch_index * self.config.limit_entities
            limit = self.config.limit_entities

            logger.info(f"Using rotation: batch {batch_index + 1}/{num_batches}, offset {offset}, eligible entities: {total_entities}/{total_entities_all} total")

        # Get entities with optional rotation
        if self.config.enable_rotation and total_entities > self.config.limit_entities:
            # Use deterministic ordering by ID for consistent rotation
            query = """
            MATCH (e:Entity)
            WHERE e.confidence < $max_confidence
            OPTIONAL MATCH (f:Fact)-[r]->(e)
            WITH e, count(DISTINCT f) AS fact_count
            RETURN e, fact_count
            ORDER BY e.id
            SKIP $offset
            LIMIT $limit
            """
        else:
            # Use fact count ordering for better prioritization when not rotating
            # Entities with fewer facts are prioritized (more likely to be duplicates)
            query = """
            MATCH (e:Entity)
            WHERE e.confidence < $max_confidence
            OPTIONAL MATCH (f:Fact)-[r]->(e)
            WITH e, count(DISTINCT f) AS fact_count
            RETURN e, fact_count
            ORDER BY fact_count ASC, e.id
            LIMIT $limit
            """

        params = {
            "max_confidence": self.config.preserve_high_confidence,
            "offset": offset,
            "limit": limit
        }

        results = await self.storage._connection_ops._execute_query(query, params)

        entities = []
        for record in results:
            try:
                entity_data = record["e"]
                entity = Entity.from_neo4j_node(entity_data)
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Failed to parse entity from deduplication candidates: {e}")

        # Mark entities as processed to track rotation (only in non-dry-run mode)
        if entities and not self.config.dry_run and self.config.enable_rotation:
            entity_ids = [e.id for e in entities]
            mark_query = """
            MATCH (e:Entity)
            WHERE e.id IN $entity_ids
            SET e.last_dedup_check = datetime(),
                e.last_dedup_job_tag = $job_tag
            """
            await self.storage._connection_ops._execute_query(
                mark_query,
                {"entity_ids": entity_ids, "job_tag": self.config.job_tag}
            )

        logger.info(f"Selected {len(entities)} entities for deduplication analysis")
        return entities

    async def _find_keyword_merge_candidates(self, entities: List[Entity]) -> List[MergeCandidate]:
        """Find merge candidates using enhanced keyword similarity."""
        candidates = []
        processed = set()
        
        for i, entity1 in enumerate(entities):
            if entity1.id in processed:
                continue
                
            similar_entities = []
            
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity2.id in processed or entity2.type != entity1.type:
                    continue
                
                # Calculate enhanced similarity for keywords
                similarity = await self._calculate_keyword_similarity(entity1, entity2)
                
                if similarity >= self.config.similarity_threshold:
                    similar_entities.append(entity2)
                    processed.add(entity2.id)
            
            if similar_entities and len(similar_entities) < self.config.max_cluster_size:
                # Get fact counts for viability analysis
                fact_counts = await self._get_fact_counts([entity1] + similar_entities)

                # Calculate average similarity score
                similarity_scores = []
                for e in similar_entities:
                    score = await self._calculate_keyword_similarity(entity1, e)
                    similarity_scores.append(score)
                avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

                candidates.append(MergeCandidate(
                    primary_entity=entity1,
                    duplicate_entities=similar_entities,
                    similarity_score=avg_similarity,
                    merge_confidence=0.0,  # Will be set by LLM validation
                    merge_reason=f"keyword_similarity_cluster_{len(similar_entities)}_entities"
                ))
                
                processed.add(entity1.id)
        
        logger.info(f"Found {len(candidates)} keyword merge candidates")
        return candidates

    async def _calculate_keyword_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate enhanced similarity for keyword entities."""
        # Use existing similarity calculator as base
        base_similarity = self.deduplicator.similarity_calculator.calculate_similarity(entity1, entity2)
        
        # Add keyword-specific enhancements
        keyword_bonus = self._calculate_keyword_specific_similarity(entity1.name, entity2.name)
        
        # Combine with weights
        enhanced_similarity = (
            base_similarity * (1 - self.config.semantic_similarity_weight) +
            keyword_bonus * self.config.semantic_similarity_weight
        )
        
        return min(1.0, enhanced_similarity)

    def _calculate_keyword_specific_similarity(self, name1: str, name2: str) -> float:
        """Calculate keyword-specific similarity patterns."""
        name1_clean = self._normalize_keyword_name(name1)
        name2_clean = self._normalize_keyword_name(name2)

        # Exact match after normalization
        if name1_clean == name2_clean:
            return 1.0

        # Check for common keyword patterns
        patterns = [
            # Plural/singular variations
            (name1_clean.rstrip('s'), name2_clean.rstrip('s')),
            # Hyphen/space variations
            (name1_clean.replace('-', ' '), name2_clean.replace('-', ' ')),
            (name1_clean.replace(' ', '-'), name2_clean.replace(' ', '-')),
            # Number/word variations
            (name1_clean.replace('3', 'drei'), name2_clean.replace('3', 'drei')),
        ]

        for pattern1, pattern2 in patterns:
            if pattern1 == pattern2:
                return 0.9

        # Check if one is a subset/extension of the other
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return 0.8

        # Word overlap with keyword-specific weights
        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())

        if words1 and words2:
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            jaccard = len(intersection) / len(union)

            # Boost score if there's significant word overlap
            if len(intersection) > 0:
                return max(jaccard, 0.6)  # Minimum score for any word overlap
            return jaccard

        return 0.0

    def _normalize_keyword_name(self, name: str) -> str:
        """Normalize keyword name for comparison."""
        return name.lower().strip().replace('_', ' ')

    async def _get_fact_counts(self, entities: List[Entity]) -> Dict[str, int]:
        """Get fact counts for entities."""
        entity_ids = [e.id for e in entities]
        
        query = """
        UNWIND $entity_ids AS entity_id
        MATCH (e:Entity {id: entity_id})
        OPTIONAL MATCH (f:Fact)-[r]->(e)
        RETURN e.id AS entity_id, count(DISTINCT f) AS fact_count
        """
        
        results = await self.storage._connection_ops._execute_query(
            query, {"entity_ids": entity_ids}
        )
        
        return {r["entity_id"]: r["fact_count"] for r in results}

    async def _validate_merges_with_viability_analysis(
        self, 
        candidates: List[MergeCandidate]
    ) -> List[MergeCandidate]:
        """Validate merges using LLM viability analysis."""
        client = await self._get_llm_client()
        if not client:
            logger.warning("No LLM available for viability analysis, using rule-based validation")
            return self._rule_based_validation(candidates)
        
        confirmed_merges = []
        
        for candidate in candidates:
            try:
                should_merge, confidence, reasoning = await self._analyze_merge_viability(
                    client, candidate
                )
                
                if should_merge:
                    candidate.merge_confidence = confidence
                    candidate.merge_reason = "llm_approved"
                    confirmed_merges.append(candidate)
                    
            except Exception as e:
                logger.warning(f"Viability analysis failed for candidate: {e}")
        
        logger.info(f"Confirmed {len(confirmed_merges)} merges via viability analysis")
        return confirmed_merges

    async def _analyze_merge_viability(
        self, 
        client, 
        candidate: MergeCandidate
    ) -> Tuple[bool, float, str]:
        """Analyze merge viability using LLM."""
        # Get fact counts for analysis
        all_entities = [candidate.primary_entity] + candidate.duplicate_entities
        fact_counts = await self._get_fact_counts(all_entities)
        
        # Create viability analysis prompt
        entity_info = []
        for entity in all_entities:
            fact_count = fact_counts.get(entity.id, 0)
            entity_info.append(f"- {entity.name} (Type: {entity.type}, Facts: {fact_count})")
        
        prompt = f"""
        Should these entities be merged? Entities:
        {chr(10).join(entity_info)}
        Similarity: {candidate.similarity_score:.2f}

        Merge if: same core concept, formatting variations, or semantic extensions.
        Don't merge if: different concepts despite similar names.

        JSON response:
        {{
            "should_merge": true/false,
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            response = await client.generate(prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return (
                    data.get("should_merge", False),
                    float(data.get("confidence", 0.0)),
                    "llm_approved"
                )
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM viability response: {e}")
        
        # Fallback to rule-based decision
        return False, 0.0, "llm_failed"

    def _rule_based_validation(self, candidates: List[MergeCandidate]) -> List[MergeCandidate]:
        """Rule-based validation fallback."""
        confirmed = []
        
        for candidate in candidates:
            # Simple rule: high similarity + same type = merge
            if (candidate.similarity_score >= 0.85 and 
                all(e.type == candidate.primary_entity.type for e in candidate.duplicate_entities)):
                candidate.merge_confidence = candidate.similarity_score
                candidate.merge_reason = "rule_based_high_similarity"
                confirmed.append(candidate)
        
        return confirmed

    async def _apply_keyword_merges(self, confirmed_merges: List[MergeCandidate]) -> int:
        """Apply confirmed keyword merges to the graph."""
        merges_applied = 0
        
        for merge in confirmed_merges:
            try:
                # Merge entities in the graph
                await self._merge_entities_in_graph(merge)
                merges_applied += 1
                
            except Exception as e:
                logger.error(f"Failed to apply merge: {e}", merge=merge)
        
        return merges_applied

    async def _merge_entities_in_graph(self, merge: MergeCandidate) -> None:
        """Merge entities in the Neo4j graph."""
        primary_id = merge.primary_entity.id
        duplicate_ids = [e.id for e in merge.duplicate_entities]

        # Update all relationships to point to primary entity
        for dup_id in duplicate_ids:
            try:
                # Try APOC first with corrected syntax
                await self._merge_entities_apoc(dup_id, primary_id)
            except Exception as e:
                # Fallback to manual relationship type handling if APOC is not available
                logger.warning(f"APOC not available, using fallback merge method: {e}")
                await self._merge_entities_fallback(dup_id, primary_id)

        # Delete duplicate entities (only after all relationships are moved)
        delete_query = """
        MATCH (e:Entity)
        WHERE e.id IN $duplicate_ids
        DELETE e
        """

        await self.storage._connection_ops._execute_query(
            delete_query, {"duplicate_ids": duplicate_ids}
        )

        logger.info(f"Merged {len(duplicate_ids)} entities into {primary_id}")

    async def _merge_entities_apoc(self, dup_id: str, primary_id: str) -> None:
        """Merge entities using APOC with corrected syntax."""
        logger.info(f"Starting APOC merge for entity {dup_id} -> {primary_id}")

        # Move incoming relationships (Facts to Entities)
        query1 = """
        MATCH (source)-[r]->(dup:Entity {id: $dup_id})
        MATCH (primary:Entity {id: $primary_id})
        WITH source, r, dup, primary, type(r) as rel_type, properties(r) as props
        WITH source, r, primary, rel_type, props + {merged_from: $dup_id, job_tag: $job_tag} as new_props
        CALL apoc.create.relationship(source, rel_type, new_props, primary) YIELD rel
        DELETE r
        RETURN count(r) as moved_count
        """

        # Move outgoing relationships (Entities to other nodes)
        query2 = """
        MATCH (dup:Entity {id: $dup_id})-[r]->(target)
        MATCH (primary:Entity {id: $primary_id})
        WITH dup, r, target, primary, type(r) as rel_type, properties(r) as props
        WITH dup, r, target, primary, rel_type, props + {merged_from: $dup_id, job_tag: $job_tag} as new_props
        CALL apoc.create.relationship(primary, rel_type, new_props, target) YIELD rel
        DELETE r
        RETURN count(r) as moved_count
        """

        # Execute relationship updates using APOC
        try:
            result1 = await self.storage._connection_ops._execute_query(
                query1, {"dup_id": dup_id, "primary_id": primary_id, "job_tag": self.config.job_tag}
            )
            moved_incoming = result1[0]["moved_count"] if result1 else 0
            logger.debug(f"APOC moved {moved_incoming} incoming relationships")

            result2 = await self.storage._connection_ops._execute_query(
                query2, {"dup_id": dup_id, "primary_id": primary_id, "job_tag": self.config.job_tag}
            )
            moved_outgoing = result2[0]["moved_count"] if result2 else 0
            logger.debug(f"APOC moved {moved_outgoing} outgoing relationships")

            logger.info(f"APOC merge completed: {moved_incoming} incoming + {moved_outgoing} outgoing relationships moved")

        except Exception as e:
            logger.error(f"APOC merge failed for {dup_id}: {e}")
            raise

    async def _merge_entities_fallback(self, dup_id: str, primary_id: str) -> None:
        """Fallback merge method when APOC is not available."""
        logger.info(f"Starting fallback merge for entity {dup_id} -> {primary_id}")

        # First, get detailed information about all relationships
        get_all_rels_query = """
        MATCH (dup:Entity {id: $dup_id})
        OPTIONAL MATCH (source)-[r1]->(dup)
        OPTIONAL MATCH (dup)-[r2]->(target)
        RETURN
            collect(DISTINCT {type: type(r1), direction: 'incoming', source_labels: labels(source)}) as incoming_rels,
            collect(DISTINCT {type: type(r2), direction: 'outgoing', target_labels: labels(target)}) as outgoing_rels
        """

        result = await self.storage._connection_ops._execute_query(
            get_all_rels_query, {"dup_id": dup_id}
        )

        if not result:
            logger.warning(f"No relationship information found for entity {dup_id}")
            return

        incoming_rels = [r for r in result[0]["incoming_rels"] if r["type"] is not None]
        outgoing_rels = [r for r in result[0]["outgoing_rels"] if r["type"] is not None]

        logger.info(f"Found {len(incoming_rels)} incoming and {len(outgoing_rels)} outgoing relationship types for {dup_id}")

        # Handle incoming relationships with more robust approach
        for rel_info in incoming_rels:
            rel_type = rel_info["type"]
            logger.debug(f"Moving incoming relationships of type {rel_type}")

            # Use a more comprehensive approach that handles all source node types
            move_incoming_query = f"""
            MATCH (source)-[r:{rel_type}]->(dup:Entity {{id: $dup_id}})
            MATCH (primary:Entity {{id: $primary_id}})
            WITH source, r, primary, properties(r) as props
            CREATE (source)-[new_r:{rel_type}]->(primary)
            SET new_r = props
            SET new_r.merged_from = $dup_id
            SET new_r.job_tag = $job_tag
            DELETE r
            RETURN count(r) as moved_count
            """

            move_result = await self.storage._connection_ops._execute_query(
                move_incoming_query,
                {"dup_id": dup_id, "primary_id": primary_id, "job_tag": self.config.job_tag}
            )
            moved_count = move_result[0]["moved_count"] if move_result else 0
            logger.debug(f"Moved {moved_count} incoming relationships of type {rel_type}")

        # Handle outgoing relationships with more robust approach
        for rel_info in outgoing_rels:
            rel_type = rel_info["type"]
            logger.debug(f"Moving outgoing relationships of type {rel_type}")

            move_outgoing_query = f"""
            MATCH (dup:Entity {{id: $dup_id}})-[r:{rel_type}]->(target)
            MATCH (primary:Entity {{id: $primary_id}})
            WITH dup, r, target, primary, properties(r) as props
            CREATE (primary)-[new_r:{rel_type}]->(target)
            SET new_r = props
            SET new_r.merged_from = $dup_id
            SET new_r.job_tag = $job_tag
            DELETE r
            RETURN count(r) as moved_count
            """

            move_result = await self.storage._connection_ops._execute_query(
                move_outgoing_query,
                {"dup_id": dup_id, "primary_id": primary_id, "job_tag": self.config.job_tag}
            )
            moved_count = move_result[0]["moved_count"] if move_result else 0
            logger.debug(f"Moved {moved_count} outgoing relationships of type {rel_type}")

        # Verify all relationships have been moved before proceeding
        verify_query = """
        MATCH (dup:Entity {id: $dup_id})
        OPTIONAL MATCH (dup)-[r]-()
        RETURN count(r) as remaining_relationships, collect(type(r)) as remaining_types
        """

        verify_result = await self.storage._connection_ops._execute_query(
            verify_query, {"dup_id": dup_id}
        )

        remaining_rels = verify_result[0]["remaining_relationships"] if verify_result else 0
        remaining_types = verify_result[0]["remaining_types"] if verify_result else []

        if remaining_rels > 0:
            logger.warning(f"Entity {dup_id} still has {remaining_rels} relationships after merge attempt",
                         remaining_types=remaining_types)

            # Try a more aggressive approach - delete relationships one by one
            for rel_type in set(remaining_types):
                if rel_type:  # Skip None types
                    force_delete_type_query = f"""
                    MATCH (dup:Entity {{id: $dup_id}})-[r:{rel_type}]-()
                    DELETE r
                    RETURN count(r) as deleted_count
                    """
                    delete_result = await self.storage._connection_ops._execute_query(
                        force_delete_type_query, {"dup_id": dup_id}
                    )
                    deleted_count = delete_result[0]["deleted_count"] if delete_result else 0
                    logger.info(f"Force deleted {deleted_count} relationships of type {rel_type}")

            # Final cleanup - delete any remaining relationships
            final_cleanup_query = """
            MATCH (dup:Entity {id: $dup_id})
            OPTIONAL MATCH (dup)-[r]-()
            DELETE r
            RETURN count(r) as final_deleted_count
            """
            final_result = await self.storage._connection_ops._execute_query(
                final_cleanup_query, {"dup_id": dup_id}
            )
            final_deleted = final_result[0]["final_deleted_count"] if final_result else 0
            if final_deleted > 0:
                logger.info(f"Final cleanup deleted {final_deleted} remaining relationships")
        else:
            logger.info(f"Successfully moved all relationships for entity {dup_id}")


async def run_keyword_deduplication(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Entrypoint to run the keyword deduplication job."""
    import os
    neo_config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
        trust_all_certificates=os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "false").lower() == "true",
    )
    storage = Neo4jStorage(neo_config)
    await storage.connect()
    try:
        cfg = KeywordDeduplicationConfig()
        if config_overrides:
            for k, v in config_overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        cfg.ensure_defaults()
        svc = KeywordDeduplicationService(storage, cfg)
        return await svc.run()
    finally:
        await storage.disconnect()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Keyword Deduplication maintenance job")
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--max-cluster-size", type=int, default=8)
    parser.add_argument("--min-facts", type=int, default=3)
    parser.add_argument("--preserve-confidence", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit-entities", type=int, default=100)
    parser.add_argument("--apply", action="store_true", help="Apply changes (disable dry-run)")
    parser.add_argument("--job-tag", type=str, default="")
    parser.add_argument("--no-rotation", action="store_true", help="Disable rotation (always process top entities)")
    parser.add_argument("--no-process-all-small", action="store_true", help="Don't process all entities when count is small")
    args = parser.parse_args()

    overrides = {
        "similarity_threshold": args.similarity_threshold,
        "max_cluster_size": args.max_cluster_size,
        "min_fact_threshold": args.min_facts,
        "preserve_high_confidence": args.preserve_confidence,
        "batch_size": args.batch_size,
        "limit_entities": args.limit_entities,
        "dry_run": not args.apply,
        "job_tag": args.job_tag,
        "enable_rotation": not args.no_rotation,
        "process_all_if_small": not args.no_process_all_small,
    }

    result = asyncio.run(run_keyword_deduplication(overrides))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
