"""Relationship Merger Maintenance Job

Performs intelligent merging of redundant relationships to reduce graph overhead.
Identifies and merges:
1. Duplicate relationships (same source, target, type)
2. Semantically equivalent relationships (different types, same meaning)
3. Bidirectional redundancy (A→B and B→A consolidation)
4. Transitive redundancy (A→B→C where A→C exists directly)

Applies changes by default (dry-run available as opt-in). Uses LLM to evaluate merge viability based on:
- Semantic similarity of relationship types
- Confidence scores
- Graph connectivity impact
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime

import structlog

from morag_graph.storage import Neo4jStorage, Neo4jConfig
from morag_reasoning.llm import LLMClient
from .base import MaintenanceJobBase, validate_positive_int, validate_float_range
from .query_optimizer import QueryOptimizer

logger = structlog.get_logger(__name__)


@dataclass
class RelationshipMergerConfig:
    similarity_threshold: float = 0.8       # Threshold for semantic similarity
    batch_size: int = 100                   # Batch size for merge operations
    dry_run: bool = False                   # Apply merges by default (set to True for preview)
    job_tag: str = ""                       # Job tag for tracking
    limit_relations: int = 1000             # Max relationships to process per run
    enable_rotation: bool = True            # Enable rotation to prevent starvation
    merge_bidirectional: bool = True        # Merge bidirectional relationships
    merge_transitive: bool = False          # Merge transitive relationships (conservative)
    min_confidence: float = 0.5             # Minimum confidence for relationships to consider
    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

    def ensure_defaults(self) -> None:
        """Ensure all configuration values are within valid ranges."""
        self.similarity_threshold = max(0.0, min(1.0, self.similarity_threshold))
        self.batch_size = max(1, self.batch_size)
        self.limit_relations = max(1, self.limit_relations)
        self.min_confidence = max(0.0, min(1.0, self.min_confidence))
        if not self.job_tag:
            self.job_tag = f"rel_merger_{int(time.time())}"


@dataclass
class RelationshipCandidate:
    """Represents a relationship that could be merged."""
    id: str
    source_id: str
    target_id: str
    type: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class MergeCandidate:
    """Represents a group of relationships that should be merged."""
    primary_relationship: RelationshipCandidate
    duplicate_relationships: List[RelationshipCandidate]
    merge_reason: str
    confidence_score: float


@dataclass
class RelationshipMergerResult:
    """Result of relationship merger operation."""
    total_relationships: int
    processed_relationships: int
    duplicate_merges: int
    semantic_merges: int
    bidirectional_merges: int
    transitive_merges: int
    total_merges: int
    processing_time: float
    dry_run: bool
    job_tag: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "total_relationships": self.total_relationships,
            "processed_relationships": self.processed_relationships,
            "duplicate_merges": self.duplicate_merges,
            "semantic_merges": self.semantic_merges,
            "bidirectional_merges": self.bidirectional_merges,
            "transitive_merges": self.transitive_merges,
            "total_merges": self.total_merges,
            "processing_time": self.processing_time,
            "dry_run": self.dry_run,
            "job_tag": self.job_tag
        }


class RelationshipMerger(MaintenanceJobBase):
    """Handles relationship merging operations."""

    def __init__(self, config: RelationshipMergerConfig, neo4j_storage: Neo4jStorage, llm_client: LLMClient):
        self.merger_config = config
        self.neo4j_storage = neo4j_storage
        self.llm_client = llm_client
        self.logger = logger.bind(component="relationship_merger")

        # Initialize base class with config dict
        config_dict = {
            'job_tag': self.merger_config.job_tag,
            'dry_run': self.merger_config.dry_run,
            'batch_size': self.merger_config.batch_size,
            'circuit_breaker_threshold': self.merger_config.circuit_breaker_threshold,
            'circuit_breaker_timeout': self.merger_config.circuit_breaker_timeout,
        }
        super().__init__(config_dict)

        self.query_optimizer = QueryOptimizer(neo4j_storage)

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate integer parameters
        errors.extend(validate_positive_int(self.merger_config.batch_size, "batch_size", min_value=1, max_value=10000))
        errors.extend(validate_positive_int(self.merger_config.limit_relations, "limit_relations", min_value=1, max_value=100000))

        # Validate float parameters
        errors.extend(validate_float_range(self.merger_config.similarity_threshold, "similarity_threshold", min_value=0.0, max_value=1.0))
        errors.extend(validate_float_range(self.merger_config.min_confidence, "min_confidence", min_value=0.0, max_value=1.0))

        return errors

    async def run(self) -> Dict[str, Any]:
        """Run the relationship merger process."""
        result = await self.run_merger()
        return {
            "total_relationships": result.total_relationships,
            "processed_relationships": result.processed_relationships,
            "duplicate_merges": result.duplicate_merges,
            "semantic_merges": result.semantic_merges,
            "bidirectional_merges": result.bidirectional_merges,
            "transitive_merges": result.transitive_merges,
            "total_merges": result.total_merges,
            "processing_time": result.processing_time,
            "dry_run": result.dry_run,
            "job_tag": result.job_tag
        }

    async def run_merger(self) -> RelationshipMergerResult:
        """Run the relationship merger process."""
        self.log_job_start()

        # Validate configuration
        config_errors = self.validate_config()
        if config_errors:
            raise ValueError(f"Configuration errors: {'; '.join(config_errors)}")

        start_time = time.time()
        self.merger_config.ensure_defaults()

        self.logger.info("Starting relationship merger", config=self.config)

        # Get relationships to process
        relationships = await self._get_merger_candidates()
        total_relationships = len(relationships)

        if not relationships:
            self.logger.info("No relationships found for merging")
            return RelationshipMergerResult(
                total_relationships=0,
                processed_relationships=0,
                duplicate_merges=0,
                semantic_merges=0,
                bidirectional_merges=0,
                transitive_merges=0,
                total_merges=0,
                processing_time=time.time() - start_time,
                dry_run=self.merger_config.dry_run,
                job_tag=self.merger_config.job_tag
            )

        self.logger.info(f"Found {total_relationships} relationships for potential merging")

        # Find merge candidates
        merge_candidates = await self._find_merge_candidates(relationships)

        # Apply merges
        duplicate_merges = 0
        semantic_merges = 0
        bidirectional_merges = 0
        transitive_merges = 0

        for merge_candidate in merge_candidates:
            if not self.merger_config.dry_run:
                await self._apply_merge(merge_candidate)
            else:
                # In dry run mode, just log what would be merged
                self.logger.info(
                    "Would merge relationships (dry run)",
                    primary_id=merge_candidate.primary_relationship.id,
                    duplicate_ids=[d.id for d in merge_candidate.duplicate_relationships],
                    reason=merge_candidate.merge_reason
                )

            # Count merge types
            if "duplicate" in merge_candidate.merge_reason.lower():
                duplicate_merges += 1
            elif "semantic" in merge_candidate.merge_reason.lower():
                semantic_merges += 1
            elif "bidirectional" in merge_candidate.merge_reason.lower():
                bidirectional_merges += 1
            elif "transitive" in merge_candidate.merge_reason.lower():
                transitive_merges += 1

        total_merges = len(merge_candidates)
        processing_time = time.time() - start_time

        result = RelationshipMergerResult(
            total_relationships=total_relationships,
            processed_relationships=len(relationships),
            duplicate_merges=duplicate_merges,
            semantic_merges=semantic_merges,
            bidirectional_merges=bidirectional_merges,
            transitive_merges=transitive_merges,
            total_merges=total_merges,
            processing_time=processing_time,
            dry_run=self.merger_config.dry_run,
            job_tag=self.merger_config.job_tag
        )

        self.logger.info("Relationship merger completed", result=result)
        return result

    async def _get_merger_candidates(self) -> List[RelationshipCandidate]:
        """Get relationships that are candidates for merging."""
        # Calculate offset for rotation if enabled
        offset = 0
        if self.merger_config.enable_rotation and self.merger_config.job_tag:
            # Use job_tag hash for deterministic offset
            offset = hash(self.merger_config.job_tag) % 1000

        query = """
        MATCH ()-[r]->()
        WHERE r.confidence >= $min_confidence
        RETURN r.id as id, startNode(r).id as source_id, endNode(r).id as target_id,
               type(r) as type, r.confidence as confidence, r.metadata as metadata
        ORDER BY r.id
        SKIP $offset
        LIMIT $limit
        """

        result = await self.neo4j_storage._connection_ops._execute_query(query, {
            "min_confidence": self.merger_config.min_confidence,
            "offset": offset,
            "limit": self.merger_config.limit_relations
        })

        relationships = []
        for record in result:
            metadata = {}
            if record.get("metadata"):
                try:
                    metadata = json.loads(record["metadata"]) if isinstance(record["metadata"], str) else record["metadata"]
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            relationships.append(RelationshipCandidate(
                id=record["id"],
                source_id=record["source_id"],
                target_id=record["target_id"],
                type=record["type"],
                confidence=record["confidence"],
                metadata=metadata
            ))

        return relationships

    async def _find_merge_candidates(self, relationships: List[RelationshipCandidate]) -> List[MergeCandidate]:
        """Find relationships that should be merged."""
        merge_candidates = []

        # 1. Find exact duplicates
        merge_candidates.extend(await self._find_duplicate_relationships(relationships))

        # 2. Find semantically equivalent relationships
        merge_candidates.extend(await self._find_semantic_equivalents(relationships))

        # 3. Find bidirectional redundancy
        if self.merger_config.merge_bidirectional:
            merge_candidates.extend(await self._find_bidirectional_redundancy(relationships))

        # 4. Find transitive redundancy
        if self.merger_config.merge_transitive:
            merge_candidates.extend(await self._find_transitive_redundancy(relationships))

        return merge_candidates

    async def _find_duplicate_relationships(self, relationships: List[RelationshipCandidate]) -> List[MergeCandidate]:
        """Find exact duplicate relationships."""
        duplicates_map: Dict[Tuple[str, str, str], List[RelationshipCandidate]] = {}

        for rel in relationships:
            key = (rel.source_id, rel.target_id, rel.type.upper())
            if key not in duplicates_map:
                duplicates_map[key] = []
            duplicates_map[key].append(rel)

        merge_candidates = []
        for key, rels in duplicates_map.items():
            if len(rels) > 1:
                # Sort by confidence (highest first)
                rels.sort(key=lambda r: r.confidence, reverse=True)
                primary = rels[0]
                duplicates = rels[1:]

                merge_candidates.append(MergeCandidate(
                    primary_relationship=primary,
                    duplicate_relationships=duplicates,
                    merge_reason="duplicate_exact",
                    confidence_score=1.0
                ))

        return merge_candidates

    async def _find_semantic_equivalents(self, relationships: List[RelationshipCandidate]) -> List[MergeCandidate]:
        """Find semantically equivalent relationships using LLM."""
        # Group relationships by source-target pairs
        pairs_map: Dict[Tuple[str, str], List[RelationshipCandidate]] = {}

        for rel in relationships:
            key = (rel.source_id, rel.target_id)
            if key not in pairs_map:
                pairs_map[key] = []
            pairs_map[key].append(rel)

        merge_candidates = []
        for key, rels in pairs_map.items():
            if len(rels) > 1:
                # Check if relationship types are semantically equivalent
                semantic_groups = await self._group_by_semantic_similarity(rels)
                for group in semantic_groups:
                    if len(group) > 1:
                        # Sort by confidence (highest first)
                        group.sort(key=lambda r: r.confidence, reverse=True)
                        primary = group[0]
                        duplicates = group[1:]

                        merge_candidates.append(MergeCandidate(
                            primary_relationship=primary,
                            duplicate_relationships=duplicates,
                            merge_reason="semantic_equivalent",
                            confidence_score=0.8
                        ))

        return merge_candidates

    async def _find_bidirectional_redundancy(self, relationships: List[RelationshipCandidate]) -> List[MergeCandidate]:
        """Find bidirectional relationships that could be consolidated."""
        # Create lookup for reverse relationships
        forward_map: Dict[Tuple[str, str, str], RelationshipCandidate] = {}
        reverse_map: Dict[Tuple[str, str, str], RelationshipCandidate] = {}
        merge_candidates = []

        for rel in relationships:
            forward_key = (rel.source_id, rel.target_id, rel.type.upper())
            reverse_key = (rel.target_id, rel.source_id, rel.type.upper())

            forward_map[forward_key] = rel
            if reverse_key in forward_map:
                # Found bidirectional pair
                reverse_rel = forward_map[reverse_key]
                if rel.confidence >= reverse_rel.confidence:
                    primary = rel
                    duplicate = reverse_rel
                else:
                    primary = reverse_rel
                    duplicate = rel

                # Only merge if they're not already processed
                if reverse_key not in reverse_map:
                    reverse_map[forward_key] = rel
                    merge_candidates.append(MergeCandidate(
                        primary_relationship=primary,
                        duplicate_relationships=[duplicate],
                        merge_reason="bidirectional_redundancy",
                        confidence_score=0.7
                    ))

        return merge_candidates

    async def _find_transitive_redundancy(self, relationships: List[RelationshipCandidate]) -> List[MergeCandidate]:
        """Find transitive relationships that could be consolidated."""
        # This is more complex and requires graph analysis
        # For now, return empty list - can be implemented later if needed
        return []

    async def _group_by_semantic_similarity(self, relationships: List[RelationshipCandidate]) -> List[List[RelationshipCandidate]]:
        """Group relationships by semantic similarity of their types."""
        if len(relationships) <= 1:
            return [relationships]

        # Extract unique relationship types
        types = list(set(rel.type for rel in relationships))
        if len(types) <= 1:
            return [relationships]

        # Use LLM to determine semantic similarity
        prompt = f"""
        Analyze these relationship types and group them by semantic similarity.
        Relationship types: {types}

        Return ONLY a JSON array of arrays containing groups of semantically equivalent types.
        Only group types that have the exact same meaning (e.g., "WORKS_FOR" and "EMPLOYED_BY").
        If no types are semantically equivalent, return an empty array: []

        Examples:
        - Input: ["WORKS_FOR", "EMPLOYED_BY", "LOCATED_IN"] → [["WORKS_FOR", "EMPLOYED_BY"]]
        - Input: ["CAUSES", "TREATS", "PREVENTS"] → []

        Response format: JSON array only, no explanation.
        """

        try:
            response = await self.safe_llm_call(self.llm_client.generate, prompt)

            # Handle empty or whitespace-only responses
            if not response or not response.strip():
                self.logger.warning("LLM returned empty response for semantic similarity")
                return []

            # Try to parse JSON response
            try:
                groups_data = json.loads(response.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
                if json_match:
                    groups_data = json.loads(json_match.group(1))
                else:
                    self.logger.warning("Failed to parse LLM response as JSON", response_preview=response[:100])
                    return []

            # Map back to relationships
            semantic_groups = []
            for group_types in groups_data:
                if len(group_types) > 1:
                    group_rels = [rel for rel in relationships if rel.type in group_types]
                    if len(group_rels) > 1:
                        semantic_groups.append(group_rels)

            return semantic_groups
        except Exception as e:
            self.logger.warning("Failed to analyze semantic similarity", error=str(e))
            return []

    async def _apply_merge(self, merge_candidate: MergeCandidate) -> None:
        """Apply a relationship merge."""
        if self.merger_config.dry_run:
            # In dry run mode, don't actually apply changes
            return

        primary = merge_candidate.primary_relationship
        duplicates = merge_candidate.duplicate_relationships

        for duplicate in duplicates:
            # Update primary relationship with merged metadata
            await self._merge_relationship_metadata(primary, duplicate)

            # Delete duplicate relationship
            await self._delete_relationship(duplicate.id)

        self.logger.info(
            "Merged relationships",
            primary_id=primary.id,
            duplicate_ids=[d.id for d in duplicates],
            reason=merge_candidate.merge_reason
        )

    async def _merge_relationship_metadata(self, primary: RelationshipCandidate, duplicate: RelationshipCandidate) -> None:
        """Merge metadata from duplicate into primary relationship."""
        # Combine metadata
        merged_metadata = primary.metadata.copy()
        merged_metadata.update(duplicate.metadata)
        
        # Add merge tracking
        if "merged_from" not in merged_metadata:
            merged_metadata["merged_from"] = []
        merged_metadata["merged_from"].append(duplicate.id)
        merged_metadata["job_tag"] = self.merger_config.job_tag
        merged_metadata["merged_at"] = datetime.utcnow().isoformat()

        # Update primary relationship
        query = """
        MATCH ()-[r {id: $rel_id}]->()
        SET r.metadata = $metadata,
            r.confidence = $confidence,
            r.updated_at = datetime()
        """

        await self.neo4j_storage._connection_ops._execute_query(query, {
            "rel_id": primary.id,
            "metadata": json.dumps(merged_metadata),
            "confidence": max(primary.confidence, duplicate.confidence)
        })

    async def _delete_relationship(self, relationship_id: str) -> None:
        """Delete a relationship by ID."""
        query = """
        MATCH ()-[r {id: $rel_id}]->()
        DELETE r
        """

        await self.neo4j_storage._connection_ops._execute_query(query, {
            "rel_id": relationship_id
        })


async def run_relationship_merger(overrides: Dict[str, Any]) -> RelationshipMergerResult:
    """Run relationship merger with configuration overrides."""
    import os

    # Create configuration
    config = RelationshipMergerConfig()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Initialize storage and LLM client
    neo4j_config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
    )
    neo4j_storage = Neo4jStorage(neo4j_config)
    await neo4j_storage.connect()

    llm_client = LLMClient()

    try:
        # Run merger
        merger = RelationshipMerger(config, neo4j_storage, llm_client)
        result = await merger.run_merger()
        return result.to_dict()
    finally:
        await neo4j_storage.disconnect()
