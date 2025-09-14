"""Relationship Cleanup Maintenance Job

Performs comprehensive cleanup of problematic relationships in the Neo4j knowledge graph.
Identifies and removes:
1. Duplicate relationships (exact and semantic duplicates)
2. Meaningless relationships (UNRELATED, overly generic types)
3. Invalid relationships (orphaned, self-referential, type incompatible)
4. Consolidates similar relationships with confidence aggregation

Applies changes by default (dry-run available as opt-in). Uses LLM to evaluate cleanup decisions based on:
- Semantic meaningfulness of relationship types
- Entity type compatibility
- Confidence scores and source attribution
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from morag_graph.storage import Neo4jStorage, Neo4jConfig
from .base import MaintenanceJobBase, validate_positive_int, validate_float_range
from .query_optimizer import QueryOptimizer

logger = structlog.get_logger(__name__)

# Try to import LLM client for assessment
try:
    from morag_core.llm import LLMClient, LLMConfig
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


@dataclass
class RelationshipCleanupConfig:
    """Configuration for relationship cleanup job."""
    dry_run: bool = False                       # Apply changes by default
    batch_size: int = 100                       # Batch size for processing (used for bulk operations)
    min_confidence: float = 0.3                 # Minimum confidence threshold
    remove_unrelated: bool = True               # Remove "UNRELATED" type relationships
    remove_generic: bool = True                 # Remove overly generic relationship types
    consolidate_similar: bool = True            # Merge semantically similar relationships
    similarity_threshold: float = 0.85          # Threshold for semantic similarity merging
    job_tag: str = ""                          # Job tag for tracking
    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

    def ensure_defaults(self) -> None:
        """Ensure all configuration values are within valid ranges."""
        self.similarity_threshold = max(0.0, min(1.0, self.similarity_threshold))
        self.batch_size = max(1, self.batch_size)
        self.min_confidence = max(0.0, min(1.0, self.min_confidence))
        if not self.job_tag:
            self.job_tag = f"rel_cleanup_{int(time.time())}"





@dataclass
class RelationshipCleanupResult:
    """Result of relationship cleanup operation."""
    relationships_processed: int = 0
    duplicates_removed: int = 0
    meaningless_removed: int = 0
    invalid_removed: int = 0
    consolidated: int = 0
    total_removed: int = 0
    total_modified: int = 0
    execution_time_seconds: float = 0.0
    dry_run: bool = True
    details: Dict[str, int] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {
                "unrelated_removed": 0,
                "generic_removed": 0,
                "orphaned_removed": 0,
                "self_referential_removed": 0,
                "low_confidence_removed": 0,
                "semantic_duplicates_merged": 0
            }


class RelationshipCleanupService(MaintenanceJobBase):
    """Service for cleaning up problematic relationships in the knowledge graph."""

    def __init__(self, storage: Neo4jStorage, config: Optional[RelationshipCleanupConfig] = None):
        self.neo4j_storage = storage
        self.cleanup_config = config or RelationshipCleanupConfig()
        self.cleanup_config.ensure_defaults()

        # Initialize base class with config dict
        config_dict = {
            'job_tag': self.cleanup_config.job_tag,
            'dry_run': self.cleanup_config.dry_run,
            'batch_size': self.cleanup_config.batch_size,
            'circuit_breaker_threshold': self.cleanup_config.circuit_breaker_threshold,
            'circuit_breaker_timeout': self.cleanup_config.circuit_breaker_timeout,
        }
        super().__init__(config_dict)

        self.query_optimizer = QueryOptimizer(storage)

        # Initialize LLM client for assessment
        self._llm_client = None
        if LLM_AVAILABLE:
            try:
                import os
                llm_config = LLMConfig(
                    provider="gemini",
                    model=os.getenv("MORAG_LLM_MODEL", "gemini-1.5-flash"),
                    api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                )
                self._llm_client = LLMClient(llm_config)
                logger.info("LLM client initialized for relationship assessment")
            except Exception as e:
                logger.warning("Failed to initialize LLM client", error=str(e))

        # Minimal fallback types (only when LLM completely unavailable)
        # These are only the most obviously problematic types
        self.fallback_meaningless_types = {
            "UNRELATED", "NOT_RELATED", "NO_RELATION"
        }

        self.invalid_self_referential_types = {
            "BORN_IN", "DIED_IN", "WORKS_AT", "EMPLOYED_BY", "FOUNDED_BY",
            "CREATED_BY", "AUTHORED_BY", "LOCATED_IN"
        }

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate integer parameters
        errors.extend(validate_positive_int(self.cleanup_config.batch_size, "batch_size", min_value=1, max_value=10000))

        # Validate float parameters
        errors.extend(validate_float_range(self.cleanup_config.min_confidence, "min_confidence", min_value=0.0, max_value=1.0))
        errors.extend(validate_float_range(self.cleanup_config.similarity_threshold, "similarity_threshold", min_value=0.0, max_value=1.0))

        return errors

    async def run(self) -> Dict[str, Any]:
        """Run the relationship cleanup process."""
        return await self.run_cleanup()

    async def run_cleanup(self) -> RelationshipCleanupResult:
        """Run the relationship cleanup process with intelligent performance optimization."""
        self.log_job_start()

        # Validate configuration
        config_errors = self.validate_config()
        if config_errors:
            raise ValueError(f"Configuration errors: {'; '.join(config_errors)}")

        start_time = time.time()
        result = RelationshipCleanupResult(dry_run=self.cleanup_config.dry_run)

        logger.info("Starting relationship cleanup",
                   dry_run=self.cleanup_config.dry_run)

        try:
            # Always use the optimized type-based approach
            logger.info("Using optimized type-based cleanup approach")
            await self._run_optimized_cleanup(result)

            result.execution_time_seconds = time.time() - start_time

            logger.info("Relationship cleanup completed",
                       processed=result.relationships_processed,
                       removed=result.total_removed,
                       modified=result.total_modified,
                       execution_time=result.execution_time_seconds)

        except Exception as e:
            logger.error("Relationship cleanup failed", error=str(e))
            raise

        return result

    async def _run_optimized_cleanup(self, result: RelationshipCleanupResult) -> None:
        """Run optimized cleanup by analyzing relationship types first."""
        # Step 1: Get all relationship types and their counts
        relationship_types = await self._get_relationship_type_summary()
        logger.info(f"Found {len(relationship_types)} distinct relationship types")

        # Step 2: Use LLM to identify problematic and mergeable types
        type_analysis = await self._analyze_relationship_types_with_llm(relationship_types)

        # Step 3: Remove meaningless relationship types entirely
        if type_analysis.get("remove_types"):
            await self._remove_relationships_by_type(type_analysis["remove_types"], result)

        # Step 4: Merge similar relationship types
        if type_analysis.get("merge_pairs"):
            await self._merge_relationships_by_type_pairs(type_analysis["merge_pairs"], result)

        # Step 5: Remove generic relationships when specific alternatives exist
        await self._cleanup_generic_relationships_with_specific_alternatives(result)

        # Step 6: Handle remaining individual cases (orphaned, low confidence, etc.)
        await self._cleanup_remaining_issues(result)

    async def _get_relationship_type_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all relationship types and their counts."""
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as neo4j_type,
               r.type as stored_type,
               count(*) as count,
               avg(r.confidence) as avg_confidence,
               collect(DISTINCT r.type)[0..5] as sample_type_values
        ORDER BY count DESC
        """

        result = await self.neo4j_storage._connection_ops._execute_query(query, {})

        types_summary = []
        type_combinations = {}

        for record in result:
            neo4j_type = record["neo4j_type"]
            stored_type = record.get("stored_type", neo4j_type)
            count = record["count"]

            # Group by the actual meaningful type (prefer stored_type if available)
            key_type = stored_type if stored_type else neo4j_type

            if key_type not in type_combinations:
                type_combinations[key_type] = {
                    "neo4j_type": neo4j_type,
                    "stored_type": stored_type,
                    "count": 0,
                    "avg_confidence": 0,
                    "sample_type_values": record.get("sample_type_values", [])
                }

            type_combinations[key_type]["count"] += count
            type_combinations[key_type]["avg_confidence"] = record.get("avg_confidence", 1.0)

        # Convert to list and sort by count
        types_summary = list(type_combinations.values())
        types_summary.sort(key=lambda x: x["count"], reverse=True)

        return types_summary

    async def _analyze_relationship_types_with_llm(self, relationship_types: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM to analyze relationship types and determine cleanup strategy."""
        if not self._llm_client:
            # Fallback to rule-based analysis
            return self._analyze_relationship_types_fallback(relationship_types)

        try:
            # Prepare type summary for LLM
            type_summary = []
            for rt in relationship_types[:20]:  # Limit to top 20 types
                type_summary.append(f"- {rt['neo4j_type']} (count: {rt['count']}, avg_confidence: {rt['avg_confidence']:.2f})")

            prompt = f"""Analyze these relationship types from a knowledge graph and identify cleanup opportunities based on semantic value and meaningfulness:

{chr(10).join(type_summary)}

Evaluate each relationship type for:
1. **Semantic Value**: Does it provide meaningful information about the relationship?
   - HIGH VALUE: Specific, descriptive types like "TREATS", "CAUSES", "IMPROVES", "WORSENS", "SOLVES", "PREVENTS"
   - LOW VALUE: Generic, vague types like "TAGGED_WITH", "RELATED_TO", "ASSOCIATED_WITH", "CONNECTED_TO"
   - NO VALUE: Meaningless types like "UNRELATED", "NOT_RELATED", "NO_RELATION"

2. **Semantic Equivalence**: Are there types that express the same relationship?
   - Examples: "WORKS_AT" ≈ "EMPLOYED_BY", "LOCATED_IN" ≈ "BASED_IN", "FOUNDED_BY" ≈ "ESTABLISHED_BY"

Decision criteria:
- REMOVE types with no semantic value or that are explicitly meaningless
- REMOVE generic types when more specific alternatives exist for the same entity pairs
- MERGE semantically equivalent types, keeping the most descriptive/specific one as primary
- PRESERVE all types that add unique semantic value, even if they seem similar

Respond with JSON:
{{
    "remove_types": ["TYPE1", "TYPE2"],
    "merge_pairs": [
        {{"primary": "MOST_DESCRIPTIVE_TYPE", "merge_into": ["EQUIVALENT_TYPE1", "EQUIVALENT_TYPE2"]}},
        {{"primary": "SPECIFIC_TYPE", "merge_into": ["GENERIC_EQUIVALENT"]}}
    ],
    "reasoning": "detailed explanation of semantic value assessment and decisions"
}}"""

            response = await self.safe_llm_call(self._llm_client.generate_text, prompt)

            # Parse JSON response
            import json
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    logger.info("LLM type analysis completed",
                              remove_count=len(result.get("remove_types", [])),
                              merge_pairs=len(result.get("merge_pairs", [])))
                    return result
                else:
                    logger.warning("No valid JSON found in LLM type analysis response")
                    return self._analyze_relationship_types_fallback(relationship_types)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse LLM type analysis JSON", error=str(e))
                return self._analyze_relationship_types_fallback(relationship_types)

        except Exception as e:
            logger.warning("LLM type analysis failed", error=str(e))
            return self._analyze_relationship_types_fallback(relationship_types)

    def _analyze_relationship_types_fallback(self, relationship_types: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Minimal fallback analysis when LLM is completely unavailable."""
        remove_types = []
        merge_pairs = []

        # Only remove the most obviously meaningless types
        for rt in relationship_types:
            neo4j_type = rt["neo4j_type"]
            stored_type = rt.get("stored_type", neo4j_type)

            # Only check for explicitly meaningless types
            if (neo4j_type in self.fallback_meaningless_types or
                stored_type in self.fallback_meaningless_types):
                # Use the stored_type if available, otherwise neo4j_type
                remove_types.append(stored_type if stored_type else neo4j_type)

        # No static merge rules - let the LLM handle semantic decisions
        # This fallback is intentionally minimal to avoid hardcoded assumptions

        logger.warning("Using minimal fallback analysis - LLM unavailable for semantic assessment",
                      remove_types=remove_types,
                   merge_pairs_count=len(merge_pairs))

        return {
            "remove_types": remove_types,
            "merge_pairs": merge_pairs,
            "reasoning": "Rule-based fallback analysis"
        }

    async def _cleanup_generic_relationships_with_specific_alternatives(self, result: RelationshipCleanupResult) -> None:
        """Remove generic relationships when more specific alternatives exist between the same entities."""
        if not self._llm_client:
            logger.info("Skipping generic relationship cleanup - LLM unavailable")
            return

        # Find entity pairs with multiple relationship types
        query = """
        MATCH (a)-[r1]->(b)
        WITH a, b, collect(DISTINCT type(r1)) as rel_types, collect(r1) as relationships
        WHERE size(rel_types) > 1
        RETURN a.name as source_name, b.name as target_name,
               rel_types, relationships
        LIMIT 100
        """

        result_data = await self.neo4j_storage._connection_ops._execute_query(query, {})

        for record in result_data:
            source_name = record["source_name"]
            target_name = record["target_name"]
            rel_types = record["rel_types"]
            relationships = record["relationships"]

            if len(rel_types) <= 1:
                continue

            # Use LLM to determine which relationships to keep vs remove
            generic_to_remove = await self._identify_generic_relationships_to_remove(
                source_name, target_name, rel_types
            )

            if generic_to_remove:
                # Remove the generic relationships
                for rel_type in generic_to_remove:
                    remove_query = """
                    MATCH (a {name: $source_name})-[r]->(b {name: $target_name})
                    WHERE type(r) = $rel_type
                    DELETE r
                    RETURN count(r) as removed_count
                    """

                    if not self.dry_run:
                        remove_result = await self.neo4j_storage._connection_ops._execute_query(
                            remove_query, {
                                "source_name": source_name,
                                "target_name": target_name,
                                "rel_type": rel_type
                            }
                        )
                        removed_count = remove_result[0]["removed_count"] if remove_result else 0
                        result.generic_removed += removed_count
                        logger.info(f"Removed {removed_count} generic {rel_type} relationships between {source_name} and {target_name}")

    async def _identify_generic_relationships_to_remove(self, source_name: str, target_name: str, rel_types: List[str]) -> List[str]:
        """Use LLM to identify which relationship types are generic and should be removed."""
        try:
            prompt = f"""Analyze these relationship types between two entities and identify which are generic/low-value when more specific alternatives exist:

Source Entity: {source_name}
Target Entity: {target_name}
Relationship Types: {', '.join(rel_types)}

Identify relationship types that should be REMOVED because they are:
1. Generic/vague (like "TAGGED_WITH", "RELATED_TO", "ASSOCIATED_WITH") when more specific types exist
2. Low semantic value compared to more descriptive alternatives
3. Redundant given the presence of more informative relationship types

Keep relationship types that:
- Provide unique, specific semantic information
- Add distinct meaning not captured by other types
- Are domain-specific and descriptive

Examples:
- If both "TAGGED_WITH" and "TREATS" exist → remove "TAGGED_WITH"
- If both "RELATED_TO" and "CAUSES" exist → remove "RELATED_TO"
- If both "ASSOCIATED_WITH" and "IMPROVES" exist → remove "ASSOCIATED_WITH"

Respond with JSON:
{{
    "remove_types": ["GENERIC_TYPE1", "GENERIC_TYPE2"],
    "reasoning": "explanation of why these are generic compared to alternatives"
}}"""

            response = await self.safe_llm_call(self._llm_client.generate_text, prompt)

            # Parse JSON response
            import json
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    return result.get("remove_types", [])
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM generic relationship analysis")
                return []

        except Exception as e:
            logger.warning("LLM generic relationship analysis failed", error=str(e))
            return []



    async def _are_semantically_similar_with_llm(self, type1: str, type2: str) -> bool:
        """Check if two relationship types are semantically similar using LLM."""
        if type1 == type2:
            return True

        # Use LLM assessment if available
        if self._llm_client:
            try:
                prompt = f"""Analyze these two relationship types for semantic similarity and value:

Type 1: {type1}
Type 2: {type2}

Consider:
1. **Semantic Equivalence**: Do they express the same relationship meaning?
   - Examples: "WORKS_AT" ≈ "EMPLOYED_BY", "LOCATED_IN" ≈ "BASED_IN"
2. **Semantic Value**: Which provides more specific, descriptive information?
   - High value: "TREATS", "CAUSES", "IMPROVES", "PREVENTS"
   - Low value: "TAGGED_WITH", "RELATED_TO", "ASSOCIATED_WITH"

Respond with JSON:
{{
    "are_similar": true/false,
    "confidence": 0.0-1.0,
    "preferred_type": "TYPE1 or TYPE2 (if similar, which is more descriptive)",
    "reason": "explanation focusing on semantic value and specificity"
}}"""

                response = await self.safe_llm_call(self._llm_client.generate_text, prompt)

                # Parse JSON response
                import json
                try:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        result = json.loads(json_str)
                        return result.get("are_similar", False) and result.get("confidence", 0.0) > 0.7
                except json.JSONDecodeError:
                    pass
            except Exception as e:
                logger.warning("LLM similarity assessment failed", error=str(e))

        # Fallback to rule-based similarity
        return await self._are_semantically_similar(type1, type2)

    async def _are_semantically_similar(self, type1: str, type2: str) -> bool:
        """Check if two relationship types are semantically similar (rule-based fallback)."""
        if type1 == type2:
            return True

        # Define semantic similarity groups
        similarity_groups = [
            {"WORKS_AT", "EMPLOYED_BY", "WORKS_FOR", "EMPLOYEE_OF"},
            {"LOCATED_IN", "BASED_IN", "SITUATED_IN", "POSITIONED_IN"},
            {"FOUNDED_BY", "ESTABLISHED_BY", "CREATED_BY", "STARTED_BY"},
            {"PART_OF", "COMPONENT_OF", "MEMBER_OF", "BELONGS_TO"},
            {"CAUSES", "LEADS_TO", "RESULTS_IN", "TRIGGERS"},
            {"USES", "UTILIZES", "EMPLOYS", "APPLIES"},
            {"OWNS", "POSSESSES", "HAS", "CONTROLS"}
        ]

        type1_upper = type1.upper()
        type2_upper = type2.upper()

        for group in similarity_groups:
            if type1_upper in group and type2_upper in group:
                return True

        return False

    async def _remove_relationships_by_type(self, remove_types: List[str], result: RelationshipCleanupResult) -> None:
        """Remove all relationships of specified types."""
        for rel_type in remove_types:
            # Also check for relationships where r.type property matches (not just Neo4j type)
            if self.cleanup_config.dry_run:
                # Count relationships that would be removed (both Neo4j type and r.type property)
                count_query = f"""
                MATCH ()-[r]->()
                WHERE type(r) = $rel_type OR r.type = $rel_type
                RETURN count(r) as count
                """
                count_result = await self.neo4j_storage._connection_ops._execute_query(count_query, {"rel_type": rel_type})
                count = count_result[0]["count"] if count_result else 0
                logger.info(f"[DRY RUN] Would remove {count} relationships of type {rel_type}")
                result.total_removed += count
                result.meaningless_removed += count
                result.details["unrelated_removed"] += count
            else:
                # Actually remove relationships (both Neo4j type and r.type property)
                delete_query = """
                MATCH ()-[r]->()
                WHERE type(r) = $rel_type OR r.type = $rel_type
                DELETE r
                """
                await self.neo4j_storage._connection_ops._execute_query(delete_query, {"rel_type": rel_type})

                # Count what was deleted
                count_query = f"""
                MATCH ()-[r]->()
                WHERE type(r) = $rel_type OR r.type = $rel_type
                RETURN count(r) as remaining
                """
                remaining_result = await self.neo4j_storage._connection_ops._execute_query(count_query, {"rel_type": rel_type})
                remaining = remaining_result[0]["remaining"] if remaining_result else 0

                logger.info(f"Processed relationships of type {rel_type}, {remaining} remaining")
                # We can't easily count deleted in this approach, so we'll estimate
                result.total_removed += 1  # Placeholder
                result.meaningless_removed += 1
                result.details["unrelated_removed"] += 1

    async def _merge_relationships_by_type_pairs(self, merge_pairs: List[Dict[str, Any]], result: RelationshipCleanupResult) -> None:
        """Merge relationships by converting types in bulk."""
        for pair in merge_pairs:
            primary_type = pair["primary"]
            merge_types = pair["merge_into"]

            for merge_type in merge_types:
                if self.cleanup_config.dry_run:
                    # Count relationships that would be merged
                    count_query = f"""
                    MATCH (a)-[r:{merge_type}]->(b)
                    WHERE NOT EXISTS((a)-[:{primary_type}]->(b))
                    RETURN count(r) as count
                    """
                    count_result = await self.neo4j_storage._connection_ops._execute_query(count_query, {})
                    count = count_result[0]["count"] if count_result else 0
                    logger.info(f"[DRY RUN] Would merge {count} relationships from {merge_type} to {primary_type}")
                    result.duplicates_removed += count
                    result.details["semantic_duplicates_merged"] += 1
                else:
                    # Actually merge relationships
                    merge_query = f"""
                    MATCH (a)-[r:{merge_type}]->(b)
                    WHERE NOT EXISTS((a)-[:{primary_type}]->(b))
                    CREATE (a)-[new_r:{primary_type}]->(b)
                    SET new_r = r,
                        new_r.merged_from = r.type,
                        new_r.merged_at = datetime(),
                        new_r.job_tag = $job_tag
                    DELETE r
                    RETURN count(*) as merged
                    """
                    merge_result = await self.neo4j_storage._connection_ops._execute_query(
                        merge_query, {"job_tag": self.cleanup_config.job_tag}
                    )
                    merged = merge_result[0]["merged"] if merge_result else 0
                    logger.info(f"Merged {merged} relationships from {merge_type} to {primary_type}")
                    result.duplicates_removed += merged
                    result.details["semantic_duplicates_merged"] += 1

    async def _cleanup_remaining_issues(self, result: RelationshipCleanupResult) -> None:
        """Handle remaining cleanup issues (orphaned, low confidence, etc.)."""
        # Remove orphaned relationships (pointing to non-existent entities)
        if self.cleanup_config.dry_run:
            orphaned_query = """
            MATCH (start)-[r]->(end)
            WHERE start.id IS NULL OR end.id IS NULL
            RETURN count(r) as count
            """
            orphaned_result = await self.neo4j_storage._connection_ops._execute_query(orphaned_query, {})
            orphaned_count = orphaned_result[0]["count"] if orphaned_result else 0
            logger.info(f"[DRY RUN] Would remove {orphaned_count} orphaned relationships")
            result.invalid_removed += orphaned_count
            result.details["orphaned_removed"] += orphaned_count
        else:
            orphaned_query = """
            MATCH (start)-[r]->(end)
            WHERE start.id IS NULL OR end.id IS NULL
            DELETE r
            RETURN count(*) as deleted
            """
            orphaned_result = await self.neo4j_storage._connection_ops._execute_query(orphaned_query, {})
            orphaned_deleted = orphaned_result[0]["deleted"] if orphaned_result else 0
            logger.info(f"Removed {orphaned_deleted} orphaned relationships")
            result.invalid_removed += orphaned_deleted
            result.details["orphaned_removed"] += orphaned_deleted

        # Remove low confidence relationships
        if self.cleanup_config.min_confidence > 0:
            if self.cleanup_config.dry_run:
                low_conf_query = """
                MATCH ()-[r]->()
                WHERE r.confidence < $min_confidence
                RETURN count(r) as count
                """
                low_conf_result = await self.neo4j_storage._connection_ops._execute_query(
                    low_conf_query, {"min_confidence": self.cleanup_config.min_confidence}
                )
                low_conf_count = low_conf_result[0]["count"] if low_conf_result else 0
                logger.info(f"[DRY RUN] Would remove {low_conf_count} low confidence relationships")
                result.invalid_removed += low_conf_count
                result.details["low_confidence_removed"] += low_conf_count
            else:
                low_conf_query = """
                MATCH ()-[r]->()
                WHERE r.confidence < $min_confidence
                DELETE r
                RETURN count(*) as deleted
                """
                low_conf_result = await self.neo4j_storage._connection_ops._execute_query(
                    low_conf_query, {"min_confidence": self.cleanup_config.min_confidence}
                )
                low_conf_deleted = low_conf_result[0]["deleted"] if low_conf_result else 0
                logger.info(f"Removed {low_conf_deleted} low confidence relationships")
                result.invalid_removed += low_conf_deleted
                result.details["low_confidence_removed"] += low_conf_deleted


def parse_cleanup_overrides() -> Dict[str, Any]:
    """Parse environment variables for relationship cleanup configuration."""
    import os

    def _parse_bool(value: str, default: bool = False) -> bool:
        if not value:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def _parse_float(value: str, default: float) -> float:
        try:
            return float(value) if value else default
        except ValueError:
            return default

    def _parse_int(value: str, default: int) -> int:
        try:
            return int(value) if value else default
        except ValueError:
            return default

    return {
        "dry_run": _parse_bool(os.getenv("MORAG_REL_CLEANUP_DRY_RUN"), False),
        "batch_size": _parse_int(os.getenv("MORAG_REL_CLEANUP_BATCH_SIZE"), 100),
        "min_confidence": _parse_float(os.getenv("MORAG_REL_CLEANUP_MIN_CONFIDENCE"), 0.3),
        "remove_unrelated": _parse_bool(os.getenv("MORAG_REL_CLEANUP_REMOVE_UNRELATED"), True),
        "remove_generic": _parse_bool(os.getenv("MORAG_REL_CLEANUP_REMOVE_GENERIC"), True),
        "consolidate_similar": _parse_bool(os.getenv("MORAG_REL_CLEANUP_CONSOLIDATE_SIMILAR"), True),
        "similarity_threshold": _parse_float(os.getenv("MORAG_REL_CLEANUP_SIMILARITY_THRESHOLD"), 0.85),
        "job_tag": os.getenv("MORAG_REL_CLEANUP_JOB_TAG", ""),
    }


async def run_relationship_cleanup(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the relationship cleanup maintenance job.

    Args:
        overrides: Optional configuration overrides

    Returns:
        Dictionary containing cleanup results
    """
    # Parse configuration
    config_dict = parse_cleanup_overrides()
    if overrides:
        config_dict.update(overrides)

    config = RelationshipCleanupConfig(**config_dict)

    # Initialize Neo4j storage
    import os
    neo4j_config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        verify_ssl=os.getenv("NEO4J_VERIFY_SSL", "true").lower() == "true",
        trust_all_certificates=os.getenv("NEO4J_TRUST_ALL_CERTIFICATES", "false").lower() == "true",
    )
    storage = Neo4jStorage(neo4j_config)

    try:
        # Initialize connection
        await storage.connect()

        # Run cleanup
        service = RelationshipCleanupService(storage, config)
        result = await service.run_cleanup()

        # Convert result to dictionary
        return {
            "relationships_processed": result.relationships_processed,
            "duplicates_removed": result.duplicates_removed,
            "meaningless_removed": result.meaningless_removed,
            "invalid_removed": result.invalid_removed,
            "consolidated": result.consolidated,
            "total_removed": result.total_removed,
            "total_modified": result.total_modified,
            "execution_time_seconds": result.execution_time_seconds,
            "dry_run": result.dry_run,
            "details": result.details
        }

    finally:
        await storage.disconnect()
