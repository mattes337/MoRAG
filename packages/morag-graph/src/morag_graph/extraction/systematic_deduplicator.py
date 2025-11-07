"""Systematic entity and relation deduplication across document chunks."""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel, Field

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from ..models import Entity, Relation
from .entity_normalizer import LLMEntityNormalizer

logger = structlog.get_logger(__name__)


class MergeDecision(BaseModel):
    """Structured merge decision from LLM."""

    should_merge: bool = Field(description="Whether entities should be merged")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    reasoning: str = Field(description="Explanation for the decision")


@dataclass
class MergeCandidate:
    """Candidate entities for merging."""

    primary_entity: Entity
    duplicate_entities: List[Entity]
    similarity_score: float
    merge_confidence: float
    merge_reason: str


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""

    original_count: int
    deduplicated_count: int
    merges_performed: int
    processing_time: float
    merge_details: List[MergeCandidate]


class EntitySimilarityCalculator:
    """Calculate similarity between entities for deduplication."""

    def __init__(self):
        self.logger = logger.bind(component="entity_similarity")

    def calculate_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity score between two entities."""
        # Name similarity (most important)
        name_sim = self._calculate_name_similarity(entity1.name, entity2.name)

        # Type similarity
        type_sim = self._calculate_type_similarity(entity1.type, entity2.type)

        # Context similarity (if available)
        context_sim = self._calculate_context_similarity(entity1, entity2)

        # Weighted combination
        similarity = name_sim * 0.6 + type_sim * 0.3 + context_sim * 0.1

        return min(1.0, max(0.0, similarity))

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity using multiple methods."""
        name1_lower = name1.lower().strip()
        name2_lower = name2.lower().strip()

        # Exact match
        if name1_lower == name2_lower:
            return 1.0

        # One name contains the other
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.8

        # Jaccard similarity of words
        words1 = set(name1_lower.split())
        words2 = set(name2_lower.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        jaccard = intersection / union if union > 0 else 0.0

        # Edit distance similarity (simplified)
        edit_sim = self._calculate_edit_similarity(name1_lower, name2_lower)

        return max(jaccard, edit_sim)

    def _calculate_edit_similarity(self, str1: str, str2: str) -> float:
        """Calculate edit distance similarity."""
        # Simple Levenshtein distance implementation
        if len(str1) == 0:
            return 0.0 if len(str2) > 0 else 1.0
        if len(str2) == 0:
            return 0.0

        # Create matrix
        matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

        # Initialize first row and column
        for i in range(len(str1) + 1):
            matrix[i][0] = i
        for j in range(len(str2) + 1):
            matrix[0][j] = j

        # Fill matrix
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i - 1] == str2[j - 1]:
                    matrix[i][j] = matrix[i - 1][j - 1]
                else:
                    matrix[i][j] = min(
                        matrix[i - 1][j] + 1,  # deletion
                        matrix[i][j - 1] + 1,  # insertion
                        matrix[i - 1][j - 1] + 1,  # substitution
                    )

        # Convert to similarity score
        max_len = max(len(str1), len(str2))
        distance = matrix[len(str1)][len(str2)]

        return 1.0 - (distance / max_len) if max_len > 0 else 0.0

    def _calculate_type_similarity(self, type1: str, type2: str) -> float:
        """Calculate type similarity."""
        if not type1 or not type2:
            return 0.5  # Neutral score for missing types

        type1_lower = type1.lower().strip()
        type2_lower = type2.lower().strip()

        if type1_lower == type2_lower:
            return 1.0

        # Check for related types
        related_types = {
            "person": ["individual", "human", "people"],
            "organization": ["company", "institution", "org", "business"],
            "location": ["place", "city", "country", "region"],
            "concept": ["idea", "theory", "principle", "notion"],
            "technology": ["tech", "tool", "system", "platform"],
        }

        for main_type, related in related_types.items():
            if (
                (type1_lower == main_type and type2_lower in related)
                or (type2_lower == main_type and type1_lower in related)
                or (type1_lower in related and type2_lower in related)
            ):
                return 0.8

        return 0.0

    def _calculate_context_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate context similarity based on source documents."""
        # Same document bonus
        if (
            entity1.source_doc_id
            and entity2.source_doc_id
            and entity1.source_doc_id == entity2.source_doc_id
        ):
            return 1.0

        # Different documents penalty
        return 0.3


class LLMMergeValidator:
    """Use LLM to validate entity merges."""

    def __init__(self, normalizer: Optional[LLMEntityNormalizer] = None):
        self.normalizer = normalizer
        self.logger = logger.bind(component="llm_merge_validator")

    async def confirm_merge(
        self, primary: Entity, candidates: List[Entity], similarity_score: float
    ) -> Tuple[bool, float, str]:
        """Confirm if entities should be merged using LLM."""
        if not self.normalizer or not self.normalizer.model:
            # Fallback to rule-based confirmation
            return self._rule_based_confirmation(primary, candidates, similarity_score)

        try:
            # Create merge confirmation prompt
            prompt = self._create_merge_prompt(primary, candidates, similarity_score)

            # Get LLM response
            response = await self._get_llm_response(prompt)

            # Parse response
            return self._parse_merge_response(response)

        except Exception as e:
            self.logger.warning(f"LLM merge validation failed: {e}")
            return self._rule_based_confirmation(primary, candidates, similarity_score)

    def _create_merge_prompt(
        self, primary: Entity, candidates: List[Entity], similarity_score: float
    ) -> str:
        """Create prompt for merge confirmation."""
        candidate_info = "\n".join(
            [
                f"- {entity.name} (Type: {entity.type}, Source: {entity.source_doc_id})"
                for entity in candidates
            ]
        )

        return f"""
        Determine if these entities should be merged:

        Primary Entity: {primary.name} (Type: {primary.type}, Source: {primary.source_doc_id})

        Candidate Entities:
        {candidate_info}

        Similarity Score: {similarity_score:.2f}

        Consider:
        1. Are these referring to the same real-world entity?
        2. Are the names variations of the same entity (e.g., "Dr. Smith" vs "John Smith")?
        3. Are they different entities that happen to have similar names?

        Respond with JSON:
        {{
            "should_merge": true/false,
            "confidence": 0.0-1.0,
            "reasoning": "explanation"
        }}
        """

    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        if not self.normalizer or not self.normalizer.model:
            raise ValueError("No LLM model available")

        try:
            response = await asyncio.to_thread(
                self.normalizer.model.generate_content, prompt
            )
            return response.text if response.text else ""
        except Exception as e:
            self.logger.error(f"LLM response failed: {e}")
            raise

    def _parse_merge_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse LLM response for merge decision."""
        # This method is now deprecated - structured generation should be used instead
        # Keeping minimal fallback for compatibility
        try:
            # Try to parse as JSON for backwards compatibility
            import json

            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]

            data = json.loads(response_clean)
            should_merge = data.get("should_merge", False)
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "LLM decision")

            return should_merge, confidence, reasoning

        except Exception as e:
            self.logger.warning(f"Legacy JSON parsing failed: {e}")
            # Conservative fallback
            return False, 0.3, f"parse_error: {str(e)}"

    def _rule_based_confirmation(
        self, primary: Entity, candidates: List[Entity], similarity_score: float
    ) -> Tuple[bool, float, str]:
        """Rule-based merge confirmation fallback."""
        # High similarity threshold for automatic merge
        if similarity_score >= 0.9:
            return True, 0.9, "high_similarity_auto_merge"

        # Medium similarity with additional checks
        if similarity_score >= 0.7:
            # Check if names are very similar and types match
            for candidate in candidates:
                if (
                    primary.type.lower() == candidate.type.lower()
                    and self._names_are_variations(primary.name, candidate.name)
                ):
                    return True, 0.8, "name_variation_type_match"

        # Conservative approach for lower similarities
        return False, similarity_score, "insufficient_similarity"

    def _names_are_variations(self, name1: str, name2: str) -> bool:
        """Check if names are likely variations of the same entity."""
        name1_words = set(name1.lower().split())
        name2_words = set(name2.lower().split())

        # One name is subset of another
        if name1_words.issubset(name2_words) or name2_words.issubset(name1_words):
            return True

        # Significant word overlap
        overlap = len(name1_words.intersection(name2_words))
        min_words = min(len(name1_words), len(name2_words))

        return overlap >= min_words * 0.7


class SystematicDeduplicator:
    """Systematic entity and relation deduplication across chunks."""

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        merge_confidence_threshold: float = 0.8,
        enable_llm_validation: bool = True,
    ):
        """Initialize systematic deduplicator.

        Args:
            similarity_threshold: Minimum similarity for merge candidates
            merge_confidence_threshold: Minimum confidence for automatic merge
            enable_llm_validation: Whether to use LLM for merge validation
        """
        self.similarity_threshold = similarity_threshold
        self.merge_confidence_threshold = merge_confidence_threshold
        self.enable_llm_validation = enable_llm_validation

        self.similarity_calculator = EntitySimilarityCalculator()
        self.merge_validator = LLMMergeValidator() if enable_llm_validation else None

        self.logger = logger.bind(component="systematic_deduplicator")

    async def deduplicate_across_chunks(
        self, entities_by_chunk: Dict[str, List[Entity]]
    ) -> Tuple[Dict[str, List[Entity]], DeduplicationResult]:
        """Systematically deduplicate entities across all chunks."""
        start_time = time.time()

        # Flatten all entities
        all_entities = []
        chunk_mapping = {}  # entity_id -> chunk_id

        for chunk_id, entities in entities_by_chunk.items():
            for entity in entities:
                all_entities.append(entity)
                chunk_mapping[entity.id] = chunk_id

        original_count = len(all_entities)

        self.logger.info(
            "Starting systematic deduplication",
            total_entities=original_count,
            chunks=len(entities_by_chunk),
        )

        # Step 1: Build similarity matrix
        merge_candidates = await self._find_merge_candidates(all_entities)

        # Step 2: Validate merges with LLM
        confirmed_merges = await self._confirm_merges_with_llm(merge_candidates)

        # Step 3: Apply merges
        deduplicated_entities, merge_mapping = self._apply_merges(
            all_entities, confirmed_merges
        )

        # Step 4: Rebuild chunk mapping
        deduplicated_by_chunk = self._rebuild_chunk_mapping(
            deduplicated_entities, chunk_mapping, merge_mapping
        )

        processing_time = time.time() - start_time

        result = DeduplicationResult(
            original_count=original_count,
            deduplicated_count=len(deduplicated_entities),
            merges_performed=len(confirmed_merges),
            processing_time=processing_time,
            merge_details=confirmed_merges,
        )

        self.logger.info(
            "Deduplication completed",
            original_entities=original_count,
            deduplicated_entities=len(deduplicated_entities),
            merges_performed=len(confirmed_merges),
            processing_time=f"{processing_time:.2f}s",
        )

        return deduplicated_by_chunk, result

    async def _find_merge_candidates(
        self, entities: List[Entity]
    ) -> List[MergeCandidate]:
        """Find potential merge candidates using similarity analysis."""
        candidates = []
        processed = set()

        for i, entity1 in enumerate(entities):
            if entity1.id in processed:
                continue

            duplicates = []

            for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                if entity2.id in processed:
                    continue

                similarity = self.similarity_calculator.calculate_similarity(
                    entity1, entity2
                )

                if similarity >= self.similarity_threshold:
                    duplicates.append(entity2)
                    processed.add(entity2.id)

            if duplicates:
                # Calculate average similarity for the group
                avg_similarity = sum(
                    self.similarity_calculator.calculate_similarity(entity1, dup)
                    for dup in duplicates
                ) / len(duplicates)

                candidates.append(
                    MergeCandidate(
                        primary_entity=entity1,
                        duplicate_entities=duplicates,
                        similarity_score=avg_similarity,
                        merge_confidence=0.0,  # Will be set by LLM validation
                        merge_reason="similarity_analysis",
                    )
                )

                processed.add(entity1.id)

        self.logger.info(f"Found {len(candidates)} merge candidates")
        return candidates

    async def _confirm_merges_with_llm(
        self, candidates: List[MergeCandidate]
    ) -> List[MergeCandidate]:
        """Confirm merges using LLM validation."""
        if not self.merge_validator:
            # Use rule-based confirmation
            return self._rule_based_merge_confirmation(candidates)

        confirmed = []

        for candidate in candidates:
            try:
                (
                    should_merge,
                    confidence,
                    reasoning,
                ) = await self.merge_validator.confirm_merge(
                    candidate.primary_entity,
                    candidate.duplicate_entities,
                    candidate.similarity_score,
                )

                if should_merge and confidence >= self.merge_confidence_threshold:
                    candidate.merge_confidence = confidence
                    candidate.merge_reason = reasoning
                    confirmed.append(candidate)

            except Exception as e:
                self.logger.warning(
                    f"Merge validation failed for candidate: {e}",
                    primary_entity=candidate.primary_entity.name,
                )

        self.logger.info(
            f"Confirmed {len(confirmed)} merges out of {len(candidates)} candidates"
        )
        return confirmed

    def _rule_based_merge_confirmation(
        self, candidates: List[MergeCandidate]
    ) -> List[MergeCandidate]:
        """Rule-based merge confirmation fallback."""
        confirmed = []

        for candidate in candidates:
            # High similarity threshold for automatic confirmation
            if candidate.similarity_score >= 0.9:
                candidate.merge_confidence = 0.9
                candidate.merge_reason = "high_similarity_rule"
                confirmed.append(candidate)
            elif candidate.similarity_score >= 0.8:
                # Additional checks for medium similarity
                primary = candidate.primary_entity

                # Check if all duplicates have same type
                same_type = all(
                    dup.type.lower() == primary.type.lower()
                    for dup in candidate.duplicate_entities
                )

                if same_type:
                    candidate.merge_confidence = 0.8
                    candidate.merge_reason = "medium_similarity_same_type"
                    confirmed.append(candidate)

        return confirmed

    def _apply_merges(
        self, entities: List[Entity], confirmed_merges: List[MergeCandidate]
    ) -> Tuple[List[Entity], Dict[str, str]]:
        """Apply confirmed merges to entity list."""
        # Create mapping from duplicate IDs to primary IDs
        merge_mapping = {}
        entities_to_remove = set()

        for merge in confirmed_merges:
            primary_id = merge.primary_entity.id

            for duplicate in merge.duplicate_entities:
                merge_mapping[duplicate.id] = primary_id
                entities_to_remove.add(duplicate.id)

        # Filter out merged entities
        deduplicated = [
            entity for entity in entities if entity.id not in entities_to_remove
        ]

        return deduplicated, merge_mapping

    def _rebuild_chunk_mapping(
        self,
        deduplicated_entities: List[Entity],
        original_chunk_mapping: Dict[str, str],
        merge_mapping: Dict[str, str],
    ) -> Dict[str, List[Entity]]:
        """Rebuild chunk mapping after deduplication."""
        chunk_entities = defaultdict(list)

        for entity in deduplicated_entities:
            # Find original chunk for this entity
            chunk_id = original_chunk_mapping.get(entity.id)

            if chunk_id:
                chunk_entities[chunk_id].append(entity)

        return dict(chunk_entities)

    async def deduplicate_relations(
        self,
        relations_by_chunk: Dict[str, List[Relation]],
        entity_merge_mapping: Dict[str, str],
    ) -> Dict[str, List[Relation]]:
        """Deduplicate relations after entity deduplication."""
        all_relations = []
        chunk_mapping = {}

        # Flatten relations and update entity references
        for chunk_id, relations in relations_by_chunk.items():
            for relation in relations:
                # Update entity references based on merge mapping
                source_id = entity_merge_mapping.get(
                    relation.source_entity_id, relation.source_entity_id
                )
                target_id = entity_merge_mapping.get(
                    relation.target_entity_id, relation.target_entity_id
                )

                # Create updated relation
                updated_relation = Relation(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    type=relation.type,
                    description=relation.description,
                    confidence=relation.confidence,
                    source_doc_id=relation.source_doc_id,
                )

                all_relations.append(updated_relation)
                chunk_mapping[updated_relation.id] = chunk_id

        # Deduplicate relations by source, target, and type
        deduplicated_relations = self._deduplicate_relation_list(all_relations)

        # Rebuild chunk mapping
        deduplicated_by_chunk = defaultdict(list)
        for relation in deduplicated_relations:
            chunk_id = chunk_mapping.get(relation.id)
            if chunk_id:
                deduplicated_by_chunk[chunk_id].append(relation)

        return dict(deduplicated_by_chunk)

    def _deduplicate_relation_list(self, relations: List[Relation]) -> List[Relation]:
        """Deduplicate list of relations."""
        seen = {}
        deduplicated = []

        for relation in relations:
            # Create key based on source, target, and type
            key = (
                relation.source_entity_id,
                relation.target_entity_id,
                relation.type.lower(),
            )

            if key not in seen:
                seen[key] = relation
                deduplicated.append(relation)
            else:
                # Keep relation with higher confidence
                if relation.confidence > seen[key].confidence:
                    # Replace in deduplicated list
                    for i, existing in enumerate(deduplicated):
                        if existing is seen[key]:
                            deduplicated[i] = relation
                            seen[key] = relation
                            break

        return deduplicated
