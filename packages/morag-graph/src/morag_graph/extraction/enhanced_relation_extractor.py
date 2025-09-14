"""Enhanced relation extractor with iterative refinement and validation."""

import structlog
import asyncio
import time
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass

try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False
    lx = None

from ..models import Entity, Relation
from .relation_extractor import RelationExtractor
from .enhanced_entity_extractor import ConfidenceEntity
from ..utils.retry_utils import retry_with_exponential_backoff

logger = structlog.get_logger(__name__)


@dataclass
class RelationCandidate:
    """Candidate relation with validation metadata."""
    relation: Relation
    confidence: float
    extraction_round: int
    validation_score: float
    context_evidence: str
    
    def __post_init__(self):
        """Update relation confidence with calculated confidence."""
        self.relation.confidence = self.confidence


@dataclass
class RelationValidationResult:
    """Result of relation validation."""
    is_valid: bool
    confidence: float
    issues: List[str]
    evidence_strength: float


class RelationValidator:
    """Validate relations using multiple validation models."""
    
    def __init__(self):
        self.logger = logger.bind(component="relation_validator")
        self.validation_models = {
            'semantic': self._validate_semantic,
            'temporal': self._validate_temporal,
            'causal': self._validate_causal,
            'spatial': self._validate_spatial
        }
    
    async def validate_relation_with_context(
        self,
        relation: Relation,
        context_text: str,
        source_entity: Optional[Entity] = None,
        target_entity: Optional[Entity] = None
    ) -> RelationValidationResult:
        """Validate relation using multiple validation models."""
        validation_results = {}
        
        # Run validation models
        for model_name, validator in self.validation_models.items():
            try:
                result = await validator(relation, context_text, source_entity, target_entity)
                validation_results[model_name] = result
            except Exception as e:
                self.logger.warning(f"Validation model {model_name} failed: {e}")
                validation_results[model_name] = RelationValidationResult(
                    is_valid=False, confidence=0.0, issues=[str(e)], evidence_strength=0.0
                )
        
        # Combine validation results
        return self._combine_validation_results(validation_results, relation)
    
    async def _validate_semantic(
        self,
        relation: Relation,
        context_text: str,
        source_entity: Optional[Entity],
        target_entity: Optional[Entity]
    ) -> RelationValidationResult:
        """Validate semantic consistency of relation."""
        issues = []
        confidence = 0.8  # Base confidence
        
        # Check if relation type makes sense for entity types
        if source_entity and target_entity:
            type_compatibility = self._check_type_compatibility(
                relation.type, source_entity.type, target_entity.type
            )
            if not type_compatibility:
                issues.append(f"Relation type '{relation.type}' incompatible with entity types")
                confidence *= 0.5
        
        # Check if entities appear in context
        context_lower = context_text.lower()
        if source_entity and source_entity.name.lower() not in context_lower:
            issues.append("Source entity not found in context")
            confidence *= 0.7
        
        if target_entity and target_entity.name.lower() not in context_lower:
            issues.append("Target entity not found in context")
            confidence *= 0.7
        
        # Check relation description quality
        if not relation.description or len(relation.description.strip()) < 10:
            issues.append("Relation description too brief or missing")
            confidence *= 0.8
        
        evidence_strength = self._calculate_evidence_strength(relation, context_text)
        
        return RelationValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            evidence_strength=evidence_strength
        )
    
    async def _validate_temporal(
        self,
        relation: Relation,
        context_text: str,
        source_entity: Optional[Entity],
        target_entity: Optional[Entity]
    ) -> RelationValidationResult:
        """Validate temporal aspects of relation."""
        issues = []
        confidence = 0.9
        
        # Check for temporal indicators in relation type
        temporal_types = ['before', 'after', 'during', 'founded', 'created', 'established']
        is_temporal = any(temp in relation.type.lower() for temp in temporal_types)
        
        if is_temporal:
            # Look for temporal evidence in context
            temporal_indicators = ['in', 'on', 'during', 'before', 'after', 'since', 'until']
            has_temporal_context = any(
                indicator in context_text.lower() for indicator in temporal_indicators
            )
            
            if not has_temporal_context:
                issues.append("Temporal relation lacks temporal context evidence")
                confidence *= 0.6
        
        return RelationValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            evidence_strength=0.7 if is_temporal else 0.9
        )
    
    async def _validate_causal(
        self,
        relation: Relation,
        context_text: str,
        source_entity: Optional[Entity],
        target_entity: Optional[Entity]
    ) -> RelationValidationResult:
        """Validate causal relationships."""
        issues = []
        confidence = 0.9
        
        # Check for causal indicators
        causal_types = ['causes', 'leads_to', 'results_in', 'enables', 'prevents']
        is_causal = any(causal in relation.type.lower() for causal in causal_types)
        
        if is_causal:
            # Look for causal evidence
            causal_indicators = ['because', 'due to', 'results in', 'leads to', 'causes']
            has_causal_context = any(
                indicator in context_text.lower() for indicator in causal_indicators
            )
            
            if not has_causal_context:
                issues.append("Causal relation lacks causal evidence in context")
                confidence *= 0.7
        
        return RelationValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            evidence_strength=0.8 if is_causal else 0.9
        )
    
    async def _validate_spatial(
        self,
        relation: Relation,
        context_text: str,
        source_entity: Optional[Entity],
        target_entity: Optional[Entity]
    ) -> RelationValidationResult:
        """Validate spatial relationships."""
        issues = []
        confidence = 0.9
        
        # Check for spatial indicators
        spatial_types = ['located_in', 'near', 'adjacent_to', 'contains', 'part_of']
        is_spatial = any(spatial in relation.type.lower() for spatial in spatial_types)
        
        if is_spatial:
            # Check if at least one entity is a location
            has_location = False
            if source_entity and 'location' in source_entity.type.lower():
                has_location = True
            if target_entity and 'location' in target_entity.type.lower():
                has_location = True
            
            if not has_location:
                issues.append("Spatial relation should involve at least one location entity")
                confidence *= 0.6
        
        return RelationValidationResult(
            is_valid=len(issues) == 0,
            confidence=confidence,
            issues=issues,
            evidence_strength=0.8 if is_spatial else 0.9
        )
    
    def _check_type_compatibility(
        self,
        relation_type: str,
        source_type: str,
        target_type: str
    ) -> bool:
        """Check if relation type is compatible with entity types."""
        relation_lower = relation_type.lower()
        source_lower = source_type.lower()
        target_lower = target_type.lower()
        
        # Define compatibility rules
        compatibility_rules = {
            'works_for': ('person', 'organization'),
            'located_in': (['person', 'organization', 'object'], 'location'),
            'founded': ('person', 'organization'),
            'created': ('person', ['object', 'concept', 'technology']),
            'part_of': (['object', 'concept'], ['object', 'concept', 'organization']),
            'uses': (['person', 'organization'], ['technology', 'object', 'method'])
        }
        
        for rel_pattern, (source_patterns, target_patterns) in compatibility_rules.items():
            if rel_pattern in relation_lower:
                # Convert to lists for uniform handling
                if isinstance(source_patterns, str):
                    source_patterns = [source_patterns]
                if isinstance(target_patterns, str):
                    target_patterns = [target_patterns]
                
                source_match = any(pattern in source_lower for pattern in source_patterns)
                target_match = any(pattern in target_lower for pattern in target_patterns)
                
                return source_match and target_match
        
        # Default to compatible if no specific rule
        return True
    
    def _calculate_evidence_strength(self, relation: Relation, context_text: str) -> float:
        """Calculate evidence strength for relation in context."""
        # Simple implementation based on description quality and context presence
        base_strength = 0.5
        
        # Bonus for detailed description
        if relation.description and len(relation.description) > 20:
            base_strength += 0.2
        
        # Bonus for relation type appearing in context
        if relation.type.lower().replace('_', ' ') in context_text.lower():
            base_strength += 0.2
        
        # Bonus for high confidence
        if relation.confidence > 0.8:
            base_strength += 0.1
        
        return min(1.0, base_strength)
    
    def _combine_validation_results(
        self,
        validation_results: Dict[str, RelationValidationResult],
        relation: Relation
    ) -> RelationValidationResult:
        """Combine validation results from multiple models."""
        all_issues = []
        confidences = []
        evidence_strengths = []
        
        for model_name, result in validation_results.items():
            all_issues.extend([f"{model_name}: {issue}" for issue in result.issues])
            confidences.append(result.confidence)
            evidence_strengths.append(result.evidence_strength)
        
        # Calculate combined confidence (weighted average)
        combined_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Calculate combined evidence strength
        combined_evidence = sum(evidence_strengths) / len(evidence_strengths) if evidence_strengths else 0.0
        
        # Relation is valid if no critical issues and confidence above threshold
        is_valid = len(all_issues) == 0 and combined_confidence >= 0.6
        
        return RelationValidationResult(
            is_valid=is_valid,
            confidence=combined_confidence,
            issues=all_issues,
            evidence_strength=combined_evidence
        )


class EnhancedRelationExtractor:
    """Enhanced relation extractor with iterative refinement and validation."""
    
    def __init__(
        self,
        base_extractor: Optional[RelationExtractor] = None,
        max_rounds: int = 2,
        confidence_threshold: float = 0.7,
        enable_validation: bool = True
    ):
        """Initialize enhanced relation extractor.
        
        Args:
            base_extractor: Base RelationExtractor instance
            max_rounds: Maximum extraction rounds
            confidence_threshold: Minimum confidence for relations
            enable_validation: Whether to enable relation validation
        """
        self.base_extractor = base_extractor or RelationExtractor()
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold
        self.enable_validation = enable_validation
        
        self.validator = RelationValidator() if enable_validation else None
        self.logger = logger.bind(component="enhanced_relation_extractor")
    
    async def extract_with_gleaning(
        self,
        text: str,
        entities: List[Entity],
        source_doc_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations with iterative refinement."""
        if not entities:
            return []
        
        start_time = time.time()
        all_relations: List[RelationCandidate] = []
        
        self.logger.info(
            "Starting enhanced relation extraction",
            max_rounds=self.max_rounds,
            entities=len(entities)
        )
        
        for round_num in range(self.max_rounds):
            round_start = time.time()
            
            self.logger.info(f"Extraction round {round_num + 1}/{self.max_rounds}")
            
            # Extract relations for this round
            if round_num == 0:
                # First round: standard extraction
                round_relations = await self.base_extractor.extract(
                    text, entities, source_doc_id
                )
            else:
                # Subsequent rounds: focused extraction on missed patterns
                round_relations = await self._extract_missed_relations(
                    text, entities, all_relations, source_doc_id
                )
            
            # Validate and score relations
            validated_relations = []
            for relation in round_relations:
                if self.enable_validation:
                    validation_result = await self._validate_relation(
                        relation, text, entities
                    )
                    
                    if validation_result.is_valid and validation_result.confidence >= self.confidence_threshold:
                        validated_relations.append(RelationCandidate(
                            relation=relation,
                            confidence=validation_result.confidence,
                            extraction_round=round_num + 1,
                            validation_score=validation_result.evidence_strength,
                            context_evidence=text[:200] + "..."
                        ))
                else:
                    # No validation - use original confidence
                    if relation.confidence >= self.confidence_threshold:
                        validated_relations.append(RelationCandidate(
                            relation=relation,
                            confidence=relation.confidence,
                            extraction_round=round_num + 1,
                            validation_score=0.8,
                            context_evidence=text[:200] + "..."
                        ))
            
            # Add new relations (with deduplication)
            new_relations = self._deduplicate_relations(all_relations + validated_relations)
            new_count = len(new_relations) - len(all_relations)
            all_relations = new_relations
            
            round_time = time.time() - round_start
            
            self.logger.info(
                f"Round {round_num + 1} completed",
                new_relations=new_count,
                total_relations=len(all_relations),
                processing_time=f"{round_time:.2f}s"
            )
            
            # Stop if no new relations found
            if new_count == 0:
                self.logger.info("No new relations found, stopping extraction")
                break
        
        total_time = time.time() - start_time
        final_relations = [rc.relation for rc in all_relations]
        
        self.logger.info(
            "Enhanced relation extraction completed",
            total_relations=len(final_relations),
            total_time=f"{total_time:.2f}s"
        )
        
        return final_relations

    async def _extract_missed_relations(
        self,
        text: str,
        entities: List[Entity],
        existing_relations: List[RelationCandidate],
        source_doc_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations that might have been missed in previous rounds."""
        # Analyze existing relations to identify patterns
        existing_pairs = set()
        existing_types = set()

        for rel_candidate in existing_relations:
            rel = rel_candidate.relation
            existing_pairs.add((rel.source_entity_id, rel.target_entity_id))
            existing_types.add(rel.type.lower())

        # Create entity pairs that haven't been explored
        unexplored_pairs = []
        entity_dict = {e.id: e for e in entities}

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                pair1 = (entity1.id, entity2.id)
                pair2 = (entity2.id, entity1.id)

                if pair1 not in existing_pairs and pair2 not in existing_pairs:
                    unexplored_pairs.append((entity1, entity2))

        if not unexplored_pairs:
            return []

        # Focus extraction on unexplored pairs
        focused_text = self._create_focused_extraction_context(
            text, unexplored_pairs[:10]  # Limit to avoid overwhelming
        )

        # Extract with focused context
        return await self.base_extractor.extract(focused_text, entities, source_doc_id)

    def _create_focused_extraction_context(
        self,
        text: str,
        entity_pairs: List[Tuple[Entity, Entity]]
    ) -> str:
        """Create focused context for extracting specific entity pairs."""
        pair_descriptions = []
        for entity1, entity2 in entity_pairs:
            pair_descriptions.append(f"{entity1.name} and {entity2.name}")

        context = f"Focus on relationships between: {', '.join(pair_descriptions[:5])}\n\n{text}"
        return context

    async def _validate_relation(
        self,
        relation: Relation,
        context_text: str,
        entities: List[Entity]
    ) -> RelationValidationResult:
        """Validate a single relation."""
        if not self.validator:
            # Return default validation result
            return RelationValidationResult(
                is_valid=True,
                confidence=relation.confidence,
                issues=[],
                evidence_strength=0.8
            )

        # Find source and target entities
        source_entity = next(
            (e for e in entities if e.id == relation.source_entity_id), None
        )
        target_entity = next(
            (e for e in entities if e.id == relation.target_entity_id), None
        )

        return await self.validator.validate_relation_with_context(
            relation, context_text, source_entity, target_entity
        )

    def _deduplicate_relations(
        self,
        relations: List[RelationCandidate]
    ) -> List[RelationCandidate]:
        """Deduplicate relations by source, target, and type."""
        seen = {}
        deduplicated = []

        for rel_candidate in relations:
            rel = rel_candidate.relation
            key = (rel.source_entity_id, rel.target_entity_id, rel.type.lower())

            if key not in seen:
                seen[key] = rel_candidate
                deduplicated.append(rel_candidate)
            else:
                # Keep relation with higher confidence
                if rel_candidate.confidence > seen[key].confidence:
                    # Replace in deduplicated list
                    for i, existing in enumerate(deduplicated):
                        if existing is seen[key]:
                            deduplicated[i] = rel_candidate
                            seen[key] = rel_candidate
                            break

        return deduplicated

    # Compatibility methods to maintain interface
    async def extract(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        source_doc_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations (compatibility method)."""
        if not entities:
            return []
        return await self.extract_with_gleaning(text, entities, source_doc_id)

    @property
    def min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return self.confidence_threshold

    @property
    def domain(self) -> str:
        """Get extraction domain."""
        return self.base_extractor.domain

    @property
    def language(self) -> Optional[str]:
        """Get extraction language."""
        return self.base_extractor.language
