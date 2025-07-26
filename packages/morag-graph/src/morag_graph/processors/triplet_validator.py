"""Comprehensive triplet validation with quality assessment and relevance scoring."""

import asyncio
import re
from typing import List, Dict, Any, Optional, NamedTuple, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from ..normalizers.entity_linker import LinkedTriplet
from ..normalizers.confidence_manager import ConfidenceScore, ConfidenceLevel
from ..normalizers.relationship_categorizer import CategorizedRelationship, RelationshipCategory

logger = structlog.get_logger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"
    VERY_STRICT = "very_strict"


class ValidationResult(Enum):
    """Validation result types."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    REJECTED = "rejected"


@dataclass
class ValidationRule:
    """Represents a validation rule with metadata."""
    name: str
    description: str
    category: str
    severity: str  # "error", "warning", "info"
    enabled: bool = True
    weight: float = 1.0


class QualityScore(NamedTuple):
    """Comprehensive quality score for a triplet."""
    overall_score: float
    component_scores: Dict[str, float]
    validation_result: ValidationResult
    validation_issues: List[str]
    quality_flags: Set[str]
    relevance_score: float
    metadata: Dict[str, Any] = {}


class ValidatedTriplet(NamedTuple):
    """Represents a fully validated triplet with quality assessment."""
    triplet: LinkedTriplet
    confidence_score: ConfidenceScore
    categorized_relationship: Optional[CategorizedRelationship]
    quality_score: QualityScore
    validation_level: ValidationLevel
    passed_validation: bool
    rejection_reason: Optional[str] = None


class TripletValidator:
    """Comprehensive triplet validator with quality assessment and relevance scoring."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize triplet validator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Configuration
        self.validation_level = ValidationLevel(self.config.get('validation_level', 'standard'))
        self.min_overall_quality = self.config.get('min_overall_quality', 0.6)
        self.min_relevance_score = self.config.get('min_relevance_score', 0.5)
        self.enable_semantic_validation = self.config.get('enable_semantic_validation', True)
        self.enable_domain_validation = self.config.get('enable_domain_validation', False)
        self.reject_low_quality = self.config.get('reject_low_quality', True)
        
        # Validation rules
        self.validation_rules = self._build_validation_rules()
        
        # Quality thresholds by validation level
        self.quality_thresholds = {
            ValidationLevel.PERMISSIVE: 0.3,
            ValidationLevel.STANDARD: 0.6,
            ValidationLevel.STRICT: 0.8,
            ValidationLevel.VERY_STRICT: 0.9
        }
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="triplet_validator")
        
        # Compiled patterns for validation
        self._validation_patterns = self._compile_validation_patterns()
        
        logger.info(
            "Triplet validator initialized",
            validation_level=self.validation_level.value,
            min_overall_quality=self.min_overall_quality,
            min_relevance_score=self.min_relevance_score,
            enable_semantic_validation=self.enable_semantic_validation
        )
    
    def _build_validation_rules(self) -> Dict[str, ValidationRule]:
        """Build validation rules based on configuration."""
        rules = {
            # Entity validation rules
            "non_empty_entities": ValidationRule(
                name="non_empty_entities",
                description="Subject and object must not be empty",
                category="entity",
                severity="error",
                weight=1.0
            ),
            "entity_length_check": ValidationRule(
                name="entity_length_check",
                description="Entities must be within reasonable length limits",
                category="entity",
                severity="warning",
                weight=0.8
            ),
            "entity_character_validation": ValidationRule(
                name="entity_character_validation",
                description="Entities should not contain invalid characters",
                category="entity",
                severity="warning",
                weight=0.6
            ),
            
            # Predicate validation rules
            "non_empty_predicate": ValidationRule(
                name="non_empty_predicate",
                description="Predicate must not be empty",
                category="predicate",
                severity="error",
                weight=1.0
            ),
            "meaningful_predicate": ValidationRule(
                name="meaningful_predicate",
                description="Predicate should be meaningful and not just filler words",
                category="predicate",
                severity="warning",
                weight=0.7
            ),
            "predicate_length_check": ValidationRule(
                name="predicate_length_check",
                description="Predicate must be within reasonable length limits",
                category="predicate",
                severity="warning",
                weight=0.5
            ),
            
            # Semantic validation rules
            "subject_object_different": ValidationRule(
                name="subject_object_different",
                description="Subject and object should be different",
                category="semantic",
                severity="warning",
                weight=0.6
            ),
            "semantic_coherence": ValidationRule(
                name="semantic_coherence",
                description="Triplet should be semantically coherent",
                category="semantic",
                severity="warning",
                weight=0.8,
                enabled=self.enable_semantic_validation
            ),
            
            # Quality validation rules
            "confidence_threshold": ValidationRule(
                name="confidence_threshold",
                description="Triplet confidence must meet minimum threshold",
                category="quality",
                severity="error",
                weight=1.0
            ),
            "entity_linking_quality": ValidationRule(
                name="entity_linking_quality",
                description="Entity linking should meet quality standards",
                category="quality",
                severity="warning",
                weight=0.7
            ),
            "sentence_quality": ValidationRule(
                name="sentence_quality",
                description="Source sentence should be of good quality",
                category="quality",
                severity="info",
                weight=0.4
            ),
            
            # Relevance validation rules
            "domain_relevance": ValidationRule(
                name="domain_relevance",
                description="Triplet should be relevant to the domain",
                category="relevance",
                severity="warning",
                weight=0.6,
                enabled=self.enable_domain_validation
            ),
            "information_value": ValidationRule(
                name="information_value",
                description="Triplet should provide meaningful information",
                category="relevance",
                severity="warning",
                weight=0.8
            )
        }
        
        return rules
    
    def _compile_validation_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for validation."""
        return {
            # Invalid characters in entities
            'invalid_entity_chars': re.compile(r'[<>{}[\]\\|`~@#$%^&*()+=]'),
            
            # Only punctuation or numbers
            'only_punctuation': re.compile(r'^[^\w\s]+$'),
            'only_numbers': re.compile(r'^\d+$'),
            
            # Meaningless predicates
            'meaningless_predicates': re.compile(r'^\s*(?:of|in|on|at|by|for|with|from|to|and|or|but|the|a|an)\s*$', re.IGNORECASE),
            
            # URLs and emails (usually not meaningful entities)
            'url_pattern': re.compile(r'https?://|www\.'),
            'email_pattern': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # Common noise patterns
            'noise_patterns': re.compile(r'^\s*(?:click|here|more|info|link|page|site|website|home|back|next|prev|previous)\s*$', re.IGNORECASE),
        }
    
    async def validate_triplets(
        self, 
        triplets: List[LinkedTriplet],
        confidence_scores: Optional[List[ConfidenceScore]] = None,
        categorized_relationships: Optional[List[CategorizedRelationship]] = None,
        source_doc_id: Optional[str] = None
    ) -> List[ValidatedTriplet]:
        """Validate triplets with comprehensive quality assessment.
        
        Args:
            triplets: List of linked triplets to validate
            confidence_scores: Optional list of confidence scores (same length as triplets)
            categorized_relationships: Optional list of categorized relationships
            source_doc_id: Optional source document ID
            
        Returns:
            List of validated triplets with quality scores
            
        Raises:
            ProcessingError: If validation fails
        """
        if not triplets:
            return []
        
        try:
            logger.debug(
                "Starting triplet validation",
                triplet_count=len(triplets),
                validation_level=self.validation_level.value,
                source_doc_id=source_doc_id
            )
            
            # Validate triplets
            validated_triplets = []
            for i, triplet in enumerate(triplets):
                try:
                    confidence_score = confidence_scores[i] if confidence_scores and i < len(confidence_scores) else None
                    categorized_rel = categorized_relationships[i] if categorized_relationships and i < len(categorized_relationships) else None
                    
                    validated = await self._validate_single_triplet(
                        triplet, confidence_score, categorized_rel, source_doc_id
                    )
                    if validated:
                        validated_triplets.append(validated)
                except Exception as e:
                    logger.warning(
                        "Failed to validate triplet",
                        triplet=f"{triplet.subject} | {triplet.predicate} | {triplet.object}",
                        error=str(e)
                    )
            
            # Filter by validation results if configured
            if self.reject_low_quality:
                filtered_triplets = [
                    t for t in validated_triplets 
                    if t.passed_validation and t.quality_score.validation_result != ValidationResult.REJECTED
                ]
            else:
                filtered_triplets = validated_triplets
            
            logger.info(
                "Triplet validation completed",
                input_triplets=len(triplets),
                validated_triplets=len(validated_triplets),
                passed_validation=len([t for t in validated_triplets if t.passed_validation]),
                filtered_triplets=len(filtered_triplets),
                validation_level=self.validation_level.value,
                source_doc_id=source_doc_id
            )
            
            return filtered_triplets
            
        except Exception as e:
            logger.error(
                "Triplet validation failed",
                error=str(e),
                error_type=type(e).__name__,
                triplet_count=len(triplets),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"Triplet validation failed: {e}")
    
    async def _validate_single_triplet(
        self, 
        triplet: LinkedTriplet,
        confidence_score: Optional[ConfidenceScore] = None,
        categorized_relationship: Optional[CategorizedRelationship] = None,
        source_doc_id: Optional[str] = None
    ) -> Optional[ValidatedTriplet]:
        """Validate a single triplet."""
        def validate_sync():
            # Calculate quality score
            quality_score = self._calculate_quality_score(triplet, confidence_score, categorized_relationship)
            
            # Determine if triplet passes validation
            threshold = self.quality_thresholds[self.validation_level]
            passed_validation = (
                quality_score.overall_score >= threshold and
                quality_score.validation_result in [ValidationResult.VALID, ValidationResult.WARNING]
            )
            
            # Determine rejection reason if applicable
            rejection_reason = None
            if not passed_validation:
                if quality_score.validation_result == ValidationResult.REJECTED:
                    rejection_reason = "Failed critical validation rules"
                elif quality_score.overall_score < threshold:
                    rejection_reason = f"Quality score {quality_score.overall_score:.2f} below threshold {threshold:.2f}"
                else:
                    rejection_reason = f"Validation result: {quality_score.validation_result.value}"
            
            return ValidatedTriplet(
                triplet=triplet,
                confidence_score=confidence_score or ConfidenceScore(
                    overall_score=triplet.confidence,
                    component_scores={'extraction_confidence': triplet.confidence},
                    confidence_level=ConfidenceLevel.MEDIUM,
                    quality_flags=set()
                ),
                categorized_relationship=categorized_relationship,
                quality_score=quality_score,
                validation_level=self.validation_level,
                passed_validation=passed_validation,
                rejection_reason=rejection_reason
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, validate_sync
        )
    
    def _calculate_quality_score(
        self, 
        triplet: LinkedTriplet,
        confidence_score: Optional[ConfidenceScore] = None,
        categorized_relationship: Optional[CategorizedRelationship] = None
    ) -> QualityScore:
        """Calculate comprehensive quality score for a triplet."""
        component_scores = {}
        validation_issues = []
        quality_flags = set()
        
        # Entity validation
        entity_score, entity_issues, entity_flags = self._validate_entities(triplet)
        component_scores['entity_quality'] = entity_score
        validation_issues.extend(entity_issues)
        quality_flags.update(entity_flags)
        
        # Predicate validation
        predicate_score, predicate_issues, predicate_flags = self._validate_predicate(triplet)
        component_scores['predicate_quality'] = predicate_score
        validation_issues.extend(predicate_issues)
        quality_flags.update(predicate_flags)
        
        # Semantic validation
        semantic_score, semantic_issues, semantic_flags = self._validate_semantics(triplet)
        component_scores['semantic_quality'] = semantic_score
        validation_issues.extend(semantic_issues)
        quality_flags.update(semantic_flags)
        
        # Confidence integration
        if confidence_score:
            component_scores['confidence_quality'] = confidence_score.overall_score
            quality_flags.update(confidence_score.quality_flags)
        else:
            component_scores['confidence_quality'] = triplet.confidence
        
        # Relationship categorization quality
        if categorized_relationship:
            component_scores['categorization_quality'] = categorized_relationship.semantic_weight
            quality_flags.add(f"category_{categorized_relationship.relationship_category.value}")
        else:
            component_scores['categorization_quality'] = 0.5
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(triplet, categorized_relationship)
        component_scores['relevance_score'] = relevance_score
        
        # Calculate overall score (weighted average)
        weights = {
            'entity_quality': 0.25,
            'predicate_quality': 0.20,
            'semantic_quality': 0.15,
            'confidence_quality': 0.20,
            'categorization_quality': 0.10,
            'relevance_score': 0.10
        }
        
        overall_score = sum(
            component_scores.get(component, 0.0) * weight
            for component, weight in weights.items()
        )
        
        # Determine validation result
        validation_result = self._determine_validation_result(overall_score, validation_issues)
        
        return QualityScore(
            overall_score=overall_score,
            component_scores=component_scores,
            validation_result=validation_result,
            validation_issues=validation_issues,
            quality_flags=quality_flags,
            relevance_score=relevance_score,
            metadata={
                'validation_level': self.validation_level.value,
                'rules_applied': len([r for r in self.validation_rules.values() if r.enabled]),
                'source_doc_id': triplet.source_doc_id
            }
        )
    
    def _validate_entities(self, triplet: LinkedTriplet) -> Tuple[float, List[str], Set[str]]:
        """Validate entities in the triplet."""
        score = 1.0
        issues = []
        flags = set()
        
        # Check non-empty entities
        if not triplet.subject or not triplet.subject.strip():
            score -= 0.5
            issues.append("Empty subject")
            flags.add("empty_subject")
        
        if not triplet.object or not triplet.object.strip():
            score -= 0.5
            issues.append("Empty object")
            flags.add("empty_object")
        
        # Check entity lengths
        if len(triplet.subject) > 100:
            score -= 0.2
            issues.append("Subject too long")
            flags.add("long_subject")
        
        if len(triplet.object) > 100:
            score -= 0.2
            issues.append("Object too long")
            flags.add("long_object")
        
        # Check for invalid characters
        if self._validation_patterns['invalid_entity_chars'].search(triplet.subject):
            score -= 0.3
            issues.append("Subject contains invalid characters")
            flags.add("invalid_subject_chars")
        
        if self._validation_patterns['invalid_entity_chars'].search(triplet.object):
            score -= 0.3
            issues.append("Object contains invalid characters")
            flags.add("invalid_object_chars")
        
        # Check for noise patterns
        if self._validation_patterns['noise_patterns'].match(triplet.subject):
            score -= 0.4
            issues.append("Subject appears to be noise")
            flags.add("noisy_subject")
        
        if self._validation_patterns['noise_patterns'].match(triplet.object):
            score -= 0.4
            issues.append("Object appears to be noise")
            flags.add("noisy_object")
        
        # Bonus for entity linking
        if triplet.subject_entity:
            score += 0.1
            flags.add("subject_linked")
        
        if triplet.object_entity:
            score += 0.1
            flags.add("object_linked")
        
        return max(0.0, min(1.0, score)), issues, flags

    def _validate_predicate(self, triplet: LinkedTriplet) -> Tuple[float, List[str], Set[str]]:
        """Validate predicate in the triplet."""
        score = 1.0
        issues = []
        flags = set()

        # Check non-empty predicate
        if not triplet.predicate or not triplet.predicate.strip():
            score -= 0.8
            issues.append("Empty predicate")
            flags.add("empty_predicate")
            return max(0.0, score), issues, flags

        # Check for meaningless predicates
        if self._validation_patterns['meaningless_predicates'].match(triplet.predicate):
            score -= 0.6
            issues.append("Meaningless predicate")
            flags.add("meaningless_predicate")

        # Check predicate length
        if len(triplet.predicate) > 50:
            score -= 0.3
            issues.append("Predicate too long")
            flags.add("long_predicate")
        elif len(triplet.predicate) < 2:
            score -= 0.4
            issues.append("Predicate too short")
            flags.add("short_predicate")

        # Check for only punctuation
        if self._validation_patterns['only_punctuation'].match(triplet.predicate):
            score -= 0.7
            issues.append("Predicate is only punctuation")
            flags.add("punctuation_only_predicate")

        # Bonus for good predicates
        good_predicate_indicators = ['is', 'are', 'has', 'have', 'works', 'creates', 'manages', 'teaches', 'studies']
        if any(indicator in triplet.predicate.lower() for indicator in good_predicate_indicators):
            score += 0.1
            flags.add("good_predicate")

        return max(0.0, min(1.0, score)), issues, flags

    def _validate_semantics(self, triplet: LinkedTriplet) -> Tuple[float, List[str], Set[str]]:
        """Validate semantic coherence of the triplet."""
        score = 1.0
        issues = []
        flags = set()

        # Check if subject and object are different
        if triplet.subject.lower().strip() == triplet.object.lower().strip():
            score -= 0.4
            issues.append("Subject and object are identical")
            flags.add("identical_subject_object")

        # Check for semantic coherence (basic heuristics)
        if self.enable_semantic_validation:
            # Check for proper noun patterns
            if any(word[0].isupper() for word in triplet.subject.split()):
                flags.add("proper_noun_subject")
                score += 0.05

            if any(word[0].isupper() for word in triplet.object.split()):
                flags.add("proper_noun_object")
                score += 0.05

            # Check for reasonable entity types
            if self._validation_patterns['only_numbers'].match(triplet.subject):
                score -= 0.2
                issues.append("Subject is only numbers")
                flags.add("numeric_subject")

            if self._validation_patterns['only_numbers'].match(triplet.object):
                score -= 0.2
                issues.append("Object is only numbers")
                flags.add("numeric_object")

        return max(0.0, min(1.0, score)), issues, flags

    def _calculate_relevance_score(
        self,
        triplet: LinkedTriplet,
        categorized_relationship: Optional[CategorizedRelationship] = None
    ) -> float:
        """Calculate relevance score for the triplet."""
        relevance = 0.5  # Base relevance

        # Boost for entity linking
        if triplet.subject_entity or triplet.object_entity:
            relevance += 0.2

        # Boost for high-value relationship categories
        if categorized_relationship:
            high_value_categories = {
                RelationshipCategory.STRUCTURAL,
                RelationshipCategory.FUNCTIONAL,
                RelationshipCategory.CAUSAL
            }
            if categorized_relationship.relationship_category in high_value_categories:
                relevance += 0.2

            # Use semantic weight from categorization
            relevance += categorized_relationship.semantic_weight * 0.1

        # Boost for proper nouns (likely important entities)
        if any(word[0].isupper() for word in triplet.subject.split()):
            relevance += 0.1

        if any(word[0].isupper() for word in triplet.object.split()):
            relevance += 0.1

        # Penalty for very short or very long sentences
        sentence_length = len(triplet.sentence)
        if sentence_length < 20:
            relevance -= 0.1
        elif sentence_length > 200:
            relevance -= 0.05

        return max(0.0, min(1.0, relevance))

    def _determine_validation_result(self, overall_score: float, validation_issues: List[str]) -> ValidationResult:
        """Determine validation result based on score and issues."""
        # Check for critical issues
        critical_issues = [
            "Empty subject", "Empty object", "Empty predicate",
            "Subject is only punctuation", "Object is only punctuation"
        ]

        if any(issue in validation_issues for issue in critical_issues):
            return ValidationResult.REJECTED

        # Determine result based on score and validation level
        threshold = self.quality_thresholds[self.validation_level]

        if overall_score >= threshold:
            if len(validation_issues) == 0:
                return ValidationResult.VALID
            else:
                return ValidationResult.WARNING
        else:
            return ValidationResult.INVALID

    def get_validation_statistics(self, validated_triplets: List[ValidatedTriplet]) -> Dict[str, Any]:
        """Get validation statistics for a set of validated triplets."""
        if not validated_triplets:
            return {}

        # Count by validation result
        result_counts = {}
        quality_scores = []
        relevance_scores = []
        passed_count = 0

        for triplet in validated_triplets:
            result = triplet.quality_score.validation_result.value
            result_counts[result] = result_counts.get(result, 0) + 1
            quality_scores.append(triplet.quality_score.overall_score)
            relevance_scores.append(triplet.quality_score.relevance_score)

            if triplet.passed_validation:
                passed_count += 1

        # Aggregate quality flags
        all_flags = set()
        flag_counts = {}
        for triplet in validated_triplets:
            flags = triplet.quality_score.quality_flags
            all_flags.update(flags)
            for flag in flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        # Component score averages
        component_averages = {}
        if validated_triplets:
            first_triplet = validated_triplets[0]
            for component in first_triplet.quality_score.component_scores.keys():
                scores = [
                    t.quality_score.component_scores.get(component, 0.0)
                    for t in validated_triplets
                ]
                component_averages[component] = sum(scores) / len(scores)

        return {
            "total_triplets": len(validated_triplets),
            "passed_validation": passed_count,
            "validation_pass_rate": passed_count / len(validated_triplets),
            "validation_result_distribution": result_counts,
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "min_quality_score": min(quality_scores),
            "max_quality_score": max(quality_scores),
            "average_relevance_score": sum(relevance_scores) / len(relevance_scores),
            "component_score_averages": component_averages,
            "quality_flag_distribution": flag_counts,
            "validation_level": self.validation_level.value,
            "quality_threshold": self.quality_thresholds[self.validation_level]
        }

    def get_validation_rules_info(self) -> Dict[str, Any]:
        """Get information about validation rules."""
        rules_info = {}

        for rule_name, rule in self.validation_rules.items():
            rules_info[rule_name] = {
                "description": rule.description,
                "category": rule.category,
                "severity": rule.severity,
                "enabled": rule.enabled,
                "weight": rule.weight
            }

        return {
            "total_rules": len(self.validation_rules),
            "enabled_rules": len([r for r in self.validation_rules.values() if r.enabled]),
            "rules_by_category": {
                category: len([r for r in self.validation_rules.values() if r.category == category])
                for category in set(r.category for r in self.validation_rules.values())
            },
            "rules_by_severity": {
                severity: len([r for r in self.validation_rules.values() if r.severity == severity])
                for severity in set(r.severity for r in self.validation_rules.values())
            },
            "rules": rules_info
        }

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("Triplet validator closed")
        except Exception as e:
            logger.warning("Error during triplet validator cleanup", error=str(e))

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
