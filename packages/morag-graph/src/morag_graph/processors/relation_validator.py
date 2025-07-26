"""Enhanced relation validation and scoring framework for MoRAG."""

import asyncio
import re
import time
from typing import List, Dict, Any, Optional, NamedTuple, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError, ValidationError
from ..models import Relation, Entity
from ..normalizers.entity_linker import LinkedTriplet

logger = structlog.get_logger(__name__)

# Optional imports for enhanced functionality
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class RelationQuality(Enum):
    """Relation quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


class ValidationCategory(Enum):
    """Categories of validation checks."""
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    CONTEXTUAL = "contextual"
    DOMAIN = "domain"
    QUALITY = "quality"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    category: ValidationCategory
    severity: str  # "error", "warning", "info"
    message: str
    score_impact: float
    metadata: Dict[str, Any] = None


class RelationScore(NamedTuple):
    """Comprehensive relation scoring."""
    overall_score: float
    quality_level: RelationQuality
    component_scores: Dict[str, float]
    validation_issues: List[ValidationIssue]
    confidence_score: float
    relevance_score: float
    context_score: float
    semantic_coherence: float
    domain_relevance: float
    metadata: Dict[str, Any] = {}


class ValidatedRelation(NamedTuple):
    """Represents a validated relation with comprehensive scoring."""
    relation: Relation
    source_triplet: LinkedTriplet
    relation_score: RelationScore
    passed_validation: bool
    rejection_reason: Optional[str] = None
    processing_time: float = 0.0


class RelationValidator:
    """Enhanced relation validator with comprehensive quality scoring and context preservation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize relation validator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Configuration
        self.min_overall_score = self.config.get('min_overall_score', 0.6)
        self.min_confidence_score = self.config.get('min_confidence_score', 0.5)
        self.min_relevance_score = self.config.get('min_relevance_score', 0.4)
        self.enable_semantic_validation = self.config.get('enable_semantic_validation', SENTENCE_TRANSFORMERS_AVAILABLE)
        self.enable_domain_validation = self.config.get('enable_domain_validation', False)
        self.enable_context_preservation = self.config.get('enable_context_preservation', True)
        self.strict_mode = self.config.get('strict_mode', False)
        
        # Scoring weights
        self.confidence_weight = self.config.get('confidence_weight', 0.3)
        self.relevance_weight = self.config.get('relevance_weight', 0.25)
        self.context_weight = self.config.get('context_weight', 0.2)
        self.semantic_weight = self.config.get('semantic_weight', 0.15)
        self.domain_weight = self.config.get('domain_weight', 0.1)
        
        # Quality thresholds
        self.quality_thresholds = {
            RelationQuality.EXCELLENT: 0.9,
            RelationQuality.GOOD: 0.75,
            RelationQuality.FAIR: 0.6,
            RelationQuality.POOR: 0.4,
            RelationQuality.INVALID: 0.0
        }
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="relation_validator")
        
        # Initialize semantic model if available
        self._semantic_model = None
        if self.enable_semantic_validation and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._init_semantic_model()
        
        # Compiled patterns for validation
        self._validation_patterns = self._compile_validation_patterns()
        
        # Domain-specific validation rules
        self._domain_rules = self._load_domain_rules()
        
        logger.info(
            "Relation validator initialized",
            min_overall_score=self.min_overall_score,
            enable_semantic_validation=self.enable_semantic_validation,
            enable_domain_validation=self.enable_domain_validation,
            enable_context_preservation=self.enable_context_preservation,
            strict_mode=self.strict_mode
        )
    
    def _init_semantic_model(self) -> None:
        """Initialize semantic model for validation."""
        try:
            model_name = self.config.get('semantic_model', 'all-MiniLM-L6-v2')
            self._semantic_model = SentenceTransformer(model_name)
            logger.info(f"Semantic model initialized for relation validation: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic model: {e}")
            self.enable_semantic_validation = False
    
    def _compile_validation_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for relation validation."""
        return {
            # Noise patterns in predicates
            'noise_predicates': re.compile(r'\b(?:is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might)\b', re.IGNORECASE),
            
            # Invalid characters
            'invalid_chars': re.compile(r'[^\w\s\-\.\,\:\;\!\?\'\"]'),
            
            # Repetitive patterns
            'repetitive': re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE),
            
            # Common filler words
            'filler_words': re.compile(r'\b(?:um|uh|like|you know|sort of|kind of|basically|actually|literally)\b', re.IGNORECASE),
            
            # Incomplete predicates
            'incomplete_predicates': re.compile(r'\b(?:and|or|but|because|since|although|however|therefore|thus|hence)\s*$', re.IGNORECASE),
        }
    
    def _load_domain_rules(self) -> Dict[str, Any]:
        """Load domain-specific validation rules."""
        # This could be loaded from configuration files or databases
        return {
            'scientific': {
                'required_patterns': [r'\b(?:study|research|experiment|analysis|investigation)\b'],
                'forbidden_patterns': [r'\b(?:magic|supernatural|mystical)\b'],
                'confidence_boost': 0.1
            },
            'business': {
                'required_patterns': [r'\b(?:company|corporation|business|market|revenue|profit)\b'],
                'forbidden_patterns': [r'\b(?:fictional|imaginary|fantasy)\b'],
                'confidence_boost': 0.05
            },
            'medical': {
                'required_patterns': [r'\b(?:patient|treatment|diagnosis|symptom|disease|medicine)\b'],
                'forbidden_patterns': [r'\b(?:unproven|alternative|homeopathic)\b'],
                'confidence_boost': 0.15
            }
        }
    
    async def validate_relations(
        self, 
        relations: List[Relation],
        source_triplets: Optional[List[LinkedTriplet]] = None,
        context_info: Optional[Dict[str, Any]] = None
    ) -> List[ValidatedRelation]:
        """Validate a list of relations with comprehensive scoring.
        
        Args:
            relations: List of relations to validate
            source_triplets: Optional list of source triplets (same length as relations)
            context_info: Optional context information
            
        Returns:
            List of validated relations with scores
            
        Raises:
            ProcessingError: If validation fails
        """
        if not relations:
            return []
        
        try:
            logger.debug(
                "Starting relation validation",
                relation_count=len(relations),
                has_source_triplets=source_triplets is not None,
                has_context=context_info is not None
            )
            
            start_time = time.time()
            
            # Validate relations in parallel batches
            batch_size = self.config.get('batch_size', 10)
            validated_relations = []
            
            for i in range(0, len(relations), batch_size):
                batch_relations = relations[i:i + batch_size]
                batch_triplets = source_triplets[i:i + batch_size] if source_triplets else [None] * len(batch_relations)
                
                batch_results = await self._validate_relation_batch(
                    batch_relations, batch_triplets, context_info
                )
                validated_relations.extend(batch_results)
            
            processing_time = time.time() - start_time
            
            # Filter results based on validation
            passed_relations = [r for r in validated_relations if r.passed_validation]
            
            logger.info(
                "Relation validation completed",
                total_relations=len(relations),
                passed_relations=len(passed_relations),
                rejection_rate=1.0 - (len(passed_relations) / len(relations)) if relations else 0,
                processing_time=processing_time
            )
            
            return validated_relations
            
        except Exception as e:
            logger.error(
                "Relation validation failed",
                error=str(e),
                error_type=type(e).__name__,
                relation_count=len(relations)
            )
            raise ProcessingError(f"Relation validation failed: {e}")

    async def _validate_relation_batch(
        self,
        relations: List[Relation],
        source_triplets: List[Optional[LinkedTriplet]],
        context_info: Optional[Dict[str, Any]]
    ) -> List[ValidatedRelation]:
        """Validate a batch of relations."""
        def validate_batch_sync():
            results = []
            for relation, triplet in zip(relations, source_triplets):
                start_time = time.time()
                try:
                    validated = self._validate_single_relation(relation, triplet, context_info)
                    processing_time = time.time() - start_time

                    # Update processing time
                    validated = validated._replace(processing_time=processing_time)
                    results.append(validated)

                except Exception as e:
                    processing_time = time.time() - start_time
                    logger.warning(
                        "Failed to validate single relation",
                        relation_id=getattr(relation, 'id', 'unknown'),
                        error=str(e)
                    )

                    # Create failed validation result
                    failed_result = ValidatedRelation(
                        relation=relation,
                        source_triplet=triplet,
                        relation_score=RelationScore(
                            overall_score=0.0,
                            quality_level=RelationQuality.INVALID,
                            component_scores={},
                            validation_issues=[ValidationIssue(
                                category=ValidationCategory.STRUCTURAL,
                                severity="error",
                                message=f"Validation failed: {e}",
                                score_impact=-1.0
                            )],
                            confidence_score=0.0,
                            relevance_score=0.0,
                            context_score=0.0,
                            semantic_coherence=0.0,
                            domain_relevance=0.0
                        ),
                        passed_validation=False,
                        rejection_reason=f"Validation error: {e}",
                        processing_time=processing_time
                    )
                    results.append(failed_result)

            return results

        return await asyncio.get_event_loop().run_in_executor(
            self._executor, validate_batch_sync
        )

    def _validate_single_relation(
        self,
        relation: Relation,
        source_triplet: Optional[LinkedTriplet],
        context_info: Optional[Dict[str, Any]]
    ) -> ValidatedRelation:
        """Validate a single relation with comprehensive scoring."""
        validation_issues = []
        component_scores = {}

        # 1. Structural validation
        structural_score, structural_issues = self._validate_structure(relation)
        component_scores['structural'] = structural_score
        validation_issues.extend(structural_issues)

        # 2. Semantic validation
        semantic_score, semantic_issues = self._validate_semantics(relation, source_triplet)
        component_scores['semantic'] = semantic_score
        validation_issues.extend(semantic_issues)

        # 3. Context validation
        context_score, context_issues = self._validate_context(relation, source_triplet, context_info)
        component_scores['context'] = context_score
        validation_issues.extend(context_issues)

        # 4. Quality assessment
        quality_scores = self._assess_quality(relation, source_triplet)
        component_scores.update(quality_scores)

        # 5. Domain validation (if enabled)
        domain_score = 0.0
        if self.enable_domain_validation:
            domain_score, domain_issues = self._validate_domain(relation, context_info)
            component_scores['domain'] = domain_score
            validation_issues.extend(domain_issues)

        # Calculate overall score
        overall_score = self._calculate_overall_score(component_scores)

        # Determine quality level
        quality_level = self._determine_quality_level(overall_score)

        # Check if relation passes validation
        passed_validation = self._check_validation_pass(overall_score, validation_issues)
        rejection_reason = None if passed_validation else self._get_rejection_reason(validation_issues)

        # Create relation score
        relation_score = RelationScore(
            overall_score=overall_score,
            quality_level=quality_level,
            component_scores=component_scores,
            validation_issues=validation_issues,
            confidence_score=component_scores.get('confidence', 0.0),
            relevance_score=component_scores.get('relevance', 0.0),
            context_score=context_score,
            semantic_coherence=semantic_score,
            domain_relevance=domain_score,
            metadata={
                'validation_timestamp': time.time(),
                'validator_version': '1.0',
                'strict_mode': self.strict_mode
            }
        )

        return ValidatedRelation(
            relation=relation,
            source_triplet=source_triplet,
            relation_score=relation_score,
            passed_validation=passed_validation,
            rejection_reason=rejection_reason
        )

    def _validate_structure(self, relation: Relation) -> Tuple[float, List[ValidationIssue]]:
        """Validate structural aspects of the relation."""
        issues = []
        score = 1.0

        # Check for required fields
        if not relation.type or not relation.type.strip():
            issues.append(ValidationIssue(
                category=ValidationCategory.STRUCTURAL,
                severity="error",
                message="Relation type is empty",
                score_impact=-0.5
            ))
            score -= 0.5

        # Check relation type validity
        if relation.type and len(relation.type) < 2:
            issues.append(ValidationIssue(
                category=ValidationCategory.STRUCTURAL,
                severity="warning",
                message="Relation type is too short",
                score_impact=-0.2
            ))
            score -= 0.2

        # Check for noise patterns in relation type
        if relation.type and self._validation_patterns['noise_predicates'].search(relation.type):
            issues.append(ValidationIssue(
                category=ValidationCategory.STRUCTURAL,
                severity="warning",
                message="Relation type contains noise words",
                score_impact=-0.3
            ))
            score -= 0.3

        # Check for invalid characters
        if relation.type and self._validation_patterns['invalid_chars'].search(relation.type):
            issues.append(ValidationIssue(
                category=ValidationCategory.STRUCTURAL,
                severity="warning",
                message="Relation type contains invalid characters",
                score_impact=-0.2
            ))
            score -= 0.2

        # Check for repetitive patterns
        if relation.type and self._validation_patterns['repetitive'].search(relation.type):
            issues.append(ValidationIssue(
                category=ValidationCategory.STRUCTURAL,
                severity="info",
                message="Relation type contains repetitive patterns",
                score_impact=-0.1
            ))
            score -= 0.1

        return max(0.0, score), issues

    def _validate_semantics(self, relation: Relation, source_triplet: Optional[LinkedTriplet]) -> Tuple[float, List[ValidationIssue]]:
        """Validate semantic aspects of the relation."""
        issues = []
        score = 1.0

        if not source_triplet:
            return 0.5, [ValidationIssue(
                category=ValidationCategory.SEMANTIC,
                severity="info",
                message="No source triplet available for semantic validation",
                score_impact=-0.5
            )]

        # Check semantic coherence using embeddings if available
        if self.enable_semantic_validation and self._semantic_model:
            try:
                coherence_score = self._calculate_semantic_coherence(relation, source_triplet)
                if coherence_score < 0.5:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.SEMANTIC,
                        severity="warning",
                        message=f"Low semantic coherence: {coherence_score:.2f}",
                        score_impact=-0.3
                    ))
                    score -= 0.3
                elif coherence_score > 0.8:
                    score += 0.1  # Bonus for high coherence
            except Exception as e:
                logger.warning(f"Semantic coherence calculation failed: {e}")

        # Check for meaningful predicate
        if relation.type and len(relation.type.split()) == 1:
            # Single word predicates are often less meaningful
            issues.append(ValidationIssue(
                category=ValidationCategory.SEMANTIC,
                severity="info",
                message="Single-word predicate may lack semantic richness",
                score_impact=-0.1
            ))
            score -= 0.1

        return max(0.0, min(1.0, score)), issues

    def _validate_context(self, relation: Relation, source_triplet: Optional[LinkedTriplet], context_info: Optional[Dict[str, Any]]) -> Tuple[float, List[ValidationIssue]]:
        """Validate contextual aspects of the relation."""
        issues = []
        score = 1.0

        if not self.enable_context_preservation:
            return 1.0, []

        # Check if context information is preserved
        if not context_info:
            issues.append(ValidationIssue(
                category=ValidationCategory.CONTEXTUAL,
                severity="info",
                message="No context information available",
                score_impact=-0.2
            ))
            score -= 0.2

        # Check source sentence quality if available
        if source_triplet and hasattr(source_triplet, 'sentence'):
            sentence_quality = self._assess_sentence_quality(source_triplet.sentence)
            if sentence_quality < 0.5:
                issues.append(ValidationIssue(
                    category=ValidationCategory.CONTEXTUAL,
                    severity="warning",
                    message=f"Low source sentence quality: {sentence_quality:.2f}",
                    score_impact=-0.3
                ))
                score -= 0.3

        # Check entity linking quality
        if source_triplet and hasattr(source_triplet, 'subject_match') and hasattr(source_triplet, 'object_match'):
            if source_triplet.subject_match and source_triplet.subject_match.confidence < 0.7:
                issues.append(ValidationIssue(
                    category=ValidationCategory.CONTEXTUAL,
                    severity="warning",
                    message=f"Low subject entity linking confidence: {source_triplet.subject_match.confidence:.2f}",
                    score_impact=-0.2
                ))
                score -= 0.2

            if source_triplet.object_match and source_triplet.object_match.confidence < 0.7:
                issues.append(ValidationIssue(
                    category=ValidationCategory.CONTEXTUAL,
                    severity="warning",
                    message=f"Low object entity linking confidence: {source_triplet.object_match.confidence:.2f}",
                    score_impact=-0.2
                ))
                score -= 0.2

        return max(0.0, score), issues

    def _assess_quality(self, relation: Relation, source_triplet: Optional[LinkedTriplet]) -> Dict[str, float]:
        """Assess various quality metrics for the relation."""
        scores = {}

        # Confidence score from source triplet
        if source_triplet and hasattr(source_triplet, 'confidence'):
            scores['confidence'] = source_triplet.confidence
        else:
            scores['confidence'] = 0.5  # Default confidence

        # Relevance score based on relation type meaningfulness
        relevance = self._calculate_relevance_score(relation)
        scores['relevance'] = relevance

        # Entity quality score
        entity_quality = self._assess_entity_quality(relation, source_triplet)
        scores['entity_quality'] = entity_quality

        return scores

    def _validate_domain(self, relation: Relation, context_info: Optional[Dict[str, Any]]) -> Tuple[float, List[ValidationIssue]]:
        """Validate domain-specific aspects of the relation."""
        issues = []
        score = 1.0

        if not context_info or 'domain' not in context_info:
            return 1.0, []  # No domain validation if no domain specified

        domain = context_info['domain'].lower()
        if domain not in self._domain_rules:
            return 1.0, []  # No rules for this domain

        domain_rule = self._domain_rules[domain]
        relation_text = f"{relation.type} {getattr(relation, 'description', '')}"

        # Check required patterns
        required_patterns = domain_rule.get('required_patterns', [])
        for pattern in required_patterns:
            if not re.search(pattern, relation_text, re.IGNORECASE):
                issues.append(ValidationIssue(
                    category=ValidationCategory.DOMAIN,
                    severity="warning",
                    message=f"Missing domain-specific pattern: {pattern}",
                    score_impact=-0.2
                ))
                score -= 0.2

        # Check forbidden patterns
        forbidden_patterns = domain_rule.get('forbidden_patterns', [])
        for pattern in forbidden_patterns:
            if re.search(pattern, relation_text, re.IGNORECASE):
                issues.append(ValidationIssue(
                    category=ValidationCategory.DOMAIN,
                    severity="error",
                    message=f"Contains forbidden domain pattern: {pattern}",
                    score_impact=-0.5
                ))
                score -= 0.5

        # Apply confidence boost if applicable
        confidence_boost = domain_rule.get('confidence_boost', 0.0)
        if confidence_boost > 0 and score > 0.5:
            score = min(1.0, score + confidence_boost)

        return max(0.0, score), issues

    def _calculate_overall_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate overall score from component scores."""
        weighted_score = 0.0
        total_weight = 0.0

        # Apply weights to different components
        weights = {
            'confidence': self.confidence_weight,
            'relevance': self.relevance_weight,
            'context': self.context_weight,
            'semantic': self.semantic_weight,
            'domain': self.domain_weight,
            'structural': 0.2,  # Fixed weight for structural
            'entity_quality': 0.1  # Fixed weight for entity quality
        }

        for component, score in component_scores.items():
            weight = weights.get(component, 0.1)  # Default weight
            weighted_score += score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _determine_quality_level(self, overall_score: float) -> RelationQuality:
        """Determine quality level based on overall score."""
        for quality_level, threshold in sorted(self.quality_thresholds.items(), key=lambda x: x[1], reverse=True):
            if overall_score >= threshold:
                return quality_level
        return RelationQuality.INVALID

    def _check_validation_pass(self, overall_score: float, validation_issues: List[ValidationIssue]) -> bool:
        """Check if relation passes validation."""
        # Check overall score threshold
        if overall_score < self.min_overall_score:
            return False

        # Check for critical errors
        critical_errors = [issue for issue in validation_issues if issue.severity == "error"]
        if critical_errors and self.strict_mode:
            return False

        return True

    def _get_rejection_reason(self, validation_issues: List[ValidationIssue]) -> str:
        """Get rejection reason from validation issues."""
        errors = [issue for issue in validation_issues if issue.severity == "error"]
        if errors:
            return f"Critical errors: {', '.join(error.message for error in errors[:3])}"

        warnings = [issue for issue in validation_issues if issue.severity == "warning"]
        if len(warnings) > 3:
            return f"Too many warnings ({len(warnings)}): {', '.join(warning.message for warning in warnings[:2])}"

        return "Overall score below threshold"

    def _calculate_semantic_coherence(self, relation: Relation, source_triplet: LinkedTriplet) -> float:
        """Calculate semantic coherence using embeddings."""
        if not self._semantic_model:
            return 0.5

        try:
            # Create text representations
            relation_text = f"{relation.type}"
            triplet_text = f"{source_triplet.subject} {source_triplet.predicate} {source_triplet.object}"

            # Get embeddings and calculate similarity
            embeddings = self._semantic_model.encode([relation_text, triplet_text])

            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(embeddings[0], embeddings[1]))
            norm1 = sum(a * a for a in embeddings[0]) ** 0.5
            norm2 = sum(b * b for b in embeddings[1]) ** 0.5

            similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.warning(f"Semantic coherence calculation failed: {e}")
            return 0.5

    def _calculate_relevance_score(self, relation: Relation) -> float:
        """Calculate relevance score based on relation type."""
        if not relation.type:
            return 0.0

        # Simple heuristics for relevance
        score = 0.5  # Base score

        # Longer, more descriptive predicates are generally more relevant
        word_count = len(relation.type.split())
        if word_count >= 2:
            score += 0.2
        if word_count >= 3:
            score += 0.1

        # Check for meaningful verbs
        meaningful_verbs = ['create', 'develop', 'establish', 'discover', 'invent', 'found', 'lead', 'manage', 'direct']
        if any(verb in relation.type.lower() for verb in meaningful_verbs):
            score += 0.2

        # Penalize noise words
        if self._validation_patterns['noise_predicates'].search(relation.type):
            score -= 0.3

        return max(0.0, min(1.0, score))

    def _assess_sentence_quality(self, sentence: str) -> float:
        """Assess the quality of the source sentence."""
        if not sentence:
            return 0.0

        score = 1.0

        # Check length
        if len(sentence) < 10:
            score -= 0.3
        elif len(sentence) > 500:
            score -= 0.2

        # Check for proper punctuation
        if not re.search(r'[.!?]$', sentence.strip()):
            score -= 0.1

        # Check for filler words
        filler_count = len(self._validation_patterns['filler_words'].findall(sentence))
        if filler_count > 0:
            score -= min(0.3, filler_count * 0.1)

        return max(0.0, score)

    def _assess_entity_quality(self, relation: Relation, source_triplet: Optional[LinkedTriplet]) -> float:
        """Assess the quality of linked entities."""
        if not source_triplet:
            return 0.5

        score = 1.0

        # Check subject entity quality
        if hasattr(source_triplet, 'subject_entity') and source_triplet.subject_entity:
            if hasattr(source_triplet, 'subject_match') and source_triplet.subject_match:
                score *= source_triplet.subject_match.confidence
        else:
            score *= 0.7  # Penalty for missing subject entity

        # Check object entity quality
        if hasattr(source_triplet, 'object_entity') and source_triplet.object_entity:
            if hasattr(source_triplet, 'object_match') and source_triplet.object_match:
                score *= source_triplet.object_match.confidence
        else:
            score *= 0.7  # Penalty for missing object entity

        return max(0.0, min(1.0, score))

    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("Relation validator closed")
        except Exception as e:
            logger.warning("Error during relation validator cleanup", error=str(e))

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
