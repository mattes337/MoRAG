"""Confidence scoring and filtering for OpenIE entity linking."""

import asyncio
from typing import List, Dict, Any, Optional, NamedTuple, Set, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import structlog

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError
from .entity_linker import LinkedTriplet, EntityMatch
from ..processors.triplet_processor import ValidatedTriplet

logger = structlog.get_logger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for filtering."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for different quality levels."""
    very_low: float = 0.3
    low: float = 0.5
    medium: float = 0.7
    high: float = 0.8
    very_high: float = 0.9


class ConfidenceScore(NamedTuple):
    """Represents a comprehensive confidence score."""
    overall_score: float
    component_scores: Dict[str, float]
    confidence_level: ConfidenceLevel
    quality_flags: Set[str]
    metadata: Dict[str, Any] = {}


class FilteredTriplet(NamedTuple):
    """Represents a triplet that passed confidence filtering."""
    triplet: LinkedTriplet
    confidence_score: ConfidenceScore
    filter_reason: Optional[str] = None


class ConfidenceManager:
    """Manages confidence scoring and filtering for OpenIE results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize confidence manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Configuration
        self.min_overall_confidence = self.config.get('min_overall_confidence', 0.6)
        self.min_entity_match_confidence = self.config.get('min_entity_match_confidence', 0.5)
        self.min_validation_score = self.config.get('min_validation_score', 0.6)
        self.require_entity_linking = self.config.get('require_entity_linking', False)
        self.enable_adaptive_thresholds = self.config.get('enable_adaptive_thresholds', True)
        
        # Confidence thresholds
        self.thresholds = ConfidenceThresholds(
            very_low=self.config.get('threshold_very_low', 0.3),
            low=self.config.get('threshold_low', 0.5),
            medium=self.config.get('threshold_medium', 0.7),
            high=self.config.get('threshold_high', 0.8),
            very_high=self.config.get('threshold_very_high', 0.9)
        )
        
        # Component weights for overall score calculation
        self.component_weights = {
            'extraction_confidence': self.config.get('weight_extraction', 0.3),
            'validation_score': self.config.get('weight_validation', 0.3),
            'entity_linking_score': self.config.get('weight_entity_linking', 0.2),
            'predicate_quality': self.config.get('weight_predicate', 0.1),
            'sentence_quality': self.config.get('weight_sentence', 0.1)
        }
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="confidence_mgr")
        
        logger.info(
            "Confidence manager initialized",
            min_overall_confidence=self.min_overall_confidence,
            min_entity_match_confidence=self.min_entity_match_confidence,
            require_entity_linking=self.require_entity_linking,
            thresholds=self.thresholds
        )
    
    async def filter_triplets(
        self, 
        triplets: List[LinkedTriplet],
        target_confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        source_doc_id: Optional[str] = None
    ) -> List[FilteredTriplet]:
        """Filter triplets based on confidence scores.
        
        Args:
            triplets: List of linked triplets to filter
            target_confidence_level: Target confidence level for filtering
            source_doc_id: Optional source document ID
            
        Returns:
            List of filtered triplets that meet confidence requirements
            
        Raises:
            ProcessingError: If filtering fails
        """
        if not triplets:
            return []
        
        try:
            logger.debug(
                "Starting confidence filtering",
                triplet_count=len(triplets),
                target_level=target_confidence_level.value,
                source_doc_id=source_doc_id
            )
            
            # Calculate confidence scores for all triplets
            scored_triplets = await self._score_triplets(triplets, source_doc_id)
            
            # Apply filtering based on target confidence level
            target_threshold = getattr(self.thresholds, target_confidence_level.value)
            filtered_triplets = []
            
            for triplet, score in scored_triplets:
                if self._passes_confidence_filter(score, target_threshold):
                    filtered_triplets.append(FilteredTriplet(
                        triplet=triplet,
                        confidence_score=score,
                        filter_reason=None
                    ))
                else:
                    # Optionally include rejected triplets with reason
                    if self.config.get('include_rejected', False):
                        filtered_triplets.append(FilteredTriplet(
                            triplet=triplet,
                            confidence_score=score,
                            filter_reason=f"Below {target_confidence_level.value} threshold ({target_threshold})"
                        ))
            
            logger.info(
                "Confidence filtering completed",
                input_triplets=len(triplets),
                filtered_triplets=len([t for t in filtered_triplets if t.filter_reason is None]),
                rejected_triplets=len([t for t in filtered_triplets if t.filter_reason is not None]),
                target_level=target_confidence_level.value,
                source_doc_id=source_doc_id
            )
            
            return filtered_triplets
            
        except Exception as e:
            logger.error(
                "Confidence filtering failed",
                error=str(e),
                error_type=type(e).__name__,
                triplet_count=len(triplets),
                source_doc_id=source_doc_id
            )
            raise ProcessingError(f"Confidence filtering failed: {e}")
    
    async def _score_triplets(
        self, 
        triplets: List[LinkedTriplet], 
        source_doc_id: Optional[str] = None
    ) -> List[Tuple[LinkedTriplet, ConfidenceScore]]:
        """Calculate confidence scores for triplets."""
        def score_triplets_sync():
            scored_triplets = []
            
            for triplet in triplets:
                try:
                    score = self._calculate_confidence_score(triplet)
                    scored_triplets.append((triplet, score))
                except Exception as e:
                    logger.warning(
                        "Failed to score triplet",
                        triplet=f"{triplet.subject} | {triplet.predicate} | {triplet.object}",
                        error=str(e)
                    )
                    # Create a low-confidence score as fallback
                    fallback_score = ConfidenceScore(
                        overall_score=0.1,
                        component_scores={'error': 0.1},
                        confidence_level=ConfidenceLevel.VERY_LOW,
                        quality_flags={'scoring_error'},
                        metadata={'error': str(e)}
                    )
                    scored_triplets.append((triplet, fallback_score))
            
            return scored_triplets
        
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, score_triplets_sync
        )
    
    def _calculate_confidence_score(self, triplet: LinkedTriplet) -> ConfidenceScore:
        """Calculate comprehensive confidence score for a triplet."""
        component_scores = {}
        quality_flags = set()
        
        # 1. Extraction confidence (from OpenIE)
        extraction_confidence = triplet.confidence
        component_scores['extraction_confidence'] = extraction_confidence
        
        if extraction_confidence >= 0.8:
            quality_flags.add('high_extraction_confidence')
        elif extraction_confidence < 0.5:
            quality_flags.add('low_extraction_confidence')
        
        # 2. Validation score (from triplet processor)
        validation_score = triplet.validation_score
        component_scores['validation_score'] = validation_score
        
        if validation_score >= 0.8:
            quality_flags.add('high_validation_score')
        elif validation_score < 0.5:
            quality_flags.add('low_validation_score')
        
        # 3. Entity linking score
        entity_linking_score = self._calculate_entity_linking_score(triplet)
        component_scores['entity_linking_score'] = entity_linking_score
        
        if entity_linking_score >= 0.8:
            quality_flags.add('high_entity_linking')
        elif entity_linking_score == 0.0:
            quality_flags.add('no_entity_linking')
        
        # 4. Predicate quality score
        predicate_quality = self._calculate_predicate_quality(triplet.predicate)
        component_scores['predicate_quality'] = predicate_quality
        
        if predicate_quality >= 0.8:
            quality_flags.add('high_quality_predicate')
        elif predicate_quality < 0.5:
            quality_flags.add('low_quality_predicate')
        
        # 5. Sentence quality score (if available)
        sentence_quality = self._extract_sentence_quality(triplet)
        component_scores['sentence_quality'] = sentence_quality
        
        if sentence_quality >= 0.8:
            quality_flags.add('high_sentence_quality')
        
        # Calculate overall score using weighted average
        overall_score = sum(
            component_scores.get(component, 0.0) * weight
            for component, weight in self.component_weights.items()
        )
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(overall_score)
        
        # Additional quality checks
        if triplet.subject_entity and triplet.object_entity:
            quality_flags.add('both_entities_linked')
        elif triplet.subject_entity or triplet.object_entity:
            quality_flags.add('partial_entity_linking')
        else:
            quality_flags.add('no_entities_linked')
        
        # Check for proper nouns
        if any(word[0].isupper() for word in triplet.subject.split()):
            quality_flags.add('proper_noun_subject')
        if any(word[0].isupper() for word in triplet.object.split()):
            quality_flags.add('proper_noun_object')
        
        return ConfidenceScore(
            overall_score=overall_score,
            component_scores=component_scores,
            confidence_level=confidence_level,
            quality_flags=quality_flags,
            metadata={
                'sentence_id': triplet.sentence_id,
                'source_doc_id': triplet.source_doc_id,
                'subject_match_type': triplet.subject_match.match_type if triplet.subject_match else None,
                'object_match_type': triplet.object_match.match_type if triplet.object_match else None
            }
        )
    
    def _calculate_entity_linking_score(self, triplet: LinkedTriplet) -> float:
        """Calculate entity linking quality score."""
        subject_score = 0.0
        object_score = 0.0
        
        # Subject entity linking
        if triplet.subject_match:
            subject_score = triplet.subject_match.confidence
            # Bonus for exact matches
            if triplet.subject_match.match_type == 'exact':
                subject_score = min(1.0, subject_score + 0.1)
        
        # Object entity linking
        if triplet.object_match:
            object_score = triplet.object_match.confidence
            # Bonus for exact matches
            if triplet.object_match.match_type == 'exact':
                object_score = min(1.0, object_score + 0.1)
        
        # Average of both scores
        return (subject_score + object_score) / 2.0
    
    def _calculate_predicate_quality(self, predicate: str) -> float:
        """Calculate predicate quality score."""
        if not predicate or not predicate.strip():
            return 0.0
        
        score = 0.5  # Base score
        
        # Good predicates (action verbs, relationships)
        good_predicates = {
            'is', 'are', 'was', 'were', 'has', 'have', 'had',
            'works at', 'lives in', 'owns', 'creates', 'makes',
            'manages', 'leads', 'teaches', 'studies', 'writes',
            'develops', 'builds', 'founded', 'established',
            'located in', 'based in', 'part of', 'member of'
        }
        
        predicate_lower = predicate.lower().strip()
        
        if predicate_lower in good_predicates:
            score += 0.3
        elif any(good in predicate_lower for good in ['work', 'live', 'own', 'create', 'manage', 'lead']):
            score += 0.2
        
        # Penalty for weak predicates
        weak_predicates = {'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to'}
        if predicate_lower in weak_predicates:
            score -= 0.3
        
        # Length considerations
        if len(predicate_lower) < 2:
            score -= 0.2
        elif len(predicate_lower) > 50:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _extract_sentence_quality(self, triplet: LinkedTriplet) -> float:
        """Extract sentence quality score from triplet metadata."""
        try:
            # Try to get sentence quality from metadata
            if 'sentence_quality_score' in triplet.metadata:
                return float(triplet.metadata['sentence_quality_score'])
            
            # Fallback: estimate based on sentence length and structure
            sentence = triplet.sentence
            if not sentence:
                return 0.5
            
            score = 0.5
            
            # Length considerations
            if 20 <= len(sentence) <= 200:
                score += 0.2
            elif len(sentence) < 10 or len(sentence) > 500:
                score -= 0.2
            
            # Word count
            word_count = len(sentence.split())
            if 5 <= word_count <= 30:
                score += 0.1
            
            # Has punctuation
            if any(p in sentence for p in '.!?'):
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default score
    
    def _determine_confidence_level(self, overall_score: float) -> ConfidenceLevel:
        """Determine confidence level based on overall score."""
        if overall_score >= self.thresholds.very_high:
            return ConfidenceLevel.VERY_HIGH
        elif overall_score >= self.thresholds.high:
            return ConfidenceLevel.HIGH
        elif overall_score >= self.thresholds.medium:
            return ConfidenceLevel.MEDIUM
        elif overall_score >= self.thresholds.low:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _passes_confidence_filter(self, score: ConfidenceScore, threshold: float) -> bool:
        """Check if a confidence score passes the filter."""
        # Primary check: overall score
        if score.overall_score < threshold:
            return False
        
        # Additional checks based on configuration
        if self.require_entity_linking and 'no_entities_linked' in score.quality_flags:
            return False
        
        # Check minimum validation score
        if score.component_scores.get('validation_score', 0.0) < self.min_validation_score:
            return False
        
        # Check minimum entity match confidence
        entity_linking_score = score.component_scores.get('entity_linking_score', 0.0)
        if entity_linking_score > 0 and entity_linking_score < self.min_entity_match_confidence:
            return False
        
        return True
    
    async def get_quality_statistics(self, triplets: List[LinkedTriplet]) -> Dict[str, Any]:
        """Get quality statistics for a set of triplets."""
        if not triplets:
            return {}
        
        scored_triplets = await self._score_triplets(triplets)
        
        scores = [score.overall_score for _, score in scored_triplets]
        confidence_levels = [score.confidence_level for _, score in scored_triplets]
        
        # Calculate statistics
        stats = {
            'total_triplets': len(triplets),
            'average_confidence': sum(scores) / len(scores),
            'min_confidence': min(scores),
            'max_confidence': max(scores),
            'confidence_distribution': {
                level.value: sum(1 for cl in confidence_levels if cl == level)
                for level in ConfidenceLevel
            },
            'quality_flags_distribution': {}
        }
        
        # Aggregate quality flags
        all_flags = set()
        for _, score in scored_triplets:
            all_flags.update(score.quality_flags)
        
        for flag in all_flags:
            stats['quality_flags_distribution'][flag] = sum(
                1 for _, score in scored_triplets if flag in score.quality_flags
            )
        
        return stats
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            if self._executor:
                self._executor.shutdown(wait=True)
            logger.info("Confidence manager closed")
        except Exception as e:
            logger.warning("Error during confidence manager cleanup", error=str(e))
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
