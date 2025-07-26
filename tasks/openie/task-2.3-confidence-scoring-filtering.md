# Task 2.3: Confidence Scoring and Filtering Mechanisms

## Objective
Implement comprehensive confidence scoring and filtering mechanisms to ensure only high-quality triplets are ingested into the knowledge graph, reducing noise and improving overall data quality.

## Scope
- Create confidence scoring system for triplets
- Implement multi-level filtering mechanisms
- Add quality assessment metrics
- Create adaptive thresholds based on data quality
- **MANDATORY**: Test thoroughly before proceeding to Phase 3

## Implementation Details

### 1. Create Confidence Scorer

**File**: `packages/morag-graph/src/morag_graph/validators/confidence_scorer.py`

```python
"""Confidence scoring system for OpenIE triplets."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import structlog
import numpy as np
from collections import defaultdict

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

class ConfidenceScorer:
    """Scores confidence for OpenIE triplets using multiple factors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize confidence scorer.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Scoring weights
        self.weights = {
            'extraction_confidence': self.config.get('extraction_confidence_weight', 0.3),
            'entity_linking_confidence': self.config.get('entity_linking_weight', 0.25),
            'predicate_normalization_confidence': self.config.get('predicate_norm_weight', 0.2),
            'sentence_quality': self.config.get('sentence_quality_weight', 0.15),
            'triplet_quality': self.config.get('triplet_quality_weight', 0.1)
        }
        
        # Confidence thresholds
        self.thresholds = {
            'high_confidence': self.config.get('high_confidence_threshold', 0.8),
            'medium_confidence': self.config.get('medium_confidence_threshold', 0.6),
            'low_confidence': self.config.get('low_confidence_threshold', 0.4),
            'minimum_acceptable': self.config.get('minimum_acceptable_threshold', 0.3)
        }
    
    async def score_triplets(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score confidence for a list of triplets.
        
        Args:
            triplets: List of triplets to score
            
        Returns:
            Triplets with confidence scores
            
        Raises:
            ProcessingError: If scoring fails
        """
        if not triplets:
            return []
        
        try:
            scored_triplets = []
            
            for triplet in triplets:
                scored_triplet = await self._score_single_triplet(triplet)
                if scored_triplet:
                    scored_triplets.append(scored_triplet)
            
            # Calculate adaptive thresholds
            adaptive_thresholds = await self._calculate_adaptive_thresholds(scored_triplets)
            
            # Apply adaptive scoring adjustments
            adjusted_triplets = await self._apply_adaptive_adjustments(
                scored_triplets, adaptive_thresholds
            )
            
            logger.info(
                "Confidence scoring completed",
                input_triplets=len(triplets),
                scored_triplets=len(scored_triplets),
                adjusted_triplets=len(adjusted_triplets)
            )
            
            return adjusted_triplets
            
        except Exception as e:
            logger.error("Confidence scoring failed", error=str(e))
            raise ProcessingError(f"Confidence scoring failed: {str(e)}")
    
    async def _score_single_triplet(self, triplet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Score confidence for a single triplet.
        
        Args:
            triplet: Triplet to score
            
        Returns:
            Triplet with confidence score or None if invalid
        """
        try:
            # Extract individual confidence components
            extraction_conf = triplet.get('confidence', 0.0)
            
            # Entity linking confidence
            entity_linking = triplet.get('entity_linking', {})
            linking_conf = entity_linking.get('linking_confidence', 0.0)
            
            # Predicate normalization confidence
            predicate_norm = triplet.get('predicate_normalization', {})
            predicate_conf = predicate_norm.get('confidence', 0.0)
            
            # Sentence quality
            sentence_quality = triplet.get('sentence_quality', 0.0)
            
            # Triplet quality
            triplet_quality = triplet.get('quality_score', 0.0)
            
            # Calculate weighted confidence score
            weighted_score = (
                self.weights['extraction_confidence'] * extraction_conf +
                self.weights['entity_linking_confidence'] * linking_conf +
                self.weights['predicate_normalization_confidence'] * predicate_conf +
                self.weights['sentence_quality'] * sentence_quality +
                self.weights['triplet_quality'] * triplet_quality
            )
            
            # Apply bonus/penalty factors
            adjusted_score = await self._apply_adjustment_factors(triplet, weighted_score)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(adjusted_score)
            
            # Create scored triplet
            scored_triplet = triplet.copy()
            scored_triplet.update({
                'overall_confidence': adjusted_score,
                'confidence_level': confidence_level,
                'confidence_components': {
                    'extraction_confidence': extraction_conf,
                    'entity_linking_confidence': linking_conf,
                    'predicate_normalization_confidence': predicate_conf,
                    'sentence_quality': sentence_quality,
                    'triplet_quality': triplet_quality,
                    'weighted_score': weighted_score,
                    'adjustment_factors': await self._get_adjustment_factors(triplet)
                }
            })
            
            return scored_triplet
            
        except Exception as e:
            logger.error(
                "Single triplet scoring failed",
                error=str(e),
                triplet=triplet
            )
            return None
    
    async def _apply_adjustment_factors(self, triplet: Dict[str, Any], base_score: float) -> float:
        """Apply adjustment factors to base confidence score.
        
        Args:
            triplet: Triplet data
            base_score: Base weighted confidence score
            
        Returns:
            Adjusted confidence score
        """
        adjusted_score = base_score
        
        # Entity linking bonus
        entity_linking = triplet.get('entity_linking', {})
        if entity_linking.get('subject_linked') and entity_linking.get('object_linked'):
            adjusted_score += 0.1  # Both entities linked
        elif entity_linking.get('subject_linked') or entity_linking.get('object_linked'):
            adjusted_score += 0.05  # One entity linked
        
        # Predicate category bonus
        predicate_norm = triplet.get('predicate_normalization', {})
        category = predicate_norm.get('category', 'OTHER')
        if category != 'OTHER':
            adjusted_score += 0.05  # Known predicate category
        
        # Sentence length penalty/bonus
        sentence_word_count = triplet.get('sentence_word_count', 0)
        if 5 <= sentence_word_count <= 20:
            adjusted_score += 0.02  # Optimal sentence length
        elif sentence_word_count > 30:
            adjusted_score -= 0.05  # Very long sentence penalty
        elif sentence_word_count < 5:
            adjusted_score -= 0.03  # Very short sentence penalty
        
        # Entity specificity bonus
        subject_entity = triplet.get('subject_entity', {})
        object_entity = triplet.get('object_entity', {})
        
        if subject_entity.get('match_type') == 'exact':
            adjusted_score += 0.03
        if object_entity.get('match_type') == 'exact':
            adjusted_score += 0.03
        
        # Normalization method bonus
        norm_method = predicate_norm.get('method', 'fallback')
        if norm_method == 'exact_match':
            adjusted_score += 0.05
        elif norm_method == 'pattern_match':
            adjusted_score += 0.03
        elif norm_method == 'fallback':
            adjusted_score -= 0.05
        
        return max(0.0, min(1.0, adjusted_score))
    
    async def _get_adjustment_factors(self, triplet: Dict[str, Any]) -> Dict[str, float]:
        """Get adjustment factors applied to triplet.
        
        Args:
            triplet: Triplet data
            
        Returns:
            Dictionary of adjustment factors
        """
        factors = {}
        
        # Entity linking factors
        entity_linking = triplet.get('entity_linking', {})
        if entity_linking.get('subject_linked') and entity_linking.get('object_linked'):
            factors['both_entities_linked'] = 0.1
        elif entity_linking.get('subject_linked') or entity_linking.get('object_linked'):
            factors['one_entity_linked'] = 0.05
        
        # Predicate category factor
        predicate_norm = triplet.get('predicate_normalization', {})
        if predicate_norm.get('category', 'OTHER') != 'OTHER':
            factors['known_predicate_category'] = 0.05
        
        # Sentence length factors
        sentence_word_count = triplet.get('sentence_word_count', 0)
        if 5 <= sentence_word_count <= 20:
            factors['optimal_sentence_length'] = 0.02
        elif sentence_word_count > 30:
            factors['long_sentence_penalty'] = -0.05
        elif sentence_word_count < 5:
            factors['short_sentence_penalty'] = -0.03
        
        return factors
    
    def _determine_confidence_level(self, score: float) -> str:
        """Determine confidence level based on score.
        
        Args:
            score: Confidence score
            
        Returns:
            Confidence level string
        """
        if score >= self.thresholds['high_confidence']:
            return 'HIGH'
        elif score >= self.thresholds['medium_confidence']:
            return 'MEDIUM'
        elif score >= self.thresholds['low_confidence']:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    async def _calculate_adaptive_thresholds(self, triplets: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate adaptive thresholds based on data distribution.
        
        Args:
            triplets: Scored triplets
            
        Returns:
            Adaptive thresholds
        """
        if not triplets:
            return self.thresholds.copy()
        
        # Extract confidence scores
        scores = [t.get('overall_confidence', 0.0) for t in triplets]
        
        if not scores:
            return self.thresholds.copy()
        
        # Calculate percentiles
        scores_array = np.array(scores)
        percentiles = {
            'p25': np.percentile(scores_array, 25),
            'p50': np.percentile(scores_array, 50),
            'p75': np.percentile(scores_array, 75),
            'p90': np.percentile(scores_array, 90)
        }
        
        # Adaptive threshold calculation
        adaptive_thresholds = {
            'high_confidence': max(self.thresholds['high_confidence'], percentiles['p75']),
            'medium_confidence': max(self.thresholds['medium_confidence'], percentiles['p50']),
            'low_confidence': max(self.thresholds['low_confidence'], percentiles['p25']),
            'minimum_acceptable': self.thresholds['minimum_acceptable']
        }
        
        return adaptive_thresholds
    
    async def _apply_adaptive_adjustments(
        self, 
        triplets: List[Dict[str, Any]], 
        adaptive_thresholds: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Apply adaptive threshold adjustments.
        
        Args:
            triplets: Scored triplets
            adaptive_thresholds: Adaptive thresholds
            
        Returns:
            Triplets with adaptive adjustments
        """
        adjusted_triplets = []
        
        for triplet in triplets:
            adjusted_triplet = triplet.copy()
            
            # Update confidence level with adaptive thresholds
            score = triplet.get('overall_confidence', 0.0)
            
            if score >= adaptive_thresholds['high_confidence']:
                confidence_level = 'HIGH'
            elif score >= adaptive_thresholds['medium_confidence']:
                confidence_level = 'MEDIUM'
            elif score >= adaptive_thresholds['low_confidence']:
                confidence_level = 'LOW'
            else:
                confidence_level = 'VERY_LOW'
            
            adjusted_triplet.update({
                'adaptive_confidence_level': confidence_level,
                'adaptive_thresholds': adaptive_thresholds
            })
            
            adjusted_triplets.append(adjusted_triplet)
        
        return adjusted_triplets
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scorer statistics and configuration.
        
        Returns:
            Dictionary with scorer statistics
        """
        return {
            'scorer_name': 'ConfidenceScorer',
            'weights': self.weights,
            'thresholds': self.thresholds,
            'total_weight': sum(self.weights.values())
        }
```

### 2. Create Filtering Service

**File**: `packages/morag-graph/src/morag_graph/validators/triplet_filter.py`

```python
"""Filtering service for OpenIE triplets."""

import asyncio
from typing import List, Dict, Any, Optional, Set
import structlog
from collections import defaultdict

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

class TripletFilter:
    """Filters triplets based on confidence and quality criteria."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize triplet filter.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Filter thresholds
        self.min_overall_confidence = self.config.get('min_overall_confidence', 0.5)
        self.min_extraction_confidence = self.config.get('min_extraction_confidence', 0.4)
        self.min_quality_score = self.config.get('min_quality_score', 0.3)
        
        # Filter criteria
        self.require_entity_linking = self.config.get('require_entity_linking', False)
        self.require_predicate_normalization = self.config.get('require_predicate_normalization', True)
        self.max_triplets_per_sentence = self.config.get('max_triplets_per_sentence', 5)
        
        # Blacklisted patterns
        self.blacklisted_predicates = set(self.config.get('blacklisted_predicates', [
            'is', 'are', 'was', 'were', 'be', 'been'
        ]))
        
        self.blacklisted_entities = set(self.config.get('blacklisted_entities', [
            'it', 'this', 'that', 'something', 'someone'
        ]))
    
    async def filter_triplets(self, triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Filter triplets based on quality criteria.
        
        Args:
            triplets: List of triplets to filter
            
        Returns:
            Dictionary with filtered triplets and statistics
            
        Raises:
            ProcessingError: If filtering fails
        """
        if not triplets:
            return {'filtered_triplets': [], 'filter_statistics': {}}
        
        try:
            # Apply filters in sequence
            confidence_filtered = await self._filter_by_confidence(triplets)
            quality_filtered = await self._filter_by_quality(confidence_filtered)
            entity_filtered = await self._filter_by_entity_requirements(quality_filtered)
            predicate_filtered = await self._filter_by_predicate_requirements(entity_filtered)
            deduped_filtered = await self._filter_duplicates(predicate_filtered)
            sentence_filtered = await self._filter_by_sentence_limits(deduped_filtered)
            
            # Calculate filter statistics
            filter_stats = await self._calculate_filter_statistics(
                triplets, sentence_filtered
            )
            
            logger.info(
                "Triplet filtering completed",
                input_triplets=len(triplets),
                output_triplets=len(sentence_filtered),
                filter_rate=len(sentence_filtered) / len(triplets) if triplets else 0
            )
            
            return {
                'filtered_triplets': sentence_filtered,
                'filter_statistics': filter_stats
            }
            
        except Exception as e:
            logger.error("Triplet filtering failed", error=str(e))
            raise ProcessingError(f"Triplet filtering failed: {str(e)}")
    
    async def _filter_by_confidence(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter triplets by confidence thresholds.
        
        Args:
            triplets: Triplets to filter
            
        Returns:
            Confidence-filtered triplets
        """
        filtered = []
        
        for triplet in triplets:
            overall_conf = triplet.get('overall_confidence', 0.0)
            extraction_conf = triplet.get('confidence', 0.0)
            
            if (overall_conf >= self.min_overall_confidence and
                extraction_conf >= self.min_extraction_confidence):
                filtered.append(triplet)
        
        return filtered
    
    async def _filter_by_quality(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter triplets by quality score.
        
        Args:
            triplets: Triplets to filter
            
        Returns:
            Quality-filtered triplets
        """
        filtered = []
        
        for triplet in triplets:
            quality_score = triplet.get('quality_score', 0.0)
            
            if quality_score >= self.min_quality_score:
                filtered.append(triplet)
        
        return filtered
    
    async def _filter_by_entity_requirements(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter triplets by entity linking requirements.
        
        Args:
            triplets: Triplets to filter
            
        Returns:
            Entity-filtered triplets
        """
        if not self.require_entity_linking:
            return triplets
        
        filtered = []
        
        for triplet in triplets:
            entity_linking = triplet.get('entity_linking', {})
            
            # Check if at least one entity is linked
            if (entity_linking.get('subject_linked') or 
                entity_linking.get('object_linked')):
                
                # Check for blacklisted entities
                subject = triplet.get('subject', '').lower().strip()
                obj = triplet.get('object', '').lower().strip()
                
                if (subject not in self.blacklisted_entities and
                    obj not in self.blacklisted_entities):
                    filtered.append(triplet)
        
        return filtered
    
    async def _filter_by_predicate_requirements(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter triplets by predicate requirements.
        
        Args:
            triplets: Triplets to filter
            
        Returns:
            Predicate-filtered triplets
        """
        filtered = []
        
        for triplet in triplets:
            predicate = triplet.get('predicate', '').lower().strip()
            original_predicate = triplet.get('original_predicate', '').lower().strip()
            
            # Check blacklisted predicates
            if (predicate not in self.blacklisted_predicates and
                original_predicate not in self.blacklisted_predicates):
                
                # Check normalization requirement
                if self.require_predicate_normalization:
                    predicate_norm = triplet.get('predicate_normalization', {})
                    if predicate_norm.get('method') != 'fallback':
                        filtered.append(triplet)
                else:
                    filtered.append(triplet)
        
        return filtered
    
    async def _filter_duplicates(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate triplets.
        
        Args:
            triplets: Triplets to deduplicate
            
        Returns:
            Deduplicated triplets
        """
        seen_triplets = {}
        filtered = []
        
        for triplet in triplets:
            # Create key based on normalized components
            subject = triplet.get('subject', '').lower().strip()
            predicate = triplet.get('predicate', '').lower().strip()
            obj = triplet.get('object', '').lower().strip()
            
            triplet_key = f"{subject}|{predicate}|{obj}"
            
            if triplet_key not in seen_triplets:
                seen_triplets[triplet_key] = triplet
                filtered.append(triplet)
            else:
                # Keep the one with higher confidence
                existing = seen_triplets[triplet_key]
                if triplet.get('overall_confidence', 0) > existing.get('overall_confidence', 0):
                    # Replace in seen_triplets and filtered list
                    seen_triplets[triplet_key] = triplet
                    for i, t in enumerate(filtered):
                        if t is existing:
                            filtered[i] = triplet
                            break
        
        return filtered
    
    async def _filter_by_sentence_limits(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter triplets by sentence-level limits.
        
        Args:
            triplets: Triplets to filter
            
        Returns:
            Sentence-limited triplets
        """
        # Group by sentence
        sentence_groups = defaultdict(list)
        for triplet in triplets:
            sentence_key = (
                triplet.get('source_sentence', ''),
                triplet.get('sentence_index', 0)
            )
            sentence_groups[sentence_key].append(triplet)
        
        filtered = []
        
        for sentence_key, sentence_triplets in sentence_groups.items():
            # Sort by confidence and take top N
            sorted_triplets = sorted(
                sentence_triplets,
                key=lambda t: t.get('overall_confidence', 0),
                reverse=True
            )
            
            # Take up to max_triplets_per_sentence
            selected_triplets = sorted_triplets[:self.max_triplets_per_sentence]
            filtered.extend(selected_triplets)
        
        return filtered
    
    async def _calculate_filter_statistics(
        self, 
        original_triplets: List[Dict[str, Any]], 
        filtered_triplets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate filtering statistics.
        
        Args:
            original_triplets: Original triplets
            filtered_triplets: Filtered triplets
            
        Returns:
            Filter statistics
        """
        total_original = len(original_triplets)
        total_filtered = len(filtered_triplets)
        
        # Calculate confidence distribution
        confidence_levels = defaultdict(int)
        for triplet in filtered_triplets:
            level = triplet.get('confidence_level', 'UNKNOWN')
            confidence_levels[level] += 1
        
        # Calculate average scores
        avg_confidence = sum(
            t.get('overall_confidence', 0) for t in filtered_triplets
        ) / total_filtered if total_filtered > 0 else 0
        
        avg_quality = sum(
            t.get('quality_score', 0) for t in filtered_triplets
        ) / total_filtered if total_filtered > 0 else 0
        
        return {
            'total_original': total_original,
            'total_filtered': total_filtered,
            'filter_rate': total_filtered / total_original if total_original > 0 else 0,
            'confidence_distribution': dict(confidence_levels),
            'average_confidence': avg_confidence,
            'average_quality': avg_quality,
            'filters_applied': {
                'min_overall_confidence': self.min_overall_confidence,
                'min_extraction_confidence': self.min_extraction_confidence,
                'min_quality_score': self.min_quality_score,
                'require_entity_linking': self.require_entity_linking,
                'require_predicate_normalization': self.require_predicate_normalization,
                'max_triplets_per_sentence': self.max_triplets_per_sentence
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics and configuration.
        
        Returns:
            Dictionary with filter statistics
        """
        return {
            'filter_name': 'TripletFilter',
            'min_overall_confidence': self.min_overall_confidence,
            'min_extraction_confidence': self.min_extraction_confidence,
            'min_quality_score': self.min_quality_score,
            'require_entity_linking': self.require_entity_linking,
            'require_predicate_normalization': self.require_predicate_normalization,
            'max_triplets_per_sentence': self.max_triplets_per_sentence,
            'blacklisted_predicates_count': len(self.blacklisted_predicates),
            'blacklisted_entities_count': len(self.blacklisted_entities)
        }
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_confidence_scorer.py`

```python
"""Tests for confidence scorer."""

import pytest
from morag_graph.validators.confidence_scorer import ConfidenceScorer

class TestConfidenceScorer:
    
    def test_initialization(self):
        """Test scorer initialization."""
        scorer = ConfidenceScorer()
        assert sum(scorer.weights.values()) <= 1.1  # Allow small floating point variance
        assert scorer.thresholds['high_confidence'] > scorer.thresholds['medium_confidence']
    
    @pytest.mark.asyncio
    async def test_score_single_triplet(self):
        """Test single triplet scoring."""
        scorer = ConfidenceScorer()
        
        triplet = {
            'confidence': 0.8,
            'entity_linking': {'linking_confidence': 0.9},
            'predicate_normalization': {'confidence': 0.7},
            'sentence_quality': 0.8,
            'quality_score': 0.75
        }
        
        result = await scorer._score_single_triplet(triplet)
        
        assert result is not None
        assert 'overall_confidence' in result
        assert 'confidence_level' in result
        assert 'confidence_components' in result
        assert 0 <= result['overall_confidence'] <= 1
    
    def test_determine_confidence_level(self):
        """Test confidence level determination."""
        scorer = ConfidenceScorer()
        
        assert scorer._determine_confidence_level(0.9) == 'HIGH'
        assert scorer._determine_confidence_level(0.7) == 'MEDIUM'
        assert scorer._determine_confidence_level(0.5) == 'LOW'
        assert scorer._determine_confidence_level(0.2) == 'VERY_LOW'
```

## Acceptance Criteria

- [ ] ConfidenceScorer class with weighted scoring system
- [ ] TripletFilter class with multi-level filtering
- [ ] Adaptive threshold calculation based on data distribution
- [ ] Comprehensive filtering criteria (confidence, quality, entities, predicates)
- [ ] Duplicate detection and removal
- [ ] Sentence-level triplet limits
- [ ] Comprehensive unit tests with >90% coverage
- [ ] Integration with triplet processing pipeline
- [ ] Statistics and monitoring capabilities
- [ ] Proper logging and error handling

## Dependencies
- Task 2.1: Entity Linking Between OpenIE and spaCy NER
- Task 2.2: Entity Normalization and Canonical Mapping

## Estimated Effort
- **Development**: 6-8 hours
- **Testing**: 4-5 hours
- **Integration**: 2-3 hours
- **Total**: 12-16 hours

## Notes
- Focus on balancing precision and recall in filtering
- Consider domain-specific confidence adjustments
- Implement configurable thresholds for different use cases
- Plan for A/B testing of different filtering strategies
