# Task 3.1: Predicate Normalization and Standardization

## Objective
Implement predicate normalization to standardize verbose OpenIE predicates into consistent, meaningful relationship types for the knowledge graph.

## Scope
- Create predicate normalizer with standardization rules
- Implement predicate taxonomy and categorization
- Add semantic similarity matching for predicates
- Handle multilingual predicate normalization
- **MANDATORY**: Test thoroughly before proceeding to Task 3.2

## Implementation Details

### 1. Create Predicate Normalizer

**File**: `packages/morag-graph/src/morag_graph/normalizers/predicate_normalizer.py`

```python
"""Predicate normalization and standardization for OpenIE."""

import re
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
import structlog
from collections import defaultdict

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

class PredicateNormalizer:
    """Normalizes OpenIE predicates to standard relationship types."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize predicate normalizer.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Load predicate mappings and rules
        self._load_predicate_mappings()
        self._load_normalization_rules()
        
        # Confidence thresholds
        self.exact_match_confidence = 1.0
        self.pattern_match_confidence = 0.9
        self.semantic_match_confidence = 0.8
        self.default_confidence = 0.5
    
    def _load_predicate_mappings(self) -> None:
        """Load predicate mappings to standard relationship types."""
        # Core relationship categories
        self.relationship_categories = {
            'EMPLOYMENT': {
                'standard_form': 'WORKS_AT',
                'patterns': [
                    'works at', 'works for', 'employed by', 'employee of',
                    'job at', 'position at', 'career at', 'hired by'
                ]
            },
            'OWNERSHIP': {
                'standard_form': 'OWNS',
                'patterns': [
                    'owns', 'possesses', 'has', 'belongs to', 'property of',
                    'owner of', 'possession of'
                ]
            },
            'LOCATION': {
                'standard_form': 'LOCATED_IN',
                'patterns': [
                    'located in', 'based in', 'situated in', 'found in',
                    'resides in', 'lives in', 'headquarters in'
                ]
            },
            'LEADERSHIP': {
                'standard_form': 'LEADS',
                'patterns': [
                    'ceo of', 'president of', 'director of', 'manager of',
                    'leads', 'heads', 'runs', 'manages', 'supervises'
                ]
            },
            'MEMBERSHIP': {
                'standard_form': 'MEMBER_OF',
                'patterns': [
                    'member of', 'part of', 'belongs to', 'affiliated with',
                    'associated with', 'connected to'
                ]
            },
            'CREATION': {
                'standard_form': 'CREATED',
                'patterns': [
                    'created', 'founded', 'established', 'built', 'developed',
                    'invented', 'designed', 'authored', 'wrote'
                ]
            },
            'EDUCATION': {
                'standard_form': 'STUDIED_AT',
                'patterns': [
                    'studied at', 'graduated from', 'attended', 'alumni of',
                    'degree from', 'educated at'
                ]
            },
            'FAMILY': {
                'standard_form': 'RELATED_TO',
                'patterns': [
                    'married to', 'spouse of', 'husband of', 'wife of',
                    'father of', 'mother of', 'son of', 'daughter of',
                    'brother of', 'sister of', 'parent of', 'child of'
                ]
            },
            'TEMPORAL': {
                'standard_form': 'OCCURRED_ON',
                'patterns': [
                    'happened on', 'occurred on', 'took place on',
                    'started on', 'ended on', 'began on'
                ]
            },
            'CAUSATION': {
                'standard_form': 'CAUSES',
                'patterns': [
                    'causes', 'leads to', 'results in', 'brings about',
                    'triggers', 'produces', 'generates'
                ]
            }
        }
        
        # Create reverse mapping for quick lookup
        self.pattern_to_category = {}
        for category, data in self.relationship_categories.items():
            for pattern in data['patterns']:
                self.pattern_to_category[pattern.lower()] = {
                    'category': category,
                    'standard_form': data['standard_form']
                }
    
    def _load_normalization_rules(self) -> None:
        """Load normalization rules and patterns."""
        # Common verb transformations
        self.verb_transformations = {
            # Present to base form
            'works': 'work',
            'lives': 'live',
            'owns': 'own',
            'leads': 'lead',
            'manages': 'manage',
            'creates': 'create',
            'studies': 'study',
            
            # Past to base form
            'worked': 'work',
            'lived': 'live',
            'owned': 'own',
            'led': 'lead',
            'managed': 'manage',
            'created': 'create',
            'studied': 'study'
        }
        
        # Preposition standardization
        self.preposition_mappings = {
            'in': 'at',
            'within': 'at',
            'inside': 'at',
            'for': 'at',
            'with': 'at'
        }
        
        # Multilingual predicate mappings
        self.multilingual_predicates = {
            # Spanish to English
            'trabaja en': 'works at',
            'vive en': 'lives in',
            'estudia en': 'studies at',
            'es ceo de': 'ceo of',
            'es director de': 'director of',
            'pertenece a': 'belongs to',
            'miembro de': 'member of',

            # German to English
            'arbeitet bei': 'works at',
            'arbeitet für': 'works for',
            'lebt in': 'lives in',
            'wohnt in': 'lives in',
            'studiert an': 'studies at',
            'ist ceo von': 'ceo of',
            'ist geschäftsführer von': 'ceo of',
            'ist direktor von': 'director of',
            'gehört zu': 'belongs to',
            'mitglied von': 'member of',
            'gegründet von': 'founded by',
            'erstellt von': 'created by',
            'entwickelt von': 'developed by',

            # Common Spanish verbs
            'trabaja': 'works',
            'vive': 'lives',
            'estudia': 'studies',
            'dirige': 'leads',
            'posee': 'owns',

            # Common German verbs
            'arbeitet': 'works',
            'lebt': 'lives',
            'wohnt': 'lives',
            'studiert': 'studies',
            'leitet': 'leads',
            'führt': 'leads',
            'besitzt': 'owns',
            'gehört': 'belongs',
            'gründet': 'founds',
            'erstellt': 'creates',
            'entwickelt': 'develops'
        }
        
        # Noise words to remove
        self.noise_words = {
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'can', 'could', 'should', 'may', 'might',
            'the', 'a', 'an', 'and', 'or', 'but'
        }
    
    async def normalize_predicates(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize predicates in triplets.
        
        Args:
            triplets: Triplets with raw predicates
            
        Returns:
            Triplets with normalized predicates
            
        Raises:
            ProcessingError: If normalization fails
        """
        if not triplets:
            return []
        
        try:
            normalized_triplets = []
            
            for triplet in triplets:
                normalized_triplet = await self._normalize_triplet_predicate(triplet)
                if normalized_triplet:
                    normalized_triplets.append(normalized_triplet)
            
            # Calculate normalization statistics
            stats = await self._calculate_normalization_stats(triplets, normalized_triplets)
            
            logger.info(
                "Predicate normalization completed",
                input_triplets=len(triplets),
                normalized_triplets=len(normalized_triplets),
                **stats
            )
            
            return normalized_triplets
            
        except Exception as e:
            logger.error("Predicate normalization failed", error=str(e))
            raise ProcessingError(f"Predicate normalization failed: {str(e)}")
    
    async def _normalize_triplet_predicate(self, triplet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize predicate in a single triplet.
        
        Args:
            triplet: Triplet with raw predicate
            
        Returns:
            Triplet with normalized predicate or None if invalid
        """
        try:
            raw_predicate = triplet.get('predicate', '').strip()
            if not raw_predicate:
                return None
            
            # Clean and preprocess predicate
            cleaned_predicate = await self._clean_predicate(raw_predicate)
            
            # Normalize predicate
            normalized_predicate, confidence, method = await self._normalize_predicate(cleaned_predicate)
            
            if not normalized_predicate:
                return None
            
            # Create normalized triplet
            normalized_triplet = triplet.copy()
            normalized_triplet.update({
                'predicate': normalized_predicate,
                'original_predicate': raw_predicate,
                'cleaned_predicate': cleaned_predicate,
                'predicate_normalization': {
                    'method': method,
                    'confidence': confidence,
                    'category': self._get_predicate_category(normalized_predicate)
                }
            })
            
            return normalized_triplet
            
        except Exception as e:
            logger.error(
                "Triplet predicate normalization failed",
                error=str(e),
                triplet=triplet
            )
            return None
    
    async def _clean_predicate(self, predicate: str) -> str:
        """Clean and preprocess predicate text.
        
        Args:
            predicate: Raw predicate text
            
        Returns:
            Cleaned predicate
        """
        # Convert to lowercase
        cleaned = predicate.lower().strip()
        
        # Handle multilingual predicates
        for foreign_pred, english_pred in self.multilingual_predicates.items():
            if foreign_pred in cleaned:
                cleaned = cleaned.replace(foreign_pred, english_pred)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove noise words at the beginning
        words = cleaned.split()
        while words and words[0] in self.noise_words:
            words.pop(0)
        
        # Remove noise words at the end
        while words and words[-1] in self.noise_words:
            words.pop()
        
        cleaned = ' '.join(words)
        
        # Transform verbs to base form
        words = cleaned.split()
        transformed_words = []
        for word in words:
            if word in self.verb_transformations:
                transformed_words.append(self.verb_transformations[word])
            else:
                transformed_words.append(word)
        
        cleaned = ' '.join(transformed_words)
        
        return cleaned.strip()
    
    async def _normalize_predicate(self, predicate: str) -> Tuple[Optional[str], float, str]:
        """Normalize a predicate to standard form.
        
        Args:
            predicate: Cleaned predicate text
            
        Returns:
            Tuple of (normalized_predicate, confidence, method)
        """
        if not predicate:
            return None, 0.0, 'none'
        
        # 1. Try exact pattern match
        if predicate in self.pattern_to_category:
            mapping = self.pattern_to_category[predicate]
            return mapping['standard_form'], self.exact_match_confidence, 'exact_match'
        
        # 2. Try partial pattern match
        for pattern, mapping in self.pattern_to_category.items():
            if pattern in predicate or predicate in pattern:
                return mapping['standard_form'], self.pattern_match_confidence, 'pattern_match'
        
        # 3. Try semantic similarity matching
        semantic_match = await self._find_semantic_match(predicate)
        if semantic_match:
            return semantic_match, self.semantic_match_confidence, 'semantic_match'
        
        # 4. Try rule-based normalization
        rule_based = await self._apply_normalization_rules(predicate)
        if rule_based:
            return rule_based, self.default_confidence, 'rule_based'
        
        # 5. Return cleaned predicate as fallback
        return predicate.upper().replace(' ', '_'), 0.3, 'fallback'
    
    async def _find_semantic_match(self, predicate: str) -> Optional[str]:
        """Find semantic match for predicate.
        
        Args:
            predicate: Predicate to match
            
        Returns:
            Best semantic match or None
        """
        # Simple keyword-based semantic matching
        # In a more advanced implementation, this could use embeddings
        
        predicate_words = set(predicate.split())
        
        best_match = None
        best_score = 0.0
        
        for pattern, mapping in self.pattern_to_category.items():
            pattern_words = set(pattern.split())
            
            # Calculate word overlap
            common_words = predicate_words.intersection(pattern_words)
            if common_words:
                score = len(common_words) / max(len(predicate_words), len(pattern_words))
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_match = mapping['standard_form']
        
        return best_match
    
    async def _apply_normalization_rules(self, predicate: str) -> Optional[str]:
        """Apply rule-based normalization.
        
        Args:
            predicate: Predicate to normalize
            
        Returns:
            Normalized predicate or None
        """
        # Check for common relationship patterns
        if any(word in predicate for word in ['work', 'employ', 'job']):
            return 'WORKS_AT'
        
        if any(word in predicate for word in ['own', 'possess', 'belong']):
            return 'OWNS'
        
        if any(word in predicate for word in ['live', 'reside', 'locate']):
            return 'LOCATED_IN'
        
        if any(word in predicate for word in ['lead', 'manage', 'direct', 'head']):
            return 'LEADS'
        
        if any(word in predicate for word in ['create', 'found', 'establish', 'build']):
            return 'CREATED'
        
        if any(word in predicate for word in ['study', 'attend', 'graduate']):
            return 'STUDIED_AT'
        
        return None
    
    def _get_predicate_category(self, normalized_predicate: str) -> str:
        """Get category for normalized predicate.
        
        Args:
            normalized_predicate: Normalized predicate
            
        Returns:
            Predicate category
        """
        for category, data in self.relationship_categories.items():
            if data['standard_form'] == normalized_predicate:
                return category
        
        return 'OTHER'
    
    async def _calculate_normalization_stats(
        self, 
        original_triplets: List[Dict[str, Any]], 
        normalized_triplets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate normalization statistics.
        
        Args:
            original_triplets: Original triplets
            normalized_triplets: Normalized triplets
            
        Returns:
            Normalization statistics
        """
        method_counts = defaultdict(int)
        category_counts = defaultdict(int)
        confidence_sum = 0.0
        
        for triplet in normalized_triplets:
            norm_info = triplet.get('predicate_normalization', {})
            method = norm_info.get('method', 'unknown')
            category = norm_info.get('category', 'OTHER')
            confidence = norm_info.get('confidence', 0.0)
            
            method_counts[method] += 1
            category_counts[category] += 1
            confidence_sum += confidence
        
        avg_confidence = confidence_sum / len(normalized_triplets) if normalized_triplets else 0
        
        return {
            'normalization_rate': len(normalized_triplets) / len(original_triplets) if original_triplets else 0,
            'average_confidence': avg_confidence,
            'method_distribution': dict(method_counts),
            'category_distribution': dict(category_counts)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get normalizer statistics and configuration.
        
        Returns:
            Dictionary with normalizer statistics
        """
        return {
            'normalizer_name': 'PredicateNormalizer',
            'relationship_categories': len(self.relationship_categories),
            'total_patterns': sum(len(data['patterns']) for data in self.relationship_categories.values()),
            'verb_transformations': len(self.verb_transformations),
            'multilingual_predicates': len(self.multilingual_predicates),
            'exact_match_confidence': self.exact_match_confidence,
            'pattern_match_confidence': self.pattern_match_confidence,
            'semantic_match_confidence': self.semantic_match_confidence
        }
```

### 2. Integration with Triplet Processor

**File**: Update `packages/morag-graph/src/morag_graph/processors/triplet_processor.py`

Add predicate normalization integration:

```python
# Add import
from morag_graph.normalizers.predicate_normalizer import PredicateNormalizer

# Update TripletProcessor.__init__
def __init__(self, config: Optional[Dict[str, Any]] = None):
    # ... existing code ...
    self.predicate_normalizer = PredicateNormalizer(config)

# Add method for normalized processing
async def process_triplets_with_normalization(self, raw_triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process triplets with predicate normalization.
    
    Args:
        raw_triplets: Raw triplets from OpenIE
        
    Returns:
        Processed triplets with normalized predicates
    """
    # Basic triplet processing
    processed_triplets = await self.process_triplets(raw_triplets)
    
    # Normalize predicates
    normalized_triplets = await self.predicate_normalizer.normalize_predicates(processed_triplets)
    
    return normalized_triplets
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_predicate_normalizer.py`

```python
"""Tests for predicate normalizer."""

import pytest
from morag_graph.normalizers.predicate_normalizer import PredicateNormalizer

class TestPredicateNormalizer:
    
    def test_initialization(self):
        """Test normalizer initialization."""
        normalizer = PredicateNormalizer()
        assert len(normalizer.relationship_categories) > 0
        assert len(normalizer.pattern_to_category) > 0
    
    @pytest.mark.asyncio
    async def test_clean_predicate(self):
        """Test predicate cleaning."""
        normalizer = PredicateNormalizer()
        
        # Test noise word removal
        result = await normalizer._clean_predicate("is the CEO of")
        assert "ceo of" in result.lower()
        
        # Test verb transformation
        result = await normalizer._clean_predicate("works at")
        assert "work at" in result
    
    @pytest.mark.asyncio
    async def test_normalize_predicate(self):
        """Test predicate normalization."""
        normalizer = PredicateNormalizer()
        
        # Test exact match
        result, confidence, method = await normalizer._normalize_predicate("works at")
        assert result == "WORKS_AT"
        assert method == "exact_match"
        assert confidence == 1.0
        
        # Test pattern match
        result, confidence, method = await normalizer._normalize_predicate("employed by")
        assert result == "WORKS_AT"
        assert method in ["exact_match", "pattern_match"]
    
    @pytest.mark.asyncio
    async def test_normalize_triplets(self):
        """Test triplet predicate normalization."""
        normalizer = PredicateNormalizer()
        
        triplets = [{
            'subject': 'John',
            'predicate': 'is the CEO of',
            'object': 'Microsoft',
            'confidence': 0.8
        }]
        
        result = await normalizer.normalize_predicates(triplets)
        
        assert len(result) == 1
        assert 'original_predicate' in result[0]
        assert 'predicate_normalization' in result[0]
        assert result[0]['predicate_normalization']['category'] in normalizer.relationship_categories
```

## Acceptance Criteria

- [ ] PredicateNormalizer class implemented with comprehensive mapping rules
- [ ] Support for exact, pattern, semantic, and rule-based matching
- [ ] Multilingual predicate normalization capabilities
- [ ] Integration with triplet processing pipeline
- [ ] Confidence scoring for normalization quality
- [ ] Predicate categorization and taxonomy
- [ ] Comprehensive unit tests with >90% coverage
- [ ] Performance optimization for large predicate sets
- [ ] Proper logging and error handling
- [ ] Statistics and monitoring capabilities

## Dependencies
- Task 1.3: Basic Triplet Extraction and Validation
- Task 2.1: Entity Linking Between OpenIE and spaCy NER
- Task 2.2: Entity Normalization and Canonical Mapping

## Estimated Effort
- **Development**: 8-10 hours
- **Testing**: 4-5 hours
- **Integration**: 2-3 hours
- **Total**: 14-18 hours

## Notes
- Focus on creating meaningful, consistent relationship types
- Consider domain-specific predicate patterns
- Plan for extensible predicate taxonomy
- Implement caching for frequently normalized predicates
