# Task 1.3: Basic Triplet Extraction and Validation

## Objective
Implement basic triplet extraction functionality with validation and quality assessment to ensure reliable (subject, predicate, object) extraction from processed sentences.

## Scope
- Create triplet extractor with validation logic
- Implement quality scoring for extracted triplets
- Add filtering and deduplication mechanisms
- Create validation rules for triplet components
- **MANDATORY**: Test thoroughly before proceeding to Phase 2

## Implementation Details

### 1. Create Triplet Processor

**File**: `packages/morag-graph/src/morag_graph/processors/triplet_processor.py`

```python
"""Triplet processing and validation for OpenIE."""

import re
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
import structlog
from collections import defaultdict

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

class TripletProcessor:
    """Processor for triplet validation and quality assessment."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize triplet processor.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        self._load_validation_rules()
    
    def _load_validation_rules(self) -> None:
        """Load validation rules and patterns."""
        # Common invalid predicates to filter out
        self.invalid_predicates = {
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'can', 'could', 'should', 'may', 'might'
        }
        
        # Minimum lengths for triplet components
        self.min_subject_length = self.config.get('min_subject_length', 2)
        self.min_predicate_length = self.config.get('min_predicate_length', 3)
        self.min_object_length = self.config.get('min_object_length', 2)
        
        # Quality thresholds
        self.min_triplet_quality = self.config.get('min_triplet_quality', 0.5)
        self.min_confidence = self.config.get('min_confidence', 0.6)
    
    async def process_triplets(self, raw_triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and validate extracted triplets.
        
        Args:
            raw_triplets: Raw triplets from OpenIE extraction
            
        Returns:
            List of validated and scored triplets
            
        Raises:
            ProcessingError: If processing fails
        """
        if not raw_triplets:
            return []
        
        try:
            # Validate individual triplets
            validated_triplets = []
            for triplet in raw_triplets:
                processed_triplet = await self._process_single_triplet(triplet)
                if processed_triplet:
                    validated_triplets.append(processed_triplet)
            
            # Deduplicate triplets
            deduplicated_triplets = await self._deduplicate_triplets(validated_triplets)
            
            # Filter by quality and confidence
            filtered_triplets = await self._filter_triplets(deduplicated_triplets)
            
            logger.info(
                "Triplet processing completed",
                raw_count=len(raw_triplets),
                validated_count=len(validated_triplets),
                deduplicated_count=len(deduplicated_triplets),
                final_count=len(filtered_triplets)
            )
            
            return filtered_triplets
            
        except Exception as e:
            logger.error("Triplet processing failed", error=str(e))
            raise ProcessingError(f"Triplet processing failed: {str(e)}")
    
    async def _process_single_triplet(self, triplet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate a single triplet.
        
        Args:
            triplet: Raw triplet dictionary
            
        Returns:
            Processed triplet or None if invalid
        """
        try:
            # Extract and clean components
            subject = self._clean_component(triplet.get('subject', ''))
            predicate = self._clean_component(triplet.get('predicate', ''))
            obj = self._clean_component(triplet.get('object', ''))
            
            # Basic validation
            if not self._validate_components(subject, predicate, obj):
                return None
            
            # Calculate quality score
            quality_score = await self._calculate_triplet_quality(subject, predicate, obj)
            
            # Create processed triplet
            processed_triplet = {
                'subject': subject,
                'predicate': predicate,
                'object': obj,
                'confidence': triplet.get('confidence', 0.0),
                'quality_score': quality_score,
                'source_sentence': triplet.get('source_sentence', ''),
                'extraction_method': triplet.get('extraction_method', 'unknown'),
                'sentence_index': triplet.get('sentence_index', 0),
                'sentence_quality': triplet.get('sentence_quality', 0.0),
                'sentence_word_count': triplet.get('sentence_word_count', 0),
                'triplet_id': self._generate_triplet_id(subject, predicate, obj)
            }
            
            return processed_triplet
            
        except Exception as e:
            logger.error(
                "Single triplet processing failed",
                error=str(e),
                triplet=triplet
            )
            return None
    
    def _clean_component(self, component: str) -> str:
        """Clean and normalize a triplet component.
        
        Args:
            component: Raw component text
            
        Returns:
            Cleaned component
        """
        if not component:
            return ''
        
        # Remove extra whitespace
        component = re.sub(r'\s+', ' ', component.strip())
        
        # Remove leading/trailing punctuation except meaningful ones
        component = re.sub(r'^[^\w\s]+|[^\w\s.!?]+$', '', component)
        
        # Handle quotes and parentheses
        component = re.sub(r'^["\']|["\']$', '', component)
        component = re.sub(r'^\(|\)$', '', component)
        
        # Normalize case for predicates (keep original case for subjects/objects)
        return component.strip()
    
    def _validate_components(self, subject: str, predicate: str, obj: str) -> bool:
        """Validate triplet components.
        
        Args:
            subject: Subject component
            predicate: Predicate component
            obj: Object component
            
        Returns:
            True if all components are valid
        """
        # Check minimum lengths
        if (len(subject) < self.min_subject_length or
            len(predicate) < self.min_predicate_length or
            len(obj) < self.min_object_length):
            return False
        
        # Check for empty or whitespace-only components
        if not all([subject.strip(), predicate.strip(), obj.strip()]):
            return False
        
        # Check for invalid predicates
        if predicate.lower().strip() in self.invalid_predicates:
            return False
        
        # Check for circular references (subject == object)
        if subject.lower().strip() == obj.lower().strip():
            return False
        
        # Check for overly generic components
        generic_terms = {'thing', 'something', 'someone', 'it', 'this', 'that'}
        if (subject.lower().strip() in generic_terms or
            obj.lower().strip() in generic_terms):
            return False
        
        return True
    
    async def _calculate_triplet_quality(self, subject: str, predicate: str, obj: str) -> float:
        """Calculate quality score for a triplet.
        
        Args:
            subject: Subject component
            predicate: Predicate component
            obj: Object component
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Length-based scoring
        if len(subject) < 5:
            score -= 0.1
        if len(predicate) < 5:
            score -= 0.1
        if len(obj) < 5:
            score -= 0.1
        
        # Specificity scoring
        if self._is_specific_entity(subject):
            score += 0.1
        if self._is_specific_entity(obj):
            score += 0.1
        if self._is_meaningful_predicate(predicate):
            score += 0.1
        
        # Penalize overly long components (likely extraction errors)
        if len(subject) > 50:
            score -= 0.2
        if len(predicate) > 30:
            score -= 0.2
        if len(obj) > 50:
            score -= 0.2
        
        # Penalize components with too many special characters
        for component in [subject, predicate, obj]:
            special_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', component)) / len(component)
            if special_ratio > 0.3:
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _is_specific_entity(self, entity: str) -> bool:
        """Check if entity appears to be specific (proper noun, etc.).
        
        Args:
            entity: Entity text
            
        Returns:
            True if entity appears specific
        """
        # Check for capitalization patterns
        words = entity.split()
        capitalized_words = [w for w in words if w[0].isupper()]
        
        # Most words capitalized suggests proper noun
        if len(capitalized_words) / len(words) > 0.5:
            return True
        
        # Check for specific patterns
        specific_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Person names
            r'\b[A-Z][a-z]+ (Inc|Corp|Ltd|LLC)\b',  # Company names
            r'\b\d+(\.\d+)?\s*(kg|lb|meter|mile|dollar|euro)\b',  # Measurements
        ]
        
        for pattern in specific_patterns:
            if re.search(pattern, entity):
                return True
        
        return False
    
    def _is_meaningful_predicate(self, predicate: str) -> bool:
        """Check if predicate is meaningful for relation extraction.
        
        Args:
            predicate: Predicate text
            
        Returns:
            True if predicate is meaningful
        """
        # Check for action verbs and meaningful relations
        meaningful_patterns = [
            r'\b(work|live|study|teach|manage|lead|create|develop|build)\b',
            r'\b(love|like|hate|prefer|enjoy|appreciate)\b',
            r'\b(own|have|possess|contain|include)\b',
            r'\b(located|based|situated|founded|established)\b',
            r'\b(member|part|component|element)\b'
        ]
        
        for pattern in meaningful_patterns:
            if re.search(pattern, predicate.lower()):
                return True
        
        # Check for compound predicates (more specific)
        if len(predicate.split()) > 1:
            return True
        
        return False
    
    async def _deduplicate_triplets(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate triplets.
        
        Args:
            triplets: List of triplets to deduplicate
            
        Returns:
            Deduplicated triplets
        """
        seen_triplets = {}
        deduplicated = []
        
        for triplet in triplets:
            triplet_key = (
                triplet['subject'].lower().strip(),
                triplet['predicate'].lower().strip(),
                triplet['object'].lower().strip()
            )
            
            if triplet_key not in seen_triplets:
                seen_triplets[triplet_key] = triplet
                deduplicated.append(triplet)
            else:
                # Keep the one with higher confidence
                existing = seen_triplets[triplet_key]
                if triplet['confidence'] > existing['confidence']:
                    # Replace in both dict and list
                    seen_triplets[triplet_key] = triplet
                    for i, t in enumerate(deduplicated):
                        if t['triplet_id'] == existing['triplet_id']:
                            deduplicated[i] = triplet
                            break
        
        return deduplicated
    
    async def _filter_triplets(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter triplets by quality and confidence thresholds.
        
        Args:
            triplets: Triplets to filter
            
        Returns:
            Filtered triplets
        """
        filtered = []
        
        for triplet in triplets:
            if (triplet['quality_score'] >= self.min_triplet_quality and
                triplet['confidence'] >= self.min_confidence):
                filtered.append(triplet)
        
        return filtered
    
    def _generate_triplet_id(self, subject: str, predicate: str, obj: str) -> str:
        """Generate unique ID for triplet.
        
        Args:
            subject: Subject component
            predicate: Predicate component
            obj: Object component
            
        Returns:
            Unique triplet identifier
        """
        import hashlib
        
        triplet_string = f"{subject.lower()}|{predicate.lower()}|{obj.lower()}"
        return hashlib.md5(triplet_string.encode()).hexdigest()[:12]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics and configuration.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            'processor_name': 'TripletProcessor',
            'min_subject_length': self.min_subject_length,
            'min_predicate_length': self.min_predicate_length,
            'min_object_length': self.min_object_length,
            'min_triplet_quality': self.min_triplet_quality,
            'min_confidence': self.min_confidence,
            'invalid_predicates_count': len(self.invalid_predicates)
        }
```

### 2. Create Triplet Validator

**File**: `packages/morag-graph/src/morag_graph/validators/triplet_validator.py`

```python
"""Validation utilities for OpenIE triplets."""

import re
from typing import List, Dict, Any, Set, Tuple
import structlog

logger = structlog.get_logger(__name__)

class TripletValidator:
    """Validator for OpenIE triplets."""
    
    def __init__(self):
        """Initialize validator with validation rules."""
        self.load_validation_rules()
    
    def load_validation_rules(self) -> None:
        """Load validation rules and patterns."""
        # Patterns for valid entities
        self.entity_patterns = [
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Proper nouns
            r'^[A-Z][a-z]+(?:\s+[a-z]+)*$',       # Mixed case
            r'^\d+(?:\.\d+)?\s*[a-zA-Z]+$',       # Numbers with units
        ]
        
        # Patterns for valid predicates
        self.predicate_patterns = [
            r'^[a-z]+(?:\s+[a-z]+)*$',            # Lowercase verbs
            r'^[a-z]+\s+(?:of|in|at|on|by|with|for|to|from)$',  # Prepositions
        ]
    
    def validate_triplet(self, triplet: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single triplet.
        
        Args:
            triplet: Triplet to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        required_fields = ['subject', 'predicate', 'object']
        for field in required_fields:
            if field not in triplet or not triplet[field]:
                issues.append(f"Missing or empty {field}")
        
        if issues:
            return False, issues
        
        # Validate components
        subject_valid, subject_issues = self._validate_component(
            triplet['subject'], 'subject'
        )
        predicate_valid, predicate_issues = self._validate_component(
            triplet['predicate'], 'predicate'
        )
        object_valid, object_issues = self._validate_component(
            triplet['object'], 'object'
        )
        
        issues.extend(subject_issues)
        issues.extend(predicate_issues)
        issues.extend(object_issues)
        
        # Additional triplet-level validations
        if triplet['subject'].lower() == triplet['object'].lower():
            issues.append("Subject and object are identical")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _validate_component(self, component: str, component_type: str) -> Tuple[bool, List[str]]:
        """Validate a triplet component.
        
        Args:
            component: Component text
            component_type: Type of component (subject, predicate, object)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Basic checks
        if not component or not component.strip():
            issues.append(f"{component_type} is empty")
            return False, issues
        
        if len(component.strip()) < 2:
            issues.append(f"{component_type} is too short")
        
        if len(component) > 100:
            issues.append(f"{component_type} is too long")
        
        # Check for excessive special characters
        special_char_count = len(re.findall(r'[^a-zA-Z0-9\s]', component))
        if special_char_count / len(component) > 0.5:
            issues.append(f"{component_type} has too many special characters")
        
        return len(issues) == 0, issues
    
    def validate_batch(self, triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a batch of triplets.
        
        Args:
            triplets: List of triplets to validate
            
        Returns:
            Validation summary
        """
        valid_count = 0
        invalid_count = 0
        all_issues = []
        
        for i, triplet in enumerate(triplets):
            is_valid, issues = self.validate_triplet(triplet)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                all_issues.extend([f"Triplet {i}: {issue}" for issue in issues])
        
        return {
            'total_triplets': len(triplets),
            'valid_triplets': valid_count,
            'invalid_triplets': invalid_count,
            'validation_rate': valid_count / len(triplets) if triplets else 0,
            'issues': all_issues
        }
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_triplet_processor.py`

```python
"""Tests for triplet processor."""

import pytest
from morag_graph.processors.triplet_processor import TripletProcessor

class TestTripletProcessor:
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = TripletProcessor()
        assert processor.min_triplet_quality == 0.5
        assert processor.min_confidence == 0.6
    
    @pytest.mark.asyncio
    async def test_process_triplets_success(self):
        """Test successful triplet processing."""
        processor = TripletProcessor()
        
        raw_triplets = [{
            'subject': 'John Smith',
            'predicate': 'works at',
            'object': 'Microsoft Corporation',
            'confidence': 0.8,
            'source_sentence': 'John Smith works at Microsoft Corporation.'
        }]
        
        result = await processor.process_triplets(raw_triplets)
        
        assert len(result) == 1
        assert result[0]['subject'] == 'John Smith'
        assert result[0]['predicate'] == 'works at'
        assert result[0]['object'] == 'Microsoft Corporation'
        assert 'quality_score' in result[0]
        assert 'triplet_id' in result[0]
    
    def test_validate_components(self):
        """Test component validation."""
        processor = TripletProcessor()
        
        # Valid components
        assert processor._validate_components('John', 'loves', 'Mary')
        
        # Invalid - too short
        assert not processor._validate_components('J', 'loves', 'Mary')
        
        # Invalid - circular reference
        assert not processor._validate_components('John', 'is', 'John')
        
        # Invalid - generic terms
        assert not processor._validate_components('something', 'does', 'thing')
    
    @pytest.mark.asyncio
    async def test_calculate_triplet_quality(self):
        """Test quality calculation."""
        processor = TripletProcessor()
        
        # High quality triplet
        quality = await processor._calculate_triplet_quality(
            'John Smith', 'works at', 'Microsoft Corporation'
        )
        assert quality > 0.7
        
        # Low quality triplet
        quality = await processor._calculate_triplet_quality(
            'a', 'is', 'b'
        )
        assert quality < 0.5
```

## Acceptance Criteria

- [ ] TripletProcessor class implemented with validation logic
- [ ] Quality scoring system for triplets
- [ ] Deduplication mechanism for identical triplets
- [ ] Filtering by confidence and quality thresholds
- [ ] Component validation (subject, predicate, object)
- [ ] TripletValidator utility class
- [ ] Comprehensive unit tests with >90% coverage
- [ ] Integration with OpenIE service
- [ ] Proper logging and error handling
- [ ] Performance optimization for large triplet sets

## Dependencies
- Task 1.1: OpenIE Dependency Integration and Service Wrapper
- Task 1.2: Sentence Segmentation and Preprocessing Pipeline

## Estimated Effort
- **Development**: 6-8 hours
- **Testing**: 4-5 hours
- **Integration**: 2-3 hours
- **Total**: 12-16 hours

## Notes
- Focus on quality over quantity for triplet extraction
- Implement robust validation to filter out noise
- Consider domain-specific validation rules for future enhancement
- Ensure scalability for processing large document sets
