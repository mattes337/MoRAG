# Task 2.1: Entity Linking Between OpenIE and spaCy NER

## Objective
Implement entity linking functionality to map OpenIE-extracted entities to canonical entities identified by spaCy NER, ensuring consistent entity representation across the knowledge graph.

## Scope
- Create entity linking service for OpenIE-spaCy integration
- Implement fuzzy matching and similarity algorithms
- Add entity resolution and canonical mapping
- Handle partial matches and entity variations
- **MANDATORY**: Test thoroughly before proceeding to Task 2.2

## Implementation Details

### 1. Create Entity Linker

**File**: `packages/morag-graph/src/morag_graph/normalizers/entity_linker.py`

```python
"""Entity linking between OpenIE and spaCy NER."""

import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
import structlog
from difflib import SequenceMatcher
from collections import defaultdict

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

class EntityLinker:
    """Links OpenIE entities to spaCy canonical entities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize entity linker.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Similarity thresholds
        self.exact_match_threshold = self.config.get('exact_match_threshold', 1.0)
        self.high_similarity_threshold = self.config.get('high_similarity_threshold', 0.9)
        self.medium_similarity_threshold = self.config.get('medium_similarity_threshold', 0.7)
        self.min_similarity_threshold = self.config.get('min_similarity_threshold', 0.5)
        
        # Entity type mappings
        self.entity_type_mappings = {
            'PERSON': ['person', 'people', 'individual', 'human'],
            'ORG': ['organization', 'company', 'corporation', 'institution'],
            'GPE': ['location', 'place', 'country', 'city', 'state'],
            'PRODUCT': ['product', 'item', 'goods', 'service'],
            'EVENT': ['event', 'occurrence', 'happening'],
            'WORK_OF_ART': ['artwork', 'creation', 'work'],
            'LAW': ['law', 'regulation', 'rule'],
            'LANGUAGE': ['language', 'tongue', 'dialect'],
            'DATE': ['date', 'time', 'period'],
            'TIME': ['time', 'moment', 'hour'],
            'PERCENT': ['percentage', 'percent', 'rate'],
            'MONEY': ['money', 'currency', 'amount'],
            'QUANTITY': ['quantity', 'amount', 'number'],
            'ORDINAL': ['ordinal', 'position', 'rank'],
            'CARDINAL': ['cardinal', 'number', 'count']
        }
    
    async def link_entities(
        self, 
        openie_triplets: List[Dict[str, Any]], 
        spacy_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Link OpenIE entities to spaCy entities.
        
        Args:
            openie_triplets: Triplets from OpenIE extraction
            spacy_entities: Entities from spaCy NER
            
        Returns:
            Triplets with linked entities
            
        Raises:
            ProcessingError: If linking fails
        """
        if not openie_triplets:
            return []
        
        try:
            # Create entity lookup structures
            entity_lookup = await self._create_entity_lookup(spacy_entities)
            
            # Link entities in each triplet
            linked_triplets = []
            for triplet in openie_triplets:
                linked_triplet = await self._link_triplet_entities(triplet, entity_lookup)
                linked_triplets.append(linked_triplet)
            
            # Calculate linking statistics
            linking_stats = await self._calculate_linking_stats(linked_triplets)
            
            logger.info(
                "Entity linking completed",
                triplets_processed=len(openie_triplets),
                spacy_entities=len(spacy_entities),
                **linking_stats
            )
            
            return linked_triplets
            
        except Exception as e:
            logger.error("Entity linking failed", error=str(e))
            raise ProcessingError(f"Entity linking failed: {str(e)}")
    
    async def _create_entity_lookup(self, spacy_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create lookup structures for efficient entity matching.
        
        Args:
            spacy_entities: Entities from spaCy NER
            
        Returns:
            Entity lookup dictionary
        """
        lookup = {
            'exact_matches': {},
            'normalized_matches': {},
            'type_groups': defaultdict(list),
            'all_entities': spacy_entities
        }
        
        for entity in spacy_entities:
            entity_text = entity.get('text', '').strip()
            entity_type = entity.get('label', 'UNKNOWN')
            entity_id = entity.get('entity_id', '')
            
            if not entity_text:
                continue
            
            # Exact match lookup
            lookup['exact_matches'][entity_text] = entity
            
            # Normalized match lookup
            normalized = self._normalize_entity_text(entity_text)
            lookup['normalized_matches'][normalized] = entity
            
            # Type-based grouping
            lookup['type_groups'][entity_type].append(entity)
            
            # Add variations (if entity has aliases or variations)
            variations = entity.get('variations', [])
            for variation in variations:
                lookup['exact_matches'][variation] = entity
                normalized_var = self._normalize_entity_text(variation)
                lookup['normalized_matches'][normalized_var] = entity
        
        return lookup
    
    async def _link_triplet_entities(
        self, 
        triplet: Dict[str, Any], 
        entity_lookup: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Link entities in a single triplet.
        
        Args:
            triplet: OpenIE triplet
            entity_lookup: Entity lookup structures
            
        Returns:
            Triplet with linked entities
        """
        linked_triplet = triplet.copy()
        
        # Link subject
        subject_link = await self._find_entity_link(
            triplet['subject'], entity_lookup
        )
        if subject_link:
            linked_triplet['subject_entity'] = subject_link
        
        # Link object
        object_link = await self._find_entity_link(
            triplet['object'], entity_lookup
        )
        if object_link:
            linked_triplet['object_entity'] = object_link
        
        # Add linking metadata
        linked_triplet['entity_linking'] = {
            'subject_linked': bool(subject_link),
            'object_linked': bool(object_link),
            'linking_confidence': self._calculate_linking_confidence(subject_link, object_link)
        }
        
        return linked_triplet
    
    async def _find_entity_link(
        self, 
        openie_entity: str, 
        entity_lookup: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find the best matching spaCy entity for an OpenIE entity.
        
        Args:
            openie_entity: Entity text from OpenIE
            entity_lookup: Entity lookup structures
            
        Returns:
            Best matching spaCy entity or None
        """
        if not openie_entity or not openie_entity.strip():
            return None
        
        entity_text = openie_entity.strip()
        
        # 1. Try exact match
        exact_match = entity_lookup['exact_matches'].get(entity_text)
        if exact_match:
            return {
                **exact_match,
                'match_type': 'exact',
                'match_confidence': 1.0,
                'original_text': entity_text
            }
        
        # 2. Try normalized match
        normalized = self._normalize_entity_text(entity_text)
        normalized_match = entity_lookup['normalized_matches'].get(normalized)
        if normalized_match:
            return {
                **normalized_match,
                'match_type': 'normalized',
                'match_confidence': 0.95,
                'original_text': entity_text
            }
        
        # 3. Try fuzzy matching
        fuzzy_match = await self._find_fuzzy_match(entity_text, entity_lookup)
        if fuzzy_match:
            return fuzzy_match
        
        # 4. Try partial matching
        partial_match = await self._find_partial_match(entity_text, entity_lookup)
        if partial_match:
            return partial_match
        
        return None
    
    def _normalize_entity_text(self, text: str) -> str:
        """Normalize entity text for matching.
        
        Args:
            text: Entity text to normalize
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        normalized = text.lower().strip()
        
        # Remove common prefixes/suffixes
        prefixes = ['the ', 'a ', 'an ', 'mr. ', 'mrs. ', 'dr. ', 'prof. ']
        suffixes = [' inc', ' corp', ' ltd', ' llc', ' co']
        
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        
        # Remove extra whitespace and punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    async def _find_fuzzy_match(
        self, 
        entity_text: str, 
        entity_lookup: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find fuzzy matches using string similarity.
        
        Args:
            entity_text: Entity text to match
            entity_lookup: Entity lookup structures
            
        Returns:
            Best fuzzy match or None
        """
        best_match = None
        best_similarity = 0.0
        
        normalized_target = self._normalize_entity_text(entity_text)
        
        # Check against all entities
        for spacy_entity in entity_lookup['all_entities']:
            spacy_text = spacy_entity.get('text', '')
            normalized_spacy = self._normalize_entity_text(spacy_text)
            
            # Calculate similarity
            similarity = SequenceMatcher(None, normalized_target, normalized_spacy).ratio()
            
            # Also check against variations
            variations = spacy_entity.get('variations', [])
            for variation in variations:
                normalized_var = self._normalize_entity_text(variation)
                var_similarity = SequenceMatcher(None, normalized_target, normalized_var).ratio()
                similarity = max(similarity, var_similarity)
            
            if similarity > best_similarity and similarity >= self.min_similarity_threshold:
                best_similarity = similarity
                best_match = spacy_entity
        
        if best_match and best_similarity >= self.min_similarity_threshold:
            match_type = 'high_fuzzy' if best_similarity >= self.high_similarity_threshold else 'medium_fuzzy'
            return {
                **best_match,
                'match_type': match_type,
                'match_confidence': best_similarity,
                'original_text': entity_text
            }
        
        return None
    
    async def _find_partial_match(
        self, 
        entity_text: str, 
        entity_lookup: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find partial matches (substring matching).
        
        Args:
            entity_text: Entity text to match
            entity_lookup: Entity lookup structures
            
        Returns:
            Best partial match or None
        """
        normalized_target = self._normalize_entity_text(entity_text)
        target_words = set(normalized_target.split())
        
        if len(target_words) < 2:  # Skip single-word partial matching
            return None
        
        best_match = None
        best_score = 0.0
        
        for spacy_entity in entity_lookup['all_entities']:
            spacy_text = spacy_entity.get('text', '')
            normalized_spacy = self._normalize_entity_text(spacy_text)
            spacy_words = set(normalized_spacy.split())
            
            # Calculate word overlap
            common_words = target_words.intersection(spacy_words)
            if len(common_words) >= 2:  # At least 2 words in common
                overlap_score = len(common_words) / max(len(target_words), len(spacy_words))
                
                if overlap_score > best_score and overlap_score >= 0.5:
                    best_score = overlap_score
                    best_match = spacy_entity
        
        if best_match and best_score >= 0.5:
            return {
                **best_match,
                'match_type': 'partial',
                'match_confidence': best_score,
                'original_text': entity_text
            }
        
        return None
    
    def _calculate_linking_confidence(
        self, 
        subject_link: Optional[Dict[str, Any]], 
        object_link: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall linking confidence for a triplet.
        
        Args:
            subject_link: Subject entity link
            object_link: Object entity link
            
        Returns:
            Overall linking confidence
        """
        confidences = []
        
        if subject_link:
            confidences.append(subject_link.get('match_confidence', 0.0))
        
        if object_link:
            confidences.append(object_link.get('match_confidence', 0.0))
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
    
    async def _calculate_linking_stats(self, linked_triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate entity linking statistics.
        
        Args:
            linked_triplets: Triplets with entity linking
            
        Returns:
            Linking statistics
        """
        total_entities = len(linked_triplets) * 2  # Subject + object per triplet
        linked_entities = 0
        exact_matches = 0
        fuzzy_matches = 0
        partial_matches = 0
        
        for triplet in linked_triplets:
            linking_info = triplet.get('entity_linking', {})
            
            if linking_info.get('subject_linked'):
                linked_entities += 1
                subject_entity = triplet.get('subject_entity', {})
                match_type = subject_entity.get('match_type', '')
                if match_type == 'exact':
                    exact_matches += 1
                elif 'fuzzy' in match_type:
                    fuzzy_matches += 1
                elif match_type == 'partial':
                    partial_matches += 1
            
            if linking_info.get('object_linked'):
                linked_entities += 1
                object_entity = triplet.get('object_entity', {})
                match_type = object_entity.get('match_type', '')
                if match_type == 'exact':
                    exact_matches += 1
                elif 'fuzzy' in match_type:
                    fuzzy_matches += 1
                elif match_type == 'partial':
                    partial_matches += 1
        
        linking_rate = linked_entities / total_entities if total_entities > 0 else 0
        
        return {
            'total_entities': total_entities,
            'linked_entities': linked_entities,
            'linking_rate': linking_rate,
            'exact_matches': exact_matches,
            'fuzzy_matches': fuzzy_matches,
            'partial_matches': partial_matches
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get linker statistics and configuration.
        
        Returns:
            Dictionary with linker statistics
        """
        return {
            'linker_name': 'EntityLinker',
            'exact_match_threshold': self.exact_match_threshold,
            'high_similarity_threshold': self.high_similarity_threshold,
            'medium_similarity_threshold': self.medium_similarity_threshold,
            'min_similarity_threshold': self.min_similarity_threshold,
            'supported_entity_types': list(self.entity_type_mappings.keys())
        }
```

### 2. Integration with OpenIE Pipeline

**File**: Update `packages/morag-graph/src/morag_graph/services/openie_service.py`

Add entity linking integration:

```python
# Add import
from morag_graph.normalizers.entity_linker import EntityLinker

# Update OpenIEService.__init__
def __init__(self, config: Optional[Dict[str, Any]] = None):
    # ... existing code ...
    self.entity_linker = EntityLinker(config)

# Add new method for integrated extraction
async def extract_triplets_with_entities(
    self, 
    text: str, 
    spacy_entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Extract triplets and link to spaCy entities.
    
    Args:
        text: Input text to process
        spacy_entities: Entities from spaCy NER
        
    Returns:
        List of triplets with linked entities
    """
    # Extract basic triplets
    triplets = await self.extract_triplets(text)
    
    # Link entities
    linked_triplets = await self.entity_linker.link_entities(triplets, spacy_entities)
    
    return linked_triplets
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_entity_linker.py`

```python
"""Tests for entity linker."""

import pytest
from morag_graph.normalizers.entity_linker import EntityLinker

class TestEntityLinker:
    
    def test_initialization(self):
        """Test linker initialization."""
        linker = EntityLinker()
        assert linker.min_similarity_threshold == 0.5
        assert linker.high_similarity_threshold == 0.9
    
    def test_normalize_entity_text(self):
        """Test entity text normalization."""
        linker = EntityLinker()
        
        # Test basic normalization
        assert linker._normalize_entity_text("John Smith") == "john smith"
        
        # Test prefix removal
        assert linker._normalize_entity_text("The Microsoft Corporation") == "microsoft corporation"
        
        # Test suffix removal
        assert linker._normalize_entity_text("Apple Inc") == "apple"
    
    @pytest.mark.asyncio
    async def test_link_entities_success(self):
        """Test successful entity linking."""
        linker = EntityLinker()
        
        openie_triplets = [{
            'subject': 'John Smith',
            'predicate': 'works at',
            'object': 'Microsoft',
            'confidence': 0.8
        }]
        
        spacy_entities = [
            {
                'text': 'John Smith',
                'label': 'PERSON',
                'entity_id': 'person_001'
            },
            {
                'text': 'Microsoft Corporation',
                'label': 'ORG',
                'entity_id': 'org_001',
                'variations': ['Microsoft', 'MSFT']
            }
        ]
        
        result = await linker.link_entities(openie_triplets, spacy_entities)
        
        assert len(result) == 1
        assert 'subject_entity' in result[0]
        assert 'object_entity' in result[0]
        assert result[0]['subject_entity']['match_type'] == 'exact'
        assert result[0]['object_entity']['match_type'] == 'exact'
    
    @pytest.mark.asyncio
    async def test_find_fuzzy_match(self):
        """Test fuzzy matching functionality."""
        linker = EntityLinker()
        
        entity_lookup = {
            'exact_matches': {},
            'normalized_matches': {},
            'type_groups': {},
            'all_entities': [
                {
                    'text': 'Microsoft Corporation',
                    'label': 'ORG',
                    'entity_id': 'org_001'
                }
            ]
        }
        
        # Should find fuzzy match
        result = await linker._find_fuzzy_match('Microsoft Corp', entity_lookup)
        assert result is not None
        assert result['match_type'] in ['high_fuzzy', 'medium_fuzzy']
        assert result['match_confidence'] > 0.5
```

## Acceptance Criteria

- [ ] EntityLinker class implemented with multiple matching strategies
- [ ] Exact, normalized, fuzzy, and partial matching algorithms
- [ ] Integration with OpenIE service for entity linking
- [ ] Confidence scoring for entity matches
- [ ] Support for entity variations and aliases
- [ ] Comprehensive unit tests with >90% coverage
- [ ] Performance optimization for large entity sets
- [ ] Proper logging and error handling
- [ ] Statistics and monitoring capabilities

## Dependencies
- Task 1.1: OpenIE Dependency Integration and Service Wrapper
- Task 1.2: Sentence Segmentation and Preprocessing Pipeline
- Task 1.3: Basic Triplet Extraction and Validation

## Estimated Effort
- **Development**: 8-10 hours
- **Testing**: 4-5 hours
- **Integration**: 3-4 hours
- **Total**: 15-19 hours

## Notes
- Focus on accuracy over speed for entity linking
- Consider domain-specific entity matching rules
- Implement caching for frequently matched entities
- Plan for multilingual entity matching in future phases
