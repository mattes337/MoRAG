# Task 2.2: Entity Normalization and Canonical Mapping

## Objective
Implement entity normalization to ensure consistent canonical forms for entities across the knowledge graph, handling variations, aliases, and multilingual representations.

## Scope
- Create entity normalization service
- Implement canonical entity mapping
- Add support for entity aliases and variations
- Handle multilingual entity normalization
- **MANDATORY**: Test thoroughly before proceeding to Task 2.3

## Implementation Details

### 1. Create Entity Normalizer

**File**: `packages/morag-graph/src/morag_graph/normalizers/entity_normalizer.py`

```python
"""Entity normalization for canonical mapping."""

import re
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
import structlog
from collections import defaultdict
import unicodedata

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

class EntityNormalizer:
    """Normalizes entities to canonical forms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize entity normalizer.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Load normalization rules
        self._load_normalization_rules()
        
        # Entity type specific normalizers
        self.type_normalizers = {
            'PERSON': self._normalize_person,
            'ORG': self._normalize_organization,
            'GPE': self._normalize_location,
            'PRODUCT': self._normalize_product,
            'EVENT': self._normalize_event,
            'DATE': self._normalize_date,
            'MONEY': self._normalize_money,
            'QUANTITY': self._normalize_quantity
        }
    
    def _load_normalization_rules(self) -> None:
        """Load normalization rules and patterns."""
        # Common abbreviations and their expansions
        self.abbreviations = {
            # Organizations
            'corp': 'corporation',
            'inc': 'incorporated',
            'ltd': 'limited',
            'llc': 'limited liability company',
            'co': 'company',
            'assoc': 'association',
            'inst': 'institute',
            'univ': 'university',
            
            # Titles
            'mr': 'mister',
            'mrs': 'missus',
            'ms': 'miss',
            'dr': 'doctor',
            'prof': 'professor',
            'sr': 'senior',
            'jr': 'junior',
            
            # Locations
            'st': 'street',
            'ave': 'avenue',
            'blvd': 'boulevard',
            'rd': 'road',
            'usa': 'united states of america',
            'uk': 'united kingdom',
            'uae': 'united arab emirates'
        }
        
        # Common entity variations
        self.entity_variations = {
            # Technology companies
            'microsoft': ['msft', 'microsoft corp', 'microsoft corporation'],
            'apple': ['apple inc', 'apple computer'],
            'google': ['alphabet', 'alphabet inc', 'google llc'],
            'amazon': ['amazon.com', 'amazon inc'],
            'facebook': ['meta', 'meta platforms'],
            
            # Countries
            'united states': ['usa', 'us', 'america', 'united states of america'],
            'united kingdom': ['uk', 'britain', 'great britain'],
            'germany': ['deutschland'],
            'spain': ['españa'],
            
            # Common names
            'john': ['johnny', 'jon'],
            'william': ['bill', 'billy', 'will'],
            'robert': ['bob', 'bobby', 'rob'],
            'michael': ['mike', 'mick']
        }
        
        # Multilingual mappings
        self.multilingual_mappings = {
            # Spanish to English
            'empresa': 'company',
            'corporación': 'corporation',
            'universidad': 'university',
            'instituto': 'institute',
            'presidente': 'president',
            'director': 'director',

            # German to English
            'unternehmen': 'company',
            'gesellschaft': 'company',
            'firma': 'company',
            'konzern': 'corporation',
            'universität': 'university',
            'institut': 'institute',
            'präsident': 'president',
            'direktor': 'director',
            'geschäftsführer': 'manager',
            'vorstand': 'board',
            'mitarbeiter': 'employee',
            'gründer': 'founder',

            # Common Spanish names
            'juan': 'john',
            'maría': 'mary',
            'josé': 'joseph',
            'carlos': 'charles',

            # Common German names
            'johann': 'john',
            'johannes': 'john',
            'wilhelm': 'william',
            'friedrich': 'frederick',
            'heinrich': 'henry',
            'maria': 'mary',
            'elisabeth': 'elizabeth',
            'katharina': 'catherine'
        }
    
    async def normalize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize a list of entities to canonical forms.
        
        Args:
            entities: List of entities to normalize
            
        Returns:
            List of normalized entities
            
        Raises:
            ProcessingError: If normalization fails
        """
        if not entities:
            return []
        
        try:
            normalized_entities = []
            
            for entity in entities:
                normalized_entity = await self._normalize_single_entity(entity)
                if normalized_entity:
                    normalized_entities.append(normalized_entity)
            
            # Create canonical mappings
            canonical_entities = await self._create_canonical_mappings(normalized_entities)
            
            logger.info(
                "Entity normalization completed",
                input_entities=len(entities),
                normalized_entities=len(normalized_entities),
                canonical_entities=len(canonical_entities)
            )
            
            return canonical_entities
            
        except Exception as e:
            logger.error("Entity normalization failed", error=str(e))
            raise ProcessingError(f"Entity normalization failed: {str(e)}")
    
    async def _normalize_single_entity(self, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a single entity.
        
        Args:
            entity: Entity to normalize
            
        Returns:
            Normalized entity or None if invalid
        """
        try:
            entity_text = entity.get('text', '').strip()
            entity_type = entity.get('label', 'UNKNOWN')
            
            if not entity_text:
                return None
            
            # Basic text normalization
            normalized_text = await self._normalize_text(entity_text)
            
            # Type-specific normalization
            if entity_type in self.type_normalizers:
                normalized_text = await self.type_normalizers[entity_type](normalized_text)
            
            # Create normalized entity
            normalized_entity = entity.copy()
            normalized_entity.update({
                'normalized_text': normalized_text,
                'original_text': entity_text,
                'canonical_form': normalized_text,  # Will be updated in canonical mapping
                'variations': self._get_entity_variations(normalized_text),
                'normalization_confidence': self._calculate_normalization_confidence(
                    entity_text, normalized_text
                )
            })
            
            return normalized_entity
            
        except Exception as e:
            logger.error(
                "Single entity normalization failed",
                error=str(e),
                entity=entity
            )
            return None
    
    async def _normalize_text(self, text: str) -> str:
        """Basic text normalization.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Unicode normalization
        normalized = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        normalized = normalized.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Expand abbreviations
        words = normalized.split()
        expanded_words = []
        for word in words:
            # Remove punctuation for lookup
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.abbreviations:
                expanded_words.append(self.abbreviations[clean_word])
            else:
                expanded_words.append(word)
        
        normalized = ' '.join(expanded_words)
        
        # Handle multilingual terms
        for foreign_term, english_term in self.multilingual_mappings.items():
            normalized = normalized.replace(foreign_term, english_term)
        
        # Remove common prefixes and suffixes
        normalized = self._remove_common_affixes(normalized)
        
        return normalized.strip()
    
    def _remove_common_affixes(self, text: str) -> str:
        """Remove common prefixes and suffixes.
        
        Args:
            text: Text to process
            
        Returns:
            Text with affixes removed
        """
        # Remove common prefixes
        prefixes = ['the ', 'a ', 'an ']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        
        # Remove common suffixes
        suffixes = [' inc', ' corp', ' ltd', ' llc', ' co', ' company']
        for suffix in suffixes:
            if text.endswith(suffix):
                text = text[:-len(suffix)]
                break
        
        return text.strip()
    
    async def _normalize_person(self, text: str) -> str:
        """Normalize person names.
        
        Args:
            text: Person name to normalize
            
        Returns:
            Normalized person name
        """
        # Remove titles
        titles = ['mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'lady']
        words = text.split()
        filtered_words = [word for word in words if word not in titles]
        
        # Capitalize first letter of each name part
        normalized_words = []
        for word in filtered_words:
            if word:
                normalized_words.append(word.capitalize())
        
        return ' '.join(normalized_words)
    
    async def _normalize_organization(self, text: str) -> str:
        """Normalize organization names.
        
        Args:
            text: Organization name to normalize
            
        Returns:
            Normalized organization name
        """
        # Handle common organization patterns
        normalized = text
        
        # Standardize legal entity types
        legal_entities = {
            'incorporated': 'inc',
            'corporation': 'corp',
            'limited': 'ltd',
            'company': 'co'
        }
        
        for full_form, abbrev in legal_entities.items():
            normalized = normalized.replace(full_form, abbrev)
        
        return normalized
    
    async def _normalize_location(self, text: str) -> str:
        """Normalize location names.
        
        Args:
            text: Location name to normalize
            
        Returns:
            Normalized location name
        """
        # Capitalize each word for proper nouns
        words = text.split()
        capitalized_words = [word.capitalize() for word in words if word]
        return ' '.join(capitalized_words)
    
    async def _normalize_product(self, text: str) -> str:
        """Normalize product names.
        
        Args:
            text: Product name to normalize
            
        Returns:
            Normalized product name
        """
        # Keep original capitalization for product names
        return text.title()
    
    async def _normalize_event(self, text: str) -> str:
        """Normalize event names.
        
        Args:
            text: Event name to normalize
            
        Returns:
            Normalized event name
        """
        return text.title()
    
    async def _normalize_date(self, text: str) -> str:
        """Normalize date expressions.
        
        Args:
            text: Date text to normalize
            
        Returns:
            Normalized date
        """
        # Basic date normalization - could be enhanced with date parsing
        return text.lower()
    
    async def _normalize_money(self, text: str) -> str:
        """Normalize money expressions.
        
        Args:
            text: Money text to normalize
            
        Returns:
            Normalized money expression
        """
        # Standardize currency symbols and amounts
        normalized = text.lower()
        
        # Convert currency names to symbols
        currency_mappings = {
            'dollars': '$',
            'euros': '€',
            'pounds': '£',
            'yen': '¥'
        }
        
        for currency_name, symbol in currency_mappings.items():
            normalized = normalized.replace(currency_name, symbol)
        
        return normalized
    
    async def _normalize_quantity(self, text: str) -> str:
        """Normalize quantity expressions.
        
        Args:
            text: Quantity text to normalize
            
        Returns:
            Normalized quantity
        """
        # Standardize units and numbers
        return text.lower()
    
    def _get_entity_variations(self, normalized_text: str) -> List[str]:
        """Get known variations for an entity.
        
        Args:
            normalized_text: Normalized entity text
            
        Returns:
            List of known variations
        """
        variations = []
        
        # Check predefined variations
        if normalized_text in self.entity_variations:
            variations.extend(self.entity_variations[normalized_text])
        
        # Generate common variations
        if ' ' in normalized_text:
            # Add acronym
            words = normalized_text.split()
            acronym = ''.join([word[0].upper() for word in words if word])
            if len(acronym) > 1:
                variations.append(acronym)
        
        return list(set(variations))
    
    def _calculate_normalization_confidence(self, original: str, normalized: str) -> float:
        """Calculate confidence in normalization.
        
        Args:
            original: Original text
            normalized: Normalized text
            
        Returns:
            Confidence score between 0 and 1
        """
        if original.lower() == normalized.lower():
            return 1.0
        
        # Calculate based on similarity and transformations applied
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, original.lower(), normalized.lower()).ratio()
        
        # Boost confidence for known transformations
        if any(abbrev in original.lower() for abbrev in self.abbreviations):
            similarity += 0.1
        
        if any(term in original.lower() for term in self.multilingual_mappings):
            similarity += 0.1
        
        return min(1.0, similarity)
    
    async def _create_canonical_mappings(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create canonical mappings for entities.
        
        Args:
            entities: Normalized entities
            
        Returns:
            Entities with canonical mappings
        """
        # Group entities by normalized text
        entity_groups = defaultdict(list)
        for entity in entities:
            normalized_text = entity.get('normalized_text', '')
            entity_groups[normalized_text].append(entity)
        
        canonical_entities = []
        
        for normalized_text, group in entity_groups.items():
            if not group:
                continue
            
            # Select canonical entity (highest confidence or most complete)
            canonical_entity = max(group, key=lambda e: (
                e.get('normalization_confidence', 0),
                len(e.get('text', '')),
                e.get('confidence', 0)
            ))
            
            # Collect all variations from the group
            all_variations = set()
            for entity in group:
                all_variations.add(entity.get('original_text', ''))
                all_variations.update(entity.get('variations', []))
            
            # Update canonical entity
            canonical_entity.update({
                'canonical_form': normalized_text,
                'all_variations': list(all_variations),
                'entity_count': len(group),
                'is_canonical': True
            })
            
            canonical_entities.append(canonical_entity)
        
        return canonical_entities
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get normalizer statistics and configuration.
        
        Returns:
            Dictionary with normalizer statistics
        """
        return {
            'normalizer_name': 'EntityNormalizer',
            'abbreviations_count': len(self.abbreviations),
            'entity_variations_count': len(self.entity_variations),
            'multilingual_mappings_count': len(self.multilingual_mappings),
            'supported_types': list(self.type_normalizers.keys())
        }
```

### 2. Integration with Entity Linker

**File**: Update `packages/morag-graph/src/morag_graph/normalizers/entity_linker.py`

Add normalization integration:

```python
# Add import
from morag_graph.normalizers.entity_normalizer import EntityNormalizer

# Update EntityLinker.__init__
def __init__(self, config: Optional[Dict[str, Any]] = None):
    # ... existing code ...
    self.entity_normalizer = EntityNormalizer(config)

# Add method for normalized linking
async def link_entities_with_normalization(
    self, 
    openie_triplets: List[Dict[str, Any]], 
    spacy_entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Link entities with normalization.
    
    Args:
        openie_triplets: Triplets from OpenIE
        spacy_entities: Entities from spaCy NER
        
    Returns:
        Triplets with normalized and linked entities
    """
    # Normalize spaCy entities first
    normalized_spacy_entities = await self.entity_normalizer.normalize_entities(spacy_entities)
    
    # Link with normalized entities
    linked_triplets = await self.link_entities(openie_triplets, normalized_spacy_entities)
    
    return linked_triplets
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_entity_normalizer.py`

```python
"""Tests for entity normalizer."""

import pytest
from morag_graph.normalizers.entity_normalizer import EntityNormalizer

class TestEntityNormalizer:
    
    def test_initialization(self):
        """Test normalizer initialization."""
        normalizer = EntityNormalizer()
        assert len(normalizer.abbreviations) > 0
        assert len(normalizer.entity_variations) > 0
    
    @pytest.mark.asyncio
    async def test_normalize_text(self):
        """Test basic text normalization."""
        normalizer = EntityNormalizer()
        
        # Test abbreviation expansion
        result = await normalizer._normalize_text("Microsoft Corp")
        assert "corporation" in result
        
        # Test multilingual mapping
        result = await normalizer._normalize_text("Juan empresa")
        assert "john" in result and "company" in result
    
    @pytest.mark.asyncio
    async def test_normalize_person(self):
        """Test person name normalization."""
        normalizer = EntityNormalizer()
        
        result = await normalizer._normalize_person("mr john smith")
        assert result == "John Smith"
    
    @pytest.mark.asyncio
    async def test_normalize_organization(self):
        """Test organization normalization."""
        normalizer = EntityNormalizer()
        
        result = await normalizer._normalize_organization("microsoft corporation")
        assert "corp" in result
    
    def test_get_entity_variations(self):
        """Test entity variation generation."""
        normalizer = EntityNormalizer()
        
        variations = normalizer._get_entity_variations("microsoft")
        assert "msft" in variations
        
        # Test acronym generation
        variations = normalizer._get_entity_variations("international business machines")
        assert "IBM" in variations
```

## Acceptance Criteria

- [ ] EntityNormalizer class implemented with type-specific normalization
- [ ] Support for abbreviation expansion and multilingual mapping
- [ ] Canonical entity mapping and variation handling
- [ ] Integration with entity linking pipeline
- [ ] Confidence scoring for normalization quality
- [ ] Comprehensive unit tests with >90% coverage
- [ ] Performance optimization for large entity sets
- [ ] Proper logging and error handling
- [ ] Statistics and monitoring capabilities

## Dependencies
- Task 2.1: Entity Linking Between OpenIE and spaCy NER

## Estimated Effort
- **Development**: 7-9 hours
- **Testing**: 4-5 hours
- **Integration**: 2-3 hours
- **Total**: 13-17 hours

## Notes
- Focus on maintaining entity meaning during normalization
- Consider domain-specific normalization rules
- Plan for extensible multilingual support
- Implement caching for frequently normalized entities
