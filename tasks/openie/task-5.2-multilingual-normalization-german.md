# Task 5.2: Multilingual Normalization with German Support

## Objective
Implement comprehensive multilingual normalization capabilities with specific focus on German language support, including entity normalization, predicate mapping, and cultural context handling.

## Scope
- Extend multilingual support to include German language
- Implement German-specific entity normalization rules
- Add German predicate mapping and verb conjugation handling
- Create German cultural context and naming conventions
- **MANDATORY**: Test thoroughly with German text samples

## Implementation Details

### 1. German Language Normalizer

**File**: `packages/morag-graph/src/morag_graph/normalizers/german_normalizer.py`

```python
"""German language normalization for OpenIE entities and predicates."""

import re
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
import structlog
import unicodedata

from morag_core.config import get_settings
from morag_core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

class GermanNormalizer:
    """Normalizes German entities and predicates to English equivalents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize German normalizer.
        
        Args:
            config: Optional configuration overrides
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Load German normalization rules
        self._load_german_mappings()
        self._load_german_grammar_rules()
        self._load_german_cultural_context()
    
    def _load_german_mappings(self) -> None:
        """Load German to English mappings."""
        # German entity types and organizations
        self.german_entity_mappings = {
            # Business entities
            'unternehmen': 'company',
            'gesellschaft': 'company',
            'firma': 'company',
            'betrieb': 'company',
            'konzern': 'corporation',
            'holding': 'holding company',
            'ag': 'corporation',  # Aktiengesellschaft
            'gmbh': 'limited liability company',  # Gesellschaft mit beschränkter Haftung
            'kg': 'limited partnership',  # Kommanditgesellschaft
            'ohg': 'general partnership',  # Offene Handelsgesellschaft
            'eg': 'cooperative',  # Eingetragene Genossenschaft
            'ev': 'association',  # Eingetragener Verein
            
            # Educational institutions
            'universität': 'university',
            'hochschule': 'university',
            'fachhochschule': 'university of applied sciences',
            'technische universität': 'technical university',
            'institut': 'institute',
            'forschungsinstitut': 'research institute',
            'akademie': 'academy',
            'schule': 'school',
            'gymnasium': 'high school',
            
            # Government and public institutions
            'regierung': 'government',
            'ministerium': 'ministry',
            'behörde': 'authority',
            'amt': 'office',
            'verwaltung': 'administration',
            'gemeinde': 'municipality',
            'stadt': 'city',
            'bundesland': 'state',
            'landkreis': 'district',
            
            # Titles and positions
            'geschäftsführer': 'ceo',
            'vorstandsvorsitzender': 'chairman',
            'vorstand': 'board member',
            'aufsichtsrat': 'supervisory board member',
            'direktor': 'director',
            'leiter': 'head',
            'manager': 'manager',
            'mitarbeiter': 'employee',
            'angestellter': 'employee',
            'arbeiter': 'worker',
            'praktikant': 'intern',
            'auszubildender': 'trainee',
            'student': 'student',
            'professor': 'professor',
            'doktor': 'doctor',
            'ingenieur': 'engineer',
            'entwickler': 'developer',
            'programmierer': 'programmer',
            'designer': 'designer',
            'berater': 'consultant',
            'verkäufer': 'salesperson',
            'kunde': 'customer',
            'klient': 'client',
            
            # Common German words in business context
            'projekt': 'project',
            'produkt': 'product',
            'service': 'service',
            'dienstleistung': 'service',
            'software': 'software',
            'technologie': 'technology',
            'innovation': 'innovation',
            'forschung': 'research',
            'entwicklung': 'development',
            'marketing': 'marketing',
            'vertrieb': 'sales',
            'finanzen': 'finance',
            'buchhaltung': 'accounting',
            'personal': 'human resources',
            'qualität': 'quality',
            'sicherheit': 'security'
        }
        
        # German predicate mappings
        self.german_predicate_mappings = {
            # Employment relationships
            'arbeitet bei': 'works at',
            'arbeitet für': 'works for',
            'ist angestellt bei': 'employed by',
            'ist beschäftigt bei': 'employed by',
            'ist mitarbeiter von': 'employee of',
            'hat eine stelle bei': 'works at',
            'ist tätig bei': 'works at',
            
            # Leadership relationships
            'ist geschäftsführer von': 'ceo of',
            'ist vorstandsvorsitzender von': 'chairman of',
            'ist direktor von': 'director of',
            'leitet': 'leads',
            'führt': 'leads',
            'verwaltet': 'manages',
            'ist verantwortlich für': 'responsible for',
            'beaufsichtigt': 'supervises',
            
            # Ownership relationships
            'besitzt': 'owns',
            'gehört': 'belongs to',
            'ist eigentümer von': 'owner of',
            'hat': 'has',
            'verfügt über': 'has',
            
            # Location relationships
            'lebt in': 'lives in',
            'wohnt in': 'lives in',
            'befindet sich in': 'located in',
            'ist ansässig in': 'based in',
            'hat seinen sitz in': 'headquartered in',
            'stammt aus': 'from',
            'kommt aus': 'from',
            
            # Education relationships
            'studiert an': 'studies at',
            'studierte an': 'studied at',
            'hat studiert an': 'studied at',
            'ist student an': 'student at',
            'promoviert an': 'pursuing phd at',
            'lehrt an': 'teaches at',
            'unterrichtet an': 'teaches at',
            'forscht an': 'researches at',
            
            # Creation relationships
            'gründete': 'founded',
            'hat gegründet': 'founded',
            'erstellt': 'created',
            'entwickelt': 'developed',
            'hat entwickelt': 'developed',
            'erfand': 'invented',
            'hat erfunden': 'invented',
            'baute': 'built',
            'hat gebaut': 'built',
            
            # Family relationships
            'ist verheiratet mit': 'married to',
            'ist der ehemann von': 'husband of',
            'ist die ehefrau von': 'wife of',
            'ist der vater von': 'father of',
            'ist die mutter von': 'mother of',
            'ist der sohn von': 'son of',
            'ist die tochter von': 'daughter of',
            'ist der bruder von': 'brother of',
            'ist die schwester von': 'sister of',
            
            # Membership relationships
            'ist mitglied von': 'member of',
            'gehört zu': 'belongs to',
            'ist teil von': 'part of',
            'ist verbunden mit': 'associated with',
            'kooperiert mit': 'cooperates with',
            'arbeitet zusammen mit': 'collaborates with'
        }
        
        # German name mappings
        self.german_name_mappings = {
            # Male names
            'johann': 'john',
            'johannes': 'john',
            'hans': 'john',
            'wilhelm': 'william',
            'willi': 'william',
            'friedrich': 'frederick',
            'fritz': 'frederick',
            'heinrich': 'henry',
            'karl': 'charles',
            'ludwig': 'louis',
            'georg': 'george',
            'michael': 'michael',
            'thomas': 'thomas',
            'andreas': 'andrew',
            'stefan': 'stephen',
            'christian': 'christian',
            'alexander': 'alexander',
            'matthias': 'matthew',
            'martin': 'martin',
            'peter': 'peter',
            'wolfgang': 'wolfgang',
            'klaus': 'klaus',
            'jürgen': 'jurgen',
            'dieter': 'dieter',
            'helmut': 'helmut',
            'bernd': 'bernd',
            'frank': 'frank',
            'rainer': 'rainer',
            'manfred': 'manfred',
            
            # Female names
            'maria': 'mary',
            'marie': 'mary',
            'anna': 'anna',
            'anne': 'anna',
            'elisabeth': 'elizabeth',
            'lisa': 'lisa',
            'katharina': 'catherine',
            'katrin': 'catherine',
            'christina': 'christina',
            'christine': 'christina',
            'barbara': 'barbara',
            'petra': 'petra',
            'sabine': 'sabine',
            'andrea': 'andrea',
            'claudia': 'claudia',
            'susanne': 'susan',
            'birgit': 'birgit',
            'gabriele': 'gabrielle',
            'monika': 'monica',
            'ursula': 'ursula',
            'ingrid': 'ingrid',
            'helga': 'helga',
            'renate': 'renate',
            'gisela': 'gisela',
            'brigitte': 'brigitte',
            'angelika': 'angelica',
            'martina': 'martina',
            'karin': 'karin',
            'doris': 'doris',
            'ruth': 'ruth',
            'dagmar': 'dagmar'
        }
    
    def _load_german_grammar_rules(self) -> None:
        """Load German grammar and conjugation rules."""
        # German verb conjugations to base forms
        self.german_verb_conjugations = {
            # arbeiten (to work)
            'arbeitet': 'arbeiten',
            'arbeitete': 'arbeiten',
            'gearbeitet': 'arbeiten',
            
            # leben (to live)
            'lebt': 'leben',
            'lebte': 'leben',
            'gelebt': 'leben',
            
            # studieren (to study)
            'studiert': 'studieren',
            'studierte': 'studieren',
            'studiert': 'studieren',
            
            # führen (to lead)
            'führt': 'führen',
            'führte': 'führen',
            'geführt': 'führen',
            
            # leiten (to lead/manage)
            'leitet': 'leiten',
            'leitete': 'leiten',
            'geleitet': 'leiten',
            
            # gründen (to found)
            'gründet': 'gründen',
            'gründete': 'gründen',
            'gegründet': 'gründen',
            
            # entwickeln (to develop)
            'entwickelt': 'entwickeln',
            'entwickelte': 'entwickeln',
            'entwickelt': 'entwickeln',
            
            # besitzen (to own)
            'besitzt': 'besitzen',
            'besaß': 'besitzen',
            'besessen': 'besitzen'
        }
        
        # German articles and determiners
        self.german_articles = {
            'der', 'die', 'das',  # definite articles
            'ein', 'eine', 'einer',  # indefinite articles
            'den', 'dem', 'des',  # declined forms
            'einen', 'einem', 'eines'  # declined forms
        }
        
        # German prepositions
        self.german_prepositions = {
            'in': 'in',
            'an': 'at',
            'auf': 'on',
            'bei': 'at',
            'von': 'from',
            'zu': 'to',
            'mit': 'with',
            'für': 'for',
            'über': 'about',
            'unter': 'under',
            'vor': 'before',
            'nach': 'after',
            'zwischen': 'between',
            'durch': 'through',
            'ohne': 'without',
            'gegen': 'against',
            'um': 'around'
        }
    
    def _load_german_cultural_context(self) -> None:
        """Load German cultural context and conventions."""
        # German company suffixes and their meanings
        self.german_company_suffixes = {
            'ag': 'aktiengesellschaft',
            'gmbh': 'gesellschaft mit beschränkter haftung',
            'kg': 'kommanditgesellschaft',
            'ohg': 'offene handelsgesellschaft',
            'eg': 'eingetragene genossenschaft',
            'ev': 'eingetragener verein',
            'ug': 'unternehmergesellschaft',
            'gbr': 'gesellschaft bürgerlichen rechts',
            'partg': 'partnerschaftsgesellschaft'
        }
        
        # German academic titles
        self.german_academic_titles = {
            'prof': 'professor',
            'dr': 'doctor',
            'dipl': 'diploma',
            'ing': 'engineer',
            'med': 'medical doctor',
            'phil': 'doctor of philosophy',
            'rer': 'doctor rerum',
            'jur': 'doctor of law',
            'theol': 'doctor of theology'
        }
        
        # German location patterns
        self.german_location_patterns = {
            'stadt': 'city',
            'dorf': 'village',
            'gemeinde': 'municipality',
            'landkreis': 'district',
            'bundesland': 'state',
            'region': 'region'
        }
    
    async def normalize_german_entity(self, entity_text: str, entity_type: str = None) -> str:
        """Normalize German entity to English equivalent.
        
        Args:
            entity_text: German entity text
            entity_type: Optional entity type hint
            
        Returns:
            Normalized English entity text
        """
        try:
            # Convert to lowercase for processing
            normalized = entity_text.lower().strip()
            
            # Handle German umlauts and special characters
            normalized = self._handle_german_characters(normalized)
            
            # Remove German articles
            normalized = self._remove_german_articles(normalized)
            
            # Apply entity-specific mappings
            if entity_type == 'PERSON':
                normalized = self._normalize_german_person(normalized)
            elif entity_type == 'ORG':
                normalized = self._normalize_german_organization(normalized)
            elif entity_type == 'GPE':
                normalized = self._normalize_german_location(normalized)
            else:
                # General entity normalization
                normalized = self._apply_general_german_mappings(normalized)
            
            # Capitalize appropriately
            normalized = self._apply_capitalization(normalized, entity_type)
            
            return normalized
            
        except Exception as e:
            logger.error(
                "German entity normalization failed",
                error=str(e),
                entity_text=entity_text
            )
            return entity_text  # Return original if normalization fails
    
    async def normalize_german_predicate(self, predicate_text: str) -> str:
        """Normalize German predicate to English equivalent.
        
        Args:
            predicate_text: German predicate text
            
        Returns:
            Normalized English predicate
        """
        try:
            # Convert to lowercase for processing
            normalized = predicate_text.lower().strip()
            
            # Handle German characters
            normalized = self._handle_german_characters(normalized)
            
            # Try direct mapping first
            if normalized in self.german_predicate_mappings:
                return self.german_predicate_mappings[normalized]
            
            # Try partial matching
            for german_pred, english_pred in self.german_predicate_mappings.items():
                if german_pred in normalized or normalized in german_pred:
                    return english_pred
            
            # Handle verb conjugations
            words = normalized.split()
            normalized_words = []
            
            for word in words:
                if word in self.german_verb_conjugations:
                    base_verb = self.german_verb_conjugations[word]
                    if base_verb in self.german_entity_mappings:
                        normalized_words.append(self.german_entity_mappings[base_verb])
                    else:
                        normalized_words.append(word)
                elif word in self.german_prepositions:
                    normalized_words.append(self.german_prepositions[word])
                elif word in self.german_entity_mappings:
                    normalized_words.append(self.german_entity_mappings[word])
                else:
                    normalized_words.append(word)
            
            return ' '.join(normalized_words)
            
        except Exception as e:
            logger.error(
                "German predicate normalization failed",
                error=str(e),
                predicate_text=predicate_text
            )
            return predicate_text
    
    def _handle_german_characters(self, text: str) -> str:
        """Handle German umlauts and special characters.
        
        Args:
            text: Text with German characters
            
        Returns:
            Text with normalized characters
        """
        # German umlaut mappings
        umlaut_mappings = {
            'ä': 'ae',
            'ö': 'oe',
            'ü': 'ue',
            'ß': 'ss'
        }
        
        for german_char, replacement in umlaut_mappings.items():
            text = text.replace(german_char, replacement)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        return text
    
    def _remove_german_articles(self, text: str) -> str:
        """Remove German articles from text.
        
        Args:
            text: Text with German articles
            
        Returns:
            Text without articles
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.german_articles]
        return ' '.join(filtered_words)
    
    def _normalize_german_person(self, person_text: str) -> str:
        """Normalize German person name.
        
        Args:
            person_text: German person name
            
        Returns:
            Normalized person name
        """
        # Handle academic titles
        for title_abbrev, title_full in self.german_academic_titles.items():
            person_text = person_text.replace(f"{title_abbrev}.", title_full)
            person_text = person_text.replace(title_abbrev, title_full)
        
        # Apply name mappings
        words = person_text.split()
        normalized_words = []
        
        for word in words:
            if word in self.german_name_mappings:
                normalized_words.append(self.german_name_mappings[word])
            elif word in self.german_academic_titles.values():
                # Keep academic titles
                normalized_words.append(word)
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def _normalize_german_organization(self, org_text: str) -> str:
        """Normalize German organization name.
        
        Args:
            org_text: German organization name
            
        Returns:
            Normalized organization name
        """
        # Handle company suffixes
        for suffix, full_form in self.german_company_suffixes.items():
            if org_text.endswith(suffix):
                org_text = org_text[:-len(suffix)].strip()
                # Add English equivalent
                if suffix in ['ag', 'gmbh']:
                    org_text += ' corporation'
                elif suffix in ['ev']:
                    org_text += ' association'
                break
        
        # Apply general mappings
        return self._apply_general_german_mappings(org_text)
    
    def _normalize_german_location(self, location_text: str) -> str:
        """Normalize German location name.
        
        Args:
            location_text: German location name
            
        Returns:
            Normalized location name
        """
        # Apply location pattern mappings
        for german_pattern, english_pattern in self.german_location_patterns.items():
            if german_pattern in location_text:
                location_text = location_text.replace(german_pattern, english_pattern)
        
        return self._apply_general_german_mappings(location_text)
    
    def _apply_general_german_mappings(self, text: str) -> str:
        """Apply general German to English mappings.
        
        Args:
            text: German text
            
        Returns:
            Text with German words mapped to English
        """
        words = text.split()
        normalized_words = []
        
        for word in words:
            if word in self.german_entity_mappings:
                normalized_words.append(self.german_entity_mappings[word])
            else:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def _apply_capitalization(self, text: str, entity_type: str = None) -> str:
        """Apply appropriate capitalization.
        
        Args:
            text: Text to capitalize
            entity_type: Entity type for context
            
        Returns:
            Properly capitalized text
        """
        if entity_type in ['PERSON', 'ORG', 'GPE']:
            # Capitalize each word for proper nouns
            return ' '.join(word.capitalize() for word in text.split())
        else:
            # Keep lowercase for general terms
            return text
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get normalizer statistics.
        
        Returns:
            Dictionary with normalizer statistics
        """
        return {
            'normalizer_name': 'GermanNormalizer',
            'entity_mappings_count': len(self.german_entity_mappings),
            'predicate_mappings_count': len(self.german_predicate_mappings),
            'name_mappings_count': len(self.german_name_mappings),
            'verb_conjugations_count': len(self.german_verb_conjugations),
            'company_suffixes_count': len(self.german_company_suffixes),
            'academic_titles_count': len(self.german_academic_titles)
        }
```

### 2. Integration with Existing Normalizers

**File**: Update `packages/morag-graph/src/morag_graph/normalizers/entity_normalizer.py`

Add German normalizer integration:

```python
# Add import
from morag_graph.normalizers.german_normalizer import GermanNormalizer

# Update EntityNormalizer.__init__
def __init__(self, config: Optional[Dict[str, Any]] = None):
    # ... existing code ...
    self.german_normalizer = GermanNormalizer(config)

# Add method for German normalization
async def normalize_german_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize entities with German language support.
    
    Args:
        entities: List of entities to normalize
        
    Returns:
        List of normalized entities with German support
    """
    normalized_entities = []
    
    for entity in entities:
        entity_text = entity.get('text', '')
        entity_type = entity.get('label', 'UNKNOWN')
        
        # Apply German normalization
        german_normalized = await self.german_normalizer.normalize_german_entity(
            entity_text, entity_type
        )
        
        # Apply general normalization
        general_normalized = await self._normalize_single_entity(entity)
        
        if general_normalized:
            general_normalized['german_normalized_text'] = german_normalized
            normalized_entities.append(general_normalized)
    
    return normalized_entities
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_german_normalizer.py`

```python
"""Tests for German normalizer."""

import pytest
from morag_graph.normalizers.german_normalizer import GermanNormalizer

class TestGermanNormalizer:
    
    def test_initialization(self):
        """Test normalizer initialization."""
        normalizer = GermanNormalizer()
        assert len(normalizer.german_entity_mappings) > 0
        assert len(normalizer.german_predicate_mappings) > 0
        assert len(normalizer.german_name_mappings) > 0
    
    def test_handle_german_characters(self):
        """Test German character handling."""
        normalizer = GermanNormalizer()
        
        # Test umlauts
        assert normalizer._handle_german_characters('Müller') == 'mueller'
        assert normalizer._handle_german_characters('Größe') == 'groesse'
        assert normalizer._handle_german_characters('Weiß') == 'weiss'
    
    @pytest.mark.asyncio
    async def test_normalize_german_entity_person(self):
        """Test German person name normalization."""
        normalizer = GermanNormalizer()
        
        result = await normalizer.normalize_german_entity('Johann Müller', 'PERSON')
        assert 'john' in result.lower()
        assert 'mueller' in result.lower()
    
    @pytest.mark.asyncio
    async def test_normalize_german_entity_organization(self):
        """Test German organization normalization."""
        normalizer = GermanNormalizer()
        
        result = await normalizer.normalize_german_entity('Volkswagen AG', 'ORG')
        assert 'corporation' in result.lower()
        
        result = await normalizer.normalize_german_entity('Deutsche Bank GmbH', 'ORG')
        assert 'corporation' in result.lower()
    
    @pytest.mark.asyncio
    async def test_normalize_german_predicate(self):
        """Test German predicate normalization."""
        normalizer = GermanNormalizer()
        
        result = await normalizer.normalize_german_predicate('arbeitet bei')
        assert result == 'works at'
        
        result = await normalizer.normalize_german_predicate('ist geschäftsführer von')
        assert result == 'ceo of'
        
        result = await normalizer.normalize_german_predicate('lebt in')
        assert result == 'lives in'
    
    def test_remove_german_articles(self):
        """Test German article removal."""
        normalizer = GermanNormalizer()
        
        result = normalizer._remove_german_articles('der große Konzern')
        assert result == 'große Konzern'
        
        result = normalizer._remove_german_articles('die deutsche Bank')
        assert result == 'deutsche Bank'
```

## Acceptance Criteria

- [ ] GermanNormalizer class with comprehensive German language support
- [ ] German entity normalization for persons, organizations, and locations
- [ ] German predicate mapping with verb conjugation handling
- [ ] German cultural context and naming conventions
- [ ] Integration with existing entity and predicate normalizers
- [ ] Proper handling of German umlauts and special characters
- [ ] German company suffix recognition and normalization
- [ ] German academic title handling
- [ ] Comprehensive unit tests with >90% coverage
- [ ] Performance optimization for German text processing

## Dependencies
- Task 2.2: Entity Normalization and Canonical Mapping
- Task 3.1: Predicate Normalization and Standardization

## Estimated Effort
- **Development**: 8-10 hours
- **Testing**: 4-5 hours
- **Integration**: 2-3 hours
- **Total**: 14-18 hours

## Notes
- Focus on German business and academic contexts
- Handle German grammatical cases and declensions
- Consider regional German variations (Austrian, Swiss)
- Plan for extensibility to other Germanic languages
