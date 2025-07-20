"""Domain-specific relation extractors for specialized knowledge domains."""

import re
from typing import List, Dict, Set, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import structlog

from morag_core.ai import Relation
from ..models import Entity as GraphEntity

logger = structlog.get_logger(__name__)


@dataclass
class DomainPattern:
    """A domain-specific pattern for relation extraction."""
    pattern: str
    relation_category: str  # Category instead of specific type
    confidence_boost: float
    entity_types: List[str]
    description: str


class BaseDomainExtractor(ABC):
    """Base class for domain-specific relation extractors."""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.logger = structlog.get_logger(__name__)
        self.patterns = self._build_patterns()
    
    @abstractmethod
    def _build_patterns(self) -> List[DomainPattern]:
        """Build domain-specific patterns."""
        pass
    
    def extract_domain_relations(
        self,
        text: str,
        entities: List[GraphEntity],
        base_relations: List[Relation]
    ) -> List[Relation]:
        """Extract domain-specific relations from text."""
        domain_relations = []
        
        # Filter entities relevant to this domain
        relevant_entities = self._filter_relevant_entities(entities)
        if len(relevant_entities) < 2:
            return domain_relations
        
        # Apply domain patterns
        for pattern in self.patterns:
            relations = self._apply_pattern(pattern, text, relevant_entities)
            domain_relations.extend(relations)
        
        # Enhance existing relations with domain knowledge
        enhanced_relations = self._enhance_existing_relations(base_relations, text, entities)
        domain_relations.extend(enhanced_relations)
        
        return domain_relations
    
    def _filter_relevant_entities(self, entities: List[GraphEntity]) -> List[GraphEntity]:
        """Filter entities relevant to this domain."""
        # Default implementation - subclasses should override
        return entities
    
    def _apply_pattern(
        self,
        pattern: DomainPattern,
        text: str,
        entities: List[GraphEntity]
    ) -> List[Relation]:
        """Apply a domain pattern to extract relations."""
        relations = []

        # Find pattern matches in text
        matches = re.finditer(pattern.pattern, text, re.IGNORECASE)

        for match in matches:
            # Find entities near the match
            nearby_entities = self._find_nearby_entities(
                text, match.start(), match.end(), entities
            )

            # Create relations between nearby entities
            for i, entity1 in enumerate(nearby_entities):
                for entity2 in nearby_entities[i+1:]:
                    if self._entities_match_pattern(entity1, entity2, pattern):
                        # Create dynamic relation type based on pattern and context
                        relation_type = self._create_dynamic_relation_type(
                            pattern, match.group(), entity1, entity2
                        )

                        relation = Relation(
                            source_entity=entity1.name,
                            target_entity=entity2.name,
                            relation_type=relation_type,
                            confidence=0.7 + pattern.confidence_boost,
                            context=match.group()
                        )
                        relations.append(relation)

        return relations

    def _create_dynamic_relation_type(
        self,
        pattern: DomainPattern,
        matched_text: str,
        entity1: GraphEntity,
        entity2: GraphEntity
    ) -> str:
        """Create a dynamic relation type based on pattern and context."""
        # Analyze the matched text to determine the specific relation type
        matched_lower = matched_text.lower()

        # Extract the key verb or relationship indicator from the matched text
        # This is a simplified approach - could be enhanced with NLP
        if pattern.relation_category == "treatment":
            if "treat" in matched_lower:
                return "treats"
            elif "cure" in matched_lower:
                return "cures"
            elif "heal" in matched_lower:
                return "heals"
            else:
                return "therapeutically_affects"

        elif pattern.relation_category == "causation":
            if "cause" in matched_lower:
                return "causes"
            elif "result" in matched_lower:
                return "results_in"
            elif "lead" in matched_lower:
                return "leads_to"
            else:
                return "causally_influences"

        elif pattern.relation_category == "prevention":
            if "prevent" in matched_lower:
                return "prevents"
            elif "protect" in matched_lower:
                return "protects_against"
            elif "block" in matched_lower:
                return "blocks"
            else:
                return "inhibits"

        elif pattern.relation_category == "technical":
            if "integrate" in matched_lower:
                return "integrates_with"
            elif "extend" in matched_lower:
                return "extends"
            elif "implement" in matched_lower:
                return "implements"
            elif "call" in matched_lower:
                return "calls"
            else:
                return "technically_interacts_with"

        elif pattern.relation_category == "business":
            if "acquire" in matched_lower:
                return "acquires"
            elif "invest" in matched_lower:
                return "invests_in"
            elif "partner" in matched_lower:
                return "partners_with"
            elif "compete" in matched_lower:
                return "competes_with"
            else:
                return "business_relates_to"

        elif pattern.relation_category == "research":
            if "study" in matched_lower:
                return "studies"
            elif "research" in matched_lower:
                return "researches"
            elif "analyze" in matched_lower:
                return "analyzes"
            elif "investigate" in matched_lower:
                return "investigates"
            else:
                return "academically_examines"

        # Default fallback
        return f"{pattern.relation_category}_relates_to"
    
    def _find_nearby_entities(
        self,
        text: str,
        start: int,
        end: int,
        entities: List[GraphEntity],
        window: int = 100
    ) -> List[GraphEntity]:
        """Find entities near a text position."""
        nearby = []
        search_start = max(0, start - window)
        search_end = min(len(text), end + window)
        search_text = text[search_start:search_end].lower()
        
        for entity in entities:
            if entity.name.lower() in search_text:
                nearby.append(entity)
        
        return nearby
    
    def _entities_match_pattern(
        self,
        entity1: GraphEntity,
        entity2: GraphEntity,
        pattern: DomainPattern
    ) -> bool:
        """Check if entities match the pattern requirements."""
        if not pattern.entity_types:
            return True
        
        entity_types = [entity1.type, entity2.type]
        return any(et in pattern.entity_types for et in entity_types)
    
    def _enhance_existing_relations(
        self,
        relations: List[Relation],
        text: str,
        entities: List[GraphEntity]
    ) -> List[Relation]:
        """Enhance existing relations with domain knowledge."""
        # Default implementation - subclasses should override
        return []


class MedicalDomainExtractor(BaseDomainExtractor):
    """Extractor for medical/healthcare domain relations."""
    
    def __init__(self):
        super().__init__("medical")
    
    def _build_patterns(self) -> List[DomainPattern]:
        """Build medical domain patterns."""
        return [
            DomainPattern(
                pattern=r'\b(\w+)\s+treats?\s+(\w+)',
                relation_category='treatment',
                confidence_boost=0.2,
                entity_types=['SUBSTANCE', 'PERSON', 'ORGANIZATION'],
                description='Treatment relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:causes?|leads? to|results? in)\s+(\w+)',
                relation_category='causation',
                confidence_boost=0.25,
                entity_types=['SUBSTANCE', 'CONCEPT'],
                description='Causal medical relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:prevents?|protects? against)\s+(\w+)',
                relation_category='prevention',
                confidence_boost=0.2,
                entity_types=['SUBSTANCE', 'CONCEPT'],
                description='Prevention relationships'
            ),
            DomainPattern(
                pattern=r'\b(?:symptoms? of|signs? of)\s+(\w+)',
                relation_category='symptom',
                confidence_boost=0.15,
                entity_types=['CONCEPT'],
                description='Symptom relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:prescribed for|used to treat)\s+(\w+)',
                relation_category='prescription',
                confidence_boost=0.2,
                entity_types=['SUBSTANCE'],
                description='Prescription relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:contraindicated|not recommended)\s+(?:with|for)\s+(\w+)',
                relation_category='contraindication',
                confidence_boost=0.2,
                entity_types=['SUBSTANCE'],
                description='Contraindication relationships'
            ),
            DomainPattern(
                pattern=r'\b(?:side effects? of|adverse effects? of)\s+(\w+)',
                relation_category='side_effect',
                confidence_boost=0.15,
                entity_types=['SUBSTANCE'],
                description='Side effect relationships'
            )
        ]
    
    def _filter_relevant_entities(self, entities: List[GraphEntity]) -> List[GraphEntity]:
        """Filter entities relevant to medical domain."""
        medical_types = {'SUBSTANCE', 'PERSON', 'ORGANIZATION', 'CONCEPT', 'METHODOLOGY'}
        return [e for e in entities if e.type in medical_types]


class TechnicalDomainExtractor(BaseDomainExtractor):
    """Extractor for technical/software domain relations."""
    
    def __init__(self):
        super().__init__("technical")
    
    def _build_patterns(self) -> List[DomainPattern]:
        """Build technical domain patterns."""
        return [
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:integrates? with|interfaces? with)\s+(\w+)',
                relation_category='technical',
                confidence_boost=0.2,
                entity_types=['TECHNOLOGY', 'SOFTWARE', 'SYSTEM'],
                description='Integration relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:extends?|inherits? from)\s+(\w+)',
                relation_category='technical',
                confidence_boost=0.25,
                entity_types=['SOFTWARE', 'TECHNOLOGY'],
                description='Inheritance relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:calls?|invokes?)\s+(\w+)',
                relation_category='technical',
                confidence_boost=0.2,
                entity_types=['SOFTWARE', 'ALGORITHM'],
                description='Function call relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:implements?|realizes?)\s+(\w+)',
                relation_category='technical',
                confidence_boost=0.2,
                entity_types=['SOFTWARE', 'TECHNOLOGY'],
                description='Implementation relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:depends? on|requires?)\s+(\w+)',
                relation_category='technical',
                confidence_boost=0.15,
                entity_types=['SOFTWARE', 'TECHNOLOGY'],
                description='Dependency relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:overrides?|replaces?)\s+(\w+)',
                relation_category='technical',
                confidence_boost=0.2,
                entity_types=['SOFTWARE'],
                description='Override relationships'
            )
        ]
    
    def _filter_relevant_entities(self, entities: List[GraphEntity]) -> List[GraphEntity]:
        """Filter entities relevant to technical domain."""
        technical_types = {'TECHNOLOGY', 'SOFTWARE', 'ALGORITHM', 'SYSTEM', 'CONCEPT'}
        return [e for e in entities if e.type in technical_types]


class BusinessDomainExtractor(BaseDomainExtractor):
    """Extractor for business/commercial domain relations."""
    
    def __init__(self):
        super().__init__("business")
    
    def _build_patterns(self) -> List[DomainPattern]:
        """Build business domain patterns."""
        return [
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:acquires?|purchases?|buys?)\s+(\w+)',
                relation_category='business',
                confidence_boost=0.25,
                entity_types=['ORGANIZATION'],
                description='Acquisition relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:invests? in|funds?)\s+(\w+)',
                relation_category='business',
                confidence_boost=0.2,
                entity_types=['ORGANIZATION', 'PERSON'],
                description='Investment relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:partners? with|collaborates? with)\s+(\w+)',
                relation_category='business',
                confidence_boost=0.15,
                entity_types=['ORGANIZATION'],
                description='Partnership relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:competes? with|rivals?)\s+(\w+)',
                relation_category='business',
                confidence_boost=0.15,
                entity_types=['ORGANIZATION'],
                description='Competition relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:sells? to|markets? to)\s+(\w+)',
                relation_category='business',
                confidence_boost=0.2,
                entity_types=['ORGANIZATION'],
                description='Sales relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:licenses?|grants? license)\s+(?:to\s+)?(\w+)',
                relation_category='business',
                confidence_boost=0.2,
                entity_types=['ORGANIZATION'],
                description='Licensing relationships'
            )
        ]
    
    def _filter_relevant_entities(self, entities: List[GraphEntity]) -> List[GraphEntity]:
        """Filter entities relevant to business domain."""
        business_types = {'ORGANIZATION', 'PERSON', 'LOCATION', 'CONCEPT'}
        return [e for e in entities if e.type in business_types]


class AcademicDomainExtractor(BaseDomainExtractor):
    """Extractor for academic/research domain relations."""
    
    def __init__(self):
        super().__init__("academic")
    
    def _build_patterns(self) -> List[DomainPattern]:
        """Build academic domain patterns."""
        return [
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:studies?|researches?|investigates?)\s+(\w+)',
                relation_category='research',
                confidence_boost=0.2,
                entity_types=['PERSON', 'ORGANIZATION', 'CONCEPT'],
                description='Research relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:analyzes?|examines?|evaluates?)\s+(\w+)',
                relation_category='research',
                confidence_boost=0.15,
                entity_types=['PERSON', 'METHODOLOGY'],
                description='Analysis relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:proves?|demonstrates?|validates?)\s+(\w+)',
                relation_category='research',
                confidence_boost=0.2,
                entity_types=['CONCEPT', 'METHODOLOGY'],
                description='Validation relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:contradicts?|disproves?|refutes?)\s+(\w+)',
                relation_category='research',
                confidence_boost=0.2,
                entity_types=['CONCEPT'],
                description='Contradiction relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:cites?|references?)\s+(\w+)',
                relation_category='research',
                confidence_boost=0.15,
                entity_types=['DOCUMENT', 'PERSON'],
                description='Citation relationships'
            ),
            DomainPattern(
                pattern=r'\b(\w+)\s+(?:builds? on|extends?|develops?)\s+(\w+)',
                relation_category='research',
                confidence_boost=0.15,
                entity_types=['CONCEPT', 'METHODOLOGY'],
                description='Development relationships'
            )
        ]
    
    def _filter_relevant_entities(self, entities: List[GraphEntity]) -> List[GraphEntity]:
        """Filter entities relevant to academic domain."""
        academic_types = {'PERSON', 'ORGANIZATION', 'CONCEPT', 'METHODOLOGY', 'DOCUMENT', 'EXPERIMENT'}
        return [e for e in entities if e.type in academic_types]


class DomainExtractorFactory:
    """Factory for creating domain-specific extractors."""
    
    _extractors = {
        'medical': MedicalDomainExtractor,
        'technical': TechnicalDomainExtractor,
        'business': BusinessDomainExtractor,
        'academic': AcademicDomainExtractor
    }
    
    @classmethod
    def create_extractor(cls, domain: str) -> Optional[BaseDomainExtractor]:
        """Create a domain extractor for the specified domain."""
        extractor_class = cls._extractors.get(domain.lower())
        if extractor_class:
            return extractor_class()
        return None
    
    @classmethod
    def get_available_domains(cls) -> List[str]:
        """Get list of available domains."""
        return list(cls._extractors.keys())
