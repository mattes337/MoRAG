"""Custom schema definitions for MoRAG-specific entity types."""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from enum import Enum

logger = logging.getLogger(__name__)


class MoragEntityType(str, Enum):
    """MoRAG-specific entity types."""
    # Core types
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    
    # Document-specific types
    DOCUMENT = "DOCUMENT"
    SECTION = "SECTION"
    FIGURE = "FIGURE"
    TABLE = "TABLE"
    REFERENCE = "REFERENCE"
    
    # Technical types
    TECHNOLOGY = "TECHNOLOGY"
    SOFTWARE = "SOFTWARE"
    ALGORITHM = "ALGORITHM"
    DATASET = "DATASET"
    MODEL = "MODEL"
    
    # Domain-specific types
    CONCEPT = "CONCEPT"
    METHODOLOGY = "METHODOLOGY"
    METRIC = "METRIC"
    EXPERIMENT = "EXPERIMENT"
    RESULT = "RESULT"
    
    # Temporal types
    EVENT = "EVENT"
    PERIOD = "PERIOD"
    MILESTONE = "MILESTONE"


class MoragRelationType(str, Enum):
    """MoRAG-specific relation types with enhanced semantic depth."""

    # === BASIC STRUCTURAL RELATIONSHIPS ===
    MENTIONS = "MENTIONS"
    CONTAINS = "CONTAINS"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"

    # === HIERARCHICAL RELATIONSHIPS ===
    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"
    BELONGS_TO = "BELONGS_TO"
    OWNS = "OWNS"
    MANAGES = "MANAGES"
    SUPERVISES = "SUPERVISES"
    REPORTS_TO = "REPORTS_TO"

    # === CAUSAL RELATIONSHIPS ===
    CAUSES = "CAUSES"
    PREVENTS = "PREVENTS"
    ENABLES = "ENABLES"
    TRIGGERS = "TRIGGERS"
    RESULTS_IN = "RESULTS_IN"
    LEADS_TO = "LEADS_TO"
    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    INHIBITS = "INHIBITS"
    ACCELERATES = "ACCELERATES"

    # === FUNCTIONAL RELATIONSHIPS ===
    DEFINES = "DEFINES"
    IMPLEMENTS = "IMPLEMENTS"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    INFLUENCES = "INFLUENCES"
    OPERATES = "OPERATES"
    CONTROLS = "CONTROLS"
    REGULATES = "REGULATES"
    MODIFIES = "MODIFIES"
    TRANSFORMS = "TRANSFORMS"

    # === TEMPORAL RELATIONSHIPS ===
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    OCCURS_DURING = "OCCURS_DURING"
    OCCURS_BEFORE = "OCCURS_BEFORE"
    OCCURS_AFTER = "OCCURS_AFTER"
    CONCURRENT_WITH = "CONCURRENT_WITH"
    INITIATES = "INITIATES"
    TERMINATES = "TERMINATES"

    # === SPATIAL RELATIONSHIPS ===
    LOCATED_IN = "LOCATED_IN"
    ADJACENT_TO = "ADJACENT_TO"
    SURROUNDS = "SURROUNDS"
    CONNECTS = "CONNECTS"
    SEPARATES = "SEPARATES"

    # === SEMANTIC RELATIONSHIPS ===
    SIMILAR_TO = "SIMILAR_TO"
    OPPOSITE_OF = "OPPOSITE_OF"
    EQUIVALENT_TO = "EQUIVALENT_TO"
    SPECIALIZES = "SPECIALIZES"
    GENERALIZES = "GENERALIZES"
    EXEMPLIFIES = "EXEMPLIFIES"
    CATEGORIZES = "CATEGORIZES"

    # === COLLABORATIVE RELATIONSHIPS ===
    COLLABORATES_WITH = "COLLABORATES_WITH"
    PARTNERS_WITH = "PARTNERS_WITH"
    COMPETES_WITH = "COMPETES_WITH"
    SUPPORTS = "SUPPORTS"
    OPPOSES = "OPPOSES"
    ASSISTS = "ASSISTS"

    # === KNOWLEDGE RELATIONSHIPS ===
    TEACHES = "TEACHES"
    LEARNS_FROM = "LEARNS_FROM"
    EXPLAINS = "EXPLAINS"
    DEMONSTRATES = "DEMONSTRATES"
    PROVES = "PROVES"
    DISPROVES = "DISPROVES"
    VALIDATES = "VALIDATES"
    CONTRADICTS = "CONTRADICTS"

    # === CREATION RELATIONSHIPS ===
    CREATES = "CREATES"
    PRODUCES = "PRODUCES"
    GENERATES = "GENERATES"
    BUILDS = "BUILDS"
    DEVELOPS = "DEVELOPS"
    DESIGNS = "DESIGNS"
    INVENTS = "INVENTS"
    DISCOVERS = "DISCOVERS"

    # === COMMUNICATION RELATIONSHIPS ===
    COMMUNICATES_WITH = "COMMUNICATES_WITH"
    INFORMS = "INFORMS"
    NOTIFIES = "NOTIFIES"
    REQUESTS = "REQUESTS"
    RESPONDS_TO = "RESPONDS_TO"
    DISCUSSES = "DISCUSSES"

    # === DOCUMENT RELATIONSHIPS ===
    REFERENCES = "REFERENCES"
    CITES = "CITES"
    DESCRIBES = "DESCRIBES"
    ILLUSTRATES = "ILLUSTRATES"
    DOCUMENTS = "DOCUMENTS"
    SUMMARIZES = "SUMMARIZES"
    QUOTES = "QUOTES"
    PARAPHRASES = "PARAPHRASES"

    # === DOMAIN-SPECIFIC RELATIONSHIPS ===
    # Medical/Health
    TREATS = "TREATS"
    DIAGNOSES = "DIAGNOSES"
    SYMPTOMS_OF = "SYMPTOMS_OF"
    SIDE_EFFECT_OF = "SIDE_EFFECT_OF"
    PRESCRIBED_FOR = "PRESCRIBED_FOR"
    CONTRAINDICATED_WITH = "CONTRAINDICATED_WITH"

    # Technical/Engineering
    INTEGRATES_WITH = "INTEGRATES_WITH"
    INTERFACES_WITH = "INTERFACES_WITH"
    EXTENDS = "EXTENDS"
    INHERITS_FROM = "INHERITS_FROM"
    OVERRIDES = "OVERRIDES"
    CALLS = "CALLS"
    RETURNS = "RETURNS"

    # Business/Economic
    SELLS_TO = "SELLS_TO"
    BUYS_FROM = "BUYS_FROM"
    INVESTS_IN = "INVESTS_IN"
    FUNDS = "FUNDS"
    ACQUIRES = "ACQUIRES"
    MERGES_WITH = "MERGES_WITH"
    LICENSES = "LICENSES"

    # Research/Academic
    STUDIES = "STUDIES"
    RESEARCHES = "RESEARCHES"
    ANALYZES = "ANALYZES"
    EVALUATES = "EVALUATES"
    COMPARES = "COMPARES"
    MEASURES = "MEASURES"
    OBSERVES = "OBSERVES"


class BaseEntitySchema(BaseModel):
    """Base schema for all MoRAG entities."""
    id: str
    name: str
    type: MoragEntityType
    confidence: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    source_document_id: Optional[str] = None
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class PersonEntity(BaseEntitySchema):
    """Schema for person entities."""
    type: MoragEntityType = MoragEntityType.PERSON
    
    # Person-specific attributes
    title: Optional[str] = None
    organization: Optional[str] = None
    role: Optional[str] = None
    email: Optional[str] = None
    expertise_areas: List[str] = Field(default_factory=list)
    publications: List[str] = Field(default_factory=list)
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v


class OrganizationEntity(BaseEntitySchema):
    """Schema for organization entities."""
    type: MoragEntityType = MoragEntityType.ORGANIZATION
    
    # Organization-specific attributes
    organization_type: Optional[str] = None  # "company", "university", "government", etc.
    industry: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    founded_year: Optional[int] = None
    size: Optional[str] = None  # "startup", "small", "medium", "large"


class TechnologyEntity(BaseEntitySchema):
    """Schema for technology entities."""
    type: MoragEntityType = MoragEntityType.TECHNOLOGY
    
    # Technology-specific attributes
    category: Optional[str] = None  # "programming_language", "framework", "tool", etc.
    version: Optional[str] = None
    vendor: Optional[str] = None
    license: Optional[str] = None
    documentation_url: Optional[str] = None
    maturity_level: Optional[str] = None  # "experimental", "stable", "deprecated"


class ConceptEntity(BaseEntitySchema):
    """Schema for concept entities."""
    type: MoragEntityType = MoragEntityType.CONCEPT
    
    # Concept-specific attributes
    domain: Optional[str] = None
    definition: Optional[str] = None
    related_concepts: List[str] = Field(default_factory=list)
    complexity_level: Optional[str] = None  # "basic", "intermediate", "advanced"
    prerequisites: List[str] = Field(default_factory=list)


class DocumentEntity(BaseEntitySchema):
    """Schema for document entities."""
    type: MoragEntityType = MoragEntityType.DOCUMENT
    
    # Document-specific attributes
    document_type: Optional[str] = None  # "paper", "report", "manual", etc.
    authors: List[str] = Field(default_factory=list)
    publication_date: Optional[datetime] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    page_count: Optional[int] = None


class BaseRelationSchema(BaseModel):
    """Base schema for all MoRAG relations."""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: MoragRelationType
    confidence: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = None
    context: Optional[str] = None
    source_document_id: Optional[str] = None
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticRelation(BaseRelationSchema):
    """Schema for semantic relations with additional properties."""
    
    # Semantic-specific attributes
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    directionality: str = Field(default="bidirectional")  # "unidirectional", "bidirectional"
    temporal_scope: Optional[str] = None  # "past", "present", "future", "timeless"
    evidence_text: Optional[str] = None
    
    @field_validator('directionality')
    @classmethod
    def validate_directionality(cls, v):
        if v not in ["unidirectional", "bidirectional"]:
            raise ValueError('Directionality must be "unidirectional" or "bidirectional"')
        return v


class TemporalRelation(BaseRelationSchema):
    """Schema for temporal relations."""
    
    # Temporal-specific attributes
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[str] = None
    temporal_precision: str = Field(default="unknown")  # "exact", "approximate", "unknown"
    
    @field_validator('end_time')
    @classmethod
    def validate_end_time(cls, v, info):
        if v and hasattr(info, 'data') and 'start_time' in info.data and info.data['start_time'] and v < info.data['start_time']:
            raise ValueError('End time must be after start time')
        return v


class DocumentRelation(BaseRelationSchema):
    """Schema for document-specific relations."""

    # Document-specific attributes
    page_number: Optional[int] = None
    section: Optional[str] = None
    citation_context: Optional[str] = None
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)


class SchemaRegistry:
    """Registry for managing custom entity and relation schemas."""

    def __init__(self):
        self.entity_schemas: Dict[MoragEntityType, type] = {
            MoragEntityType.PERSON: PersonEntity,
            MoragEntityType.ORGANIZATION: OrganizationEntity,
            MoragEntityType.TECHNOLOGY: TechnologyEntity,
            MoragEntityType.CONCEPT: ConceptEntity,
            MoragEntityType.DOCUMENT: DocumentEntity,
        }

        self.relation_schemas: Dict[str, type] = {
            "semantic": SemanticRelation,
            "temporal": TemporalRelation,
            "document": DocumentRelation,
            "base": BaseRelationSchema
        }

    def get_entity_schema(self, entity_type: MoragEntityType) -> type:
        """Get schema class for entity type."""
        return self.entity_schemas.get(entity_type, BaseEntitySchema)

    def get_relation_schema(self, relation_category: str = "base") -> type:
        """Get schema class for relation category."""
        return self.relation_schemas.get(relation_category, BaseRelationSchema)

    def register_entity_schema(self, entity_type: MoragEntityType, schema_class: type):
        """Register custom entity schema."""
        self.entity_schemas[entity_type] = schema_class
        logger.info(f"Registered entity schema for {entity_type}")

    def register_relation_schema(self, category: str, schema_class: type):
        """Register custom relation schema."""
        self.relation_schemas[category] = schema_class
        logger.info(f"Registered relation schema for {category}")

    def validate_entity(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate entity data against appropriate schema."""
        entity_type = MoragEntityType(entity_data.get('type', 'CONCEPT'))
        schema_class = self.get_entity_schema(entity_type)

        try:
            validated_entity = schema_class(**entity_data)
            return validated_entity.model_dump()
        except Exception as e:
            logger.error(f"Entity validation failed: {e}")
            # Fall back to base schema with minimal required fields
            fallback_data = {
                'id': entity_data.get('id', 'unknown'),
                'name': entity_data.get('name', 'unknown'),
                'type': entity_type,
                'confidence': entity_data.get('confidence', 0.5),
                **{k: v for k, v in entity_data.items() if k not in ['id', 'name', 'type', 'confidence']}
            }
            try:
                base_entity = BaseEntitySchema(**fallback_data)
                return base_entity.model_dump()
            except Exception:
                # Ultimate fallback - return original data
                return entity_data

    def validate_relation(self, relation_data: Dict[str, Any], category: str = "base") -> Dict[str, Any]:
        """Validate relation data against appropriate schema."""
        schema_class = self.get_relation_schema(category)

        try:
            validated_relation = schema_class(**relation_data)
            return validated_relation.model_dump()
        except Exception as e:
            logger.error(f"Relation validation failed: {e}")
            # Fall back to base schema
            try:
                base_relation = BaseRelationSchema(**relation_data)
                return base_relation.model_dump()
            except Exception:
                # Ultimate fallback - return original data
                return relation_data


# Global schema registry instance
schema_registry = SchemaRegistry()
