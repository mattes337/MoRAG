# Step 9: Custom Schema and Entity Types

**Duration**: 3-4 days  
**Phase**: Advanced Features  
**Prerequisites**: Steps 1-8 completed, temporal queries working

## Objective

Define MoRAG-specific entity types and relationship schemas in Graphiti, enabling domain-specific knowledge representation and enhanced semantic understanding.

## Deliverables

1. Custom entity type definitions using Pydantic models
2. Domain-specific relationship schemas
3. Validation framework for custom types
4. Enhanced search with semantic type awareness
5. Migration tools for existing entity types

## Implementation

### 1. Define Custom Entity Types

**File**: `packages/morag-graph/src/morag_graph/graphiti/custom_schema.py`

```python
"""Custom schema definitions for MoRAG-specific entity types."""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
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
    """MoRAG-specific relation types."""
    # Basic relationships
    MENTIONS = "MENTIONS"
    CONTAINS = "CONTAINS"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"
    
    # Hierarchical relationships
    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"
    BELONGS_TO = "BELONGS_TO"
    
    # Semantic relationships
    DEFINES = "DEFINES"
    IMPLEMENTS = "IMPLEMENTS"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    INFLUENCES = "INFLUENCES"
    
    # Temporal relationships
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    OCCURS_DURING = "OCCURS_DURING"
    
    # Document relationships
    REFERENCES = "REFERENCES"
    CITES = "CITES"
    DESCRIBES = "DESCRIBES"
    ILLUSTRATES = "ILLUSTRATES"


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
    
    @validator('confidence')
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
    
    @validator('email')
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
    
    @validator('directionality')
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
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if v and 'start_time' in values and values['start_time'] and v < values['start_time']:
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
            return validated_entity.dict()
        except Exception as e:
            logger.error(f"Entity validation failed: {e}")
            # Fall back to base schema
            base_entity = BaseEntitySchema(**entity_data)
            return base_entity.dict()
    
    def validate_relation(self, relation_data: Dict[str, Any], category: str = "base") -> Dict[str, Any]:
        """Validate relation data against appropriate schema."""
        schema_class = self.get_relation_schema(category)
        
        try:
            validated_relation = schema_class(**relation_data)
            return validated_relation.dict()
        except Exception as e:
            logger.error(f"Relation validation failed: {e}")
            # Fall back to base schema
            base_relation = BaseRelationSchema(**relation_data)
            return base_relation.dict()


# Global schema registry instance
schema_registry = SchemaRegistry()
```

### 2. Create Schema-Aware Storage Service

**File**: `packages/morag-graph/src/morag_graph/graphiti/schema_storage.py`

```python
"""Schema-aware storage service for custom entity types."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .entity_storage import GraphitiEntityStorage, EntityStorageResult
from .custom_schema import schema_registry, MoragEntityType, MoragRelationType
from morag_graph.models import Entity, Relation

logger = logging.getLogger(__name__)


class SchemaAwareEntityStorage(GraphitiEntityStorage):
    """Enhanced entity storage with custom schema validation."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.schema_registry = schema_registry
    
    async def store_typed_entity(
        self,
        entity_data: Dict[str, Any],
        auto_deduplicate: bool = True,
        validate_schema: bool = True
    ) -> EntityStorageResult:
        """Store entity with schema validation.
        
        Args:
            entity_data: Raw entity data
            auto_deduplicate: Whether to check for existing entities
            validate_schema: Whether to validate against schema
            
        Returns:
            EntityStorageResult with validation info
        """
        try:
            # Validate against custom schema if requested
            if validate_schema:
                validated_data = self.schema_registry.validate_entity(entity_data)
                logger.debug(f"Entity validated against schema: {validated_data.get('type')}")
            else:
                validated_data = entity_data
            
            # Convert to MoRAG Entity model
            entity = self._dict_to_entity(validated_data)
            
            # Store using parent method
            result = await self.store_entity(entity, auto_deduplicate)
            
            # Add schema validation info to result
            if hasattr(result, 'metadata'):
                result.metadata = result.metadata or {}
                result.metadata['schema_validated'] = validate_schema
                result.metadata['entity_type'] = validated_data.get('type')
            
            return result
            
        except Exception as e:
            logger.error(f"Schema-aware entity storage failed: {e}")
            return EntityStorageResult(
                success=False,
                error=f"Schema validation/storage error: {str(e)}"
            )
    
    async def store_typed_relation(
        self,
        relation_data: Dict[str, Any],
        relation_category: str = "base",
        validate_schema: bool = True
    ) -> Dict[str, Any]:
        """Store relation with schema validation.
        
        Args:
            relation_data: Raw relation data
            relation_category: Category for schema selection
            validate_schema: Whether to validate against schema
            
        Returns:
            Storage result
        """
        try:
            # Validate against custom schema if requested
            if validate_schema:
                validated_data = self.schema_registry.validate_relation(relation_data, relation_category)
                logger.debug(f"Relation validated against {relation_category} schema")
            else:
                validated_data = relation_data
            
            # Convert to MoRAG Relation model
            relation = self._dict_to_relation(validated_data)
            
            # Store using parent method
            result = await self.store_relation(relation)
            
            return {
                'success': result.success,
                'relation_id': result.relation_id,
                'episode_id': result.episode_id,
                'schema_validated': validate_schema,
                'relation_category': relation_category,
                'error': result.error
            }
            
        except Exception as e:
            logger.error(f"Schema-aware relation storage failed: {e}")
            return {
                'success': False,
                'error': f"Schema validation/storage error: {str(e)}"
            }
    
    async def enhance_existing_entities(
        self,
        entity_type_filter: Optional[MoragEntityType] = None,
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """Enhance existing entities with schema-based validation and enrichment.
        
        Args:
            entity_type_filter: Optional filter for specific entity types
            batch_size: Number of entities to process per batch
            
        Returns:
            Enhancement results
        """
        results = {
            'total_processed': 0,
            'enhanced_entities': 0,
            'validation_errors': 0,
            'errors': []
        }
        
        try:
            # Search for existing entities
            search_query = "adapter_type:entity"
            if entity_type_filter:
                search_query += f" type:{entity_type_filter.value}"
            
            search_results = await self.graphiti.search(query=search_query, limit=1000)
            
            for i in range(0, len(search_results), batch_size):
                batch = search_results[i:i + batch_size]
                
                for result in batch:
                    try:
                        metadata = getattr(result, 'metadata', {})
                        
                        # Skip if already schema-validated
                        if metadata.get('schema_validated'):
                            continue
                        
                        # Validate and enhance
                        enhanced_data = self.schema_registry.validate_entity(metadata)
                        
                        # Update episode with enhanced data
                        # Note: This would require episode update functionality in Graphiti
                        # For now, we'll log the enhancement
                        logger.info(f"Enhanced entity {metadata.get('id')} with schema validation")
                        
                        results['enhanced_entities'] += 1
                        
                    except Exception as e:
                        results['validation_errors'] += 1
                        results['errors'].append(str(e))
                    
                    results['total_processed'] += 1
                
                logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} entities")
        
        except Exception as e:
            results['errors'].append(f"Enhancement process failed: {str(e)}")
        
        return results
    
    def _dict_to_entity(self, entity_data: Dict[str, Any]) -> Entity:
        """Convert dictionary to MoRAG Entity model."""
        from morag_graph.models import EntityType
        
        # Map custom types to MoRAG types
        morag_type = self._map_to_morag_type(entity_data.get('type'))
        
        return Entity(
            id=entity_data['id'],
            name=entity_data['name'],
            type=morag_type,
            confidence=entity_data.get('confidence', 0.5),
            attributes=entity_data.get('metadata', {}),
            source_doc_id=entity_data.get('source_document_id')
        )
    
    def _dict_to_relation(self, relation_data: Dict[str, Any]) -> Relation:
        """Convert dictionary to MoRAG Relation model."""
        from morag_graph.models import RelationType
        
        # Map custom types to MoRAG types
        morag_type = self._map_to_morag_relation_type(relation_data.get('relation_type'))
        
        return Relation(
            id=relation_data['id'],
            source_entity_id=relation_data['source_entity_id'],
            target_entity_id=relation_data['target_entity_id'],
            relation_type=morag_type,
            confidence=relation_data.get('confidence', 0.5),
            attributes=relation_data.get('metadata', {}),
            source_doc_id=relation_data.get('source_document_id')
        )
    
    def _map_to_morag_type(self, custom_type: str) -> 'EntityType':
        """Map custom entity type to MoRAG EntityType."""
        from morag_graph.models import EntityType
        
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORGANIZATION': EntityType.ORGANIZATION,
            'LOCATION': EntityType.LOCATION,
            'TECHNOLOGY': EntityType.TECHNOLOGY,
            'CONCEPT': EntityType.CONCEPT,
            'DOCUMENT': EntityType.DOCUMENT,
        }
        
        return mapping.get(custom_type, EntityType.UNKNOWN)
    
    def _map_to_morag_relation_type(self, custom_type: str) -> 'RelationType':
        """Map custom relation type to MoRAG RelationType."""
        from morag_graph.models import RelationType
        
        mapping = {
            'MENTIONS': RelationType.MENTIONS,
            'CONTAINS': RelationType.CONTAINS,
            'RELATED_TO': RelationType.RELATED_TO,
            'REFERENCES': RelationType.REFERENCES,
        }
        
        return mapping.get(custom_type, RelationType.UNKNOWN)


class SchemaAwareSearchService:
    """Search service with schema-aware filtering and enhancement."""
    
    def __init__(self, storage_service: SchemaAwareEntityStorage):
        self.storage_service = storage_service
        self.graphiti = storage_service.graphiti
    
    async def search_by_entity_type(
        self,
        entity_type: MoragEntityType,
        query: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search entities by specific type.
        
        Args:
            entity_type: Entity type to search for
            query: Optional additional query
            limit: Maximum results
            
        Returns:
            List of typed entities
        """
        search_parts = [f"type:{entity_type.value}", "adapter_type:entity"]
        
        if query:
            search_parts.append(query)
        
        search_query = " AND ".join(search_parts)
        
        results = await self.graphiti.search(query=search_query, limit=limit)
        
        typed_entities = []
        for result in results:
            metadata = getattr(result, 'metadata', {})
            
            # Validate against schema
            try:
                validated_data = schema_registry.validate_entity(metadata)
                typed_entities.append(validated_data)
            except Exception as e:
                logger.warning(f"Schema validation failed for search result: {e}")
                typed_entities.append(metadata)
        
        return typed_entities
    
    async def search_semantic_relations(
        self,
        relation_type: MoragRelationType,
        entity_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for semantic relations of specific type.
        
        Args:
            relation_type: Relation type to search for
            entity_id: Optional entity ID filter
            limit: Maximum results
            
        Returns:
            List of semantic relations
        """
        search_parts = [f"relation_type:{relation_type.value}", "adapter_type:relation"]
        
        if entity_id:
            search_parts.append(f"(source_entity_id:{entity_id} OR target_entity_id:{entity_id})")
        
        search_query = " AND ".join(search_parts)
        
        results = await self.graphiti.search(query=search_query, limit=limit)
        
        semantic_relations = []
        for result in results:
            metadata = getattr(result, 'metadata', {})
            
            # Validate against semantic relation schema
            try:
                validated_data = schema_registry.validate_relation(metadata, "semantic")
                semantic_relations.append(validated_data)
            except Exception as e:
                logger.warning(f"Semantic relation validation failed: {e}")
                semantic_relations.append(metadata)
        
        return semantic_relations
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_custom_schema.py`

```python
"""Unit tests for custom schema functionality."""

import pytest
from datetime import datetime
from morag_graph.graphiti.custom_schema import (
    PersonEntity, OrganizationEntity, TechnologyEntity,
    SemanticRelation, TemporalRelation, SchemaRegistry,
    MoragEntityType, MoragRelationType
)


class TestCustomEntitySchemas:
    """Test custom entity schema definitions."""
    
    def test_person_entity_validation(self):
        """Test person entity schema validation."""
        person_data = {
            'id': 'person_1',
            'name': 'John Doe',
            'type': MoragEntityType.PERSON,
            'confidence': 0.9,
            'title': 'Dr.',
            'organization': 'MIT',
            'role': 'Professor',
            'email': 'john.doe@mit.edu',
            'expertise_areas': ['AI', 'Machine Learning']
        }
        
        person = PersonEntity(**person_data)
        
        assert person.name == 'John Doe'
        assert person.type == MoragEntityType.PERSON
        assert person.confidence == 0.9
        assert person.title == 'Dr.'
        assert 'AI' in person.expertise_areas
    
    def test_person_entity_invalid_email(self):
        """Test person entity with invalid email."""
        person_data = {
            'id': 'person_1',
            'name': 'John Doe',
            'type': MoragEntityType.PERSON,
            'confidence': 0.9,
            'email': 'invalid-email'
        }
        
        with pytest.raises(ValueError, match="Invalid email format"):
            PersonEntity(**person_data)
    
    def test_organization_entity_validation(self):
        """Test organization entity schema validation."""
        org_data = {
            'id': 'org_1',
            'name': 'Microsoft Corporation',
            'type': MoragEntityType.ORGANIZATION,
            'confidence': 0.95,
            'organization_type': 'company',
            'industry': 'Technology',
            'location': 'Redmond, WA',
            'founded_year': 1975
        }
        
        org = OrganizationEntity(**org_data)
        
        assert org.name == 'Microsoft Corporation'
        assert org.organization_type == 'company'
        assert org.founded_year == 1975
    
    def test_technology_entity_validation(self):
        """Test technology entity schema validation."""
        tech_data = {
            'id': 'tech_1',
            'name': 'Python',
            'type': MoragEntityType.TECHNOLOGY,
            'confidence': 0.9,
            'category': 'programming_language',
            'version': '3.9',
            'license': 'PSF',
            'maturity_level': 'stable'
        }
        
        tech = TechnologyEntity(**tech_data)
        
        assert tech.name == 'Python'
        assert tech.category == 'programming_language'
        assert tech.maturity_level == 'stable'


class TestCustomRelationSchemas:
    """Test custom relation schema definitions."""
    
    def test_semantic_relation_validation(self):
        """Test semantic relation schema validation."""
        relation_data = {
            'id': 'rel_1',
            'source_entity_id': 'entity_1',
            'target_entity_id': 'entity_2',
            'relation_type': MoragRelationType.RELATED_TO,
            'confidence': 0.8,
            'strength': 0.7,
            'directionality': 'bidirectional',
            'evidence_text': 'Both entities are mentioned together'
        }
        
        relation = SemanticRelation(**relation_data)
        
        assert relation.relation_type == MoragRelationType.RELATED_TO
        assert relation.strength == 0.7
        assert relation.directionality == 'bidirectional'
    
    def test_semantic_relation_invalid_directionality(self):
        """Test semantic relation with invalid directionality."""
        relation_data = {
            'id': 'rel_1',
            'source_entity_id': 'entity_1',
            'target_entity_id': 'entity_2',
            'relation_type': MoragRelationType.RELATED_TO,
            'confidence': 0.8,
            'directionality': 'invalid'
        }
        
        with pytest.raises(ValueError, match="Directionality must be"):
            SemanticRelation(**relation_data)
    
    def test_temporal_relation_validation(self):
        """Test temporal relation schema validation."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 2, 12, 0, 0)
        
        relation_data = {
            'id': 'temp_rel_1',
            'source_entity_id': 'entity_1',
            'target_entity_id': 'entity_2',
            'relation_type': MoragRelationType.PRECEDES,
            'confidence': 0.9,
            'start_time': start_time,
            'end_time': end_time,
            'temporal_precision': 'exact'
        }
        
        relation = TemporalRelation(**relation_data)
        
        assert relation.start_time == start_time
        assert relation.end_time == end_time
        assert relation.temporal_precision == 'exact'
    
    def test_temporal_relation_invalid_times(self):
        """Test temporal relation with invalid time order."""
        start_time = datetime(2024, 1, 2, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 0, 0)  # Before start time
        
        relation_data = {
            'id': 'temp_rel_1',
            'source_entity_id': 'entity_1',
            'target_entity_id': 'entity_2',
            'relation_type': MoragRelationType.PRECEDES,
            'confidence': 0.9,
            'start_time': start_time,
            'end_time': end_time
        }
        
        with pytest.raises(ValueError, match="End time must be after start time"):
            TemporalRelation(**relation_data)


class TestSchemaRegistry:
    """Test schema registry functionality."""
    
    def test_schema_registry_entity_validation(self):
        """Test entity validation through registry."""
        registry = SchemaRegistry()
        
        person_data = {
            'id': 'person_1',
            'name': 'Jane Smith',
            'type': 'PERSON',
            'confidence': 0.85,
            'title': 'Dr.',
            'email': 'jane@example.com'
        }
        
        validated = registry.validate_entity(person_data)
        
        assert validated['name'] == 'Jane Smith'
        assert validated['type'] == 'PERSON'
        assert validated['title'] == 'Dr.'
    
    def test_schema_registry_relation_validation(self):
        """Test relation validation through registry."""
        registry = SchemaRegistry()
        
        relation_data = {
            'id': 'rel_1',
            'source_entity_id': 'entity_1',
            'target_entity_id': 'entity_2',
            'relation_type': 'RELATED_TO',
            'confidence': 0.8,
            'strength': 0.6,
            'directionality': 'unidirectional'
        }
        
        validated = registry.validate_relation(relation_data, "semantic")
        
        assert validated['relation_type'] == 'RELATED_TO'
        assert validated['strength'] == 0.6
        assert validated['directionality'] == 'unidirectional'
    
    def test_schema_registry_custom_registration(self):
        """Test custom schema registration."""
        registry = SchemaRegistry()
        
        # Register custom entity schema
        class CustomEntity(PersonEntity):
            custom_field: str = "default"
        
        registry.register_entity_schema(MoragEntityType.PERSON, CustomEntity)
        
        # Test that custom schema is used
        schema_class = registry.get_entity_schema(MoragEntityType.PERSON)
        assert schema_class == CustomEntity
```

## Validation Checklist

- [ ] Custom entity types defined with appropriate attributes
- [ ] Relation schemas support semantic and temporal properties
- [ ] Schema validation works correctly for all types
- [ ] Registry manages multiple schema types effectively
- [ ] Storage service integrates schema validation
- [ ] Search service supports type-aware filtering
- [ ] Migration tools handle existing data
- [ ] Performance acceptable for schema validation
- [ ] Unit tests cover all schema functionality
- [ ] Documentation explains custom type usage

## Success Criteria

1. **Type Safety**: Schema validation prevents invalid data
2. **Extensibility**: Easy to add new entity and relation types
3. **Performance**: Schema validation doesn't significantly impact storage
4. **Compatibility**: Works with existing MoRAG entity types
5. **Usability**: Clear API for working with custom types

## Next Steps

After completing this step:
1. Test schema validation with real data
2. Create domain-specific entity types for your use case
3. Validate performance impact of schema validation
4. Proceed to [Step 10: Hybrid Search Enhancement](./step-10-hybrid-search.md)

## Performance Considerations

- Schema validation adds processing overhead
- Cache validated schemas for repeated use
- Batch validation for large datasets
- Optional validation for performance-critical paths
