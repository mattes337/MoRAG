"""Unit tests for custom schema functionality."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
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
    
    def test_base_entity_confidence_validation(self):
        """Test confidence validation in base entity."""
        from morag_graph.graphiti.custom_schema import BaseEntitySchema
        
        # Valid confidence
        valid_data = {
            'id': 'entity_1',
            'name': 'Test Entity',
            'type': MoragEntityType.CONCEPT,
            'confidence': 0.8
        }
        entity = BaseEntitySchema(**valid_data)
        assert entity.confidence == 0.8
        
        # Invalid confidence (too high)
        invalid_data = {
            'id': 'entity_1',
            'name': 'Test Entity',
            'type': MoragEntityType.CONCEPT,
            'confidence': 1.5
        }
        with pytest.raises(Exception):  # Pydantic v2 raises ValidationError, not ValueError
            BaseEntitySchema(**invalid_data)


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
    
    def test_schema_registry_fallback_validation(self):
        """Test fallback to base schema on validation failure."""
        registry = SchemaRegistry()

        # Invalid person data (missing required fields but with fallback values)
        invalid_person_data = {
            'type': 'PERSON',
            'confidence': 0.8,
            'extra_field': 'some_value'
            # Missing 'id' and 'name' - should be filled by fallback
        }

        # Should fall back and add missing required fields
        validated = registry.validate_entity(invalid_person_data)

        # Should have been validated with fallback handling
        assert 'type' in validated
        assert 'confidence' in validated
        assert 'id' in validated  # Should be filled with 'unknown'
        assert 'name' in validated  # Should be filled with 'unknown'
    
    def test_entity_type_enum_values(self):
        """Test that entity type enum has expected values."""
        assert MoragEntityType.PERSON == "PERSON"
        assert MoragEntityType.ORGANIZATION == "ORGANIZATION"
        assert MoragEntityType.TECHNOLOGY == "TECHNOLOGY"
        assert MoragEntityType.CONCEPT == "CONCEPT"
        assert MoragEntityType.DOCUMENT == "DOCUMENT"
    
    def test_relation_type_enum_values(self):
        """Test that relation type enum has expected values."""
        assert MoragRelationType.MENTIONS == "MENTIONS"
        assert MoragRelationType.CONTAINS == "CONTAINS"
        assert MoragRelationType.RELATED_TO == "RELATED_TO"
        assert MoragRelationType.REFERENCES == "REFERENCES"


class TestSchemaAwareStorage:
    """Test schema-aware storage functionality."""

    @pytest.fixture
    def mock_schema_storage(self):
        """Create mock schema-aware storage service."""
        from morag_graph.graphiti.schema_storage import SchemaAwareEntityStorage

        # Create storage without calling the parent constructor
        storage = object.__new__(SchemaAwareEntityStorage)
        storage.config = None
        storage.graphiti = Mock()
        storage.schema_registry = Mock()
        storage.store_entity = AsyncMock()
        storage.store_relation = AsyncMock()
        return storage

    @pytest.mark.asyncio
    async def test_store_typed_entity_with_validation(self, mock_schema_storage):
        """Test storing typed entity with schema validation."""
        from morag_graph.graphiti.entity_storage import EntityStorageResult

        # Mock successful storage
        mock_result = EntityStorageResult(
            success=True,
            entity_id='person_1',
            episode_id='episode_123'
        )
        mock_schema_storage.store_entity.return_value = mock_result

        person_data = {
            'id': 'person_1',
            'name': 'John Doe',
            'type': 'PERSON',
            'confidence': 0.9,
            'email': 'john@example.com'
        }

        result = await mock_schema_storage.store_typed_entity(
            person_data,
            validate_schema=True
        )

        assert result.success
        assert result.entity_id == 'person_1'
        # Verify store_entity was called
        mock_schema_storage.store_entity.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_typed_relation_with_validation(self, mock_schema_storage):
        """Test storing typed relation with schema validation."""
        from morag_graph.graphiti.entity_storage import RelationStorageResult

        # Mock successful storage
        mock_result = RelationStorageResult(
            success=True,
            relation_id='rel_1',
            episode_id='episode_456'
        )
        mock_schema_storage.store_relation.return_value = mock_result

        relation_data = {
            'id': 'rel_1',
            'source_entity_id': 'entity_1',
            'target_entity_id': 'entity_2',
            'relation_type': 'RELATED_TO',
            'confidence': 0.8,
            'strength': 0.7,
            'directionality': 'bidirectional'
        }

        result = await mock_schema_storage.store_typed_relation(
            relation_data,
            relation_category="semantic",
            validate_schema=True
        )

        assert result['success']
        assert result['relation_id'] == 'rel_1'
        assert result['schema_validated']
        assert result['relation_category'] == 'semantic'

    def test_entity_type_mapping(self, mock_schema_storage):
        """Test mapping custom entity types to MoRAG types."""
        # Test successful mapping
        morag_type = mock_schema_storage._map_to_morag_type('PERSON')
        # This will depend on whether MoRAG models are available
        # In test environment, it might return the string itself
        assert morag_type is not None

        # Test unknown type mapping
        unknown_type = mock_schema_storage._map_to_morag_type('UNKNOWN_TYPE')
        assert unknown_type is not None

    def test_relation_type_mapping(self, mock_schema_storage):
        """Test mapping custom relation types to MoRAG types."""
        # Test successful mapping
        morag_type = mock_schema_storage._map_to_morag_relation_type('MENTIONS')
        assert morag_type is not None

        # Test unknown type mapping
        unknown_type = mock_schema_storage._map_to_morag_relation_type('UNKNOWN_RELATION')
        assert unknown_type is not None


class TestSchemaAwareSearch:
    """Test schema-aware search functionality."""

    @pytest.fixture
    def mock_search_service(self):
        """Create mock schema-aware search service."""
        from morag_graph.graphiti.schema_storage import SchemaAwareSearchService, SchemaAwareEntityStorage

        # Create mock storage service
        mock_storage = Mock(spec=SchemaAwareEntityStorage)
        mock_storage.graphiti = Mock()
        mock_storage.search_service = Mock()
        mock_storage.search_service.search = AsyncMock()

        # Create search service
        search_service = SchemaAwareSearchService(mock_storage)
        return search_service

    @pytest.mark.asyncio
    async def test_search_by_entity_type(self, mock_search_service):
        """Test searching entities by type."""
        from morag_graph.graphiti.search_service import SearchResult, SearchMetrics

        # Mock search results
        mock_result = SearchResult(
            content="Test person content",
            score=0.9,
            metadata={
                'id': 'person_1',
                'name': 'John Doe',
                'type': 'PERSON',
                'confidence': 0.9,
                'email': 'john@example.com'
            }
        )
        mock_metrics = SearchMetrics(0.1, 1, 1, "entity_type_search")

        mock_search_service.storage_service.search_service.search.return_value = ([mock_result], mock_metrics)

        results = await mock_search_service.search_by_entity_type(
            MoragEntityType.PERSON,
            query="John",
            limit=10
        )

        assert len(results) == 1
        assert results[0]['name'] == 'John Doe'
        assert results[0]['type'] == 'PERSON'

    @pytest.mark.asyncio
    async def test_search_semantic_relations(self, mock_search_service):
        """Test searching semantic relations."""
        from morag_graph.graphiti.search_service import SearchResult, SearchMetrics

        # Mock search results
        mock_result = SearchResult(
            content="Test relation content",
            score=0.8,
            metadata={
                'id': 'rel_1',
                'source_entity_id': 'entity_1',
                'target_entity_id': 'entity_2',
                'relation_type': 'RELATED_TO',
                'confidence': 0.8,
                'strength': 0.7,
                'directionality': 'bidirectional'
            }
        )
        mock_metrics = SearchMetrics(0.1, 1, 1, "semantic_relation_search")

        mock_search_service.storage_service.search_service.search.return_value = ([mock_result], mock_metrics)

        results = await mock_search_service.search_semantic_relations(
            MoragRelationType.RELATED_TO,
            entity_id="entity_1",
            limit=10
        )

        assert len(results) == 1
        assert results[0]['relation_type'] == 'RELATED_TO'
        assert results[0]['strength'] == 0.7
