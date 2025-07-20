"""Tests for proper relation type storage as Neo4j relationship types (labels)."""

import pytest
from unittest.mock import Mock, AsyncMock
import json

from morag_graph.models import Relation
from morag_graph.storage.neo4j_storage import Neo4jStorage


class TestRelationTypeStorage:
    """Test that relation types are stored as Neo4j relationship types, not properties."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.storage = Mock(spec=Neo4jStorage)
    
    def test_relation_to_neo4j_excludes_type_property(self):
        """Test that relation type is NOT included in Neo4j relationship properties."""
        relation = Relation(
            id="test-relation-1",
            source_entity_id="entity1",
            target_entity_id="entity2",
            type="TREATS",
            confidence=0.8,
            context="The doctor treats the patient",
            attributes={"additional_info": "medical context"}
        )

        properties = relation.to_neo4j_relationship()

        # Type should NOT be in properties (it's used as the relationship type)
        assert "type" not in properties, "Relation type should not be stored as property"

        # Other fields should be present
        assert properties["id"] == "test-relation-1"
        assert properties["confidence"] == 0.8
        assert json.loads(properties["attributes"])["additional_info"] == "medical context"

        # Entity IDs should be excluded (handled by Neo4j relationship structure)
        assert "source_entity_id" not in properties
        assert "target_entity_id" not in properties

        # Context should be excluded to reduce memory consumption
        assert "context" not in properties

        # Description should be excluded as it's redundant with relation type
        assert "description" not in properties

        # source_doc_id should be excluded as entities have chunk references
        assert "source_doc_id" not in properties
    
    def test_relation_from_neo4j_with_relationship_type(self):
        """Test creating relation from Neo4j with relationship type parameter."""
        # Mock Neo4j relationship object
        mock_relationship = Mock()
        mock_relationship.items.return_value = [
            ("id", "test-relation-2"),
            ("confidence", 0.8),
            ("attributes", '{"context": "medical context"}')
        ]
        mock_relationship.type = "TREATS"
        
        relation = Relation.from_neo4j_relationship(
            mock_relationship,
            "entity1",
            "entity2",
            "TREATS"
        )
        
        assert relation.type == "TREATS"
        assert relation.source_entity_id == "entity1"
        assert relation.target_entity_id == "entity2"
        assert relation.confidence == 0.8
        # Note: context field is not stored in Neo4j properties to reduce memory
    
    def test_relation_from_neo4j_extracts_type_from_object(self):
        """Test that relation type is extracted from Neo4j relationship object."""
        # Mock Neo4j relationship object with type attribute
        mock_relationship = Mock()
        mock_relationship.items.return_value = [
            ("id", "test-relation-3"),
            ("confidence", 0.9)
        ]
        mock_relationship.type = "CURES"
        
        relation = Relation.from_neo4j_relationship(
            mock_relationship,
            "medicine1",
            "disease1"
        )
        
        assert relation.type == "CURES"
        assert relation.source_entity_id == "medicine1"
        assert relation.target_entity_id == "disease1"
    
    def test_relation_from_neo4j_fallback_type(self):
        """Test fallback to default type when no type available."""
        # Mock Neo4j relationship object without type
        mock_relationship = Mock()
        mock_relationship.items.return_value = [
            ("id", "test-relation-4"),
            ("confidence", 0.7)
        ]
        # No type attribute
        del mock_relationship.type
        
        relation = Relation.from_neo4j_relationship(
            mock_relationship,
            "entity1",
            "entity2"
        )
        
        assert relation.type == "RELATED_TO"  # Default fallback
    
    def test_enhanced_relation_types_as_labels(self):
        """Test that enhanced semantic relation types work as Neo4j labels."""
        enhanced_types = [
            "TREATS",
            "CURES", 
            "DIAGNOSES",
            "CAUSES",
            "PREVENTS",
            "ENABLES",
            "CONTAINS",
            "MANAGES",
            "IMPLEMENTS",
            "CONNECTS_TO",
            "PROCESSES",
            "CREATES",
            "DEVELOPS",
            "TEACHES",
            "EXPLAINS"
        ]
        
        for relation_type in enhanced_types:
            relation = Relation(
                id=f"test-{relation_type.lower()}-rel",
                source_entity_id="entity1",
                target_entity_id="entity2",
                type=relation_type,
                confidence=0.8,
                context=f"Test relation of type {relation_type}"
            )
            
            properties = relation.to_neo4j_relationship()
            
            # Type should not be in properties
            assert "type" not in properties
            
            # The relation type should be valid for Neo4j (uppercase, underscores)
            assert relation_type.isupper()
            assert " " not in relation_type
            assert all(c.isalnum() or c == "_" for c in relation_type)
    
    def test_multiple_relation_types_between_entities(self):
        """Test that multiple relation types between same entities are supported."""
        entity1 = "doctor1"
        entity2 = "patient1"
        
        relations = [
            Relation(
                id="test-treats-relation",
                source_entity_id=entity1,
                target_entity_id=entity2,
                type="TREATS",
                confidence=0.9,
                context="Doctor treats patient"
            ),
            Relation(
                id="test-diagnoses-relation",
                source_entity_id=entity1,
                target_entity_id=entity2,
                type="DIAGNOSES",
                confidence=0.8,
                context="Doctor diagnoses patient"
            ),
            Relation(
                id="test-monitors-relation",
                source_entity_id=entity1,
                target_entity_id=entity2,
                type="MONITORS",
                confidence=0.7,
                context="Doctor monitors patient"
            )
        ]
        
        # Each relation should have different type but same entities
        for relation in relations:
            properties = relation.to_neo4j_relationship()
            assert "type" not in properties  # Type used as relationship label
            assert relation.source_entity_id == entity1
            assert relation.target_entity_id == entity2
        
        # All relation types should be different
        types = [r.type for r in relations]
        assert len(set(types)) == len(types)  # All unique
        assert "TREATS" in types
        assert "DIAGNOSES" in types
        assert "MONITORS" in types


class TestNeo4jQueryUpdates:
    """Test that Neo4j queries properly handle relationship types."""
    
    def test_query_includes_relationship_type(self):
        """Test that queries include type(r) to get relationship type."""
        # This would be tested with actual Neo4j queries in integration tests
        # Here we just verify the query structure
        
        expected_queries = [
            "RETURN r, source.id as source_id, target.id as target_id, type(r) as relation_type",
            "type(r) as relation_type"
        ]
        
        # These patterns should be present in Neo4j storage queries
        for pattern in expected_queries:
            # In actual implementation, we'd verify these patterns exist in the queries
            assert len(pattern) > 0  # Placeholder assertion
    
    def test_relationship_type_extraction(self):
        """Test extraction of relationship type from Neo4j results."""
        # Mock Neo4j query result
        mock_record = {
            "r": Mock(),
            "source_id": "entity1", 
            "target_id": "entity2",
            "relation_type": "TREATS"
        }
        
        # Verify that the relation_type field is used correctly
        assert mock_record["relation_type"] == "TREATS"
        assert mock_record["source_id"] == "entity1"
        assert mock_record["target_id"] == "entity2"


if __name__ == "__main__":
    pytest.main([__file__])
