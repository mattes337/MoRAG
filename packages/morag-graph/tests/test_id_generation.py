"""Unit tests for unified ID generation system."""

import pytest
from unittest.mock import patch
from morag_graph.utils.id_generation import (
    UnifiedIDGenerator,
    IDValidator,
    IDCollisionDetector,
    IDValidationError,
    IDCollisionError
)


class TestUnifiedIDGenerator:
    """Test cases for UnifiedIDGenerator."""
    
    def test_generate_document_id_with_checksum(self):
        """Test document ID generation with checksum."""
        doc_id = UnifiedIDGenerator.generate_document_id(
            source_file="test.pdf",
            checksum="abc123"
        )
        assert doc_id.startswith("doc_")
        assert "test.pdf" in doc_id
        assert "abc123" in doc_id
        assert len(doc_id.split("_")) == 3  # doc_filename_checksum
    
    def test_generate_document_id_without_checksum(self):
        """Test document ID generation without checksum."""
        with patch('time.time', return_value=1234567890):
            doc_id = UnifiedIDGenerator.generate_document_id(
                source_file="test.pdf"
            )
        assert doc_id.startswith("doc_")
        assert "test.pdf" in doc_id
        assert "1234567890" in doc_id
    
    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        chunk_id = UnifiedIDGenerator.generate_chunk_id(
            document_id="doc_test.pdf_abc123",
            chunk_index=5
        )
        expected = "doc_test.pdf_abc123:chunk:5"
        assert chunk_id == expected
    
    def test_generate_entity_id(self):
        """Test entity ID generation."""
        entity_id = UnifiedIDGenerator.generate_entity_id(
            name="John Doe",
            entity_type="PERSON",
            source_doc_id="doc_test.pdf_abc123"
        )
        assert entity_id.startswith("ent_")
        assert "john_doe" in entity_id.lower()
        assert "person" in entity_id.lower()

    def test_generate_relation_id(self):
        """Test relation ID generation."""
        relation_id = UnifiedIDGenerator.generate_relation_id(
            source_entity_id="ent_john_doe_person_abc123",
            target_entity_id="ent_company_org_abc123",
            relation_type="WORKS_FOR"
        )
        assert relation_id.startswith("rel_")
        assert "works_for" in relation_id.lower()
    
    def test_parse_id_type(self):
        """Test ID type parsing."""
        assert UnifiedIDGenerator.parse_id_type("doc_test.pdf_abc123") == "document"
        assert UnifiedIDGenerator.parse_id_type("doc_test.pdf_abc123:chunk:5") == "chunk"
        assert UnifiedIDGenerator.parse_id_type("ent_john_doe_person_abc123") == "entity"
        assert UnifiedIDGenerator.parse_id_type("rel_works_for_abc123") == "relation"
        assert UnifiedIDGenerator.parse_id_type("invalid_id") == "unknown"
    
    def test_extract_document_id_from_chunk(self):
        """Test extracting document ID from chunk ID."""
        chunk_id = "doc_test.pdf_abc123:chunk:5"
        doc_id = UnifiedIDGenerator.extract_document_id_from_chunk(chunk_id)
        assert doc_id == "doc_test.pdf_abc123"
    
    def test_extract_chunk_index_from_chunk_id(self):
        """Test extracting chunk index from chunk ID."""
        chunk_id = "doc_test.pdf_abc123:chunk:5"
        index = UnifiedIDGenerator.extract_chunk_index_from_chunk_id(chunk_id)
        assert index == 5
    
    def test_extract_chunk_index_invalid_id(self):
        """Test extracting chunk index from invalid ID."""
        with pytest.raises(ValueError):
            UnifiedIDGenerator.extract_chunk_index_from_chunk_id("invalid_id")


class TestIDValidator:
    """Test cases for IDValidator."""
    
    def test_validate_document_id_valid(self):
        """Test validation of valid document ID."""
        valid_id = "doc_test.pdf_abc123"
        assert IDValidator.validate_document_id(valid_id) is True
    
    def test_validate_document_id_invalid(self):
        """Test validation of invalid document ID."""
        with pytest.raises(IDValidationError):
            IDValidator.validate_document_id("invalid_doc_id")
    
    def test_validate_chunk_id_valid(self):
        """Test validation of valid chunk ID."""
        valid_id = "doc_test.pdf_abc123:chunk:5"
        assert IDValidator.validate_chunk_id(valid_id) is True
    
    def test_validate_chunk_id_invalid(self):
        """Test validation of invalid chunk ID."""
        with pytest.raises(IDValidationError):
            IDValidator.validate_chunk_id("doc_test.pdf_abc123:invalid:5")
    
    def test_validate_entity_id_valid(self):
        """Test validation of valid entity ID."""
        valid_id = "ent_john_doe_person_abc123"
        assert IDValidator.validate_entity_id(valid_id) is True
    
    def test_validate_entity_id_invalid(self):
        """Test validation of invalid entity ID."""
        with pytest.raises(IDValidationError):
            IDValidator.validate_entity_id("invalid_entity_id")
    
    def test_validate_relation_id_valid(self):
        """Test validation of valid relation ID."""
        valid_id = "rel_works_for_abc123def456"
        assert IDValidator.validate_relation_id(valid_id) is True
    
    def test_validate_relation_id_invalid(self):
        """Test validation of invalid relation ID."""
        with pytest.raises(IDValidationError):
            IDValidator.validate_relation_id("invalid_relation_id")
    
    def test_is_unified_format(self):
        """Test unified format detection."""
        assert IDValidator.is_unified_format("doc_test.pdf_abc123") is True
        assert IDValidator.is_unified_format("doc_test.pdf_abc123:chunk:5") is True
        assert IDValidator.is_unified_format("ent_john_doe_person_abc123") is True
        assert IDValidator.is_unified_format("rel_works_for_abc123def456") is True
        assert IDValidator.is_unified_format("legacy_uuid_format") is False


class TestIDCollisionDetector:
    """Test cases for IDCollisionDetector."""
    
    def test_check_collision_no_collision(self):
        """Test collision detection with no collision."""
        detector = IDCollisionDetector()
        new_id = "doc_test.pdf_abc123"
        existing_ids = ["doc_other.pdf_def456", "doc_another.pdf_ghi789"]
        
        # Should not raise exception
        detector.check_collision(new_id, existing_ids)
    
    def test_check_collision_with_collision(self):
        """Test collision detection with collision."""
        detector = IDCollisionDetector()
        new_id = "doc_test.pdf_abc123"
        existing_ids = ["doc_test.pdf_abc123", "doc_other.pdf_def456"]
        
        with pytest.raises(IDCollisionError):
            detector.check_collision(new_id, existing_ids)
    
    def test_batch_check_collisions_no_collisions(self):
        """Test batch collision detection with no collisions."""
        detector = IDCollisionDetector()
        new_ids = ["doc_test1.pdf_abc123", "doc_test2.pdf_def456"]
        existing_ids = ["doc_other.pdf_ghi789"]
        
        # Should not raise exception
        detector.batch_check_collisions(new_ids, existing_ids)
    
    def test_batch_check_collisions_with_collisions(self):
        """Test batch collision detection with collisions."""
        detector = IDCollisionDetector()
        new_ids = ["doc_test1.pdf_abc123", "doc_test2.pdf_def456"]
        existing_ids = ["doc_test1.pdf_abc123", "doc_other.pdf_ghi789"]
        
        with pytest.raises(IDCollisionError):
            detector.batch_check_collisions(new_ids, existing_ids)
    
    def test_get_collision_report_no_collisions(self):
        """Test collision report with no collisions."""
        detector = IDCollisionDetector()
        new_ids = ["doc_test1.pdf_abc123", "doc_test2.pdf_def456"]
        existing_ids = ["doc_other.pdf_ghi789"]
        
        report = detector.get_collision_report(new_ids, existing_ids)
        assert report['has_collisions'] is False
        assert len(report['collisions']) == 0
    
    def test_get_collision_report_with_collisions(self):
        """Test collision report with collisions."""
        detector = IDCollisionDetector()
        new_ids = ["doc_test1.pdf_abc123", "doc_test2.pdf_def456"]
        existing_ids = ["doc_test1.pdf_abc123", "doc_other.pdf_ghi789"]
        
        report = detector.get_collision_report(new_ids, existing_ids)
        assert report['has_collisions'] is True
        assert len(report['collisions']) == 1
        assert "doc_test1.pdf_abc123" in report['collisions']


class TestIDGenerationIntegration:
    """Integration tests for ID generation system."""
    
    def test_document_chunk_id_consistency(self):
        """Test that chunk IDs are consistent with document IDs."""
        doc_id = UnifiedIDGenerator.generate_document_id(
            source_file="test.pdf",
            checksum="abc123"
        )
        
        chunk_id = UnifiedIDGenerator.generate_chunk_id(
            document_id=doc_id,
            chunk_index=0
        )
        
        extracted_doc_id = UnifiedIDGenerator.extract_document_id_from_chunk(chunk_id)
        assert extracted_doc_id == doc_id
    
    def test_entity_relation_id_consistency(self):
        """Test that relation IDs are consistent with entity IDs."""
        source_entity_id = UnifiedIDGenerator.generate_entity_id(
            name="John Doe",
            entity_type="PERSON",
            source_doc_id="doc_test.pdf_abc123"
        )

        target_entity_id = UnifiedIDGenerator.generate_entity_id(
            name="Acme Corp",
            entity_type="ORGANIZATION",
            source_doc_id="doc_test.pdf_abc123"
        )

        relation_id = UnifiedIDGenerator.generate_relation_id(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type="WORKS_FOR"
        )
        
        # Validate all IDs
        assert IDValidator.validate_entity_id(source_entity_id)
        assert IDValidator.validate_entity_id(target_entity_id)
        assert IDValidator.validate_relation_id(relation_id)
    
    def test_id_determinism(self):
        """Test that ID generation is deterministic."""
        # Generate same document ID multiple times
        doc_id_1 = UnifiedIDGenerator.generate_document_id(
            source_file="test.pdf",
            checksum="abc123"
        )
        doc_id_2 = UnifiedIDGenerator.generate_document_id(
            source_file="test.pdf",
            checksum="abc123"
        )
        assert doc_id_1 == doc_id_2
        
        # Generate same entity ID multiple times
        entity_id_1 = UnifiedIDGenerator.generate_entity_id(
            name="John Doe",
            entity_type="PERSON",
            source_doc_id="doc_test.pdf_abc123"
        )
        entity_id_2 = UnifiedIDGenerator.generate_entity_id(
            name="John Doe",
            entity_type="PERSON",
            source_doc_id="doc_test.pdf_abc123"
        )
        assert entity_id_1 == entity_id_2
    
    def test_id_uniqueness(self):
        """Test that different inputs generate unique IDs."""
        doc_id_1 = UnifiedIDGenerator.generate_document_id(
            source_file="test1.pdf",
            checksum="abc123"
        )
        doc_id_2 = UnifiedIDGenerator.generate_document_id(
            source_file="test2.pdf",
            checksum="def456"
        )
        assert doc_id_1 != doc_id_2
        
        entity_id_1 = UnifiedIDGenerator.generate_entity_id(
            name="John Doe",
            entity_type="PERSON",
            source_doc_id="doc_test.pdf_abc123"
        )
        entity_id_2 = UnifiedIDGenerator.generate_entity_id(
            name="Jane Smith",
            entity_type="PERSON",
            source_doc_id="doc_test.pdf_abc123"
        )
        assert entity_id_1 != entity_id_2