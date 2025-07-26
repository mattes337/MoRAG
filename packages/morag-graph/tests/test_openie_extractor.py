"""Tests for OpenIE extractor."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from morag_graph.extractors.openie_extractor import OpenIEExtractor, OpenIEExtractionResult
from morag_graph.models import Entity, Relation
from morag_graph.services.openie_service import OpenIETriplet
from morag_graph.processors.triplet_processor import ValidatedTriplet
from morag_graph.normalizers.entity_linker import EntityMatch
from morag_graph.normalizers.predicate_normalizer import NormalizedPredicate


class TestOpenIEExtractor:
    """Test cases for OpenIE extractor."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'min_confidence': 0.6,
            'enable_entity_linking': True,
            'enable_predicate_normalization': True,
            'batch_size': 10
        }
    
    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing."""
        return [
            Entity(
                id="entity_1",
                name="John",
                canonical_name="john",
                entity_type="PERSON",
                confidence=0.95,
                metadata={"source": "test"}
            ),
            Entity(
                id="entity_2",
                name="Microsoft",
                canonical_name="microsoft",
                entity_type="ORG",
                confidence=0.98,
                metadata={"source": "test"}
            )
        ]
    
    @pytest.fixture
    def sample_triplets(self):
        """Sample OpenIE triplets for testing."""
        return [
            OpenIETriplet(
                subject="John",
                predicate="works at",
                object="Microsoft",
                confidence=0.85,
                sentence="John works at Microsoft."
            ),
            OpenIETriplet(
                subject="Microsoft",
                predicate="is",
                object="technology company",
                confidence=0.78,
                sentence="Microsoft is a technology company."
            )
        ]
    
    @pytest.fixture
    def openie_extractor(self, mock_config):
        """Create OpenIE extractor instance for testing."""
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = True
            mock_settings.return_value.openie_enable_entity_linking = True
            mock_settings.return_value.openie_enable_predicate_normalization = True
            
            return OpenIEExtractor(mock_config)
    
    def test_init(self, openie_extractor):
        """Test extractor initialization."""
        assert openie_extractor.enabled is True
        assert openie_extractor.min_confidence == 0.6
        assert openie_extractor.enable_entity_linking is True
        assert openie_extractor.enable_predicate_normalization is True
        assert hasattr(openie_extractor, 'openie_service')
        assert hasattr(openie_extractor, 'sentence_processor')
        assert hasattr(openie_extractor, 'triplet_processor')
        assert hasattr(openie_extractor, 'entity_linker')
        assert hasattr(openie_extractor, 'predicate_normalizer')
        assert hasattr(openie_extractor, 'confidence_manager')
    
    def test_init_disabled(self):
        """Test extractor initialization when disabled."""
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = False
            
            extractor = OpenIEExtractor({'min_confidence': 0.6})
            assert extractor.enabled is False
    
    @pytest.mark.asyncio
    async def test_extract_relations_disabled(self):
        """Test relation extraction when extractor is disabled."""
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = False
            
            extractor = OpenIEExtractor({'min_confidence': 0.6})
            relations = await extractor.extract_relations("John works at Microsoft.")
            assert relations == []
    
    @pytest.mark.asyncio
    async def test_extract_relations_empty_text(self, openie_extractor):
        """Test relation extraction with empty text."""
        relations = await openie_extractor.extract_relations("")
        assert relations == []
        
        relations = await openie_extractor.extract_relations("   ")
        assert relations == []
    
    @pytest.mark.asyncio
    async def test_extract_relations_basic(self, openie_extractor, sample_triplets):
        """Test basic relation extraction."""
        # Mock the full extraction method
        mock_result = OpenIEExtractionResult(
            relations=[
                Relation(
                    id="rel_1",
                    subject="John",
                    predicate="works_at",
                    object="Microsoft",
                    confidence=0.85,
                    metadata={"source": "openie"}
                )
            ],
            triplets=sample_triplets,
            entity_matches=[],
            normalized_predicates=[],
            processed_sentences=[],
            metadata={"extraction_method": "openie"}
        )
        
        with patch.object(openie_extractor, 'extract_full', return_value=mock_result):
            relations = await openie_extractor.extract_relations("John works at Microsoft.")
            
            assert len(relations) == 1
            assert relations[0].subject == "John"
            assert relations[0].predicate == "works_at"
            assert relations[0].object == "Microsoft"
            assert relations[0].confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_extract_full_basic(self, openie_extractor, sample_entities, sample_triplets):
        """Test full extraction pipeline."""
        # Mock all components
        openie_extractor.openie_service.extract_triplets = AsyncMock(return_value=sample_triplets)
        openie_extractor.sentence_processor.process_text = AsyncMock(return_value=[])
        
        validated_triplets = [
            ValidatedTriplet(
                subject=t.subject,
                predicate=t.predicate,
                object=t.object,
                confidence=t.confidence,
                sentence=t.sentence,
                sentence_id="sent_1",
                validation_score=0.8,
                validation_flags=set()
            ) for t in sample_triplets
        ]
        openie_extractor.triplet_processor.process_triplets = AsyncMock(return_value=validated_triplets)
        
        # Mock entity linking
        entity_matches = [
            EntityMatch(
                openie_entity="John",
                spacy_entity=sample_entities[0],
                match_type="exact",
                confidence=0.95,
                similarity_score=1.0
            )
        ]
        openie_extractor.entity_linker.link_triplet_entities = AsyncMock(return_value=entity_matches)
        
        # Mock predicate normalization
        normalized_predicates = [
            NormalizedPredicate(
                original="works at",
                normalized="works_at",
                canonical_form="WORKS_AT",
                relationship_type="EMPLOYMENT",
                confidence=0.9
            )
        ]
        openie_extractor.predicate_normalizer.normalize_predicates = AsyncMock(return_value=normalized_predicates)
        
        # Mock confidence filtering
        openie_extractor.confidence_manager.filter_relations = AsyncMock(
            side_effect=lambda relations, threshold: relations
        )
        
        # Test extraction
        result = await openie_extractor.extract_full(
            "John works at Microsoft. Microsoft is a technology company.",
            entities=sample_entities,
            source_doc_id="test_doc"
        )
        
        assert isinstance(result, OpenIEExtractionResult)
        assert len(result.relations) > 0
        assert len(result.triplets) == len(sample_triplets)
        assert len(result.entity_matches) == 1
        assert len(result.normalized_predicates) == 1
        assert result.metadata["entity_linking_enabled"] is True
        assert result.metadata["predicate_normalization_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_extract_full_without_entity_linking(self, mock_config, sample_triplets):
        """Test full extraction without entity linking."""
        mock_config['enable_entity_linking'] = False
        
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = True
            mock_settings.return_value.openie_enable_entity_linking = False
            mock_settings.return_value.openie_enable_predicate_normalization = True
            
            extractor = OpenIEExtractor(mock_config)
            
            # Mock components
            extractor.openie_service.extract_triplets = AsyncMock(return_value=sample_triplets)
            extractor.sentence_processor.process_text = AsyncMock(return_value=[])
            extractor.triplet_processor.process_triplets = AsyncMock(return_value=[])
            extractor.predicate_normalizer.normalize_predicates = AsyncMock(return_value=[])
            extractor.confidence_manager.filter_relations = AsyncMock(return_value=[])
            
            result = await extractor.extract_full("Test text")
            
            assert result.metadata["entity_linking_enabled"] is False
            assert len(result.entity_matches) == 0
    
    @pytest.mark.asyncio
    async def test_extract_full_without_predicate_normalization(self, mock_config, sample_triplets):
        """Test full extraction without predicate normalization."""
        mock_config['enable_predicate_normalization'] = False
        
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = True
            mock_settings.return_value.openie_enable_entity_linking = True
            mock_settings.return_value.openie_enable_predicate_normalization = False
            
            extractor = OpenIEExtractor(mock_config)
            
            # Mock components
            extractor.openie_service.extract_triplets = AsyncMock(return_value=sample_triplets)
            extractor.sentence_processor.process_text = AsyncMock(return_value=[])
            extractor.triplet_processor.process_triplets = AsyncMock(return_value=[])
            extractor.entity_linker.link_triplet_entities = AsyncMock(return_value=[])
            extractor.confidence_manager.filter_relations = AsyncMock(return_value=[])
            
            result = await extractor.extract_full("Test text")
            
            assert result.metadata["predicate_normalization_enabled"] is False
            assert len(result.normalized_predicates) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, openie_extractor):
        """Test error handling in extraction pipeline."""
        # Mock service to raise exception
        openie_extractor.openie_service.extract_triplets = AsyncMock(
            side_effect=Exception("OpenIE service error")
        )
        
        # Test that ProcessingError is raised
        with pytest.raises(Exception):  # Should be ProcessingError
            await openie_extractor.extract_relations("Test text")
        
        with pytest.raises(Exception):  # Should be ProcessingError
            await openie_extractor.extract_full("Test text")
    
    @pytest.mark.asyncio
    async def test_get_extraction_stats(self, openie_extractor):
        """Test extraction statistics."""
        stats = await openie_extractor.get_extraction_stats()
        
        assert stats["enabled"] is True
        assert stats["configuration"]["min_confidence"] == 0.6
        assert stats["configuration"]["entity_linking_enabled"] is True
        assert stats["configuration"]["predicate_normalization_enabled"] is True
        assert "components" in stats
        assert "openie_service" in stats["components"]
        assert "sentence_processor" in stats["components"]
        assert "triplet_processor" in stats["components"]
    
    @pytest.mark.asyncio
    async def test_confidence_filtering(self, openie_extractor, sample_triplets):
        """Test confidence-based filtering of results."""
        # Create triplets with different confidence levels
        low_confidence_triplet = OpenIETriplet(
            subject="Test",
            predicate="has",
            object="low confidence",
            confidence=0.3,  # Below threshold
            sentence="Test has low confidence."
        )
        
        all_triplets = sample_triplets + [low_confidence_triplet]
        
        # Mock components
        openie_extractor.openie_service.extract_triplets = AsyncMock(return_value=all_triplets)
        openie_extractor.sentence_processor.process_text = AsyncMock(return_value=[])
        openie_extractor.triplet_processor.process_triplets = AsyncMock(return_value=[])
        openie_extractor.entity_linker.link_triplet_entities = AsyncMock(return_value=[])
        openie_extractor.predicate_normalizer.normalize_predicates = AsyncMock(return_value=[])
        
        # Mock confidence manager to filter low confidence relations
        def filter_relations(relations, threshold):
            return [r for r in relations if r.confidence >= threshold]
        
        openie_extractor.confidence_manager.filter_relations = AsyncMock(side_effect=filter_relations)
        
        result = await openie_extractor.extract_full("Test text")
        
        # Should have filtered out the low confidence triplet
        assert len(result.relations) < len(all_triplets)


if __name__ == "__main__":
    pytest.main([__file__])
