"""Integration tests for the complete OpenIE pipeline."""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from morag_graph.extractors.openie_extractor import OpenIEExtractor, OpenIEExtractionResult
from morag_graph.models import Entity, Relation
from morag_graph.services.openie_service import OpenIETriplet
from morag_graph.processors.triplet_processor import ValidatedTriplet
from morag_graph.normalizers.entity_linker import EntityMatch
from morag_graph.normalizers.predicate_normalizer import NormalizedPredicate
from morag_graph.storage.neo4j_storage import Neo4jStorage


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    John works at Microsoft. The company was founded by Bill Gates.
    Microsoft develops software products. Bill Gates is a philanthropist.
    The company is headquartered in Seattle.
    """


@pytest.fixture
def sample_entities():
    """Sample entities from spaCy NER."""
    return [
        Entity(
            id="entity_1",
            name="John",
            canonical_name="john",
            entity_type="PERSON",
            confidence=0.95,
            metadata={"source": "spacy"}
        ),
        Entity(
            id="entity_2", 
            name="Microsoft",
            canonical_name="microsoft",
            entity_type="ORG",
            confidence=0.98,
            metadata={"source": "spacy"}
        ),
        Entity(
            id="entity_3",
            name="Bill Gates",
            canonical_name="bill_gates",
            entity_type="PERSON",
            confidence=0.97,
            metadata={"source": "spacy"}
        ),
        Entity(
            id="entity_4",
            name="Seattle",
            canonical_name="seattle",
            entity_type="GPE",
            confidence=0.92,
            metadata={"source": "spacy"}
        )
    ]


@pytest.fixture
def mock_openie_triplets():
    """Mock OpenIE triplets."""
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
            predicate="was founded by",
            object="Bill Gates",
            confidence=0.82,
            sentence="The company was founded by Bill Gates."
        ),
        OpenIETriplet(
            subject="Microsoft",
            predicate="develops",
            object="software products",
            confidence=0.78,
            sentence="Microsoft develops software products."
        ),
        OpenIETriplet(
            subject="Bill Gates",
            predicate="is",
            object="philanthropist",
            confidence=0.75,
            sentence="Bill Gates is a philanthropist."
        ),
        OpenIETriplet(
            subject="company",
            predicate="is headquartered in",
            object="Seattle",
            confidence=0.80,
            sentence="The company is headquartered in Seattle."
        )
    ]


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "min_confidence": 0.6,
        "enable_entity_linking": True,
        "enable_predicate_normalization": True,
        "batch_size": 10
    }


class TestOpenIEIntegration:
    """Integration tests for OpenIE pipeline."""
    
    @pytest.mark.asyncio
    async def test_openie_extractor_initialization(self, mock_config):
        """Test OpenIE extractor initialization."""
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = True
            mock_settings.return_value.openie_enable_entity_linking = True
            mock_settings.return_value.openie_enable_predicate_normalization = True
            
            extractor = OpenIEExtractor(mock_config)
            
            assert extractor.enabled
            assert extractor.enable_entity_linking
            assert extractor.enable_predicate_normalization
            assert extractor.min_confidence == 0.6
    
    @pytest.mark.asyncio
    async def test_extract_relations_basic(self, sample_text, mock_config, mock_openie_triplets):
        """Test basic relation extraction."""
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = True
            mock_settings.return_value.openie_enable_entity_linking = False
            mock_settings.return_value.openie_enable_predicate_normalization = False
            
            extractor = OpenIEExtractor(mock_config)
            
            # Mock the OpenIE service
            extractor.openie_service.extract_triplets = AsyncMock(return_value=mock_openie_triplets)
            
            # Mock other components
            extractor.sentence_processor.process_text = AsyncMock(return_value=[])
            extractor.triplet_processor.process_triplets = AsyncMock(
                return_value=[
                    ValidatedTriplet(
                        subject=t.subject,
                        predicate=t.predicate,
                        object=t.object,
                        confidence=t.confidence,
                        sentence=t.sentence,
                        sentence_id="sent_1",
                        validation_score=0.8,
                        validation_flags=set()
                    ) for t in mock_openie_triplets
                ]
            )
            extractor.confidence_manager.filter_relations = AsyncMock(
                side_effect=lambda relations, threshold: relations
            )
            
            # Test extraction
            relations = await extractor.extract_relations(sample_text, source_doc_id="test_doc")
            
            assert len(relations) == len(mock_openie_triplets)
            assert all(isinstance(r, Relation) for r in relations)
            assert relations[0].subject == "John"
            assert relations[0].predicate == "works at"
            assert relations[0].object == "Microsoft"
    
    @pytest.mark.asyncio
    async def test_extract_full_with_entity_linking(
        self, sample_text, sample_entities, mock_config, mock_openie_triplets
    ):
        """Test full extraction with entity linking."""
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = True
            mock_settings.return_value.openie_enable_entity_linking = True
            mock_settings.return_value.openie_enable_predicate_normalization = True
            
            extractor = OpenIEExtractor(mock_config)
            
            # Mock components
            extractor.openie_service.extract_triplets = AsyncMock(return_value=mock_openie_triplets)
            extractor.sentence_processor.process_text = AsyncMock(return_value=[])
            
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
                ) for t in mock_openie_triplets
            ]
            extractor.triplet_processor.process_triplets = AsyncMock(return_value=validated_triplets)
            
            # Mock entity linking
            entity_matches = [
                EntityMatch(
                    openie_entity="John",
                    spacy_entity=sample_entities[0],
                    match_type="exact",
                    confidence=0.95,
                    similarity_score=1.0
                ),
                EntityMatch(
                    openie_entity="Microsoft",
                    spacy_entity=sample_entities[1],
                    match_type="exact",
                    confidence=0.98,
                    similarity_score=1.0
                )
            ]
            extractor.entity_linker.link_triplet_entities = AsyncMock(return_value=entity_matches)
            
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
            extractor.predicate_normalizer.normalize_predicates = AsyncMock(
                return_value=normalized_predicates
            )
            
            extractor.confidence_manager.filter_relations = AsyncMock(
                side_effect=lambda relations, threshold: relations
            )
            
            # Test full extraction
            result = await extractor.extract_full(
                sample_text, sample_entities, source_doc_id="test_doc"
            )
            
            assert isinstance(result, OpenIEExtractionResult)
            assert len(result.relations) > 0
            assert len(result.triplets) == len(mock_openie_triplets)
            assert len(result.entity_matches) == 2
            assert len(result.normalized_predicates) > 0
            assert result.metadata["entity_linking_enabled"]
            assert result.metadata["predicate_normalization_enabled"]
    
    @pytest.mark.asyncio
    async def test_disabled_extractor(self, sample_text, mock_config):
        """Test behavior when OpenIE is disabled."""
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = False
            
            extractor = OpenIEExtractor(mock_config)
            
            relations = await extractor.extract_relations(sample_text)
            result = await extractor.extract_full(sample_text)
            
            assert len(relations) == 0
            assert len(result.relations) == 0
            assert result.metadata["enabled"] == False
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, mock_config):
        """Test handling of empty input."""
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = True
            
            extractor = OpenIEExtractor(mock_config)
            
            # Test empty string
            relations = await extractor.extract_relations("")
            result = await extractor.extract_full("")
            
            assert len(relations) == 0
            assert len(result.relations) == 0
            assert result.metadata["empty_input"] == True
            
            # Test whitespace only
            relations = await extractor.extract_relations("   \n\t  ")
            result = await extractor.extract_full("   \n\t  ")
            
            assert len(relations) == 0
            assert len(result.relations) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, sample_text, mock_config):
        """Test error handling in extraction pipeline."""
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = True
            
            extractor = OpenIEExtractor(mock_config)
            
            # Mock service to raise exception
            extractor.openie_service.extract_triplets = AsyncMock(
                side_effect=Exception("OpenIE service error")
            )
            
            # Test that ProcessingError is raised
            with pytest.raises(Exception):  # ProcessingError
                await extractor.extract_relations(sample_text)
            
            with pytest.raises(Exception):  # ProcessingError
                await extractor.extract_full(sample_text)
    
    @pytest.mark.asyncio
    async def test_get_extraction_stats(self, mock_config):
        """Test extraction statistics."""
        with patch('morag_graph.extractors.openie_extractor.get_settings') as mock_settings:
            mock_settings.return_value.openie_enabled = True
            mock_settings.return_value.openie_enable_entity_linking = True
            mock_settings.return_value.openie_enable_predicate_normalization = True
            
            extractor = OpenIEExtractor(mock_config)
            
            stats = await extractor.get_extraction_stats()
            
            assert stats["enabled"] == True
            assert stats["configuration"]["min_confidence"] == 0.6
            assert stats["configuration"]["entity_linking_enabled"] == True
            assert stats["configuration"]["predicate_normalization_enabled"] == True


@pytest.mark.asyncio
async def test_neo4j_integration_mock():
    """Test Neo4j integration with mocked storage."""
    # Mock Neo4j storage
    mock_storage = MagicMock(spec=Neo4jStorage)
    mock_storage.initialize_openie_schema = AsyncMock()
    mock_storage.store_openie_triplets = AsyncMock(return_value={
        "triplets_stored": 5,
        "relationships_created": 5,
        "nodes_created": 8,
        "source_doc_id": "test_doc"
    })
    
    # Mock triplets
    triplets = [
        ValidatedTriplet(
            subject="John",
            predicate="works at",
            object="Microsoft",
            confidence=0.85,
            sentence="John works at Microsoft.",
            sentence_id="sent_1",
            validation_score=0.8,
            validation_flags=set()
        )
    ]
    
    # Test schema initialization
    await mock_storage.initialize_openie_schema()
    mock_storage.initialize_openie_schema.assert_called_once()
    
    # Test triplet storage
    result = await mock_storage.store_openie_triplets(
        triplets, source_doc_id="test_doc"
    )
    
    assert result["triplets_stored"] == 5
    assert result["relationships_created"] == 5
    assert result["nodes_created"] == 8
    mock_storage.store_openie_triplets.assert_called_once_with(
        triplets, source_doc_id="test_doc"
    )
