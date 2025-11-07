"""Tests for enhanced entity and relation extraction components."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict

# Import the enhanced extraction components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-graph', 'src'))

from morag_graph.extraction.enhanced_entity_extractor import (
    EnhancedEntityExtractor,
    ConfidenceEntity,
    EntityConfidenceModel,
    BasicGleaningStrategy,
    ContextualGleaningStrategy,
    SemanticGleaningStrategy
)
from morag_graph.extraction.enhanced_relation_extractor import (
    EnhancedRelationExtractor,
    RelationValidator,
    RelationCandidate
)
from morag_graph.extraction.systematic_deduplicator import (
    SystematicDeduplicator,
    EntitySimilarityCalculator,
    LLMMergeValidator,
    MergeCandidate
)
from morag_graph.models import Entity, Relation


class TestEntityConfidenceModel:
    """Test the EntityConfidenceModel."""

    def setup_method(self):
        """Set up test fixtures."""
        self.confidence_model = EntityConfidenceModel()

    @pytest.mark.asyncio
    async def test_score_entity_high_confidence(self):
        """Test scoring entity with high confidence factors."""
        entity = Entity(
            name="Python",
            type="TECHNOLOGY",
            confidence=0.9,
            source_doc_id="test_doc"
        )

        text = "Python is a programming language used for data science and web development."
        existing_entities = []

        score = await self.confidence_model.score_entity(entity, text, existing_entities)

        # Should have high score due to exact name match and good base confidence
        assert score > 0.8
        assert score <= 1.0

    @pytest.mark.asyncio
    async def test_score_entity_with_duplicates(self):
        """Test scoring entity with existing duplicates."""
        entity = Entity(
            name="Python",
            type="TECHNOLOGY",
            confidence=0.8,
            source_doc_id="test_doc"
        )

        existing_entity = ConfidenceEntity(
            entity=Entity(
                name="Python Programming",
                type="TECHNOLOGY",
                confidence=0.9,
                source_doc_id="test_doc"
            ),
            confidence=0.9,
            extraction_round=1,
            gleaning_strategy="basic"
        )

        text = "Python is a programming language."
        existing_entities = [existing_entity]

        score = await self.confidence_model.score_entity(entity, text, existing_entities)

        # Should have lower score due to duplicate penalty
        assert score < 0.8


class TestGleaningStrategies:
    """Test the different gleaning strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_extractor = Mock()
        self.mock_extractor.extract = AsyncMock(return_value=[])

    @pytest.mark.asyncio
    async def test_basic_gleaning_strategy(self):
        """Test basic gleaning strategy."""
        strategy = BasicGleaningStrategy()

        text = "Test text for extraction"
        existing_entities = []

        await strategy.extract(text, existing_entities, self.mock_extractor, "test_doc")

        # Should call base extractor
        self.mock_extractor.extract.assert_called_once_with(text, "test_doc")

    @pytest.mark.asyncio
    async def test_contextual_gleaning_strategy(self):
        """Test contextual gleaning strategy."""
        strategy = ContextualGleaningStrategy()

        # Mock existing entities with specific types
        existing_entities = [
            ConfidenceEntity(
                entity=Entity(name="John", type="PERSON", confidence=0.8),
                confidence=0.8,
                extraction_round=1,
                gleaning_strategy="basic"
            )
        ]

        text = "John works at Microsoft in Seattle."

        # Mock the focused extractor creation
        with patch.object(strategy, '_create_focused_extractor') as mock_create:
            mock_focused_extractor = Mock()
            mock_focused_extractor.extract = AsyncMock(return_value=[])
            mock_create.return_value = mock_focused_extractor

            await strategy.extract(text, existing_entities, self.mock_extractor, "test_doc")

            # Should create focused extractor for missing types
            mock_create.assert_called_once()
            mock_focused_extractor.extract.assert_called_once()


class TestEnhancedEntityExtractor:
    """Test the EnhancedEntityExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_base_extractor = Mock()
        self.mock_base_extractor.extract = AsyncMock(return_value=[])
        self.mock_base_extractor.domain = "test"
        self.mock_base_extractor.language = "en"

        self.extractor = EnhancedEntityExtractor(
            base_extractor=self.mock_base_extractor,
            max_rounds=2,
            target_confidence=0.8,
            enable_gleaning=True
        )

    @pytest.mark.asyncio
    async def test_extract_with_gleaning_disabled(self):
        """Test extraction with gleaning disabled."""
        extractor = EnhancedEntityExtractor(
            base_extractor=self.mock_base_extractor,
            enable_gleaning=False
        )

        text = "Test text"
        await extractor.extract_with_gleaning(text, "test_doc")

        # Should call base extractor directly
        self.mock_base_extractor.extract.assert_called_once_with(text, "test_doc")

    @pytest.mark.asyncio
    async def test_extract_with_gleaning_enabled(self):
        """Test extraction with gleaning enabled."""
        # Mock entities returned by base extractor
        mock_entities = [
            Entity(name="Python", type="TECHNOLOGY", confidence=0.9),
            Entity(name="Microsoft", type="ORGANIZATION", confidence=0.8)
        ]

        self.mock_base_extractor.extract.return_value = mock_entities

        text = "Python is used at Microsoft for development."
        result = await self.extractor.extract_with_gleaning(text, "test_doc")

        # Should return entities
        assert isinstance(result, list)
        # Base extractor should be called at least once
        assert self.mock_base_extractor.extract.call_count >= 1

    def test_deduplicate_entities(self):
        """Test entity deduplication."""
        entities = [
            ConfidenceEntity(
                entity=Entity(name="Python", type="TECHNOLOGY", confidence=0.9),
                confidence=0.9,
                extraction_round=1,
                gleaning_strategy="basic"
            ),
            ConfidenceEntity(
                entity=Entity(name="python", type="technology", confidence=0.7),
                confidence=0.7,
                extraction_round=2,
                gleaning_strategy="contextual"
            ),
            ConfidenceEntity(
                entity=Entity(name="Java", type="TECHNOLOGY", confidence=0.8),
                confidence=0.8,
                extraction_round=1,
                gleaning_strategy="basic"
            )
        ]

        deduplicated = self.extractor._deduplicate_entities(entities)

        # Should keep 2 entities (Python with higher confidence, Java)
        assert len(deduplicated) == 2

        # Should keep the higher confidence Python entity
        python_entity = next(e for e in deduplicated if e.entity.name == "Python")
        assert python_entity.confidence == 0.9


class TestEntitySimilarityCalculator:
    """Test the EntitySimilarityCalculator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = EntitySimilarityCalculator()

    def test_calculate_similarity_exact_match(self):
        """Test similarity calculation for exact matches."""
        entity1 = Entity(name="Python", type="TECHNOLOGY", confidence=0.9)
        entity2 = Entity(name="Python", type="TECHNOLOGY", confidence=0.8)

        similarity = self.calculator.calculate_similarity(entity1, entity2)

        # Should have very high similarity
        assert similarity > 0.9

    def test_calculate_similarity_partial_match(self):
        """Test similarity calculation for partial matches."""
        entity1 = Entity(name="Dr. John Smith", type="PERSON", confidence=0.9)
        entity2 = Entity(name="John Smith", type="PERSON", confidence=0.8)

        similarity = self.calculator.calculate_similarity(entity1, entity2)

        # Should have high similarity (one name contains the other)
        assert similarity > 0.7
        assert similarity < 1.0

    def test_calculate_similarity_different_entities(self):
        """Test similarity calculation for different entities."""
        entity1 = Entity(name="Python", type="TECHNOLOGY", confidence=0.9)
        entity2 = Entity(name="Java", type="TECHNOLOGY", confidence=0.8)

        similarity = self.calculator.calculate_similarity(entity1, entity2)

        # Should have low similarity
        assert similarity < 0.5

    def test_calculate_name_similarity_jaccard(self):
        """Test Jaccard similarity calculation."""
        similarity = self.calculator._calculate_name_similarity(
            "Machine Learning Algorithm",
            "Deep Learning Algorithm"
        )

        # Should have moderate similarity (shared words: Learning, Algorithm)
        assert 0.3 < similarity < 0.8


class TestSystematicDeduplicator:
    """Test the SystematicDeduplicator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.deduplicator = SystematicDeduplicator(
            similarity_threshold=0.7,
            merge_confidence_threshold=0.8,
            enable_llm_validation=False  # Disable for testing
        )

    @pytest.mark.asyncio
    async def test_deduplicate_across_chunks_no_duplicates(self):
        """Test deduplication with no duplicate entities."""
        entities_by_chunk = {
            "chunk1": [
                Entity(name="Python", type="TECHNOLOGY", confidence=0.9),
                Entity(name="Java", type="TECHNOLOGY", confidence=0.8)
            ],
            "chunk2": [
                Entity(name="Microsoft", type="ORGANIZATION", confidence=0.9),
                Entity(name="Google", type="ORGANIZATION", confidence=0.8)
            ]
        }

        result_chunks, result = await self.deduplicator.deduplicate_across_chunks(entities_by_chunk)

        # Should have same number of entities (no duplicates)
        assert result.original_count == 4
        assert result.deduplicated_count == 4
        assert result.merges_performed == 0

    @pytest.mark.asyncio
    async def test_deduplicate_across_chunks_with_duplicates(self):
        """Test deduplication with duplicate entities."""
        entities_by_chunk = {
            "chunk1": [
                Entity(name="Python", type="TECHNOLOGY", confidence=0.9),
                Entity(name="Microsoft", type="ORGANIZATION", confidence=0.8)
            ],
            "chunk2": [
                Entity(name="python", type="technology", confidence=0.7),  # Duplicate
                Entity(name="Google", type="ORGANIZATION", confidence=0.9)
            ]
        }

        result_chunks, result = await self.deduplicator.deduplicate_across_chunks(entities_by_chunk)

        # Should have fewer entities after deduplication
        assert result.original_count == 4
        assert result.deduplicated_count < 4
        assert result.merges_performed > 0


class TestRelationValidator:
    """Test the RelationValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = RelationValidator()

    @pytest.mark.asyncio
    async def test_validate_semantic_relation(self):
        """Test semantic validation of relations."""
        relation = Relation(
            source_entity_id="person1",
            target_entity_id="org1",
            type="WORKS_FOR",
            description="John works for Microsoft",
            confidence=0.9
        )

        source_entity = Entity(name="John", type="PERSON", confidence=0.9)
        target_entity = Entity(name="Microsoft", type="ORGANIZATION", confidence=0.9)
        context_text = "John Smith is a software engineer at Microsoft Corporation."

        result = await self.validator._validate_semantic(
            relation, context_text, source_entity, target_entity
        )

        # Should be valid with high confidence
        assert result.is_valid
        assert result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_validate_incompatible_relation(self):
        """Test validation of incompatible relation types."""
        relation = Relation(
            source_entity_id="tech1",
            target_entity_id="tech2",
            type="WORKS_FOR",  # Incompatible: technology can't work for technology
            description="Python works for Java",
            confidence=0.8
        )

        source_entity = Entity(name="Python", type="TECHNOLOGY", confidence=0.9)
        target_entity = Entity(name="Java", type="TECHNOLOGY", confidence=0.9)
        context_text = "Python and Java are both programming languages."

        result = await self.validator._validate_semantic(
            relation, context_text, source_entity, target_entity
        )

        # Should have issues due to incompatible types
        assert not result.is_valid or result.confidence < 0.7
        assert len(result.issues) > 0


class TestEnhancedRelationExtractor:
    """Test the EnhancedRelationExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_base_extractor = Mock()
        self.mock_base_extractor.extract = AsyncMock(return_value=[])
        self.mock_base_extractor.domain = "test"
        self.mock_base_extractor.language = "en"

        self.extractor = EnhancedRelationExtractor(
            base_extractor=self.mock_base_extractor,
            max_rounds=2,
            confidence_threshold=0.7,
            enable_validation=False  # Disable for simpler testing
        )

    @pytest.mark.asyncio
    async def test_extract_with_no_entities(self):
        """Test extraction with no entities."""
        result = await self.extractor.extract_with_gleaning("Test text", [], "test_doc")

        # Should return empty list
        assert result == []
        # Base extractor should not be called
        self.mock_base_extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_with_entities(self):
        """Test extraction with entities."""
        entities = [
            Entity(name="John", type="PERSON", confidence=0.9),
            Entity(name="Microsoft", type="ORGANIZATION", confidence=0.8)
        ]

        mock_relations = [
            Relation(
                source_entity_id=entities[0].id,
                target_entity_id=entities[1].id,
                type="WORKS_FOR",
                description="John works for Microsoft",
                confidence=0.8
            )
        ]

        self.mock_base_extractor.extract.return_value = mock_relations

        text = "John Smith works at Microsoft Corporation."
        result = await self.extractor.extract_with_gleaning(text, entities, "test_doc")

        # Should return relations
        assert isinstance(result, list)
        # Base extractor should be called
        self.mock_base_extractor.extract.assert_called()

    def test_deduplicate_relations(self):
        """Test relation deduplication."""
        relations = [
            RelationCandidate(
                relation=Relation(
                    source_entity_id="e1",
                    target_entity_id="e2",
                    type="WORKS_FOR",
                    confidence=0.9
                ),
                confidence=0.9,
                extraction_round=1,
                validation_score=0.8,
                context_evidence="test"
            ),
            RelationCandidate(
                relation=Relation(
                    source_entity_id="e1",
                    target_entity_id="e2",
                    type="works_for",  # Same type, different case
                    confidence=0.7
                ),
                confidence=0.7,
                extraction_round=2,
                validation_score=0.6,
                context_evidence="test"
            )
        ]

        deduplicated = self.extractor._deduplicate_relations(relations)

        # Should keep only one relation (higher confidence)
        assert len(deduplicated) == 1
        assert deduplicated[0].confidence == 0.9


if __name__ == "__main__":
    pytest.main([__file__])
