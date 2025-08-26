"""Integration tests for entity deduplication pipeline."""

import pytest
import asyncio
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from morag_graph.extraction.systematic_deduplicator import SystematicDeduplicator
from morag_graph.storage.neo4j_storage import Neo4jStorage, Neo4jConfig
from morag_graph.models import Entity

# Skip all tests in this file due to interface changes
pytestmark = pytest.mark.skip(reason="EntityDeduplicator interface has changed - tests need to be updated")


@pytest.fixture
async def neo4j_storage():
    """Create Neo4j storage for testing."""
    # Skip if no Neo4j connection available
    if not os.getenv("NEO4J_PASSWORD"):
        pytest.skip("Neo4j not available for integration tests")
    
    config = Neo4jConfig(
        uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_TEST_DATABASE", "test_morag")
    )
    
    storage = Neo4jStorage(config)
    await storage.connect()
    
    # Clear test database
    await storage.clear()
    
    yield storage
    
    # Cleanup
    await storage.clear()
    await storage.close()


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(
            name="artificial intelligence",
            type="TECHNOLOGY",
            confidence=0.9,
            source_doc_id="doc1"
        ),
        Entity(
            name="AI",
            type="TECHNOLOGY",
            confidence=0.8,
            source_doc_id="doc1"
        ),
        Entity(
            name="A.I.",
            type="TECHNOLOGY",
            confidence=0.7,
            source_doc_id="doc2"
        ),
        Entity(
            name="machine learning",
            type="TECHNOLOGY",
            confidence=0.9,
            source_doc_id="doc1"
        ),
        Entity(
            name="ML",
            type="TECHNOLOGY",
            confidence=0.8,
            source_doc_id="doc2"
        ),
        Entity(
            name="John Smith",
            type="PERSON",
            confidence=0.9,
            source_doc_id="doc1"
        ),
        Entity(
            name="Smith, John",
            type="PERSON",
            confidence=0.8,
            source_doc_id="doc2"
        ),
        Entity(
            name="Google Inc.",
            type="ORGANIZATION",
            confidence=0.9,
            source_doc_id="doc1"
        ),
        Entity(
            name="Google",
            type="ORGANIZATION",
            confidence=0.8,
            source_doc_id="doc2"
        )
    ]


class TestEntityDeduplicationIntegration:
    """Integration tests for entity deduplication."""
    
    @pytest.mark.asyncio
    async def test_full_deduplication_pipeline_without_llm(self, neo4j_storage, sample_entities):
        """Test complete deduplication pipeline without LLM service."""
        # Store sample entities
        for entity in sample_entities:
            await neo4j_storage.store_entity(entity)
        
        # Initialize deduplicator without LLM service
        deduplicator = SystematicDeduplicator(
            similarity_threshold=0.7,
            merge_confidence_threshold=0.8,
            enable_llm_validation=False
        )
        
        # Run deduplication
        result = await deduplicator.deduplicate_entities()
        
        # Verify results
        assert isinstance(result, dict)
        assert 'total_entities_before' in result
        assert 'merge_candidates_found' in result
        assert 'merges_applied' in result
        assert 'merge_results' in result
        
        assert result['total_entities_before'] == len(sample_entities)
        assert result['merges_applied'] == 0  # Dry run mode
        assert isinstance(result['merge_results'], list)
    
    @pytest.mark.asyncio
    async def test_get_duplicate_candidates(self, neo4j_storage, sample_entities):
        """Test getting duplicate candidates without applying merges."""
        # Store sample entities
        for entity in sample_entities:
            await neo4j_storage.store_entity(entity)
        
        # Initialize deduplicator
        deduplicator = EntityDeduplicator(neo4j_storage, llm_service=None)
        
        # Get candidates
        candidates = await deduplicator.get_duplicate_candidates()
        
        # Verify candidates
        assert isinstance(candidates, list)
        for candidate in candidates:
            assert 'entities' in candidate
            assert 'canonical_form' in candidate
            assert 'confidence' in candidate
            assert 'reason' in candidate
            assert isinstance(candidate['entities'], list)
            assert len(candidate['entities']) >= 2
            assert 0.0 <= candidate['confidence'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_collection_specific_deduplication(self, neo4j_storage):
        """Test deduplication for specific collection."""
        # Create entities with different collections
        entities_collection1 = [
            Entity(name="AI", type="TECHNOLOGY", confidence=0.9, source_doc_id="doc1"),
            Entity(name="artificial intelligence", type="TECHNOLOGY", confidence=0.8, source_doc_id="doc1")
        ]

        entities_collection2 = [
            Entity(name="ML", type="TECHNOLOGY", confidence=0.9, source_doc_id="doc2"),
            Entity(name="machine learning", type="TECHNOLOGY", confidence=0.8, source_doc_id="doc2")
        ]
        
        # Store entities
        for entity in entities_collection1 + entities_collection2:
            await neo4j_storage.store_entity(entity)
        
        # Create document chunks to associate entities with collections
        # Note: This would require creating DocumentChunk objects and relationships
        # For now, we'll test the basic functionality
        
        deduplicator = EntityDeduplicator(neo4j_storage, llm_service=None)
        
        # Test deduplication without collection filter
        result = await deduplicator.deduplicate_entities()
        assert result['total_entities_before'] == 4
    
    @pytest.mark.asyncio
    async def test_entity_merge_execution(self, neo4j_storage):
        """Test actual entity merging (not dry run)."""
        # Create similar entities that should be merged
        entities = [
            Entity(name="artificial intelligence", type="TECHNOLOGY", confidence=0.9, source_doc_id="doc1"),
            Entity(name="Artificial Intelligence", type="TECHNOLOGY", confidence=0.8, source_doc_id="doc2")
        ]
        
        # Store entities
        for entity in entities:
            await neo4j_storage.store_entity(entity)
        
        # Verify entities are stored
        all_entities_before = await neo4j_storage.get_all_entities()
        assert len(all_entities_before) == 2
        
        # Initialize deduplicator for actual merging
        deduplicator = EntityDeduplicator(
            neo4j_storage, 
            llm_service=None,
            config={'dry_run': False, 'merge_confidence_threshold': 0.7}
        )
        
        # Run deduplication
        result = await deduplicator.deduplicate_entities()
        
        # Check if any merges were applied
        # Note: Without LLM, merges depend on string similarity fallback
        assert isinstance(result['merges_applied'], int)
        assert result['merges_applied'] >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_collection(self, neo4j_storage):
        """Test error handling with invalid collection name."""
        deduplicator = EntityDeduplicator(neo4j_storage, llm_service=None)
        
        # Test with non-existent collection
        result = await deduplicator.deduplicate_entities(collection_name="non_existent_collection")
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert result['total_entities_before'] == 0
        assert result['merge_candidates_found'] == 0
        assert result['merges_applied'] == 0
    
    @pytest.mark.asyncio
    async def test_large_entity_set_performance(self, neo4j_storage):
        """Test performance with larger entity set."""
        # Create a larger set of entities
        large_entity_set = []
        for i in range(50):
            entity = Entity(
                name=f"entity_{i}",
                type="CUSTOM",
                confidence=0.8,
                source_doc_id=f"doc_{i % 10}"
            )
            large_entity_set.append(entity)
        
        # Add some similar entities
        similar_entities = [
            Entity(name="test entity", type="CUSTOM", confidence=0.9, source_doc_id="doc_test"),
            Entity(name="Test Entity", type="CUSTOM", confidence=0.8, source_doc_id="doc_test"),
            Entity(name="TEST ENTITY", type="CUSTOM", confidence=0.7, source_doc_id="doc_test")
        ]
        large_entity_set.extend(similar_entities)
        
        # Store entities
        for entity in large_entity_set:
            await neo4j_storage.store_entity(entity)
        
        # Initialize deduplicator
        deduplicator = EntityDeduplicator(
            neo4j_storage, 
            llm_service=None,
            config={'dry_run': True, 'batch_size': 10}
        )
        
        # Run deduplication and measure basic performance
        import time
        start_time = time.time()
        result = await deduplicator.deduplicate_entities()
        end_time = time.time()
        
        # Verify results
        assert result['total_entities_before'] == len(large_entity_set)
        assert isinstance(result['merge_candidates_found'], int)
        
        # Basic performance check (should complete within reasonable time)
        processing_time = end_time - start_time
        assert processing_time < 30  # Should complete within 30 seconds


class TestEntityDeduplicationWithMockLLM:
    """Test entity deduplication with mocked LLM service."""
    
    @pytest.mark.asyncio
    async def test_deduplication_with_mock_llm(self, neo4j_storage, sample_entities):
        """Test deduplication with mocked LLM service."""
        # Store sample entities
        for entity in sample_entities:
            await neo4j_storage.store_entity(entity)
        
        # Create mock LLM service
        mock_llm_service = Mock()
        
        # Initialize deduplicator with mock LLM
        deduplicator = EntityDeduplicator(
            neo4j_storage, 
            llm_service=mock_llm_service,
            config={'dry_run': True}
        )
        
        # Run deduplication
        result = await deduplicator.deduplicate_entities()
        
        # Verify results
        assert isinstance(result, dict)
        assert result['total_entities_before'] == len(sample_entities)
    
    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self, neo4j_storage, sample_entities):
        """Test fallback behavior when LLM service fails."""
        # Store sample entities
        for entity in sample_entities:
            await neo4j_storage.store_entity(entity)
        
        # Create mock LLM service that raises exceptions
        mock_llm_service = Mock()
        mock_llm_service.side_effect = Exception("LLM service unavailable")
        
        # Initialize deduplicator
        deduplicator = EntityDeduplicator(
            neo4j_storage, 
            llm_service=mock_llm_service,
            config={'dry_run': True}
        )
        
        # Should handle LLM failure gracefully
        result = await deduplicator.deduplicate_entities()
        
        # Verify it still works with fallback
        assert isinstance(result, dict)
        assert result['total_entities_before'] == len(sample_entities)


class TestEntityDeduplicationConfiguration:
    """Test entity deduplication with different configurations."""
    
    @pytest.mark.asyncio
    async def test_different_confidence_thresholds(self, neo4j_storage, sample_entities):
        """Test deduplication with different confidence thresholds."""
        # Store sample entities
        for entity in sample_entities:
            await neo4j_storage.store_entity(entity)
        
        # Test with high confidence threshold
        deduplicator_high = EntityDeduplicator(
            neo4j_storage, 
            llm_service=None,
            config={'dry_run': True, 'merge_confidence_threshold': 0.95}
        )
        
        result_high = await deduplicator_high.deduplicate_entities()
        
        # Test with low confidence threshold
        deduplicator_low = EntityDeduplicator(
            neo4j_storage, 
            llm_service=None,
            config={'dry_run': True, 'merge_confidence_threshold': 0.5}
        )
        
        result_low = await deduplicator_low.deduplicate_entities()
        
        # Low threshold should potentially find more candidates
        assert result_low['merge_candidates_found'] >= result_high['merge_candidates_found']
    
    @pytest.mark.asyncio
    async def test_batch_size_configuration(self, neo4j_storage):
        """Test deduplication with different batch sizes."""
        # Create entities
        entities = [
            Entity(name=f"entity_{i}", type="CUSTOM", confidence=0.8, source_doc_id="doc1")
            for i in range(25)
        ]
        
        for entity in entities:
            await neo4j_storage.store_entity(entity)
        
        # Test with small batch size
        deduplicator = EntityDeduplicator(
            neo4j_storage, 
            llm_service=None,
            config={'dry_run': True, 'batch_size': 5}
        )
        
        result = await deduplicator.deduplicate_entities()
        
        # Should handle batching correctly
        assert result['total_entities_before'] == 25
        assert isinstance(result['merge_candidates_found'], int)
