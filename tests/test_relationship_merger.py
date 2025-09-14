"""Tests for relationship merger maintenance job."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from morag_graph.maintenance.relationship_merger import (
    RelationshipMerger,
    RelationshipMergerConfig,
    RelationshipCandidate,
    MergeCandidate,
    run_relationship_merger
)


@pytest.fixture
def config():
    """Create test configuration."""
    return RelationshipMergerConfig(
        similarity_threshold=0.8,
        batch_size=10,
        dry_run=False,
        limit_relations=100,
        enable_rotation=False,
        merge_bidirectional=True,
        merge_transitive=False,
        min_confidence=0.5
    )


@pytest.fixture
def mock_neo4j_storage():
    """Create mock Neo4j storage."""
    storage = MagicMock()
    storage._connection_ops = MagicMock()
    storage._connection_ops._execute_query = AsyncMock()
    return storage


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = MagicMock()
    client.generate_text = AsyncMock()
    return client


@pytest.fixture
def sample_relationships():
    """Create sample relationships for testing."""
    return [
        RelationshipCandidate(
            id="rel1",
            source_id="entity1",
            target_id="entity2",
            type="WORKS_FOR",
            confidence=0.9,
            metadata={}
        ),
        RelationshipCandidate(
            id="rel2",
            source_id="entity1",
            target_id="entity2",
            type="WORKS_FOR",
            confidence=0.8,
            metadata={}
        ),
        RelationshipCandidate(
            id="rel3",
            source_id="entity1",
            target_id="entity2",
            type="EMPLOYED_BY",
            confidence=0.85,
            metadata={}
        ),
        RelationshipCandidate(
            id="rel4",
            source_id="entity2",
            target_id="entity1",
            type="EMPLOYS",
            confidence=0.7,
            metadata={}
        )
    ]


class TestRelationshipMergerConfig:
    """Test relationship merger configuration."""

    def test_ensure_defaults(self):
        """Test configuration validation."""
        config = RelationshipMergerConfig(
            similarity_threshold=1.5,  # Invalid
            batch_size=0,  # Invalid
            min_confidence=-0.1  # Invalid
        )
        
        config.ensure_defaults()
        
        assert config.similarity_threshold == 1.0
        assert config.batch_size == 1
        assert config.min_confidence == 0.0


class TestRelationshipMerger:
    """Test relationship merger functionality."""

    @pytest.mark.asyncio
    async def test_find_duplicate_relationships(self, config, mock_neo4j_storage, mock_llm_client, sample_relationships):
        """Test finding exact duplicate relationships."""
        merger = RelationshipMerger(config, mock_neo4j_storage, mock_llm_client)
        
        # Test with relationships that have exact duplicates
        duplicates = await merger._find_duplicate_relationships(sample_relationships[:2])
        
        assert len(duplicates) == 1
        assert duplicates[0].primary_relationship.id == "rel1"  # Higher confidence
        assert len(duplicates[0].duplicate_relationships) == 1
        assert duplicates[0].duplicate_relationships[0].id == "rel2"
        assert duplicates[0].merge_reason == "duplicate_exact"

    @pytest.mark.asyncio
    async def test_find_semantic_equivalents(self, config, mock_neo4j_storage, mock_llm_client, sample_relationships):
        """Test finding semantically equivalent relationships."""
        merger = RelationshipMerger(config, mock_neo4j_storage, mock_llm_client)

        # Mock LLM response for semantic similarity
        mock_llm_client.generate = AsyncMock(return_value='[["WORKS_FOR", "EMPLOYED_BY"]]')

        # Test with relationships that are semantically similar
        test_rels = [sample_relationships[0], sample_relationships[2]]  # WORKS_FOR and EMPLOYED_BY
        semantic_groups = await merger._find_semantic_equivalents(test_rels)

        assert len(semantic_groups) == 1
        assert semantic_groups[0].primary_relationship.id == "rel1"  # Higher confidence
        assert len(semantic_groups[0].duplicate_relationships) == 1
        assert semantic_groups[0].duplicate_relationships[0].id == "rel3"
        assert semantic_groups[0].merge_reason == "semantic_equivalent"

    @pytest.mark.asyncio
    async def test_get_merger_candidates(self, config, mock_neo4j_storage, mock_llm_client):
        """Test getting merger candidates from database."""
        merger = RelationshipMerger(config, mock_neo4j_storage, mock_llm_client)
        
        # Mock database response
        mock_neo4j_storage._connection_ops._execute_query.return_value = [
            {
                "id": "rel1",
                "source_id": "entity1",
                "target_id": "entity2",
                "type": "WORKS_FOR",
                "confidence": 0.9,
                "metadata": "{}"
            }
        ]
        
        candidates = await merger._get_merger_candidates()
        
        assert len(candidates) == 1
        assert candidates[0].id == "rel1"
        assert candidates[0].type == "WORKS_FOR"
        assert candidates[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_apply_merge_dry_run(self, config, mock_neo4j_storage, mock_llm_client):
        """Test merge application in dry run mode."""
        config.dry_run = True
        merger = RelationshipMerger(config, mock_neo4j_storage, mock_llm_client)
        
        primary = RelationshipCandidate("rel1", "e1", "e2", "WORKS_FOR", 0.9, {})
        duplicate = RelationshipCandidate("rel2", "e1", "e2", "WORKS_FOR", 0.8, {})
        
        merge_candidate = MergeCandidate(
            primary_relationship=primary,
            duplicate_relationships=[duplicate],
            merge_reason="duplicate_exact",
            confidence_score=1.0
        )
        
        # Should not call any database operations in dry run
        await merger._apply_merge(merge_candidate)
        
        # Verify no database calls were made
        mock_neo4j_storage._connection_ops._execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_merge_real(self, config, mock_neo4j_storage, mock_llm_client):
        """Test merge application in real mode."""
        config.dry_run = False
        merger = RelationshipMerger(config, mock_neo4j_storage, mock_llm_client)
        
        primary = RelationshipCandidate("rel1", "e1", "e2", "WORKS_FOR", 0.9, {})
        duplicate = RelationshipCandidate("rel2", "e1", "e2", "WORKS_FOR", 0.8, {})
        
        merge_candidate = MergeCandidate(
            primary_relationship=primary,
            duplicate_relationships=[duplicate],
            merge_reason="duplicate_exact",
            confidence_score=1.0
        )
        
        # Should call database operations
        await merger._apply_merge(merge_candidate)
        
        # Verify database calls were made (update + delete)
        assert mock_neo4j_storage._connection_ops._execute_query.call_count == 2

    @pytest.mark.asyncio
    async def test_run_merger_empty_result(self, config, mock_neo4j_storage, mock_llm_client):
        """Test merger with no relationships to process."""
        merger = RelationshipMerger(config, mock_neo4j_storage, mock_llm_client)
        
        # Mock empty database response
        mock_neo4j_storage._connection_ops._execute_query.return_value = []
        
        result = await merger.run_merger()
        
        assert result.total_relationships == 0
        assert result.processed_relationships == 0
        assert result.total_merges == 0
        assert result.dry_run == config.dry_run


class TestRunRelationshipMerger:
    """Test the main run function."""

    @pytest.mark.asyncio
    async def test_run_relationship_merger_with_overrides(self):
        """Test running relationship merger with configuration overrides."""
        overrides = {
            "similarity_threshold": 0.9,
            "batch_size": 50,
            "dry_run": False,
            "limit_relations": 500
        }
        
        with patch('morag_graph.maintenance.relationship_merger.Neo4jStorage') as mock_storage_class, \
             patch('morag_graph.maintenance.relationship_merger.Neo4jConfig') as mock_config_class, \
             patch('morag_graph.maintenance.relationship_merger.LLMClient') as mock_llm_class:
            
            # Setup mocks
            mock_storage = AsyncMock()
            mock_storage_class.return_value = mock_storage
            mock_config_class.from_env.return_value = MagicMock()
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm
            
            # Mock the merger run method
            with patch.object(RelationshipMerger, 'run_merger') as mock_run:
                mock_result = MagicMock()
                mock_run.return_value = mock_result
                
                result = await run_relationship_merger(overrides)

                # Verify the merger was called
                mock_run.assert_called_once()
                assert result == mock_result.to_dict()
                
                # Verify storage connection lifecycle
                mock_storage.connect.assert_called_once()
                mock_storage.disconnect.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
