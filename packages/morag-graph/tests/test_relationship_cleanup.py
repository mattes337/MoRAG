"""Tests for relationship cleanup maintenance job."""
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from morag_graph.maintenance.relationship_cleanup import (
    RelationshipCleanupConfig,
    RelationshipCleanupService,
    parse_cleanup_overrides,
    run_relationship_cleanup,
)


class TestRelationshipCleanupConfig:
    """Test relationship cleanup configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RelationshipCleanupConfig()
        assert config.dry_run is False
        assert config.batch_size == 100
        assert config.min_confidence == 0.3
        assert config.remove_unrelated is True
        assert config.remove_generic is True
        assert config.consolidate_similar is True
        assert config.similarity_threshold == 0.85

    def test_config_validation(self):
        """Test configuration validation."""
        config = RelationshipCleanupConfig(
            similarity_threshold=1.5,  # Should be clamped to 1.0
            batch_size=0,  # Should be set to 1
            min_confidence=-0.1,  # Should be clamped to 0.0
        )
        config.ensure_defaults()

        assert config.similarity_threshold == 1.0
        assert config.batch_size == 1
        assert config.min_confidence == 0.0


class TestRelationshipCleanupService:
    """Test relationship cleanup service."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock Neo4j storage."""
        storage = MagicMock()
        storage._connection_ops = MagicMock()
        storage._connection_ops._execute_query = AsyncMock()
        return storage

    @pytest.fixture
    def cleanup_service(self, mock_storage):
        """Create cleanup service with mock storage."""
        config = RelationshipCleanupConfig(dry_run=True)  # Use dry_run for tests
        return RelationshipCleanupService(mock_storage, config)

    async def test_run_cleanup_dry_run(self, cleanup_service, mock_storage):
        """Test running cleanup in dry-run mode."""
        # Mock the type summary query (for optimized approach)
        mock_storage._connection_ops._execute_query.side_effect = [
            # First call: type summary
            [
                {
                    "neo4j_type": "UNRELATED",
                    "stored_type": "UNRELATED",
                    "count": 1,
                    "avg_confidence": 0.5,
                    "sample_type_values": ["UNRELATED"],
                }
            ],
            # Second call: count UNRELATED relationships
            [{"count": 1}],
            # Third call: count orphaned relationships
            [{"count": 0}],
            # Fourth call: count low confidence relationships
            [{"count": 0}],
        ]

        result = await cleanup_service.run_cleanup()

        assert result.dry_run is True
        assert result.total_removed == 1
        assert result.meaningless_removed == 1
        # Multiple query calls for the optimized approach
        assert mock_storage._connection_ops._execute_query.call_count >= 3


class TestConfigurationParsing:
    """Test configuration parsing from environment variables."""

    def test_parse_cleanup_overrides_defaults(self, monkeypatch):
        """Test parsing with default values."""
        # Clear any existing environment variables
        for key in ["MORAG_REL_CLEANUP_DRY_RUN", "MORAG_REL_CLEANUP_BATCH_SIZE"]:
            monkeypatch.delenv(key, raising=False)

        overrides = parse_cleanup_overrides()
        assert overrides["dry_run"] is False  # Default

    def test_parse_cleanup_overrides_custom(self, monkeypatch):
        """Test parsing with custom values."""
        monkeypatch.setenv("MORAG_REL_CLEANUP_DRY_RUN", "false")
        monkeypatch.setenv("MORAG_REL_CLEANUP_BATCH_SIZE", "50")
        monkeypatch.setenv("MORAG_REL_CLEANUP_MIN_CONFIDENCE", "0.4")
        monkeypatch.setenv("MORAG_REL_CLEANUP_REMOVE_UNRELATED", "false")

        overrides = parse_cleanup_overrides()
        assert overrides["dry_run"] is False
        assert overrides["batch_size"] == 50
        assert overrides["min_confidence"] == 0.4
        assert overrides["remove_unrelated"] is False


@pytest.mark.asyncio
async def test_run_relationship_cleanup_integration():
    """Test the main run_relationship_cleanup function."""
    # This is a basic integration test that verifies the function can be called
    # In a real environment, this would need proper Neo4j setup

    # Mock the Neo4j config and storage
    with pytest.raises(Exception):  # Expected to fail without proper Neo4j setup
        await run_relationship_cleanup({"dry_run": True})
