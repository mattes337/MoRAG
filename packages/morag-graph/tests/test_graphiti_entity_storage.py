"""Tests for Graphiti entity storage functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from morag_graph.models import Entity, Relation
from morag_graph.graphiti.entity_storage import (
    GraphitiEntityStorage, EntityStorageResult, RelationStorageResult,
    create_entity_storage
)
from morag_graph.graphiti.migration_utils import (
    Neo4jToGraphitiMigrator, MigrationStats, MigrationResult,
    create_migrator
)
from morag_graph.graphiti import GraphitiConfig


class TestEntityStorageResult:
    """Test EntityStorageResult dataclass."""
    
    def test_init_success(self):
        """Test successful result initialization."""
        result = EntityStorageResult(
            success=True,
            entity_id="test-entity-123",
            episode_id="episode-456"
        )
        
        assert result.success is True
        assert result.entity_id == "test-entity-123"
        assert result.episode_id == "episode-456"
        assert result.deduplication_info is None
        assert result.error is None
    
    def test_init_failure(self):
        """Test failure result initialization."""
        result = EntityStorageResult(
            success=False,
            entity_id="test-entity-123",
            error="Storage failed"
        )
        
        assert result.success is False
        assert result.entity_id == "test-entity-123"
        assert result.error == "Storage failed"
        assert result.episode_id is None


class TestRelationStorageResult:
    """Test RelationStorageResult dataclass."""
    
    def test_init_success(self):
        """Test successful result initialization."""
        result = RelationStorageResult(
            success=True,
            relation_id="test-relation-123",
            episode_id="episode-789"
        )
        
        assert result.success is True
        assert result.relation_id == "test-relation-123"
        assert result.episode_id == "episode-789"
        assert result.missing_entities == []
        assert result.error is None
    
    def test_init_with_missing_entities(self):
        """Test initialization with missing entities."""
        result = RelationStorageResult(
            success=False,
            relation_id="test-relation-123",
            missing_entities=["entity1", "entity2"],
            error="Missing entities"
        )
        
        assert result.success is False
        assert result.missing_entities == ["entity1", "entity2"]
        assert result.error == "Missing entities"


class TestGraphitiEntityStorage:
    """Test GraphitiEntityStorage functionality."""
    
    def create_test_entity(self):
        """Create a test entity."""
        return Entity(
            id="test-entity-123",
            name="Test Entity",
            type="PERSON",
            description="A test entity for unit tests",
            confidence=0.95
        )
    
    def create_test_relation(self):
        """Create a test relation."""
        return Relation(
            id="test-relation-456",
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            type="RELATED_TO",
            description="A test relation",
            confidence=0.85
        )
    
    def test_init(self):
        """Test storage initialization."""
        storage = GraphitiEntityStorage()
        
        assert storage.config is None
        assert storage.entity_adapter is not None
        assert storage.relation_adapter is not None
        assert len(storage._entity_cache) == 0
        assert len(storage._relation_cache) == 0
    
    def test_init_with_config(self):
        """Test storage initialization with config."""
        config = GraphitiConfig(openai_api_key="test-key")
        storage = GraphitiEntityStorage(config)
        
        assert storage.config == config
    
    @pytest.mark.asyncio
    async def test_store_entity_no_graphiti(self):
        """Test entity storage when Graphiti is not available."""
        storage = GraphitiEntityStorage()
        storage.graphiti = None  # Simulate unavailable Graphiti
        
        entity = self.create_test_entity()
        result = await storage.store_entity(entity)
        
        assert result.success is False
        assert result.entity_id == entity.id
        assert "Graphiti instance not available" in result.error
    
    @pytest.mark.asyncio
    async def test_store_entity_success(self):
        """Test successful entity storage."""
        storage = GraphitiEntityStorage()
        
        # Mock Graphiti instance
        mock_graphiti = AsyncMock()
        mock_graphiti.add_episode = AsyncMock(return_value="episode-123")
        mock_graphiti.search = AsyncMock(return_value=[])  # No existing entities
        storage.graphiti = mock_graphiti
        
        entity = self.create_test_entity()
        result = await storage.store_entity(entity)
        
        assert result.success is True
        assert result.entity_id == entity.id
        assert result.episode_id == "episode-123"
        assert entity.name.lower() in storage._entity_cache
    
    @pytest.mark.asyncio
    async def test_store_entity_with_deduplication(self):
        """Test entity storage with deduplication."""
        storage = GraphitiEntityStorage()
        
        # Mock existing entity in cache
        entity = self.create_test_entity()
        storage._entity_cache[entity.name.lower()] = "existing-episode-123"
        
        # Mock Graphiti instance
        mock_graphiti = AsyncMock()
        storage.graphiti = mock_graphiti
        
        result = await storage.store_entity(entity, auto_deduplicate=True)
        
        assert result.success is True
        assert result.deduplication_info is not None
        assert result.deduplication_info["source"] == "cache"
    
    @pytest.mark.asyncio
    async def test_store_relation_no_graphiti(self):
        """Test relation storage when Graphiti is not available."""
        storage = GraphitiEntityStorage()
        storage.graphiti = None  # Simulate unavailable Graphiti
        
        relation = self.create_test_relation()
        result = await storage.store_relation(relation)
        
        assert result.success is False
        assert result.relation_id == relation.id
        assert "Graphiti instance not available" in result.error
    
    @pytest.mark.asyncio
    async def test_store_relation_success(self):
        """Test successful relation storage."""
        storage = GraphitiEntityStorage()
        
        # Mock Graphiti instance
        mock_graphiti = AsyncMock()
        mock_graphiti.add_episode = AsyncMock(return_value="episode-456")
        mock_graphiti.search = AsyncMock(return_value=[
            # Mock existing entities
            MagicMock(metadata={"adapter_type": "entity", "morag_entity_id": "entity-1"}),
            MagicMock(metadata={"adapter_type": "entity", "morag_entity_id": "entity-2"})
        ])
        storage.graphiti = mock_graphiti
        
        relation = self.create_test_relation()
        result = await storage.store_relation(relation)
        
        assert result.success is True
        assert result.relation_id == relation.id
        assert result.episode_id == "episode-456"
        assert len(result.missing_entities) == 0
    
    @pytest.mark.asyncio
    async def test_store_relation_missing_entities(self):
        """Test relation storage with missing entities."""
        storage = GraphitiEntityStorage()
        
        # Mock Graphiti instance with no existing entities
        mock_graphiti = AsyncMock()
        mock_graphiti.search = AsyncMock(return_value=[])
        storage.graphiti = mock_graphiti
        
        relation = self.create_test_relation()
        result = await storage.store_relation(relation, ensure_entities_exist=True)
        
        assert result.success is False
        assert len(result.missing_entities) == 2
        assert "entity-1" in result.missing_entities
        assert "entity-2" in result.missing_entities
    
    @pytest.mark.asyncio
    async def test_store_entities_batch(self):
        """Test batch entity storage."""
        storage = GraphitiEntityStorage()
        
        # Mock Graphiti instance
        mock_graphiti = AsyncMock()
        mock_graphiti.add_episode = AsyncMock(side_effect=["episode-1", "episode-2"])
        mock_graphiti.search = AsyncMock(return_value=[])  # No existing entities
        storage.graphiti = mock_graphiti
        
        entities = [self.create_test_entity(), self.create_test_entity()]
        entities[1].id = "test-entity-456"
        entities[1].name = "Test Entity 2"
        
        results = await storage.store_entities_batch(entities)
        
        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].episode_id == "episode-1"
        assert results[1].episode_id == "episode-2"
    
    def test_get_storage_stats(self):
        """Test storage statistics."""
        storage = GraphitiEntityStorage()
        storage._entity_cache["entity1"] = "episode1"
        storage._relation_cache["rel1"] = "episode2"
        
        stats = storage.get_storage_stats()
        
        assert stats["entity_cache_size"] == 1
        assert stats["relation_cache_size"] == 1
        assert "graphiti_available" in stats
        assert "entity_adapter_stats" in stats
        assert "relation_adapter_stats" in stats


class TestMigrationStats:
    """Test MigrationStats functionality."""
    
    def test_init(self):
        """Test stats initialization."""
        stats = MigrationStats()
        
        assert stats.entities_processed == 0
        assert stats.entities_migrated == 0
        assert stats.relations_processed == 0
        assert stats.relations_migrated == 0
        assert stats.start_time is None
        assert stats.end_time is None
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        stats = MigrationStats()
        stats.start_time = datetime(2024, 1, 1, 12, 0, 0)
        stats.end_time = datetime(2024, 1, 1, 12, 0, 30)
        
        assert stats.duration_seconds == 30.0
    
    def test_success_rates(self):
        """Test success rate calculations."""
        stats = MigrationStats()
        stats.entities_processed = 10
        stats.entities_migrated = 8
        stats.relations_processed = 5
        stats.relations_migrated = 4
        
        assert stats.entity_success_rate == 0.8
        assert stats.relation_success_rate == 0.8
    
    def test_success_rates_zero_division(self):
        """Test success rates with zero processed items."""
        stats = MigrationStats()
        
        assert stats.entity_success_rate == 0.0
        assert stats.relation_success_rate == 0.0


class TestNeo4jToGraphitiMigrator:
    """Test Neo4jToGraphitiMigrator functionality."""
    
    def test_init(self):
        """Test migrator initialization."""
        mock_neo4j = MagicMock()
        migrator = Neo4jToGraphitiMigrator(mock_neo4j)
        
        assert migrator.neo4j_storage == mock_neo4j
        assert migrator.graphiti_storage is not None
        assert migrator.batch_size == 100
        assert isinstance(migrator.stats, MigrationStats)
    
    def test_init_with_config(self):
        """Test migrator initialization with config."""
        mock_neo4j = MagicMock()
        config = GraphitiConfig(openai_api_key="test-key")
        migrator = Neo4jToGraphitiMigrator(mock_neo4j, config, batch_size=50)
        
        assert migrator.batch_size == 50
        assert migrator.graphiti_storage.config == config
    
    def test_get_migration_summary(self):
        """Test migration summary generation."""
        mock_neo4j = MagicMock()
        migrator = Neo4jToGraphitiMigrator(mock_neo4j)
        
        # Set some test stats
        migrator.stats.entities_processed = 10
        migrator.stats.entities_migrated = 8
        migrator.errors = ["Error 1", "Error 2"]
        migrator.warnings = ["Warning 1"]
        
        summary = migrator.get_migration_summary()
        
        assert "stats" in summary
        assert "errors" in summary
        assert "warnings" in summary
        assert "graphiti_storage_stats" in summary
        assert summary["stats"]["entities"]["processed"] == 10
        assert summary["stats"]["entities"]["migrated"] == 8
        assert len(summary["errors"]) == 2
        assert len(summary["warnings"]) == 1


class TestCreateFunctions:
    """Test creation functions."""
    
    def test_create_entity_storage(self):
        """Test entity storage creation function."""
        storage = create_entity_storage()
        
        assert isinstance(storage, GraphitiEntityStorage)
        assert storage.config is None
    
    def test_create_entity_storage_with_config(self):
        """Test entity storage creation with config."""
        config = GraphitiConfig(openai_api_key="test-key")
        storage = create_entity_storage(config)
        
        assert isinstance(storage, GraphitiEntityStorage)
        assert storage.config == config
    
    def test_create_migrator(self):
        """Test migrator creation function."""
        mock_neo4j = MagicMock()
        migrator = create_migrator(mock_neo4j)
        
        assert isinstance(migrator, Neo4jToGraphitiMigrator)
        assert migrator.neo4j_storage == mock_neo4j
        assert migrator.batch_size == 100
    
    def test_create_migrator_with_config(self):
        """Test migrator creation with config."""
        mock_neo4j = MagicMock()
        config = GraphitiConfig(openai_api_key="test-key")
        migrator = create_migrator(mock_neo4j, config, batch_size=50)
        
        assert isinstance(migrator, Neo4jToGraphitiMigrator)
        assert migrator.batch_size == 50
